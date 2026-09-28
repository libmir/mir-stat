/++
Internal widening count storage accessed through stable state.

Counters start as ubyte and the entire buffer widens to ushort, uint, then
ulong when an increment requires it. Reads always return a ulong snapshot.
Widening preserves the shape, indices, and existing proxy/view ownership;
it temporarily needs both the old and new buffers and copies every counter.
The storage does not shrink. Incrementing ulong.max is an unrecoverable error,
including in release builds. Only unweighted increments are supported.

Long contiguous batches with IntegralAxis or RegularAxis and plain numeric
input iterators select the counter type once per batch or promotion. Promotion
happens at the observation that needs it, not in a preliminary pass. Other
layouts, custom axes/fields, and short batches retain scalar insertion. This
keeps borrowed buffers out of paths that can invoke mutating user callbacks.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.shared_counts;

import mir.ndslice.allocation: rcslice;
import mir.ndslice.iterator: FieldIterator;
import mir.ndslice.slice: Slice, sliced;
import mir.rc.array: RCI;
import mir.rc.ptr: mir_rcptr, createRC;
import std.traits: isMutable;

package(mir.stat.descriptive.histogram):

// Keep the state at a stable address. Only it owns the current buffer; views
// and proxies retain the state, never a pointer into a replaceable buffer.
// Reference counting manages ownership, not concurrent access to counters.
struct SharedCountState
{
private:
    enum Width { byte_, short_, int_, long_ }
    Width width;
    size_t length;
    // Only the selected buffer owns memory. Separate typed handles keep
    // allocation, destruction, and access @safe without a manually managed union.
    Slice!(RCI!ubyte) bytes;
    Slice!(RCI!ushort) shorts;
    Slice!(RCI!uint) ints;
    Slice!(RCI!ulong) longs;

    void incrementBuffer(T, U)(ref Slice!(RCI!T) current,
        ref Slice!(RCI!U) next, size_t index) @safe pure nothrow @nogc
    {
        if (current[index] < T.max)
        {
            ++current[index];
            return;
        }
        widenAndIncrement(current, next, index);
    }

    // Keep allocation and copying out of the ordinary increment path so that
    // the small check-and-increment helper can be inlined by the compiler.
    void widenAndIncrement(T, U)(ref Slice!(RCI!T) current,
        ref Slice!(RCI!U) next, size_t index) @safe pure nothrow @nogc
    {
        // Publish only after allocation and copying. The increment is applied
        // in the wider type, so the triggering count cannot wrap to zero.
        auto replacement = rcslice!U(length);
        foreach (i; 0 .. length)
            replacement[i] = current[i];
        ++replacement[index];
        next = replacement;
        current = typeof(current).init;
        width = cast(Width)(width + 1);
    }
public:
    this(size_t length) @safe pure nothrow @nogc
    {
        this.length = length;
        bytes = rcslice!ubyte(length);
        bytes[] = 0;
    }

    ulong count(size_t index) const @safe pure nothrow @nogc
    {
        final switch (width)
        {
        case Width.byte_: return bytes[index];
        case Width.short_: return shorts[index];
        case Width.int_: return ints[index];
        case Width.long_: return longs[index];
        }
    }

    // Keep the terminal width outside the general dispatcher so ordinary
    // ulong increments can inline without pulling in the narrower cases.
    void increment(size_t index) @safe pure nothrow @nogc
    {
        if (width == Width.long_)
        {
            if (longs[index] == ulong.max)
                assert(0, "Histogram count exceeds ulong.max");
            ++longs[index];
        }
        else
            incrementNarrow(index);
    }
private:
    void incrementNarrow(size_t index) @safe pure nothrow @nogc
    {
        final switch (width)
        {
        case Width.byte_: incrementBuffer(bytes, shorts, index); break;
        case Width.short_: incrementBuffer(shorts, ints, index); break;
        case Width.int_: incrementBuffer(ints, longs, index); break;
        case Width.long_: assert(0, "Expected narrow counters");
        }
    }
}

struct SharedCountProxy(State)
{
private:
    mir_rcptr!State owner;
    size_t index;
public:
    ulong count() const @safe pure nothrow @nogc
    {
        return owner.count(index);
    }

    void opUnary(string op : "++")() @safe pure nothrow @nogc
        if (isMutable!State)
    {
        owner.increment(index);
    }
}

// FieldIterator supplies position arithmetic and ndslice integration. Field
// indexing creates owning proxies; lightConst preserves the same state while
// removing mutation capability, without copying the counters.
struct SharedCountField(State)
{
private:
    mir_rcptr!State owner;
public:
    auto opIndex(ptrdiff_t index) @safe pure nothrow @nogc
    {
        assert(index >= 0 && cast(size_t) index < owner.length);
        return SharedCountProxy!State(owner, cast(size_t) index);
    }

    // FieldIterator forwards indexed increments here. The field already owns
    // the state for the duration of the call, so immediate insertion needs no
    // temporary owning proxy (and no extra reference-count updates).
    void opIndexUnary(string op : "++")(ptrdiff_t index) @safe pure nothrow @nogc
        if (isMutable!State)
    {
        assert(index >= 0 && cast(size_t) index < owner.length);
        owner.increment(cast(size_t) index);
    }

    auto lightConst()() const @property @safe pure nothrow @nogc
    {
        return SharedCountField!(const State)(owner.lightConst);
    }
}

auto sharedCountSlice(size_t length) @safe pure nothrow @nogc
{
    auto owner = createRC!SharedCountState(length);
    auto field = SharedCountField!SharedCountState(owner);
    return FieldIterator!(typeof(field))(0, field).sliced(length);
}

// Saved proxies and bin views follow replacement; snapshots stay independent.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.topology: stride;
    alias A = IntegralAxis!(int, AxisOptions());
    auto storage = sharedCountSlice(8);
    auto joint = storage.stride(2).sliced(2, 2);
    auto h = HistogramAccumulator!(typeof(joint), A, A)(joint, A(2, 0), A(2, 0));
    auto saved = h.counts[0, 1];
    auto view = h.bins;
    h.put(0, 1);
    auto snapshot = view[1];
    auto copy = h;

    // Retain the old buffer only in this test to demonstrate that no writes
    // reach it after replacement. Production proxies never retain it.
    auto state = storage.iterator._field.owner;
    auto old = state.bytes;
    foreach (i; 1 .. 256) copy.put(0, 1);
    saved++;
    copy.put(0, 1);
    assert(saved.count() == 258 && view[1].count == 258);
    assert(snapshot.count == 1 && old[2] == 255);
    assert(storage[1].count() == 0 && storage[3].count() == 0);
    saved++;
    assert(view[1].count == 259 && old[2] == 255);
    const readOnly = h;
    assert(readOnly.bins[1].count == 259);
    static assert(!__traits(compiles, readOnly.put(0, 1)));
    auto frozen = joint.lightConst;
    assert(frozen[0, 1].count() == 259);
    static assert(!__traits(compiles, frozen[0, 1]++));
    auto independent = sharedCountSlice(8);
    assert(independent[2].count() == 0);
    auto empty = sharedCountSlice(0);
    assert(empty.length == 0);
}

// Seed near the large boundaries instead of performing billions of updates.
// Each transition preserves all cells and releases the previous typed handle.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto storage = sharedCountSlice(3);
    auto saved = storage[1];
    auto readOnly = storage.lightConst;
    auto state = storage.iterator._field.owner;
    alias W = SharedCountState.Width;
    assert(state.width == W.byte_);
    state.bytes[0] = 7;
    state.bytes[1] = ubyte.max - 1;
    ++storage[1];
    assert(saved.count() == ubyte.max && state.width == W.byte_);
    ++storage[1];
    assert(saved.count() == 256 && state.width == W.short_);
    assert(state.bytes.length == 0);
    state.shorts[1] = ushort.max - 1;
    saved++;
    assert(saved.count() == ushort.max && state.width == W.short_);
    saved++;
    assert(saved.count() == 65536 && state.width == W.int_);
    assert(state.shorts.length == 0);
    state.ints[1] = uint.max - 1;
    ++storage[1];
    assert(saved.count() == uint.max && state.width == W.int_);
    ++storage[1];
    assert(saved.count() == cast(ulong) uint.max + 1 && state.width == W.long_);
    assert(state.ints.length == 0);
    state.longs[1] = ulong.max - 1;
    saved++;
    assert(saved.count() == ulong.max);
    assert(readOnly[1].count() == ulong.max);
    assert(readOnly[0].count() == 7 && readOnly[2].count() == 0);
}

// Overflow fails before mutation. Release builds may halt instead of throwing,
// so catching the error is tested only when runtime assertions are enabled.
version(mir_stat_test)
version(assert)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    auto storage = sharedCountSlice(1);
    auto state = storage.iterator._field.owner;
    state.bytes = typeof(state.bytes).init;
    state.longs = rcslice!ulong(1);
    state.longs[0] = ulong.max;
    state.width = SharedCountState.Width.long_;
    bool rejected;
    try { ++storage[0]; }
    catch (AssertError) { rejected = true; }
    assert(rejected && storage[0].count() == ulong.max);
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, AxisOptions());
    auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(1,0));
    uint[65] value = 0;
    rejected = false;
    try { h.put(value[]); }
    catch (AssertError) { rejected = true; }
    assert(rejected && storage[0].count() == ulong.max);
}

// Proxies and views own their state, so returning them needs no borrowed owner.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(int, AxisOptions());
    static auto cell() @safe pure nothrow @nogc
    {
        auto storage = sharedCountSlice(2);
        storage[1]++;
        return storage[1];
    }
    auto saved = cell();
    saved++;
    assert(saved.count() == 2);
    static auto bins() @safe pure nothrow @nogc
    {
        auto storage = sharedCountSlice(2);
        auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(2, 0));
        h.put(1);
        return h.bins;
    }
    auto view = bins();
    assert(view[1].count == 1);
}

// A batch may insert through aliases, but must not replace its storage or
// change its shape/strides while traversing. Widening is supported explicitly.
template isSharedCountStorage(S)
{
    import mir.ndslice.slice: isSlice;
    static if (isSlice!S)
        enum isSharedCountStorage = isSharedCountIterator!(S.Iterator);
    else
        enum isSharedCountStorage = false;
}
private template isSharedCountIterator(I)
{
    import mir.ndslice.iterator: StrideIterator;
    static if (is(I == FieldIterator!(SharedCountField!SharedCountState)))
        enum isSharedCountIterator = true;
    else static if (is(I == StrideIterator!Inner, Inner))
        enum isSharedCountIterator = isSharedCountIterator!Inner;
    else
        enum isSharedCountIterator = false;
}
// Borrow a typed buffer only when coordinate reads and axis classification cannot
// call user code that replaces it through an alias. Custom axes and fields keep
// ordinary insertion. Extend this deliberately, not by testing syntax alone.
private template isBatchAxis(A)
{
    import mir.stat.descriptive.histogram.axis: IntegralAxis, RegularAxis;
    import std.traits: isNumeric;
    static if (is(A == IntegralAxis!Args, Args...) || is(A == RegularAxis!Args, Args...))
        enum isBatchAxis = isNumeric!(A.BinType);
    else
        enum isBatchAxis = false;
}
private template isBatchInputIterator(I)
{
    import mir.ndslice.iterator: StrideIterator;
    import std.traits: isNumeric;
    static if (is(I == T*, T))
        enum isBatchInputIterator = isNumeric!T;
    else static if (is(I == StrideIterator!Inner, Inner))
        enum isBatchInputIterator = isBatchInputIterator!Inner;
    else
        enum isBatchInputIterator = false;
}
private template canBatchCounts(H, Inputs...)
{
    enum canBatchCounts = () {
        static if (typeof(H.init.counts).S != 0 ||
            !is(typeof(H.init.counts.iterator) == FieldIterator!(SharedCountField!SharedCountState)))
            return false;
        else
        {
            static foreach (i; 0 .. H.N)
                static if (!isBatchAxis!(typeof(H.init.axis[i])) ||
                    !isBatchInputIterator!(Inputs[i].Iterator))
                    return false;
            return true;
        }
    }();
}
// Share scalar classification, including underflow/overflow and circular axes.
// The storage constructor already verified the extents against the axes.
private size_t countRowIndex(size_t column = 0, H, Inputs...)(
    ref H h, size_t position, Inputs inputs)
{
    static if (column == 0)
        return h.storageIndex!0(inputs[0][position]);
    else
        return countRowIndex!(column - 1)(h, position, inputs) *
            h.counts._lengths[column] + h.storageIndex!column(inputs[column][position]);
}
// No owning proxies, layout dispatch, or representation dispatch in this loop.
// Inputs are descriptors copied once, not aliases reloaded after every increment.
private size_t consumeTypedCounts(T, bool hasOffset, H, S, Inputs...)(
    ref SharedCountState state, ref H h, S cells, size_t offset,
    size_t position, Inputs inputs)
{
    for (; position < inputs[0].length; ++position)
    {
        const index = countRowIndex!(H.N - 1)(h, position, inputs);
        if (cells[index] == T.max)
        {
            static if (is(T == ulong))
                assert(0, "Histogram count exceeds ulong.max");
            else
            {
                // Consume the triggering observation once, then reselect the
                // widened buffer before reading the next observation.
                static if (hasOffset) state.increment(index + offset);
                else state.increment(index);
                return position + 1;
            }
        }
        ++cells[index];
    }
    return position;
}
private size_t consumeCountBuffer(T, H, S, Inputs...)(ref SharedCountState state,
    ref H h, S cells, size_t offset, size_t position, Inputs inputs)
{
    if (offset == 0)
        return consumeTypedCounts!(T, false)(state, h, cells, 0, position, inputs);
    return consumeTypedCounts!(T, true)(state, h, cells[offset .. $], offset, position, inputs);
}
// Only unweighted counts enter this path. Factories retain shape validation and
// nested traversal, selecting this function at a one-dimensional input row.
void insertSharedCounts(H, Inputs...)(ref H h, auto ref Inputs inputs)
    if (isSharedCountStorage!(typeof(h.counts)) && Inputs.length == H.N)
{
    normalizeCountInputs!0(h, inputs);
}
private void normalizeCountInputs(size_t column, H, Inputs...)(ref H h, auto ref Inputs inputs)
{
    import mir.ndslice.slice: isSlice;
    static if (column == Inputs.length)
        insertCountViews(h, inputs);
    else static if (isSlice!(Inputs[column]))
        normalizeCountInputs!(column + 1)(h, inputs);
    else
        normalizeCountInputs!(column + 1)(h, inputs[0 .. column],
            inputs[column][].sliced, inputs[column + 1 .. $]);
}
private void putCountRow(size_t column, size_t columns, H, Args...)(
    ref H h, size_t index, auto ref Args args)
{
    static if (column < columns)
        putCountRow!(column + 1, columns)(h, index, args, args[column][index]);
    else
        h.put(args[columns .. $]);
}
private void insertCountViews(H, Inputs...)(ref H h, auto ref Inputs inputs)
{
    static if (!canBatchCounts!(H, Inputs))
    {
        foreach (i; 0 .. inputs[0].length)
            putCountRow!(0, Inputs.length)(h, i, inputs);
    }
    else
    {
        // Avoid typed setup for short rows. Scalar insertion remains unchanged.
        if (inputs[0].length <= 64)
        {
            foreach (i; 0 .. inputs[0].length)
                putCountRow!(0, Inputs.length)(h, i, inputs);
            return;
        }
        auto iterator = h.counts.iterator;
        auto owner = iterator._field.owner;
        assert(iterator._index >= 0);
        const offset = cast(size_t) iterator._index;
        size_t position;
        while (position < inputs[0].length)
        {
            final switch (owner.width)
            {
            case SharedCountState.Width.byte_:
                position = consumeCountBuffer!ubyte(*owner, h, owner.bytes.lightScope, offset, position, inputs); break;
            case SharedCountState.Width.short_:
                position = consumeCountBuffer!ushort(*owner, h, owner.shorts.lightScope, offset, position, inputs); break;
            case SharedCountState.Width.int_:
                position = consumeCountBuffer!uint(*owner, h, owner.ints.lightScope, offset, position, inputs); break;
            case SharedCountState.Width.long_:
                position = consumeCountBuffer!ulong(*owner, h, owner.longs.lightScope, offset, position, inputs); break;
            }
        }
    }
}
// Three-axis batches preserve logical layout, offsets, and saved views on widening.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.topology: stride;
    alias A = IntegralAxis!(uint, AxisOptions());
    auto storage = sharedCountSlice(20);
    auto counts = storage[1 .. 17].stride(2).sliced(2, 2, 2);
    auto h = HistogramAccumulator!(typeof(counts), A, A, A)(counts, A(2,0), A(2,0), A(2,0));
    auto saved = h.counts[1,1,1];
    uint[128] x, y, z;
    foreach (i; 0 .. 128) { x[i]=(i%8)/4; y[i]=(i%4)/2; z[i]=i%2; }
    foreach (i; 0 .. 20) insertSharedCounts(h, x, y, z);
    foreach (i; 0 .. 20)
        assert(storage[i].count() == (i < 17 && i % 2 == 1 ? 320 : 0));
    assert(saved.count() == 320);
}

// Reentrant axis insertion can widen; use the new buffer without replaying mapping.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, AxisOptions());
    static struct CallbackAxis
    {
        A inner;
        alias inner this;
        mir_rcptr!SharedCountState owner;
        size_t calls;
        size_t index(uint value) @safe pure nothrow @nogc
        {
            if (calls++ == 0) owner.increment(0);
            return inner.index(value);
        }
    }
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(ubyte, ushort, uint, ulong))
    {{
        auto storage = sharedCountSlice(1);
        auto owner = storage.iterator._field.owner;
        owner.bytes = typeof(owner.bytes).init;
        enum ulong initial = is(T == ulong) ? ulong.max - 129 : T.max;
        static if (is(T == ubyte)) { owner.bytes=rcslice!T(1); owner.bytes[]=initial; }
        else static if (is(T == ushort)) { owner.shorts=rcslice!T(1); owner.shorts[]=initial; owner.width=SharedCountState.Width.short_; }
        else static if (is(T == uint)) { owner.ints=rcslice!T(1); owner.ints[]=initial; owner.width=SharedCountState.Width.int_; }
        else { owner.longs=rcslice!T(1); owner.longs[]=initial; owner.width=SharedCountState.Width.long_; }
        auto axis = CallbackAxis(A(1,0), owner, 0);
        auto h = HistogramAccumulator!(typeof(storage), CallbackAxis)(storage, axis);
        const uint[128] values = 0;
        h.put(values[]);
        assert(storage[0].count() == initial + 129);
        assert(h.axis[0].calls == 128);
    }}
}

// Non-indexable ranges retain ordinary insertion; empty batches do no work.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, AxisOptions());
    struct Input
    {
        uint front;
        bool empty() const @safe pure nothrow @nogc { return front == 2; }
        void popFront() @safe pure nothrow @nogc { ++front; }
    }
    auto storage = sharedCountSlice(2);
    auto h = HistogramAccumulator!(typeof(storage), A)(storage, A(2,0));
    h.put(Input());
    const uint[] empty;
    h.put(empty);
    uint[0] emptyStatic;
    h.put(emptyStatic);
    assert(storage[0].count() == 1 && storage[1].count() == 1);
}

// Every promotion consumes its triggering row once, including offset storage.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import std.meta: AliasSeq;
    alias A = IntegralAxis!(uint, AxisOptions());
    static foreach (T; AliasSeq!(ubyte, ushort, uint, ulong))
    {{
        auto storage = sharedCountSlice(6);
        auto owner = storage.iterator._field.owner;
        enum ulong initial = is(T == ulong) ? 10 : T.max - 1;
        owner.bytes = typeof(owner.bytes).init;
        static if (is(T == ubyte)) { owner.bytes=rcslice!T(6); owner.bytes[]=initial; }
        else static if (is(T == ushort)) { owner.shorts=rcslice!T(6); owner.shorts[]=initial; owner.width=SharedCountState.Width.short_; }
        else static if (is(T == uint)) { owner.ints=rcslice!T(6); owner.ints[]=initial; owner.width=SharedCountState.Width.int_; }
        else { owner.longs=rcslice!T(6); owner.longs[]=initial; owner.width=SharedCountState.Width.long_; }
        auto counts = storage[1 .. 5].sliced(2, 2);
        auto h = HistogramAccumulator!(typeof(counts), A, A)(counts, A(2,0), A(2,0));
        auto saved = storage[1];
        uint[128] x, y;
        foreach (i; 0 .. 128) { x[i]=cast(uint)(i%2); y[i]=x[i]; }
        insertSharedCounts(h, x[], y[]);
        assert(saved.count() == initial + 64);
        assert(storage[4].count() == initial + 64);
        foreach (i; [0,2,3,5]) assert(storage[i].count() == initial);
    }}
}

// Joint regular axes use the same underflow/overflow classification as scalar put.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: RegularAxis, AxisOptions;
    alias A = RegularAxis!(double, AxisOptions(false, true, true));
    auto batch = sharedCountSlice(16).sliced(4, 4);
    auto scalar = sharedCountSlice(16).sliced(4, 4);
    auto h = HistogramAccumulator!(typeof(batch), A, A)(batch, A(2,0,2), A(2,0,2));
    auto reference = HistogramAccumulator!(typeof(scalar), A, A)(scalar, A(2,0,2), A(2,0,2));
    double[128] x, y;
    foreach (i; 0 .. 128)
    {
        x[i] = cast(int)(i%4) - 0.5;
        y[i] = cast(int)((i/4)%4) - 0.5;
    }
    foreach (repeat; 0 .. 33)
    {
        insertSharedCounts(h, x[], y[]);
        foreach (i; 0 .. x.length) reference.put(x[i], y[i]);
    }
    foreach (i; 0 .. 4)
        foreach (j; 0 .. 4)
        {
            assert(h.counts[i,j].count() == 264);
            assert(h.counts[i,j].count() == reference.counts[i,j].count());
        }
}
// Contiguous joint batches flatten unequal extents and accept strided inputs.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    import mir.ndslice.topology: stride;
    alias A = IntegralAxis!(uint, AxisOptions());
    auto counts = sharedCountSlice(24).sliced(2, 3, 4);
    auto h = HistogramAccumulator!(typeof(counts), A, A, A)(counts, A(2,0), A(3,0), A(4,0));
    uint[288] x, y, z;
    foreach (i; 0 .. 144)
    {
        x[2*i] = cast(uint)(i%24)/12;
        y[2*i] = cast(uint)(i%12)/4;
        z[2*i] = cast(uint)(i%4);
        x[2*i+1] = y[2*i+1] = z[2*i+1] = 99;
    }
    foreach (repeat; 0 .. 44)
        insertSharedCounts(h, x[].sliced.stride(2), y[].sliced.stride(2), z[].sliced.stride(2));
    foreach (i; 0 .. 2)
        foreach (j; 0 .. 3)
            foreach (k; 0 .. 4)
                assert(h.counts[i,j,k].count() == 264);
}
// A custom input field can widen through an alias while producing coordinates.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.stat.descriptive.histogram.axis: IntegralAxis, AxisOptions;
    alias A = IntegralAxis!(uint, AxisOptions());
    static struct Coordinates
    {
        mir_rcptr!SharedCountState owner;
        uint opIndex(ptrdiff_t index) @safe pure nothrow @nogc
        {
            if (index == 0) owner.increment(0);
            return 0;
        }
    }
    auto counts = sharedCountSlice(1);
    auto owner = counts.iterator._field.owner;
    owner.bytes[0] = ubyte.max;
    auto h = HistogramAccumulator!(typeof(counts), A)(counts, A(1,0));
    auto input = FieldIterator!Coordinates(0, Coordinates(owner)).sliced(128);
    static assert(!canBatchCounts!(typeof(h), typeof(input)));
    h.put(input);
    assert(counts[0].count() == ubyte.max + 129);
}
