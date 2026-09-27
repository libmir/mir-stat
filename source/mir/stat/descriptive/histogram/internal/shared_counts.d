/++
Internal fixed-width prototype for count storage accessed through stable state.

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
    Slice!(RCI!ulong) counters;
public:
    this(size_t length) @safe pure nothrow @nogc
    {
        counters = rcslice!ulong(length);
        counters[] = 0;
    }

    // Exercise replacement before introducing narrow counters or widening.
    // Allocate and copy first, so the state changes only when the new buffer
    // is ready. Length and cell indices remain unchanged.
    void replaceBuffer() @safe pure nothrow @nogc
    {
        auto replacement = rcslice!ulong(counters.length);
        foreach (i; 0 .. counters.length)
            replacement[i] = counters[i];
        counters = replacement;
    }
}

struct SharedCountProxy(State)
{
private:
    mir_rcptr!State owner;
    size_t index;
public:
    ulong get() const @safe pure nothrow @nogc
    {
        return owner.counters[index];
    }

    void opUnary(string op : "++")() @safe pure nothrow @nogc
        if (isMutable!State)
    {
        ++owner.counters[index];
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
        assert(index >= 0 && cast(size_t) index < owner.counters.length);
        return SharedCountProxy!State(owner, cast(size_t) index);
    }

    // FieldIterator forwards indexed increments here. The field already owns
    // the state for the duration of the call, so immediate insertion needs no
    // temporary owning proxy (and no extra reference-count updates).
    void opIndexUnary(string op : "++")(ptrdiff_t index) @safe pure nothrow @nogc
        if (isMutable!State)
    {
        assert(index >= 0 && cast(size_t) index < owner.counters.length);
        ++owner.counters[cast(size_t) index];
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
    auto old = state.counters;
    state.replaceBuffer();
    saved++;
    copy.put(0, 1);
    assert(saved.get() == 3 && view[1].count == 3);
    assert(snapshot.count == 1 && old[2] == 1);
    assert(storage[1].get() == 0 && storage[3].get() == 0);
    state.replaceBuffer();
    saved++;
    assert(view[1].count == 4 && old[2] == 1);
    const readOnly = h;
    assert(readOnly.bins[1].count == 4);
    static assert(!__traits(compiles, readOnly.put(0, 1)));
    auto frozen = joint.lightConst;
    assert(frozen[0, 1].get() == 4);
    static assert(!__traits(compiles, frozen[0, 1]++));
    auto independent = sharedCountSlice(8);
    assert(independent[2].get() == 0);
    auto empty = sharedCountSlice(0);
    empty.iterator._field.owner.replaceBuffer();
    assert(empty.length == 0);
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
    assert(saved.get() == 2);
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
