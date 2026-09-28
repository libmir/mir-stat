/++
Internal helpers for updating and merging histogram cells.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.cell;

import std.traits: Unqual, isNumeric;

package(mir.stat.descriptive.histogram):

// Reading is independent of mutation. count() may return a value or a reference;
// readCount below always produces an independent numeric snapshot.
template isReadableCount(Cell)
{
    enum isReadableCount = isNumeric!Cell || __traits(compiles, {
        const Cell cell = Cell.init;
        static assert(isNumeric!(typeof(cell.count())));
    });
}

template CountValueType(Cell)
    if (isReadableCount!Cell)
{
    static if (isNumeric!Cell)
        alias CountValueType = Unqual!Cell;
    else
        alias CountValueType = Unqual!(typeof((const Cell).init.count()));
}

CountValueType!Cell readCount(Cell)(auto ref const Cell cell)
    if (isReadableCount!Cell)
{
    static if (isNumeric!Cell)
        return cell;
    else
        return cell.count();
}

// Check the actual indexed increment. Nested built-in arrays yield lvalues;
// ndslice indexing may instead return a temporary proxy.
template isIncrementableCountStorage(Storage, size_t dimensions = 1)
    if (dimensions > 0)
{
    import mir.ndslice.slice: isSlice;
    import std.traits: isArray;
    static if (isSlice!Storage)
        enum isIncrementableCountStorage = __traits(compiles, {
            Storage storage;
            size_t[dimensions] indices;
            ++storage[indices];
        });
    else static if (dimensions == 1)
        enum isIncrementableCountStorage = __traits(compiles, {
            Storage storage;
            ++storage[0];
        });
    else static if (isArray!Storage)
        enum isIncrementableCountStorage = __traits(compiles, {
            Storage storage;
            static assert(isIncrementableCountStorage!(typeof(storage[0]), dimensions - 1));
        });
    else
        enum isIncrementableCountStorage = false;
}

template acceptsCellSamples(Cell, Samples...)
{
    enum acceptsCellSamples = !isNumeric!(Unqual!Cell) && __traits(compiles, {
        Cell cell;
        Samples samples;
        cell.put(samples);
    });
}

template acceptsCellMerge(Cell)
{
    enum acceptsCellMerge = (acceptsCellSamples!(Cell, const(Unqual!Cell)) ||
        __traits(compiles, {
            Cell destination;
            const(Unqual!Cell) source;
            destination += source;
        }));
}

// Cells with a sample interface or a full-state merge remain accumulator values.
// Reading a count alone must not discard the rest of their state in bin views.
template isCountValue(Cell)
{
    enum isCountValue = isNumeric!Cell ||
        (isReadableCount!Cell && !__traits(hasMember, Cell, "put") &&
            !acceptsCellMerge!(Unqual!Cell));
}

// Both histogram merging and projection must combine full accumulator state.
// Keep the source by reference, including cells with expensive or disabled copies.
void mergeCell(D, S)(ref D destination, auto ref const S source)
    if (is(Unqual!D == Unqual!S) && acceptsCellMerge!D)
{
    static if (acceptsCellSamples!(D, const(Unqual!S)))
        destination.put(source);
    else
        destination += source;
}

// Prefer the accumulator's merge operation and do not copy its source state.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Cell
    {
        int value;
        @disable this(this);
        void put(ref const Cell source) @safe pure nothrow @nogc
        {
            value += source.value;
        }
        void opOpAssign(string op)(ref const Cell source)
            if (op == "+")
        {
            assert(0, "put must take precedence");
        }
    }
    Cell destination;
    Cell source;
    source.value = 7;
    mergeCell(destination, source);
    assert(destination.value == 7 && source.value == 7);
    static assert(!acceptsCellMerge!(const Cell));
}

// Infer the count type from const-readable operations, without a type alias.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct Plain { ulong value; }
    struct GenericGetter { ulong get() const { return 0; } }
    static assert(!isReadableCount!GenericGetter);
    struct WrongValue { string count() const { return ""; } }
    struct MutableRead {
        ulong count() { return 0; }
    }
    struct Good {
        ulong count() const { return 0; }
    }
    struct Floating {
        double count() const { return 0; }
    }
    static assert(isReadableCount!ulong && !isReadableCount!Plain);
    static assert(!isReadableCount!WrongValue && !isReadableCount!MutableRead);
    static assert(isReadableCount!Good && isReadableCount!(const Good));
    static assert(is(CountValueType!(const Good) == ulong));
    static assert(is(CountValueType!Floating == double));
    static assert(is(CountValueType!(const ubyte) == ubyte));
    static assert(!__traits(compiles, CountValueType!Plain));
    assert(readCount(Good()) == 0);
    assert(readCount(Floating()) == 0);
    const ubyte narrow = 3;
    assert(readCount(narrow) == 3);
    struct Reference {
        ulong value;
        ref const(ulong) count() const return { return value; }
    }
    Reference cell = Reference(7);
    auto snapshot = readCount(cell);
    static assert(is(typeof(snapshot) == ulong));
    cell.value = 9;
    assert(snapshot == 7);
    static assert(isIncrementableCountStorage!(ulong[2][3], 2));
    static assert(!isIncrementableCountStorage!(const(ulong)[2][3], 2));
    static assert(!isIncrementableCountStorage!(Good[]));
}
