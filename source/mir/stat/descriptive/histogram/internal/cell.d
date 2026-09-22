/++
Internal helpers for updating and merging histogram cells.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.cell;

import std.traits: Unqual, isNumeric;

package(mir.stat.descriptive.histogram):

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
    enum acceptsCellMerge = acceptsCellSamples!(Cell, const(Unqual!Cell)) ||
        __traits(compiles, {
            Cell destination;
            const(Unqual!Cell) source;
            destination += source;
        });
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
