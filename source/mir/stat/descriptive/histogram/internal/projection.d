/++
Internal helpers for histogram projections.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.projection;

package(mir.stat.descriptive.histogram):

template validMarginalAxes(size_t rank, dimensions...)
{
    import std.traits: isIntegral;
    enum validMarginalAxes = () {
        static if (dimensions.length == 0 || dimensions.length >= rank)
            return false;
        else
        {
            static foreach (i, dimension; dimensions)
            {
                static if (!is(typeof(dimension)))
                    return false;
                else static if (!isIntegral!(typeof(dimension)))
                    return false;
                else static if (dimension < 0 || dimension >= rank)
                    return false;
                else static foreach (previous; dimensions[0 .. i])
                    static if (previous == dimension)
                        return false;
            }
            return true;
        }
    }();
}

// Destination has empty initialized cells, the selected shape, and does not alias
// source. Walk every stored source coordinate once, including end bins.
// Partial ndslices are handles; auto ref preserves nested static-array storage.
void projectCells(size_t rank, alias dimensions, D, S)(ref D destination, auto ref const S source)
{
    import mir.stat.descriptive.histogram.internal.cell: mergeCell;
    import mir.stat.descriptive.histogram.internal.view: needsScopedSliceRow;
    static void add(size_t depth, T, V)(auto ref T destination, auto ref const V value,
        const ref size_t[rank] indices)
    {
        static if (depth == dimensions.length)
            mergeCell(destination, value);
        else
            add!(depth + 1)(destination[indices[dimensions[depth]]], value, indices);
    }

    static void accumulate(size_t depth, T)(ref D destination,
        auto ref const T source, ref size_t[rank] indices)
    {
        static if (depth == rank)
            add!(0)(destination, source, indices);
        else
            foreach (i; 0 .. source.length)
            {
                indices[depth] = i;
                static if (needsScopedSliceRow!T && depth + 1 < rank)
                {
                    // DMD 2.111/2.112 mishandle direct temporary const-row arguments.
                    scope auto row = source[i];
                    accumulate!(depth + 1)(destination, row, indices);
                }
                else
                    accumulate!(depth + 1)(destination, source[i], indices);
            }
    }

    size_t[rank] indices;
    accumulate!(0)(destination, source, indices);
}

// Recursive array traversal must borrow cells that cannot be copied.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Cell
    {
        ulong value;
        @disable this(this);
        void put(ref const Cell other) @safe pure nothrow @nogc { value += other.value; }
    }
    Cell[2][2] source;
    source[0][1].value = 7;
    source[1][0].value = 3;
    Cell[2] marginal;
    enum dimensions = [0];
    projectCells!(2, dimensions)(marginal, source);
    assert(marginal[0].value == 7 && marginal[1].value == 3);
    assert(source[0][1].value == 7 && source[1][0].value == 3);
}
