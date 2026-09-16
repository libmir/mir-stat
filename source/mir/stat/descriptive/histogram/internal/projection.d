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

// Destination is already zeroed, has the selected shape, and does not alias
// source. Walk every stored source coordinate once, including end bins.
// Partial ndslices are handles; auto ref preserves nested static-array storage.
void projectCounts(size_t rank, alias dimensions, D, S)(ref D destination, auto ref const S source)
{
    static void add(size_t depth, T, V)(auto ref T destination, V value,
        const ref size_t[rank] indices)
    {
        static if (depth == dimensions.length)
            destination += value;
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
                accumulate!(depth + 1)(destination, source[i], indices);
            }
    }

    size_t[rank] indices;
    accumulate!(0)(destination, source, indices);
}
