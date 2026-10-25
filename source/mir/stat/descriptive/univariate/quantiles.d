/++
Medians, ordinary and weighted quantiles, and robust measures of spread.

Choose an ordinary or weighted quantile function for the input data.
The `rc` and `make` variants provide allocation choices; $(REF quantileInto, mir, stat, descriptive, univariate, quantiles)
writes into caller-provided output. The algorithm enums describe the available
quantile definitions.

For weighted means and sums, see
$(REF wmean, mir, stat, descriptive, univariate, means) and
$(REF wsum, mir, stat, descriptive, univariate, sums). These functions and weighted quantiles
take observations before weights. Their normalization and weight requirements
depend on the selected statistic; see each function's documentation.

Import `mir.stat.descriptive.univariate.quantiles` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Ordinary quantiles) $(TD
        $(REF median, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF quantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF rcquantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF makeQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF quantileInto, mir, stat, descriptive, univariate, quantiles)$(NBSP)
    ))
    $(TR $(TD Weighted quantiles) $(TD
        $(REF weightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF rcWeightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF makeWeightedQuantile, mir, stat, descriptive, univariate, quantiles)$(NBSP)
    ))
    $(TR $(TD Measures of spread) $(TD
        $(REF interquartileRange, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF medianAbsoluteDeviation, mir, stat, descriptive, univariate, quantiles)$(NBSP)
    ))
    $(TR $(TD Algorithm selection) $(TD
        $(REF QuantileAlgo, mir, stat, descriptive, univariate, quantiles)$(NBSP)
        $(REF WeightedQuantileAlgo, mir, stat, descriptive, univariate, quantiles)$(NBSP)
    ))
))

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall, Ilya Yaroshenko
Copyright: 2022-3 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.univariate.quantiles;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType;
import mir.stat.descriptive.univariate.means: mean;

/++
Computes the median of `slice`.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

Can also pass a boolean variable, `allowModify`, that allows the input slice to
be modified. By default, a reference-counted copy is made.

Params:
    F = output type
    allowModify = Allows the input slice to be modified, default is false
Returns:
    the median of the slice

See_also:
    $(REF mean, mir, stat, descriptive, univariate, means)
+/
template median(F, bool allowModify = false)
{
    import std.traits: Unqual;

    /++
    Params:
        slice = slice
    +/
    @nogc
    meanType!F median(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        static assert (!allowModify ||
                       isMutable!(slice.DeepElement),
                           "allowModify must be false or the input must be mutable");
        alias G = typeof(return);
        size_t len = slice.elementCount;
        assert(len > 0, "median: slice must have length greater than zero");

        import mir.ndslice.topology: as, flattened;

        static if (!allowModify) {
            import mir.ndslice.allocation: rcslice;

            if (len > 2) {
                auto view = slice.lightScope;
                auto val = view.as!(Unqual!(slice.DeepElement)).rcslice;
                auto temp = val.lightScope.flattened;
                return .median!(G, true)(temp);
            } else {
                import std.traits: isIntegral;
                static if ((isIntegral!(slice.DeepElement) ||
                            isFloatingPoint!(slice.DeepElement)) && isFloatingPoint!G) {
                    // Retain integral endpoints until averaging; converting
                    // them independently can lose cancellation at the midpoint.
                    auto flat = slice.flattened;
                    return len == 1 ? cast(G) flat[0] : medianMidpoint!G(flat[0], flat[1]);
                } else {
                    return mean!G(slice);
                }
            }
        } else {
            import mir.ndslice.sorting: partitionAt;

            auto temp = slice.flattened;

            if (len > 5) {
                size_t half_n = len / 2;
                partitionAt(temp, half_n);
                if (len % 2 == 1) {
                    return cast(G) temp[half_n];
                } else {
                    // The lower-half partition cannot change the upper median.
                    // Read it first to avoid combining the final two loads after
                    // partitioning has just written individual elements.
                    auto upper = temp[half_n];
                    // Move the largest value in the first half to half_n - 1.
                    partitionAt(temp[0 .. half_n], half_n - 1);
                    return medianMidpoint!G(temp[half_n - 1], upper);
                }
            } else {
                return smallMedianImpl!(G)(temp);
            }
        }
    }
}

/// ditto
template median(bool allowModify = false)
{
    import core.lifetime: move;
    import mir.primitives: DeepElementType;

    /// ditto
    meanType!(Slice!(Iterator, N, kind))
        median(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        static assert (!allowModify ||
                       isMutable!(DeepElementType!(Slice!(Iterator, N, kind))),
                           "allowModify must be false or the input must be mutable");
        alias F = typeof(return);
        return .median!(F, allowModify)(slice.move);
    }
}

/++
Params:
    ar = array
+/
meanType!(T[]) median(T)(scope const T[] ar...)
{
    import mir.ndslice.slice: sliced;

    alias F = typeof(return);
    return median!(F, false)(ar.sliced);
}

/++
Params:
    sliceLike = type that satisfies `isConvertibleToSlice!T && !isSlice!T`
+/
auto median(T)(T sliceLike)
    if (isConvertibleToSlice!T && !isSlice!T)
{
    import mir.ndslice.slice: toSlice;
    return median(sliceLike.toSlice);
}

/// Median of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x0 = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10, 5].sliced;
    assert(x0.median == 5);

    auto x1 = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10].sliced;
    assert(x1.median == 5);
}

/// Median of dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    auto x0 = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10, 5];
    assert(x0.median == 5);

    auto x1 = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10];
    assert(x1.median == 5);
}

/// Median of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;

    auto x0 = [
        [9.0, 1, 0, 2,  3],
        [4.0, 6, 8, 7, 10]
    ].fuse;

    assert(x0.median == 5);
}

/// Row median of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    auto result = [1.75, 1.75].sliced;

    // Use byDim or alongDim with map to compute median of row/column.
    assert(x.byDim!0.map!median.all!approxEqual(result));
    assert(x.alongDim!1.map!median.all!approxEqual(result));
}

/// Can allow original slice to be modified or set output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x0 = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10, 5].sliced;
    assert(x0.median!true == 5);

    auto x1 = [9, 1, 0, 2, 3, 4, 6, 8, 7, 10].sliced;
    assert(x1.median!(float, true) == 5);
}

/// Arbitrary median
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(median(0, 1, 2, 3, 4) == 2);
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;

    static immutable x = [9.0, 1, 0, 2, 3];
    assert(x.sliced.median == 2);
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [9.0, 1, 0, 2, 3, 4, 6, 8, 7, 10, 5];

    auto x = RCArray!double(11);
    foreach(i, ref e; x)
        e = a[i];

    assert(x.median.approxEqual(5));
}

/++
For integral slices, can pass output type as template parameter to ensure output
type is correct
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [9, 1, 0, 2, 3, 4, 6, 8, 7, 10].sliced;
    assert(x.median!float == 5f);

    auto y = x.median;
    assert(y == 5.0);
    static assert(is(typeof(y) == double));
}

// additional logic tests
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3, 3, 2, 0, 2, 0].sliced;
    assert(x.median!float.approxEqual(2));

    x[] = [2, 2, 4, 0, 4, 3];
    assert(x.median!float.approxEqual(2.5));
    x[] = [1, 4, 5, 4, 4, 3];
    assert(x.median!float.approxEqual(4));
    x[] = [1, 5, 3, 5, 2, 2];
    assert(x.median!float.approxEqual(2.5));
    x[] = [4, 3, 2, 1, 4, 5];
    assert(x.median!float.approxEqual(3.5));
    x[] = [4, 5, 3, 5, 5, 4];
    assert(x.median!float.approxEqual(4.5));
    x[] = [3, 3, 3, 0, 0, 1];
    assert(x.median!float.approxEqual(2));
    x[] = [4, 2, 2, 1, 2, 5];
    assert(x.median!float.approxEqual(2));
    x[] = [2, 3, 1, 4, 5, 5];
    assert(x.median!float.approxEqual(3.5));
    x[] = [1, 1, 4, 5, 5, 5];
    assert(x.median!float.approxEqual(4.5));
    x[] = [2, 4, 0, 5, 1, 0];
    assert(x.median!float.approxEqual(1.5));
    x[] = [3, 5, 2, 5, 4, 2];
    assert(x.median!float.approxEqual(3.5));
    x[] = [3, 5, 4, 1, 4, 3];
    assert(x.median!float.approxEqual(3.5));
    x[] = [4, 2, 0, 3, 1, 3];
    assert(x.median!float.approxEqual(2.5));
    x[] = [100, 4, 5, 0, 5, 1];
    assert(x.median!float.approxEqual(4.5));
    x[] = [100, 5, 4, 0, 5, 1];
    assert(x.median!float.approxEqual(4.5));
    x[] = [100, 5, 4, 0, 1, 5];
    assert(x.median!float.approxEqual(4.5));
    x[] = [4, 5, 100, 1, 5, 0];
    assert(x.median!float.approxEqual(4.5));
    x[] = [0, 1, 2, 2, 3, 4];
    assert(x.median!float.approxEqual(2));
    x[] = [0, 2, 2, 3, 4, 5];
    assert(x.median!float.approxEqual(2.5));
}

// smallMedianImpl tests
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x0 = [9.0, 1, 0, 2, 3].sliced;
    assert(x0.median.approxEqual(2));

    auto x1 = [9.0, 1, 0, 2].sliced;
    assert(x1.median.approxEqual(1.5));

    auto x2 = [9.0, 0, 1].sliced;
    assert(x2.median.approxEqual(1));

    auto x3 = [1.0, 0].sliced;
    assert(x3.median.approxEqual(0.5));

    auto x4 = [1.0].sliced;
    assert(x4.median.approxEqual(1));
}

// Check issue #328 fixed
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.topology: iota;

    auto x = iota(18);
    auto y = median(x);
    assert(y == 8.5);
}

// Average selected observations without overflowing their sum or converting
// wide integers before cancellation. Keep generic arithmetic for other types.
private F medianMidpoint(F, T)(T a, T b)
{
    import std.traits: isIntegral, CommonType;

    static if (isFloatingPoint!T && isFloatingPoint!F) {
        import std.math: fabs;
        alias Work = CommonType!(T, F, double);
        const Work x = a, y = b, sum = x + y;
        // Every float sum fits in double. For wider inputs, halve separately
        // only on overflow; adding first otherwise preserves tiny midpoints.
        static if (T.sizeof >= double.sizeof) {
            if (fabs(sum) > Work.max)
                return cast(F)(x * 0.5 + y * 0.5);
        }
        return cast(F)(sum * 0.5);
    } else static if (isIntegral!T && isFloatingPoint!F) {
        // Two integers narrower than 64 bits have an exact sum in long and
        // double. Wider inputs need the split-integer interpolation path.
        static if (T.sizeof < long.sizeof)
            return cast(F)(cast(double)(cast(long) a + cast(long) b) * 0.5);
        else
            return interpolateQuantile!F(a, b, 0.5);
    } else {
        return cast(F)(a + b) / cast(F) 2;
    }
}

private pure @trusted nothrow @nogc
F smallMedianImpl(F, Iterator)(Slice!Iterator slice)
{
    size_t n = slice.elementCount;

    assert(n > 0, "smallMedianImpl: slice must have elementCount greater than 0");
    assert(n <= 5, "smallMedianImpl: slice must have elementCount of 5 or less");

    import mir.functional: naryFun;
    import mir.ndslice.sorting: medianOf;
    import mir.utility: swapStars;

    auto sliceI0 = slice._iterator;

    if (n == 1) {
        return cast(F) *sliceI0;
    }

    auto sliceI1 = sliceI0;
    ++sliceI1;

    if (n > 2) {
        auto sliceI2 = sliceI1;
        ++sliceI2;
        alias less = naryFun!("a < b");

        if (n == 3) {
            medianOf!less(sliceI0, sliceI1, sliceI2);
            return cast(F) *sliceI1;
        } else {
            auto sliceI3 = sliceI2;
            ++sliceI3;
            if (n == 4) {
                // Put min in slice[0], lower median in slice[1]
                medianOf!less(sliceI0, sliceI1, sliceI2, sliceI3);
                // Ensure slice[2] < slice[3]
                medianOf!less(sliceI2, sliceI3);
                return medianMidpoint!F(*sliceI1, *sliceI2);
            } else {
                auto sliceI4 = sliceI3;
                ++sliceI4;
                medianOf!less(sliceI0, sliceI1, sliceI2, sliceI3, sliceI4);
                return cast(F) *sliceI2;
            }
        }
    } else {
        return medianMidpoint!F(*sliceI0, *sliceI1);
    }
}

// smallMedianImpl tests
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x0 = [9.0, 1, 0, 2, 3].sliced;
    assert(x0.smallMedianImpl!double.approxEqual(2));

    auto x1 = [9.0, 1, 0, 2].sliced;
    assert(x1.smallMedianImpl!double.approxEqual(1.5));

    auto x2 = [9.0, 0, 1].sliced;
    assert(x2.smallMedianImpl!double.approxEqual(1));

    auto x3 = [1.0, 0].sliced;
    assert(x3.smallMedianImpl!double.approxEqual(0.5));

    auto x4 = [1.0].sliced;
    assert(x4.smallMedianImpl!double.approxEqual(1));

    auto x5 = [2.0, 1, 0, 9].sliced;
    assert(x5.smallMedianImpl!double.approxEqual(1.5));

    auto x6 = [1.0, 2, 0, 9].sliced;
    assert(x6.smallMedianImpl!double.approxEqual(1.5));

    auto x7 = [1.0, 0, 9, 2].sliced;
    assert(x7.smallMedianImpl!double.approxEqual(1.5));
}

// Cover the short, small-sort, and partitioned median paths separately from
// quantile interpolation: all must average the selected pair consistently.
version(mir_stat_test)
private @safe pure nothrow @nogc void testFloatingMedianMidpoints(T)()
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;

    T[8] storage;
    const T tiny = T.min_normal * T.epsilon;
    foreach (n; [1, 2, 3, 4, 5, 6, 8]) {
        auto values = storage[0 .. n];
        foreach (value; [T.max, -T.max, tiny, -tiny, T.infinity, -T.infinity]) {
            values[] = value;
            assert(median(values.sliced) == value);
            assert(median!true(values.sliced) == value);
        }
        foreach (i, ref value; values)
            value = cast(T)(n - 1 - i);
        assert(median(values.sliced) == cast(T)(n - 1) / 2);
        // The default overload must leave the input unchanged.
        foreach (i, value; values)
            assert(value == cast(T)(n - 1 - i));
        assert(median!true(values.sliced) == cast(T)(n - 1) / 2);
    }
    T[2] extremes = [-T.max, T.max];
    assert(median(extremes[].sliced) == 0);
    assert(median!true(extremes[].sliced) == 0);
    T[2] infinities = [-T.infinity, T.infinity];
    assert(median(infinities[].sliced).isNaN);
    assert(median!true(infinities[].sliced).isNaN);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    testFloatingMedianMidpoints!float();
    testFloatingMedianMidpoints!double();
    testFloatingMedianMidpoints!real();
}

// Integral addition must not wrap before conversion to the result type.
version(mir_stat_test)
private @safe pure nothrow @nogc void testIntegralMedianMidpoints(T)()
{
    import mir.ndslice.slice: sliced;

    T[6] storage;
    foreach (n; [2, 4, 6]) {
        auto values = storage[0 .. n];
        foreach (value; [T.min, T.max]) {
            values[] = value;
            assert(median(values.sliced) == cast(double) value);
            assert(median!true(values.sliced) == cast(double) value);
        }
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;

    testIntegralMedianMidpoints!int();
    testIntegralMedianMidpoints!uint();
    testIntegralMedianMidpoints!long();
    testIntegralMedianMidpoints!ulong();

    long[6] storage;
    foreach (n; [2, 4, 6]) {
        auto values = storage[0 .. n];
        values[0 .. n / 2] = long.min;
        values[n / 2 .. $] = long.max;
        assert(median(values.sliced) == -0.5);
        assert(median!true(values.sliced) == -0.5);
    }
    int[4] equal = [int.max, int.max, int.max, int.max];
    assert(median!float(equal[].sliced) == cast(float) int.max);
    assert(medianAbsoluteDeviation(equal[].sliced) == 0);
}

// The short no-copy path retains qualifiers, layouts, and generic conversion.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    const double[2] values = [double.max, double.max];
    assert(median(values[].sliced(1, 2).universal) == double.max);
    const double[1] single = [3];
    assert(median(single[].sliced) == 3);

    struct Wrapper {
        double value;
        alias value this;
    }
    Wrapper[4] wrapped = [Wrapper(1), Wrapper(3), Wrapper(2), Wrapper(4)];
    assert(median(wrapped[].sliced) == 2.5);
    assert(median(wrapped[0 .. 2].sliced) == 2);
}

/++
Algorithms used to calculate the quantile of an input `x` at probability `p`.

These algorithms match the same provided in R's (as of version 3.6.2) `quantile`
function. In turn, these were discussed in Hyndman and Fan (1996).

All sample quantiles are defined as weighted averages of consecutive order
statistics. For each `quantileAlgo`, the sample quantile is given by
(using R's 1-based indexing notation):

    (1 - `gamma`) * `x$(SUBSCRIPT j)` + `gamma` * `x$(SUBSCRIPT j + 1)`


where `x$(SUBSCRIPT j)` is the `j`th order statistic. `gamma` is a function of
`j = floor(np + m)` and `g = np + m - j` where `n` is the sample size, `p` is
the probability, and `m` is a constant determined by the quantile type.

$(BOOKTABLE ,
    $(TR
        $(TH Type)
        $(TH m)
        $(TH gamma)
    )
    $(LEADINGROWN 3, Discontinuous sample quantile)
    $(T3 type1, 0, 0 if `g = 0` and 1 otherwise.)
    $(T3 type2, 0, 0.5 if `g = 0` and 1 otherwise.)
    $(T3 type3, -0.5, 0 if `g = 0` and `j` is even and 1 otherwise.)
    $(LEADINGROWN 3, Continuous sample quantile)
    $(T3 type4, 0, `gamma = g`)
    $(T3 type5, 0.5, `gamma = g`)
    $(T3 type6, `p`, `gamma = g`)
    $(T3 type7, `1 - p`, `gamma = g`)
    $(T3 type8, `(p + 1) / 3`, `gamma = g`)
    $(T3 type9, `p / 4 + 3 / 8`, `gamma = g`)
)

References:
    Hyndman, R. J. and Fan, Y. (1996) Sample quantiles in statistical packages, American Statistician 50, 361--365. 10.2307/2684934.

See_also:
    $(LINK2 https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/quantile, quantile)
+/
enum QuantileAlgo {
    /++
    $(H4 Discontinuous sample quantile)

    Inverse of empirical distribution function.
    +/
    type1,
    /++
    Similar to type1, but averages at discontinuities.
    +/
    type2,
    /++
    SAS definition: nearest even order statistic.
    +/
    type3,
    /++
    $(H4 Continuous sample quantile)

    Linear interpolation of the empirical cdf.
    +/
    type4,
    /++
    A piece-wise linear function hwere the knots are the values midway through
    the steps of the empirical cdf. Popular amongst hydrologists.
    +/
    type5,
    /++
    Used by Minitab and by SPSS.
    +/
    type6,
    /++
    This is used by S and is the default for R.
    +/
    type7,
    /++
    The resulting quantile estimates are approximately median-unbiased
    regardless of the distribution of the input. Preferred by Hyndman and Fan
    (1996).
    +/
    type8,
    /++
    The resulting quantile estimates are approximately unbiased for the expected
    order statistics of the input is normally distributed.
    +/
    type9
}

/++
For all $(LREF QuantileAlgo) except $(LINK2 #.QuantileAlgo.type1, $(D QuantileAlgo.type1)) and $(LINK2 #.QuantileAlgo.type3, $(D QuantileAlgo.type3)),
this is an alias to the $(MATHREF stat, meanType) of `T`

For $(LINK2 #.QuantileAlgo.type1, $(D QuantileAlgo.type1)) and $(LINK2 #.QuantileAlgo.type3, $(D QuantileAlgo.type3)), this is an alias to the
$(MATHREF sum, elementType) of `T`.
+/
package(mir.stat)
template quantileType(T, QuantileAlgo quantileAlgo)
{
    static if (quantileAlgo == QuantileAlgo.type1 ||
               quantileAlgo == QuantileAlgo.type3)
    {
        import mir.math.sum: elementType;

        alias quantileType = elementType!T;
    }
    else
    {
        alias quantileType = meanType!T;
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static assert(is(quantileType!(int[], QuantileAlgo.type1) == int));
    static assert(is(quantileType!(double[], QuantileAlgo.type1) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type1) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type2) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type2) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type2) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type3) == int));
    static assert(is(quantileType!(double[], QuantileAlgo.type3) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type3) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type4) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type4) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type4) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type5) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type5) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type5) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type6) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type6) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type6) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type7) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type7) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type7) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type8) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type8) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type8) == float));

    static assert(is(quantileType!(int[], QuantileAlgo.type9) == double));
    static assert(is(quantileType!(double[], QuantileAlgo.type9) == double));
    static assert(is(quantileType!(float[], QuantileAlgo.type9) == float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex: Complex;

    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type1) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type2) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type3) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type4) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type5) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type6) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type7) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type8) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type9) == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.complex: Complex;

    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type1) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type2) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type3) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type4) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type5) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type6) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type7) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type8) == Complex!float));
    static assert(is(quantileType!(Complex!(float)[], QuantileAlgo.type9) == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    static assert(is(quantileType!(Foo[], QuantileAlgo.type7) == float));

    static assert(is(quantileType!(Foo[], QuantileAlgo.type1) == Foo));
    static assert(is(quantileType!(Foo[], QuantileAlgo.type3) == Foo));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex: Complex;
    static struct Foo {
        Complex!float x;
        alias x this;
    }

    static assert(is(quantileType!(Foo[], QuantileAlgo.type7) == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.complex: Complex;
    static struct Foo {
        Complex!float x;
        alias x this;
    }

    static assert(is(quantileType!(Foo[], QuantileAlgo.type7) == Complex!float));
}

// Keep selected observations exact until interpolation or subtraction is required.
private struct QuantileParts(T, G)
{
    T low, high;
    G fraction;
}

// Splitting 64-bit integers before conversion preserves cancellation such as the
// midpoint of long.min and long.max, even when real has only 53 significand bits.
private F interpolateQuantile(F, T, G)(T a, T b, G fraction)
{
    import std.traits: isIntegral, CommonType;
    static if (isIntegral!T)
    {
        enum long radix = 1L << 32;
        const real t = fraction;
        const real high = (1 - t) * (a / radix) + t * (b / radix);
        const real low = (1 - t) * (a % radix) + t * (b % radix);
        return cast(F)(high * radix + low);
    }
    else
    {
        alias Work = CommonType!(T, F, G, double);
        const Work t = fraction;
        const Work x = a, y = b;
        import std.math: isFinite;
        if (!isFinite(x) || !isFinite(y)) return cast(F)(x * (1 - t) + y * t);
        // Adding opposite signs cannot overflow. At the midpoint, add before
        // halving so subnormal operands are not rounded separately.
        if (t == 0.5 && (x < 0) != (y < 0)) return cast(F)((x + y) / 2);
        // Preserve subnormals for nearby same-sign endpoints; avoid overflow
        // in their difference when signs differ.
        return cast(F)((x < 0) != (y < 0) ?
            x * (1 - t) + y * t : x + (y - x) * t);
    }
}

@fmamath private @safe pure nothrow @nogc
auto quantileImpl(F, QuantileAlgo quantileAlgo, Iterator, G)(Slice!Iterator slice, G p)
    if ((isFloatingPoint!F || quantileAlgo == QuantileAlgo.type1 ||
         quantileAlgo == QuantileAlgo.type3) && isFloatingPoint!G)
{
    return selectQuantile!(quantileAlgo, F)(slice, p);
}

// Share rank calculation and partitioning, but select the result representation
// at compile time. Scalar callers interpolate directly; callers such as integral
// IQR retain the endpoints and fraction until after subtracting the quantiles.
// Package visibility also lets robustScale retain exact integral median parts.
@fmamath package(mir.stat) @safe pure nothrow @nogc
auto selectQuantile(QuantileAlgo quantileAlgo, F = void, Iterator, G)(Slice!Iterator slice, G p)
    if (isFloatingPoint!G)
{
    assert(p >= 0 && p <= 1, "quantileImpl: p must be between 0 and 1");
    size_t n = slice.elementCount;
    assert(n > 1, "quantileImpl: slice.elementCount must be greater than 1");

    import mir.math.common: floor;
    import mir.ndslice.sorting: partitionAt;
    import std.traits: Unqual, CommonType;
    import mir.primitives: DeepElementType;
    alias T = Unqual!(DeepElementType!(typeof(slice)));

    alias GG = CommonType!(Unqual!G, double);
    static if (is(F == void)) alias Parts = QuantileParts!(T, GG);
    const GG probability = p;

    GG m;

    static if (quantileAlgo == QuantileAlgo.type1) {
        m = 0;
    } else static if (quantileAlgo == QuantileAlgo.type2) {
        m = 0;
    } else static if (quantileAlgo == QuantileAlgo.type3) {
        m = -0.5;
    } else static if (quantileAlgo == QuantileAlgo.type4) {
        m = 0;
    } else static if (quantileAlgo == QuantileAlgo.type5) {
        m = 0.5;
    } else static if (quantileAlgo == QuantileAlgo.type6) {
        m = probability;
    } else static if (quantileAlgo == QuantileAlgo.type7) {
        m = 1 - probability;
    } else static if (quantileAlgo == QuantileAlgo.type8) {
        m = (probability + 1) / 3;
    } else static if (quantileAlgo == QuantileAlgo.type9) {
        m = probability / 4 + cast(GG) 3 / 8;
    }

    GG g = n * probability + m - 1; //note: 0-based, not 1-based indexing

    GG pre_j = floor(g);
    GG pre_j_1 = pre_j + 1;
    size_t j;
    if (pre_j >= (n - 1)) { //note: 0-based, not 1-based indexing
        j = n - 1;
    } else if (pre_j < 0) {
        j = 0;
    } else {
        j = cast(size_t) pre_j;
    }

    size_t j_1;
    if (pre_j_1 >= (n - 1)) { //note: 0-based, not 1-based indexing
        j_1 = n - 1;
    } else if (pre_j_1 < 0) {
        j_1 = 0;
    } else {
        j_1 = cast(size_t) pre_j_1;
    }

    g -= j;
    GG gamma;

    static if (quantileAlgo == QuantileAlgo.type1) {
        if (g == 0) {
            gamma = 0;
        } else {
            gamma = 1;
        }
    } else static if (quantileAlgo == QuantileAlgo.type2) {
        if (g == 0) {
            gamma = 0.5;
        } else {
            gamma = 1;
        }
    } else static if (quantileAlgo == QuantileAlgo.type3) {
        if (g == 0 && (j + 1) % 2 == 0) { //need to adjust because 0-based indexing
            gamma = 0;
        } else {
            gamma = 1;
        }
    } else {
        gamma = g;
    }

    if (gamma == 0) {
        partitionAt(slice, j);
        static if (is(F == void)) return Parts(slice[j], slice[j], 0);
        else return cast(F) slice[j];
    } else if (gamma == 1) {
        partitionAt(slice, j_1);
        static if (is(F == void)) return Parts(slice[j_1], slice[j_1], 0);
        else return cast(F) slice[j_1];
    } else if (j != j_1) {
        partitionAt(slice, j_1);
        partitionAt(slice[0 .. j_1], j);
        static if (is(F == void)) return Parts(slice[j], slice[j_1], gamma);
        else return interpolateQuantile!F(slice[j], slice[j_1], gamma);
    } else {
        partitionAt(slice, j);
        static if (is(F == void)) return Parts(slice[j], slice[j], 0);
        else return cast(F) slice[j];
    }
}

// Returning a scalar directly must preserve tiny floating-point observations
// and the midpoint rounding safeguards for each supported floating type.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T tiny = T.min_normal * T.epsilon;
        T[2] equal = [tiny, tiny];
        assert((quantileImpl!(T, QuantileAlgo.type7)(equal[].sliced, 0.5)) == tiny);
        T[2] mixed = [-tiny, 2 * tiny];
        assert((quantileImpl!(T, QuantileAlgo.type7)(mixed[].sliced, 0.5)) == 0);
        T[2] extremes = [-T.max, T.max];
        assert((quantileImpl!(T, QuantileAlgo.type7)(extremes[].sliced, 0.5)) == 0);
        T[2] infinite = [T.infinity, T.infinity];
        assert((quantileImpl!(T, QuantileAlgo.type7)(infinite[].sliced, 0.5)) == T.infinity);
    }}
    // Round to the requested result type only after interpolating observations
    // whose precision may exceed both the result and probability types.
    real[2] precise = [-1.0L, 1.0L + real.epsilon];
    assert(rcquantile!double(precise, 0.5) == cast(double)(real.epsilon / 2));
    double[2] wider = [-1.0, 1.0 + double.epsilon];
    assert(rcquantile!float(wider, 0.5f) == cast(float)(double.epsilon / 2));
}

/++
Algorithms for weighted quantile estimation.

By default, weights represent relative probability masses and need not sum to
one. The probability-weight algorithms have their own definitions, documented
below; they do not simply replicate observations and apply an unweighted type.

$(LREF frequencyType7) and $(LREF frequencyType8) instead interpret weights as
whole-number occurrence counts. They reproduce the corresponding ordinary
quantile types for the conceptual expanded sample without allocating it.
Select the interpretation through the algorithm, not the weights' numeric type.
+/
enum WeightedQuantileAlgo
{
    /++
    Inverse weighted empirical cumulative distribution function.

    Use this to select an observed outcome at a cumulative probability threshold.
    For example, after reweighting simulated outcomes, the 90th percentile is
    the smallest outcome at or below which at least 90 percent of the weight
    lies. This is the default when the weighted empirical distribution itself
    is the object of interest and no interpolation is wanted.

    After ignoring zero weights, sort observations by value. For probability
    $(TT 0 < p <= 1), return the smallest observed value whose cumulative weight
    is at least $(TT p * totalWeight). At $(TT p = 0), return the smallest
    positive-weight observation. At $(TT p = 1), return the largest.

    No interpolation is performed. Equal values contribute their combined mass.
    With equal positive weights this corresponds mathematically to
    $(LINK2 #.QuantileAlgo.type1, $(D QuantileAlgo.type1)), not the type7 default of $(LREF quantile).
    Quantiles can jump when a probability crosses a cumulative-weight boundary.
    +/
    inverseCDF,
    /++
    Weighted Harrell-Davis smoothing estimator following Akinshin.

    Use this when estimating quantiles that should change smoothly as observation
    weights change. For example, a time series can give more weight to recent
    measurements and gradually less weight to older ones. Averaging contributions
    across observations avoids the abrupt switches between observed values made
    by inverseCDF, but can make the estimate sensitive to extreme observations.

    Let $(TT t_i) be cumulative normalized weights of sorted observations, with
    $(TT t_0 = 0) and $(TT t_n = 1). Compute Kish effective sample size
    $(TT nEff = (sum w)^2 / sum(w^2)) from the original positive-weight rows.
    For $(TT 0 < p < 1), set $(TT a = (nEff + 1)*p) and
    $(TT b = (nEff + 1)*(1-p)). Observation i receives coefficient
    $(TT I(t_i; a, b) - I(t_(i-1); a, b)), where I is the regularized
    incomplete beta CDF. The estimate is the sum of coefficient times value.

    This smooths across observations rather than selecting an exact empirical
    quantile. Equal weights recover ordinary Harrell-Davis, not type7.
    Small changes in weights give continuous estimates at interior probabilities,
    but extreme observations can have substantial influence. It is not a robust
    replacement for inverseCDF and does not promise greater statistical efficiency
    for every distribution.

    Do not combine tied rows before calculating nEff: splitting a weight between
    identical observations changes nEff and can change the estimate, even though
    the empirical distribution is unchanged. Rescaling every weight by the same
    positive constant does not change the mathematical estimate.

    As an explicit endpoint extension, p=0 and p=1 return the positive-weight
    minimum and maximum. These endpoint values need not be continuous as a
    weight reaches zero. Integral observations infer double output; floating
    observations retain their type. An explicit output type must be floating-point.

    References:
        $(LINK2 https://arxiv.org/abs/2304.07265, Akinshin (2023), Weighted quantile estimators).
    +/
    harrellDavis,
    /++
    Trimmed weighted Harrell-Davis estimator following Akinshin.

    Use this when Harrell-Davis smoothing is useful but extreme observations
    should have less opportunity to influence a central estimate. For example,
    an unusually large measurement can affect an HD median even with a small
    weight; trimming can exclude its contribution. This is an optional estimate
    with restricted contributions, not a guarantee that outliers are excluded
    at every requested probability.

    Use the effective sample size and beta shapes of $(LREF harrellDavis).
    Choose the interval [L, R] of width $(TT 1 / sqrt(nEff)) that contains
    the greatest beta probability. Intersect each observation's cumulative
    weight interval with [L, R], integrate the beta density over that
    intersection, and normalize the retained coefficients to sum to one.
    The width is measured on the cumulative probability axis; it is not
    the amount of beta probability retained.

    Observations outside this interval contribute nothing to the estimate.
    This can reduce outlier influence, but upper or lower quantiles may still
    include extreme observations. Renormalization can increase their influence
    when they remain inside the interval. This does not trim histogram tails.

    Equal weights recover trimmed ordinary Harrell-Davis with this width.
    Effective size is computed before combining ties, so splitting a row can
    change the result. Weight rescaling, output types, and the explicit p=0/1
    minimum/maximum extension follow $(LREF harrellDavis).

    References:
        $(LINK2 https://arxiv.org/abs/2304.07265, Akinshin (2023), Weighted quantile estimators).
    +/
    trimmedHarrellDavis,
    /++
    Weighted type 7 using Akinshin's effective-size uniform interval.

    Use this for probability or importance weights when ordinary type 7 is the
    desired equal-weight convention. For example, simulated outcomes can start
    with equal weights and later be reweighted to represent another scenario.
    This provides interpolation using arithmetic interval overlaps, without the
    beta-function evaluations required by Harrell-Davis. For weights recording
    how many times each outcome occurred, use $(LREF frequencyType7).

    Compute $(TT nEff = (sum w)^2 / sum(w^2)) from the original positive-weight
    rows. Set $(TT h = (nEff - 1)*p + 1), clamped to [1, nEff]. Each sorted
    observation receives its normalized cumulative-weight interval's overlap
    with $(TT [(h-1)/nEff, h/nEff]), divided by the width $(TT 1/nEff).
    The estimate is the sum of these coefficients times observations.

    Equal weights recover $(LINK2 #.QuantileAlgo.type7, $(D QuantileAlgo.type7)). Unequal weights can
    give contributions from more than two observations. This definition uses
    probability weights, not replication counts. Rescaling all weights has no
    mathematical effect, while splitting a row can change nEff and the result.
    Zero-weight rows are ignored; interior estimates vary continuously as
    weights tend to zero. Do not combine ties before computing nEff.

    At p=0 and p=1 this API explicitly returns the positive-weight extrema.
    The interior limits need not equal these endpoints: for values [0,10,20]
    and weights [1,1,8], the limit as p approaches zero is approximately 15.45,
    while the value at p=0 is 0. This is part of the definition, not roundoff.
    Integral observations infer double output; floating observations retain
    their type. Explicit output types must be floating-point.

    References:
        $(LINK2 https://arxiv.org/abs/2304.07265, Akinshin (2023), Weighted quantile estimators).
    +/
    type7,
    /++
    Weighted type 8 using Akinshin's effective-size uniform interval.

    Use this for probability or importance weights when ordinary type 8 is the
    desired equal-weight convention. Like $(LREF type7), it offers arithmetic
    interpolation as weights change, without beta-function evaluations. For
    example, use it when reweighting simulated outcomes whose original quantiles
    were calculated with type 8. Use $(LREF frequencyType8) when the weights
    instead count repeated observations.

    Uses the same uniform-interval construction as $(LREF type7), with
    $(TT h = (nEff + 1/3)*p + 1/3), clamped to [1, nEff].

    Equal weights recover $(LINK2 #.QuantileAlgo.type8, $(D QuantileAlgo.type8)). The approximate
    median-unbiasedness of unweighted type 8 is not promised for arbitrary
    weights. The weight, output-type, and endpoint rules of $(LREF type7)
    apply, including possible jumps at p=0 and p=1. Clamping h can produce
    constant estimates over intervals of interior probabilities.
    +/
    type8,
    /++
    Inverse weighted empirical CDF with averaging at exact boundaries.

    Use this when an empirical quantile should use the midpoint if the requested
    probability falls exactly between two distinct outcomes. For example, when
    half the total weight belongs to values at or below 0 and the next observed
    value is 10, this convention gives a median of 5. Away from an exact boundary
    it selects an observed value, so it does not provide continuous smoothing.

    Treat equal observations as one value with their combined weight. As with
    $(LREF inverseCDF), select the first value whose cumulative weight reaches
    the target. If the target exactly equals the cumulative weight through
    that entire group, average that value and the next distinct value.
    Otherwise return the selected value. Zero-weight rows are ignored.

    For equal positive weights this agrees mathematically with
    $(LINK2 #.QuantileAlgo.type2, $(D QuantileAlgo.type2)). For example, the median of equally weighted
    [0,10] is 5. A boundary inside a group of equal values does not trigger
    averaging with the next distinct value. This estimator remains discontinuous
    when probabilities or weights cross a cumulative-weight boundary.

    p=0 and p=1 return the positive-weight minimum and maximum. Boundary
    equality is tested on computed cumulative masses without a tolerance;
    roundoff can affect whether averaging occurs. Integral observations infer
    double output; floating observations retain their type. Explicit output
    types must be floating-point. No effective sample size is used.
    +/
    averagedInverseCDF,
    /++
    Compute ordinary type 7 quantiles from observations stored with occurrence counts.

    Use this when repeated observations have been summarized in a frequency
    table, such as the number of customers giving each survey rating. A value
    of 4 with frequency 12 represents twelve observations equal to 4. This
    option gives the quantiles of the original sample without reconstructing it.
    For example, values [0,10] with frequencies [2,2] represent [0,0,10,10];
    the type 7 first quartile is 0.

    Choose $(LREF type7) instead when weights describe relative probability or
    importance rather than occurrence counts. That option gives 2.5 for the
    same first-quartile example. Increasing every occurrence count changes the
    sample size, whereas rescaling probability weights leaves their relative
    contributions unchanged. Select the interpretation explicitly, regardless
    of whether the weights are stored as integers or floating-point values.

    Each weight must be a finite nonnegative whole number. Integer and
    floating-point weight types are accepted, but their type never selects
    this interpretation automatically. Zero counts are ignored. The positive
    total N must not exceed the smaller of ulong.max and 2^^real.mant_dig,
    so every cumulative count is exactly representable in the working precision.
    Invalid counts and excessive totals are rejected with assertions.

    Use the ordinary type 7 rank $(TT h = (N-1)*p + 1). Locate the adjacent
    order statistics through cumulative integer counts and interpolate; no
    expanded sample is allocated. Multiplying every frequency can change the
    answer. Splitting or combining identical observations preserves the
    conceptual sample. Floating-point rank arithmetic can round near boundaries.

    p=0/1 return the positive-count extrema. Integral observations infer double;
    floating observations retain their type. Explicit F must be floating-point.
    +/
    frequencyType7,
    /++
    Compute ordinary type 8 quantiles from observations stored with occurrence counts.

    Use this for the same frequency-table inputs as $(LREF frequencyType7)
    when the desired unweighted quantile convention is $(LINK2 #.QuantileAlgo.type8, $(D QuantileAlgo.type8)).
    For example, if survey ratings are stored with the number of responses for
    each rating, this produces type 8 quantiles of those responses without
    allocating one observation per response. Choose $(LREF type8) when the
    weights instead express relative probability or importance.

    Uses rank $(TT h = (N + 1/3)*p + 1/3), clamped to [1,N], where N is the
    total number of occurrences. The count validation, allocation, endpoint,
    and output rules of $(LREF frequencyType7) apply.
    +/
    frequencyType8,
}

private enum weightedQuantileUsesEffectiveSize(WeightedQuantileAlgo algorithm) =
    algorithm == WeightedQuantileAlgo.harrellDavis ||
    algorithm == WeightedQuantileAlgo.trimmedHarrellDavis ||
    algorithm == WeightedQuantileAlgo.type7 || algorithm == WeightedQuantileAlgo.type8;

private enum weightedQuantileUsesFrequencies(WeightedQuantileAlgo algorithm) =
    algorithm == WeightedQuantileAlgo.frequencyType7 || algorithm == WeightedQuantileAlgo.frequencyType8;

// Counts must also be exact in the floating-point rank calculation.
static if (real.mant_dig < 64)
    private enum ulong weightedQuantileMaxFrequency = 1UL << real.mant_dig;
else
    private enum ulong weightedQuantileMaxFrequency = ulong.max;

private struct WeightedQuantileEntry(T, WeightedQuantileAlgo algorithm)
{
    T value;
    static if (weightedQuantileUsesFrequencies!algorithm)
        ulong cumulative;
    else
        real cumulative;
    static if (weightedQuantileUsesEffectiveSize!algorithm)
        real upper;
}

private template WeightedQuantileResult(Data, WeightedQuantileAlgo algorithm)
{
    import mir.primitives: DeepElementType;
    import std.traits: Unqual;
    alias Element = Unqual!(DeepElementType!Data);
    static if (algorithm != WeightedQuantileAlgo.inverseCDF)
        alias WeightedQuantileResult = quantileType!(Element, QuantileAlgo.type7);
    else
        alias WeightedQuantileResult = Element;
}

// Fixed-width maximum-mass interval for the beta shapes reachable here:
// a+b=nEff+1>=2, so both shapes cannot be below one. Boundary modes have
// one-sided intervals; interior modes have equal endpoint densities.
private real weightedTrimmedHDLower(real a, real b, real width)
    @safe pure nothrow @nogc
{
    import std.algorithm: min, max;
    import std.math: log1p;
    if (width >= 1 || a <= 1) return 0;
    if (b <= 1) return 1 - width;
    if (a == b) return (1 - width) / 2;
    const mode = (a - 1) / (a + b - 2);
    real low = max(0.0L, mode - width), high = min(mode, 1 - width);
    // Stop when the bracket has no representable midpoint. A bound also
    // handles roots very close to zero without a long subnormal search.
    foreach (_; 0 .. 256)
    {
        const mid = low + (high - low) / 2;
        if (mid == low || mid == high) break;
        const difference = (a - 1) * log1p(-width / (mid + width))
            + (b - 1) * log1p(width / (1 - mid - width));
        if (difference < 0) low = mid;
        else high = mid;
    }
    return low + (high - low) / 2;
}

// Both prefixes and suffixes were accumulated independently. Subtracting a
// rounded prefix from one would lose precisely the small upper masses we need.
private auto weightedHarrellDavisAt(F, bool trimmed, Workspace)(scope Workspace workspace,
    real probability, real effectiveSize)
{
    import mir.stat.distribution.beta: betaCDF;
    import std.math: fabs;
    const lowValue = cast(real) workspace[0].value;
    const highValue = cast(real) workspace[$ - 1].value;
    if (workspace[0].value == workspace[$ - 1].value)
        return cast(F) workspace[0].value;
    real scale = 0;
    const a = (effectiveSize + 1) * probability;
    const b = (effectiveSize + 1) * (1 - probability);
    const total = workspace[$ - 1].cumulative;
    Summator!(real, Summation.kahan) coefficients, estimate;
    static if (trimmed)
    {
        import std.algorithm: min, max;
        import std.math: sqrt;
        const width = min(1.0L, 1 / sqrt(effectiveSize));
        const lower = weightedTrimmedHDLower(a, b, width);
        const upper = min(1.0L, lower + width);
        // Excluded observations must not set the scale: a huge excluded value
        // could otherwise make every retained value underflow on division.
        foreach (i; 0 .. workspace.length)
        {
            const left = i == 0 ? 0.0L : workspace[i - 1].cumulative / total;
            const right = i + 1 == workspace.length ? 1.0L : workspace[i].cumulative / total;
            const leftTail = left >= lower ? workspace[i].upper / total : 1 - lower;
            const rightTail = right <= upper ?
                (i + 1 == workspace.length ? 0.0L : workspace[i + 1].upper / total) : 1 - upper;
            if (min(right, upper) <= max(left, lower) && rightTail >= leftTail) continue;
            scale = max(scale, fabs(cast(real) workspace[i].value));
        }
        if (scale == 0) return cast(F) 0;
    }
    else
        scale = fabs(lowValue) > fabs(highValue) ? fabs(lowValue) : fabs(highValue);
    real lowerCDF = 0;
    foreach (i; 0 .. workspace.length)
    {
        static if (trimmed)
        {
            const left = i == 0 ? 0.0L : workspace[i - 1].cumulative / total;
            const right = i + 1 == workspace.length ? 1.0L : workspace[i].cumulative / total;
            const clippedLeft = max(left, lower), clippedRight = min(right, upper);
            // Preserve independently accumulated suffixes for unclipped edges.
            const leftTail = left >= lower ? workspace[i].upper / total : 1 - lower;
            const rightTail = right <= upper ?
                (i + 1 == workspace.length ? 0.0L : workspace[i + 1].upper / total) : 1 - upper;
            // Equal rounded prefixes may still enclose a nonzero upper mass.
            if (clippedRight <= clippedLeft && rightTail >= leftTail) continue;
            lowerCDF = betaCDF(clippedLeft, a, b);
            const upperCDF = betaCDF(clippedRight, a, b);
        }
        else
            const upperCDF = i + 1 == workspace.length ? 1.0L :
                betaCDF(workspace[i].cumulative / total, a, b);
        real coefficient;
        if (upperCDF <= 0.5L)
            coefficient = upperCDF - lowerCDF;
        else
        {
            static if (trimmed)
                const upperTail = betaCDF(rightTail, b, a);
            else
                const upperTail = i + 1 == workspace.length ? 0.0L :
                    betaCDF(workspace[i + 1].upper / total, b, a);
            if (lowerCDF >= 0.5L)
            {
                static if (trimmed)
                    coefficient = betaCDF(leftTail, b, a) - upperTail;
                else
                    coefficient = betaCDF(workspace[i].upper / total, b, a) - upperTail;
            }
            else
                coefficient = 1 - lowerCDF - upperTail;
        }
        // Roundoff in nearly identical beta evaluations must not create a
        // negative averaging weight. Renormalize the resulting coefficients.
        if (coefficient < 0)
            coefficient = 0;
        coefficients.put(coefficient);
        estimate.put(coefficient * (cast(real) workspace[i].value / scale));
        lowerCDF = upperCDF;
    }
    assert(coefficients.sum > 0, "weightedQuantile: invalid Harrell-Davis coefficients");
    auto normalized = estimate.sum / coefficients.sum;
    const low = lowValue / scale, high = highValue / scale;
    if (normalized < low) normalized = low;
    if (normalized > high) normalized = high;
    return cast(F) (normalized * scale);
}

private real weightedUniformOverlap(Workspace)(scope Workspace workspace,
    size_t i, real lower, real upper, real total, bool reflected)
{
    import std.algorithm: min, max;
    const left = reflected ?
        (i + 1 == workspace.length ? 0.0L : workspace[i + 1].upper / total) :
        (i == 0 ? 0.0L : workspace[i - 1].cumulative / total);
    const right = reflected ? workspace[i].upper / total :
        (i + 1 == workspace.length ? 1.0L : workspace[i].cumulative / total);
    return min(right, upper) - max(left, lower);
}

// Uniform-window counterpart of the beta-weighted estimators. Use suffixes
// for upper windows so small upper masses survive rounded cumulative prefixes.
private auto weightedUniformQuantileAt(F, WeightedQuantileAlgo algorithm, Workspace)(
    scope Workspace workspace, real probability, real effectiveSize)
{
    import std.algorithm: min, max;
    import std.math: fabs;
    const lowValue = cast(real) workspace[0].value;
    const highValue = cast(real) workspace[$ - 1].value;
    if (workspace[0].value == workspace[$ - 1].value)
        return cast(F) workspace[0].value;
    // Reflect upper probabilities before computing h, avoiding 1 - roundedCDF.
    const reflected = probability > 0.5L;
    const p = reflected ? 1 - probability : probability;
    static if (algorithm == WeightedQuantileAlgo.type7)
        const rawH = (effectiveSize - 1) * p + 1;
    else
        const rawH = (effectiveSize + 1.0L / 3) * p + 1.0L / 3;
    const h = min(effectiveSize, max(1.0L, rawH));
    const lower = (h - 1) / effectiveSize, upper = h / effectiveSize;
    const total = workspace[$ - 1].cumulative;
    real scale = 0;
    foreach (i; 0 .. workspace.length)
        if (weightedUniformOverlap(workspace, i, lower, upper, total, reflected) > 0)
            scale = max(scale, fabs(cast(real) workspace[i].value));
    if (scale == 0) return cast(F) 0;
    Summator!(real, Summation.kahan) coefficients, estimate;
    foreach (i; 0 .. workspace.length)
    {
        const overlap = weightedUniformOverlap(workspace, i, lower, upper, total, reflected);
        if (overlap <= 0) continue;
        // The common width cancels when normalizing the retained overlaps.
        coefficients.put(overlap);
        estimate.put(overlap * (cast(real) workspace[i].value / scale));
    }
    assert(coefficients.sum > 0, "weightedQuantile: invalid uniform coefficients");
    const normalized = min(highValue / scale, max(lowValue / scale,
        estimate.sum / coefficients.sum));
    return cast(F) (normalized * scale);
}

private auto frequencyQuantileAt(F, WeightedQuantileAlgo algorithm, Workspace)(
    scope Workspace workspace, real probability)
{
    import std.math: floor;
    const total = workspace[$ - 1].cumulative;
    static if (algorithm == WeightedQuantileAlgo.frequencyType7)
        const rank = cast(real)(total - 1) * probability;
    else
        // Equivalent to (N+1/3)*p-2/3, with an exact zero correction at p=0.5.
        const rank = cast(real)(total - 1) * probability + (4 * probability - 2) / 3;
    if (rank <= 0) return cast(F) workspace[0].value;
    if (rank >= cast(real)(total - 1)) return cast(F) workspace[$ - 1].value;
    const index = cast(ulong) floor(rank);
    const fraction = rank - index;
    size_t first = 0, last = workspace.length - 1;
    while (first < last)
    {
        const mid = first + (last - first) / 2;
        if (workspace[mid].cumulative <= index) first = mid + 1;
        else last = mid;
    }
    if (fraction == 0 || workspace[first].cumulative > index + 1)
        return cast(F) workspace[first].value;
    // Every stored count is positive, so the next row owns the next rank.
    return interpolateQuantile!F(workspace[first].value, workspace[first + 1].value, fraction);
}

private auto weightedQuantileAt(F, WeightedQuantileAlgo algorithm, Workspace, P)(
    scope Workspace workspace, P probability, real effectiveSize)
{
    import std.math: isFinite;
    static assert(isFloatingPoint!P, "Probabilities must be floating-point values");
    assert(isFinite(probability) && probability >= 0 && probability <= 1,
        "weightedQuantile: probability must be finite and in [0, 1]");
    if (probability == 0)
        return cast(F) workspace[0].value;
    if (probability == 1)
        return cast(F) workspace[$ - 1].value;
    static if (weightedQuantileUsesFrequencies!algorithm)
        return frequencyQuantileAt!(F, algorithm)(workspace, cast(real) probability);
    else static if (algorithm == WeightedQuantileAlgo.type7 || algorithm == WeightedQuantileAlgo.type8)
        return weightedUniformQuantileAt!(F, algorithm)(workspace, cast(real) probability, effectiveSize);
    else static if (algorithm == WeightedQuantileAlgo.harrellDavis ||
        algorithm == WeightedQuantileAlgo.trimmedHarrellDavis)
        return weightedHarrellDavisAt!(F, algorithm == WeightedQuantileAlgo.trimmedHarrellDavis)(workspace, cast(real) probability, effectiveSize);
    else
    {
        const target = cast(real) probability * workspace[$ - 1].cumulative;
        size_t low = 0;
        size_t high = workspace.length - 1;
        while (low < high)
        {
            const mid = low + (high - low) / 2;
            if (workspace[mid].cumulative < target)
                low = mid + 1;
            else
                high = mid;
        }
        static if (algorithm == WeightedQuantileAlgo.averagedInverseCDF)
        {
            // Locate the end of the equal-value group without changing scratch
            // storage or adding a linear scan per probability.
            size_t next = low + 1, end = workspace.length;
            while (next < end)
            {
                const mid = next + (end - next) / 2;
                if (workspace[mid].value == workspace[low].value)
                    next = mid + 1;
                else
                    end = mid;
            }
            if (next < workspace.length && workspace[next - 1].cumulative == target)
            {
                // Same-sign subtraction and opposite-sign addition cannot
                // overflow. Avoid halving each operand, which loses subnormals.
                return interpolateQuantile!F(workspace[low].value, workspace[next].value, 0.5L);
            }
        }
        return cast(F) workspace[low].value;
    }
}

private auto allocatedWeightedQuantile(F, WeightedQuantileAlgo algorithm,
    QuantileAllocation allocation, Context, Weights, Data, P)(
    ref Context context, scope auto ref Weights weights,
    scope auto ref Data data, scope auto ref P probabilities)
{
    import mir.ndslice.sorting: sort;
    import mir.primitives: DeepElementType;
    import std.traits: Unqual, isNumeric;
    import std.math: isFinite;

    static assert(algorithm == WeightedQuantileAlgo.inverseCDF ||
        algorithm == WeightedQuantileAlgo.averagedInverseCDF ||
        weightedQuantileUsesFrequencies!algorithm ||
        algorithm == WeightedQuantileAlgo.harrellDavis ||
        algorithm == WeightedQuantileAlgo.trimmedHarrellDavis ||
        algorithm == WeightedQuantileAlgo.type7 || algorithm == WeightedQuantileAlgo.type8,
        "Unsupported weighted quantile algorithm");
    scope auto values = quantileSlice(data);
    scope auto masses = quantileSlice(weights);
    static assert(typeof(values).N == 1 && typeof(masses).N == 1,
        "Weighted quantile values and weights must be one-dimensional");
    alias Element = Unqual!(DeepElementType!(typeof(values)));
    alias Weight = Unqual!(DeepElementType!(typeof(masses)));
    static assert(is(Unqual!(DeepElementType!Data) == Element),
        "Weighted quantile input conversion changes its element type. " ~
        "Convert the input to a Mir slice explicitly before calling.");
    static assert(isNumeric!Element && isNumeric!Weight && isNumeric!F,
        "Weighted quantiles require numeric values, weights, and output");
    assert(values.length == masses.length,
        "weightedQuantile: values and weights must have equal lengths");
    size_t count;
    static if (weightedQuantileUsesFrequencies!algorithm) ulong frequencyTotal;
    real maximum = 0;
    foreach (i; 0 .. masses.length)
    {
        const w = cast(real) masses[i];
        assert(isFinite(w) && w >= 0,
            "weightedQuantile: weights must be finite and nonnegative");
        static if (weightedQuantileUsesFrequencies!algorithm)
        {
            import std.traits: isIntegral;
            import std.math: floor;
            static if (isIntegral!Weight)
                assert(masses[i] <= weightedQuantileMaxFrequency,
                    "weightedQuantile: frequency is too large");
            else
                assert(w <= cast(real) weightedQuantileMaxFrequency && floor(w) == w,
                    "weightedQuantile: frequencies must be representable whole numbers");
            const frequency = cast(ulong) masses[i];
            assert(frequency <= weightedQuantileMaxFrequency - frequencyTotal,
                "weightedQuantile: total frequency is too large");
            frequencyTotal += frequency;
        }
        if (w == 0)
            continue;
        assert(isFinite(cast(real) values[i]),
            "weightedQuantile: positive-weight values must be finite");
        ++count;
        if (w > maximum)
            maximum = w;
    }
    assert(count != 0, "weightedQuantile: at least one positive weight is required");
    alias Entry = WeightedQuantileEntry!(Element, algorithm);
    auto scratch = allocateQuantiles!(Entry, allocation)(context, count);
    static if (allocation == QuantileAllocation.custom)
        scope(exit) releaseQuantiles(context, scratch);
    scope auto workspace = scratch.lightScope;
    size_t j;
    foreach (i; 0 .. masses.length)
        if (masses[i] > 0)
            workspace[j++] = Entry(values[i], cast(typeof(Entry.init.cumulative)) masses[i]);

    // Keep observations in their original type: large integers must not collapse
    // to equal floating-point values. Order tied masses consistently as well.
    sort!((a, b) => a.value < b.value ||
        (a.value == b.value && a.cumulative < b.cumulative))(workspace);
    static if (weightedQuantileUsesEffectiveSize!algorithm)
    {
        Summator!(real, Summation.kahan) suffix, squares;
        foreach_reverse (i; 0 .. workspace.length)
        {
            const mass = workspace[i].cumulative / maximum;
            suffix.put(mass);
            squares.put(mass * mass);
            workspace[i].upper = suffix.sum;
        }
    }
    static if (weightedQuantileUsesFrequencies!algorithm)
    {
        ulong cumulativeCount;
        foreach (i; 0 .. workspace.length)
        {
            cumulativeCount += workspace[i].cumulative;
            workspace[i].cumulative = cumulativeCount;
        }
    }
    else
    {
        Summator!(real, Summation.kahan) cumulative;
        foreach (i; 0 .. workspace.length)
        {
            // Scaling before accumulation avoids overflow in the raw weight sum.
            cumulative.put(workspace[i].cumulative / maximum);
            workspace[i].cumulative = cumulative.sum;
        }
    }
    real effectiveSize = 1;
    static if (weightedQuantileUsesEffectiveSize!algorithm)
    {
        effectiveSize = cumulative.sum * cumulative.sum / squares.sum;
        if (effectiveSize < 1) effectiveSize = 1;
        if (effectiveSize > count) effectiveSize = count;
        // Separate accumulation directions may round differently at the edges.
        foreach (i; 0 .. workspace.length)
            if (workspace[i].upper > cumulative.sum)
                workspace[i].upper = cumulative.sum;
    }

    static if (isFloatingPoint!(Unqual!P))
        return weightedQuantileAt!(F, algorithm)(workspace, probabilities, effectiveSize);
    else
    {
        scope auto levels = quantileSlice(probabilities);
        static assert(typeof(levels).N == 1, "Probabilities must be one-dimensional");
        auto result = allocateQuantiles!(Unqual!F, allocation)(context, levels.length);
        static if (allocation == QuantileAllocation.custom)
            scope(failure) releaseQuantiles(context, result);
        foreach (i; 0 .. levels.length)
            result[i] = weightedQuantileAt!(F, algorithm)(workspace, levels[i], effectiveSize);
        return result;
    }
}

private auto dispatchWeightedQuantile(F, WeightedQuantileAlgo algorithm,
    QuantileAllocation allocation, Context, Weights, Data, P...)(
    ref Context context, scope auto ref Weights weights, scope auto ref Data data,
    scope auto ref P probabilities)
{
    import std.traits: CommonType, Unqual;
    static if (P.length == 0)
    {
        const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
        return allocatedWeightedQuantile!(F, algorithm, allocation)(context, weights, data, levels);
    }
    else static if (P.length == 1)
        return allocatedWeightedQuantile!(F, algorithm, allocation)(context, weights, data, probabilities[0]);
    else
    {
        alias G = CommonType!P;
        static assert(isFloatingPoint!(Unqual!G), "Probabilities must be floating-point values");
        G[P.length] levels = [probabilities];
        return allocatedWeightedQuantile!(F, algorithm, allocation)(context, weights, data, levels);
    }
}

/++
Compute weighted quantile estimates with GC-allocated scratch and results.

Pass observations first, then weights, followed by probabilities, as in
`weightedQuantile(data, weights, probabilities)`. This matches the data-first
argument order of weighted means and sums.

Use weights to change the probability mass assigned to observations, for example
when reweighting simulated outcomes. $(LINK2 #.WeightedQuantileAlgo.inverseCDF, $(D WeightedQuantileAlgo.inverseCDF))
returns observed values without interpolation; it is the default.
With equal weights, this matches ordinary type 1, not the interpolating type 7
default of $(LREF quantile). Select $(LINK2 #.WeightedQuantileAlgo.type7, $(D WeightedQuantileAlgo.type7))
explicitly when that equal-weight convention is wanted.

Weights and observations must be one-dimensional arrays or Mir slices of equal
length. Weights must be finite and nonnegative, with at least one positive weight.
Zero-weight observations are ignored, including their values. Positive-weight
values must be finite. Probabilities must be finite and in [0, 1].
These requirements are checked with assertions.

Inputs are never modified. A scalar probability returns a scalar. A probability
array/slice or multiple probabilities returns an owning slice. Omitting
probabilities requests $(TT [0, 0.25, 0.5, 0.75, 1]). For inverseCDF the inferred
output type is the unqualified observation type, including integral types.
All other algorithms promote integral observations to double and preserve
floating observation types. An explicit F selects the result type; algorithms
other than inverseCDF require a floating-point F.

Except for frequencyType7 and frequencyType8, the mathematical result is
invariant to positive rescaling of all weights. Frequency algorithms instead
use exact cumulative integer counts and the total frequency as sample size.
For probability-weight algorithms, cumulative masses use compensated summation
in $(TT real), after dividing by the
largest weight to avoid raw-sum overflow. Boundary comparisons use the computed
masses without a tolerance. Rounding and underflow can affect tiny masses or
probabilities close to a boundary; arbitrary rescaling is not promised to give
bitwise-identical results. Endpoints select the extrema of positive-weight data.
Signed zeros compare equal; the sign of a selected zero is unspecified.

A paired workspace is sorted once. For n observations and k probabilities,
inverseCDF, averagedInverseCDF, frequencyType7, and frequencyType8 take
$(TT O(n log n + k log n)) time. Harrell-Davis takes
$(TT O(n log n + k*n)) time, including beta CDF evaluations, and reuses effective
sample size and cumulative masses across probabilities. Trimmed Harrell-Davis
adds a bounded interval search per probability and skips excluded intervals.
Weighted type 7 and type 8 also take $(TT O(n log n + k*n)) time, using
arithmetic interval overlaps without beta evaluations or a root search.
All algorithms use $(TT O(n + k)) storage. Scratch storage is needed even for a scalar result.

Harrell-Davis evaluates coefficients using lower or upper tails to reduce
cancellation. Computed negative coefficients are clipped to zero and the
coefficient sum is normalized to correct roundoff. Values are scaled during
accumulation to avoid overflow; roundoff outside the input range is clamped.
Arithmetic uses real, whose precision depends on the compiler and platform.
These safeguards do not guarantee uniform accuracy in extreme beta tails.

Params:
    F = explicit result type; inferred from observations when omitted
    algorithm = weighted quantile definition, default inverseCDF
See_also:
    $(LREF WeightedQuantileAlgo), $(LREF rcWeightedQuantile), $(LREF makeWeightedQuantile)
+/
template weightedQuantile(F, WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
    if (algorithm == WeightedQuantileAlgo.inverseCDF || isFloatingPoint!F)
{
    /++
    Params:
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto weightedQuantile(Data, Weights, P...)(
        scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        QuantileContext context;
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.gc)(
            context, weights, data, probabilities);
    }
}

/// ditto
template weightedQuantile(WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
{
    /++
    Params:
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto weightedQuantile(Data, Weights, P...)(
        scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        alias F = WeightedQuantileResult!(Data, algorithm);
        QuantileContext context;
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.gc)(
            context, weights, data, probabilities);
    }
}

// Same-typed inputs must be interpreted as observations first in every API.
// Mallocator deallocation makes the custom-allocation calls @system.
version(mir_stat_test)
@system pure nothrow
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;

    const double[3] data = [2, 10, 20], weights = [1, 1, 8];
    // The cumulative masses are 1, 2, 10: the median observation is 20.
    // Reversing these arrays would instead select 8.
    assert(weightedQuantile(data, weights, 0.5) == 20);
    assert(weightedQuantile!double(data, weights, 0.5) == 20);
    assert(rcWeightedQuantile(data, weights, 0.5) == 20);
    assert(rcWeightedQuantile!double(data, weights, 0.5) == 20);
    assert(makeWeightedQuantile(Mallocator.instance, data, weights, 0.5) == 20);
    assert(makeWeightedQuantile!double(Mallocator.instance, data, weights, 0.5) == 20);
}

/// Select an observed value using cumulative weights, without interpolation.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    const int[3] values = [0, 10, 20];
    const int[3] weights = [1, 2, 1];
    // Sorted cumulative weights are 1, 3, 4. Half the total is 2,
    // so the median is the first observation whose cumulative weight reaches 2.
    auto median = weightedQuantile(values, weights, 0.5);
    static assert(is(typeof(median) == int));
    assert(median == 10);
    // At an exact boundary, select the observation on that boundary.
    assert(weightedQuantile(values, weights, 0.25) == 0);
    assert(weightedQuantile(values, weights) == [0, 0, 10, 10, 20]);
}

/// Zero weights exclude observations, including from the minimum and maximum.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    const double[5] values = [double.nan, -100, 10, 20, 100];
    const int[5] weights = [0, 0, 1, 1, 0];
    // Only 10 and 20 participate. A zero-weight NaN is ignored too;
    // a NaN with positive weight would violate the input requirements.
    assert(weightedQuantile(values, weights, 0.0, 0.5, 1.0) == [10.0, 10, 20]);
}

/// Increasing an outcome's probability mass moves the median toward it.
/// Harrell-Davis optionally smooths across outcomes instead of selecting one.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    const int[4] values = [0, 10, 20, 30];
    const double[4] equal = [1, 1, 1, 1];
    const double[4] reweighted = [1, 1, 1, 97];
    assert(weightedQuantile(values, equal, 0.5) == 10);
    assert(weightedQuantile(values, reweighted, 0.5) == 30);
    assert(weightedQuantile(values, reweighted) == [0, 30, 30, 30, 30]);
    import mir.math.common: approxEqual;
    auto smooth = weightedQuantile!(WeightedQuantileAlgo.harrellDavis)(values, reweighted, 0.5);
    static assert(is(typeof(smooth) == double));
    assert(smooth.approxEqual(29.450762222351227));
}

/// Trimmed smoothing can exclude extreme outcomes from a central estimate.
/// The same outlier can still contribute to quantiles near the upper tail.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    const double[4] values = [1, 2, 3, 10000];
    const double[4] weights = [1, 4, 4, 1];
    auto median = weightedQuantile!(WeightedQuantileAlgo.trimmedHarrellDavis)(
        values[].sliced, weights[].sliced, 0.5);
    assert(median.approxEqual(2.5));
    assert(weightedQuantile(values, weights, 0.5) == 2);
    auto smooth = weightedQuantile!(WeightedQuantileAlgo.harrellDavis)(values, weights, 0.5);
    assert(smooth.approxEqual(292.59361886338557));
}

/// Types 7 and 8 recover the corresponding ordinary quantiles with equal weights.
/// Reweighting simulated outcomes changes the estimated quartiles.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    const int[4] values = [0, 10, 20, 30];
    const int[4] equal = [1, 1, 1, 1], weights = [1, 1, 1, 7];
    auto q7 = weightedQuantile!(WeightedQuantileAlgo.type7)(values, equal, 0.25);
    auto q8 = weightedQuantile!(WeightedQuantileAlgo.type8)(values, equal, 0.25);
    assert(q7.approxEqual(7.5));
    assert(q8.approxEqual(25.0 / 6));
    auto reweighted = weightedQuantile!(WeightedQuantileAlgo.type7)(
        values[].sliced, weights[].sliced, [0.25, 0.5, 0.75]);
    assert(reweighted[0].approxEqual(25.0));
    assert(reweighted[1].approxEqual(375.0 / 13));
    assert(reweighted[2].approxEqual(30.0));
}

/// Averaging gives the midpoint when half the weight lies on either side.
/// A cumulative boundary inside tied observations still returns their value.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    const int[4] values = [0, 0, 10, 20], weights = [1, 1, 1, 1];
    auto q = weightedQuantile!(WeightedQuantileAlgo.averagedInverseCDF)(
        values, weights, [0.25, 0.5, 0.75]);
    assert(q == [0.0, 5.0, 15.0]);
    assert(weightedQuantile(values, weights, 0.5) == 0);
}

/// Frequency counts reproduce repeated observations without allocating them.
/// Unlike probability weights, scaling frequencies can change a quantile.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    const int[2] values = [0, 10], counts = [2, 2], once = [1, 1];
    alias frequency = weightedQuantile!(WeightedQuantileAlgo.frequencyType7);
    assert(frequency(values, counts, 0.25) == 0);
    assert(frequency(values, once, 0.25) == 2.5);
    assert(weightedQuantile!(WeightedQuantileAlgo.type7)(values, counts, 0.25) == 2.5);
}

/++
Compute weighted quantiles with reference-counted scratch and result storage.
Uses the definitions, input requirements, and numerical rules of $(LREF weightedQuantile).
Inputs remain unchanged. A scalar probability returns a scalar; otherwise the
owning RC result survives the inputs. Omitted probabilities request quartiles.
RC allocation supports $(TT @nogc) use, including scratch storage.
Pass observations before weights: `rcWeightedQuantile(data, weights, probabilities)`.

Params:
    F = explicit result type; inferred from observations when omitted
    algorithm = weighted quantile definition, default inverseCDF
+/
template rcWeightedQuantile(F, WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
    if (algorithm == WeightedQuantileAlgo.inverseCDF || isFloatingPoint!F)
{
    /++
    Params:
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto rcWeightedQuantile(Data, Weights, P...)(
        scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        QuantileContext context;
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.rc)(
            context, weights, data, probabilities);
    }
}

/// ditto
template rcWeightedQuantile(WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
{
    /++
    Params:
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto rcWeightedQuantile(Data, Weights, P...)(
        scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        alias F = WeightedQuantileResult!(Data, algorithm);
        QuantileContext context;
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.rc)(
            context, weights, data, probabilities);
    }
}

/// Mir slices can be used to obtain owning results without GC allocation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    const int[4] values = [30, 10, 20, 0];
    const double[4] weights = [5, 1, 1, 1];
    auto result = rcWeightedQuantile(values[].sliced, weights[].sliced, 0.0, 0.25, 1.0);
    assert(result == [0, 10, 30]);
    assert(values[] == [30, 10, 20, 0]);
}

/++
Compute weighted quantiles using a caller-selected allocator.
Uses the definitions, input requirements, and numerical rules of $(LREF weightedQuantile).
Scratch is released before returning, including on failure. A scalar result
requires no disposal. For a result slice, the caller must dispose its field with
the same allocator after all uses. Attributes depend on the allocator.
Inputs remain unchanged; omitted probabilities request quartiles.
The allocator comes first, then observations and weights:
`makeWeightedQuantile(allocator, data, weights, probabilities)`.

Params:
    F = explicit result type; inferred from observations when omitted
    algorithm = weighted quantile definition, default inverseCDF
+/
template makeWeightedQuantile(F, WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
    if (algorithm == WeightedQuantileAlgo.inverseCDF || isFloatingPoint!F)
{
    /++
    Params:
        allocator = allocator for scratch and result storage
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto makeWeightedQuantile(Allocator, Data, Weights, P...)(
        ref Allocator allocator, scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.custom)(
            allocator, weights, data, probabilities);
    }
}

/// ditto
template makeWeightedQuantile(WeightedQuantileAlgo algorithm = WeightedQuantileAlgo.inverseCDF)
{
    /++
    Params:
        allocator = allocator for scratch and result storage
        data = observations corresponding to the weights
        weights = relative probability masses, or nonnegative whole-number occurrence counts
            for $(LINK2 #.WeightedQuantileAlgo.frequencyType7, $(D WeightedQuantileAlgo.frequencyType7)) and $(LINK2 #.WeightedQuantileAlgo.frequencyType8, $(D WeightedQuantileAlgo.frequencyType8))
        probabilities = scalar, array/slice, or variadic probabilities; omitted for quartiles
    +/
    auto makeWeightedQuantile(Allocator, Data, Weights, P...)(
        ref Allocator allocator, scope auto ref Data data, scope auto ref Weights weights,
        scope auto ref P probabilities)
    {
        alias F = WeightedQuantileResult!(Data, algorithm);
        return dispatchWeightedQuantile!(F, algorithm, QuantileAllocation.custom)(
            allocator, weights, data, probabilities);
    }
}

/// The caller disposes custom-allocated results; scratch is released automatically.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    const int[3] values = [0, 10, 20];
    const int[3] weights = [1, 2, 1];
    auto result = makeWeightedQuantile(Mallocator.instance, values, weights);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [0, 0, 10, 10, 20]);
    assert(makeWeightedQuantile!double(Mallocator.instance, values, weights, 0.5) == 10.0);
}

// A decoding container must expose its converted element type explicitly.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    import std.experimental.allocator.mallocator: Mallocator;

    static struct PackedMeasurements(bool decode)
    {
        alias DeepElement = short;
        short[] raw;

        @property auto asSlice() return scope @safe pure nothrow @nogc
        {
            static if (decode)
                return raw.sliced.map!(x => x * 0.5);
            else
                return raw.sliced;
        }
    }

    short[3] raw = [1, 3, 5];
    const int[3] weights = [1, 1, 1];
    auto packed = PackedMeasurements!true(raw[]);
    static assert(!__traits(compiles, weightedQuantile(packed, weights, 0.5)));
    static assert(!__traits(compiles, rcWeightedQuantile(packed, weights, 0.5)));
    static assert(!__traits(compiles,
        makeWeightedQuantile(Mallocator.instance, packed, weights, 0.5)));

    // A conversion that preserves the element type remains valid.
    auto unchanged = PackedMeasurements!false(raw[]);
    assert(rcWeightedQuantile(unchanged, weights, 0.5) == 3);

    auto decoded = packed.asSlice;
    auto gcMedian = weightedQuantile(decoded, weights, 0.5);
    auto rcMedian = rcWeightedQuantile(decoded, weights, 0.5);
    static assert(is(typeof(gcMedian) == double) && is(typeof(rcMedian) == double));
    assert(gcMedian == 1.5 && rcMedian == 1.5);
    // Selecting a different output type remains supported.
    auto explicitMedian = rcWeightedQuantile!float(decoded, weights, 0.5);
    static assert(is(typeof(explicitMedian) == float));
    assert(explicitMedian == 1.5f);
}

// Boundaries, ties, ignored values, and original integral precision.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    const double[5] values = [double.nan, 30, 10, 10, 0];
    const double[5] weights = [0, 1, 1, 2, 1];
    const double[5] scaled = [0, 8, 8, 16, 8];
    const double[5] levels = [1, 0.5, 0, 0.25, 0.75];
    auto result = rcWeightedQuantile(values, weights, levels);
    assert(result == [30.0, 10, 0, 10, 10]);
    assert(result == rcWeightedQuantile(values, scaled, levels));
    const double[3] mergedValues = [0, 10, 30];
    const double[3] mergedWeights = [1, 3, 1];
    assert(result == rcWeightedQuantile(mergedValues, mergedWeights, levels));

    const int[4] data = [30, 10, 20, 0];
    const int[4] equal = [1, 1, 1, 1];
    assert(rcWeightedQuantile(data, equal) == rcquantile!(QuantileAlgo.type1)(data));
    assert(rcWeightedQuantile(data, equal, 0.5) == 10);
    assert(rcWeightedQuantile(data, equal, 0.5 + double.epsilon) == 20);
    const ulong[2] large = [ulong.max - 1, ulong.max];
    const int[2] pair = [1, 1];
    assert(rcWeightedQuantile(large, pair, 0.5) == ulong.max - 1);
    assert(rcWeightedQuantile(large, pair, 1.0) == ulong.max);

    const real[2] huge = [real.max, real.max];
    const real[2] tiny = [real.min_normal, real.min_normal];
    assert(rcWeightedQuantile(large, huge, 0.5) == ulong.max - 1);
    assert(rcWeightedQuantile(large, tiny, 0.5) == ulong.max - 1);
    const int[2] single = [0, 1];
    assert(rcWeightedQuantile(large, single) ==
        [ulong.max, ulong.max, ulong.max, ulong.max, ulong.max]);
}

// Owned results outlive local inputs, and explicit result types are respected.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    static auto fromLocal()
    {
        const int[3] data = [20, 0, 10];
        const int[3] weights = [1, 1, 2];
        const float[2] levels = [0, 1];
        return weightedQuantile!float(data, weights, levels);
    }
    assert(fromLocal() == [0.0f, 20.0f]);
    const int[2] data = [0, 10];
    const int[2] weights = [1, 1];
    assert(weightedQuantile!double(data, weights, 0.0, 1.0) == [0.0, 10]);
    assert(rcWeightedQuantile!float(data, weights, 0.5f) == 0.0f);
    const double[] empty = [];
    assert(rcWeightedQuantile(data, weights, empty).length == 0);
}

// Strided Mir slices and RC results with independent lifetimes.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    static auto fromLocal()
    {
        const int[6] data = [20, -1, 0, -1, 10, -1];
        const int[6] weights = [1, -1, 1, -1, 2, -1];
        const double[4] levels = [0, -1, 1, -1];
        return rcWeightedQuantile(data[].sliced(3, 2).transposed[0],
            weights[].sliced(3, 2).transposed[0], levels[].sliced(2, 2).transposed[0]);
    }
    auto result = fromLocal();
    assert(result == [0, 20]);
}

// A safe allocator preserves safe use of the custom factory.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.experimental.allocator.gc_allocator: GCAllocator;
    struct SafeAllocator
    {
        enum alignment = GCAllocator.alignment;
        void[] allocate(size_t bytes) @safe pure nothrow
        {
            return GCAllocator.instance.allocate(bytes);
        }
        bool deallocate(void[] memory) @safe pure nothrow @nogc
        {
            // Leave reclamation to the GC so aliases cannot become dangling.
            return true;
        }
    }
    SafeAllocator allocator;
    const int[3] values = [0, 10, 20];
    const int[3] weights = [1, 2, 1];
    auto result = makeWeightedQuantile(allocator, values, weights);
    assert(result == [0, 0, 10, 10, 20]);
}

// Reject invalid inputs; release custom scratch and result storage on failures.
version(mir_stat_test)
@system pure
unittest
{
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import std.experimental.allocator: dispose;
    const double[2] data = [0, 10];
    const double[2] weights = [1, 1];
    foreach (bad; [-1.0, double.nan, double.infinity])
    {
        double[2] badWeights = [1, bad];
        assertThrown!AssertError(rcWeightedQuantile(data, badWeights, 0.5));
    }
    const double[2] zero = [0, 0];
    assertThrown!AssertError(rcWeightedQuantile(data, zero, 0.5));
    assertThrown!AssertError(rcWeightedQuantile(data, weights[0 .. 1], 0.5));
    assertThrown!AssertError(rcWeightedQuantile(data[0 .. 0], weights[0 .. 0], 0.5));
    foreach (bad; [double.nan, double.infinity])
    {
        double[2] badData = [0, bad];
        assertThrown!AssertError(rcWeightedQuantile(badData, weights, 0.5));
    }
    foreach (bad; [-0.1, 1.1, double.nan, double.infinity])
        assertThrown!AssertError(rcWeightedQuantile(data, weights, bad));

    QuantileTestAllocator!() allocator;
    auto result = makeWeightedQuantile(allocator, data, weights, 0.0, 1.0);
    assert(result == [0.0, 10]);
    assert(allocator.allocations == 2 && allocator.releases == 1);
    allocator.dispose(result.field);
    assert(allocator.releases == 2);

    QuantileTestAllocator!true failing;
    failing.failAt = 2;
    assertThrown!Exception(makeWeightedQuantile(failing, data, weights));
    assert(failing.allocations == 2 && failing.releases == 1);
    QuantileTestAllocator!() invalid;
    assertThrown!AssertError(makeWeightedQuantile(invalid, data, weights, 0.0, 2.0));
    assert(invalid.allocations == 2 && invalid.releases == 2);
}

// High-precision reference values, including tails and effective-size semantics.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias hd = rcWeightedQuantile!(WeightedQuantileAlgo.harrellDavis);
    const double[3] values = [0, 10, 100];
    const double[3] zero = [1, 0, 1];
    const double[3] tiny = [1, 0.000001, 1];
    assert(hd(values, zero, 0.5).approxEqual(50.0));
    assert(hd(values, tiny, 0.5).approxEqual(49.999974535212, 1e-12, 1e-12));
    assert(hd(values, tiny, 0.4).approxEqual(34.43961434178727, 1e-12, 1e-12));
    const double[3] splitValues = [0, 10, 30];
    const double[3] splitWeights = [1, 2, 1];
    const double[4] duplicateValues = [0, 10, 10, 30];
    const double[4] equal = [1, 1, 1, 1];
    assert(hd(splitValues, splitWeights, 0.5).approxEqual(11.680673694094532));
    assert(hd(duplicateValues, equal, 0.5).approxEqual(11.265849975501613));

    const double[3] tailValues = [0, 1, 1e12];
    const double[3] tailWeights = [1, 1, 1e-12];
    assert(hd(tailValues, tailWeights, 0.75).approxEqual(1142.889745000284, 1e-12, 1e-12));
    assert(hd(tailValues, tailWeights, 0.5).approxEqual(0.5000006002111957, 1e-12, 1e-12));
    // The difficult beta tail is less accurate on platforms with double-sized real.
    assert(hd(tailValues, tailWeights, 1 - 1e-8).approxEqual(999999195275.2699, 1e-10, 0));
    const double[4] outlier = [1, 2, 3, 10000];
    const double[4] outlierWeights = [1, 4, 4, 1];
    assert(hd(outlier, outlierWeights, 0.5).approxEqual(292.59361886338557));
    double[64] regularValues, regularWeights;
    foreach (i; 0 .. regularValues.length)
    {
        regularValues[i] = cast(double) i * i;
        regularWeights[i] = 1;
    }
    assert(hd(regularValues, regularWeights, 0.5).approxEqual(1007.8484848484848, 1e-12, 1e-12));
    assert(hd(tailValues, tailWeights, 1e-8).approxEqual(2.044415538269786e-9, 1e-12, 1e-22));
}

// Equal-weight HD with two values has a beta(1.5,1.5) median of their midpoint.
// Scaling observations protects finite extreme values from intermediate overflow.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static foreach (algorithm; [WeightedQuantileAlgo.harrellDavis,
        WeightedQuantileAlgo.trimmedHarrellDavis, WeightedQuantileAlgo.type7,
        WeightedQuantileAlgo.type8, WeightedQuantileAlgo.averagedInverseCDF])
    {{
        import mir.math.common: approxEqual;
        import std.math: isFinite;
        alias hd = rcWeightedQuantile!algorithm;
        const real[2] weights = [real.max, real.max];
        const real[2] extreme = [-real.max, real.max];
        const real[2] positive = [real.max / 2, real.max];
        auto mid = hd(extreme, weights, 0.5);
        assert(isFinite(mid) && (mid / real.max).approxEqual(0.0L, 0.0L, 1e-14L));
        auto positiveMid = hd(positive, weights, 0.5);
        assert(isFinite(positiveMid) && (positiveMid / real.max).approxEqual(0.75L));
        const real[2] constant = [real.max, real.max];
        assert(hd(constant, weights, 0.25) == real.max);
        const double[3] singleValues = [double.nan, 7, double.infinity];
        const int[3] singleWeights = [0, 1, 0];
        assert(hd(singleValues, singleWeights) == [7.0, 7, 7, 7, 7]);
        const int[2] integerValues = [0, 10];
        const int[2] integerWeights = [1, 1];
        static assert(is(typeof(hd(integerValues, integerWeights, 0.5)) == double));
        const float[2] floatValues = [0, 10];
        static assert(is(typeof(hd(floatValues, integerWeights, 0.5)) == float));
        static assert(!__traits(compiles,
            rcWeightedQuantile!(int, algorithm)(integerValues, integerWeights, 0.5)));
        assert(hd(integerValues, integerWeights, 0.0, 1.0) == [0.0, 10]);
    }}
}

// Invariants and monotonicity on interior probabilities, plus rescaled weights.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static foreach (algorithm; [WeightedQuantileAlgo.harrellDavis,
        WeightedQuantileAlgo.trimmedHarrellDavis, WeightedQuantileAlgo.type7,
        WeightedQuantileAlgo.type8])
    {{
        import mir.math.common: approxEqual;
        alias hd = rcWeightedQuantile!algorithm;
        const double[4] weights = [1, 1, 1, 97];
        const double[4] scaled = [1e200, 1e200, 1e200, 97e200];
        const int[4] values = [0, 10, 20, 30];
        double[21] levels;
        foreach (i; 0 .. levels.length) levels[i] = i / 20.0;
        auto result = hd(values, weights, levels);
        auto scaledResult = hd(values, scaled, levels);
        foreach (i; 0 .. levels.length)
        {
            assert(result[i] >= 0 && result[i] <= 30);
            assert(result[i].approxEqual(scaledResult[i], 1e-12, 1e-12));
            if (i) assert(result[i] >= result[i - 1]);
        }
    }}
}

// All allocation factories use the same smoothing kernel and release scratch.
version(mir_stat_test)
@system pure nothrow
unittest
{
    static foreach (algorithm; [WeightedQuantileAlgo.harrellDavis,
        WeightedQuantileAlgo.trimmedHarrellDavis, WeightedQuantileAlgo.type7,
        WeightedQuantileAlgo.type8, WeightedQuantileAlgo.averagedInverseCDF,
        WeightedQuantileAlgo.frequencyType7, WeightedQuantileAlgo.frequencyType8])
    {{
        import mir.math.common: approxEqual;
        import std.experimental.allocator: dispose;
        import mir.ndslice.slice: sliced;
        import mir.ndslice.dynamic: transposed;
        const int[6] data = [30, -1, 0, -1, 10, -1];
        const int[6] weights = [1, -1, 1, -1, 2, -1];
        const double[3] levels = [0, 0.5, 1];
        auto values = data[].sliced(3, 2).transposed[0];
        auto masses = weights[].sliced(3, 2).transposed[0];
        auto gc = weightedQuantile!algorithm(values, masses, levels);
        auto rc = rcWeightedQuantile!algorithm(values, masses, levels);
        QuantileTestAllocator!() allocator;
        auto custom = makeWeightedQuantile!(double, algorithm)(allocator, values, masses, levels);
        assert(allocator.allocations == 2 && allocator.releases == 1);
        foreach (i; 0 .. levels.length)
            assert(gc[i].approxEqual(rc[i]) && gc[i].approxEqual(custom[i]));
        allocator.dispose(custom.field);
        assert(allocator.releases == 2);
        auto scalar = makeWeightedQuantile!algorithm(allocator, values, masses, 0.5);
        assert(scalar.approxEqual(algorithm == WeightedQuantileAlgo.harrellDavis ?
            11.680673694094532 : algorithm == WeightedQuantileAlgo.trimmedHarrellDavis ?
            10.753130990579251 : 10.0));
        assert(allocator.allocations == 3 && allocator.releases == 3);
    }}
}

// Trimmed-HD reference values from 80-digit beta integration, including ties,
// vanishing weights, and a tail that trimming does not remove.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias thd = rcWeightedQuantile!(WeightedQuantileAlgo.trimmedHarrellDavis);
    const double[3] values = [0, 10, 100];
    const double[3] zero = [1, 0, 1], tiny = [1, 0.000001, 1];
    assert(thd(values, zero, 0.5).approxEqual(50.0));
    assert(thd(values, tiny, 0.5).approxEqual(49.999968881235418, 1e-12, 1e-12));
    assert(thd(values, tiny, 0.4).approxEqual(24.518539568626075, 1e-12, 1e-12));
    const int[3] tiedWeights = [1, 2, 1], tiedValues = [0, 10, 30];
    const int[4] splitWeights = [1, 1, 1, 1], splitValues = [0, 10, 10, 30];
    assert(thd(tiedValues, tiedWeights, 0.5).approxEqual(10.753130990579251));
    assert(thd(splitValues, splitWeights, 0.5).approxEqual(10.0));
    const double[3] tailValues = [0, 1, 1e12], tailWeights = [1, 1, 1e-12];
    assert(thd(tailValues, tailWeights, 0.5).approxEqual(0.500000000000389, 1e-12, 1e-12));
    assert(thd(tailValues, tailWeights, 0.75).approxEqual(1193.996801217712, 1e-12, 1e-12));
    // Prefixes round to one, but the independently summed suffix retains mass.
    const double[3] roundedValues = [0, 1, 1e30], roundedWeights = [1, 1, 1e-30];
    assert(thd(roundedValues, roundedWeights, 0.99).approxEqual(1.2890527282600647e29, 1e-12, 0));
}

// The interval is fixed-width, with boundary modes and symmetric/asymmetric
// interior modes. These checks exercise the interval solver independently.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: log;
    assert(weightedTrimmedHDLower(1, 1, 1) == 0);
    assert(weightedTrimmedHDLower(0.5L, 3, 0.5L) == 0);
    assert(weightedTrimmedHDLower(3, 0.5L, 0.5L) == 0.5L);
    assert(weightedTrimmedHDLower(3, 3, 0.5L) == 0.25L);
    const left = weightedTrimmedHDLower(2, 5, 0.25L);
    const right = left + 0.25L;
    assert(left > 0 && right < 1);
    assert((log(left) + 4 * log(1-left)).approxEqual(
        log(right) + 4 * log(1-right), 1e-13L, 1e-13L));
    assert(weightedTrimmedHDLower(5, 2, 0.25L).approxEqual(1-right, 1e-13L, 1e-13L));
}

// Equal-weight agreement across sample sizes and a probability grid, including
// clamping at the boundaries and exact order-statistic positions.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    static foreach (algorithm; [WeightedQuantileAlgo.type7, WeightedQuantileAlgo.type8])
    {{
        enum ordinary = algorithm == WeightedQuantileAlgo.type7 ? QuantileAlgo.type7 : QuantileAlgo.type8;
        double[12] values, weights;
        foreach (i; 0 .. values.length) { values[i] = cast(double) i*i - 7; weights[i] = 1; }
        foreach (n; 1 .. values.length + 1)
            foreach (k; 0 .. 41)
            {
                const p = k / 40.0;
                const actual = rcWeightedQuantile!algorithm(values[0 .. n], weights[0 .. n], p);
                const expected = n == 1 ? values[0] :
                    rcquantile!(double, ordinary)(values[0 .. n], p);
                assert(actual.approxEqual(expected, 1e-12, 1e-12));
            }
    }}
}

// Exact-rational reference cases: tiny weights, splitting tied rows, reflection,
// endpoint jumps, and a weighted interval spanning more than two observations.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    static foreach (algorithm; [WeightedQuantileAlgo.type7, WeightedQuantileAlgo.type8])
    {{
        alias q = rcWeightedQuantile!algorithm;
        const double[4] values = [0, 1, 1, 100];
        const double[4] zero = [1, 0, 0, 1], tiny = [1, 1e-6, 1e-6, 1];
        assert(q(values, zero, 0.5).approxEqual(50.0));
        assert(q(values, tiny, 0.5).approxEqual(49.999901999902, 1e-12, 1e-12));
        const int[3] concentrated = [1, 1, 8], points = [0, 10, 20];
        const int[3] reflected = [0, -10, -20];
        const expected = algorithm == WeightedQuantileAlgo.type7 ? 15.454555757575758 : 170.0 / 11;
        assert(q(points, concentrated, 1e-6).approxEqual(expected, 1e-12, 1e-12));
        assert(q(points, concentrated, 0.0) == 0);
        assert(q(points, concentrated, 1.0) == 20);
        assert(q(points, concentrated, 0.25).approxEqual(-q(reflected, concentrated, 0.75)));
        const int[3] unsplitValues = [0, 10, 30], unsplitWeights = [1, 2, 1];
        const int[4] splitValues = [0, 10, 10, 30], splitWeights = [1, 1, 1, 1];
        assert(q(unsplitValues, unsplitWeights, 0.01).approxEqual(
            algorithm == WeightedQuantileAlgo.type7 ? 3.5 : 10.0 / 3));
        assert(q(splitValues, splitWeights, 0.01).approxEqual(
            algorithm == WeightedQuantileAlgo.type7 ? 0.3 : 0.0, 1e-12, 1e-12));
        const int[4] multiValues = [0, 10, 20, 40], multiWeights = [3, 1, 1, 4];
        assert(q(multiValues, multiWeights, 0.5).approxEqual(70.0 / 3));
    }}
}

// Boundary behavior, ties, zeros, rescaling, and agreement with ordinary type 2.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import std.math: nextafter;
    alias q = rcWeightedQuantile!(WeightedQuantileAlgo.averagedInverseCDF);
    const int[4] values = [0, 0, 10, 20], weights = [1, 1, 1, 1];
    assert(q(values, weights, nextafter(0.5, 0.0)) == 0);
    assert(q(values, weights, 0.5) == 5);
    assert(q(values, weights, nextafter(0.5, 1.0)) == 10);
    const int[3] combinedValues = [0, 10, 20], combinedWeights = [2, 1, 1];
    const int[3] scaled = [8, 4, 4];
    const int[5] zeroWeights = [1, 0, 1, 1, 1];
    const double[5] zeroValues = [0, double.nan, 0, 10, 20];
    foreach (i; 0 .. 17)
    {
        const p = i / 16.0;
        const result = q(values, weights, p);
        assert(result == q(combinedValues, combinedWeights, p));
        assert(result == q(combinedValues, scaled, p));
        assert(result == q(zeroValues, zeroWeights, p));
        assert(result.approxEqual(rcquantile!(double, QuantileAlgo.type2)(values, p)));
    }
    const int[2] equal = [1, 1];
    const real[2] negative = [-real.max, -real.max / 2];
    assert((q(negative, equal, 0.5) / real.max).approxEqual(-0.75L));
    // Neighboring same-sign values at the bottom of the representable range.
    const real tiny = nextafter(0.0L, 1.0L);
    const real[2] subnormal = [tiny, 3 * tiny];
    assert(q(subnormal, equal, 0.5) == 2 * tiny);
    const real[2] mixedSigns = [-tiny, 2 * tiny];
    assert(q(mixedSigns, equal, 0.5L) == 0);
    assert(rcquantile(mixedSigns, 0.5L) == 0);
    static foreach (algorithm; [WeightedQuantileAlgo.frequencyType7,
        WeightedQuantileAlgo.frequencyType8])
        assert(rcWeightedQuantile!algorithm(mixedSigns, equal, 0.5L) == 0);
}

// Compare conceptual replication against genuinely expanded samples, including
// unsorted values, ties, zero frequencies, and floating-point whole counts.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    static foreach (algorithm; [WeightedQuantileAlgo.frequencyType7, WeightedQuantileAlgo.frequencyType8])
    {{
        alias q = rcWeightedQuantile!algorithm;
        enum ordinary = algorithm == WeightedQuantileAlgo.frequencyType7 ? QuantileAlgo.type7 : QuantileAlgo.type8;
        const int[4] values = [20, 0, 10, 10];
        foreach (a; 0 .. 4) foreach (b; 0 .. 4) foreach (c; 1 .. 4)
        {
            int[4] counts = [a, b, c, 1];
            double[4] floating = [a, b, c, 1];
            int[10] expanded;
            size_t n;
            foreach (i; 0 .. 4) foreach (_; 0 .. counts[i]) expanded[n++] = values[i];
            foreach (k; 0 .. 21)
            {
                const p = k / 20.0;
                const expected = rcquantile!(double, ordinary)(expanded[0 .. n], p);
                assert(q(values, counts, p).approxEqual(expected, 1e-12, 1e-12));
                assert(q(values, floating, p).approxEqual(expected, 1e-12, 1e-12));
            }
        }
        const int[2] ones = [1, 1];
        const real[2] extremes = [-real.max, real.max];
        assert(q(extremes, ones, 0.5) == 0);
        const real[2] positive = [real.max / 2, real.max];
        assert((q(positive, ones, 0.5) / real.max).approxEqual(0.75L));
        const int[2] single = [0, 1], data = [0, 10];
        assert(q(data, single) == [10.0, 10, 10, 10, 10]);
        // Large counts stay compact; total equals the supported maximum.
        const ulong[2] large = [weightedQuantileMaxFrequency - 1, 1];
        assert(q(data, large, 0.5) == 0);
        assert(q(data, large, 1.0) == 10);
        static assert(is(typeof(q(data, ones, 0.5)) == double));
        static assert(!__traits(compiles, rcWeightedQuantile!(int, algorithm)(data, ones, 0.5)));
    }}
}

// Reject fractional counts and totals outside the exact working-count range.
// Catching AssertError requires @system under DIP1000.
version(mir_stat_test)
@system pure
unittest
{
    import std.exception: assertThrown;
    import core.exception: AssertError;
    static foreach (algorithm; [WeightedQuantileAlgo.frequencyType7, WeightedQuantileAlgo.frequencyType8])
    {{
        const int[2] data = [0, 10];
        foreach (bad; [0.5, -1.0, double.nan, double.infinity, 18446744073709551616.0])
        {
            double[2] counts = [bad, 1];
            assertThrown!AssertError(rcWeightedQuantile!algorithm(data, counts, 0.5));
        }
        const ulong[2] overflow = [weightedQuantileMaxFrequency, 1];
        assertThrown!AssertError(rcWeightedQuantile!algorithm(data, overflow, 0.5));
        static if (real.mant_dig < 64)
        {
            const ulong[2] tooLarge = [weightedQuantileMaxFrequency + 1, 0];
            assertThrown!AssertError(rcWeightedQuantile!algorithm(data, tooLarge, 0.5));
        }
    }}
}

// Excluded extremes cannot erase tiny contributing values during scaling.
// Use real's full range so this exercises both extended and double-sized real.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    const int[3] weights = [1, 1, 1];
    const real tiny = real.min_normal;
    const real[3] values = [tiny, 2*tiny, real.max];
    const real[3] reflected = [-tiny, -2*tiny, -real.max];
    const real[3] zeros = [0, 0, real.max];
    static foreach (algorithm; [WeightedQuantileAlgo.type7, WeightedQuantileAlgo.type8])
    {{
        alias q = rcWeightedQuantile!algorithm;
        assert((q(values, weights, 0.5) / tiny).approxEqual(2.0L));
        assert((q(reflected, weights, 0.5) / tiny).approxEqual(-2.0L));
        assert(q(zeros, weights, 0.5) == 0);
        const real[5] upper = [-real.max, tiny, 2*tiny, 3*tiny, 4*tiny];
        const int[5] equal = [1, 1, 1, 1, 1];
        // Exercise the independently accumulated upper-mass path as well.
        const expected = algorithm == WeightedQuantileAlgo.type7 ? 3.0L : 10.0L/3;
        assert((q(upper, equal, 0.75) / tiny).approxEqual(expected));
    }}
    alias trimmed = rcWeightedQuantile!(WeightedQuantileAlgo.trimmedHarrellDavis);
    const int[5] equal = [1, 1, 1, 1, 1];
    const real[5] five = [tiny, 2*tiny, 3*tiny, 4*tiny, real.max];
    const real[5] negative = [-tiny, -2*tiny, -3*tiny, -4*tiny, -real.max];
    const real[5] retainedZero = [-real.max, 0, 0, 0, real.max];
    assert((trimmed(five, equal, 0.5) / tiny).approxEqual(3.0L));
    assert((trimmed(negative, equal, 0.5) / tiny).approxEqual(-3.0L));
    assert(trimmed(retainedZero, equal, 0.5) == 0);
}

private enum QuantileAllocation { gc, rc, custom }
private struct QuantileContext {}

private auto quantileSlice(T)(return scope auto ref T value)
{
    import mir.ndslice.slice: isSlice, toSlice;
    static if (isSlice!T)
        return value;
    else
        return value.toSlice;
}

private auto allocateQuantiles(T, QuantileAllocation allocation, Context)(
    ref Context context, size_t length)
{
    static if (allocation == QuantileAllocation.gc)
    {
        import mir.ndslice.slice: sliced;
        return (new T[length]).sliced;
    }
    else static if (allocation == QuantileAllocation.rc)
    {
        import mir.ndslice.allocation: rcslice;
        return rcslice!T([length], T.init);
    }
    else
    {
        import mir.ndslice.allocation: makeSlice;
        return makeSlice!T(context, length);
    }
}

private void releaseQuantiles(Context, Storage)(ref Context context, Storage storage)
{
    import std.experimental.allocator: dispose;
    context.dispose(storage.field);
}

// Keep scratch ownership alive until all partitioning and output writes finish.
private auto allocatedQuantile(F, QuantileAlgo algorithm, bool modify,
    QuantileAllocation allocation, Context, Data, Probability)(
    ref Context context, scope auto ref Data data, scope auto ref Probability probability)
{
    import mir.ndslice.topology: flattened;
    import mir.primitives: DeepElementType;
    import mir.ndslice.slice: IteratorOf;
    import std.traits: Unqual;

    alias Result = Unqual!(quantileType!(F, algorithm));
    scope auto source = quantileSlice(data);
    static if (modify)
        scope auto workspace = source.flattened;
    else
    {
        alias Element = Unqual!(DeepElementType!(typeof(source)));
        auto scratch = allocateQuantiles!(Element, allocation)(context, source.elementCount);
        static if (allocation == QuantileAllocation.custom)
            scope(exit) releaseQuantiles(context, scratch);
        scope auto flat = source.lightScope.flattened;
        foreach (i; 0 .. flat.length)
            scratch[i] = flat[i];
        scope auto workspace = scratch.lightScope;
    }

    static if (isFloatingPoint!(Unqual!Probability))
        return quantileImpl!(Result, algorithm, IteratorOf!(typeof(workspace)),
            Unqual!Probability)(workspace, probability);
    else
    {
        auto result = allocateQuantiles!(Result, allocation)(context, probability.length);
        static if (allocation == QuantileAllocation.custom)
            scope(failure) releaseQuantiles(context, result);
        quantileInto!algorithm(workspace, probability, result.lightScope);
        return result;
    }
}

private auto dispatchQuantile(F, QuantileAlgo algorithm, bool modify,
    QuantileAllocation allocation, Context, Data, P...)(
    ref Context context, scope auto ref Data data, scope auto ref P probabilities)
{
    import std.traits: CommonType, Unqual;
    import mir.ndslice.slice: sliced;
    static if (P.length == 0)
    {
        const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
        return allocatedQuantile!(F, algorithm, modify, allocation)(
            context, data, levels[].sliced);
    }
    else static if (P.length == 1)
    {
        static if (isFloatingPoint!(Unqual!(P[0])))
            return allocatedQuantile!(F, algorithm, modify, allocation)(
                context, data, probabilities[0]);
        else
            return allocatedQuantile!(F, algorithm, modify, allocation)(
                context, data, quantileSlice(probabilities[0]));
    }
    else
    {
        alias G = CommonType!P;
        static assert(isFloatingPoint!(Unqual!G), "Probabilities must be floating-point values");
        G[P.length] levels = [probabilities];
        return allocatedQuantile!(F, algorithm, modify, allocation)(
            context, data, levels[].sliced);
    }
}

/++
Compute quantiles into caller-provided output without allocating.

The observations in workspace may be rearranged by partitioning; they need not
end up sorted. Probabilities are read without modification. Output must have one
element per probability, and its element type determines the result type.
Workspace, probabilities, and output must not overlap.
Use $(LREF quantile), $(LREF rcquantile), or $(LREF makeQuantile) when a copy of
the observations or allocated output is desired.

Params:
    algorithm = quantile algorithm, default type7
    workspace = mutable observations, as a slice or array
    probabilities = one-dimensional floating-point probabilities
    output = one-dimensional mutable result storage
+/
template quantileInto(QuantileAlgo algorithm = QuantileAlgo.type7)
{
    void quantileInto(Data, P, Output)(
        scope auto ref Data workspace, scope auto ref P probabilities, scope auto ref Output output)
    {
        import mir.ndslice.topology: flattened;
        import mir.ndslice.slice: IteratorOf;
        import mir.primitives: DeepElementType;
        import std.traits: Unqual;
        scope auto values = quantileSlice(workspace).flattened;
        scope auto levels = quantileSlice(probabilities);
        scope auto destination = quantileSlice(output);
        static assert(typeof(levels).N == 1 && typeof(destination).N == 1,
            "Probabilities and output must be one-dimensional");
        alias F = Unqual!(DeepElementType!(typeof(destination)));
        alias G = Unqual!(DeepElementType!(typeof(levels)));
        static assert(isFloatingPoint!G, "Probabilities must be floating-point values");
        static assert(isFloatingPoint!F || algorithm == QuantileAlgo.type1 ||
            algorithm == QuantileAlgo.type3, "Interpolated quantiles require floating-point output");
        assert(destination.length == levels.length, "quantileInto: output length must match probabilities");
        foreach (i; 0 .. levels.length)
            destination[i] = quantileImpl!(F, algorithm, IteratorOf!(typeof(values)), G)(values, levels[i]);
    }
}

/// ditto
template quantileInto(string algorithm)
{
    mixin("alias quantileInto = .quantileInto!(QuantileAlgo." ~ algorithm ~ ");");
}

/// Compute into caller-provided output without allocating.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double[5] workspace = [3, 1, 4, 2, 0];
    const float[2] probabilities = [0.25f, 0.75f];
    double[2] output;
    quantileInto(workspace, probabilities, output);
    assert(output[] == [1.0, 3.0]);
    assert(probabilities[] == [0.25f, 0.75f]);
}

/++
Compute quantiles with reference-counted scratch and result storage.
A scalar probability returns a scalar; multiple probabilities return an owning
RC slice. The result remains valid after the inputs go out of scope.

By default observations are copied before partitioning. Setting allowModifySlice
permits rearranging the input and avoids that scratch allocation. When input
modification is enabled, observations must not overlap the probabilities.
Use $(LREF quantileInto) for caller-provided output.
Quantile algorithms and inferred result types follow $(LREF QuantileAlgo).
Omitting probabilities returns the minimum, quartiles, and maximum using
`[0, 0.25, 0.5, 0.75, 1]`.

Params:
    F = explicit output type; inferred from observations when omitted
    algorithm = quantile algorithm, default type7
    allowModifySlice = allow rearranging observations, default false
+/
template rcquantile(F, QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
    if (isFloatingPoint!F || algorithm == QuantileAlgo.type1 || algorithm == QuantileAlgo.type3)
{
    /++
    Params:
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto rcquantile(Data, P...)(scope auto ref Data data, scope auto ref P probabilities)
    {
        QuantileContext context;
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.rc)(
            context, data, probabilities);
    }
}

/// ditto
template rcquantile(QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
{
    /++
    Params:
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto rcquantile(Data, P...)(scope auto ref Data data, scope auto ref P probabilities)
    {
        alias F = quantileType!(typeof(quantileSlice(data)), algorithm);
        QuantileContext context;
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.rc)(
            context, data, probabilities);
    }
}

/// Omit probabilities to obtain the minimum, quartiles, and maximum.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    int[5] data = [4, 0, 3, 1, 2];
    auto result = rcquantile(data);
    assert(result == [0.0, 1, 2, 3, 4]);
    assert(data[] == [4, 0, 3, 1, 2]);
}

/// ditto
template rcquantile(F, string algorithm, bool allowModifySlice = false)
{
    mixin("alias rcquantile = .rcquantile!(F, QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

/// ditto
template rcquantile(string algorithm, bool allowModifySlice = false)
{
    mixin("alias rcquantile = .rcquantile!(QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

/// RC allocation supports quantile calculation without the GC.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double[5] observations = [4, 0, 3, 1, 2];
    auto result = rcquantile!float(observations[], 0.25, 0.75);
    assert(result == [1.0f, 3.0f]);
    assert(observations[] == [4.0, 0.0, 3.0, 1.0, 2.0]);
}

/++
Compute quantiles using a caller-selected allocator for scratch and results.
Scratch storage is released before returning, including on failure. For multiple
probabilities, the caller owns the returned result and must release its field
through the same allocator after all uses. Construction failure releases any
result allocation as well. Attributes depend on the allocator.

By default observations are copied before partitioning. Setting allowModifySlice
permits rearranging the input and avoids that scratch allocation. When input
modification is enabled, observations must not overlap the probabilities.
Use $(LREF quantileInto) for caller-provided output.
Quantile algorithms and inferred result types follow $(LREF QuantileAlgo).
Omitting probabilities returns the minimum, quartiles, and maximum using
`[0, 0.25, 0.5, 0.75, 1]`.

Params:
    F = explicit output type; inferred from observations when omitted
    algorithm = quantile algorithm, default type7
    allowModifySlice = allow rearranging observations, default false
+/
template makeQuantile(F, QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
    if (isFloatingPoint!F || algorithm == QuantileAlgo.type1 || algorithm == QuantileAlgo.type3)
{
    /++
    Params:
        allocator = allocator used for scratch and result storage
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto makeQuantile(Allocator, Data, P...)(ref Allocator allocator, scope auto ref Data data, scope auto ref P probabilities)
    {
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.custom)(
            allocator, data, probabilities);
    }
}

/// ditto
template makeQuantile(QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
{
    /++
    Params:
        allocator = allocator used for scratch and result storage
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto makeQuantile(Allocator, Data, P...)(ref Allocator allocator, scope auto ref Data data, scope auto ref P probabilities)
    {
        alias F = quantileType!(typeof(quantileSlice(data)), algorithm);
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.custom)(
            allocator, data, probabilities);
    }
}

/// Omit probabilities to obtain the minimum, quartiles, and maximum.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    int[5] data = [4, 0, 3, 1, 2];
    auto result = makeQuantile(Mallocator.instance, data);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [0.0, 1, 2, 3, 4]);
    assert(data[] == [4, 0, 3, 1, 2]);
}

/// ditto
template makeQuantile(F, string algorithm, bool allowModifySlice = false)
{
    mixin("alias makeQuantile = .makeQuantile!(F, QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

/// ditto
template makeQuantile(string algorithm, bool allowModifySlice = false)
{
    mixin("alias makeQuantile = .makeQuantile!(QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

/// Custom allocation releases scratch automatically; the caller releases results.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    double[5] observations = [4, 0, 3, 1, 2];
    auto result = makeQuantile!float(Mallocator.instance, observations[], 0.25, 0.75);
    scope(exit) Mallocator.instance.dispose(result.field);
    assert(result == [1.0f, 3.0f]);
    // A scalar result leaves no allocation for the caller to release.
    assert(makeQuantile(Mallocator.instance, observations[], 0.5) == 2.0);
}

/++
Compute quantiles with GC-allocated scratch and result storage.
A scalar probability returns a scalar; a probability slice/array or multiple
probabilities returns an owning GC slice. The result type is independent of the
probability type. Use $(LREF rcquantile) for the former reference-counted behavior.

By default observations are copied before partitioning. Setting allowModifySlice
permits rearranging the input and avoids that scratch allocation. When input
modification is enabled, observations must not overlap the probabilities.
Use $(LREF quantileInto) for caller-provided output.
Quantile algorithms and inferred result types follow $(LREF QuantileAlgo).
Omitting probabilities returns the minimum, quartiles, and maximum using
`[0, 0.25, 0.5, 0.75, 1]`.

Params:
    F = explicit output type; inferred from observations when omitted
    algorithm = quantile algorithm, default type7
    allowModifySlice = allow rearranging observations, default false
+/
template quantile(F, QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
    if (isFloatingPoint!F || algorithm == QuantileAlgo.type1 || algorithm == QuantileAlgo.type3)
{
    /++
    Params:
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto quantile(Data, P...)(scope auto ref Data data, scope auto ref P probabilities)
    {
        QuantileContext context;
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.gc)(
            context, data, probabilities);
    }
}

/// ditto
template quantile(QuantileAlgo algorithm = QuantileAlgo.type7, bool allowModifySlice = false)
{
    /++
    Params:
        data = observations as a slice or array
        probabilities = a scalar, probability slice/array, or variadic probabilities; omitted for quartiles
    +/
    auto quantile(Data, P...)(scope auto ref Data data, scope auto ref P probabilities)
    {
        alias F = quantileType!(typeof(quantileSlice(data)), algorithm);
        QuantileContext context;
        return dispatchQuantile!(F, algorithm, allowModifySlice, QuantileAllocation.gc)(
            context, data, probabilities);
    }
}

/// Omit probabilities to obtain the minimum, quartiles, and maximum.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    int[5] data = [4, 0, 3, 1, 2];
    auto result = quantile(data);
    assert(result == [0.0, 1, 2, 3, 4]);
    assert(data[] == [4, 0, 3, 1, 2]);
}

/// ditto
template quantile(F, string algorithm, bool allowModifySlice = false)
{
    mixin("alias quantile = .quantile!(F, QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

/// ditto
template quantile(string algorithm, bool allowModifySlice = false)
{
    mixin("alias quantile = .quantile!(QuantileAlgo." ~ algorithm ~ ", allowModifySlice);");
}

// Defaults preserve result precision, algorithm selection, and owning storage.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.primitives: DeepElementType;
    import std.meta: AliasSeq;
    static foreach (factory; AliasSeq!(quantile, rcquantile))
    {{
        const float[6] data = [9, 0, 7, 2, 5, 3];
        const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
        auto result = factory(data[].sliced);
        static assert(is(DeepElementType!(typeof(result)) == float));
        assert(result == factory(data, levels));
        auto precise = factory!real(data[]);
        static assert(is(DeepElementType!(typeof(precise)) == real));
        assert(precise == factory!real(data, levels));
        assert(factory!"type1"(data) == factory!"type1"(data, levels));
        float[6] workspace = data;
        assert((factory!(QuantileAlgo.type7, true)(workspace) == result));
    }}
    auto escaped = () @safe pure nothrow {
        double[5] local = [4, 3, 2, 1, 0];
        return quantile(local);
    }();
    assert(escaped == [0.0, 1, 2, 3, 4]);
}

version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    import std.experimental.allocator: dispose;
    import mir.primitives: DeepElementType;
    const int[6] data = [9, 0, 7, 2, 5, 3];
    const double[5] levels = [0, 0.25, 0.5, 0.75, 1];
    auto result = makeQuantile!real(Mallocator.instance, data);
    scope(exit) Mallocator.instance.dispose(result.field);
    static assert(is(DeepElementType!(typeof(result)) == real));
    assert(result == rcquantile!real(data, levels));
    auto discrete = makeQuantile!"type1"(Mallocator.instance, data);
    scope(exit) Mallocator.instance.dispose(discrete.field);
    assert(discrete == rcquantile!"type1"(data, levels));
}

/// GC results own their storage independently of local inputs.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: IteratorOf;
    static auto fromLocal()
    {
        double[5] observations = [4, 0, 3, 1, 2];
        const float[3] probabilities = [0.25f, 0.5f, 0.75f];
        auto result = quantile(observations[], probabilities[]);
        static assert(is(IteratorOf!(typeof(result)) == double*));
        observations[] = -1;
        return result;
    }
    auto result = fromLocal();
    assert(result == [1.0, 2.0, 3.0]);
}

version(mir_stat_test)
private struct QuantileTestAllocator(bool throwing = false)
{
    import std.experimental.allocator.mallocator: Mallocator;
    enum alignment = Mallocator.alignment;
    size_t allocations;
    size_t releases;
    size_t failAt;

    auto allocate(size_t bytes)
    {
        ++allocations;
        static if (throwing)
            if (failAt && allocations == failAt)
                throw new Exception("quantile allocation failure");
        return Mallocator.instance.allocate(bytes);
    }

    bool deallocate(void[] memory) @system pure nothrow @nogc
    {
        ++releases;
        return Mallocator.instance.deallocate(memory);
    }
}

// Scratch is released before return, while result storage remains caller-owned.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator: dispose;
    double[5] observations = [4, 0, 3, 1, 2];
    const float[3] probabilities = [0.25f, 0.5f, 0.75f];
    QuantileTestAllocator!() allocator;
    auto result = makeQuantile!double(allocator, observations[], probabilities[]);
    assert(allocator.allocations == 2 && allocator.releases == 1);
    assert(result == [1.0, 2.0, 3.0]);
    allocator.dispose(result.field);
    assert(allocator.releases == 2);

    assert(makeQuantile(allocator, observations[], 0.5) == 2.0);
    assert(allocator.allocations == 3 && allocator.releases == 3);

    // Mutable scratch supplied by the caller needs no temporary allocation.
    auto inPlace = makeQuantile!(double, "type7", true)(
        allocator, observations[], probabilities[]);
    assert(allocator.allocations == 4 && allocator.releases == 3);
    assert(inPlace == [1.0, 2.0, 3.0]);
    allocator.dispose(inPlace.field);
    assert(makeQuantile!(QuantileAlgo.type7, true)(
        allocator, observations[], 0.5) == 2.0);
    assert(allocator.allocations == 4 && allocator.releases == 4);
}

// A safe allocator gives the custom factory an @safe usage path.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.experimental.allocator.gc_allocator: GCAllocator;
    struct SafeAllocator
    {
        enum alignment = GCAllocator.alignment;
        size_t releases;
        void[] allocate(size_t bytes) @safe pure nothrow
        {
            return GCAllocator.instance.allocate(bytes);
        }
        bool deallocate(void[] memory) @safe pure nothrow @nogc
        {
            // Reclamation belongs to the GC; existing aliases remain valid.
            ++releases;
            return true;
        }
    }
    SafeAllocator allocator;
    const double[4] observations = [3, 0, 2, 1];
    const float[2] probabilities = [0.25f, 0.75f];
    auto result = makeQuantile(allocator, observations, probabilities);
    assert(result == [0.75, 2.25] && allocator.releases == 1);
}

// Results survive local stack-backed observations and probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static auto fromLocal()
    {
        double[3] observations = [2, 0, 1];
        const float[2] probabilities = [0, 1];
        return rcquantile(observations[], probabilities[]);
    }
    auto result = fromLocal();
    assert(result == [0.0, 2.0]);
}

// Exceptions during allocation/copying and invalid probabilities release storage.
version(mir_stat_test)
@system pure
unittest
{
    import std.exception: assertThrown;
    import core.exception: AssertError;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: map;
    double[3] observations = [2, 0, 1];
    float[2] probabilities = [0.25f, 0.75f];

    QuantileTestAllocator!true allocator;
    allocator.failAt = 2;
    assertThrown!Exception(makeQuantile(allocator, observations[], probabilities[]));
    assert(allocator.allocations == 2 && allocator.releases == 1);

    static double failCopy(double value) @safe pure
    {
        throw new Exception("quantile observation failure");
    }
    QuantileTestAllocator!() copyAllocator;
    assertThrown!Exception(makeQuantile(copyAllocator,
        observations[].sliced.map!failCopy, probabilities[]));
    assert(copyAllocator.allocations == 1 && copyAllocator.releases == 1);

    QuantileTestAllocator!() invalidAllocator;
    probabilities[1] = 2;
    assertThrown!AssertError(makeQuantile(invalidAllocator, observations[], probabilities[]));
    assert(invalidAllocator.allocations == 2 && invalidAllocator.releases == 2);
}

// Every algorithm shares the same results across allocation policies.
version(mir_stat_test)
@system pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.experimental.allocator: dispose;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    static foreach (algorithm; AliasSeq!(QuantileAlgo.type1, QuantileAlgo.type2,
        QuantileAlgo.type3, QuantileAlgo.type4, QuantileAlgo.type5, QuantileAlgo.type6,
        QuantileAlgo.type7, QuantileAlgo.type8, QuantileAlgo.type9))
    {{
        const int[6] observations = [5, 0, 4, 1, 3, 2];
        const float[3] probabilities = [0, 0.5f, 1];
        auto data = observations[].sliced(2, 3).transposed;
        auto gc = quantile(data, probabilities[]);
        auto rc = rcquantile!algorithm(data, probabilities[]);
        auto typedGc = quantile!algorithm(data, probabilities[]);
        QuantileTestAllocator!() allocator;
        auto custom = makeQuantile!algorithm(allocator, data, probabilities[]);
        scope(exit) allocator.dispose(custom.field);
        foreach (i; 0 .. probabilities.length)
        {
            const expected = rcquantile!algorithm(data, probabilities[i]);
            assert(rc[i] == expected && typedGc[i] == expected && custom[i] == expected);
        }
        assert(gc[0] == 0 && gc[1] == 2.5 && gc[2] == 5);
        assert(observations[] == [5, 0, 4, 1, 3, 2]);
    }}
}

// Into supports strided outputs and independent probability/result element types.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    int[4] observations = [4, 1, 3, 2];
    const double[2] probabilities = [0, 1];
    int[4] backing = [-1, -1, -1, -1];
    auto output = backing[].sliced(2, 2).transposed[0];
    quantileInto!"type1"(observations[], probabilities[], output);
    assert(backing[] == [1, -1, 4, -1]);

    double[0] noProbabilities;
    double[0] noOutput;
    quantileInto(observations[], noProbabilities[], noOutput[]);
    static assert(!__traits(compiles,
        quantileInto(observations[], probabilities[], probabilities[])));
}

// Reject the wrong destination size before modifying observations or output.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    double[3] observations = [2, 0, 1];
    double[2] probabilities = [0.25, 0.75];
    double[1] output = [-1];
    // Catching AssertError requires @system; the calculation itself is @safe.
    bool rejected;
    try
        quantileInto(observations[], probabilities[], output[]);
    catch (AssertError)
        rejected = true;
    assert(rejected);
    assert(observations[] == [2.0, 0.0, 1.0] && output[0] == -1);
}

// Scalar quantiles need no allocation when observations may be rearranged.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double[5] observations = [4, 0, 3, 1, 2];
    assert(quantile!(QuantileAlgo.type7, true)(observations, 0.5) == 2.0);
    assert(quantile!(float, "type7", true)(observations, 0.25) == 1.0f);

    // The default RC path preserves observations without using the GC.
    const double[5] original = [4, 0, 3, 1, 2];
    assert(rcquantile(original, 0.5) == 2.0);
    assert(original[] == [4.0, 0.0, 3.0, 1.0, 2.0]);
}

// Probability precision must not select the allocated quantile result type.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced, IteratorOf;
    import mir.rc.array: RCI;
    import std.meta: AliasSeq;

    static foreach (F; AliasSeq!(float, double, real))
    {{
        static foreach (G; AliasSeq!(float, double, real))
        {{
            double[4] data = [0, 1, 2, 3];
            const G[2] probabilities = [G(0.25), G(0.75)];
            auto result = data[].sliced.rcquantile!F(probabilities[].sliced);
            auto variadic = data[].sliced.rcquantile!F(probabilities[0], probabilities[1]);
            static assert(is(IteratorOf!(typeof(result)) == RCI!F));
            static assert(is(IteratorOf!(typeof(variadic)) == RCI!F));
            foreach (i; 0 .. probabilities.length)
            {
                const expected = data[].sliced.rcquantile!F(probabilities[i]);
                assert(result[i] == expected && variadic[i] == expected);
            }
            assert(data[] == [0.0, 1.0, 2.0, 3.0]);
            assert(probabilities[0] == G(0.25) && probabilities[1] == G(0.75));
        }}
    }}
}

// Discontinuous quantiles preserve integers that float cannot represent exactly.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced, IteratorOf;
    import mir.rc.array: RCI;
    static foreach (algorithm; [QuantileAlgo.type1, QuantileAlgo.type3])
    {{
        int[2] data = [16777217, 16777219];
        float[2] probabilities = [0.25f, 0.75f];
        auto result = data[].sliced.rcquantile!algorithm(probabilities[].sliced);
        auto variadic = data[].sliced.rcquantile!algorithm(0.25f, 0.75f);
        static assert(is(IteratorOf!(typeof(result)) == RCI!int));
        static assert(is(IteratorOf!(typeof(variadic)) == RCI!int));
        foreach (i; 0 .. probabilities.length)
        {
            const expected = data[].sliced.rcquantile!algorithm(probabilities[i]);
            assert(result[i] == expected && variadic[i] == expected);
        }
        assert(result[0] == 16777217);
    }}
}

// An integral result must not round the probabilities before quantile selection.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    int[4] data = [1, 2, 3, 4];
    double[2] probabilities = [0.5, 1.0];
    auto result = data[].sliced.rcquantile!(int, QuantileAlgo.type1)(probabilities[].sliced);
    auto variadic = data[].sliced.rcquantile!(int, QuantileAlgo.type1)(0.5, 1.0);
    assert(result[0] == 2 && result[1] == 4);
    assert(variadic[0] == 2 && variadic[1] == 4);
}

/// Simple example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;

    assert(x.quantile(0.5).approxEqual(2.0));

    auto qtile = [0.25, 0.75].sliced;

    assert(x.quantile(qtile).all!approxEqual([1.0, 3.0]));
    assert(x.quantile(0.25, 0.75).all!approxEqual([1.0, 3.0]));
}

//no change in x by default
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;
    auto x_copy = x.dup;
    auto result = x.quantile(0.5);

    assert(result.approxEqual(2.0));
    assert(x.all!approxEqual(x_copy));
}

/// Quantile of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;

    assert(x.quantile(0.5).approxEqual(5.20));

    auto qtile = [0.25, 0.75].sliced;

    assert(x.quantile(qtile).all!approxEqual([3.250, 8.500]));
}

/// Quantile of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto x = [
        [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2],
        [2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5]
    ].fuse;

    assert(x.quantile(0.5).approxEqual(5.20));

    auto qtile = [0.25, 0.75].sliced;

    assert(x.quantile(qtile).all!approxEqual([3.250, 8.500]));
}

/// Row quantile of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, byDim, map, flattened;

    auto x = [
        [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2],
        [2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5]
    ].fuse;

    auto result0 = [5.200, 5.700];

    // Use byDim or alongDim with map to compute median of row/column.
    assert(x.byDim!0.map!(a => a.quantile(0.5)).all!approxEqual(result0));
    assert(x.alongDim!1.map!(a => a.quantile(0.5)).all!approxEqual(result0));

    auto qtile = [0.25, 0.75].sliced;
    auto result1 = [[3.750, 7.600], [2.825, 9.025]];

    assert(x.byDim!0.map!(a => a.quantile(qtile)).all!(all!approxEqual)(result1));
}

/// Allow modification of input
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;
    auto x_copy = x.dup;

    auto result = x.quantile!(QuantileAlgo.type7, true)(0.5);
    assert(!x.all!approxEqual(x_copy));
}

/// Double-check probability is not modified
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;

    auto qtile = [0.25, 0.75].sliced;
    auto qtile_copy = qtile.dup;

    auto result = x.quantile!("type7", false)(qtile);
    assert(result.all!approxEqual([1.0, 3.0]));
    assert(qtile.all!approxEqual(qtile_copy));
}

// Multi-probability results own their storage after the inputs go out of scope.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced, IteratorOf;
    import mir.rc.array: RCI;

    static auto fromLocalInputs(bool variadic, bool modifyInput)()
    {
        import mir.ndslice.allocation: rcslice;
        static immutable samples = [4.0, 0, 3, 1, 2];
        static immutable levels = [0.25, 0.5, 0.75];
        auto values = rcslice!double(samples);
        auto probabilities = rcslice!double(levels);
        static if (variadic)
            auto result = values.rcquantile!(double, QuantileAlgo.type7, modifyInput)(
                probabilities[0], probabilities[1], probabilities[2]);
        else
            auto result = values.rcquantile!(double, QuantileAlgo.type7, modifyInput)(
                probabilities);

        // This also catches the old borrowed return type deterministically,
        // without depending on whether an allocator reuses the freed block.
        static assert(is(IteratorOf!(typeof(result)) == RCI!double));
        values[] = -1;
        probabilities[] = 0;
        return result;
    }

    static foreach (variadic; [false, true])
    {
        static foreach (modifyInput; [false, true])
        {
            {
                auto result = fromLocalInputs!(variadic, modifyInput)();
                assert(result == [1.0, 2.0, 3.0]);
                // Copies must retain ownership when the original is released.
                auto retained = result;
                result = typeof(result).init;
                assert(retained == [1.0, 2.0, 3.0]);
            }
        }
    }


}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;

    assert(x.quantile!"type1"(0.5).approxEqual(5.20));
    assert(x.quantile!"type2"(0.5).approxEqual(5.20));
    assert(x.quantile!"type3"(0.5).approxEqual(5.20));
    assert(x.quantile!"type4"(0.5).approxEqual(5.20));
    assert(x.quantile!"type5"(0.5).approxEqual(5.20));
    assert(x.quantile!"type6"(0.5).approxEqual(5.20));
    assert(x.quantile!"type7"(0.5).approxEqual(5.20));
    assert(x.quantile!"type8"(0.5).approxEqual(5.20));
    assert(x.quantile!"type9"(0.5).approxEqual(5.20));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto a = [1, 1e100, 1, -1e100].sliced;

    auto x = a * 10_000;

    auto result0 = x.quantile!float(0.5);
    assert(result0 == 10_000f);
    static assert(is(typeof(result0) == float));

    auto result1 = x.quantile!(float, "type8")(0.5);
    assert(result1 == 10_000f);
    static assert(is(typeof(result1) == float));
}

/// Support for integral and user-defined types for type 1 & 3
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.topology: repeat;

    auto x = uint.max.repeat(3);
    assert(x.quantile!(uint, "type1")(0.5) == uint.max);
    assert(x.quantile!(uint, "type3")(0.5) == uint.max);

    static struct Foo {
        float x;
        alias x this;
    }

    Foo[] foo = [Foo(1f), Foo(2f), Foo(3f)];
    assert(foo.quantile!"type1"(0.5) == 2f);
    assert(foo.quantile!"type3"(0.5) == 2f);
}

/// Compute quantile along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: as, iota, alongDim, map, repeat;

    auto x = [
        [0.0, 1, 3],
        [4.0, 5, 7]
    ].fuse;

    assert(x.quantile(0.5).approxEqual(3.5));

    auto m0 = [2.0, 3.0, 5.0];
    assert(x.alongDim!0.map!(a => a.quantile(0.5)).all!approxEqual(m0));
    assert(x.alongDim!(-2).map!(a => a.quantile(0.5)).all!approxEqual(m0));

    auto m1 = [1.0, 5.0];
    assert(x.alongDim!1.map!(a => a.quantile(0.5)).all!approxEqual(m1));
    assert(x.alongDim!(-1).map!(a => a.quantile(0.5)).all!approxEqual(m1));

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!(a => a.quantile(0.5)).all!approxEqual(iota([3, 4, 5], 3 * 4 * 5 / 2)));
}

/// Support for array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;

    double[] x = [3.0, 1.0, 4.0, 2.0, 0.0];

    assert(x.quantile(0.5).approxEqual(2.0));

    double[] qtile = [0.25, 0.75];

    assert(x.quantile(qtile).all!approxEqual([1.0, 3.0]));
}

//@nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
                          2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5];

    assert(x.sliced.rcquantile(0.5).approxEqual(5.20));

    static immutable qtile = [0.25, 0.75];
    static immutable result = [3.250, 8.500];

    assert(x.sliced.rcquantile(qtile).all!approxEqual(result));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
                          2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5];

    auto x = RCArray!double(20);
    foreach(i, ref e; x)
        e = a[i];

    assert(x.rcquantile(0.5).approxEqual(5.20));

    auto qtile = RCArray!double(2);
    qtile[0] = 0.25;
    qtile[1] = 0.75;
    static immutable result = [3.250, 8.500];

    assert(x.rcquantile(qtile).all!approxEqual(result));
}

//x.length = 20, qtile at tenths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;
    auto qtile = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].sliced;

    assert(x.quantile!"type1"(qtile.dup).all!approxEqual([0.2, 1.0, 2.2, 3.5, 4.5, 5.2, 5.8, 8.2, 8.5, 9.2, 9.8]));
    assert(x.quantile!"type2"(qtile.dup).all!approxEqual([0.2, 1.4, 2.35, 3.65, 4.85, 5.2, 6.0, 8.35, 8.85, 9.2, 9.8]));
    assert(x.quantile!"type3"(qtile.dup).all!approxEqual([0.2, 1.0, 2.2, 3.5, 4.5, 5.2, 5.8, 8.2, 8.5, 9.2, 9.8]));
    assert(x.quantile!"type4"(qtile.dup).all!approxEqual([0.2, 1.0, 2.2, 3.5, 4.5, 5.2, 5.8, 8.2, 8.5, 9.2, 9.8]));
    assert(x.quantile!"type5"(qtile.dup).all!approxEqual([0.20, 1.40, 2.35, 3.65, 4.85, 5.20, 6.00, 8.35, 8.85, 9.20, 9.80]));
    assert(x.quantile!"type6"(qtile.dup).all!approxEqual([0.20, 1.08, 2.26, 3.59, 4.78, 5.20, 6.04, 8.41, 9.06, 9.20, 9.80]));
    assert(x.quantile!"type7"(qtile.dup).all!approxEqual([0.20, 1.72, 2.44, 3.71, 4.92, 5.20, 5.96, 8.29, 8.64, 9.20, 9.80]));
    assert(x.quantile!"type8"(qtile.dup).all!approxEqual([0.200000, 1.293333, 2.320000, 3.630000, 4.826667, 5.200000, 6.013333, 8.370000, 8.920000, 9.200000, 9.800000]));
    assert(x.quantile!"type9"(qtile.dup).all!approxEqual([0.2000, 1.3200, 2.3275, 3.6350, 4.8325, 5.2000, 6.0100, 8.3650, 8.9025, 9.200, 9.800]));
}

//x.length = 20, qtile at 5s
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;
    auto qtile = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95].sliced;

    assert(x.quantile!"type1"(qtile.dup).all!approxEqual([0.2, 1.8, 2.5, 3.8, 5.2, 5.2, 6.2, 8.5, 9.2, 9.2]));
    assert(x.quantile!"type2"(qtile.dup).all!approxEqual([0.60, 2.00, 3.00, 4.15, 5.20, 5.50, 7.20, 8.50, 9.20, 9.50]));
    assert(x.quantile!"type3"(qtile.dup).all!approxEqual([0.2, 1.8, 2.5, 3.8, 5.2, 5.2, 6.2, 8.5, 9.2, 9.2]));
    assert(x.quantile!"type4"(qtile.dup).all!approxEqual([0.2, 1.8, 2.5, 3.8, 5.2, 5.2, 6.2, 8.5, 9.2, 9.2]));
    assert(x.quantile!"type5"(qtile.dup).all!approxEqual([0.60, 2.00, 3.00, 4.15, 5.20, 5.50, 7.20, 8.50, 9.20, 9.50]));
    assert(x.quantile!"type6"(qtile.dup).all!approxEqual([0.240, 1.860, 2.750, 4.045, 5.200, 5.530, 7.500, 8.500, 9.200, 9.770]));
    assert(x.quantile!"type7"(qtile.dup).all!approxEqual([0.960, 2.140, 3.250, 4.255, 5.200, 5.470, 6.900, 8.500, 9.200, 9.230]));
    assert(x.quantile!"type8"(qtile.dup).all!approxEqual([0.480000, 1.953333, 2.916667, 4.115000, 5.200000, 5.510000, 7.300000, 8.500000, 9.200000, 9.590000]));
    assert(x.quantile!"type9"(qtile.dup).all!approxEqual([0.51000, 1.96500, 2.93750, 4.12375, 5.20000, 5.50750, 7.27500, 8.50000, 9.20000, 9.56750]));
}

//x.length = 21, qtile at tenths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [ 1.0, 9.3, 0.2, 8.1, 5.5, 3.3, 4.3, 7.9, 5.0, 5.0,
              10.0, 2.4, 1.7, 2.1, 3.6, 5.0, 8.8, 9.8, 6.0, 8.8,
               8.8].sliced;
    auto qtile = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].sliced;

    assert(x.quantile!"type1"(qtile.dup).all!approxEqual([0.2, 1.7, 2.4, 3.6, 5.0, 5.0, 6.0, 8.1, 8.8, 9.3, 10.0]));
    assert(x.quantile!"type2"(qtile.dup).all!approxEqual([0.2, 1.7, 2.4, 3.6, 5.0, 5.0, 6.0, 8.1, 8.8, 9.3, 10.0]));
    assert(x.quantile!"type3"(qtile.dup).all!approxEqual([0.2, 1.0, 2.1, 3.3, 4.3, 5.0, 6.0, 8.1, 8.8, 9.3, 10.0]));
    assert(x.quantile!"type4"(qtile.dup).all!approxEqual([0.20, 1.07, 2.16, 3.39, 4.58, 5.00, 5.80, 8.04, 8.80, 9.25, 10.00]));
    assert(x.quantile!"type5"(qtile.dup).all!approxEqual([0.20, 1.42, 2.31, 3.54, 4.93, 5.00, 6.19, 8.24, 8.80, 9.50, 10.00]));
    assert(x.quantile!"type6"(qtile.dup).all!approxEqual([0.20, 1.14, 2.22, 3.48, 4.86, 5.00, 6.38, 8.38, 8.80, 9.70, 10.00]));
    assert(x.quantile!"type7"(qtile.dup).all!approxEqual([0.2, 1.7, 2.4, 3.6, 5.0, 5.0, 6.0, 8.1, 8.8, 9.3, 10.0]));
    assert(x.quantile!"type8"(qtile.dup).all!approxEqual([0.200000, 1.326667, 2.280000, 3.520000, 4.906667, 5.000000, 6.253333, 8.286667, 8.800000, 9.566667, 10.000000]));
    assert(x.quantile!"type9"(qtile.dup).all!approxEqual([0.2000, 1.3500, 2.2875, 3.5250, 4.9125, 5.0000, 6.2375, 8.2750, 8.8000, 9.5500, 10.0000]));
}

//x.length = 21, qtile at 5s
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [ 1.0, 9.3, 0.2, 8.1, 5.5, 3.3, 4.3, 7.9, 5.0, 5.0,
              10.0, 2.4, 1.7, 2.1, 3.6, 5.0, 8.8, 9.8, 6.0, 8.8,
               8.8].sliced;
    auto qtile = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95].sliced;

    assert(x.quantile!"type1"(qtile.dup).all!approxEqual([1.0, 2.1, 3.3, 4.3, 5.0, 5.5, 7.9, 8.8, 8.8, 9.8]));
    assert(x.quantile!"type2"(qtile.dup).all!approxEqual([1.0, 2.1, 3.3, 4.3, 5.0, 5.5, 7.9, 8.8, 8.8, 9.8]));
    assert(x.quantile!"type3"(qtile.dup).all!approxEqual([0.2, 1.7, 2.4, 3.6, 5.0, 5.5, 7.9, 8.8, 8.8, 9.8]));
    assert(x.quantile!"type4"(qtile.dup).all!approxEqual([0.240, 1.760, 2.625, 3.845, 5.000, 5.275, 7.235, 8.625, 8.800, 9.775]));
    assert(x.quantile!"type5"(qtile.dup).all!approxEqual([0.640, 1.960, 3.075, 4.195, 5.000, 5.525, 7.930, 8.800, 8.975, 9.890]));
    assert(x.quantile!"type6"(qtile.dup).all!approxEqual([0.28, 1.82, 2.85, 4.09, 5.00, 5.55, 7.96, 8.80, 9.15, 9.98]));
    assert(x.quantile!"type7"(qtile.dup).all!approxEqual([1.0, 2.1, 3.3, 4.3, 5.0, 5.5, 7.9, 8.8, 8.8, 9.8]));
    assert(x.quantile!"type8"(qtile.dup).all!approxEqual([0.520000, 1.913333, 3.000000, 4.160000, 5.000000, 5.533333, 7.940000, 8.800000, 9.033333, 9.920000]));
    assert(x.quantile!"type9"(qtile.dup).all!approxEqual([0.55000, 1.92500, 3.01875, 4.16875, 5.00000, 5.53125, 7.93750, 8.80000, 9.01875, 9.91250]));
}

// Integral interpolation uses the result precision, including with float probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    int[2] values = [16_777_216, 16_777_218];
    assert(rcquantile(values, 0.5f) == 16_777_217);
    float[1] probabilities = [0.5f];
    double[1] output;
    quantileInto(values, probabilities, output);
    assert(output[0] == 16_777_217);
    long[2] extremes = [long.min, long.max];
    uint[2] weights = [1, 1];
    assert(rcquantile(extremes, 0.5) == -0.5);
    static foreach (algorithm; [WeightedQuantileAlgo.averagedInverseCDF,
        WeightedQuantileAlgo.frequencyType7, WeightedQuantileAlgo.frequencyType8])
        assert(rcWeightedQuantile!algorithm(extremes, weights, 0.5) == -0.5);
    ulong[2] nearby = [ulong.max - 2, ulong.max];
    assert(rcquantile!(QuantileAlgo.type1)(nearby, 1.0) == ulong.max);
}

// The GC adapter preserves interpolation precision.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    int[2] values = [16_777_216, 16_777_218];
    assert(quantile(values, 0.5f) == 16_777_217);
}

// Mallocator requires @system, but custom interpolation remains @nogc.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import std.experimental.allocator.mallocator: Mallocator;
    int[2] values = [16_777_216, 16_777_218];
    assert(makeQuantile(Mallocator.instance, values, 0.5f) == 16_777_217);
}

// Subtract integral quantiles before rounding their common large offset.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    enum long base = 1L << 53;
    long[3] values = [base + 2, base, base + 1];
    const original = values;
    assert(interquartileRange(values[].sliced) == 1);
    assert(interquartileRange(values[].sliced, 0.25f) == 1);
    assert(interquartileRange(values[].sliced, 0.25, 0.75) == 1);
    assert(interquartileRange!(QuantileAlgo.type1)(values[].sliced) == 2);
    assert(interquartileRange!(double, QuantileAlgo.type1)(values[].sliced) == 2);
    // Type 6 clamps these probabilities to the endpoints. Preserve their
    // exact difference even when floating-point conversion would lose it.
    long[2] endpoints = [base, base + 1];
    assert(interquartileRange!(QuantileAlgo.type6)(endpoints[].sliced, 0, 1) == 1);
    assert(values == original);
    assert(interquartileRange!(QuantileAlgo.type7, true)(values[].sliced) == 1);
    ulong[3] top = [ulong.max - 2, ulong.max - 1, ulong.max];
    assert(interquartileRange(top[].sliced) == 1);
    long[3] bottom = [long.min, long.min + 1, long.min + 2];
    assert(interquartileRange(bottom[].sliced) == 1);
    assert(interquartileRange!(QuantileAlgo.type1)(bottom[].sliced, 0.75, 0.25) == -2);
}

// Integer probability arguments select endpoints just like floating literals.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    int[3] integers = [2, 4, 8];
    double[3] floating = [2, 4, 8];
    assert(interquartileRange(integers[].sliced, 0, 1) == 6);
    assert(interquartileRange(floating[].sliced, 0, 1) == 6);
    assert(interquartileRange(integers[].sliced, 0) == 6);
    assert(interquartileRange(floating[].sliced, 0) == 6);
    assert(interquartileRange!float(floating[].sliced, 0, 1) == 6);
    assert(interquartileRange!(QuantileAlgo.type1)(integers[].sliced, 0, 1) == 6);
    assert(interquartileRange!double(floating[].sliced, 0, 0.5L) == 2);
}

// Floating spreads retain interpolation offsets until the final subtraction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    double[2] adjacent = [0x1p60, 0x1p60 + 256];
    assert(interquartileRange(adjacent[].sliced) == 128);
    assert(interquartileRange(adjacent[].sliced, 0.75, 0.25) == -128);
    assert(interquartileRange(adjacent[].sliced, 0.5, 0.5) == 0);
    double[5] values = [0x1p60, 0x1p60 + 256, 0x1p60 + 512,
        0x1p60 + 768, 0x1p60 + 1024];
    const original = values;
    assert(interquartileRange!float(values[].sliced) == 512);
    assert(values == original);
    assert(interquartileRange!(float, QuantileAlgo.type7, true)(values[].sliced) == 512);
    assert(interquartileRange!(QuantileAlgo.type1)(adjacent[].sliced) == 256);
}

// Weighted endpoint distances can fit even when their subtraction overflows.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[2] extremes = [-T.max, T.max];
        assert(interquartileRange(extremes[].sliced) == T.max);
        assert(interquartileRange(extremes[].sliced, 0, 1) == T.infinity);
        assert(interquartileRange(extremes[].sliced, 0.5, 0.5) == 0);
        T[2] infinite = [T.infinity, T.infinity];
        assert(interquartileRange(infinite[].sliced).isNaN);
    }}
}

// Scale a finite endpoint distance without overflowing the subtraction when
// the weighted distance still fits. Zero weight contributes nothing, even
// when the unweighted distance would overflow.
private W weightedQuantileGap(W, T, G)(T a, T b, G fraction)
{
    const W weight = fraction;
    if (weight == 0)
        return 0;
    const W x = a, y = b, gap = y - x;
    import std.math: isFinite;
    return isFinite(gap) ? gap * weight : y * weight - x * weight;
}

// Shared with robustScale so integral spreads are not reconstructed by
// subtracting two separately rounded quantiles.
package(mir.stat) auto quantileSpread(F, QuantileAlgo algorithm, bool modify, S, P, Q)(
    S source, P lo, Q hi)
{
    import std.traits: isIntegral, Unqual, CommonType;
    import mir.primitives: DeepElementType;
    import mir.ndslice.topology: flattened;
    alias T = Unqual!(DeepElementType!S);
    // Accept integer probability arguments independently of the result type,
    // while retaining the precision of floating-point arguments.
    alias Probability = CommonType!(P, Q);
    static if (isIntegral!Probability) alias Level = double;
    else alias Level = Probability;
    auto lowerProbability = cast(Level) lo;
    auto upperProbability = cast(Level) hi;
    static if (isIntegral!T)
    {
        static if (modify) scope auto workspace = source.flattened;
        else
        {
            import mir.ndslice.allocation: rcslice;
            import mir.ndslice.topology: as;
            auto owner = source.lightScope.flattened.as!T.rcslice;
            scope auto workspace = owner.lightScope;
        }
        const lower = selectQuantile!algorithm(workspace, lowerProbability);
        const upper = selectQuantile!algorithm(workspace, upperProbability);
        const negative = upper.low < lower.low;
        const ulong distance = negative ?
            cast(ulong) lower.low - cast(ulong) upper.low :
            cast(ulong) upper.low - cast(ulong) lower.low;
        static if (isIntegral!F)
        {
            // Selection algorithms have integral results. Do not round a large
            // exact difference through floating point or silently wrap it.
            if (negative)
            {
                static if (F.min < 0)
                {
                    const ulong limit = cast(ulong) F.max + 1;
                    assert(distance <= limit, "interquartileRange: result is out of range");
                    if (distance == limit) return F.min;
                    return cast(F) -cast(long) distance;
                }
                else assert(false, "interquartileRange: negative unsigned result");
            }
            assert(distance <= F.max, "interquartileRange: result is out of range");
            return cast(F) distance;
        }
        else
        {
            const real anchor = negative ? -cast(real) distance : cast(real) distance;
            const real upperOffset = upper.fraction *
                cast(real)(cast(ulong) upper.high - cast(ulong) upper.low);
            const real lowerOffset = lower.fraction *
                cast(real)(cast(ulong) lower.high - cast(ulong) lower.low);
            return cast(F)(anchor + (upperOffset - lowerOffset));
        }
    }
    else static if (isFloatingPoint!T && isFloatingPoint!F)
    {
        static if (modify) scope auto workspace = source.flattened;
        else
        {
            import mir.ndslice.allocation: rcslice;
            auto owner = rcslice!T([source.elementCount], T.init);
            scope auto flat = source.lightScope.flattened;
            foreach (i; 0 .. flat.length)
                owner[i] = flat[i];
            scope auto workspace = owner.lightScope;
        }
        auto lower = selectQuantile!algorithm(workspace, lowerProbability);
        auto upper = selectQuantile!algorithm(workspace, upperProbability);
        alias Work = CommonType!(T, F, Level, double);
        import std.math: isFinite;
        if (!isFinite(lower.low) || !isFinite(lower.high) ||
            !isFinite(upper.low) || !isFinite(upper.high))
        {
            const Work a = lower.fraction == 0 ? cast(Work)lower.low :
                interpolateQuantile!Work(lower.low, lower.high, lower.fraction);
            const Work b = upper.fraction == 0 ? cast(Work)upper.low :
                interpolateQuantile!Work(upper.low, upper.high, upper.fraction);
            return cast(F)(b - a);
        }
        const negative = lowerProbability > upperProbability;
        if (negative)
        {
            auto tmp = lower;
            lower = upper;
            upper = tmp;
        }
        // Retain the selected observations until subtraction. Separately
        // rounded quantiles can lose a small spread around a large offset.
        // For ordered probabilities, combine nonnegative interval lengths.
        Work value;
        if (lower.low == upper.low && lower.high == upper.high)
            value = weightedQuantileGap!Work(lower.low, lower.high, upper.fraction - lower.fraction);
        else
            value = (cast(Work)upper.low - cast(Work)lower.high) +
                weightedQuantileGap!Work(lower.low, lower.high, 1 - lower.fraction) +
                weightedQuantileGap!Work(upper.low, upper.high, upper.fraction);
        return cast(F)(negative ? -value : value);
    }
    else
    {
        auto values = rcquantile!(F, algorithm, modify)(source, lowerProbability, upperProbability);
        return values[1] - values[0];
    }
}

/++
Computes the interquartile range of the input.

For integral observations, the difference is computed before converting a common
large offset to floating point. Selection algorithms retain integral results;
the difference must fit the result type. Probabilities remain floating point
regardless of the result type.

For floating-point observations, interpolation and subtraction are combined
before rounding to the result type. This preserves small spreads around large
common offsets that can be lost by subtracting separately rounded quantiles.

By default, this function computes the result defined by $(LREF quantile), i.e.
`result = quantile(x, 0.75) - quantile(x, 0.25)`. There are also overloads for
providing a low value, as in `result = quantile(x, 1 - low) - quantile(x, low)`
and both a low and high value, as in `result = quantile(x, high) - quantile(x, low)`.

For all $(LREF QuantileAlgo) except $(LINK2 #.QuantileAlgo.type1, $(D QuantileAlgo.type1)) and $(LINK2 #.QuantileAlgo.type3, $(D QuantileAlgo.type3)),
by default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

For $(LINK2 #.QuantileAlgo.type1, $(D QuantileAlgo.type1)) and $(LINK2 #.QuantileAlgo.type3, $(D QuantileAlgo.type3)), the return type is the
$(MATHREF sum, elementType) of the input.

Params:
    F = controls type of output
    quantileAlgo = algorithm for calculating quantile (default: $(LINK2 #.QuantileAlgo.type7, $(D QuantileAlgo.type7)))
    allowModifySlice = controls whether the input is modified in place, default is false

Returns:
    The interquartile range of the input.

See_also:
    $(LREF quantile)
+/
template interquartileRange(F, QuantileAlgo quantileAlgo = QuantileAlgo.type7,
                            bool allowModifySlice = false)
{
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;

    /++
    Params:
        slice = slice
    +/
    @fmamath quantileType!(F, quantileAlgo) interquartileRange(
        Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
    {
        import core.lifetime: move;

        alias FF = typeof(return);
        return quantileSpread!(FF, quantileAlgo, allowModifySlice)(slice.move, 0.25, 0.75);
    }

    /++
    Params:
        slice = slice
        lo = low value
    +/
    @fmamath quantileType!(F, quantileAlgo) interquartileRange(
        Iterator, size_t N, SliceKind kind, P)(
            Slice!(Iterator, N, kind) slice,
            P lo)
    {
        import core.lifetime: move;

        alias FF = typeof(return);
        return quantileSpread!(FF, quantileAlgo, allowModifySlice)(slice.move, lo, 1 - lo);
    }

    /++
    Params:
        slice = slice
        lo = low value
        hi = high value
    +/
    @fmamath quantileType!(F, quantileAlgo) interquartileRange(
        Iterator, size_t N, SliceKind kind, P, Q)(
            Slice!(Iterator, N, kind) slice,
            P lo,
            Q hi)
    {
        import core.lifetime: move;

        alias FF = typeof(return);
        return quantileSpread!(FF, quantileAlgo, allowModifySlice)(slice.move, lo, hi);
    }

    /++
    Params:
        array = array
    +/
    @fmamath quantileType!(F[], quantileAlgo) interquartileRange(scope F[] array...)
    {
        import mir.ndslice.slice: sliced;

        alias FF = typeof(return);
        return .interquartileRange!(FF, quantileAlgo, allowModifySlice)(array.sliced);
    }

    /// ditto
    @fmamath auto interquartileRange(SliceLike)(SliceLike x)
        if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
    {
        import mir.ndslice.slice: toSlice;
        return .interquartileRange!(F, quantileAlgo, allowModifySlice)(x.toSlice);
    }
}

/// ditto
template interquartileRange(QuantileAlgo quantileAlgo = QuantileAlgo.type7,
                            bool allowModifySlice = false)
{
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;

    /// ditto
    @fmamath quantileType!(Slice!(Iterator), quantileAlgo)
        interquartileRange(Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .interquartileRange!(F, quantileAlgo, allowModifySlice)(slice.move);
    }

    /// ditto
    @fmamath quantileType!(Slice!(Iterator), quantileAlgo)
        interquartileRange(Iterator, size_t N, SliceKind kind, F)(
            Slice!(Iterator, N, kind) slice,
            F lo)
    {
        import core.lifetime: move;

        alias FF = typeof(return);
        return .interquartileRange!(FF, quantileAlgo, allowModifySlice)(slice.move, lo);
    }

    /// ditto
    @fmamath quantileType!(Slice!(Iterator), quantileAlgo)
        interquartileRange(Iterator, size_t N, SliceKind kind, F)(
            Slice!(Iterator, N, kind) slice,
            F lo,
            F hi)
    {
        import core.lifetime: move;

        alias FF = typeof(return);
        return .interquartileRange!(FF, quantileAlgo, allowModifySlice)(slice.move, lo, hi);
    }

    /// ditto
    @fmamath quantileType!(T[], quantileAlgo)
        interquartileRange(T)(scope T[] array...)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .interquartileRange!(F, quantileAlgo, allowModifySlice)(array);
    }

    /// ditto
    @fmamath auto interquartileRange(SliceLike)(SliceLike x)
        if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
    {
        import mir.ndslice.slice: toSlice;
        alias F = quantileType!(typeof(x.toSlice), quantileAlgo);
        return .interquartileRange!(F, quantileAlgo, allowModifySlice)(x);
    }
}

/// ditto
template interquartileRange(F, string quantileAlgo, bool allowModifySlice = false)
{
    mixin("alias interquartileRange = .interquartileRange!(F, QuantileAlgo." ~ quantileAlgo ~ ", allowModifySlice);");
}

/// ditto
template interquartileRange(string quantileAlgo, bool allowModifySlice = false)
{
    mixin("alias interquartileRange = .interquartileRange!(QuantileAlgo." ~ quantileAlgo ~ ", allowModifySlice);");
}

/// Simple example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;

    assert(x.interquartileRange.approxEqual(2.0));
    assert(x.interquartileRange(0.25).approxEqual(2.0));
    assert(x.interquartileRange(0.25, 0.75).approxEqual(2.0));
}

//no change in x by default
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;
    auto x_copy = x.dup;
    auto result = x.interquartileRange;

    assert(x.all!approxEqual(x_copy));
}

/// Interquartile Range of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;

    assert(x.interquartileRange.approxEqual(5.25));
}

/// Interquartile Range of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto x = [
        [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2],
        [2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5]
    ].fuse;

    assert(x.interquartileRange.approxEqual(5.25));
}

/// Allow modification of input
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 4.0, 2.0, 0.0].sliced;
    auto x_copy = x.dup;

    auto result = x.interquartileRange!(QuantileAlgo.type7, true);
    assert(!x.all!approxEqual(x_copy));
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
              2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5].sliced;

    assert(x.interquartileRange!"type1".approxEqual(6.0));
    assert(x.interquartileRange!"type2".approxEqual(5.5));
    assert(x.interquartileRange!"type3".approxEqual(6.0));
    assert(x.interquartileRange!"type4".approxEqual(6.0));
    assert(x.interquartileRange!"type5".approxEqual(5.5));
    assert(x.interquartileRange!"type6".approxEqual(5.75));
    assert(x.interquartileRange!"type7".approxEqual(5.25));
    assert(x.interquartileRange!"type8".approxEqual(5.583333));
    assert(x.interquartileRange!"type9".approxEqual(5.5625));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto a = [1, 1e34, 1, -1e34, 0].sliced;

    auto x = a * 10_000;

    auto result0 = x.interquartileRange!float;
    assert(result0.approxEqual(10_000));
    static assert(is(typeof(result0) == float));

    auto result1 = x.interquartileRange!(float, "type8");
    assert(result1.approxEqual(6.666667e37));
    static assert(is(typeof(result1) == float));
}

/// Support for array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    double[] x = [3.0, 1.0, 4.0, 2.0, 0.0];

    assert(x.interquartileRange.approxEqual(2.0));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
                          2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5];

    auto x = RCArray!double(20);
    foreach(i, ref e; x)
        e = a[i];

    assert(x.interquartileRange.approxEqual(5.25));
    assert(x.interquartileRange!double.approxEqual(5.25));
    // The asSlice adapter must forward both result type and algorithm.
    auto selected = x.interquartileRange!(float, QuantileAlgo.type1);
    static assert(is(typeof(selected) == float));
    assert(selected == 6);
}

// Arbitrary test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(interquartileRange(3.0, 1.0, 4.0, 2.0, 0.0).approxEqual(2.0));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [1.0, 9.8, 0.2, 8.5, 5.8, 3.5, 4.5, 8.2, 5.2, 5.2,
                          2.5, 1.8, 2.2, 3.8, 5.2, 9.2, 6.2, 9.2, 9.2, 8.5];

    assert(x.sliced.interquartileRange.approxEqual(5.25));
}

/++
Calculates the median absolute deviation about the median of the input.

The input is not modified. Integral observations retain their differences before
conversion to floating point, so a large common offset does not hide a small
spread. Choosing a narrower floating-point output type does not narrow the
center before the deviations are calculated.

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    F = output type

Returns:
    The median absolute deviation of the input
+/
template medianAbsoluteDeviation(F)
{
    /++
    Params:
        slice = slice
    +/
    @fmamath meanType!F medianAbsoluteDeviation(Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
    {
        import core.lifetime: move;
        import mir.math.common: fabs;
        import mir.ndslice.topology: map;
        import mir.stat.transform: center;
        import std.traits: Unqual, isIntegral, CommonType;

        alias G = typeof(return);
        static assert(isFloatingPoint!G, "medianAbsoluteDeviation: output type must be floating point");
        alias Element = Unqual!(typeof(slice).DeepElement);
        static if (isIntegral!Element && Element.sizeof == 8)
        {
            import mir.ndslice.allocation: rcslice;
            import mir.ndslice.topology: flattened;
            auto flat = slice.lightScope.flattened;
            assert(flat.length > 0, "medianAbsoluteDeviation: slice must have length greater than zero");
            if (flat.length == 1) return G(0);
            if (flat.length == 2)
            {
                const a = flat[0], b = flat[1];
                const ulong gap = a < b ? cast(ulong) b - cast(ulong) a :
                    cast(ulong) a - cast(ulong) b;
                return cast(G) gap / 2;
            }
            // Bias signed integers into unsigned sort order. Distances remain
            // unchanged, and unsigned storage can hold the full-domain gap.
            enum ulong bias = Element.min < 0 ? 1UL << 63 : 0;
            auto owner = flat.map!(a => cast(ulong) a ^ bias).rcslice;
            auto workspace = owner.lightScope;
            const middle = selectQuantile!(QuantileAlgo.type7)(workspace, 0.5);
            const gap = middle.high - middle.low;
            const anchor = middle.fraction == 0 ? middle.low : middle.low + gap / 2;
            const ulong half = middle.fraction == 0 ? 0 : gap & 1;
            // With a half-integral center every deviation has fractional part
            // .5. Store just its integer part, then restore that common half.
            foreach (ref value; workspace)
                value = value <= anchor ? anchor - value : value - anchor - half;
            const deviations = selectQuantile!(QuantileAlgo.type7)(workspace, 0.5);
            // Twice the MAD is integral and no greater than the input range,
            // so this sum fits in ulong even for the full signed domain.
            // Combine before conversion, avoiding a second rounding of .5.
            const ulong twiceMad = deviations.fraction == 0 ?
                2 * deviations.low + half : deviations.low + deviations.high + half;
            return cast(G) twiceMad / 2;
        }
        else static if (isFloatingPoint!Element || isIntegral!Element)
        {
            // F describes the output, not the precision of the center. In
            // particular, do not narrow double observations to a float center.
            static if (isIntegral!Element) alias Work = CommonType!(G, double);
            else alias Work = CommonType!(G, Element);
            return slice.move.center!(median!(Work, false)).map!fabs.median!(G, false);
        }
        else
            return slice.move.center!(median!(G, false)).map!fabs.median!(G, false);
    }
}

/// ditto
@fmamath meanType!(Slice!(Iterator, N, kind))
    medianAbsoluteDeviation(Iterator, size_t N, SliceKind kind)(
        Slice!(Iterator, N, kind) slice)
{
    import core.lifetime: move;

    alias F = typeof(return);
    return medianAbsoluteDeviation!F(slice.move);
}

/// ditto
@fmamath meanType!(T[]) medianAbsoluteDeviation(T)(scope const T[] ar...)
{
    import mir.ndslice.slice: sliced;

    alias G = typeof(return);
    return medianAbsoluteDeviation!G(ar.sliced);
}

/// ditto
@fmamath auto medianAbsoluteDeviation(SliceLike)(SliceLike x)
    if (isConvertibleToSlice!SliceLike && !isSlice!SliceLike)
{
    import mir.ndslice.slice: toSlice;
    return medianAbsoluteDeviation(x.toSlice);
}

/// medianAbsoluteDeviation of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;

    assert(x.medianAbsoluteDeviation.approxEqual(1.25));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.medianAbsoluteDeviation.approxEqual(1.25));
}

/// Median Absolute Deviation of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.medianAbsoluteDeviation.approxEqual(1.25));
}

/// Median Absolute Deviation of dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.medianAbsoluteDeviation.approxEqual(1.25));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.medianAbsoluteDeviation.approxEqual(1.25));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    assert(a.medianAbsoluteDeviation.approxEqual(1.25));
}

// Translation must not hide the spread of nearby 64-bit integers.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;

    foreach (base; [long.min, 0L, 1L << 60, long.max - 8])
    {
        long[8] data;
        foreach (i, ref value; data)
            value = base + cast(long) i;
        foreach (n; 1 .. 9)
        {
            // Exact deviations for the consecutive samples of lengths 1..8.
            static immutable expected = [0.0, 0.5, 1, 1, 1, 1.5, 2, 2];
            const exact = expected[n - 1];
            assert(medianAbsoluteDeviation(data[0 .. n].sliced) == exact);
            assert(medianAbsoluteDeviation!float(data[0 .. n].sliced) == exact);
        }
        foreach (i, value; data)
            assert(value == base + cast(long) i); // Selection only rearranges our copy.
    }

    ulong[5] upper = [ulong.max - 4, ulong.max - 3, ulong.max - 2,
                      ulong.max - 1, ulong.max];
    assert(medianAbsoluteDeviation(upper[].sliced) == 1);
    assert(medianAbsoluteDeviation(upper[3 .. 5].sliced) == 0.5);
    assert(medianAbsoluteDeviation(upper[4 .. 5].sliced) == 0);

    long[2] extremes = [long.min, long.max];
    assert(medianAbsoluteDeviation(extremes[].sliced) == 0x1p63);
    assert(medianAbsoluteDeviation!float(extremes[].sliced) == 0x1p63f);
    long[4] wide = [long.min, long.min, long.max, long.max];
    assert(medianAbsoluteDeviation(wide[].sliced) == 0x1p63);
    // Combining the median deviation and its shared half after conversion
    // would round twice and lose one here. The exact MAD is 2^52 + 1.
    ulong[4] rounding = [0, 1UL << 53, (1UL << 53) + 3, (1UL << 54) + 2];
    assert(medianAbsoluteDeviation(rounding[].sliced) == 0x1.0000000000001p52);

    ulong[4] unsignedWide = [0, 0, ulong.max, ulong.max];
    assert(medianAbsoluteDeviation(unsignedWide[].sliced) == 0x1p63);
}

// Two-element deviations must be independent of input order.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;

    long[2] reversed = [long.max, long.min];
    assert(medianAbsoluteDeviation(reversed[].sliced) == 0x1p63);
    assert(reversed[0] == long.max && reversed[1] == long.min);
    ulong[2] adjacent = [ulong.max, ulong.max - 1];
    assert(medianAbsoluteDeviation(adjacent[].sliced) == 0.5);
    long[2] equal = [long.min, long.min];
    assert(medianAbsoluteDeviation(equal[].sliced) == 0);
}

// A narrow result must not round the center before subtraction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;

    double[5] data = [0x1p60, 0x1p60 + 256, 0x1p60 + 512,
                      0x1p60 + 768, 0x1p60 + 1024];
    auto result = medianAbsoluteDeviation!float(data[].sliced);
    static assert(is(typeof(result) == float));
    assert(result == 256);
    int[5] integers = [int.max - 4, int.max - 3, int.max - 2,
                      int.max - 1, int.max];
    assert(medianAbsoluteDeviation!float(integers[].sliced) == 1);
}

// Integer centering also works through strided, const, and slice-like inputs.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.rc.array: RCArray;

    enum long base = 1L << 60;
    const long[6] data = [base, base + 1, base + 2,
                          base + 3, base + 4, base + 5];
    auto matrix = data[].sliced(2, 3).transposed;
    assert(medianAbsoluteDeviation(matrix) == 1.5);
    assert(medianAbsoluteDeviation(matrix[1]) == 1.5);
    auto owner = RCArray!long(6);
    foreach (i, ref value; owner)
        value = data[i];
    assert(medianAbsoluteDeviation(owner) == 1.5);
}
