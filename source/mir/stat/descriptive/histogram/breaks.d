/++
This module contains algorithms for calculating breaks for histograms.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(REF_ALTTEXT $(TT $2), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.histogram.breaks;

import mir.math.stat: VarianceAlgo;
import mir.primitives: hasShape;
import mir.stat.descriptive.univariate: QuantileAlgo;
import std.traits: isIntegral;

/++
Computes the number of breaks for a histogram using the Sturges' formula.

Calculates the ceiling of the base 2 logarithm of the number of elements of the
input and then adds 1.

Sturges' formula implicitly assumes an approximately normal distribution.

Params:
    CountType = the type that is used to count in histogram bins
Returns:
    The number of breaks

See_also:
    $(LREF scott),
    $(LREF freedmanDiaconis),
    $(WEB en.wikipedia.org/wiki/Histogram, Histogram)
+/
template sturges(CountType)
    if(isIntegral!CountType)
{
    import mir.primitives: hasShape, elementCount;

    /++
    Params:
        x = input
    +/
    CountType sturges(T)(T x)
        if (hasShape!T)
    {
        size_t n = x.elementCount;

        return sturges(n);
    }

    /++
    Params:
        n = count
    +/
    CountType sturges(size_t n)
    {
        assert(n > 0, "sturges: elementCount must be greater than zero");

        import mir.math.common: ceil, log2;

        return cast(CountType) (ceil(log2(cast(double) n)) + 1);
    }
}

/++
Params:
    x = input
+/
size_t sturges(T)(T x)
    if (hasShape!T)
{
    return .sturges!size_t(x);
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto k = x.sturges!size_t;
    assert(k == 4);
    static assert(is(typeof(k) == size_t));

    auto l = x.sturges;
    assert(l == 4);
}

/++
Calculates the number of breaks for a histogram given a step size `h`.

The formula calculates the ceiling of the result from first calculating the
difference between the maximum value of the input and the minimum value of the
input and then dividing that by the step size.

Input must be nonempty and finite, and the width must be positive and finite.
The resulting count must fit CountType. These requirements are checked with
assertions. Constant input returns zero. Integral data with an integral width
uses exact integer arithmetic; other combinations use at least double precision.

Params:
    CountType = the type that is used to count in histogram bins
Returns:
    The number of breaks

See_also:
    $(LREF scott),
    $(LREF sturges),
    $(LREF freedmanDiaconis),
    $(WEB en.wikipedia.org/wiki/Histogram, Histogram)
+/
template binsFromWidth(CountType)
    if(isIntegral!CountType)
{
    import mir.internal.utility: isFloatingPoint;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice;
    import std.traits: Unqual;

    /++
    Params:
        slice = input observations
        h = positive, finite bin width
    +/
    CountType binsFromWidth(Iterator, size_t N, SliceKind kind, H)(
        Slice!(Iterator, N, kind) slice, H h)
    {
        import mir.algorithm.iteration: minmaxIndex;
        import mir.math.common: ceil;
        import mir.ndslice.topology: flattened;
        import mir.primitives: elementCount;
        import std.math: isFinite;
        import std.traits: CommonType;

        alias T = Unqual!(typeof(slice).DeepElement);
        assert(slice.elementCount > 0, "binsFromWidth: input must not be empty");
        assert(h > 0, "binsFromWidth: width must be positive");
        static if (isFloatingPoint!H)
            assert(isFinite(h), "binsFromWidth: width must be finite");
        static if (isFloatingPoint!T)
            foreach (value; slice.flattened)
                assert(isFinite(value), "binsFromWidth: input must be finite");

        auto indexes = slice.minmaxIndex;
        const low = slice[indexes[0]];
        const high = slice[indexes[1]];
        static if (isIntegral!T)
        {
            // Unsigned subtraction preserves the exact nonnegative distance,
            // even across zero or between neighboring values near ulong.max.
            const ulong span = cast(ulong) high - cast(ulong) low;
            static if (isIntegral!H)
            {
                const ulong width = cast(ulong) h;
                // Avoid both floating-point rounding and span + width overflow.
                const ulong count = span / width + (span % width != 0);
                assert(count <= cast(ulong) CountType.max,
                    "binsFromWidth: bin count must fit CountType");
                return cast(CountType) count;
            }
        }
        static if (!(isIntegral!T && isIntegral!H))
        {
            alias F = CommonType!(double, T, H);
            static if (isIntegral!T)
                const F distance = cast(F) span;
            else
                const F distance = cast(F) high - cast(F) low;
            const F width = cast(F) h;
            // Opposite finite extremes can have an infinite difference even
            // though division by the requested width gives a finite bin count.
            const F ratio = isFinite(distance) ? distance / width
                : cast(F) high / width - cast(F) low / width;
            F count = ceil(ratio);
            // A positive span still needs one bin if the ratio underflows.
            if (high > low && count == 0)
                count = 1;
            // An exclusive power-of-two bound avoids rounding CountType.max
            // upwards and accidentally accepting an out-of-range conversion.
            enum F limit = F(CountType.max / 2 + 1) * 2;
            assert(isFinite(count) && count >= 0 && count < limit,
                "binsFromWidth: bin count must fit CountType");
            return cast(CountType) count;
        }
    }

    /++
    Params:
        array = input observations
        h = positive, finite bin width
    +/
    CountType binsFromWidth(T, H)(T[] array, H h)
        if (isFloatingPoint!(Unqual!T) || isIntegral!(Unqual!T))
    {
        import mir.ndslice.slice: sliced;

        return binsFromWidth(array.sliced, h);
    }

    /++
    Params:
        withAsSlice = input observations exposed through asSlice
        h = positive, finite bin width
    +/
    CountType binsFromWidth(T, H)(T withAsSlice, H h)
        if (hasAsSlice!T)
    {
        return binsFromWidth(withAsSlice.asSlice, h);
    }
}

/// binsFromWidth example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].sliced;

    auto k = x.binsFromWidth!size_t(5.496295);
    assert(k == 3);
    static assert(is(typeof(k) == size_t));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.rc.array: RCArray;
    import mir.algorithm.iteration: minmaxPos, minPos, maxPos, minmaxIndex, minIndex, maxIndex;

    static immutable a = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];
    auto y = x.asSlice;

    auto k = x.binsFromWidth!size_t(5.496295);
    assert(k == 3);
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    auto k = x.binsFromWidth!size_t(5.496295);
    assert(k == 3);
}

/++
Computes the number of breaks for a histogram using the Scott's normal
reference rule.

Calculates the break width using 3.49 times the sample standard deviation
divided by the cubed root of the number of elements of the input. The number of
breaks is then calculated using `binsFromWidth`.

It is considered optimal for normally distributed random samples.

Params:
    CountType = the type that is used to count in histogram bins
    varianceAlgo = Algorithm used to calculate variance
Returns:
    The number of breaks

See_also:
    $(LREF sturges),
    $(LREF freedmanDiaconis),
    $(LREF binsFromWidth),
    $(WEB en.wikipedia.org/wiki/Histogram, Histogram)
+/
template scott(CountType, VarianceAlgo varianceAlgo = VarianceAlgo.online)
    if(isIntegral!CountType)
{
    import mir.internal.utility: isFloatingPoint;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice;
    import std.traits: isIterable, Unqual;

    /++
    Params:
        r = range
    +/
    CountType scott(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;
        import mir.math.common: pow, sqrt;
        import mir.math.stat: meanType, VarianceAccumulator;
        import mir.math.sum: ResolveSummationType, Summation;
        import mir.primitives: elementCount;

        alias F = meanType!(Range);
        auto varianceAccumulator = VarianceAccumulator!(F, varianceAlgo, ResolveSummationType!(Summation.appropriate, Range, F))(r);
        auto h = 3.49 * varianceAccumulator.variance!F(false).sqrt / (pow(varianceAccumulator.count, 1.0 / 3));
        assert(h > 0, "scott: bin width must be greater than zero");

        return binsFromWidth!CountType(r, h);
    }
}

/++
Params:
    varianceAlgo = Algorithm used to calculate variance
+/
template scott(VarianceAlgo varianceAlgo = VarianceAlgo.online)
{
    import mir.internal.utility: isFloatingPoint;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice;
    import std.traits: isIterable, Unqual;

    /++
    Params:
        r = range
    +/
    size_t scott(Range)(Range r)
        if (isIterable!Range)
    {
        return .scott!(size_t, varianceAlgo)(r);
    }
}

/// Scott example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].sliced;

    auto k = x.scott!size_t;
    assert(k == 3);
    static assert(is(typeof(k) == size_t));

    auto l = x.scott;
    assert(l == 3);
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];

    auto k = x.scott!size_t;
    assert(k == 3);
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

    auto k = x.scott!size_t;
    assert(k == 3);
}

/++
Computes the number of breaks for a histogram using Freedman-Diaconis' choice.

Calculates the break width using two times the interquartile range of the input
divided by the cubed root of the number of elements of the input. The number of
breaks is then calculated using `binsFromWidth`.

When the interquartile range is zero, this function makes the same adjustments
as R's `nclass.FD` function, which steadily widens the range used in the
interquartile range to a maximum width of [1/512, 511/512] (with an adjustment
to the multiplier as well).

It is less sensitive to the standard deviation of outliers than Scott's rule.

Params:
    CountType = the type that is used to count in histogram bins
    quantileAlgo = the quantile algorithm used (default: QuantileAlgo.type7)
    allowModifySlice = controls whether the input is modified in place (default: false)
Returns:
    The number of breaks

See_also:
    $(LREF sturges),
    $(LREF freedmanDiaconis),
    $(LREF binsFromWidth),
    $(WEB en.wikipedia.org/wiki/Histogram, Histogram)
+/
template freedmanDiaconis(CountType,
                          QuantileAlgo quantileAlgo = QuantileAlgo.type7,
                          bool allowModifySlice = false)
    if(isIntegral!CountType)
{
    import mir.internal.utility: isFloatingPoint;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice;
    import std.traits: Unqual;

    /++
    Params:
        slice = slice
    +/
    CountType freedmanDiaconis(Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
    {
        import core.lifetime: move;
        import mir.math.common: pow;
        import mir.ndslice.topology: flattened;
        import mir.primitives: elementCount;
        import mir.stat.descriptive.univariate: interquartileRange, quantileType;

        alias F = quantileType!(Slice!(Iterator), quantileAlgo);
        // This allows the interquartileRange to be calculated over the slice
        // multiple times using allowModifySlice=true. This helps reduce some
        // work
        static if (!allowModifySlice) {
            import mir.ndslice.allocation: rcslice;
            import mir.ndslice.topology: as;

            auto view = slice.lightScope;
            auto val = view.as!(Unqual!(slice.DeepElement)).rcslice;
            auto temp = val.lightScope.flattened;
        } else {
            auto temp = slice.flattened;
        }
        auto iqr = temp.interquartileRange!(F, quantileAlgo, true);
        typeof(iqr) h;
        if (iqr == 0) {
            ushort divisor = 8;
            F q;
            while (iqr == 0 && divisor <= 512)
            {
                q = 1.0 / divisor;
                iqr = temp.interquartileRange!(F, quantileAlgo, true)(q, 1 - q);
                divisor = cast(ushort) (divisor << 1);
            }
            assert(iqr > 0, "freedmanDiaconis: interquartile range must be larger than zero");
            // R makes this adjustment to the multiplier, instead of times 2 for
            // this case
            h = iqr / (1 - 2 * q);
        } else {
            h = 2 * iqr;
        }
        h /= pow(temp.elementCount, 1.0 / 3);
        return binsFromWidth!CountType(temp.move, h);
    }

    /++
    Params:
        array = array
    +/
    CountType freedmanDiaconis(T)(T[] array)
        if (isFloatingPoint!(Unqual!T))
    {
        import mir.ndslice.slice: sliced;

        return freedmanDiaconis(array.sliced);
    }

    /++
    Params:
        withAsSlice = withAsSlice
    +/
    CountType freedmanDiaconis(T)(T withAsSlice)
        if (hasAsSlice!T)
    {
        return freedmanDiaconis(withAsSlice.asSlice);
    }
}

template freedmanDiaconis(QuantileAlgo quantileAlgo = QuantileAlgo.type7,
                          bool allowModifySlice = false)
{
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice;

    /++
    Params:
        slice = slice
    +/
    size_t freedmanDiaconis(Iterator, size_t N, SliceKind kind)(
            Slice!(Iterator, N, kind) slice)
    {
        return .freedmanDiaconis!(size_t, quantileAlgo, allowModifySlice)(slice);
    }

    /++
    Params:
        array = array
    +/
    size_t freedmanDiaconis(T)(T[] array)
        if (isFloatingPoint!(Unqual!T))
    {
        import mir.ndslice.slice: sliced;

        return .freedmanDiaconis!(size_t, quantileAlgo, allowModifySlice)(array.sliced);
    }

    /++
    Params:
        withAsSlice = withAsSlice
    +/
    size_t freedmanDiaconis(T)(T withAsSlice)
        if (hasAsSlice!T)
    {
        return .freedmanDiaconis!(size_t, quantileAlgo, allowModifySlice)(withAsSlice.asSlice);
    }
}

/// freedmanDiaconis Example
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto k = x.freedmanDiaconis!size_t;
    assert(k == ((7.0 - 0.0) / 3.5));
    static assert(is(typeof(k) == size_t));

    auto l = x.freedmanDiaconis;
    assert(l == ((7.0 - 0.0) / 3.5));
}

// Test example vs. R in extreme case
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: ceil, pow;
    import mir.stat.descriptive: quantile;
    import mir.ndslice.allocation: slice;

    auto x = slice!double([500], 0.0);
    x[0] = 1;

    auto k = x.freedmanDiaconis!size_t;
    assert(k == ceil(((1.0 - 0.0) / ((x.quantile(1.0 - 1.0 / 512) / (1.0 - 2.0 / 512)) / pow(500.0, 1.0 / 3)))));
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.rc.array: RCArray;

    static immutable a = [0.0, 1, 2, 3, 4, 5, 6, 7];

    auto x = RCArray!double(8);
    foreach(i, ref e; x)
        e = a[i];

    auto k = x.freedmanDiaconis!size_t;
    assert(k == ((7.0 - 0.0) / 3.5));
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7];

    auto k = x.freedmanDiaconis!size_t;
    assert(k == ((7.0 - 0.0) / 3.5));
}

// Break selection can use reference-counted working storage without the GC.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: rcslice;
    static immutable double[8] data = [0, 1, 2, 3, 4, 5, 6, 7];
    auto values = rcslice!double(data[]);
    assert(sturges(values) == 4);
    assert(binsFromWidth!uint(values, 2.0) == 4);
    assert(scott(values) > 0);
    assert(freedmanDiaconis(values) > 0);
}

// Exact integer arithmetic, including full-width spans and nearby large values.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;

    int[2] ordinary = [0, 11];
    assert(binsFromWidth!uint(ordinary[], 5) == 3);
    assert(binsFromWidth!uint(ordinary[].sliced, 5) == 3);
    int[2] extreme = [int.min, int.max];
    assert(binsFromWidth!uint(extreme[], 2147483648.0) == 2);
    assert(binsFromWidth!uint(extreme[], 2147483648UL) == 2);
    long[2] full = [long.min, long.max];
    assert(binsFromWidth!ulong(full[], 1) == ulong.max);
    assert(binsFromWidth!ulong(full[], ulong.max) == 1);
    ulong[2] nearby = [ulong.max - 11, ulong.max];
    assert(binsFromWidth!uint(nearby[], 5) == 3);
    assert(binsFromWidth!uint(nearby[], 5.0) == 3);
    int[2] constant = [7, 7];
    assert(binsFromWidth!uint(constant[], 5) == 0);
    static foreach (C; AliasSeq!(byte, ubyte, short, ushort, int, uint, long, ulong))
    {{
        ulong[2] atLimit = [0, cast(ulong) C.max];
        assert(binsFromWidth!C(atLimit[], 1) == C.max);
    }}
}

// Finite floating-point extremes, including overflowing spans and tiny ratios.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[2] wide = [-T.max, T.max];
        assert(binsFromWidth!uint(wide[], T.max) == 2);
        T[2] tiny = [0, nextUp(T(0))];
        assert(binsFromWidth!uint(tiny[], T.max) == 1);
        T[2] ordinary = [0, 11];
        assert(binsFromWidth!uint(ordinary[], 5) == 3);
        T[2] constant = [7, 7];
        assert(binsFromWidth!uint(constant[], T(5)) == 0);
    }}
}

// Invalid inputs and counts must be rejected before conversion to CountType.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;
    int[] empty;
    assertThrown!AssertError(binsFromWidth!uint(empty, 1));
    int[2] ordinary = [0, 11];
    assertThrown!AssertError(binsFromWidth!uint(ordinary[], 0));
    assertThrown!AssertError(binsFromWidth!uint(ordinary[], -1));
    static foreach (C; AliasSeq!(byte, ubyte, short, ushort, int, uint, long))
    {{
        ulong[2] tooMany = [0, cast(ulong) C.max + 1];
        assertThrown!AssertError(binsFromWidth!C(tooMany[], 1));
    }}
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T[2] ordinaryFloat = [0, 11];
        foreach (width; [T(0), T(-1), T.nan, T.infinity])
            assertThrown!AssertError(binsFromWidth!uint(ordinaryFloat[], width));
        foreach (invalid; [T.nan, T.infinity, -T.infinity])
        {
            T[3] input = [0, invalid, 1];
            assertThrown!AssertError(binsFromWidth!uint(input[], 1));
        }
        T[2] tooMany = [0, 256];
        assertThrown!AssertError(binsFromWidth!ubyte(tooMany[], T(1)));
        T[2] atLimit = [0, 255];
        assert(binsFromWidth!ubyte(atLimit[], T(1)) == 255);
        T[2] beyondUlong = [0, T(2) ^^ 64];
        assertThrown!AssertError(binsFromWidth!ulong(beyondUlong[], T(1)));
        T[2] enormous = [0, T.max];
        assertThrown!AssertError(binsFromWidth!ulong(enormous[], T.min_normal));
    }}
}

// Arrays use the same element-count semantics as slices, with default or
// explicitly selected count types. No caller-side UFCS import is required.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    double[8] values = [0, 1, 2, 3, 4, 5, 6, 7];
    auto array = values[];
    const(double)[] readOnly = array;
    auto defaultCount = sturges(array);
    auto explicitCount = sturges!uint(array);
    static assert(is(typeof(defaultCount) == size_t));
    static assert(is(typeof(explicitCount) == uint));
    assert(defaultCount == 4);
    assert(explicitCount == 4);
    assert(sturges(readOnly) == 4);
    assert(sturges!uint(readOnly) == 4);
    assert(sturges!uint(array.sliced) == explicitCount);
}
