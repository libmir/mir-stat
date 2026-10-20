/++
Arithmetic, geometric, harmonic, and weighted means.

Use the functions to summarize a range or array. The accumulator types
provide the corresponding arithmetic, geometric, and weighted mean operations through `put`.

Import `mir.stat.descriptive.univariate.means` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

Weighted means use `wmean(values, weights)` and divide by total weight by
default. $(REF AssumeWeights, mir, stat, descriptive, univariate, means)
can skip normalization when callers guarantee that weights sum to one.
For unnormalized weighted sums, see $(REF wsum, mir, stat, descriptive, univariate, sums).

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF mean, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF gmean, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF hmean, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF wmean, mir, stat, descriptive, univariate, means)$(NBSP)
    ))
    $(TR $(TD Accumulators) $(TD
        $(REF MeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF GMeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF WMeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
    ))
    $(TR $(TD Weight assumptions) $(TD $(REF AssumeWeights, mir, stat, descriptive, univariate, means)))
))

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall, Ilya Yaroshenko
Copyright: 2022-3 Mir Stat Authors.

Macros:
SUB2REF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, descriptive, $1)$(NBSP)
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.univariate.means;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType;

/++
Output range for mean.
+/
struct MeanAccumulator(T, Summation summation)
{
    import mir.primitives: elementCount, hasShape;
    import std.traits: isIterable;

    ///
    size_t count;
    ///
    Summator!(T, summation) summator;

    ///
    F mean(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) summator.sum / cast(F) count;
    }

    ///
    F sum(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) summator.sum;
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        static if (hasShape!Range)
        {
            count += r.elementCount;
            summator.put(r);
        }
        else
        {
            foreach(x; r)
            {
                count++;
                summator.put(x);
            }
        }
    }

    ///
    void put()(T x)
    {
        count++;
        summator.put(x);
    }

    ///
    void put(F = T)(MeanAccumulator!(F, summation) m)
    {
        import mir.stat.internal.summation : mergeSummator;

        count += m.count;
        mergeSummator(summator, m.summator);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    MeanAccumulator!(double, Summation.pairwise) x;
    x.put([0.0, 1, 2, 3, 4].sliced);
    assert(x.mean == 2);
    x.put(5);
    assert(x.mean == 2.5);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    MeanAccumulator!(float, Summation.pairwise) x;
    x.put([0, 1, 2, 3, 4].sliced);
    assert(x.mean == 2);
    assert(x.sum == 10);
    x.put(5);
    assert(x.mean == 2.5);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25];
    double[] y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    MeanAccumulator!(float, Summation.pairwise) m0;
    m0.put(x);
    MeanAccumulator!(float, Summation.pairwise) m1;
    m1.put(y);
    m0.put(m1);
    assert(m0.mean == 29.25 / 12);
}

// Merging compensated sums must retain contributions below the rounded sum.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta : AliasSeq;

    static foreach (mode; AliasSeq!(Summation.kbn, Summation.kb2))
    {{
        MeanAccumulator!(double, mode) source, merged;
        source.put(0x1p54);
        source.put(1.0);
        merged.put(source);
        merged.put(-0x1p54);
        assert(merged.count == 3);
        assert(merged.sum == 1);
        assert(merged.mean == 1.0 / 3);

        // The merge must not consume or alter the source's corrections.
        source.put(-0x1p54);
        assert(source.sum == 1);
        assert(source.count == 3);
    }}
}

// Retain scalar merging for summators without a direct merge operation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    alias Accumulator = MeanAccumulator!(double, Summation.decimal);
    Accumulator source, merged;
    source.put(1.0);
    source.put(3.0);
    merged.put(2.0);
    merged.put(source);
    assert(merged.count == 3 && merged.sum == 6);
    assert(merged.mean == 2);
    assert(source.count == 2 && source.sum == 4);
    merged.put(Accumulator.init);
    assert(merged.count == 3 && merged.sum == 6);
    merged.put(merged);
    assert(merged.count == 6 && merged.sum == 12);
    assert(merged.mean == 2);
}

// Precise summation also retains overflow internally until later cancellation.
// Copying this summator by value currently requires @system in mir-algorithm.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    alias Accumulator = MeanAccumulator!(double, Summation.precise);
    Accumulator source, merged;
    source.put(double.max);
    source.put(double.max);
    source.put(1.0);
    merged.put(source);
    merged.put(-double.max);
    merged.put(-double.max);
    assert(merged.count == 5);
    assert(merged.sum == 1);
    assert(merged.mean == 0.2);

    source.put(-double.max);
    source.put(-double.max);
    assert(source.sum == 1);

    // Empty and self merges preserve the accumulated state and count.
    merged.put(Accumulator.init);
    assert(merged.count == 5 && merged.sum == 1);
    merged.put(merged);
    assert(merged.count == 10 && merged.sum == 2);
    assert(merged.mean == 0.2);
}

// Retry a nonfinite floating-point mean without overflowing the raw sum.
// The caller supplies an independently saved traversal of the same observations.
private T recoverOverflowingMean(T, Range)(scope Range values, size_t count, T original)
{
    import std.math : abs, isFinite;

    if (!count)
        return original;

    // A power of two at least twice the count leaves headroom for the large
    // terms' sum. Multiplication by its reciprocal is exact unless it enters
    // the subnormal range; keep those small terms in a separate unscaled sum.
    T scale = 2;
    for (size_t n = count - 1; n; n >>= 1)
        scale *= 2;
    const T inverse = 1 / scale;
    const T threshold = T.min_normal * scale;
    Summator!(T, Summation.precise) high, low;
    T lower = T.infinity, upper = -T.infinity;
    foreach (value; values)
    {
        const T x = cast(T) value;
        // Genuine NaN/Inf inputs (including overflow on conversion to T) are
        // not finite-input sum overflow. Preserve the original result.
        if (!isFinite(x))
            return original;
        if (x < lower) lower = x;
        if (x > upper) upper = x;
        if (abs(x) < threshold)
            low.put(x);
        else
            high.put(x * inverse);
    }

    const T highSum = high.sum;
    const T unscaled = highSum * scale;
    T result;
    if (isFinite(unscaled))
    {
        // Cancellation can cross the two sums. Add the large subtotal to the
        // still-precise small sum before rounding, so a tiny remainder is not
        // lost by rounding the two opposing subtotals independently.
        low.put(unscaled);
        result = low.sum / cast(T) count;
    }
    else
    {
        // The unscaled subtotal itself overflows: divide before restoring its
        // scale. At this magnitude the small sum cannot cancel the large one.
        result = highSum * (scale / cast(T) count) + low.sum / cast(T) count;
    }
    // Rounding on rescaling must not leave the finite input interval.
    return result < lower ? lower : result > upper ? upper : result;
}

/++
Computes the mean of the input.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

For built-in floating-point results, a nonfinite result is retried with
scaled precise summation when the input can be traversed again. This supports
arrays, slices whose flattened view is a forward range, and other forward
ranges. For example, the mean of two maximum finite values is still that
maximum value, even though their sum overflows. A forward range's saved
traversal must yield the same observations as the first traversal.

The retry preserves the original result if an observation is nonfinite after
conversion to the result type. Finite results use the selected summation
algorithm without a retry; this is not a general correction for cancellation
error. Single-pass inputs and $(LREF MeanAccumulator) do not retry.

Params:
    F = controls type of output
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    The mean of all the elements in the input, must be floating point or complex type

See_also:
    $(MATHREF_ALT sum, Summation)
+/
template mean(F, Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable, isFloatingPoint;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!F mean(Range)(Range r)
        if (isIterable!Range)
    {
        alias G = typeof(return);
        MeanAccumulator!(G, ResolveSummationType!(summation, Range, G)) mean;
        static if (isFloatingPoint!G)
        {
            import std.range.primitives : isForwardRange, save;
            static if (isSlice!Range)
            {
                import mir.ndslice.topology : flattened;
                auto flat = r.flattened;
                static if (isForwardRange!(typeof(flat)))
                    auto saved = flat.save;
            }
            else static if (isForwardRange!Range)
                auto saved = r.save;
        }
        mean.put(r.move);
        const result = mean.mean;
        static if (__traits(compiles, saved))
        {
            import std.math : isFinite;
            if (!isFinite(result))
                return recoverOverflowingMean!G(saved.move, mean.count, result);
        }
        return result;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!F mean(scope const F[] ar...)
    {
        alias G = typeof(return);
        MeanAccumulator!(G, ResolveSummationType!(summation, const(G)[], G)) mean;
        mean.put(ar);
        const result = mean.mean;
        static if (isFloatingPoint!G)
        {
            import std.math : isFinite;
            if (!isFinite(result))
                return recoverOverflowingMean!G(ar, mean.count, result);
        }
        return result;
    }
}

/// ditto
template mean(Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!Range mean(Range)(Range r)
        if (isIterable!Range)
    {
        alias F = typeof(return);
        return .mean!(F, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath meanType!T mean(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .mean!(F, summation)(ar);
    }
}

/// ditto
template mean(F, string summation)
{
    mixin("alias mean = .mean!(F, Summation." ~ summation ~ ");");
}

/// ditto
template mean(string summation)
{
    mixin("alias mean = .mean!(Summation." ~ summation ~ ");");
}

// Finite observations can have a representable mean but an overflowing sum.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta : AliasSeq;
    import std.math : isNaN;
    import mir.ndslice.slice : sliced;
    import mir.ndslice.dynamic : transposed;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise,
        Summation.kbn, Summation.kb2, Summation.precise))
    {{
        T[4] values = [T.max, T.max, T.max, T.max];
        assert(mean!(T, mode)(values[]) == T.max);
        assert(mean!(T, mode)(T.max, T.max) == T.max);
        assert(mean!(T, mode)(values[].sliced) == T.max);
        assert(mean!(T, mode)(values[].sliced(2, 2).transposed) == T.max);
        assert(mean!(T, mode)(values[].sliced(2, 2).transposed[0]) == T.max);
        values[] = -T.max;
        assert(mean!(T, mode)(values[]) == -T.max);

        T[] empty;
        assert(isNaN(mean!(T, mode)(empty)));
        assert(isNaN(mean!(T, mode)(T.nan, T(1))));
        // Preserve the selected summator's handling of infinities.
        MeanAccumulator!(T, mode) original;
        original.put(T.infinity);
        original.put(T(1));
        const actual = mean!(T, mode)(T.infinity, T(1));
        assert(isNaN(original.mean) ? isNaN(actual) : original.mean == actual);
    }}

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Dividing every input first would erase this subnormal remainder.
        const T tiny = T.min_normal * T.epsilon;
        T[9] values = [T.max, T.max, -T.max, -T.max,
            tiny, tiny, tiny, tiny, tiny];
        assert(mean!(T, Summation.naive)(values[]) == tiny);

        // Cancellation between the scaled and unscaled groups must happen
        // before their subtotals are rounded. The exact mean is 5/9 * tiny,
        // which rounds to tiny rather than zero.
        const T boundary = T.min_normal * 32;
        T[9] cross = [T.max, T.max, -T.max, -T.max,
            boundary, -boundary / 2, -boundary / 2, 5 * tiny, 0];
        assert(mean!(T, Summation.naive)(cross[]) == tiny);
        cross[] = -cross[];
        assert(mean!(T, Summation.naive)(cross[]) == -tiny);
    }}
    // Conversion overflow is not repaired as if it were sum overflow.
    assert(mean!float(double.max, double.max) == float.infinity);
}

// Save forward ranges before the first pass advances a shared cursor.
version(mir_stat_test)
unittest
{
    static struct Forward
    {
        const(double)[] data;
        size_t* position;
        size_t ownPosition;
        bool empty() const { return (position ? *position : ownPosition) == data.length; }
        double front() const { return data[position ? *position : ownPosition]; }
        void popFront() { if (position) ++*position; else ++ownPosition; }
        auto save() { return Forward(data, null, position ? *position : ownPosition); }
    }
    static struct Single
    {
        const(double)[] data;
        size_t* position;
        bool empty() const { return *position == data.length; }
        double front() const { return data[*position]; }
        void popFront() { ++*position; }
    }
    double[2] values = [double.max, double.max];
    size_t position;
    assert(mean(Forward(values[], &position)) == double.max);
    assert(position == 2);
    position = 0;
    assert(mean(Single(values[], &position)) == double.infinity);
    assert(position == 2);
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.complex;
    alias C = Complex!double;

    assert(mean([1.0, 2, 3]) == 2);
    assert(mean([C(1, 3), C(2), C(3)]) == C(2, 1));

    assert(mean!float([0, 1, 2, 3, 4, 5].sliced(3, 2)) == 2.5);

    static assert(is(typeof(mean!float([1, 2, 3])) == float));
}

/// Mean of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    assert(x.mean == 29.25 / 12);
}

/// Mean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;

    assert(x.mean == 29.25 / 12);
}

/// Column mean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto result = [1, 4.25, 3.25, 1.5, 2.5, 2.125];

    // Use byDim or alongDim with map to compute mean of row/column.
    assert(x.byDim!1.map!mean.all!approxEqual(result));
    assert(x.alongDim!0.map!mean.all!approxEqual(result));

    // FIXME
    // Without using map, computes the mean of the whole slice
    // assert(x.byDim!1.mean == x.sliced.mean);
    // assert(x.alongDim!0.mean == x.sliced.mean);
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;

    //Set sum algorithm or output type

    auto a = [1, 1e100, 1, -1e100].sliced;

    auto x = a * 10_000;

    assert(x.mean!"kbn" == 20_000 / 4);
    assert(x.mean!"kb2" == 20_000 / 4);
    assert(x.mean!"precise" == 20_000 / 4);
    assert(x.mean!(double, "precise") == 20_000.0 / 4);

    auto y = uint.max.repeat(3);
    assert(y.mean!ulong == 12884901885 / 3);
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;

    auto y = x.mean;
    assert(y.approxEqual(29.0 / 12, 1.0e-10));
    static assert(is(typeof(y) == double));

    assert(x.mean!float.approxEqual(29f / 12, 1.0e-10));
}

/++
Mean works for complex numbers and other user-defined types (provided they
can be converted to a floating point or complex type)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex;
    alias C = Complex!double;

    auto x = [C(1.0, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    assert(x.mean.approxEqual(C(2.5, 3.5)));
}

/// Compute mean tensors along specified dimention of tensors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice: alongDim, iota, as, map;
    /++
      [[0,1,2],
       [3,4,5]]
     +/
    auto x = iota(2, 3).as!double;
    assert(x.mean == (5.0 / 2.0));

    auto m0 = [(0.0+3.0)/2.0, (1.0+4.0)/2.0, (2.0+5.0)/2.0];
    assert(x.alongDim!0.map!mean == m0);
    assert(x.alongDim!(-2).map!mean == m0);

    auto m1 = [(0.0+1.0+2.0)/3.0, (3.0+4.0+5.0)/3.0];
    assert(x.alongDim!1.map!mean == m1);
    assert(x.alongDim!(-1).map!mean == m1);

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!mean == iota([3, 4, 5], 3 * 4 * 5 / 2));
}

/// Arbitrary mean
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(mean(1.0, 2, 3) == 2);
    assert(mean!float(1, 2, 3) == 2);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    assert([1.0, 2, 3, 4].mean == 2.5);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!mean;
    assert(y.all!approxEqual([1.5, 3.5]));
    static assert(is(meanType!(typeof(y)) == double));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    assert(x.sliced.mean == 29.25 / 12);
    assert(x.sliced.mean!float == 29.25 / 12);
}

///
package(mir)
template hmeanType(T)
{
    import mir.math.sum: sumType;

    alias U = sumType!T;

    static if (__traits(compiles, {
        U t = U.init + cast(U) 1; //added for when U.init = 0
        auto temp = cast(U) 1 / t + cast(U) 1 / t;
    })) {
        alias V = typeof(cast(U) 1 / ((cast(U) 1 / U.init + cast(U) 1 / U.init) / cast(U) 2));
        alias hmeanType = statType!V;
    } else {
        static assert(0, "hmeanType: Can't calculate hmean of elements of type " ~ U.stringof);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static assert(is(hmeanType!(int[]) == double));
    static assert(is(hmeanType!(double[]) == double));
    static assert(is(hmeanType!(float[]) == float));
    static assert(is(hmeanType!(Complex!float[]) == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static struct Foo {
        float x;
        alias x this;
    }

    static struct Bar {
        Complex!float x;
        alias x this;
    }

    static assert(is(hmeanType!(Foo[]) == float));
    static assert(is(hmeanType!(Bar[]) == Complex!float));
}

/*
Recover a nonnegative harmonic mean when its reciprocal calculation overflows
or enters the subnormal range. Dividing the smallest positive observation by
each observation produces ratios in [0, 1], including at least one ratio of 1.
Their sum cannot underflow, and multiplying the scale by count / sum avoids
forming the potentially unrepresentable reciprocal of an observation.

Only replayable built-in floating-point inputs use this recovery. NaN and
negative observations retain the original result; they do not satisfy the
nonnegative-input argument above. Callers save the traversal before consuming
it and check the reciprocal mean once, outside the ordinary reduction loop.
*/
private T recoverHarmonicMean(T, Range)(Range values, T original)
{
    import std.math : isNaN;
    import std.range.primitives : save;

    T scale = T.infinity, upper = 0;
    size_t count;
    bool hasZero;
    foreach (value; values.save)
    {
        // Check before narrowing: a negative observation can round to -0.
        if (isNaN(value) || value < 0)
            return original;
        const T x = cast(T) value;
        ++count;
        hasZero |= x == 0;
        if (x > 0 && x < scale)
            scale = x;
        if (x > upper)
            upper = x;
    }
    // A zero dominates the reciprocal sum, but only after ruling out NaN and
    // negative inputs. Preserve the original sign when it already yielded zero.
    if (hasZero)
        return original == 0 ? original : T(0);
    if (!count || scale == T.infinity)
        return original;

    Summator!(T, Summation.kbn) ratios;
    foreach (value; values)
        ratios.put(scale / cast(T) value);
    const result = scale * (cast(T) count / ratios.sum);
    // Rescaling must not round outside the interval of positive observations.
    return result < scale ? scale : result > upper ? upper : result;
}

/++
Computes the harmonic mean of the input.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

For replayable built-in floating-point inputs and results, a nonfinite, zero,
or subnormal reciprocal mean triggers a scaled recalculation. This avoids losing
a representable harmonic mean when individual reciprocals or their sum
overflow, or when reciprocals are too small to retain normal precision.
The recalculation applies to nonnegative observations, including zero and
positive infinity, and uses compensated summation. A zero observation gives
zero unless a negative value or NaN is also present.
Recovery arithmetic uses the result type; requesting a narrower result type
can still lose values when observations are converted to that type.

Ordinary inputs retain the selected summation algorithm. Negative or NaN
observations retain the original calculation, as do single-pass inputs and
complex or user-defined element types. A saved forward traversal must yield
the same observations as the original traversal.

Params:
    F = controls type of output
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    harmonic mean of all the elements of the input, must be floating point or complex type

See_also:
    $(MATHREF_ALT sum, Summation)
+/
template hmean(F, Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable;

    /++
    Params:
        r = range
    +/
    @fmamath hmeanType!F hmean(Range)(Range r)
        if (isIterable!Range)
    {
        import mir.ndslice.topology: map;

        alias G = typeof(return);
        auto numerator = cast(G) 1;

        import std.range.primitives : isForwardRange, ElementType, save;
        static if (isFloatingPoint!G)
        {
            static if (isSlice!Range)
            {
                import mir.ndslice.topology : flattened;
                auto flat = r.flattened;
                static if (isForwardRange!(typeof(flat)) && isFloatingPoint!(ElementType!(typeof(flat))))
                    auto saved = flat.save;
            }
            else static if (isForwardRange!Range && isFloatingPoint!(ElementType!Range))
                auto saved = r.save;
        }

        static if (summation == Summation.fast && __traits(compiles, r.move.map!"numerator / a"))
        {
            auto inverseMean = r.move.map!"numerator / a".mean!(G, summation);
        }
        else
        {
            MeanAccumulator!(G, ResolveSummationType!(summation, Range, G)) imean;
            foreach (e; r)
                imean.put(numerator / e);
            auto inverseMean = imean.mean;
        }
        auto result = numerator / inverseMean;
        static if (__traits(compiles, saved))
        {
            import std.math : isNormal;
            if (!isNormal(inverseMean))
                return recoverHarmonicMean!G(saved.move, result);
        }
        return result;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath hmeanType!F hmean(scope const F[] ar...)
    {
        alias G = typeof(return);

        auto numerator = cast(G) 1;

        static if (summation == Summation.fast && __traits(compiles, ar.map!"numerator / a"))
        {
            auto inverseMean = ar.map!"numerator / a".mean!(G, summation);
        }
        else
        {
            MeanAccumulator!(G, ResolveSummationType!(summation, const(G)[], G)) imean;
            foreach (e; ar)
                imean.put(numerator / e);
            auto inverseMean = imean.mean;
        }
        auto result = numerator / inverseMean;
        static if (isFloatingPoint!G && isFloatingPoint!F)
        {
            import std.math : isNormal;
            if (!isNormal(inverseMean))
                return recoverHarmonicMean!G(ar, result);
        }
        return result;
    }
}

/// ditto
template hmean(Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable;

    /++
    Params:
        r = range
    +/
    @fmamath hmeanType!Range hmean(Range)(Range r)
        if (isIterable!Range)
    {
        alias F = typeof(return);
        return .hmean!(F, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath hmeanType!T hmean(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .hmean!(F, summation)(ar);
    }
}

/// ditto
template hmean(F, string summation)
{
    mixin("alias hmean = .hmean!(F, Summation." ~ summation ~ ");");
}

/// ditto
template hmean(string summation)
{
    mixin("alias hmean = .hmean!(Summation." ~ summation ~ ");");
}

// Recover representable means despite overflowing or subnormal reciprocals.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta : AliasSeq;
    import std.math : isNaN, abs;
    import mir.ndslice.slice : sliced;
    import mir.ndslice.dynamic : transposed;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise,
        Summation.kbn, Summation.kb2, Summation.precise, Summation.fast))
    {{
        foreach (x; [T.min_normal * T.epsilon, T.min_normal / 4, T.max])
        {
            const T[2] values = [x, x];
            assert(values[].hmean!(T, mode) == x);
            assert(hmean!(T, mode)(x, x) == x);
            assert(values[].sliced.hmean!(T, mode) == x);
        }
        // H(x, 2x) = 4x/3, independently of the implementation.
        const T x = T.min_normal / 4;
        const T expected = x * (T(4) / 3);
        assert(abs(hmean!(T, mode)(x, 2*x) - expected)
            <= 2 * T.min_normal * T.epsilon);
        assert(hmean!(T, mode)(T(0), T(4)) == 0);
        assert(hmean!(T, mode)(T.infinity, T(4)) == 8);
        assert(hmean!(T, mode)(T.infinity, T.infinity) == T.infinity);
        assert(hmean!(T, mode)(T.nan, T(4)).isNaN);
        assert(hmean!(T, mode)(T(0), T.nan).isNaN);
        assert(hmean!(T, mode)(T(-2), T(4)) == -8);
        const T[] empty;
        assert(empty.hmean!(T, mode).isNaN);
    }}
    const double[4] matrix = [double.max, double.max, double.max, double.max];
    assert(matrix[].sliced(2, 2).transposed.hmean == double.max);
    assert(matrix[].sliced(2, 2).transposed[0].hmean == double.max);

    // A negative double must not become an accepted zero during float recovery.
    const double tiny = cast(double)(float.min_normal * float.epsilon) / 2;
    const double[2] mixed = [-tiny, tiny];
    assert(mixed[].hmean!float.isNaN);
}

// Recovery must use an independent saved traversal, not an advanced cursor.
version(mir_stat_test)
unittest
{
    static struct Forward
    {
        const(double)[] data;
        size_t* position;
        size_t ownPosition;
        bool empty() const { return (position ? *position : ownPosition) == data.length; }
        double front() const { return data[position ? *position : ownPosition]; }
        void popFront() { if (position) ++*position; else ++ownPosition; }
        auto save() { return Forward(data, null, position ? *position : ownPosition); }
    }
    static struct Single
    {
        const(double)[] data;
        size_t* position;
        bool empty() const { return *position == data.length; }
        double front() const { return data[*position]; }
        void popFront() { ++*position; }
    }
    double[2] values = [double.min_normal / 4, double.min_normal / 4];
    size_t position;
    assert(hmean(Forward(values[], &position)) == values[0]);
    assert(position == 2);
    position = 0;
    assert(hmean!(Summation.naive)(Single(values[], &position)) == 0);
    assert(position == 2);
}

/// Harmonic mean of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [20.0, 100.0, 2000.0, 10.0, 5.0, 2.0].sliced;

    assert(x.hmean.approxEqual(6.97269));
}

/// Harmonic mean of matrix
version(mir_stat_test)
pure @safe
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [20.0, 100.0, 2000.0],
        [10.0, 5.0, 2.0]
    ].fuse;

    assert(x.hmean.approxEqual(6.97269));
}

/// Column harmonic mean of matrix
version(mir_stat_test)
pure @safe
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [20.0, 100.0, 2000.0],
        [ 10.0, 5.0, 2.0]
    ].fuse;

    auto y = [13.33333, 9.52381, 3.996004];

    // Use byDim or alongDim with map to compute mean of row/column.
    assert(x.byDim!1.map!hmean.all!approxEqual(y));
    assert(x.alongDim!0.map!hmean.all!approxEqual(y));
}

/// Can also pass arguments to hmean
version(mir_stat_test)
pure @safe nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: repeat;
    import mir.ndslice.slice: sliced;

    //Set sum algorithm or output type
    auto x = [1, 1e-100, 1, -1e-100].sliced;

    assert(x.hmean!"kb2".approxEqual(2));
    assert(x.hmean!"precise".approxEqual(2));
    assert(x.hmean!(double, "precise").approxEqual(2));

    //Provide the summation type
    assert(float.max.repeat(3).hmean!double.approxEqual(float.max));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [20, 100, 2000, 10, 5, 2].sliced;

    auto y = x.hmean;

    assert(y.approxEqual(6.97269));
    static assert(is(typeof(y) == double));

    assert(x.hmean!float.approxEqual(6.97269));
}

/++
hmean works for complex numbers and other user-defined types (provided they
can be converted to a floating point or complex type)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex.math: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.complex;
    alias C = Complex!double;

    auto x = [C(1, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    assert(x.hmean.approxEqual(C(1.97110904, 3.14849332)));
}

/// Arbitrary harmonic mean
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = hmean(20.0, 100, 2000, 10, 5, 2);
    assert(x.approxEqual(6.97269));

    auto y = hmean!float(20, 100, 2000, 10, 5, 2);
    assert(y.approxEqual(6.97269));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [20.0, 100.0, 2000.0, 10.0, 5.0, 2.0];

    assert(x.sliced.hmean.approxEqual(6.97269));
    assert(x.sliced.hmean!float.approxEqual(6.97269));
}

private
F nthroot(F)(in F x, in size_t n)
    if (isFloatingPoint!F)
{
    import mir.math.common: sqrt, pow;

    if (n > 2) {
        return pow(x, cast(F) 1 / cast(F) n);
    } else if (n == 2) {
        return sqrt(x);
    } else if (n == 1) {
        return x;
    } else {
        return cast(F) 1;
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(nthroot(9.0, 0).approxEqual(1));
    assert(nthroot(9.0, 1).approxEqual(9));
    assert(nthroot(9.0, 2).approxEqual(3));
    assert(nthroot(9.5, 2).approxEqual(3.08220700));
    assert(nthroot(9.0, 3).approxEqual(2.08008382));
}

/++
Output range for gmean.
+/
struct GMeanAccumulator(T)
    if (isMutable!T && isFloatingPoint!T)
{
    import mir.math.numeric: ProdAccumulator;
    import mir.primitives: elementCount, hasShape;

    ///
    size_t count;
    ///
    ProdAccumulator!T prodAccumulator;

    /++
    Return the geometric mean in type `F`. Zero and infinite results are
    preserved when `F` has a narrower exponent range than the stored product.

    The average product exponent is computed in `F`. For very large or very
    small observations, rounding this exponent can reduce the accuracy of
    the result even when the stored product does not overflow or underflow.
    +/
    F gmean(F = T)() const @property
        if (isFloatingPoint!F)
    {
        import mir.math.common: exp2, floor;
        import std.math : isFinite;

        const F root = nthroot(cast(F) prodAccumulator.mantissa, count);
        const F exponent = cast(F) prodAccumulator.exp / count;
        if (count && (exponent >= F.max_exp || exponent < F.min_exp - 1))
        {
            // A narrower result type can overflow or underflow the scaling
            // factor even when the root is zero or infinite. Preserve the root
            // instead of producing NaN from zero * infinity or infinity * zero.
            // Same-range and wider results keep the original scaling path.
            static if (F.max_exp < T.max_exp || F.min_exp > T.min_exp)
            {
                if (root == 0 || !isFinite(root))
                    return root;
            }
            // Keep exp2 in the normal range and apply the mantissa before
            // the final scaling, which may legitimately produce a subnormal.
            const F halfExponent = floor(exponent / 2);
            return (root * exp2(exponent - halfExponent)) * exp2(halfExponent);
        }
        return root * exp2(exponent);
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        static if (hasShape!Range)
        {
            count += r.elementCount;
            prodAccumulator.put(r);
        }
        else
        {
            foreach(x; r)
            {
                count++;
                prodAccumulator.put(x);
            }
        }
    }

    ///
    void put()(T x)
    {
        count++;
        prodAccumulator.put(x);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    GMeanAccumulator!double x;
    x.put([1.0, 2, 3, 4].sliced);
    assert(x.gmean.approxEqual(2.21336384));
    x.put(5);
    assert(x.gmean.approxEqual(2.60517108));
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    GMeanAccumulator!float x;
    x.put([1, 2, 3, 4].sliced);
    assert(x.gmean.approxEqual(2.21336384));
    x.put(5);
    assert(x.gmean.approxEqual(2.60517108));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: fabs;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double))
    {
        static assert(gmean(T.max) == T.max);
        static assert(gmean(T.min_normal * T.epsilon) == T.min_normal * T.epsilon);
    }

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T smallest = T.min_normal * T.epsilon;
        foreach (value; [T.max, smallest, 3 * smallest])
        {
            GMeanAccumulator!T accumulator;
            accumulator.put(value);
            assert(accumulator.gmean == value);
            assert(gmean(value) == value);
            foreach (i; 1 .. 17)
            {
                accumulator.put(value);
                auto result = accumulator.gmean;
                assert(result > 0 && result <= T.max);
                assert(fabs(result / value - 1) <= 8 * T.epsilon);
            }
            assert(gmean(value, value) == value);
        }

        // Exact powers exercise a fractional average exponent in the
        // subnormal range, as well as a large finite geometric mean.
        assert(gmean(smallest, 4 * smallest) == 2 * smallest);
        assert(fabs(gmean(T.max / 4, T.max) / (T.max / 2) - 1) <= 8 * T.epsilon);
        assert(fabs(gmean(T(1), T(4)) - 2) <= 8 * T.epsilon);
        assert(gmean(T(0), T(4)) == 0);
        assert(gmean(T.infinity, T(4)) == T.infinity);
        auto nanResult = gmean(T.nan, T(4));
        assert(nanResult != nanResult);
    }}

    GMeanAccumulator!double mixed;
    mixed.put(cast(double) float.max);
    assert(mixed.gmean!float == float.max);
    mixed = GMeanAccumulator!double.init;
    mixed.put(cast(double) (float.min_normal * float.epsilon));
    assert(mixed.gmean!float == float.min_normal * float.epsilon);
}

// Preserve special values when the result has a narrower exponent range.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math : isNaN;
    GMeanAccumulator!double zero, infinite;
    zero.put(0.0);
    infinite.put(double.infinity);
    foreach (i; 0 .. 32)
    {
        zero.put(double.max);
        infinite.put(double.min_normal * double.epsilon);
    }
    assert(zero.gmean!float == 0);
    assert(infinite.gmean!float == float.infinity);
    infinite.put(0.0);
    assert(infinite.gmean!float.isNaN);
    // Empty input retains its existing result.
    assert(GMeanAccumulator!double.init.gmean == double.infinity);
}

///
package(mir)
template gmeanType(T)
{
    // TODO: including copy because visibility in mir.math.numeric is set to package
    private template prodType(T)
    {
        import mir.math.sum: elementType;

        alias U = elementType!T;

        static if (__traits(compiles, {
            auto temp = U.init * U.init;
            temp *= U.init;
        })) {
            alias V = typeof(U.init * U.init);
            alias prodType = statType!(V, false);
        } else {
            static assert(0, "prodType: Can't prod elements of type " ~ U.stringof);
        }
    }

    alias U = prodType!T;

    static if (__traits(compiles, {
        auto temp = U.init * U.init;
        auto a = nthroot(temp, 2);
        temp *= U.init;
    })) {
        alias V = typeof(nthroot(U.init * U.init, 2));
        alias gmeanType = statType!(V, false);
    } else {
        static assert(0, "gmeanType: Can't calculate gmean of elements of type " ~ U.stringof);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static assert(is(gmeanType!int == double));
    static assert(is(gmeanType!double == double));
    static assert(is(gmeanType!float == float));
    static assert(is(gmeanType!(int[]) == double));
    static assert(is(gmeanType!(double[]) == double));
    static assert(is(gmeanType!(float[]) == float));
}

/++
Computes the geometric average of the input.

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

Params:
    r = range, must be finite iterable
Returns:
    The geometric average of all the elements in the input, must be floating point type

See_also:
    $(MATHREF_ALT numeric, prod)
+/
@fmamath gmeanType!F gmean(F, Range)(Range r)
    if (isFloatingPoint!F && isIterable!Range)
{
    import core.lifetime: move;

    alias G = typeof(return);
    GMeanAccumulator!G gmean;
    gmean.put(r.move);
    return gmean.gmean;
}

/// ditto
@fmamath gmeanType!Range gmean(Range)(Range r)
    if (isIterable!Range)
{
    import core.lifetime: move;

    alias G = typeof(return);
    return .gmean!(G, Range)(r.move);
}

/++
Params:
    ar = values
+/
@fmamath gmeanType!F gmean(F)(scope const F[] ar...)
    if (isFloatingPoint!F)
{
    alias G = typeof(return);
    GMeanAccumulator!G gmean;
    gmean.put(ar);
    return gmean.gmean;
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(gmean([1.0, 2, 3]).approxEqual(1.81712059));

    assert(gmean!float([1, 2, 3, 4, 5, 6].sliced(3, 2)).approxEqual(2.99379516));

    static assert(is(typeof(gmean!float([1, 2, 3])) == float));
}

/// Geometric mean of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 2.0].sliced;

    assert(x.gmean.approxEqual(2.36178395));
}

/// Geometric mean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    auto x = [
        [3.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 2.0]
    ].fuse;

    assert(x.gmean.approxEqual(2.36178395));
}

/// Column gmean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    auto x = [
        [3.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 2.0]
    ].fuse;
    auto result = [2.44948974, 2.73861278, 2.73861278, 1.41421356, 2.29128784, 2.91547594];

    // Use byDim or alongDim with map to compute mean of row/column.
    assert(x.byDim!1.map!gmean.all!approxEqual(result));
    assert(x.alongDim!0.map!gmean.all!approxEqual(result));

    // FIXME
    // Without using map, computes the mean of the whole slice
    // assert(x.byDim!1.gmean.all!approxEqual(result));
    // assert(x.alongDim!0.gmean.all!approxEqual(result));
}

/// Can also set output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;

    auto x = [5120.0, 7340032, 32, 3758096384].sliced;

    assert(x.gmean!float.approxEqual(259281.45295212));

    auto y = uint.max.repeat(2);
    assert(y.gmean!float.approxEqual(cast(float) uint.max));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [5, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 10].sliced;

    auto y = x.gmean;
    static assert(is(typeof(y) == double));

    assert(x.gmean!float.approxEqual(2.79160522));
}

/// gean works for user-defined types, provided the nth root can be taken for them
version(mir_stat_test)
@safe pure nothrow
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [Foo(1.0), Foo(2.0), Foo(3.0)].sliced;
    assert(x.gmean.approxEqual(1.81712059));
}

/// Compute gmean tensors along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, iota, map;

    auto x = [
        [1.0, 2, 3],
        [4.0, 5, 6]
    ].fuse;

    assert(x.gmean.approxEqual(2.99379516));

    auto result0 = [2.0, 3.16227766, 4.24264069];
    assert(x.alongDim!0.map!gmean.all!approxEqual(result0));
    assert(x.alongDim!(-2).map!gmean.all!approxEqual(result0));

    auto result1 = [1.81712059, 4.93242414];
    assert(x.alongDim!1.map!gmean.all!approxEqual(result1));
    assert(x.alongDim!(-1).map!gmean.all!approxEqual(result1));

    auto y = [
        [
            [1.0, 2, 3],
            [4.0, 5, 6]
        ], [
            [7.0, 8, 9],
            [10.0, 9, 10]
        ]
    ].fuse;

    auto result3 = [
        [2.64575131, 4.0,        5.19615242],
        [6.32455532, 6.70820393, 7.74596669]
    ];
    assert(y.alongDim!0.map!gmean.all!approxEqual(result3));
}

/// Arbitrary gmean
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(gmean(1.0, 2, 3).approxEqual(1.81712059));
    assert(gmean!float(1, 2, 3).approxEqual(1.81712059));
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert([1.0, 2, 3, 4].gmean.approxEqual(2.21336384));
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert(gmean([1, 2, 3]).approxEqual(1.81712059));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [3.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 2.0];

    assert(x.sliced.gmean.approxEqual(2.36178395));
    assert(x.sliced.gmean!float.approxEqual(2.36178395));
}

private void putter2(Slices, T, U, Summation summation1, Summation summation2)
    (scope Slices slices, ref Summator!(T, summation1) seed1, ref Summator!(U, summation2) seed2)
{
    import mir.functional: Tuple;
    static if (is(Slices == Tuple!(V1, V2), V1, V2)) {
        seed1.put(slices[0]);
        seed2.put(slices[1]);
    } else {
        import mir.ndslice.internal: frontOf2;
        while (!slices.empty)
        {
            frontOf2!(slices)[0].putter2(seed1, seed2);
            slices.popFront;
        }
    }
}

/++
Assumptions used for weighted moments
+/
enum AssumeWeights : bool
{
    /++
    Primary, does not assume weights sum to one
    +/
    primary,

    /++
    Assumes weights sum to one
    +/
    sumToOne
}

/++
Output range for wmean.
+/
struct WMeanAccumulator(T, Summation summation, AssumeWeights assumeWeights,
                        U = T, Summation weightsSummation = summation)
{
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, kindOf;
    import std.range.primitives: isInputRange;
    import std.traits: isIterable;

    ///
    Summator!(T, summation) wsummator;

    static if (!assumeWeights) {
        ///
        Summator!(U, weightsSummation) weights;
    }

    ///
    F wmean(F = T)() const @safe @property pure nothrow @nogc
    {
        static if (assumeWeights) {
            return this.wsum!F;
        } else {
            assert(this.weight!F != 0, "weight must not equal zero");
            return this.wsum!F / this.weight!F;
        }
    }

    // Match scalar put: convert each operand before multiplying.
    private static auto weightedProduct(A, B)(A a, B b)
    {
        return cast(T) a * cast(U) b;
    }

    ///
    F wsum(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) wsummator.sum;
    }

    ///
    F weight(F = U)() const @safe @property pure nothrow @nogc
    {
        return cast(F) weights.sum;
    }

    ///
    void put(Slice1, Slice2)(Slice1 s, Slice2 w)
        if (isSlice!Slice1 && isSlice!Slice2)
    {
        static assert (Slice1.N == Slice2.N, "s and w must have the same number of dimensions");
        static assert (kindOf!Slice1 == kindOf!Slice2, "s and w must have the same kind");

        import mir.ndslice.slice: Contiguous;
        import mir.ndslice.topology: zip, map;
        import mir.functional: naryFun;
        import mir.math.sum: elementType;

        // Preserve the existing mapping for matching types, including its code generation.
        static if (is(elementType!Slice1 == T) && is(elementType!Slice2 == U))
            alias product = naryFun!"a * b";
        else
            alias product = weightedProduct;

        assert(s._lengths == w._lengths, "WMeanAcumulator.put: both slices must have the same lengths");

        static if (kindOf!Slice1 != Contiguous && Slice1.N > 1) {
            assert(s.strides == w.strides, "WMeanAccumulator.put: cannot put canonical and universal slices when strides do not match");
            auto combine = s.zip!true(w);
        } else {
            auto combine = s.zip!false(w);
        }

        static if (assumeWeights) {
            auto combine2 = combine.map!product;
            wsummator.put(combine2);
        } else {
            auto combine2 = combine.map!("b", product);
            combine2.putter2(weights, wsummator);
        }
    }

    ///
    void put(SliceLike1, SliceLike2)(SliceLike1 s, SliceLike2 w)
        if (isConvertibleToSlice!SliceLike1 && !isSlice!SliceLike1 &&
            isConvertibleToSlice!SliceLike2 && !isSlice!SliceLike2)
    {
        import mir.ndslice.slice: toSlice;
        this.put(s.toSlice, w.toSlice);
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range && !assumeWeights)
    {
        import mir.primitives: hasShape, elementCount;
        static if (hasShape!Range) {
            // Summation can consume the cursor shared by copies of an input range.
            auto count = r.elementCount;
            wsummator.put(r);
            weights.put(cast(U) count);
        } else {
            foreach(x; r)
            {
                this.put(x);
            }
        }
    }

    ///
    void put(RangeA, RangeB)(RangeA r, RangeB w)
        if (isInputRange!RangeA && !isConvertibleToSlice!RangeA &&
            isInputRange!RangeB && !isConvertibleToSlice!RangeB)
    {
        while (!r.empty && !w.empty)
        {
            this.put(r.front, w.front);
            r.popFront;
            w.popFront;
        }
        assert(r.empty && w.empty, "WMeanAccumulator.put: lengths must match");
    }

    ///
    void put()(T x, U w)
    {
        static if (!assumeWeights) {
            weights.put(w);
        }
        wsummator.put(x * w);
    }

    ///
    void put()(T x)
        if (!assumeWeights)
    {
        weights.put(cast(U) 1);
        wsummator.put(x);
    }

    ///
    void put(F = T, G = U)(WMeanAccumulator!(F, summation, assumeWeights, G, weightsSummation) wm)
        if (!assumeWeights) // because calculating is easier. When assumeWeightsSumtoOne = true, need to divide original wsummator and wm by 2.
    {
        weights.put(cast(U) wm.weights);
        wsummator.put(cast(T) wm.wsummator);
    }
}

// Capture the shape before summation consumes a shared input cursor.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    struct Cursor { double[] values; }

    struct SharedInput
    {
        Cursor* remaining;
        bool empty() { return remaining.values.length == 0; }
        double front() { return remaining.values[0]; }
        void popFront() { remaining.values = remaining.values[1 .. $]; }
        @property size_t[1] shape() const { return [remaining.values.length]; }
    }

    static foreach (method; [Summation.naive, Summation.pairwise])
    {{
        auto values = new Cursor;
        values.values = [1, 2, 3];
        WMeanAccumulator!(double, method, AssumeWeights.primary) accumulator;
        accumulator.put(SharedInput(values));
        assert(values.values.length == 0);
        assert(accumulator.weight == 3);
        assert(accumulator.wsum == 6);
        assert(accumulator.wmean == 2);

        // An exhausted range contributes neither observations nor weight.
        accumulator.put(SharedInput(values));
        assert(accumulator.weight == 3);
        assert(accumulator.wsum == 6);

        values.values = [4, 5];
        accumulator.put(SharedInput(values));
        assert(accumulator.weight == 5);
        assert(accumulator.wsum == 15);
        assert(accumulator.wmean == 3);
    }}
}

/// Assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    x.put([0.0, 1, 2, 3, 4].sliced, [0.2, 0.2, 0.2, 0.2, 0.2].sliced);
    x.wmean.should == 2;
    x.put(5, 0.0);
    x.wmean.should == 2;
}

// dynamic array test, assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    x.put([0.0, 1, 2, 3, 4], [0.2, 0.2, 0.2, 0.2, 0.2]);
    x.wmean.should == 2;
}

// static array test, assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    static immutable y = [0.0, 1, 2, 3, 4];
    static immutable w = [0.2, 0.2, 0.2, 0.2, 0.2];
    x.put(y, w);
    x.wmean.should == 2;
}

// 2-d slice test, assume weights sum to 1
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.fuse: fuse;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    auto y = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;
    auto w = [
        [1.0 / 21, 2.0 / 21, 3.0 / 21],
        [4.0 / 21, 5.0 / 21, 6.0 / 21]
    ].fuse;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// universal 2-d slice test, assume weights sum to 1, using map
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: iota, map, universal;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    auto y = iota([2, 3]).universal;
    auto w = iota([2, 3], 1).map!(a => a / 21.0).universal;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// 2-d canonical slice test, assume weights sum to 1, using map
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: canonical, iota, map;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    auto y = iota([2, 3]).canonical;
    auto w = iota([2, 3], 1).map!(a => a / 21.0).canonical;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

/// Do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    x.put([0.0, 1, 2, 3, 4].sliced, [1, 2, 3, 4, 5].sliced);
    x.wmean.shouldApprox == 40.0 / 15;
    x.put(5, 6);
    x.wmean.shouldApprox == 70.0 / 21;
}

// dynamic array test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    x.put([0.0, 1, 2, 3, 4], [1, 2, 3, 4, 5]);
    x.wmean.shouldApprox == 40.0 / 15;
}

// static array test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    static immutable y = [0.0, 1, 2, 3, 4];
    static immutable w = [1, 2, 3, 4, 5];
    x.put(y, w);
    x.wmean.shouldApprox == 40.0 / 15;
}

// 2-d slice test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.fuse: fuse;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;
    auto w = [
        [1.0, 2, 3],
        [4.0, 5, 6]
    ].fuse;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// universal slice test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: iota, universal;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota(6).universal;
    auto w = iota([6], 1).universal;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// canonical slice test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: canonical, iota;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota(6).canonical;
    auto w = iota([6], 1).canonical;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// 2-d universal slice test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: iota, universal;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota([2, 3]).universal;
    auto w = iota([2, 3], 1).universal;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// 2-d canonical slice test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: canonical, iota;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota([2, 3]).canonical;
    auto w = iota([2, 3], 1).canonical;
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

/// Assume no weights, like MeanAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    x.put([0.0, 1, 2, 3, 4].sliced);
    x.wmean.shouldApprox == 2;
    x.put(5);
    x.wmean.shouldApprox == 2.5;
}

// dynamic array test, assume no weights
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    x.put([0.0, 1, 2, 3, 4]);
    x.wmean.should == 2;
}

// static array test, assume no weights
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    static immutable y = [0.0, 1, 2, 3, 4];
    x.put(y);
    x.wmean.should == 2;
}

// Adding WMeanAccmulators
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25];
    double[] y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    WMeanAccumulator!(float, Summation.pairwise, AssumeWeights.primary) m0;
    m0.put(x);
    WMeanAccumulator!(float, Summation.pairwise, AssumeWeights.primary) m1;
    m1.put(y);
    m0.put(m1);
    m0.wmean.shouldApprox == 29.25 / 12;
}

// repeat test, assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: slicedField;
    import mir.ndslice.topology: iota, map, repeat;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota(6);
    auto w = repeat(1.0, 6).map!(a => a / 6.0).slicedField;
    x.put(y, w);
    x.wmean.shouldApprox == 15.0 / 6;
}

// repeat test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: slicedField;
    import mir.ndslice.topology: iota, repeat;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota(6);
    auto w = repeat(1.0, 6).slicedField;
    x.put(y, w);
    x.wmean.shouldApprox == 15.0 / 6;
}

// range test without shape, assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import std.algorithm: map;
    import std.range: iota;
    import mir.test: shouldApprox;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.sumToOne) x;
    auto y = iota(6);
    auto w = iota(1, 7).map!(a => a / 21.0);
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// range test without shape, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: shouldApprox;
    import std.range: iota;

    WMeanAccumulator!(double, Summation.pairwise, AssumeWeights.primary) x;
    auto y = iota(6);
    auto w = iota(1, 7);
    x.put(y, w);
    x.wmean.shouldApprox == 70.0 / 21;
}

// complex test, do not assume weights sum to 1
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.math.sum: Summation;
    import mir.test: should;
    alias C = Complex!double;

    WMeanAccumulator!(C, Summation.pairwise, AssumeWeights.primary, double) x;
    x.put([C(1, 3), C(2), C(3)]);
    x.wmean.should == C(2, 1);
}

/++
Computes the weighted mean of the input.

Weights are algebraic multipliers and may be signed. By default, the weighted
sum is divided by the total weight, which must be nonzero (checked by assertion).
AssumeWeights.sumToOne skips that division and its nonzero-total check; it is a
caller assumption, not a normalization or validation step. No finite or
nonnegative-weight validation is performed.

Zero-weight observations are still multiplied, so a floating-point NaN or
infinity multiplied by zero propagates NaN. Unlike weighted quantiles,
zero weights do not exclude observations.

By default, if `F` is not floating point type or complex type, then the result
will have a `double` type if `F` is implicitly convertible to a floating point
type or a type for which `isComplex!F` is true.

Params:
    F = controls type of output
    summation = algorithm for calculating sums (default: Summation.appropriate)
    assumeWeights = whether weights are assumed to sum to one (default: AssumeWeights.primary)
    G = controls the type of weights
    weightsSummation = algorithm for summing weights (default: Summation.appropriate)
Returns:
    The weighted mean of all the elements in the input, must be floating point or complex type

See_also:
    $(MATHREF sum, Summation),
    $(SUB2REF univariate, mean),
    $(SUB2REF univariate, meanType)
+/
template wmean(F, Summation summation = Summation.appropriate,
               AssumeWeights assumeWeights = AssumeWeights.primary,
               G = F, Summation weightsSummation = Summation.appropriate)
    if (!is(F : AssumeWeights))
{
    import mir.math.common: fmamath;
    import mir.ndslice.slice: isConvertibleToSlice;
    import mir.stat.descriptive.univariate.internal.types: meanType;
    import std.traits: isIterable;

    /++
    Params:
        s = slice-like
        w = weights
    +/
    @fmamath meanType!F wmean(SliceA, SliceB)(SliceA s, SliceB w)
        if (isConvertibleToSlice!SliceA && isConvertibleToSlice!SliceB)
    {
        import core.lifetime: move;

        alias H = typeof(return);
        WMeanAccumulator!(H, ResolveSummationType!(summation, SliceA, H), assumeWeights,
                          G, ResolveSummationType!(weightsSummation, SliceB, G)) wmean;
        wmean.put(s.move, w.move);
        return wmean.wmean;
    }

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!F wmean(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias H = typeof(return);
        WMeanAccumulator!(H, ResolveSummationType!(summation, Range, H), assumeWeights, G, ResolveSummationType!(weightsSummation, Range, G)) wmean;
        wmean.put(r.move);
        return wmean.wmean;
    }
}

/// ditto
template wmean(Summation summation = Summation.appropriate,
               AssumeWeights assumeWeights = AssumeWeights.primary,
               Summation weightsSummation = Summation.appropriate)
{
    import mir.math.common: fmamath;
    import mir.ndslice.slice: isConvertibleToSlice;
    import mir.stat.descriptive.univariate.internal.types: meanType;
    import std.traits: isIterable;

    /++
    Params:
        s = slice-like
        w = weights
    +/
    @fmamath meanType!SliceA wmean(SliceA, SliceB)(SliceA s, SliceB w)
        if (isConvertibleToSlice!SliceA && isConvertibleToSlice!SliceB)
    {
        import core.lifetime: move;
        import mir.math.sum: sumType;

        alias F = typeof(return);
        return .wmean!(F, summation, assumeWeights, sumType!SliceB, weightsSummation)(s.move, w.move);
    }

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath meanType!Range wmean(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .wmean!(F, summation, assumeWeights, F, weightsSummation)(r.move);
    }
}

/// ditto
template wmean(F, AssumeWeights assumeWeights, Summation summation = Summation.appropriate,
               G = F, Summation weightsSummation = Summation.appropriate)
    if (!is(F : AssumeWeights))
{
    alias wmean = .wmean!(F, summation, assumeWeights, G, weightsSummation);
}

// AssumeWeights has a bool base type. Preserve the original argument type so
// enum calls select the enum overload, while boolean-compatible constants retain
// the legacy string-summation form.
/// ditto
template wmean(F, alias assumeWeights, string summation = "appropriate",
               G = F, string weightsSummation = "appropriate")
    if (!is(F : AssumeWeights) && !is(typeof(assumeWeights) == AssumeWeights) &&
        __traits(compiles, { enum bool value = assumeWeights; }))
{
    mixin("alias wmean = .wmean!(F, Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, G, Summation." ~ weightsSummation ~ ");");
}

// Preserve enum calls that previously used the bool overload with an explicit
// string summation argument. Requiring that argument avoids the short-form clash.
/// ditto
template wmean(F, AssumeWeights assumeWeights, string summation,
               G = F, string weightsSummation = "appropriate")
    if (!is(F : AssumeWeights))
{
    mixin("alias wmean = .wmean!(F, Summation." ~ summation ~ ", assumeWeights, G, Summation." ~ weightsSummation ~ ");");
}

/// ditto
template wmean(bool assumeWeights, string summation = "appropriate",
               string weightsSummation = "appropriate")
{
    mixin("alias wmean = .wmean!(Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, Summation." ~ weightsSummation ~ ");");
}

/// ditto
template wmean(F, string summation, bool assumeWeights = false,
               G = F, string weightsSummation = "appropriate")
    if (!is(F : AssumeWeights))
{
    mixin("alias wmean = .wmean!(F, Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, G, Summation." ~ weightsSummation ~ ");");
}

/// ditto
template wmean(string summation, bool assumeWeights = false,
               string weightsSummation = "appropriate")
{
    mixin("alias wmean = .wmean!(Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, Summation." ~ weightsSummation ~ ");");
}

/// ditto
template wmean(F, string summation, G, string weightsSummation, bool assumeWeights)
    if (!is(F : AssumeWeights))
{
    mixin("alias wmean = .wmean!(F, Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, G, Summation." ~ weightsSummation ~ ");");
}

/// ditto
template wmean(string summation, string weightsSummation, bool assumeWeights = false)
{
    mixin("alias wmean = .wmean!(Summation." ~ summation ~ ", cast(AssumeWeights) assumeWeights, Summation." ~ weightsSummation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice.slice: sliced;
    import mir.test: should, shouldApprox;
    alias C = Complex!double;

    wmean([1.0, 2, 3], [1, 2, 3]).shouldApprox == (1.0 + 4.0 + 9.0) / 6;
    wmean!true([1.0, 2, 3], [1.0 / 6, 2.0 / 6, 3.0 / 6]).shouldApprox == (1.0 + 4.0 + 9.0) / 6;
    // Optimization can change the last rounding bit of this fractional mean.
    auto complexMean = wmean([C(1, 3), C(2), C(3)], [1, 2, 3]);
    complexMean.re.shouldApprox(1e-14, 1e-14) == 14.0 / 6;
    complexMean.im.shouldApprox(1e-14, 1e-14) == 3.0 / 6;

    wmean!float([0, 1, 2, 3, 4, 5].sliced(3, 2), [1, 2, 3, 4, 5, 6].sliced(3, 2)).shouldApprox == 70.0 / 21;

    static assert(is(typeof(wmean!float([1, 2, 3], [1, 2, 3])) == float));
}

// The typed assumption shorthand must not also match the legacy bool overload.
// Exercise both input shapes and preserve explicit output/weight types.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.weighted: legacyMean = wmean;
    import mir.ndslice.slice: sliced;
    enum normalized = true;
    enum one = 1;
    auto values = [2, 8];
    auto weights = [0.25, 0.75];
    static foreach (assumption; [AssumeWeights.primary, AssumeWeights.sumToOne])
    {{
        auto result = wmean!(float, assumption)(values, weights);
        static assert(is(typeof(result) == float));
        assert(result == 6.5f);
        assert(wmean!(float, assumption, Summation.pairwise, double,
            Summation.kbn)(values.sliced, weights.sliced) == result);
        assert(legacyMean!(float, assumption)(values, weights) == result);
    }}
    assert(wmean!(float, AssumeWeights.sumToOne, "appropriate")(values, weights) == 6.5f);
    assert(wmean!(float, AssumeWeights.primary, "pairwise", double, "kbn")(
        values, [1.0, 3.0]) == 6.5f);
    assert(wmean!(float, normalized)(values, weights) == 6.5f);
    assert(wmean!(float, one)(values, weights) == 6.5f);
    assert(wmean!(float, false)(values, [1.0, 3.0]) == 6.5f);
    assert(wmean!(float, true, "pairwise", double, "kbn")(values, weights) == 6.5f);
    assert(wmean!(float, "pairwise", true, double, "kbn")(values, weights) == 6.5f);
    assert(wmean!true(values, weights) == 6.5);
    assert(wmean!(float, Summation.pairwise, AssumeWeights.sumToOne,
        double, Summation.kbn)(values, weights) == 6.5f);

    // This form previously compiled with an explicit summation argument but
    // inferred double from the integer range instead of honoring float.
    auto singleInput = wmean!(float, AssumeWeights.primary, Summation.pairwise)(values);
    static assert(is(typeof(singleInput) == float));
    assert(singleInput == 5.0f);
    assert(wmean!(float, AssumeWeights.primary)(values.sliced) == 5.0f);
    static assert(!__traits(compiles, wmean!(float, 2)(values, weights)));
}

/// Signed weights can place the mean outside the observation range.
/// Zero weights do not filter NaNs; normalized weights can skip division.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.math: isNaN;
    assert(wmean([2.0, 8.0], [-1.0, 2.0]) == 14);
    assert(wmean([double.nan, 8.0], [0.0, 1.0]).isNaN);
    assert(wmean!(double, AssumeWeights.sumToOne)([2.0, 8.0], [0.25, 0.75]) == 6.5);
}

/// If weights are not provided, then behaves like mean
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice.slice: sliced;
    import mir.test: should;
    alias C = Complex!double;

    wmean([1.0, 2, 3]).should == 2;
    wmean([C(1, 3), C(2), C(3)]).should == C(2, 1);

    wmean!float([0, 1, 2, 3, 4, 5].sliced(3, 2)).should == 2.5;

    static assert(is(typeof(wmean!float([1, 2, 3])) == float));
}

/// Weighted mean of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: iota, map;
    import mir.test: shouldApprox;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto w = iota([12], 1);
    auto w_SumToOne = w.map!(a => a / 78.0);

    x.wmean.shouldApprox == 29.25 / 12;
    x.wmean(w).shouldApprox == 203.0 / 78;
    x.wmean!true(w_SumToOne).shouldApprox == 203.0 / 78;
}

/// Weighted mean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: iota, map;
    import mir.test: shouldApprox;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto w = iota([2, 6], 1);
    auto w_SumToOne = w.map!(a => a / 78.0);

    x.wmean.shouldApprox == 29.25 / 12;
    x.wmean(w).shouldApprox == 203.0 / 78;
    x.wmean!true(w_SumToOne).shouldApprox == 203.0 / 78;
}

/// Column mean of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, iota, map, universal;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto w = iota([2], 1).universal;
    auto result = [4.0 / 3, 16.0 / 3, 11.5 / 3, 4.0 / 3, 6.5 / 3, 4.25 / 3];

    // Use byDim or alongDim with map to compute mean of row/column.
    assert(x.byDim!1.map!(a => a.wmean(w)).all!approxEqual(result));
    assert(x.alongDim!0.map!(a => a.wmean(w)).all!approxEqual(result));

    // FIXME
    // Without using map, computes the mean of the whole slice
    // assert(x.byDim!1.wmean(w) == x.sliced.wmean);
    // assert(x.alongDim!0.wmean(w) == x.sliced.wmean);
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat, universal;
    import mir.test: shouldApprox;

    //Set sum algorithm (also for weights) or output type

    auto a = [1, 1e100, 1, -1e100].sliced;

    auto x = a * 10_000;
    auto w1 = [1, 1, 1, 1].sliced;
    auto w2 = [0.25, 0.25, 0.25, 0.25].sliced;

    x.wmean!"kbn"(w1).shouldApprox == 20_000 / 4;
    x.wmean!(true, "kbn")(w2).shouldApprox == 20_000 / 4;
    x.wmean!("kbn", true)(w2).shouldApprox == 20_000 / 4;
    x.wmean!("kbn", true, "pairwise")(w2).shouldApprox == 20_000 / 4;
    x.wmean!(true, "kbn", "pairwise")(w2).shouldApprox == 20_000 / 4;
    x.wmean!"kb2"(w1).shouldApprox == 20_000 / 4;
    x.wmean!"precise"(w1).shouldApprox == 20_000 / 4;
    x.wmean!(double, "precise")(w1).shouldApprox == 20_000.0 / 4;

    auto y = uint.max.repeat(3);
    y.wmean!ulong([1, 1, 1].sliced.universal).shouldApprox == 12884901885 / 3;
}

/++
For integral slices, can pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;
    auto w = [1, 2, 3,  4,  5,  6,
              7, 8, 9, 10, 11, 12].sliced;

    auto y = x.wmean(w);
    y.shouldApprox(1.0e-10) == 204.0 / 78;
    static assert(is(typeof(y) == double));

    x.wmean!float(w).shouldApprox(1.0e-10) == 204f / 78;
}

/++
Mean works for complex numbers and other user-defined types (provided they
can be converted to a floating point or complex type)
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice.slice: sliced;
    import mir.test: should;
    alias C = Complex!double;

    auto x = [C(1.0, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    auto w = [1, 2, 3, 4].sliced;
    x.wmean(w).should == C(3, 4);
}

/// Compute weighted mean tensors along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, as, iota, map, universal;
    /++
      [[0,1,2],
       [3,4,5]]
     +/
    auto x = [
        [0, 1, 2],
        [3, 4, 5]
    ].fuse.as!double;
    auto w = [
        [1, 2, 3],
        [4, 5, 6]
    ].fuse;
    auto w1 = [1, 2].sliced.universal;
    auto w2 = [1, 2, 3].sliced;

    assert(x.wmean(w) == (70.0 / 21));

    auto m0 = [(0.0 + 6.0) / 3, (1.0 + 8.0) / 3, (2.0 + 10.0) / 3];
    assert(x.alongDim!0.map!(a => a.wmean(w1)) == m0);
    assert(x.alongDim!(-2).map!(a => a.wmean(w1)) == m0);

    auto m1 = [(0.0 + 2.0 + 6.0) / 6, (3.0 + 8.0 + 15.0) / 6];
    assert(x.alongDim!1.map!(a => a.wmean(w2)) == m1);
    assert(x.alongDim!(-1).map!(a => a.wmean(w2)) == m1);

    assert(iota(2, 3, 4, 5).as!double.alongDim!0.map!wmean == iota([3, 4, 5], 3 * 4 * 5 / 2));
}

// Empty batches leave weighted accumulators unchanged, including shapes whose
// inner dimension is empty and ranges that do not convert to ndslice.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate.sums: WSummator;
    import std.algorithm: filter;
    import std.meta: AliasSeq;
    double[] empty;
    static foreach (summation; AliasSeq!(Summation.naive, Summation.pairwise))
    {{
        WMeanAccumulator!(double, summation, AssumeWeights.primary) mean;
        WSummator!(double, summation) sum;
        mean.put(empty.sliced, empty.sliced);
        sum.put(empty.sliced, empty.sliced);
        assert(mean.weight == 0 && mean.wsum == 0 && sum.wsum == 0);
        mean.put(4.0, 2.0);
        sum.put(4.0, 2.0);
        mean.put(empty.sliced, empty.sliced);
        mean.put(empty.sliced(2, 0), empty.sliced(2, 0));
        mean.put(empty.filter!(x => true), empty.filter!(x => true));
        sum.put(empty.sliced, empty.sliced);
        sum.put(empty.sliced(2, 0), empty.sliced(2, 0));
        sum.put(empty.filter!(x => true), empty.filter!(x => true));
        assert(mean.weight == 2 && mean.wmean == 4 && mean.wsum == 8);
        assert(sum.wsum == 8);
    }}
}

// Empty-input handling must still diagnose unequal range lengths.
version(mir_stat_test)
@system unittest
{
    import mir.stat.descriptive.univariate.sums: WSummator;
    import std.algorithm: filter;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[] empty;
    double[2] data = [1, 2];
    WMeanAccumulator!(double, Summation.naive, AssumeWeights.primary) mean;
    WSummator!(double, Summation.naive) sum;
    assertThrown!AssertError(mean.put(empty.filter!(x => true), data[].filter!(x => true)));
    assertThrown!AssertError(mean.put(data[].filter!(x => true), empty.filter!(x => true)));
    assertThrown!AssertError(sum.put(empty.filter!(x => true), data[].filter!(x => true)));
    assertThrown!AssertError(sum.put(data[].filter!(x => true), empty.filter!(x => true)));
    assertThrown!AssertError(mean.put(data[].filter!(x => true), data[0 .. 1].filter!(x => true)));
    assertThrown!AssertError(sum.put(data[].filter!(x => true), data[0 .. 1].filter!(x => true)));
}

// test chaining
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;
    [1.0, 2, 3, 4].wmean.should == 2.5;
}

// additional alongDim tests
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.stat.descriptive.univariate.internal.types: meanType;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto w = iota([2], 2);
    auto y = x.alongDim!1.map!(a => a.wmean(w));
    static assert(is(meanType!(typeof(y)) == double));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];
    static immutable w = [1.0, 2, 3,  4,  5,  6,
                            7, 8, 9, 10, 11, 12];

    x.wmean.shouldApprox == 29.25 / 12;
    x.wmean(w).shouldApprox == 203.0 / 78;
}
