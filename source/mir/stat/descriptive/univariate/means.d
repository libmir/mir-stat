/++
Arithmetic, geometric, and harmonic means.

Use the functions to summarize a range or array. The accumulator types
provide the corresponding arithmetic and geometric mean operations through `put`.

Import `mir.stat.descriptive.univariate.means` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Functions) $(TD
        $(REF mean, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF gmean, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF hmean, mir, stat, descriptive, univariate, means)$(NBSP)
    ))
    $(TR $(TD Accumulators) $(TD
        $(REF MeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
        $(REF GMeanAccumulator, mir, stat, descriptive, univariate, means)$(NBSP)
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
        const T x = cast(T) value;
        if (isNaN(x) || x < 0)
            return original;
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

    ///
    F gmean(F = T)() const @property
        if (isFloatingPoint!F)
    {
        import mir.math.common: exp2, floor;

        const F root = nthroot(cast(F) prodAccumulator.mantissa, count);
        const F exponent = cast(F) prodAccumulator.exp / count;
        if (count && (exponent >= F.max_exp || exponent < F.min_exp - 1))
        {
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
