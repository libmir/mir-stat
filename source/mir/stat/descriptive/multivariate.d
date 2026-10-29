/++
This module contains algorithms for multivariate descriptive statistics.

Weighted paired observations use $(LREF weightedCovariance),
$(LREF weightedCorrelation), $(LREF WCovarianceAccumulator), and
$(LREF WCorrelationAccumulator). Pass the two observation inputs first, followed
by their shared weights. WeightedMomentNormalization shares the population,
frequencySample and relativeSample choices with weighted variance; correlation
has no normalization selector because the common factor cancels.

Accumulator merges preserve stored correction terms when the element type and
summation algorithm match and use KBN, KB2, or precise summation. Derived moments
without corresponding stored summation state, mixed-type/mixed-summation merges,
and other summation algorithms retain scalar merging.

Numerical limits:
Unweighted accumulators store unscaled sums and moments in their element type. A finite
mathematical covariance or correlation does not guarantee that these intermediate
values fit. Naive algorithms also subtract raw moments and can lose precision
when observations have a large common offset; prefer `online`, `twoPass`, or
`hybrid` for such data. Requesting a wider result type cannot recover information
already lost during accumulation. Rescale inputs when the stored sums or moments
would exceed the element type's range.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))

+/

module mir.stat.descriptive.multivariate;

import mir.internal.utility: isFloatingPoint;
import mir.math.sum: Summation, Summator;
import std.traits: isMutable, isIterable;


/*
Merge stored summation state when both accumulators use the same compensated
summator. Reading a single rounded sum first can discard its correction.
Centered moments also have a between-group correction; add that separately
only on the state-preserving path so it is not rounded into the source sum.

Other combinations retain the original getter, conversion, and addition.
Some source algorithms expose a derived moment rather than a stored summator,
so field availability is checked at compile time. Merge parameters are copied
by value by the public overloads, keeping self-merges independent.
*/
private void mergeMultivariateSum(string member, string getter, F,
    T, Summation mode, Source, Correction...)(
    ref Summator!(T, mode) target, ref Source source, Correction correction)
    if (Correction.length <= 1)
{
    import std.traits: Unqual;
    static if ((mode == Summation.kbn || mode == Summation.kb2 ||
        mode == Summation.precise) &&
        is(Unqual!(typeof(__traits(getMember, source, member))) == Summator!(T, mode)))
    {
        import mir.stat.internal.summation: mergeSummator;
        mergeSummator(target, __traits(getMember, source, member));
        static if (Correction.length)
            target.put(correction[0]);
    }
    else
    {
        static if (Correction.length)
            target.put(mixin("source." ~ getter ~ "!F") + correction[0]);
        else
            target.put(mixin("source." ~ getter ~ "!F"));
    }
}

// Exercise every stored sum independently: a rounded getter would lose the
// low-order one. Synthetic state isolates merging from update-rounding errors.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    void check(A)()
    {
        enum large = 0x1p54;
        A source;
        source.put(0.0, 0.0);
        static foreach (field; AliasSeq!("summatorLeft", "summatorRight",
            "summatorOfProducts", "summatorOfSquaresLeft", "summatorOfSquaresRight",
            "centeredSummatorOfProducts", "centeredSummatorOfSquaresLeft",
            "centeredSummatorOfSquaresRight"))
        {
            static if (__traits(hasMember, A, field))
            {
                __traits(getMember, source, field).put(large);
                __traits(getMember, source, field).put(1.0);
            }
        }
        A target;
        target.put(source);
        assert(target.count == 1 && source.count == 1);
        A self = source;
        self.put(self);
        assert(self.count == 2);
        static foreach (field; AliasSeq!("summatorLeft", "summatorRight",
            "summatorOfProducts", "summatorOfSquaresLeft", "summatorOfSquaresRight",
            "centeredSummatorOfProducts", "centeredSummatorOfSquaresLeft",
            "centeredSummatorOfSquaresRight"))
        {
            static if (__traits(hasMember, A, field))
            {
                __traits(getMember, target, field).put(-large);
                assert(__traits(getMember, target, field).sum == 1);
                __traits(getMember, source, field).put(-large);
                assert(__traits(getMember, source, field).sum == 1);
                __traits(getMember, self, field).put(-2 * large);
                assert(__traits(getMember, self, field).sum == 2);
            }
        }
    }

    static foreach (mode; AliasSeq!(Summation.kbn, Summation.kb2, Summation.precise))
    {
        static foreach (algo; AliasSeq!(CovarianceAlgo.naive, CovarianceAlgo.online,
            CovarianceAlgo.hybrid, CovarianceAlgo.assumeZeroMean))
            check!(CovarianceAccumulator!(double, algo, mode))();
        static foreach (algo; AliasSeq!(CorrelationAlgo.naive, CorrelationAlgo.online,
            CorrelationAlgo.hybrid, CorrelationAlgo.assumeZeroMean,
            CorrelationAlgo.assumeStandardized))
            check!(CorrelationAccumulator!(double, algo, mode))();
    }
}

// Preserve small product contributions through public observation/merge calls.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (mode; AliasSeq!(Summation.kbn, Summation.kb2, Summation.precise))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.naive, mode),
        CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.naive, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.assumeStandardized, mode)))
    {{
        A source, target;
        source.put(0x1p54, 1);
        source.put(1, 1);
        target.put(source);
        target.put(-0x1p54, 1);
        static if (__traits(hasMember, A, "sumOfProducts"))
            assert(target.sumOfProducts == 1);
        else
            assert(target.centeredSumOfProducts == 1);
    }}
}

// Different algorithms and summation/type combinations keep their merge API.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    void check(Source, Target)()
    {
        float[3] x = [1, 2, 3];
        float[3] y = [2, 4, 6];
        auto source = Source(x[], y[]);
        Target target;
        target.put(source);
        assert(target.count == 3 && target.centeredSumOfProducts == 4);
        assert(target.meanLeft == 2 && target.meanRight == 4);
        target.put(source);
        assert(target.count == 6 && target.centeredSumOfProducts == 8);
        static if (__traits(hasMember, Target, "centeredSumOfSquaresLeft"))
        {
            assert(target.centeredSumOfSquaresLeft == 4);
            assert(target.centeredSumOfSquaresRight == 16);
        }
    }
    static foreach (sourceAlgo; AliasSeq!(CovarianceAlgo.naive,
        CovarianceAlgo.online, CovarianceAlgo.twoPass, CovarianceAlgo.hybrid))
    static foreach (targetAlgo; AliasSeq!(CovarianceAlgo.online, CovarianceAlgo.hybrid))
    {
        check!(CovarianceAccumulator!(double, sourceAlgo, Summation.kbn),
            CovarianceAccumulator!(double, targetAlgo, Summation.kbn))();
        check!(CovarianceAccumulator!(float, sourceAlgo, Summation.pairwise),
            CovarianceAccumulator!(double, targetAlgo, Summation.kbn))();
    }
    static foreach (sourceAlgo; AliasSeq!(CorrelationAlgo.naive,
        CorrelationAlgo.online, CorrelationAlgo.twoPass, CorrelationAlgo.hybrid))
    static foreach (targetAlgo; AliasSeq!(CorrelationAlgo.online, CorrelationAlgo.hybrid))
    {
        check!(CorrelationAccumulator!(double, sourceAlgo, Summation.kbn),
            CorrelationAccumulator!(double, targetAlgo, Summation.kbn))();
        check!(CorrelationAccumulator!(float, sourceAlgo, Summation.kbn),
            CorrelationAccumulator!(double, targetAlgo, Summation.naive))();
    }
}

/*
Form the between-group cross product without discarding a tiny difference
before the other difference can bring the result back into the normal range.
The usual order avoids multiplying two large differences first. When its inner
product is subnormal, scale the larger difference down for weights below one.
For larger weights, multiply the differences first only if that product is
normal; otherwise retain the usual order to avoid rounding a subnormal twice.
This is not a general correctly rounded three-factor product.
*/
private T multivariateMergeProduct(T)(T left, T right, T weight)
{
    version (DigitalMars) pragma(inline, true);
    const T scaledRight = right * weight;
    static if (isFloatingPoint!T)
    {
        import mir.math.common: fabs;
        if (fabs(scaledRight) < T.min_normal && fabs(left) > 1)
        {
            if (weight < 1)
                return (left * weight) * right;
            const T product = left * right;
            if (fabs(product) >= T.min_normal)
                return product * weight;
        }
    }
    return left * scaledRight;
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T tiny = T.min_normal;
        tiny *= T.epsilon;
        // Powers of two give exact references without a special-function call.
        enum T large = T(2) ^^ (T.mant_dig + 2);
        const T expected = tiny * (large / 2);
        foreach (sign; [T(-1), T(1)])
        {
            assert(multivariateMergeProduct(sign * large, tiny, T(0.5)) == sign * expected);
            assert(multivariateMergeProduct(sign * large, 3 * tiny, T(0.5)) == sign * 3 * expected);
            assert(multivariateMergeProduct(sign * large, 3 * tiny, T(1.5)) == sign * 9 * expected);
        }
        // If the unweighted product is also subnormal, keep the usual order.
        assert(multivariateMergeProduct(T(1.5), tiny, T(2.5)) == 3 * tiny);
        assert(multivariateMergeProduct(T(0), tiny, T(0.5)) == 0);
        assert(multivariateMergeProduct(T(2), T(3), T(0.5)) == 3);
    }}
}

version(mir_stat_test)
private void testMultivariateTinyMerge(A)() pure nothrow @nogc
{
    alias T = typeof(A.init.meanLeft);
    T tiny = T.min_normal;
    tiny *= T.epsilon;
    enum T large = T(2) ^^ (T.mant_dig + 2);
    const T expected = tiny * (large / 2);
    foreach (reverse; [false, true])
    foreach (sign; [T(-1), T(1)])
    {
        A target, source;
        target.put(T(0), T(0));
        if (reverse)
            source.put(tiny, sign * large);
        else
            source.put(sign * large, tiny);
        target.put(source);
        assert(target.centeredSumOfProducts == sign * expected);
        assert(source.centeredSumOfProducts == 0);
    }
}

version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (mode; AliasSeq!(Summation.naive, Summation.kbn,
        Summation.kb2, Summation.precise))
    {
        testMultivariateTinyMerge!(CovarianceAccumulator!(double, CovarianceAlgo.online, mode))();
        testMultivariateTinyMerge!(CovarianceAccumulator!(double, CovarianceAlgo.hybrid, mode))();
        testMultivariateTinyMerge!(CorrelationAccumulator!(double, CorrelationAlgo.online, mode))();
        testMultivariateTinyMerge!(CorrelationAccumulator!(double, CorrelationAlgo.hybrid, mode))();
    }
    static foreach (T; AliasSeq!(float, real))
    {
        testMultivariateTinyMerge!(CovarianceAccumulator!(T, CovarianceAlgo.hybrid, Summation.naive))();
        testMultivariateTinyMerge!(CorrelationAccumulator!(T, CorrelationAlgo.hybrid, Summation.naive))();
    }
}

// Convert before multiplying, matching scalar insertion without mapper context.
private template multivariateProduct(T, size_t left = 0, size_t right = 1)
{
    T multivariateProduct(Args...)(Args args)
        if (left < Args.length && right < Args.length)
    {
        return cast(T) args[left] * cast(T) args[right];
    }
}

/*
Cancel the common count factor when naive correlation's scaled moments overflow.
Keep this uncommon path out of line so it does not inhibit ordinary getter
inlining. Raw moments still need to fit, and naive centering still loses precision
for observations tightly clustered around a large mean.
*/
pragma(inline, false)
private F naiveCorrelationFallback(F)(size_t count, F left, F right,
    F product, F squareLeft, F squareRight)
{
    import mir.math.common: sqrt;
    const F inverseCount = F(1) / count;
    const F meanLeft = left * inverseCount;
    const F meanRight = right * inverseCount;
    return (product - meanLeft * right) /
        (sqrt(squareLeft - meanLeft * left) * sqrt(squareRight - meanRight * right));
}

// Repeated self-merges reach the count-product boundary without a huge input.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive)))
    {{
        A a;
        a.put(1, 1);
        foreach (i; 0 .. size_t.sizeof * 4)
            a.put(a);
        assert(a.count == (size_t(1) << (size_t.sizeof * 4)));
        assert(a.covariance(true) == 0);
        assert(a.covariance(false) == 0);
    }}
}

// Slice products use the accumulator precision, just like scalar insertion.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;
    void check(A, V)(V value)
    {
        V[2] values = [value, -value];
        A batch;
        batch.put(values[].sliced, values[].sliced);
        A scalar;
        foreach (v; values)
            scalar.put(v, v);
        assert(batch.centeredSumOfProducts == scalar.centeredSumOfProducts);
        static if (__traits(hasMember, A, "sumOfSquaresLeft"))
        {
            assert(batch.sumOfSquaresLeft == scalar.sumOfSquaresLeft);
            assert(batch.sumOfSquaresRight == scalar.sumOfSquaresRight);
        }
        static if (__traits(hasMember, A, "centeredSumOfSquaresLeft"))
        {
            assert(batch.centeredSumOfSquaresLeft == scalar.centeredSumOfSquaresLeft);
            assert(batch.centeredSumOfSquaresRight == scalar.centeredSumOfSquaresRight);
        }
    }
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.naive, mode),
        CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.naive, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.assumeStandardized, mode)))
    {
        check!A(50000);
        check!A(0x1p70f);
    }
}

// Cancel the common count factor before normalizing finite naive moments.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.math: fabs;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Four times the stored square sum overflows, while the sum itself fits.
        static if (T.max_exp == 128) enum T large = 0x1p62f;
        else static if (T.max_exp == 1024) enum T large = 0x1p510;
        else enum T large = 0x1p8190L;
        T[4] x = [large, -large, large, -large];
        T[4] y = [-large, large, -large, large];
        T[4] small = [1, -1, 1, -1];
        alias A = CorrelationAccumulator!(T, CorrelationAlgo.naive, Summation.naive);
        A a = A(x[].sliced, x[].sliced);
        A b = A(x[].sliced, y[].sliced);
        A c = A(x[].sliced, small[].sliced);
        assert(fabs(a.correlation - 1) <= 4 * T.epsilon);
        assert(fabs(b.correlation + 1) <= 4 * T.epsilon);
        // A finite numerator divided by the overflowing denominator used to give zero.
        assert(fabs(c.correlation - 1) <= 4 * T.epsilon);
    }}
}

// Compute the count weight before scaling the between-group product.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    void check(A)()
    {
        A left, right;
        foreach (i; 0 .. 16)
        {
            left.put(0, 0);
            right.put(0x1p508, 0x1p508);
        }
        left.put(right);
        assert(left.count == 32 && left.centeredSumOfProducts == 0x1p1019);
        A first, second;
        first.put(0, 0);
        second.put(0x1.4p512, 0x1.4p512);
        first.put(second);
        assert(first.centeredSumOfProducts == 0x1.9p1023);
    }
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)))
        check!A();

    // Unequal covariance factors must not overflow when weighted individually.
    alias C = CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive);
    C small, large;
    foreach (i; 0 .. 4) { small.put(0, 0); large.put(0x1p-1022, 0x1p1020); }
    small.put(large);
    assert(small.centeredSumOfProducts == 0.5);
}

// Zero-mean sources have implicit zero sums, but stored centered moments.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    void check(Source, Target)()
    {
        Source source;
        source.put(1, 2);
        source.put(-1, -2);
        Target empty;
        empty.put(source);
        assert(empty.count == 2 && empty.meanLeft == 0 && empty.meanRight == 0);
        assert(empty.centeredSumOfProducts == 4);

        Target populated;
        populated.put(2, 4);
        populated.put(4, 8);
        populated.put(source);
        // Combined observations: (2,4), (4,8), (1,2), (-1,-2).
        assert(populated.count == 4);
        assert(populated.meanLeft == 1.5 && populated.meanRight == 3);
        assert(populated.centeredSumOfProducts == 26);
        assert(source.count == 2 && source.centeredSumOfProducts == 4);
        static if (__traits(hasMember, Target, "centeredSumOfSquaresLeft"))
        {
            assert(empty.centeredSumOfSquaresLeft == 2);
            assert(empty.centeredSumOfSquaresRight == 8);
            assert(populated.centeredSumOfSquaresLeft == 13);
            assert(populated.centeredSumOfSquaresRight == 52);
            assert(source.centeredSumOfSquaresLeft == 2);
            assert(source.centeredSumOfSquaresRight == 8);
        }
    }
    static foreach (mode; AliasSeq!(Summation.kbn, Summation.kb2, Summation.precise))
    {
        check!(CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, mode),
            CovarianceAccumulator!(double, CovarianceAlgo.hybrid, mode))();
        check!(CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, mode),
            CorrelationAccumulator!(double, CorrelationAlgo.hybrid, mode))();
    }
}

// Add the between-group correction without first rounding it into a source sum.
version(mir_stat_test)
pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (mode; AliasSeq!(Summation.kbn, Summation.kb2, Summation.precise))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, mode),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.online, mode),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, mode)))
    {{
        A source, target;
        source.put(0, 0);
        source.put(2, 2);
        target.put(-2, -2);
        target.put(0, 0);
        // Keep the counts and means; isolate cancellation of centered products.
        source.centeredSummatorOfProducts.put(-2);
        source.centeredSummatorOfProducts.put(0x1p54);
        source.centeredSummatorOfProducts.put(1);
        target.centeredSummatorOfProducts.put(-2);
        target.centeredSummatorOfProducts.put(-0x1p54);
        static if (__traits(hasMember, A, "centeredSummatorOfSquaresLeft"))
        {
            source.centeredSummatorOfSquaresLeft.put(-2);
            source.centeredSummatorOfSquaresLeft.put(0x1p54);
            source.centeredSummatorOfSquaresLeft.put(1);
            source.centeredSummatorOfSquaresRight.put(-2);
            source.centeredSummatorOfSquaresRight.put(0x1p54);
            source.centeredSummatorOfSquaresRight.put(1);
        }
        target.put(source);
        // One retained low-order product plus a between-group correction of four.
        assert(target.count == 4 && target.centeredSumOfProducts == 5);
        static if (__traits(hasMember, A, "centeredSummatorOfSquaresLeft"))
        {
            target.centeredSummatorOfSquaresLeft.put(-0x1p54);
            target.centeredSummatorOfSquaresRight.put(-0x1p54);
            // Two from the destination, one from the source, and four between groups.
            assert(target.centeredSumOfSquaresLeft == 7);
            assert(target.centeredSumOfSquaresRight == 7);
        }
    }}
}

// Convert stored sums to the requested result type before dividing by count.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (mode; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kbn))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(float, CovarianceAlgo.naive, mode),
        CovarianceAccumulator!(float, CovarianceAlgo.online, mode),
        CovarianceAccumulator!(float, CovarianceAlgo.twoPass, mode),
        CovarianceAccumulator!(float, CovarianceAlgo.hybrid, mode),
        CorrelationAccumulator!(float, CorrelationAlgo.naive, mode),
        CorrelationAccumulator!(float, CorrelationAlgo.online, mode),
        CorrelationAccumulator!(float, CorrelationAlgo.twoPass, mode),
        CorrelationAccumulator!(float, CorrelationAlgo.hybrid, mode)))
    {{
        float[3] values = [1, 0, 0];
        auto a = A(values[], values[]);
        assert(a.meanLeft!double == 1.0 / 3);
        assert(a.meanRight!double == 1.0 / 3);
        assert(a.meanLeft!real == 1.0L / 3);
        assert(a.meanRight!real == 1.0L / 3);
        // Default access still returns a float rounded at float precision.
        static assert(is(typeof(a.meanRight) == float));
        assert(a.meanLeft == float(1.0 / 3));
        assert(a.meanRight == float(1.0 / 3));
        assert(a.meanRight!double != cast(double) a.meanRight);
    }}
}

// Merging empty partitions must preserve state, including when both sides
// are empty and observations are added afterward.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (summation; AliasSeq!(Summation.naive, Summation.pairwise))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, summation),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, summation),
        CorrelationAccumulator!(double, CorrelationAlgo.online, summation),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, summation)))
    {{
        A a, empty;
        a.put(empty);
        assert(a.count == 0 && a.centeredSumOfProducts == 0);
        a.put(1.0, 2.0);
        a.put(2.0, 4.0);
        a.put(3.0, 6.0);
        a.put(empty);
        assert(a.count == 3 && a.centeredSumOfProducts == 4);
        assert(a.meanLeft == 2 && a.meanRight == 4);
        static if (__traits(hasMember, A, "correlation"))
        {
            assert(a.centeredSumOfSquaresLeft == 2);
            assert(a.centeredSumOfSquaresRight == 8);
            assert(a.correlation > 1 - 1e-14 && a.correlation < 1 + 1e-14);
        }
        A copy;
        copy.put(a);
        assert(copy.count == 3 && copy.centeredSumOfProducts == 4);
    }}
}

// Large means must not overflow an empty destination's zero correction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (summation; AliasSeq!(Summation.naive, Summation.pairwise))
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, summation),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, summation),
        CorrelationAccumulator!(double, CorrelationAlgo.online, summation),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, summation)))
    {{
        A source, target;
        source.put(1e200, -1e200);
        target.put(source);
        assert(target.count == 1);
        assert(target.meanLeft == 1e200 && target.meanRight == -1e200);
        assert(target.centeredSumOfProducts == 0);
        // Continue after the merge and exercise a nonempty destination too.
        target.put(1e200, -1e200);
        target.put(source);
        assert(target.count == 3 && target.centeredSumOfProducts == 0);
        static if (__traits(hasMember, A, "correlation"))
        {
            assert(target.centeredSumOfSquaresLeft == 0);
            assert(target.centeredSumOfSquaresRight == 0);
        }
    }}
}

// Empty arrays, slices, and ranges are valid for every algorithm.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import std.algorithm: filter;

    void check(A)()
    {
        double[3] x = [1, 2, 3];
        double[3] y = [2, 4, 6];
        auto empty = x[0 .. 0];
        auto expected = A(x[], y[]);

        void checkInput(X, Y)(X left, Y right)
        {
            auto a = A(left, right);
            assert(a.count == 0);
            static if (__traits(hasMember, A, "put"))
            {
                a.put(left, right);
                assert(a.count == 0);
                a.put(x[], y[]);
                a.put(left, right);
                assert(a.count == expected.count);
                assert(a.centeredSumOfProducts == expected.centeredSumOfProducts);
                static if (__traits(hasMember, A, "centeredSumOfSquaresLeft"))
                {
                    assert(a.centeredSumOfSquaresLeft == expected.centeredSumOfSquaresLeft);
                    assert(a.centeredSumOfSquaresRight == expected.centeredSumOfSquaresRight);
                }
            }
        }

        checkInput(empty, empty);
        checkInput(empty.sliced, empty.sliced);
        checkInput(empty.filter!(v => true), empty.filter!(v => true));
    }

    static foreach (algo; __traits(allMembers, CovarianceAlgo))
        check!(CovarianceAccumulator!(double, __traits(getMember, CovarianceAlgo, algo), Summation.naive))();
    static foreach (algo; __traits(allMembers, CorrelationAlgo))
        check!(CorrelationAccumulator!(double, __traits(getMember, CorrelationAlgo, algo), Summation.naive))();
}

// Range insertion must reject unequal lengths, including nested batches.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.algorithm: filter;
    import std.meta: AliasSeq;
    import std.exception: assertThrown;

    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)))
    {{
        double[3] x = [1, 2, 3];
        double[3] y = [2, 4, 6];
        // Filters exercise the range overload rather than slice conversion.
        foreach (n; [0, 2])
        {
            A leftShort, rightShort;
            assertThrown!AssertError(leftShort.put(
                x[0 .. n].filter!(v => true), y[].filter!(v => true)));
            assertThrown!AssertError(rightShort.put(
                x[].filter!(v => true), y[0 .. n].filter!(v => true)));
        }
        A equal;
        equal.put(x[].filter!(v => true), y[].filter!(v => true));
        assert(equal.count == 3 && equal.centeredSumOfProducts == 4);

        double[][2] rowsX = [x[0 .. 1], x[1 .. 3]];
        double[][2] rowsY = [y[0 .. 1], y[1 .. 3]];
        foreach (n; [0, 1])
        {
            A leftShort, rightShort;
            assertThrown!AssertError(leftShort.put(
                rowsX[0 .. n].filter!(v => true), rowsY[].filter!(v => true)));
            assertThrown!AssertError(rightShort.put(
                rowsX[].filter!(v => true), rowsY[0 .. n].filter!(v => true)));
        }
        A nested;
        nested.put(rowsX[].filter!(v => true), rowsY[].filter!(v => true));
        assert(nested.count == 3 && nested.centeredSumOfProducts == 4);
    }}
}

// Two-pass batches must save independent cursors; online insertion needs only one pass.
version(mir_stat_test)
unittest
{
    import std.meta: AliasSeq;
    import std.range.primitives: isInputRange, isForwardRange;

    class Cursor(T)
    {
        T[] data;
        this(T[] data) { this.data = data; }
    }
    struct Input(T)
    {
        Cursor!T cursor;
        bool empty() { return cursor.data.length == 0; }
        T front() { return cursor.data[0]; }
        void popFront() { cursor.data = cursor.data[1 .. $]; }
    }
    struct Forward(T, bool shaped = false)
    {
        Input!T input;
        alias input this;
        static if (shaped)
            @property size_t[1] shape() const { return [input.cursor.data.length]; }
        @property Forward save()
        {
            return Forward(Input!T(new Cursor!T(input.cursor.data)));
        }
    }

    static assert(isInputRange!(Input!double));
    static assert(!isForwardRange!(Input!double));
    static assert(isForwardRange!(Forward!double));
    import mir.primitives: hasShape;
    static assert(!hasShape!(Forward!double));
    static assert(hasShape!(Forward!(double, true)));
    double[] x = [1, 2, 3], y = [2, 4, 6];

    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive),
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)))
    {{
        auto left = Input!double(new Cursor!double(x));
        auto right = Input!double(new Cursor!double(y));
        static assert(!__traits(compiles, A(left, right)));
        auto a = A(Forward!double(left), Forward!double(right));
        assert(a.count == 3 && a.centeredSumOfProducts == 4);
        static if (__traits(hasMember, A, "centeredSumOfSquaresLeft"))
        {
            assert(a.centeredSumOfSquaresLeft == 2);
            assert(a.centeredSumOfSquaresRight == 8);
        }
        static if (__traits(hasMember, A, "put"))
            static assert(!__traits(compiles, a.put(left, right)));

        auto shaped = A(
            Forward!(double, true)(Input!double(new Cursor!double(x))),
            Forward!(double, true)(Input!double(new Cursor!double(y))));
        assert(shaped.count == 3 && shaped.centeredSumOfProducts == 4);
    }}

    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive)))
    {{
        auto a = A(Input!double(new Cursor!double(x)), Input!double(new Cursor!double(y)));
        assert(a.count == 3 && a.centeredSumOfProducts == 4);
    }}

    // The outer range of hybrid batches is traversed only once.
    static foreach (A; AliasSeq!(
        CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive),
        CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)))
    {{
        auto left = Input!(double[])(new Cursor!(double[])([x[0 .. 1], x[1 .. $]]));
        auto right = Input!(double[])(new Cursor!(double[])([y[0 .. 1], y[1 .. $]]));
        A a;
        a.put(left, right);
        assert(a.count == 3 && a.centeredSumOfProducts == 4);
    }}
}

private void putter3(Slices, T, U, Summation summation1, Summation summation2, Summation summation3)
    (scope Slices slices, ref Summator!(T, summation1) seed1, ref Summator!(U, summation2) seed2, ref Summator!(U, summation3) seed3)
{
    import mir.functional: Tuple;
    static if (is(Slices == Tuple!(V1, V2, V3), V1, V2, V3)) {
        seed1.put(slices[0]);
        seed2.put(slices[1]);
        seed3.put(slices[2]);
    } else {
        import mir.ndslice.internal: frontOfDim;
        while (!slices.empty)
        {
            frontOfDim!(0, slices)[0].putter3(seed1, seed2, seed3);
            slices.popFront;
        }
    }
}

/++
Covariance algorithms.

See Also:
    $(WEB en.wikipedia.org/wiki/Algorithms_for_calculating_variance, Algorithms for calculating variance).
+/
enum CovarianceAlgo
{
    /++
    Performs Welford's online algorithm for updating covariance. While it only
    iterates each input once, it can be slower for smaller inputs. However, it
    is also more accurate. Can also `put` another CovarianceAccumulator of the
    same type, which uses the parallel algorithm from Chan et al.
    +/
    online,
   
    /++
    Calculates covariance using E(x*y) - E(x)*E(y) (alowing for adjustments for
    population/sample variance). This algorithm can be numerically unstable.
    +/
    naive,

    /++
    Calculates covariance using a two-pass algorithm whereby the inputs are first
    centered and then the sum of products is calculated from that. May be faster
    than `online` and generally more accurate than the `naive` algorithm.
    +/
    twoPass,

    /++
    Calculates covariance assuming the mean of the inputs is zero.
    +/
    assumeZeroMean,

    /++
    When slices, slice-like objects, or ranges are the inputs, uses the two-pass
    algorithm. When an individual data-point is added, uses the online algorithm.
    +/
    hybrid
}

///
struct CovarianceAccumulator(T, CovarianceAlgo covarianceAlgo, Summation summation)
    if (isMutable!T && covarianceAlgo == CovarianceAlgo.naive)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S summatorOfProducts;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX &&
            isInputRange!RangeY)
    {
        import core.lifetime: move;
        this.put(x.move, y.move);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CovarianceAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip, map;

        _count += x.length;
        summatorLeft.put(x);
        summatorRight.put(y);
        summatorOfProducts.put(x.zip(y).map!(multivariateProduct!T));
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        summatorOfProducts.put(x * y);
    }

    ///
    void put(U, Summation sumAlgo)(CovarianceAccumulator!(U, covarianceAlgo, sumAlgo) v)
    {
        _count += v.count;
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", U)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", U)(summatorRight, v);
            mergeMultivariateSum!("summatorOfProducts", "sumOfProducts", U)(summatorOfProducts, v);
        }
        else
        {
            summatorLeft.put(v.sumLeft!U);
            summatorRight.put(v.sumRight!U);
            summatorOfProducts.put(v.sumOfProducts!U);
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F sumOfProducts(F = T)() @property
    {
        return cast(F) summatorOfProducts.sum;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return sumOfProducts!F - sumLeft!F * sumRight!F / count;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
        in (count + isPopulation > 1, "More data points required")
    {
        return sumOfProducts!F / (count + isPopulation - 1) -
            (sumLeft!F * sumRight!F) * (F(1) / (cast(F) count * (count + isPopulation - 1)));
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == 82.25 / 12 - (29.25 * 36) / (12 * 12);
    v.covariance(false).shouldApprox == 82.25 / 11 - (29.25 * 36) / (12 * 12) * (12.0 / 11);

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 94.25 / 13 - (33.25 * 39) / (13 * 13);
    v.covariance(false).shouldApprox == 94.25 / 12 - (33.25 * 39) / (13 * 13) * (13.0 / 12);
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == 82.25 / 12 - (29.25 * 36) / (12 * 12);
    v.covariance(false).shouldApprox == 82.25 / 11 - (29.25 * 36) / (12 * 12) * (12.0 / 11);

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 94.25 / 13 - (33.25 * 39) / (13 * 13);
    v.covariance(false).shouldApprox == 94.25 / 12 - (33.25 * 39) / (13 * 13) * (13.0 / 12);
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v;
    v.put(x, y);
    v.covariance(true).should == 10.0 / 5;
}

///
struct CovarianceAccumulator(T, CovarianceAlgo covarianceAlgo, Summation summation)
    if (isFloatingPoint!T && isMutable!T && covarianceAlgo == CovarianceAlgo.online)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S centeredSummatorOfProducts;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;
        this.put(x.move, y.move);
    }

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CovarianceAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip;

        foreach(e; x.zip(y)) {
            this.put(e[0], e[1]);
        }
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY)
    {
        while (!x.empty || !y.empty)
        {
            assert(!x.empty && !y.empty, "x and y must have the same length");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        T delta = x;
        if (count > 0) {
            delta -= meanLeft;
        }
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(delta * (y - meanRight));
    }

    ///
    void put(U, CovarianceAlgo covAlgo, Summation sumAlgo)(CovarianceAccumulator!(U, covAlgo, sumAlgo) v)
        if (covAlgo != CovarianceAlgo.assumeZeroMean)
    {
        // Empty partitions contribute no observations or centered moments.
        if (v.count == 0)
            return;
        size_t oldCount = count;
        // An empty destination has no between-group correction. Avoid Inf * 0.
        T deltaLeft = 0;
        T deltaRight = 0;
        if (count > 0) {
            deltaLeft = v.meanLeft;
            deltaRight = v.meanRight;
            deltaLeft -= meanLeft!T;
            deltaRight -= meanRight!T;
        }
        _count += v.count;
        // Form the count weight first, as in variance merging, so count factors
        // cannot overflow an otherwise representable between-group correction.
        const T mergeWeight = cast(T) v.count * (cast(T) oldCount / count);
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", T)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", T)(summatorRight, v);
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v, multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
        }
        else
        {
            summatorLeft.put(v.sumLeft!T);
            summatorRight.put(v.sumRight!T);
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T + multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
        in (count + isPopulation > 1, "More data points required")
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive)(4.0, 3.0);
    v.covariance(true).should == 0;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: chunks, iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    CovarianceAccumulator!(double, CovarianceAlgo.online, Summation.naive) v;
    v.put(x, y);
    v.covariance(true).should == 10 / 5;
    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v2;
    v2.put(x.chunks(1), y.chunks(1));
    v2.covariance(true).should == 10 / 5;
}

///
struct CovarianceAccumulator(T, CovarianceAlgo covarianceAlgo, Summation summation)
    if (isMutable!T && covarianceAlgo == CovarianceAlgo.twoPass)
{
    import mir.functional: naryFun;
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;
    import std.range.primitives: isForwardRange, save;
    import mir.stat.descriptive.univariate: MeanAccumulator;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    private S summatorLeft;
    ///
    private S summatorRight;
    ///
    private S centeredSummatorOfProducts;

    ///
    this(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
         Slice!(IteratorX, 1, kindX) x, Slice!(IteratorY, 1, kindY) y)
     in
     {
        assert(x.length == y.length,
               "CovarianceAcumulator.put: both vectors must have the same length");
     }
     do
    {
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: map, vmap, zip;

        _count = x.length;
        summatorLeft.put(x.lightScope);
        summatorRight.put(y.lightScope);
        if (_count == 0)
            return;
        centeredSummatorOfProducts.put(x.vmap(LeftOp!("-", T)(meanLeft)).zip(y.vmap(LeftOp!("-", T)(meanRight))).map!(naryFun!"a * b"));
    }

    ///
    this(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice, y.toSlice);
    }

    /// Generic ranges must be forward ranges: the first pass uses saved cursors to compute means.
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isForwardRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isForwardRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        import mir.primitives: elementCount, hasShape;

        static if (hasShape!RangeX && hasShape!RangeY) {
            assert(x.elementCount == y.elementCount);
            _count += x.elementCount;
            summatorLeft.put(x.save);
            summatorRight.put(y.save);
        } else {
            import std.range: zip;

            foreach(a, b; zip(x.save, y.save)) {
                _count++;
                summatorLeft.put(a);
                summatorRight.put(b);
            }
        }

        if (x.empty && y.empty)
            return;

        T xMean = meanLeft;
        T yMean = meanRight;
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            centeredSummatorOfProducts.put((x.front - xMean) * (y.front - yMean));
            x.popFront;
            y.popFront;
        }
    }

    ///
    this()(T x, T y)
    {
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(0);
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
        in (count + isPopulation > 1, "More data points required")
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox ==3;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check Vmap
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x = a + 1;
    auto y = b - 1;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(4.0, 3.0);
    v.centeredSumOfProducts.should == 0;
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];
    auto y = RCArray!double(12);
    foreach(i, ref e; y)
        e = b[i];

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x, y);
    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto y1 = iota(-3, 2);
    auto v1 = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x1, y1);
    v1.covariance(true).should == 10.0 / 5;

    // this version can't use elementCount
    auto x2 = x1.map!(a => 2 * a);
    auto y2 = y1.map!(a => 2 * a);
    auto v2 = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x2, y2);
    v2.covariance(true).should == 40.0 / 5;
}

///
struct CovarianceAccumulator(T, CovarianceAlgo covarianceAlgo, Summation summation)
    if (isMutable!T && covarianceAlgo == CovarianceAlgo.assumeZeroMean)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice, isConvertibleToSlice, isSlice;
    import mir.primitives: isInputRange, front, empty, popFront;

    private size_t _count;

    ///
    Summator!(T, summation) centeredSummatorOfProducts;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        this.put(x, y);
    }

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CovarianceAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip, map;

        _count += x.length;
        centeredSummatorOfProducts.put(x.zip(y).map!(multivariateProduct!T));
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        _count++;
        centeredSummatorOfProducts.put(x * y);
    }

    ///
    void put(U, Summation sumAlgo)(CovarianceAccumulator!(U, covarianceAlgo, sumAlgo) v)
    {
        _count += v.count;
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v);
        }
        else
        {
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T);
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return 0;
    }
    ///
    F sumRight(F = T)() @property
    {
        return 0;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return 0;
    }
    ///
    F meanRight(F = T)() @property
    {
        return 0;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
        in (count + isPopulation > 1, "More data points required")
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.transform: center;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x = a.center;
    auto y = b.center;

    CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 6.5 / 13;
    v.covariance(false).shouldApprox == 6.5 / 12;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: mean;
    import mir.test: should, shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto aMean = a.mean;
    auto bMean = b.mean;
    auto x = a.dup;
    auto y = b.dup;
    for (size_t i; i < a.length; i++) {
        x[i] -= aMean;
        y[i] -= bMean;
    }

    CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 6.5 / 13;
    v.covariance(false).shouldApprox == 6.5 / 12;
   
    v.meanLeft.should == 0;
    v.meanRight.should == 0;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive)(x.center, y.center);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto b1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto a2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto meanA = (a1.sum + a2.sum) / 12;
    auto meanB = (b1.sum + b2.sum) / 12;
    auto x1 = a1 - meanA;
    auto y1 = b1 - meanB;
    auto x2 = a2 - meanA;
    auto y2 = b2 - meanB;

    CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive)(4.0, 3.0);
    v.centeredSumOfProducts.should == 12;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive)(x, y);
    v.centeredSumOfProducts.should == 0;
}

///
struct CovarianceAccumulator(T, CovarianceAlgo covarianceAlgo, Summation summation)
    if (isFloatingPoint!T && isMutable!T && covarianceAlgo == CovarianceAlgo.hybrid)
{
    import mir.functional: naryFun;
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;
    import std.range.primitives: isForwardRange, save;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S centeredSummatorOfProducts;

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    this(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CovarianceAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: map, vmap, zip;

        _count += x.length;
        summatorLeft.put(x.lightScope);
        summatorRight.put(y.lightScope);
        if (_count == 0)
            return;
        centeredSummatorOfProducts.put(x.vmap(LeftOp!("-", T)(meanLeft)).zip(y.vmap(LeftOp!("-", T)(meanRight))).map!(naryFun!"a * b"));
    }

    ///
    this(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice, y.toSlice);
    }

    /// Generic scalar batches must be forward ranges because their observations are traversed twice.
    /// An outer range of batches may be single-pass; each inner batch must support its own calculation.
   this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY)
    {
        static if (is(elementType!RangeX : T) && is(elementType!RangeY : T)) {
            static assert(isForwardRange!RangeX && isForwardRange!RangeY,
                "Hybrid accumulation requires forward ranges for scalar batches; " ~
                "use the online algorithm for single-pass inputs.");
            import mir.primitives: elementCount, hasShape;

            static if (hasShape!RangeX && hasShape!RangeY) {
                assert(x.elementCount == y.elementCount);
                _count += x.elementCount;
                summatorLeft.put(x.save);
                summatorRight.put(y.save);
            } else {
                import std.range: zip;

                foreach(a, b; zip(x.save, y.save)) {
                    _count++;
                    summatorLeft.put(a);
                    summatorRight.put(b);
                }
            }

            if (x.empty && y.empty)
                return;

            T xMean = meanLeft;
            T yMean = meanRight;
            while (!x.empty || !y.empty)
            {
                assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                       "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
                centeredSummatorOfProducts.put((x.front - xMean) * (y.front - yMean));
                x.popFront;
                y.popFront;
            }
        } else {
            this.put(x, y);
        }
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        static if (is(elementType!RangeX : T) && is(elementType!RangeY : T)) {
            static assert((isConvertibleToSlice!RangeX && isConvertibleToSlice!RangeY) ||
                          (isForwardRange!RangeX && isForwardRange!RangeY),
                "Hybrid accumulation requires forward ranges for scalar batches; " ~
                "use the online algorithm for single-pass inputs.");
            auto v = typeof(this)(x, y);
            this.put(v);
        } else {
            while (!x.empty || !y.empty)
            {
                assert(!x.empty && !y.empty, "x and y must have the same length");
                this.put(x.front, y.front);
                x.popFront;
                y.popFront;
            }
        }
    }

    ///
    void put()(T x, T y)
    {
        T delta = x;
        if (count > 0) {
            delta -= meanLeft;
        }
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(delta * (y - meanRight));
    }

    ///
    void put(U, CovarianceAlgo covAlgo, Summation sumAlgo)(CovarianceAccumulator!(U, covAlgo, sumAlgo) v)
    {
        // Empty partitions contribute no observations or centered moments.
        if (v.count == 0)
            return;
        size_t oldCount = count;
        // An empty destination has no between-group correction. Avoid Inf * 0.
        T deltaLeft = 0;
        T deltaRight = 0;
        if (oldCount > 0) {
            deltaLeft = v.meanLeft!T;
            deltaRight = v.meanRight!T;
            deltaLeft -= meanLeft;
            deltaRight -= meanRight;
        }
        _count += v.count;
        // Form the count weight first, as in variance merging, so count factors
        // cannot overflow an otherwise representable between-group correction.
        const T mergeWeight = cast(T) v.count * (cast(T) oldCount / count);
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", T)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", T)(summatorRight, v);
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v, multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
        }
        else
        {
            summatorLeft.put(v.sumLeft!T);
            summatorRight.put(v.sumRight!T);
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T + multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
        in (count + isPopulation > 1, "More data points required")
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    CovarianceAccumulator!(double, CovarianceAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CovarianceAccumulator!(double, CovarianceAlgo.twoPass, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
}

// Check adding CovarianceAccumultors (assumeZeroMean)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    auto a1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto b1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto a2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x1 = a1.center;
    auto y1 = b1.center;
    auto x2 = a2.center;
    auto y2 = b2.center;

    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CovarianceAccumulator!(double, CovarianceAlgo.assumeZeroMean, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -1.9375 / 12; //note: different from above due to inconsistent centering
    v1.covariance(false).shouldApprox == -1.9375 / 11; //note: different from above due to inconsistent centering
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive)(4.0, 3.0);
    v.covariance(true).should == 0;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.algorithm: map;
    import std.range: chunks, iota;

    auto x1 = iota(0, 5);
    auto y1 = iota(-3, 2);
    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    v1.covariance(true).should == 10.0 / 5;
    // this version can't use elementCount
    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v2;
    auto x2 = x1.map!(a => 2 * a);
    auto y2 = y1.map!(a => 2 * a);
    v2.put(x2, y2);
    v2.covariance(true).should == 40.0 / 5;
    CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive) v3;
    v3.put(x1.chunks(1), y1.chunks(1));
    v3.covariance(true).should == 10.0 / 5;
    auto v4 = CovarianceAccumulator!(double, CovarianceAlgo.hybrid, Summation.naive)(x1.chunks(1), y1.chunks(1));
    v4.covariance(true).should == 10.0 / 5;
}

/++
Calculates the covariance of the inputs.

If `x` and `y` are both slices or convertible to slices, then they must be
one-dimensional.

For generic ranges of scalar observations, the twoPass and hybrid algorithms
require forward ranges because they traverse the observations twice. The default
algorithm is hybrid. Use the online algorithm for single-pass sources such as
file or stream ranges. Arrays and slices continue to use their dedicated paths.

By default, if `F` is not floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.

For inputs already centered or standardized, use assumeZeroMean to skip
centering. Both sample and population z-scores satisfy this assumption; no
input standardization option is needed. isPopulation selects the output
normalization independently: false divides the sum of products by n - 1,
true by n. The result is covariance in the supplied coordinates, not the
original unscaled data. Inputs must have mean zero over the complete dataset;
this assumption is not validated.

Params:
    F = controls type of output
    covarianceAlgo = algorithm for calculating covariance (default: CovarianceAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    The covariance of the inputs
+/
template covariance(F,
                    CovarianceAlgo covarianceAlgo = CovarianceAlgo.hybrid,
                    Summation summation = Summation.appropriate)
    if (isFloatingPoint!F)
{
    import mir.math.common: fmamath;
    import mir.primitives: isInputRange;
    import mir.math.sum: ResolveSummationType;
    /++
    Params:
        x = range, must be finite iterable
        y = range, must be finite iterable
        isPopulation = true if population covariance, false if sample covariance (default)
    +/
    @fmamath F covariance(RangeX, RangeY)(RangeX x, RangeY y, bool isPopulation = false)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;

        auto covarianceAccumulator = CovarianceAccumulator!(F, covarianceAlgo, ResolveSummationType!(summation, RangeX, F))(x.move, y.move);
        return covarianceAccumulator.covariance(isPopulation);
    }
}

/// ditto
template covariance(
    CovarianceAlgo covarianceAlgo = CovarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    import mir.math.common: fmamath;
    import mir.primitives: isInputRange;
    import mir.stat.descriptive.univariate: meanType;
    import std.traits: CommonType;
    /++
    Params:
        x = range, must be finite iterable
        y = range, must be finite iterable
        isPopulation = true if population covariance, false if sample covariance (default)
    +/
    @fmamath CommonType!(meanType!RangeX, meanType!RangeY) covariance(RangeX, RangeY)(RangeX x, RangeY y, bool isPopulation = false)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .covariance!(F, covarianceAlgo, summation)(x.move, y.move, isPopulation);
    }
}

/// ditto
template covariance(F, string covarianceAlgo, string summation = "appropriate")
{
    mixin("alias covariance = .covariance!(F, CovarianceAlgo." ~ covarianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/// ditto
template covariance(string covarianceAlgo, string summation = "appropriate")
{
    mixin("alias covariance = .covariance!(CovarianceAlgo." ~ covarianceAlgo ~ ", Summation." ~ summation ~ ");");
}

/++
Reuse standardized columns when calculating covariance, without centering them
again. Matching covariance normalization to the z-score convention gives the
correlation; choosing the other normalization changes the result's scale.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.transform: zscore;
    import mir.math.common: approxEqual;
    double[3] x = [10, 20, 30];
    double[3] y = [20, 30, 10];

    // Default z-scores use sample standard deviations.
    auto sx = x[].zscore;
    auto sy = y[].zscore;
    assert(covariance!"assumeZeroMean"(sx, sy).approxEqual(-0.5));
    assert(covariance!"assumeZeroMean"(sx, sy, true).approxEqual(-1.0 / 3));

    // Population z-scores use the same covariance API.
    auto px = x[].zscore(true);
    auto py = y[].zscore(true);
    assert(covariance!"assumeZeroMean"(px, py, true).approxEqual(-0.5));
    assert(covariance!"assumeZeroMean"(px, py).approxEqual(-0.75));
}

// Free covariance agrees with standardized correlation accumulation for either
// input convention, independently of the requested covariance normalization.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.stat.transform: zscore;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (method; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kahan))
    static foreach (convention; AliasSeq!(InputStandardization.sample, InputStandardization.population))
    {{
        T[3] x = [T(10), 20, 30];
        T[3] y = [T(20), 30, 10];
        enum populationInput = convention == InputStandardization.population;
        auto zx = x[].zscore(populationInput);
        auto zy = y[].zscore(populationInput);
        alias A = CorrelationAccumulator!(T, CorrelationAlgo.assumeStandardized, method, convention);
        const accumulator = A(zx, zy);
        foreach (populationOutput; [false, true])
        {
            T expected = populationInput
                ? (populationOutput ? T(-0.5) : T(-0.75))
                : (populationOutput ? T(-1) / 3 : T(-0.5));
            auto result = covariance!(CovarianceAlgo.assumeZeroMean, method)(zx, zy, populationOutput);
            assert(result.approxEqual(expected));
            assert(result.approxEqual(accumulator.covariance(populationOutput)));
            assert(covariance!(T, CovarianceAlgo.assumeZeroMean, method)(zx, zy, populationOutput)
                .approxEqual(expected));
            assert(covariance!("assumeZeroMean", "pairwise")(zx, zy, populationOutput)
                .approxEqual(expected));
            assert(covariance!(T, "assumeZeroMean", "pairwise")(zx, zy, populationOutput)
                .approxEqual(expected));
        }
        assert(covariance!(CovarianceAlgo.assumeZeroMean, method)(zx, zy, populationInput)
            .approxEqual(accumulator.correlation));
    }}
}

/// Covariance of vectors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    x.covariance(y, true).shouldApprox == -5.5 / 12;
    x.covariance(y).shouldApprox == -5.5 / 11;
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    auto x = a + 10.0 ^^ 9;
    auto y = b + 10.0 ^^ 9;

    x.covariance(y).shouldApprox == -5.5 / 11;

    // The naive algorithm is numerically unstable in this case
    assert(!x.covariance!"naive"(y).approxEqual(-5.5 / 11));

    // The two-pass algorithm provides the same answer as hybrid
    x.covariance!"twoPass"(y).shouldApprox == -5.5 / 11;

    // And the assumeZeroMean algorithm is way off
    assert(!x.covariance!"assumeZeroMean"(y).approxEqual(-5.5 / 11));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;
    import mir.test: shouldApprox;

    //Set population covariance, covariance algorithm, sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto b = [1.0e100, 1, 1, -1e100].sliced;
    auto x = a * 10_000;
    auto y = b * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect (for `y` the same is true
    for the first and fourth). Further, after centering and multiplying `x` and
    `y`, the third numbers in the slice has precision too low to be included in
    the centered sum of the products.
    +/
    x.covariance(y).shouldApprox == 1.0e208 / 3;
    x.covariance(y, true).shouldApprox == 1.0e208 / 4;

    x.covariance!("online")(y).shouldApprox == 1.0e208 / 3;
    x.covariance!("online", "kbn")(y).shouldApprox == 1.0e208 / 3;
    x.covariance!("online", "kb2")(y).shouldApprox == 1.0e208 / 3;
    x.covariance!("online", "precise")(y).shouldApprox == 1.0e208 / 3;
    x.covariance!(double, "online", "precise")(y).shouldApprox == 1.0e208 / 3;

    auto z = uint.max.repeat(3);
    z.covariance!float(z).shouldApprox == 0.0;
    static assert(is(typeof(z.covariance!float(z)) == float));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;
    auto y = [6, 3, 7, 1, 1, 1,
              9, 5, 3, 1, 3, 7].sliced;

    x.covariance(y).shouldApprox == -18.583333 / 11;
    static assert(is(typeof(x.covariance(y)) == double));

    x.covariance!float(y).shouldApprox == -18.583333 / 11;
    static assert(is(typeof(x.covariance!float(y)) == float));
}

// make sure works with dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: shouldApprox;

    double[] x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                    2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    double[] y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                   9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    x.covariance(y).shouldApprox == -5.5 / 11;
}

/// Works with @nogc
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;

    x.covariance(y, true).shouldApprox == -5.5 / 12;
    x.covariance(y).shouldApprox == -5.5 / 11;
}


/++
Correlation algorithms.

See Also:
    $(LREF CovarianceAlgo)
    $(WEB en.wikipedia.org/wiki/Algorithms_for_calculating_variance, Algorithms for calculating variance).
+/
enum CorrelationAlgo
{
    /++
    Performs Welford's online algorithm for updating correlation. While it only
    iterates each input once, it can be slower for smaller inputs. However, it
    is also more accurate. Can also `put` another CorrelationAccumulator of the
    same type, which uses the parallel algorithm from Chan et al.
    +/
    online,

    /++
    Calculates correlation using (E(x*y) - E(x)*E(y))/(sqrt(E(x^2)-E(x)^2)*sqrt(E(y^2)-E(y)^2)) (alowing for adjustments for
    population/sample variance). This algorithm can be numerically unstable.
    +/
    naive,

    /++
    Calculates correlation using a two-pass algorithm whereby the inputs are first
    centered and then the sum of products is calculated from that. May be faster
    than `online` and generally more accurate than the `naive` algorithm.
    +/
    twoPass,

    /++
    Calculates correlation assuming the mean of the inputs is zero.
    +/
    assumeZeroMean,

    /++
    Calculates correlation from inputs already standardized to mean zero and
    standard deviation one. InputStandardization selects sample (default) or
    population standardization. No means or variances are calculated.
    +/
    assumeStandardized,

    /++
    When slices, slice-like objects, or ranges are the inputs, uses the two-pass
    algorithm. When an individual data-point is added, uses the online algorithm.
    +/
    hybrid
}

/++
Convention used to standardize inputs to CorrelationAlgo.assumeStandardized.
This describes the input data, not a choice of sample or population correlation.
+/
enum InputStandardization
{
    /// Squared standardized values sum to n - 1, as with zscore's default.
    sample,
    /// Squared standardized values sum to n, as with zscore(true).
    population,
}

/++
Accumulate correlation when input standardization has already been performed,
for example when reusing standardized columns in several pairwise comparisons.
Only the observation count and sum of products are stored. Correlation divides
the sum of products by n - 1 for sample-standardized inputs or by n for
population-standardized inputs. There is no centering, variance calculation,
square root, or validation that the inputs are standardized.

Both inputs must use the same convention and be standardized over the complete
dataset being accumulated. Chunks of that dataset may be inserted or merged;
independently standardizing each chunk does not satisfy this contract. Read
correlation and inferred moments only after all chunks have been accumulated.
At least two observations with nonzero original variance are required.

covariance(isPopulation) describes the standardized input coordinates, not the
original unscaled data. Its output normalization is independent of the input
standardization convention.

Params:
    T = stored product type
    correlationAlgo = CorrelationAlgo.assumeStandardized
    summation = summation algorithm
    inputStandardization = input z-score convention, sample by default
+/
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation,
    InputStandardization inputStandardization = InputStandardization.sample)
    if (isMutable!T && correlationAlgo == CorrelationAlgo.assumeStandardized &&
        (inputStandardization == InputStandardization.sample ||
         inputStandardization == InputStandardization.population))
{
    import mir.math.sum: elementType;
    import mir.ndslice.slice: Slice, SliceKind, isConvertibleToSlice, isSlice;
    import mir.primitives: isInputRange, front, empty, popFront;

    private size_t _count;
    /// Product summation state.
    Summator!(T, summation) centeredSummatorOfProducts;

    /// Construct from paired ranges.
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        put(x, y);
    }

    /// Construct from one pair; add the remaining standardized observations before reading correlation.
    this()(T x, T y) { put(x, y); }

    /// Add paired slices, preserving their layouts.
    void put(IX, IY, SliceKind kx, SliceKind ky)(Slice!(IX, 1, kx) x, Slice!(IY, 1, ky) y)
    {
        import mir.ndslice.topology: zip, map;
        assert(x.length == y.length, "CorrelationAccumulator.put: lengths must match");
        _count += x.length;
        centeredSummatorOfProducts.put(x.zip(y).map!(multivariateProduct!T));
    }

    /// Add built-in arrays or other slice-convertible inputs.
    void put(X, Y)(X x, Y y)
        if (isConvertibleToSlice!X && !isSlice!X && isConvertibleToSlice!Y && !isSlice!Y)
    {
        import mir.ndslice.slice: toSlice;
        put(x.toSlice, y.toSlice);
    }

    /// Add paired input ranges; empty ranges add no observations.
    void put(X, Y)(X x, Y y)
        if (isInputRange!X && !isConvertibleToSlice!X && is(elementType!X : T) &&
            isInputRange!Y && !isConvertibleToSlice!Y && is(elementType!Y : T))
    {
        while (!x.empty && !y.empty)
        {
            put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
        assert(x.empty && y.empty, "CorrelationAccumulator.put: lengths must match");
    }

    /// Add one standardized observation pair.
    void put()(T x, T y)
    {
        ++_count;
        centeredSummatorOfProducts.put(x * y);
    }

    /// Merge chunks standardized together, using the same input convention.
    void put(U, Summation otherSummation)(
        const CorrelationAccumulator!(U, correlationAlgo, otherSummation, inputStandardization) other)
    {
        _count += other.count;
        static if (is(T == U) && summation == otherSummation &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, other);
        }
        else
        {
            centeredSummatorOfProducts.put(other.centeredSumOfProducts!T);
        }
    }

const:
    /// Number of accumulated pairs.
    size_t count() @property { return _count; }
    /// Zero by the input standardization assumption.
    F sumLeft(F = T)() @property { return 0; }
    /// ditto
    F sumRight(F = T)() @property { return 0; }
    /// ditto
    F meanLeft(F = T)() @property { return 0; }
    /// ditto
    F meanRight(F = T)() @property { return 0; }
    /// Accumulated sum of products.
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    /// Inferred from the count and standardization convention, not measured.
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        // Adjust the integer count before converting. DMD 2.113 with -O -inline
        // can miscompile the conditional floating-point result for empty state.
        // https://github.com/dlang/dmd/issues/23918
        auto n = count;
        static if (inputStandardization == InputStandardization.sample)
            if (n)
                --n;
        return cast(F) n;
    }
    /// ditto
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return centeredSumOfSquaresLeft!F;
    }
    /++
    Covariance of the standardized coordinates.
    Params:
        isPopulation = true divides by n, false divides by n - 1
    +/
    F covariance(F = T)(bool isPopulation) @property
    {
        assert(count > (isPopulation ? 0 : 1), "Insufficient observations for covariance");
        return centeredSumOfProducts!F / (isPopulation ? count : count - 1);
    }
    /// Pearson correlation using the input standardization convention.
    F correlation(F = T)() @property
    {
        assert(count > 1, "Standardized correlation requires at least two observations");
        return centeredSumOfProducts!F / centeredSumOfSquaresLeft!F;
    }
}

/++
Reuse columns that have already been standardized, avoiding repeated centering
and variance calculations when computing their correlation. zscore defaults to
sample standardization, matching assumeStandardized's default convention.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.transform: zscore;
    import mir.math.common: approxEqual;
    import mir.math.sum: Summation;
    double[3] x = [10, 20, 30];
    double[3] y = [20, 30, 10];
    auto standardizedX = x[].zscore;
    auto standardizedY = y[].zscore;
    alias A = CorrelationAccumulator!(double, CorrelationAlgo.assumeStandardized, Summation.pairwise);
    A first, second;
    // These are chunks of the same standardized columns, not separately scaled chunks.
    first.put(standardizedX[0 .. 1], standardizedY[0 .. 1]);
    second.put(standardizedX[1 .. $], standardizedY[1 .. $]);
    first.put(second);
    assert(first.count == 3 && first.correlation.approxEqual(-0.5));
    assert(first.covariance(false).approxEqual(-0.5));
    assert(first.covariance(true).approxEqual(-1.0 / 3));
}

/++
For population z-scores, select population input standardization explicitly.
The correlation is unchanged; the sum of products has a different scale.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.transform: zscore;
    import mir.math.common: approxEqual;
    import mir.math.sum: Summation;
    double[3] x = [10, 20, 30];
    double[3] y = [20, 30, 10];
    auto zx = x[].zscore(true);
    auto zy = y[].zscore(true);
    alias A = CorrelationAccumulator!(double, CorrelationAlgo.assumeStandardized,
        Summation.pairwise, InputStandardization.population);
    const accumulator = A(zx, zy);
    assert(accumulator.correlation.approxEqual(-0.5));
    assert(accumulator.covariance(true).approxEqual(-0.5));
    assert(accumulator.covariance(false).approxEqual(-0.75));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.algorithm: map;
    import std.range: iota;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (method; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kahan))
    {{
        alias A = CorrelationAccumulator!(T, CorrelationAlgo.assumeStandardized, method);
        T[3] x = [T(-1), 0, 1];
        T[3] y = [T(0), 1, -1];
        const arrayResult = A(x[], y[]);
        assert(arrayResult.count == 3 && arrayResult.correlation == T(-0.5));
        assert(arrayResult.centeredSumOfProducts == -1);
        assert(arrayResult.centeredSumOfSquaresLeft == 2);
        assert(arrayResult.centeredSumOfSquaresRight == 2);
        assert(arrayResult.sumLeft == 0 && arrayResult.sumRight == 0);
        assert(arrayResult.meanLeft == 0 && arrayResult.meanRight == 0);
        assert(arrayResult.correlation!double == -0.5);

        A scalar = A(x[0], y[0]);
        assert(scalar.centeredSumOfSquaresLeft == 0);
        assert(scalar.centeredSumOfSquaresRight == 0);
        scalar.put(x[1], y[1]);
        scalar.put(x[2], y[2]);
        assert(scalar.correlation == T(-0.5));
        A ranges;
        ranges.put(x[0 .. 0], y[0 .. 0]);
        ranges.put(iota(0, 0).map!(i => T(i - 1)), iota(0, 0).map!(i => T(i == 2 ? -1 : i)));
        assert(ranges.count == 0 && ranges.centeredSumOfProducts == 0);
        assert(ranges.centeredSumOfSquaresLeft == 0);
        assert(ranges.centeredSumOfSquaresRight == 0);
        assert(ranges.centeredSumOfSquaresLeft!double == 0);
        ranges.put(iota(0, 3).map!(i => T(i - 1)), iota(0, 3).map!(i => T(i == 2 ? -1 : i)));
        assert(ranges.correlation == T(-0.5));
        T[6] bx = [T(-1), 99, 0, 99, 1, 99];
        T[6] by = [T(0), 99, 1, 99, -1, 99];
        auto stridedResult = A(bx[].sliced.stride(2), by[].sliced.stride(2));
        assert(stridedResult.correlation == T(-0.5));
        A merged = A(x[0 .. 1], y[0 .. 1]);
        const other = CorrelationAccumulator!(T, CorrelationAlgo.assumeStandardized,
            Summation.naive)(x[1 .. $], y[1 .. $]);
        merged.put(other);
        assert(merged.count == 3 && merged.correlation == T(-0.5));
        alias P = CorrelationAccumulator!(T, CorrelationAlgo.assumeStandardized, method,
            InputStandardization.population);
        static assert(!__traits(compiles, scalar.put(P.init)));
        P emptyPopulation;
        assert(emptyPopulation.centeredSumOfSquaresLeft == 0);
        assert(emptyPopulation.centeredSumOfSquaresRight == 0);
        const singlePopulation = P(x[0], y[0]);
        assert(singlePopulation.centeredSumOfSquaresLeft == 1);
        assert(singlePopulation.centeredSumOfSquaresRight == 1);
        // A different assumption changes the divisor; it does not normalize data.
        const population = P(x[], y[]);
        assert(population.centeredSumOfSquaresLeft == 3);
        assert(population.correlation.approxEqual(T(-1) / 3));
    }}
}

// Invalid lengths and insufficient observations must be diagnosed.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    import std.range: iota;
    import std.algorithm: map;
    alias A = CorrelationAccumulator!(double, CorrelationAlgo.assumeStandardized, Summation.naive);
    A a;
    void rejects(scope void delegate() pure nothrow @nogc operation)
    {
        bool rejected;
        try { operation(); } catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    rejects(() { auto r = a.correlation; });
    rejects(() { auto r = a.covariance(true); });
    a.put(0, 0);
    rejects(() { auto r = a.correlation; });
    rejects(() { auto r = a.covariance(false); });
    double[2] x;
    rejects(() { a.put(x[], x[0 .. 1]); });
    rejects(() { a.put(iota(0, 2).map!(i => double(i)), iota(0, 1).map!(i => double(i))); });
}

///
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation)
    if (isMutable!T && correlationAlgo == CorrelationAlgo.naive)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S summatorOfProducts;
    ///
    S summatorOfSquaresLeft;
    ///
    S summatorOfSquaresRight;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX &&
            isInputRange!RangeY)
    {
        import core.lifetime: move;
        this.put(x.move, y.move);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CorrelationAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip, map;

        _count += x.length;
        summatorLeft.put(x);
        summatorRight.put(y);
        summatorOfProducts.put(x.zip(y).map!(multivariateProduct!T));
        summatorOfSquaresLeft.put(x.map!(multivariateProduct!(T, 0, 0)));
        summatorOfSquaresRight.put(y.map!(multivariateProduct!(T, 0, 0)));
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        summatorOfProducts.put(x * y);
        summatorOfSquaresLeft.put(x * x);
        summatorOfSquaresRight.put(y * y);
    }

    ///
    void put(U, Summation sumAlgo)(CorrelationAccumulator!(U, correlationAlgo, sumAlgo) v)
    {
        _count += v.count;
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", U)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", U)(summatorRight, v);
            mergeMultivariateSum!("summatorOfProducts", "sumOfProducts", U)(summatorOfProducts, v);
            mergeMultivariateSum!("summatorOfSquaresLeft", "sumOfSquaresLeft", U)(summatorOfSquaresLeft, v);
            mergeMultivariateSum!("summatorOfSquaresRight", "sumOfSquaresRight", U)(summatorOfSquaresRight, v);
        }
        else
        {
            summatorLeft.put(v.sumLeft!U);
            summatorRight.put(v.sumRight!U);
            summatorOfProducts.put(v.sumOfProducts!U);
            summatorOfSquaresLeft.put(v.sumOfSquaresLeft!U);
            summatorOfSquaresRight.put(v.sumOfSquaresRight!U);
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F sumOfProducts(F = T)() @property
    {
        return cast(F) summatorOfProducts.sum;
    }
    ///
    F sumOfSquaresLeft(F = T)() @property
    {
        return cast(F) summatorOfSquaresLeft.sum;
    }
    ///
    F sumOfSquaresRight(F = T)() @property
    {
        return cast(F) summatorOfSquaresRight.sum;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return sumOfProducts!F - sumLeft!F * sumRight!F / count;
    }
    ///
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        return sumOfSquaresLeft!F - count * meanLeft!F * meanLeft!F;
    }
    ///
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return sumOfSquaresRight!F - count * meanRight!F * meanRight!F;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
    {
        return sumOfProducts!F / (count + isPopulation - 1) -
            (sumLeft!F * sumRight!F) * (F(1) / (cast(F) count * (count + isPopulation - 1)));
    }
    ///
    F correlation(F = T)() @property
        in (centeredSumOfSquaresLeft > 0, "`x` must have centered sum of squares larger than zero")
        in (centeredSumOfSquaresRight > 0, "`y` must have centered sum of squares larger than zero")
    {
        import mir.math.common: sqrt;
        // DMD otherwise stops inlining the ordinary path after adding the fallback.
        version (DigitalMars)
            pragma(inline, true);
        const F numerator = count * sumOfProducts!F - sumLeft!F * sumRight!F;
        const F denominator = sqrt(count * sumOfSquaresLeft!F - sumLeft!F * sumLeft!F) *
            sqrt(count * sumOfSquaresRight!F - sumRight!F * sumRight!F);
        const F result = numerator / denominator;
        static if (isFloatingPoint!F)
        {
            if (!(denominator <= F.max && result == result))
                return naiveCorrelationFallback(count, sumLeft!F, sumRight!F,
                    sumOfProducts!F, sumOfSquaresLeft!F, sumOfSquaresRight!F);
        }
        return result;
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == 82.25 / 12 - (29.25 * 36) / (12 * 12);
    v.covariance(false).shouldApprox == 82.25 / 11 - (29.25 * 36) / (12 * 12) * (12.0 / 11);

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 94.25 / 13 - (33.25 * 39) / (13 * 13);
    v.covariance(false).shouldApprox == 94.25 / 12 - (33.25 * 39) / (13 * 13) * (13.0 / 12);

    v.correlation.shouldApprox == -0.0611234;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == 82.25 / 12 - (29.25 * 36) / (12 * 12);
    v.covariance(false).shouldApprox == 82.25 / 11 - (29.25 * 36) / (12 * 12) * (12.0 / 11);

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);

    v.covariance(true).shouldApprox == 94.25 / 13 - (33.25 * 39) / (13 * 13);
    v.covariance(false).shouldApprox == 94.25 / 12 - (33.25 * 39) / (13 * 13) * (13.0 / 12);

    v.correlation.shouldApprox == -0.0611234;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
    v.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
    v1.correlation.shouldApprox == -0.0623684;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should, shouldApprox;
    import std.range: iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v;
    v.put(x, y);
    v.covariance(true).should == 10.0 / 5;
    v.correlation.shouldApprox == 1;
}

///
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation)
    if (isFloatingPoint!T && isMutable!T && correlationAlgo == CorrelationAlgo.online)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S centeredSummatorOfProducts;
    ///
    S centeredSummatorOfSquaresLeft;
    ///
    S centeredSummatorOfSquaresRight;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;
        this.put(x.move, y.move);
    }

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CorrelationAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip;

        foreach(e; x.zip(y)) {
            this.put(e[0], e[1]);
        }
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY)
    {
        while (!x.empty || !y.empty)
        {
            assert(!x.empty && !y.empty, "x and y must have the same length");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        T deltaX = x;
        T deltaY = y;
        if (count > 0) {
            deltaX -= meanLeft;
            deltaY -= meanRight;
        }
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(deltaX * (y - meanRight));
        centeredSummatorOfSquaresLeft.put(deltaX * (x - meanLeft));
        centeredSummatorOfSquaresRight.put(deltaY * (y - meanRight));
    }

    ///
    void put(U, CorrelationAlgo covAlgo, Summation sumAlgo)(CorrelationAccumulator!(U, covAlgo, sumAlgo) v)
        if (!is(covAlgo == CorrelationAlgo.assumeZeroMean))
    {
        // Empty partitions contribute no observations or centered moments.
        if (v.count == 0)
            return;
        size_t oldCount = count;
        // An empty destination has no between-group correction. Avoid Inf * 0.
        T deltaLeft = 0;
        T deltaRight = 0;
        if (count > 0) {
            deltaLeft = v.meanLeft;
            deltaRight = v.meanRight;
            deltaLeft -= meanLeft!T;
            deltaRight -= meanRight!T;
        }
        _count += v.count;
        // Form the count weight first, as in variance merging, so count factors
        // cannot overflow an otherwise representable between-group correction.
        const T mergeWeight = cast(T) v.count * (cast(T) oldCount / count);
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", T)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", T)(summatorRight, v);
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v, multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
            mergeMultivariateSum!("centeredSummatorOfSquaresLeft", "centeredSumOfSquaresLeft", T)(centeredSummatorOfSquaresLeft, v, deltaLeft * (deltaLeft * mergeWeight));
            mergeMultivariateSum!("centeredSummatorOfSquaresRight", "centeredSumOfSquaresRight", T)(centeredSummatorOfSquaresRight, v, deltaRight * (deltaRight * mergeWeight));
        }
        else
        {
            summatorLeft.put(v.sumLeft!T);
            summatorRight.put(v.sumRight!T);
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T + multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
            centeredSummatorOfSquaresLeft.put(v.centeredSumOfSquaresLeft!T + deltaLeft * (deltaLeft * mergeWeight));
            centeredSummatorOfSquaresRight.put(v.centeredSumOfSquaresRight!T + deltaRight * (deltaRight * mergeWeight));
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresLeft.sum;
    }
        ///
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresRight.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
    ///
    F correlation(F = T)() @property
        in (centeredSumOfSquaresLeft > 0, "`x` must have centered sum of squares larger than zero")
        in (centeredSumOfSquaresRight > 0, "`y` must have centered sum of squares larger than zero")
    {
        import mir.math.common: sqrt;
        return centeredSumOfProducts!F / (sqrt(centeredSumOfSquaresLeft!F) * sqrt(centeredSumOfSquaresRight!F));
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;

    v.correlation.shouldApprox == -0.0611234;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;

    v.correlation.shouldApprox == -0.0611234;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (assumeZeroMean)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    auto a1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto b1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto a2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x1 = a1.center;
    auto y1 = b1.center;
    auto x2 = a2.center;
    auto y2 = b2.center;

    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -1.9375 / 12; //note: different from above due to inconsistent centering
    v1.covariance(false).shouldApprox == -1.9375 / 11; //note: different from above due to inconsistent centering

    v1.correlation.shouldApprox == -0.0229089; //note: different from above due to inconsistent centering
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive)(4.0, 3.0);
    v.covariance(true).should == 0;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should, shouldApprox;
    import std.range: chunks, iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    CorrelationAccumulator!(double, CorrelationAlgo.online, Summation.naive) v;
    v.put(x, y);
    v.covariance(true).should == 10 / 5;
    v.correlation.shouldApprox == 1;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v2;
    v2.put(x.chunks(1), y.chunks(1));
    v2.covariance(true).should == 10 / 5;
    v2.correlation.shouldApprox == 1;
}

///
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation)
    if (isMutable!T && correlationAlgo == CorrelationAlgo.twoPass)
{
    import mir.functional: naryFun;
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;
    import std.range.primitives: isForwardRange, save;
    import mir.stat.descriptive.univariate: MeanAccumulator;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    private S summatorLeft;
    ///
    private S summatorRight;
    ///
    private S centeredSummatorOfProducts;
    ///
    private S centeredSummatorOfSquaresLeft;
    ///
    private S centeredSummatorOfSquaresRight;

    ///
    this(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
         Slice!(IteratorX, 1, kindX) x, Slice!(IteratorY, 1, kindY) y)
     in
     {
        assert(x.length == y.length,
               "CorrelationAcumulator.put: both vectors must have the same length");
     }
     do
    {
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: map, vmap, zip;

        _count = x.length;
        summatorLeft.put(x.lightScope);
        summatorRight.put(y.lightScope);
        if (_count == 0)
            return;
        auto z = x.vmap(LeftOp!("-", T)(meanLeft)).zip(y.vmap(LeftOp!("-", T)(meanRight))).map!("a * b", "a * a", "b * b");
        z.putter3(centeredSummatorOfProducts,
                  centeredSummatorOfSquaresLeft,
                  centeredSummatorOfSquaresRight);
    }

    ///
    this(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice, y.toSlice);
    }

    /// Generic ranges must be forward ranges: the first pass uses saved cursors to compute means.
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isForwardRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isForwardRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        import mir.primitives: elementCount, hasShape;

        static if (hasShape!RangeX && hasShape!RangeY) {
            assert(x.elementCount == y.elementCount);
            _count += x.elementCount;
            summatorLeft.put(x.save);
            summatorRight.put(y.save);
        } else {
            import std.range: zip;

            foreach(a, b; zip(x.save, y.save)) {
                _count++;
                summatorLeft.put(a);
                summatorRight.put(b);
            }
        }

        if (x.empty && y.empty)
            return;

        T xMean = meanLeft;
        T yMean = meanRight;
        T xDeMean;
        T yDeMean;
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            xDeMean = x.front - xMean;
            yDeMean = y.front - yMean;
            centeredSummatorOfProducts.put(xDeMean * yDeMean);
            centeredSummatorOfSquaresLeft.put(xDeMean * xDeMean);
            centeredSummatorOfSquaresRight.put(yDeMean * yDeMean);
            x.popFront;
            y.popFront;
        }
    }

    ///
    this()(T x, T y)
    {
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(0);
        centeredSummatorOfSquaresLeft.put(0);
        centeredSummatorOfSquaresRight.put(0);
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresLeft.sum;
    }
    ///
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresRight.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
    ///
    F correlation(F = T)() @property
        in (centeredSumOfSquaresLeft > 0, "`x` must have centered sum of squares larger than zero")
        in (centeredSumOfSquaresRight > 0, "`y` must have centered sum of squares larger than zero")
    {
        import mir.math.common: sqrt;
        return centeredSumOfProducts!F / (sqrt(centeredSumOfSquaresLeft!F) * sqrt(centeredSumOfSquaresRight!F));
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox ==3;

    v.correlation.shouldApprox == -0.0623684;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;
}

// Check Vmap
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x = a + 1;
    auto y = b - 1;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(4.0, 3.0);
    v.centeredSumOfProducts.should == 0;
    v.count.should == 1;
}

// withAsSlice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array: RCArray;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    auto x = RCArray!double(12);
    foreach(i, ref e; x)
        e = a[i];
    auto y = RCArray!double(12);
    foreach(i, ref e; y)
        e = b[i];

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x, y);
    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
    v.correlation.shouldApprox == -0.0623684;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should, shouldApprox;
    import std.range: iota;
    import std.algorithm: map;

    auto x1 = iota(0, 5);
    auto y1 = iota(-3, 2);
    auto v1 = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x1, y1);
    v1.covariance(true).should == 10.0 / 5;
    v1.correlation.shouldApprox == 1;

    // this version can't use elementCount
    auto x2 = x1.map!(a => 2 * a);
    auto y2 = y1.map!(a => 2 * a);
    auto v2 = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x2, y2);
    v2.covariance(true).should == 40.0 / 5;
    v2.correlation.shouldApprox == 1;
}

///
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation)
    if (isMutable!T && correlationAlgo == CorrelationAlgo.assumeZeroMean)
{
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: Slice, SliceKind, hasAsSlice, isConvertibleToSlice, isSlice;
    import mir.primitives: isInputRange, front, empty, popFront;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S centeredSummatorOfProducts;
    ///
    S centeredSummatorOfSquaresLeft;
    ///
    S centeredSummatorOfSquaresRight;

    ///
    this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        this.put(x, y);
    }

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    void put(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CorrelationAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.topology: zip, map;

        _count += x.length;
        auto z = x.zip(y).map!(multivariateProduct!T,
            multivariateProduct!(T, 0, 0), multivariateProduct!(T, 1, 1));
        z.putter3(centeredSummatorOfProducts,
                  centeredSummatorOfSquaresLeft,
                  centeredSummatorOfSquaresRight);
    }

    ///
    void put(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this.put(x.toSlice, y.toSlice);
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX && is(elementType!RangeX : T) &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY && is(elementType!RangeY : T))
    {
        while (!x.empty || !y.empty)
        {
            assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                   "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
            this.put(x.front, y.front);
            x.popFront;
            y.popFront;
        }
    }

    ///
    void put()(T x, T y)
    {
        _count++;
        centeredSummatorOfProducts.put(x * y);
        centeredSummatorOfSquaresLeft.put(x * x);
        centeredSummatorOfSquaresRight.put(y * y);
    }

    ///
    void put(U, Summation sumAlgo)(CorrelationAccumulator!(U, correlationAlgo, sumAlgo) v)
    {
        _count += v.count;
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v);
            mergeMultivariateSum!("centeredSummatorOfSquaresLeft", "centeredSumOfSquaresLeft", T)(centeredSummatorOfSquaresLeft, v);
            mergeMultivariateSum!("centeredSummatorOfSquaresRight", "centeredSumOfSquaresRight", T)(centeredSummatorOfSquaresRight, v);
        }
        else
        {
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T);
            centeredSummatorOfSquaresLeft.put(v.centeredSumOfSquaresLeft!T);
            centeredSummatorOfSquaresRight.put(v.centeredSumOfSquaresRight!T);
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return 0;
    }
    ///
    F sumRight(F = T)() @property
    {
        return 0;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return 0;
    }
    ///
    F meanRight(F = T)() @property
    {
        return 0;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresLeft.sum;
    }
    ///
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresRight.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
    ///
    F correlation(F = T)() @property
        in (centeredSumOfSquaresLeft > 0, "`x` must have centered sum of squares larger than zero")
        in (centeredSumOfSquaresRight > 0, "`y` must have centered sum of squares larger than zero")
    {
        import mir.math.common: sqrt;
        return centeredSumOfProducts!F / (sqrt(centeredSumOfSquaresLeft!F) * sqrt(centeredSumOfSquaresRight!F));
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.transform: center;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x = a.center;
    auto y = b.center;

    CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 6.5 / 13;
    v.covariance(false).shouldApprox == 6.5 / 12;

    v.correlation.shouldApprox == 0.0628802;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.univariate: mean;
    import mir.stat.transform: center;
    import mir.test: should, shouldApprox;

    auto a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto aMean = a.mean;
    auto bMean = b.mean;
    auto x = a.dup;
    auto y = b.dup;
    for (size_t i; i < a.length; i++) {
        x[i] -= aMean;
        y[i] -= bMean;
    }

    CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == 6.5 / 13;
    v.covariance(false).shouldApprox == 6.5 / 12;

    v.correlation.shouldApprox == 0.0628802;

    v.meanLeft.should == 0;
    v.meanRight.should == 0;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive)(x.center, y.center);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;
    v.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto b1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto a2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto meanA = (a1.sum + a2.sum) / 12;
    auto meanB = (b1.sum + b2.sum) / 12;
    auto x1 = a1 - meanA;
    auto y1 = b1 - meanB;
    auto x2 = a2 - meanA;
    auto y2 = b2 - meanB;

    CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;
    v1.correlation.shouldApprox == -0.0623684;
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive)(4.0, 3.0);
    v.centeredSumOfProducts.should == 12;
    v.count.should == 1;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;

    auto x = iota(0, 5);
    auto y = iota(-3, 2);
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive)(x, y);
    v.centeredSumOfProducts.should == 0; // different from other algorithms because these don't have mean of zero
}

///
struct CorrelationAccumulator(T, CorrelationAlgo correlationAlgo, Summation summation)
    if (isFloatingPoint!T && isMutable!T && correlationAlgo == CorrelationAlgo.hybrid)
{
    import mir.functional: naryFun;
    import mir.math.sum: elementType, Summator;
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
    import mir.primitives: isInputRange, front, empty, popFront;
    import std.range.primitives: isForwardRange, save;

    ///
    private size_t _count;
    ///
    alias S = Summator!(T, summation);
    ///
    S summatorLeft;
    ///
    S summatorRight;
    ///
    S centeredSummatorOfProducts;
    ///
    S centeredSummatorOfSquaresLeft;
    ///
    S centeredSummatorOfSquaresRight;

    ///
    this()(T x, T y)
    {
        this.put(x, y);
    }

    ///
    this(IteratorX, IteratorY, SliceKind kindX, SliceKind kindY)(
        Slice!(IteratorX, 1, kindX) x,
        Slice!(IteratorY, 1, kindY) y
    )
    in
    {
        assert(x.length == y.length,
               "CorrelationAcumulator.put: both vectors must have the same length");
    }
    do
    {
        import mir.ndslice.internal: LeftOp;
        import mir.ndslice.topology: map, vmap, zip;

        _count += x.length;
        summatorLeft.put(x.lightScope);
        summatorRight.put(y.lightScope);
        if (_count == 0)
            return;
        auto z = x.vmap(LeftOp!("-", T)(meanLeft)).zip(y.vmap(LeftOp!("-", T)(meanRight))).map!("a * b", "a * a", "b * b");
        z.putter3(centeredSummatorOfProducts,
                  centeredSummatorOfSquaresLeft,
                  centeredSummatorOfSquaresRight);
    }

    ///
    this(SliceLikeX, SliceLikeY)(SliceLikeX x, SliceLikeY y)
        if (isConvertibleToSlice!SliceLikeX && !isSlice!SliceLikeX &&
            isConvertibleToSlice!SliceLikeY && !isSlice!SliceLikeY)
    {
        import mir.ndslice.slice: toSlice;
        this(x.toSlice, y.toSlice);
    }

    /// Generic scalar batches must be forward ranges because their observations are traversed twice.
    /// An outer range of batches may be single-pass; each inner batch must support its own calculation.
   this(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && !isConvertibleToSlice!RangeX &&
            isInputRange!RangeY && !isConvertibleToSlice!RangeY)
    {
        static if (is(elementType!RangeX : T) && is(elementType!RangeY : T)) {
            static assert(isForwardRange!RangeX && isForwardRange!RangeY,
                "Hybrid accumulation requires forward ranges for scalar batches; " ~
                "use the online algorithm for single-pass inputs.");
            import mir.primitives: elementCount, hasShape;

            static if (hasShape!RangeX && hasShape!RangeY) {
                assert(x.elementCount == y.elementCount);
                _count += x.elementCount;
                summatorLeft.put(x.save);
                summatorRight.put(y.save);
            } else {
                import std.range: zip;

                foreach(a, b; zip(x.save, y.save)) {
                    _count++;
                    summatorLeft.put(a);
                    summatorRight.put(b);
                }
            }

            if (x.empty && y.empty)
                return;

            T xMean = meanLeft;
            T yMean = meanRight;
            T xDeMean;
            T yDeMean;
            while (!x.empty || !y.empty)
            {
                assert(!(!x.empty && y.empty) && !(x.empty && !y.empty),
                       "x and y must both be empty at the same time, one cannot be empty while the other has remaining items");
                xDeMean = x.front - xMean;
                yDeMean = y.front - yMean;
                centeredSummatorOfProducts.put(xDeMean * yDeMean);
                centeredSummatorOfSquaresLeft.put(xDeMean * xDeMean);
                centeredSummatorOfSquaresRight.put(yDeMean * yDeMean);
                x.popFront;
                y.popFront;
            }
        } else {
            this.put(x, y);
        }
    }

    ///
    void put(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        static if (is(elementType!RangeX : T) && is(elementType!RangeY : T)) {
            static assert((isConvertibleToSlice!RangeX && isConvertibleToSlice!RangeY) ||
                          (isForwardRange!RangeX && isForwardRange!RangeY),
                "Hybrid accumulation requires forward ranges for scalar batches; " ~
                "use the online algorithm for single-pass inputs.");
            auto v = typeof(this)(x, y);
            this.put(v);
        } else {
            while (!x.empty || !y.empty)
            {
                assert(!x.empty && !y.empty, "x and y must have the same length");
                this.put(x.front, y.front);
                x.popFront;
                y.popFront;
            }
        }
    }

    ///
    void put()(T x, T y)
    {
        T deltaX = x;
        T deltaY = y;
        if (count > 0) {
            deltaX -= meanLeft;
            deltaY -= meanRight;
        }
        _count++;
        summatorLeft.put(x);
        summatorRight.put(y);
        centeredSummatorOfProducts.put(deltaX * (y - meanRight));
        centeredSummatorOfSquaresLeft.put(deltaX * (x - meanLeft));
        centeredSummatorOfSquaresRight.put(deltaY * (y - meanRight));
    }

    ///
    void put(U, CorrelationAlgo covAlgo, Summation sumAlgo)(CorrelationAccumulator!(U, covAlgo, sumAlgo) v)
    {
        // Empty partitions contribute no observations or centered moments.
        if (v.count == 0)
            return;
        size_t oldCount = count;
        // An empty destination has no between-group correction. Avoid Inf * 0.
        T deltaLeft = 0;
        T deltaRight = 0;
        if (count > 0) {
            deltaLeft = v.meanLeft;
            deltaRight = v.meanRight;
            deltaLeft -= meanLeft!T;
            deltaRight -= meanRight!T;
        }
        _count += v.count;
        // Form the count weight first, as in variance merging, so count factors
        // cannot overflow an otherwise representable between-group correction.
        const T mergeWeight = cast(T) v.count * (cast(T) oldCount / count);
        static if (is(T == U) && summation == sumAlgo &&
            (summation == Summation.kbn || summation == Summation.kb2 ||
             summation == Summation.precise))
        {
            mergeMultivariateSum!("summatorLeft", "sumLeft", T)(summatorLeft, v);
            mergeMultivariateSum!("summatorRight", "sumRight", T)(summatorRight, v);
            mergeMultivariateSum!("centeredSummatorOfProducts", "centeredSumOfProducts", T)(centeredSummatorOfProducts, v, multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
            mergeMultivariateSum!("centeredSummatorOfSquaresLeft", "centeredSumOfSquaresLeft", T)(centeredSummatorOfSquaresLeft, v, deltaLeft * (deltaLeft * mergeWeight));
            mergeMultivariateSum!("centeredSummatorOfSquaresRight", "centeredSumOfSquaresRight", T)(centeredSummatorOfSquaresRight, v, deltaRight * (deltaRight * mergeWeight));
        }
        else
        {
            summatorLeft.put(v.sumLeft!T);
            summatorRight.put(v.sumRight!T);
            centeredSummatorOfProducts.put(v.centeredSumOfProducts!T + multivariateMergeProduct(deltaLeft, deltaRight, mergeWeight));
            centeredSummatorOfSquaresLeft.put(v.centeredSumOfSquaresLeft!T + deltaLeft * (deltaLeft * mergeWeight));
            centeredSummatorOfSquaresRight.put(v.centeredSumOfSquaresRight!T + deltaRight * (deltaRight * mergeWeight));
        }
    }

const:

    ///
    size_t count() @property
    {
        return _count;
    }
    ///
    F sumLeft(F = T)() @property
    {
        return cast(F) summatorLeft.sum;
    }
    ///
    F sumRight(F = T)() @property
    {
        return cast(F) summatorRight.sum;
    }
    ///
    F meanLeft(F = T)() @property
    {
        return sumLeft!F / count;
    }
    ///
    F meanRight(F = T)() @property
    {
        return sumRight!F / count;
    }
    ///
    F centeredSumOfProducts(F = T)() @property
    {
        return cast(F) centeredSummatorOfProducts.sum;
    }
    ///
    F centeredSumOfSquaresLeft(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresLeft.sum;
    }
    ///
    F centeredSumOfSquaresRight(F = T)() @property
    {
        return cast(F) centeredSummatorOfSquaresRight.sum;
    }
    ///
    F covariance(F = T)(bool isPopulation) @property
    {
        return centeredSumOfProducts!F / (count + isPopulation - 1);
    }
    ///
    F correlation(F = T)() @property
        in (centeredSumOfSquaresLeft > 0, "`x` must have centered sum of squares larger than zero")
        in (centeredSumOfSquaresRight > 0, "`y` must have centered sum of squares larger than zero")
    {
        import mir.math.common: sqrt;
        return centeredSumOfProducts!F / (sqrt(centeredSumOfSquaresLeft!F) * sqrt(centeredSumOfSquaresRight!F));
    }
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;

    v.correlation.shouldApprox == -0.0611234;
}

// Check dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25];

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v;
    v.put(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.meanLeft.shouldApprox == 2.4375;
    v.meanRight.shouldApprox == 3;

    v.correlation.shouldApprox == -0.0623684;

    v.put(4.0, 3.0);
    v.covariance(true).shouldApprox == -5.5 / 13;
    v.covariance(false).shouldApprox == -5.5 / 12;

    v.correlation.shouldApprox == -0.0611234;
}

// rcslice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;
    auto v = CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)(x, y);

    v.covariance(true).shouldApprox == -5.5 / 12;
    v.covariance(false).shouldApprox == -5.5 / 11;

    v.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (naive)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    CorrelationAccumulator!(double, CorrelationAlgo.naive, Summation.naive) v2;
    v2.put(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (twoPass)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto y1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto x2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CorrelationAccumulator!(double, CorrelationAlgo.twoPass, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -5.5 / 12;
    v1.covariance(false).shouldApprox == -5.5 / 11;

    v1.correlation.shouldApprox == -0.0623684;
}

// Check adding CorrelationAccumultors (assumeZeroMean)
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sum, Summation;
    import mir.ndslice.slice: sliced;
    import mir.stat.transform: center;
    import mir.test: shouldApprox;

    auto a1 = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25].sliced;
    auto b1 = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5].sliced;
    auto a2 = [  2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto b2 = [ 9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;
    auto x1 = a1.center;
    auto y1 = b1.center;
    auto x2 = a2.center;
    auto y2 = b2.center;

    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    auto v2 = CorrelationAccumulator!(double, CorrelationAlgo.assumeZeroMean, Summation.naive)(x2, y2);
    v1.put(v2);

    v1.covariance(true).shouldApprox == -1.9375 / 12; //note: different from above due to inconsistent centering
    v1.covariance(false).shouldApprox == -1.9375 / 11; //note: different from above due to inconsistent centering

    v1.correlation.shouldApprox == -0.0229089; //note: different from above due to inconsistent centering
}

// Initializing with one point
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    auto v = CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)(4.0, 3.0);
    v.covariance(true).should == 0;
}

// Test input range
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should, shouldApprox;
    import std.algorithm: map;
    import std.range: chunks, iota;

    auto x1 = iota(0, 5);
    auto y1 = iota(-3, 2);
    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v1;
    v1.put(x1, y1);
    v1.covariance(true).should == 10.0 / 5;
    v1.correlation.shouldApprox == 1;
    // this version can't use elementCount
    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v2;
    auto x2 = x1.map!(a => 2 * a);
    auto y2 = y1.map!(a => 2 * a);
    v2.put(x2, y2);
    v2.covariance(true).should == 40.0 / 5;
    v2.correlation.shouldApprox == 1;
    CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive) v3;
    v3.put(x1.chunks(1), y1.chunks(1));
    v3.covariance(true).should == 10.0 / 5;
    v3.correlation.shouldApprox == 1;
    auto v4 = CorrelationAccumulator!(double, CorrelationAlgo.hybrid, Summation.naive)(x1.chunks(1), y1.chunks(1));
    v4.covariance(true).should == 10.0 / 5;
    v4.correlation.shouldApprox == 1;
}

/++
Calculates the correlation of the inputs.

If `x` and `y` are both slices or convertible to slices, then they must be
one-dimensional.

For generic ranges of scalar observations, the twoPass and hybrid algorithms
require forward ranges because they traverse the observations twice. The default
algorithm is hybrid. Use the online algorithm for single-pass sources such as
file or stream ranges. Arrays and slices continue to use their dedicated paths.

With assumeStandardized, inputs must already have mean zero and unit standard
deviation over the complete dataset. Select the input z-score convention with
inputStandardization; the default matches zscore's sample convention. Both
inputs must use the same convention, and at least two observations are required.
The function does not standardize or validate the input values.

Params:
    F = controls type of output
    correlationAlgo = algorithm for calculating correlation (default: CorrelationAlgo.hybrid)
    summation = algorithm for calculating sums (default: Summation.appropriate)
    inputStandardization = sample or population z-scores for assumeStandardized;
        other algorithms use the default only
Returns:
    The correlation of the inputs
+/
template correlation(F,
                     CorrelationAlgo correlationAlgo = CorrelationAlgo.hybrid,
                     Summation summation = Summation.appropriate,
                     InputStandardization inputStandardization = InputStandardization.sample)
    if (isFloatingPoint!F && (correlationAlgo == CorrelationAlgo.assumeStandardized ||
        inputStandardization == InputStandardization.sample))
{
    import mir.math.common: fmamath;
    import mir.primitives: isInputRange;
    import mir.math.sum: ResolveSummationType;
    /++
    Params:
        x = range, must be finite iterable
        y = range, must be finite iterable
    +/
    @fmamath F correlation(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;

        static if (correlationAlgo == CorrelationAlgo.assumeStandardized)
            alias Accumulator = CorrelationAccumulator!(F, correlationAlgo,
                ResolveSummationType!(summation, RangeX, F), inputStandardization);
        else
            alias Accumulator = CorrelationAccumulator!(F, correlationAlgo,
                ResolveSummationType!(summation, RangeX, F));
        auto correlationAccumulator = Accumulator(x.move, y.move);
        return correlationAccumulator.correlation();
    }
}

/// ditto
template correlation(
    CorrelationAlgo correlationAlgo = CorrelationAlgo.hybrid,
    Summation summation = Summation.appropriate,
    InputStandardization inputStandardization = InputStandardization.sample)
    if (correlationAlgo == CorrelationAlgo.assumeStandardized ||
        inputStandardization == InputStandardization.sample)
{
    import mir.math.common: fmamath;
    import mir.primitives: isInputRange;
    import mir.stat.descriptive.univariate: stdevType;
    import std.traits: CommonType;
    /++
    Params:
        x = range, must be finite iterable
        y = range, must be finite iterable
    +/
    @fmamath CommonType!(stdevType!RangeX, stdevType!RangeY) correlation(RangeX, RangeY)(RangeX x, RangeY y)
        if (isInputRange!RangeX && isInputRange!RangeY)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .correlation!(F, correlationAlgo, summation, inputStandardization)(x.move, y.move);
    }
}

/// ditto
template correlation(F, string correlationAlgo, string summation = "appropriate",
    InputStandardization inputStandardization = InputStandardization.sample)
{
    mixin("alias correlation = .correlation!(F, CorrelationAlgo." ~ correlationAlgo ~ ", Summation." ~ summation ~ ", inputStandardization);");
}

/// ditto
template correlation(string correlationAlgo, string summation = "appropriate",
    InputStandardization inputStandardization = InputStandardization.sample)
{
    mixin("alias correlation = .correlation!(CorrelationAlgo." ~ correlationAlgo ~ ", Summation." ~ summation ~ ", inputStandardization);");
}

/++
Reuse standardized columns to compute correlation without repeating centering
and variance calculations. Match the input convention to the zscore call.
+/
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.transform: zscore;
    import mir.math.common: approxEqual;
    double[3] x = [10, 20, 30];
    double[3] y = [20, 30, 10];
    // zscore and assumeStandardized both default to sample standardization.
    assert(correlation!"assumeStandardized"(x[].zscore, y[].zscore).approxEqual(-0.5));
    // Population z-scores require the population input convention explicitly.
    assert(correlation!("assumeStandardized", "pairwise", InputStandardization.population)(
        x[].zscore(true), y[].zscore(true)).approxEqual(-0.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (method; AliasSeq!(Summation.naive, Summation.pairwise, Summation.kahan))
    {{
        T[3] x = [T(-1), 0, 1];
        T[3] y = [T(0), 1, -1];
        assert(correlation!(double, "assumeStandardized")(x[], y[]) == -0.5);
        assert(correlation!(CorrelationAlgo.assumeStandardized, method)(x[], y[]) == T(-0.5));
        // Selecting a different convention changes the divisor, not the inputs.
        assert(correlation!(T, CorrelationAlgo.assumeStandardized, method,
            InputStandardization.population)(x[], y[]).approxEqual(T(-1) / 3));
        assert(correlation!(T, "assumeStandardized", "pairwise",
            InputStandardization.population)(x[], y[]).approxEqual(T(-1) / 3));
    }}
}

/// Correlation of vectors
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                2.0,   7.5,   5.0,  1.0,  1.5,  0.0].sliced;
    auto y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    x.correlation(y).shouldApprox == -0.0623684;
}

/// Can also set algorithm type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto a = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
               9.25, -0.75,   2.5, 1.25,   -1, 2.25].sliced;

    auto x = a + 10.0 ^^ 9;
    auto y = b + 10.0 ^^ 9;

    x.correlation(y).shouldApprox == -0.0623684;

    // The naive algorithm is numerically unstable in this case
    //assert(!x.correlation!"naive"(y).approxEqual(-0.0623684));

    // The two-pass algorithm provides the same answer as hybrid
    x.correlation!"twoPass"(y).shouldApprox == -0.0623684;

    // And the assumeZeroMean algorithm is way off
    assert(!x.correlation!"assumeZeroMean"(y).approxEqual(-0.0623684));
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;
    import mir.test: shouldApprox;

    //Set population covariance, covariance algorithm, sum algorithm or output type

    auto a = [1.0, 1e100, 1, -1e100].sliced;
    auto b = [1.0e100, 1, 1, -1e100].sliced;
    auto x = a * 10_000;
    auto y = b * 10_000;

    /++
    Due to Floating Point precision, when centering `x`, subtracting the mean
    from the second and fourth numbers has no effect (for `y` the same is true
    for the first and fourth). Further, after centering and multiplying `x` and
    `y`, the third numbers in the slice has precision too low to be included in
    the centered sum of the products. For the calculations below, the "true"
    correlation should be a tiny amount above 0.5, but it is as if the
    calculation happens between [0, 1, 0, -1] and [1, 0, 0, -1].
    +/
    x.correlation(y).shouldApprox == 0.5;

    x.correlation!("online")(y).shouldApprox == 0.5;
    x.correlation!("online", "kbn")(y).shouldApprox == 0.5;
    x.correlation!("online", "kb2")(y).shouldApprox == 0.5;
    x.correlation!("online", "precise")(y).shouldApprox == 0.5;
    x.correlation!(double, "online", "precise")(y).shouldApprox == 0.5;

    auto z1 = [uint.max - 2, uint.max - 1, uint.max].sliced;
    auto z2 = [uint.max - 3, uint.max - 2, uint.max - 1].sliced;
    z1.correlation(z2).shouldApprox == 1.0;
    static assert(is(typeof(z1.correlation!float(z2)) == float));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    auto x = [0, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 0].sliced;
    auto y = [6, 3, 7, 1, 1, 1,
              9, 5, 3, 1, 3, 7].sliced;

    x.correlation(y).shouldApprox == -0.27934577;
    static assert(is(typeof(x.correlation(y)) == double));

    x.correlation!float(y).shouldApprox == -0.27934577;
    static assert(is(typeof(x.correlation!float(y)) == float));
}

// make sure works with dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: shouldApprox;

    double[] x = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                    2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    double[] y = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                   9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    x.correlation(y).shouldApprox == -0.0623684;
}

/// Works with @nogc
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.allocation: mininitRcslice;
    import mir.test: shouldApprox;

    static immutable a = [  0.0,   1.0,   1.5,  2.0,  3.5, 4.25,
                            2.0,   7.5,   5.0,  1.0,  1.5,  0.0];
    static immutable b = [-0.75,   6.0, -0.25, 8.25, 5.75,  3.5,
                           9.25, -0.75,   2.5, 1.25,   -1, 2.25];
    auto x = mininitRcslice!double(12);
    auto y = mininitRcslice!double(12);
    x[] = a;
    y[] = b;

    x.correlation(y).shouldApprox == -0.0623684;
}

// Re-export the common policy used by all weighted moment statistics.
public import mir.stat.descriptive.weighting: WeightedMomentNormalization;
private import mir.primitives: DeepElementType;

private template weightedPairReplayable(R)
{
    import mir.ndslice.slice: isSlice, isConvertibleToSlice, toSlice;
    import std.traits: isStaticArray;
    import std.range.primitives: isInputRange, isForwardRange, ElementType;
    static if (isStaticArray!R)
        enum weightedPairReplayable = weightedPairReplayable!(typeof(R.init[0]));
    else static if (isConvertibleToSlice!R && !isSlice!R)
        enum weightedPairReplayable = weightedPairReplayable!(typeof(R.init.toSlice));
    else static if (isSlice!R)
        enum weightedPairReplayable = weightedPairReplayable!(typeof(R.init[0]));
    else static if (isInputRange!R)
        enum weightedPairReplayable = isForwardRange!R && weightedPairReplayable!(ElementType!R);
    else
        enum weightedPairReplayable = true;
}

private void weightedPairEach(Sink, X, Y, W)(ref Sink sink, scope X x, scope Y y, scope W w)
{
    import mir.ndslice.slice: isSlice, isConvertibleToSlice, toSlice;
    import std.traits: isStaticArray;
    import std.range.primitives: isInputRange, isForwardRange;
    // Index static arrays directly instead of taking the address of a scope
    // array containing pointers, which DIP1000 rejects in @safe code.
    static if (isConvertibleToSlice!X && !isSlice!X && !isStaticArray!X)
        weightedPairEach(sink, x.toSlice, y, w);
    else static if (isConvertibleToSlice!Y && !isSlice!Y && !isStaticArray!Y)
        weightedPairEach(sink, x, y.toSlice, w);
    else static if (isConvertibleToSlice!W && !isSlice!W && !isStaticArray!W)
        weightedPairEach(sink, x, y, w.toSlice);
    else static if ((isStaticArray!X || isSlice!X) &&
        (isStaticArray!Y || isSlice!Y) && (isStaticArray!W || isSlice!W))
    {
        static if (isSlice!X && isSlice!Y && isSlice!W)
        {
            static assert(X.N == Y.N && X.N == W.N, "weighted moments: ranks must match");
            assert(x.shape == y.shape && x.shape == w.shape, "weighted moments: shapes must match");
        }
        else
            assert(x.length == y.length && x.length == w.length, "weighted moments: lengths must match");
        foreach (i; 0 .. x.length) weightedPairEach(sink, x[i], y[i], w[i]);
    }
    else static if ((isStaticArray!X || isStaticArray!Y || isStaticArray!W) &&
        (isStaticArray!X || isInputRange!X) &&
        (isStaticArray!Y || isInputRange!Y) &&
        (isStaticArray!W || isInputRange!W))
    {
        static if (isStaticArray!X) const length = x.length;
        else static if (isStaticArray!Y) const length = y.length;
        else const length = w.length;
        static if (isStaticArray!X) assert(x.length == length, "weighted moments: lengths must match");
        else static if (isForwardRange!X) x = x.save;
        static if (isStaticArray!Y) assert(y.length == length, "weighted moments: lengths must match");
        else static if (isForwardRange!Y) y = y.save;
        static if (isStaticArray!W) assert(w.length == length, "weighted moments: lengths must match");
        else static if (isForwardRange!W) w = w.save;
        foreach (i; 0 .. length)
        {
            static if (!isStaticArray!X) assert(!x.empty, "weighted moments: lengths must match");
            enum xElement = isStaticArray!X ? "x[i]" : "x.front";
            static if (!isStaticArray!Y) assert(!y.empty, "weighted moments: lengths must match");
            enum yElement = isStaticArray!Y ? "y[i]" : "y.front";
            static if (!isStaticArray!W) assert(!w.empty, "weighted moments: lengths must match");
            enum wElement = isStaticArray!W ? "w[i]" : "w.front";
            mixin("weightedPairEach(sink, " ~ xElement ~ ", " ~ yElement ~ ", " ~ wElement ~ ");");
            static if (!isStaticArray!X) x.popFront;
            static if (!isStaticArray!Y) y.popFront;
            static if (!isStaticArray!W) w.popFront;
        }
        static if (!isStaticArray!X) assert(x.empty, "weighted moments: lengths must match");
        static if (!isStaticArray!Y) assert(y.empty, "weighted moments: lengths must match");
        static if (!isStaticArray!W) assert(w.empty, "weighted moments: lengths must match");
    }
    else static if (isInputRange!X && isInputRange!Y && isInputRange!W)
    {
        static if (isForwardRange!X) x = x.save;
        static if (isForwardRange!Y) y = y.save;
        static if (isForwardRange!W) w = w.save;
        while (!x.empty && !y.empty && !w.empty)
        {
            weightedPairEach(sink, x.front, y.front, w.front);
            x.popFront; y.popFront; w.popFront;
        }
        assert(x.empty && y.empty && w.empty, "weighted moments: lengths must match");
    }
    else sink.put(x, y, w);
}

private void weightedScaleSum(T, Summation mode)(ref Summator!(T, mode) sum, T factor)
{
    if (factor != 1) sum = cast(T) sum.sum * factor;
}

// Normalize and restore units together, without losing a representable result
// to overflow or underflow in an intermediate quotient or product.
private T weightedRestoreProduct(T)(T value, T left, T right, T denominator)
{
    import std.math: frexp, ldexp;
    if (value == 0 || left == 0 || right == 0) return T(0);
    int ev, el, er, ed;
    const v = frexp(value, ev), l = frexp(left, el), r = frexp(right, er);
    const d = frexp(denominator, ed);
    return ldexp(((v * l) * r) / d, ev + el + er - ed);
}

private void weightedMergeSum(T, Summation mode)(ref Summator!(T, mode) target,
    const ref Summator!(T, mode) source, T a = 1, T b = 1, T c = 1)
{
    import mir.stat.internal.summation: mergeSummator;
    if (a == 1 && b == 1 && c == 1) mergeSummator(target, source);
    else target.put(((cast(T) source.sum * a) * b) * c);
}

private struct WeightedCrossMoments(T, Summation mode, bool withSquares)
{
    alias Sum = Summator!(T, mode);
    Sum cross;
    static if (withSquares) Sum left, right;
    T scaleLeft = 0, scaleRight = 0;

    void scaleWeight(T factor)
    {
        weightedScaleSum(cross, factor);
        static if (withSquares)
        {
            weightedScaleSum(left, factor);
            weightedScaleSum(right, factor);
        }
    }

    void scaleTo(T x, T y)
    {
        import std.math: abs;
        x = abs(x); y = abs(y);
        if (x > scaleLeft)
        {
            const ratio = scaleLeft / x;
            weightedScaleSum(cross, ratio);
            static if (withSquares)
            {
                weightedScaleSum(left, ratio);
                weightedScaleSum(left, ratio);
            }
            scaleLeft = x;
        }
        if (y > scaleRight)
        {
            const ratio = scaleRight / y;
            weightedScaleSum(cross, ratio);
            static if (withSquares)
            {
                weightedScaleSum(right, ratio);
                weightedScaleSum(right, ratio);
            }
            scaleRight = y;
        }
    }

    void put(T x, T y, T weight)
    {
        if (weight == 0) return;
        scaleTo(x, y);
        const nx = scaleLeft == 0 ? T(0) : x / scaleLeft;
        const ny = scaleRight == 0 ? T(0) : y / scaleRight;
        cross.put(multivariateMergeProduct(nx, ny, weight));
        static if (withSquares)
        {
            left.put((nx * weight) * nx);
            right.put((ny * weight) * ny);
        }
    }

    void merge(const ref WeightedCrossMoments other, T weightRatio = 1)
    {
        if (weightRatio == 0) return;
        scaleTo(other.scaleLeft, other.scaleRight);
        const rx = scaleLeft == 0 ? T(1) : other.scaleLeft / scaleLeft;
        const ry = scaleRight == 0 ? T(1) : other.scaleRight / scaleRight;
        weightedMergeSum(cross, other.cross, weightRatio, rx, ry);
        static if (withSquares)
        {
            weightedMergeSum(left, other.left, weightRatio, rx, rx);
            weightedMergeSum(right, other.right, weightRatio, ry, ry);
        }
    }
}

private struct WeightedPairAccumulator(T, WeightedMomentNormalization normalization,
    CovarianceAlgo algorithm, Summation summation, bool withSquares)
    if (isFloatingPoint!T && isMutable!T &&
        (algorithm == CovarianceAlgo.online || algorithm == CovarianceAlgo.twoPass ||
         algorithm == CovarianceAlgo.hybrid))
{
    private alias Sum = Summator!(T, summation);
    private alias Moments = WeightedCrossMoments!(T, summation, withSquares);
    private Sum mass, weightSquares, weightPairs;
    private Moments moments;
    private T weightScale = 0;
    private T originLeft = 0, originRight = 0, offsetLeft = 0, offsetRight = 0;
    private real frequency = 0;
    private size_t positiveCount;

    private static void shiftMean(ref T origin, ref T offset, T shift)
    {
        T leading = origin + shift;
        T rounded = leading - origin;
        T trailing = (origin - (leading - rounded)) + (shift - rounded) + offset;
        origin = leading + trailing;
        rounded = origin - leading;
        offset = (leading - (origin - rounded)) + (trailing - rounded);
    }

    private void rescale(T scale)
    {
        if (weightScale == 0) { weightScale = scale; return; }
        if (scale <= weightScale) return;
        const ratio = weightScale / scale;
        weightedScaleSum(mass, ratio);
        weightedScaleSum(weightSquares, ratio);
        weightedScaleSum(weightSquares, ratio);
        weightedScaleSum(weightPairs, ratio);
        weightedScaleSum(weightPairs, ratio);
        moments.scaleWeight(ratio);
        weightScale = scale;
    }

    /// Insert a pair with unit weight.
    void put(X, Y)(X x, Y y) if (is(X : T) && is(Y : T)) { put(x, y, 1); }

    /// Insert a pair. A zero weight excludes both values before conversion.
    void put(X, Y, W)(scope X x, scope Y y, scope W w)
        if (is(X : T) && is(Y : T) && is(W : real))
    {
        import std.math: isFinite, floor;
        const raw = cast(real) w;
        assert(isFinite(raw) && raw >= 0, "weighted moments: weights must be finite and nonnegative");
        if (raw == 0) return;
        const weight = cast(T) w, vx = cast(T) x, vy = cast(T) y;
        assert(isFinite(weight) && weight > 0, "weighted moments: weight must fit accumulator type");
        assert(isFinite(vx) && isFinite(vy), "weighted moments: positive-weight observations must be finite");
        static if (normalization == WeightedMomentNormalization.frequencySample)
        {
            enum real limit = 2.0L ^^ T.mant_dig;
            assert(raw == floor(raw) && raw <= limit - frequency,
                "weighted moments: frequencies must be whole counts with an exactly representable total");
            frequency += raw;
        }
        rescale(weight);
        const a = cast(T) mass.sum, b = weight / weightScale;
        mass.put(b); weightSquares.put(b * b); weightPairs.put(a * b);
        if (positiveCount == 0) { originLeft = vx; originRight = vy; }
        else
        {
            const dx = (vx - originLeft) - offsetLeft;
            const dy = (vy - originRight) - offsetRight;
            const total = cast(T) mass.sum;
            // Divide the larger mass first so a tiny smaller mass survives.
            moments.put(dx, dy, a < b ? (b / total) * a : (a / total) * b);
            shiftMean(originLeft, offsetLeft, dx * (b / total));
            shiftMean(originRight, offsetRight, dy * (b / total));
        }
        ++positiveCount;
    }

    private struct CenteredPass
    {
        T originLeft, originRight, offsetLeft, offsetRight, weightScale;
        Moments moments;
        Sum residualLeft, residualRight;
        void put(X, Y, W)(X x, Y y, W w)
        {
            if (w == 0) return;
            const weight = cast(T) w / weightScale;
            if (weight == 0) return;
            const dx = (cast(T) x - originLeft) - offsetLeft;
            const dy = (cast(T) y - originRight) - offsetRight;
            const sx = moments.scaleLeft, sy = moments.scaleRight;
            moments.put(dx, dy, weight);
            if (moments.scaleLeft > sx) weightedScaleSum(residualLeft, sx / moments.scaleLeft);
            if (moments.scaleRight > sy) weightedScaleSum(residualRight, sy / moments.scaleRight);
            if (moments.scaleLeft != 0) residualLeft.put(weight * (dx / moments.scaleLeft));
            if (moments.scaleRight != 0) residualRight.put(weight * (dy / moments.scaleRight));
        }
    }

    /// Insert equally shaped arrays, slices, or nested ranges.
    void put(X, Y, W)(scope X x, scope Y y, scope W w)
        if (isIterable!X && isIterable!Y && isIterable!W)
    {
        WeightedPairAccumulator batch;
        weightedPairEach(batch, x, y, w);
        enum replayable = weightedPairReplayable!X && weightedPairReplayable!Y && weightedPairReplayable!W;
        static if (algorithm == CovarianceAlgo.twoPass)
            static assert(replayable, "weighted moments: twoPass requires replayable inputs");
        static if (algorithm != CovarianceAlgo.online && replayable)
        {
            if (batch.positiveCount)
            {
                CenteredPass second;
                second.originLeft = batch.originLeft; second.originRight = batch.originRight;
                second.offsetLeft = batch.offsetLeft; second.offsetRight = batch.offsetRight;
                second.weightScale = batch.weightScale;
                weightedPairEach(second, x, y, w);
                const rx = cast(T) second.residualLeft.sum, ry = cast(T) second.residualRight.sum;
                const total = cast(T) batch.mass.sum;
                second.moments.cross.put(-multivariateMergeProduct(rx, ry, T(1) / total));
                static if (withSquares)
                {
                    second.moments.left.put(-(rx / total) * rx);
                    second.moments.right.put(-(ry / total) * ry);
                    if (second.moments.left.sum < 0) second.moments.left = T(0);
                    if (second.moments.right.sum < 0) second.moments.right = T(0);
                }
                // Replace moments through scalar resets, then merge their stored
                // corrections. This also supports older precise-summator compilers.
                batch.moments.cross = T(0);
                static if (withSquares) { batch.moments.left = T(0); batch.moments.right = T(0); }
                batch.moments.scaleLeft = second.moments.scaleLeft;
                batch.moments.scaleRight = second.moments.scaleRight;
                batch.moments.merge(second.moments);
                shiftMean(batch.originLeft, batch.offsetLeft, (rx / total) * second.moments.scaleLeft);
                shiftMean(batch.originRight, batch.offsetRight, (ry / total) * second.moments.scaleRight);
            }
        }
        put(batch);
    }

    /// Merge the same accumulator type, including self-merges.
    void put(const ref WeightedPairAccumulator other)
    {
        if (other.positiveCount == 0) return;
        // Only compare addresses; neither pointer escapes or is dereferenced.
        // The trusted boundary also supports builds without DIP1000.
        if (() @trusted { return &other is &this; }())
        {
            // Build an independent snapshot via summator merges, avoiding
            // precise summation's storage-copy postblit in safe code.
            WeightedPairAccumulator snapshot;
            snapshot.put(other);
            put(snapshot);
            return;
        }
        static if (normalization == WeightedMomentNormalization.frequencySample)
        {
            enum real limit = 2.0L ^^ T.mant_dig;
            assert(other.frequency <= limit - frequency, "weighted moments: merged frequency total is too large");
            frequency += other.frequency;
        }
        const commonScale = weightScale > other.weightScale ? weightScale : other.weightScale;
        rescale(commonScale);
        const ratio = other.weightScale / weightScale;
        const a = cast(T) mass.sum, b = cast(T) other.mass.sum * ratio;
        const dx = (other.originLeft - originLeft) + (other.offsetLeft - offsetLeft);
        const dy = (other.originRight - originRight) + (other.offsetRight - offsetRight);
        weightedMergeSum(mass, other.mass, ratio);
        weightedMergeSum(weightSquares, other.weightSquares, ratio, ratio);
        weightedMergeSum(weightPairs, other.weightPairs, ratio, ratio);
        weightPairs.put(a * b);
        moments.merge(other.moments, ratio);
        if (positiveCount == 0)
        {
            originLeft = other.originLeft; originRight = other.originRight;
            offsetLeft = other.offsetLeft; offsetRight = other.offsetRight;
        }
        else
        {
            const total = cast(T) mass.sum;
            // Divide the larger mass first so a tiny smaller mass survives.
            moments.put(dx, dy, a < b ? (b / total) * a : (a / total) * b);
            shiftMean(originLeft, offsetLeft, dx * (b / total));
            shiftMean(originRight, offsetRight, dy * (b / total));
        }
        positiveCount += other.positiveCount;
    }

    /// Number of positive-weight pairs, not the sum of frequencies.
    size_t count() const @property { return positiveCount; }
    /// Total weight; may overflow even when the result is finite.
    T weight() const @property { return weightScale * cast(T) mass.sum; }
    /// Weighted mean of the left input; requires positive total weight.
    T meanLeft() const @property { assert(positiveCount); return originLeft + offsetLeft; }
    /// Weighted mean of the right input; requires positive total weight.
    T meanRight() const @property { assert(positiveCount); return originRight + offsetRight; }
    /// Kish weight concentration diagnostic, not a universal degrees of freedom.
    T effectiveSampleSize() const @property
    {
        assert(positiveCount);
        const total = cast(T) mass.sum;
        return total / (cast(T) weightSquares.sum / total);
    }
    private T denominator() const
    {
        assert(positiveCount, "weighted moments: no positive-weight observations");
        static if (normalization == WeightedMomentNormalization.population)
            T result = cast(T) mass.sum;
        else static if (normalization == WeightedMomentNormalization.frequencySample)
        {
            assert(frequency > 1, "weighted moments: total frequency must exceed one");
            T result = cast(T) ((frequency - 1) / weightScale);
        }
        else
        {
            static assert(normalization == WeightedMomentNormalization.relativeSample);
            T result = 2 * (cast(T) weightPairs.sum / cast(T) mass.sum);
        }
        assert(result > 0, "weighted moments: insufficient effective weight");
        return result;
    }
    /// Covariance under the selected policy (population for WCorrelationAccumulator).
    T covariance() const @property
    {
        return weightedRestoreProduct(cast(T) moments.cross.sum,
            moments.scaleLeft, moments.scaleRight, denominator());
    }
    static if (withSquares)
    {
        /// Pearson correlation; NaN if either weighted input has zero spread.
        T correlation() const @property
        {
            import std.math: sqrt;
            assert(positiveCount, "weighted correlation: no positive-weight observations");
            const xx = cast(T) moments.left.sum, yy = cast(T) moments.right.sum;
            if (xx <= 0 || yy <= 0) return T.nan;
            const sx = sqrt(xx), sy = sqrt(yy), xy = cast(T) moments.cross.sum;
            // Dividing by the smaller root first avoids premature underflow.
            return sx < sy ? (xy / sx) / sy : (xy / sy) / sx;
        }
    }
}

/++
Mergeable weighted covariance accumulator. Use put(x, y, weight) for a pair,
put(x, y, weights) for a batch, and put(other) to merge the same accumulator type.
Unit-weight pairs can be inserted with put(x, y). The read-only properties are
count (positive-weight pairs), weight (total weight), meanLeft, meanRight,
effectiveSampleSize (Kish concentration diagnostic), and covariance.
Empty batches and empty merges
are no-ops; getters other than count and weight require positive total weight.

Normalization and input requirements are those of $(LREF weightedCovariance).
The default is population normalization, hybrid updates, and KBN summation.
Online performs one pass; twoPass requires replayable inputs; hybrid uses a
corrected second pass for replayable inputs and online updates otherwise.
Scalar insertions always use online updates.

Weight and residual scales reduce intermediate overflow. Means retain a trailing
correction, and matching-scale merges preserve stored summator corrections where
supported by the summation algorithm. Rescaling transfers rounded sums; pairwise
merges also transfer rounded sums. Results need not be bitwise independent of
partitioning. Weight ratios, centered differences, and normalized moments remain
subject to the accumulator type's range and rounding. Convert large integers
only after subtracting a common offset if their differences would otherwise be lost.
+/
template WCovarianceAccumulator(T,
    WeightedMomentNormalization normalization = WeightedMomentNormalization.population,
    CovarianceAlgo covarianceAlgo = CovarianceAlgo.hybrid,
    Summation summation = Summation.kbn)
{
    alias WCovarianceAccumulator = WeightedPairAccumulator!(T, normalization, covarianceAlgo, summation, false);
}

/++
Mergeable weighted Pearson correlation. Supports the same insertion, merge,
weight validation and numerical contracts as $(LREF WCovarianceAccumulator).
The default algorithm is hybrid and the default summation is KBN. It provides
the same properties plus correlation.
There is no normalization parameter: the common covariance/variance correction
cancels. The covariance getter reports population covariance. Constant inputs
produce NaN correlation; empty or all-zero-weight inputs cannot be queried.
+/
template WCorrelationAccumulator(T,
    CorrelationAlgo correlationAlgo = CorrelationAlgo.hybrid,
    Summation summation = Summation.kbn)
    if (correlationAlgo == CorrelationAlgo.online || correlationAlgo == CorrelationAlgo.twoPass ||
        correlationAlgo == CorrelationAlgo.hybrid)
{
    private enum algorithm = correlationAlgo == CorrelationAlgo.online ? CovarianceAlgo.online :
        correlationAlgo == CorrelationAlgo.twoPass ? CovarianceAlgo.twoPass : CovarianceAlgo.hybrid;
    alias WCorrelationAccumulator = WeightedPairAccumulator!(T, WeightedMomentNormalization.population,
        algorithm, summation, true);
}

private template WeightedPairType(X, Y)
{
    import std.traits: CommonType;
    import mir.stat.descriptive.univariate: stdevType;
    alias WeightedPairType = CommonType!(stdevType!(DeepElementType!X), stdevType!(DeepElementType!Y));
}

/++
Weighted covariance measures whether two inputs tend to move together. Positive
covariance means larger values tend to occur together; negative covariance means
larger values of one tend to occur with smaller values of the other. Units are
the product of the two input units.

Choose WeightedMomentNormalization.population (the default) to describe the
weighted distribution. Choose frequencySample when weights count repeated pairs
and you want the sample covariance of the expanded data. Choose relativeSample
for fixed relative weights on independent pairs sharing a common mean vector and
covariance matrix. That correction estimates the covariance of observations,
not uncertainty of their weighted means, and is not a survey-design correction.

For x=[0,10], y=[0,20], weights=[1,3], the results are 37.5 (population), 50
(frequencySample), and 100 (relativeSample). Frequency weights represent the four
pairs (0,0), (10,20), (10,20), (10,20). Relative weights instead describe two pairs
with influence 1:3. Scaling all weights preserves population and relativeSample
results; scaling frequency counts changes the represented sample size.

Weights must be finite and nonnegative. Zero weights exclude both coordinates,
even NaN or infinity. Positive weights must remain positive and finite in the
accumulator type, and both corresponding values must be finite. Frequency-sample
weights must be whole counts totaling at most 2^^T.mant_dig, and the total must
exceed one when querying covariance. Relative-sample covariance needs at least
two positive-weight pairs and a representable positive correction denominator.
Preconditions are checked by assertions.

Arguments are observations first, then their shared weights. Arrays, independently
strided slices, and nested ranges must have matching shapes and lengths. No input
is modified; single-pass cursors are consumed. Integral observations infer double;
mixed observation types use their common promoted type. F explicitly selects a
floating accumulator and result type. Weights do not determine the result type.
See $(LREF WCovarianceAccumulator) for algorithms and numerical limits.
+/
template weightedCovariance(F,
    WeightedMomentNormalization normalization = WeightedMomentNormalization.population,
    CovarianceAlgo covarianceAlgo = CovarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
    if (isFloatingPoint!F)
{
    ///
    F weightedCovariance(X, Y, W)(scope X x, scope Y y, scope W weights)
        if (isIterable!X && isIterable!Y && isIterable!W)
    {
        import mir.math.sum: ResolveSummationType;
        WCovarianceAccumulator!(F, normalization, covarianceAlgo,
            ResolveSummationType!(summation, F[], F)) accumulator;
        accumulator.put(x, y, weights);
        return accumulator.covariance;
    }
}

/// ditto
template weightedCovariance(
    WeightedMomentNormalization normalization = WeightedMomentNormalization.population,
    CovarianceAlgo covarianceAlgo = CovarianceAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    WeightedPairType!(X, Y) weightedCovariance(X, Y, W)(scope X x, scope Y y, scope W weights)
        if (isIterable!X && isIterable!Y && isIterable!W)
    {
        return .weightedCovariance!(WeightedPairType!(X, Y), normalization, covarianceAlgo, summation)(x, y, weights);
    }
}

/// Choose the policy according to what each weight represents.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.multivariate: weightedCovariance, WeightedMomentNormalization;
    import mir.math.common: approxEqual;
    alias N = WeightedMomentNormalization;
    double[2] x = [0, 10], y = [0, 20], weights = [1, 3];
    assert(weightedCovariance(x, y, weights).approxEqual(37.5));
    assert(weightedCovariance!(N.frequencySample)(x, y, weights).approxEqual(50));
    assert(weightedCovariance!(N.relativeSample)(x, y, weights).approxEqual(100));
    double[2] probabilities = [0.25, 0.75];
    assert(weightedCovariance!(N.relativeSample)(x, y, probabilities).approxEqual(100));
}

/++
Weighted Pearson correlation describes the strength and direction of a linear
relationship on a dimensionless scale. It divides the weighted centered cross
product by the square roots of the two weighted centered sums of squares.
For x=[0,10], y=[0,20], weights=[1,3], correlation is 1: the positive linear
relationship is perfect even though the two inputs use different scales.

No population/sample selector is needed: a common normalization factor cancels
from numerator and denominator. Whole-number weights give the correlation of
the expanded sample; relative weights give the same result after a common
positive rescaling. Fractional weights are accepted without implying repeated
observations. This function computes Pearson correlation, not a survey-design
uncertainty estimate or a weighted rank correlation.

Inputs, algorithms and numerical limits follow $(LREF weightedCovariance).
A constant input has zero spread, so correlation is undefined and returns NaN.
The calculation uses scaled moments directly, allowing finite correlation even
when covariance or individual variances cannot be represented. No result is
clamped to [-1,1]; floating-point rounding may slightly exceed these limits.
+/
template weightedCorrelation(F,
    CorrelationAlgo correlationAlgo = CorrelationAlgo.hybrid,
    Summation summation = Summation.appropriate)
    if (isFloatingPoint!F)
{
    ///
    F weightedCorrelation(X, Y, W)(scope X x, scope Y y, scope W weights)
        if (isIterable!X && isIterable!Y && isIterable!W)
    {
        import mir.math.sum: ResolveSummationType;
        WCorrelationAccumulator!(F, correlationAlgo,
            ResolveSummationType!(summation, F[], F)) accumulator;
        accumulator.put(x, y, weights);
        return accumulator.correlation;
    }
}

/// ditto
template weightedCorrelation(
    CorrelationAlgo correlationAlgo = CorrelationAlgo.hybrid,
    Summation summation = Summation.appropriate)
{
    ///
    WeightedPairType!(X, Y) weightedCorrelation(X, Y, W)(scope X x, scope Y y, scope W weights)
        if (isIterable!X && isIterable!Y && isIterable!W)
    {
        return .weightedCorrelation!(WeightedPairType!(X, Y), correlationAlgo, summation)(x, y, weights);
    }
}

/// Streaming pairs and merging chunks use the same correlation definition.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.multivariate: weightedCorrelation, WCorrelationAccumulator;
    import mir.math.common: approxEqual;
    import std.math: isNaN;
    double[3] x = [0, 10, 20], y = [0, 20, 40], w = [1, 3, 2];
    assert(weightedCorrelation(x, y, w).approxEqual(1));
    WCorrelationAccumulator!double left, right;
    left.put(x[0 .. 2], y[0 .. 2], w[0 .. 2]);
    right.put(x[2], y[2], w[2]);
    left.put(right);
    assert(left.count == 3 && left.weight == 6);
    assert(left.correlation.approxEqual(1));
    double[3] constant = [5, 5, 5];
    assert(weightedCorrelation(x, constant, w).isNaN);
}

// All policies agree with variance on the diagonal and with expanded counts.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    import mir.stat.descriptive.univariate: weightedVariance, WeightedVarianceNormalization;
    alias N = WeightedMomentNormalization;
    static assert(is(N == WeightedVarianceNormalization));
    double[4] x = [1, 2, 4, 8], y = [9, 3, -1, 6];
    int[4] w = [1, 3, 2, 1];
    double[7] expandedX = [1, 2, 2, 2, 4, 4, 8];
    double[7] expandedY = [9, 3, 3, 3, -1, -1, 6];
    static foreach (algo; [CovarianceAlgo.online, CovarianceAlgo.twoPass, CovarianceAlgo.hybrid])
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        assert((weightedCovariance!(N.frequencySample, algo, mode)(x, y, w))
            .approxEqual(covariance(expandedX[], expandedY[]), 1e-12, 1e-12));
        static foreach (norm; [N.population, N.frequencySample, N.relativeSample])
        {{
            alias A = WCovarianceAccumulator!(double, norm, algo, mode);
            A batch, stream, left, right, empty;
            batch.put(x, y, w);
            foreach (i; 0 .. x.length) stream.put(x[i], y[i], w[i]);
            left.put(x[0 .. 2], y[0 .. 2], w[0 .. 2]);
            right.put(x[2 .. $], y[2 .. $], w[2 .. $]);
            left.put(right); left.put(empty);
            assert(stream.covariance.approxEqual(batch.covariance, 1e-12, 1e-12));
            assert(left.covariance.approxEqual(batch.covariance, 1e-12, 1e-12));
            assert(left.meanLeft.approxEqual(batch.meanLeft));
            assert(left.meanRight.approxEqual(batch.meanRight));
            assert(left.effectiveSampleSize.approxEqual(batch.effectiveSampleSize));
            empty.put(batch);
            assert(empty.covariance == batch.covariance);
            assert((weightedCovariance!(norm, algo, mode)(x, x, w))
                .approxEqual(weightedVariance!norm(x, w), 1e-12, 1e-12));
            auto before = batch.covariance;
            batch.put(batch);
            const factor = norm == N.frequencySample ? 12.0 / 13 : 1.0;
            // Relative weights count duplicated rows as independent observations.
            const relativeFactor = (49.0 - 15) / (49.0 - 7.5);
            assert(batch.covariance.approxEqual(before *
                (norm == N.relativeSample ? relativeFactor : factor), 1e-12, 1e-12));
            assert(batch.count == 8 && batch.weight.approxEqual(14));
        }}
    }}
    static foreach (algo; [CorrelationAlgo.online, CorrelationAlgo.twoPass, CorrelationAlgo.hybrid])
    static foreach (mode; [Summation.naive, Summation.pairwise, Summation.kbn, Summation.kb2, Summation.precise])
    {{
        alias A = WCorrelationAccumulator!(double, algo, mode);
        A left, right, stream;
        left.put(x[0 .. 2], y[0 .. 2], w[0 .. 2]);
        right.put(x[2 .. $], y[2 .. $], w[2 .. $]);
        foreach (i; 0 .. x.length) stream.put(x[i], y[i], w[i]);
        left.put(right);
        auto expected = correlation(expandedX[], expandedY[]);
        assert(left.correlation.approxEqual(expected, 1e-12, 1e-12));
        assert(stream.correlation.approxEqual(expected, 1e-12, 1e-12));
        assert((weightedCorrelation!(algo, mode)(x, y, w)).approxEqual(expected, 1e-12, 1e-12));
        left.put(left);
        assert(left.correlation.approxEqual(expected, 1e-12, 1e-12));
    }}
}

// Independent pairwise references test signs, order, offsets and partitioning.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    alias N = WeightedMomentNormalization;
    foreach (trial; 0 .. 30)
    {
        double[12] x, y, w;
        real xy = 0, xx = 0, yy = 0, total = 0, pairWeight = 0;
        foreach (i; 0 .. x.length)
        {
            x[i] = 1e12 + cast(int)((i * 19 + trial * 7) % 31) - 15;
            y[i] = -1e12 + cast(int)((i * 11 + trial * 3) % 23) - 11;
            w[i] = 1 + (i * 13 + trial) % 7;
            total += w[i];
            foreach (j; 0 .. i)
            {
                const dx = cast(real) x[i] - x[j], dy = cast(real) y[i] - y[j];
                const weight = cast(real) w[i] * w[j];
                xy += weight * dx * dy; xx += weight * dx * dx; yy += weight * dy * dy;
                pairWeight += weight;
            }
        }
        static foreach (norm; [N.population, N.frequencySample, N.relativeSample])
        {{
            const denom = norm == N.population ? total * total :
                norm == N.frequencySample ? total * (total - 1) : 2 * pairWeight;
            const expected = cast(double)(xy / denom);
            WCovarianceAccumulator!(double, norm) left, right, reversed;
            const cut = trial % 11 + 1;
            left.put(x[0 .. cut], y[0 .. cut], w[0 .. cut]);
            right.put(x[cut .. $], y[cut .. $], w[cut .. $]);
            left.put(right);
            foreach_reverse (i; 0 .. x.length) reversed.put(x[i], y[i], w[i]);
            foreach (actual; [left.covariance, reversed.covariance, weightedCovariance!norm(x, y, w)])
                assert(actual.approxEqual(expected, 1e-12, 1e-12));
        }}
        import std.math: sqrt;
        const expectedCorrelation = cast(double)((xy / sqrt(xx)) / sqrt(yy));
        assert(weightedCorrelation(x, y, w).approxEqual(expectedCorrelation, 1e-12, 1e-12));
    }
}

// Scales for x and y must remain independent, even in a partition merge.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp, isNaN;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        T large = ldexp(T(1), T.max_exp / 2 + 4);
        T small = 1 / large;
        T[2] x = [0, large], y = [0, small], w = [1, 3];
        foreach (sign; [T(-1), T(1)])
        {
            y[1] = sign * small;
            assert(weightedCovariance(x, y, w).approxEqual(sign * T(3) / 16, T.epsilon * 64, T(0)));
            assert(weightedCorrelation(x, y, w).approxEqual(sign, T.epsilon * 64, T(0)));
            WCorrelationAccumulator!T left, right;
            left.put(x[0], y[0], w[0]); right.put(x[1], y[1], w[1]);
            left.put(right);
            assert(left.correlation.approxEqual(sign, T.epsilon * 64, T(0)));
            assert(left.covariance.approxEqual(sign * T(3) / 16, T.epsilon * 64, T(0)));
        }
        assert(weightedCorrelation(x, x, w).approxEqual(T(1), T.epsilon * 64, T(0)));
        assert(weightedCorrelation(y, y, w).approxEqual(T(1), T.epsilon * 64, T(0)));
        T[3] maskedX = [T.nan, 0, 10], maskedY = [T.infinity, 0, -20], mask = [0, 1, 3];
        assert(weightedCovariance(maskedX, maskedY, mask).approxEqual(T(-37.5)));
        assert(weightedCorrelation(maskedX, maskedY, mask).approxEqual(T(-1)));
        T[2] constant = [1, 1];
        assert(weightedCorrelation(x, constant, w).isNaN);
        WCorrelationAccumulator!T single;
        single.put(1, 2, 3);
        assert(single.correlation.isNaN);
    }}
    double[3] x = [1, 2, 4], y = [3, 0, -1];
    foreach (scale; [1e-300, 1.0, 1e300])
    {
        double[3] weights = [scale, 2 * scale, 3 * scale];
        double[3] ordinary = [1, 2, 3];
        assert(weightedCovariance(x, y, weights).approxEqual(weightedCovariance(x, y, ordinary)));
        assert(weightedCorrelation(x, y, weights).approxEqual(weightedCorrelation(x, y, ordinary)));
    }
}

// Each array may have different strides, and nested forward ranges are replayed.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    import mir.math.common: approxEqual;
    import std.algorithm: filter;
    double[6] x = [1, 2, 3, 4, 5, 6], y = [6, 2, 5, 1, 4, 3];
    double[6] backing = [1, 4, 2, 5, 3, 6], w = [1, 2, 3, 4, 5, 6];
    auto expected = weightedCovariance(x, y, w);
    assert(weightedCovariance(x[].sliced(2, 3), y[].sliced(2, 3),
        backing[].sliced(3, 2).transposed).approxEqual(expected));
    assert(weightedCovariance(x[].filter!(a => true), y[], w[]).approxEqual(expected));
    const double[6] immutableX = x;
    assert(weightedCovariance(immutableX[], y, w).approxEqual(expected));
    double[][] nestedX = [[1.0, 2], [3.0, 4, 5, 6]];
    double[][] nestedY = [[6.0, 2], [5.0, 1, 4, 3]];
    double[][] nestedW = [[1.0, 2], [3.0, 4, 5, 6]];
    assert(weightedCovariance(nestedX, nestedY, nestedW).approxEqual(expected));
    int[2] integral = [0, 1];
    float[2] floats = [0, 1];
    real[2] reals = [0, 1];
    static assert(is(typeof(weightedCovariance(integral, integral, integral)) == double));
    static assert(is(typeof(weightedCorrelation(floats, reals, integral)) == real));
    static assert(is(typeof(weightedCovariance!float(x, y, w)) == float));
}

// Hybrid handles single-pass cursors; explicit twoPass rejects them.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    struct Cursor
    {
        double[] values;
        size_t[] position;
        bool empty() const @property { return position[0] == values.length; }
        double front() const @property { return values[position[0]]; }
        void popFront() { ++position[0]; }
    }
    double[3] x = [1, 2, 4], y = [2, 7, 6], w = [1, 3, 2];
    size_t[1] ix, iy, iw;
    auto xr = Cursor(x[], ix[]), yr = Cursor(y[], iy[]), wr = Cursor(w[], iw[]);
    static assert(!__traits(compiles, weightedCovariance!(WeightedMomentNormalization.population,
        CovarianceAlgo.twoPass)(xr, yr, wr)));
    assert(weightedCovariance(xr, yr, wr).approxEqual(weightedCovariance(x, y, w)));
    assert(ix[0] == 3 && iy[0] == 3 && iw[0] == 3);
}

// Contracts reject invalid weights, values and shapes instead of truncating.
version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    static void rejects(alias operation)()
    {
        bool rejected;
        try { operation(); } catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    alias N = WeightedMomentNormalization;
    rejects!(() { WCovarianceAccumulator!double a; auto v = a.covariance; })();
    rejects!(() { WCorrelationAccumulator!double a; auto v = a.correlation; })();
    rejects!(() { WCovarianceAccumulator!double a; a.put(0, 1, -1); })();
    rejects!(() { WCovarianceAccumulator!double a; a.put(0, 1, double.nan); })();
    rejects!(() { WCovarianceAccumulator!double a; a.put(0, 1, double.infinity); })();
    rejects!(() { WCovarianceAccumulator!double a; a.put(double.nan, 1, 1); })();
    rejects!(() { WCorrelationAccumulator!double a; a.put(0, double.infinity, 1); })();
    rejects!(() { WCovarianceAccumulator!(double, N.frequencySample) a; a.put(0, 1, 1.5); })();
    rejects!(() { WCovarianceAccumulator!(float, N.frequencySample) a; a.put(0, 1, 0x1p24); a.put(1, 0, 1); })();
    rejects!(() { WCovarianceAccumulator!(double, N.frequencySample) a; a.put(0, 1, 1); auto v = a.covariance; })();
    rejects!(() { WCovarianceAccumulator!(double, N.relativeSample) a; a.put(0, 1, 1); auto v = a.covariance; })();
    rejects!(() { WCovarianceAccumulator!double a; double[2] x; double[1] w; a.put(x, x, w); })();
    rejects!(() { WCovarianceAccumulator!float a; a.put(0, 1, 1e-300); })();
}

// Same-scale merges preserve compensated state, including an aliased source.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (mode; [Summation.kbn, Summation.kb2, Summation.precise])
    static foreach (A; AliasSeq!(WCovarianceAccumulator!(double,
        WeightedMomentNormalization.population, CovarianceAlgo.online, mode),
        WCorrelationAccumulator!(double, CorrelationAlgo.online, mode)))
    {{
        A source, target;
        source.put(0, 0, 1);
        source.moments.scaleLeft = source.moments.scaleRight = 1;
        source.moments.cross.put(0x1p54);
        source.moments.cross.put(1.0);
        target.put(source);
        target.moments.cross.put(-0x1p54);
        assert(target.moments.cross.sum == 1);
        source.put(source);
        source.moments.cross.put(-0x1p55);
        assert(source.moments.cross.sum == 2);
    }}
}

// Restore observation units before a tiny normalized covariance rounds to zero.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    double[3] x = [0x1p500, 0, 0], y = [-0x1p500, 0, 0];
    double[3] weights = [0x1p-1074, 1, 1];
    // The tiny pair contributes -2^-74 to the cross moment; total weight is 2.
    enum expected = -0x1p-75;
    static foreach (algo; [CovarianceAlgo.online, CovarianceAlgo.twoPass, CovarianceAlgo.hybrid])
    {{
        assert((weightedCovariance!(WeightedMomentNormalization.population, algo)(x, y, weights))
            .approxEqual(expected, 1e-14, 0));
        WCovarianceAccumulator!(double, WeightedMomentNormalization.population, algo) left, right;
        left.put(x[0], y[0], weights[0]);
        right.put(x[1 .. $], y[1 .. $], weights[1 .. $]);
        left.put(right);
        assert(left.covariance.approxEqual(expected, 1e-14, 0));
    }}
}

// Exercise every position of a pointer-bearing static array among slices
// and forward ranges without weakening @safe lifetime checking.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import std.algorithm.iteration: filter;
    double[4] a = [2, 4, 6, 8], b = [3, 7, 8, 12], w = [1, 2, 3, 4];
    double[][2] x = [a[0 .. 2], a[2 .. 4]];
    double[][2] y = [b[0 .. 2], b[2 .. 4]];
    double[][2] weights = [w[0 .. 2], w[2 .. 4]];
    static foreach (xs; ["x", "a[].sliced(2, 2)", "a[].sliced(2, 2).filter!(row => true)"])
    static foreach (ys; ["y", "b[].sliced(2, 2)", "b[].sliced(2, 2).filter!(row => true)"])
    static foreach (ws; ["weights", "w[].sliced(2, 2)", "w[].sliced(2, 2).filter!(row => true)"])
    {
        assert(mixin("weightedCovariance(" ~ xs ~ ", " ~ ys ~ ", " ~ ws ~ ")").approxEqual(
            weightedCovariance(a[], b[], w[])));
        assert(mixin("weightedCorrelation(" ~ xs ~ ", " ~ ys ~ ", " ~ ws ~ ")").approxEqual(
            weightedCorrelation(a[], b[], w[])));
    }
    const cx = x;
    assert(weightedCovariance(cx, y, weights).approxEqual(weightedCovariance(a[], b[], w[])));
    assert(x[0].ptr == a.ptr && x[0].length == 2 && y[0].length == 2 && weights[0].length == 2);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.common: approxEqual;
    struct Cursor
    {
        double[] data;
        size_t[] index;
        @safe pure nothrow @nogc:
        @property bool empty() const { return index[0] == data.length; }
        @property double front() const { return data[index[0]]; }
        void popFront() { ++index[0]; }
    }
    double[3] x = [2, 4, 8], y = [3, 7, 9], w = [1, 2, 3];
    size_t[1] xi, yi, wi;
    const expected = weightedCovariance(x[], y[], w[]);
    assert(weightedCovariance(x, Cursor(y[], yi[]), Cursor(w[], wi[])).approxEqual(expected));
    assert(yi[0] == 3 && wi[0] == 3);
    yi[0] = wi[0] = 0;
    assert(weightedCovariance(Cursor(x[], xi[]), y, Cursor(w[], wi[])).approxEqual(expected));
    assert(xi[0] == 3 && wi[0] == 3);
    xi[0] = wi[0] = 0;
    assert(weightedCovariance(Cursor(x[], xi[]), Cursor(y[], yi[]), w).approxEqual(expected));
    assert(xi[0] == 3 && yi[0] == 3);
}

version(mir_stat_test)
unittest
{
    import std.algorithm.iteration: filter;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[2] x = [2, 4];
    double[3] w = [1, 2, 3];
    assertThrown!AssertError(weightedCovariance(x, x, w));
    static foreach (r; ["w[].filter!(v => true)", "w[0 .. 1].filter!(v => true)"])
    {
        assertThrown!AssertError(mixin("weightedCovariance(x, x, " ~ r ~ ")"));
        assertThrown!AssertError(mixin("weightedCovariance(x, " ~ r ~ ", x)"));
        assertThrown!AssertError(mixin("weightedCovariance(" ~ r ~ ", x, x)"));
    }
}
