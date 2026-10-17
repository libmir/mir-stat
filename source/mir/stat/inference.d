/++
This module contains statistical inference algorithms.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: Ilia Ki, John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/
module mir.stat.inference;

public import mir.stat.descriptive.univariate: KurtosisAlgo, Summation;
import mir.internal.utility: isFloatingPoint;

/++
Tests whether a sample departs from a normal distribution.

Requires at least eight observations and positive sample variance. The
chi-squared calibration is an approximation; results for small samples
should be interpreted cautiously.

Params:
    kurtosisAlgo = algorithm for calculating skewness and kurtosis (default: KurtosisAlgo.online)
    summation = algorithm for calculating sums (default: Summation.appropriate)
Returns:
    The nonnegative D’Agostino-Pearson statistic, the sum of the squared
    transformed skewness and kurtosis statistics. The p-value is written to
    $(D p); a small p-value is evidence against normality.
References:
    D’Agostino, R. B. (1971), “An omnibus test of normality for moderate and large sample size”, Biometrika, 58, 341-348;
    D’Agostino, R. and Pearson, E. S. (1973), “Tests for departure from normality”, Biometrika, 60, 613-622
+/
template dAgostinoPearsonTest(
    KurtosisAlgo kurtosisAlgo = KurtosisAlgo.online, 
    Summation summation = Summation.appropriate)
{
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
        p = output p-value under the normality null hypothesis
    +/
    F dAgostinoPearsonTest(Range, F)(Range r, out F p)
        if(isFloatingPoint!F && isIterable!Range)
    {
        import core.lifetime: move;
        import mir.stat.descriptive.univariate: KurtosisAccumulator;
        import mir.stat.distribution.chi2: chi2CCDF;
        import mir.math.sum: ResolveSummationType;

        KurtosisAccumulator!(F, kurtosisAlgo, ResolveSummationType!(summation, Range, F)) kurtosisAccumulator = r;
        auto kurtosisStat = kurtosisTestImpl!F(kurtosisAccumulator);

        auto skewnessStat = skewnessTestImpl!F(kurtosisAccumulator);
        auto stat = skewnessStat * skewnessStat + kurtosisStat * kurtosisStat;
        p = chi2CCDF(stat, 2);
        return stat;
    }
}

/// ditto
template dAgostinoPearsonTest(string kurtosisAlgo, string summation = "appropriate")
{
    mixin("alias dAgostinoPearsonTest = .dAgostinoPearsonTest!(KurtosisAlgo." ~ kurtosisAlgo ~ ", Summation." ~ summation ~ ");");
}

///
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    double p;
    x.dAgostinoPearsonTest(p).shouldApprox == 4.151936053369771;
    p.shouldApprox == 0.12543494432988342;

    p = p.nan;
    x.dAgostinoPearsonTest!"threePass"(p).shouldApprox == 4.151936053369771;
    p.shouldApprox == 0.12543494432988342;
}

private F skewnessTestImpl(F, Accumulator)(ref const Accumulator acc)
    if (isFloatingPoint!F)
{
    import mir.math.common: sqrt;
    auto b2 = acc.skewness!F(true);
    auto n = acc.count;
    assert(n > 7, "skewnessTestImpl: count must be larger than seven");
    const F invN = 1 / F(n);
    const F y = b2 * sqrt(F(n) / 6 * (1 + invN) * (1 + 3 * invN) / (1 - 2 * invN));
    // Compute beta2 - 3 directly as a polynomial in 1/n. Forming beta2
    // first loses this small positive correction for large sample counts.
    const F excess = 36 * invN * (1 + invN * (-5 + invN * (-19 + 35 * invN)))
        / ((1 - 2 * invN) * (1 + 5 * invN) * (1 + 7 * invN) * (1 + 9 * invN));
    // Rationalize sqrt(4 + 2 * excess) - 2 to retain w2 - 1.
    const F wm1 = 2 * excess / (sqrt(4 + 2 * excess) + 2);
    import mir.stat.internal.shape_transform: log1pScaled;
    const F delta = 1 / sqrt(F(0.5) * log1pScaled(wm1, F(1)));
    const F y_alpha = y * sqrt(wm1 / 2);
    import std.math: asinh;
    // The equivalent log expression loses precision for negative or tiny inputs.
    return delta * asinh(y_alpha);
}

// The skewness transformation is odd, including close to zero.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isFinite;

    static foreach (F; AliasSeq!(float, double, real))
    {{
        struct Sample
        {
            size_t count;
            F value;
            G skewness(G)(bool bias) const { return cast(G) value; }
        }

        foreach (value; [F(0), F.epsilon / 16, F(0.5), F(999)])
        {
            auto positive = Sample(1_000_000, value);
            auto negative = Sample(1_000_000, -value);
            auto a = skewnessTestImpl!F(positive);
            auto b = skewnessTestImpl!F(negative);
            assert(isFinite(a) && isFinite(b));
            assert(a == -b); // Only the sign changes in this transformation.
            if (value > 0)
                assert(a > 0);
            else
                assert(a == 0);
        }
    }}
}

// Reflecting observations changes skewness direction, not the normality statistic.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isFinite, abs;

    static foreach (F; AliasSeq!(float, double, real))
    {{
        auto values = new F[4096];
        values[] = 0;
        values[0] = 1;
        F positiveP, negativeP;
        auto positive = dAgostinoPearsonTest(values, positiveP);
        values[0] = -1;
        auto negative = dAgostinoPearsonTest(values, negativeP);
        assert(isFinite(positive) && isFinite(negative));
        assert(abs(positive - negative) <= 32 * F.epsilon * positive);
        assert(positiveP == negativeP);
    }}
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate: SkewnessAccumulator, SkewnessAlgo;
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];
    SkewnessAccumulator!(double, SkewnessAlgo.naive, Summation.naive) v = x;

    auto zsk = v.skewnessTestImpl!double;
    zsk.shouldApprox == 1.7985465327962042;
    import mir.stat.distribution.normal: normalCCDF;
    auto p = zsk.normalCCDF * 2;
    p.shouldApprox == 0.07209044155600682;
}

private F kurtosisTestImpl(F, Accumulator)(ref const Accumulator acc)
    if (isFloatingPoint!F)
{
    import mir.math.common: copysign, sqrt, fabs, pow;
    auto b2 = acc.kurtosis!F(true, true);
    auto n = acc.count;
    assert(n > 7, "kurtosisTestImpl: count must be larger than seven");
    const F invN = 1 / F(n);
    // Ratios in 1/n avoid both integer count-product overflow and large
    // floating-point polynomial intermediates. Keep the centering correction
    // separate so raw kurtosis close to three does not lose it.
    const F varb2 = 24 * invN * (1 - 2 * invN) * (1 - 3 * invN)
        / ((1 + invN) * (1 + invN) * (1 + 3 * invN) * (1 + 5 * invN));
    const F x = ((b2 - 3) + 6 * invN / (1 + invN)) / sqrt(varb2);
    const F beta1sqrt = 6 * (1 + invN * (-5 + 2 * invN)) / ((1 + 7 * invN) * (1 + 9 * invN))
        * sqrt(6 * invN * (1 + 3 * invN) * (1 + 5 * invN) / ((1 - 2 * invN) * (1 - 3 * invN)));
    const F a = 6 + 8 / beta1sqrt * (2 / beta1sqrt + sqrt(1 + 4 / (beta1sqrt * beta1sqrt)));
    const F u = x * sqrt(2 / (a - 4));
    const F denom = 1 + u;
    const F t2 = pow((1 - 2 / a) / denom.fabs, 1 / F(3)).copysign(denom);
    assert(denom);
    // Near t2 == 1, use 1 - t2 = (1 - t2^^3) / (1 + t2 + t2^^2).
    // Compute the numerator from u and 2/a rather than subtracting rounded
    // quantities. Away from this region retain the signed cube-root path.
    const F difference = fabs(u) < F(0.5)
        ? ((u + 2 / a) / denom) / (1 + t2 + t2 * t2) - 2 / (9 * a)
        : (1 - 2 / (9 * a)) - t2;
    return difference * sqrt(F(4.5) * a);
}

version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate: KurtosisAccumulator;
    import mir.math.common: approxEqual, pow;
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0];
    KurtosisAccumulator!(double, KurtosisAlgo.naive, Summation.naive) v = x;

    auto zku = v.kurtosisTestImpl!double;
    zku.shouldApprox == 0.9576880612895426;
    import mir.stat.distribution.normal: normalCCDF;
    auto p = zku.normalCCDF * 2;
    p.shouldApprox == 0.3382200786902009;
}

// Isolate transformation arithmetic from moment accumulation. References were
// evaluated with 100-digit Decimal arithmetic on exact count and moment inputs.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: abs;
    static foreach (F; AliasSeq!(float, double, real))
    {{
        struct Moments
        {
            size_t count;
            G skewness(G)(bool bias) const { return G(0.5); }
            G kurtosis(G)(bool bias, bool raw) const { return G(3); }
        }
        void check(size_t n, F expectedSkew, F expectedKurt)
        {
            auto moments = Moments(n);
            const skew = skewnessTestImpl!F(moments);
            const kurt = kurtosisTestImpl!F(moments);
            assert(abs(skew - expectedSkew) <= 32 * F.epsilon * abs(expectedSkew));
            assert(abs(kurt - expectedKurt) <= 32 * F.epsilon * abs(expectedKurt));
        }
        check(8, F(0.839544007798288332864968595651L), F(1.02215389244148716135048594706L));
        check(1000, F(6.15918764459358292939089196599L), F(0.114091666164366715029400696363L));
        check(1_000_000, F(193.136722952322031951390239522L), F(0.00367416480738591728012238201737L));
        check(100_000_000, F(1931.35076700742462039442236608L), F(0.000367423391607055735682500746800L));
        static if (size_t.sizeof >= 8)
        {
            check(5_000_000_000UL, F(13656.7110896711254488375137214L), F(0.0000519615240296125288184878377331L));
            check(100_000_000_000UL, F(61074.6685912467210489382092199L), F(0.0000116189500364146501493598175601L));
        }
    }}
}
