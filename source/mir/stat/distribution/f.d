/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.f;

import mir.internal.utility: isFloatingPoint;

/++
Computes the F probability density function (PDF).

Params:
    x = value to evaluate PDF
    df1 = degrees of freedom parameter #1
    df2 = degrees of freedom parameter #2

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution)
+/
@safe pure nothrow @nogc
T fPDF(T)(const T x, const T df1, const T df2)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (df1 > 0, "df1 must be greater than zero")
    in (df2 > 0, "df2 must be greater than zero")
{
    import mir.math.common: exp, pow;
    import std.mathspecial: beta;
    const T a = df1 * T(0.5), b = df2 * T(0.5);
    // Use the direct beta-density form only while its intermediates remain
    // normal. Large shapes need the centered logarithmic calculation.
    if (x > 0 && x < T.infinity && a < 32 && b < 32)
    {
        const T ratio = df1 / df2;
        const T r = ratio * x;
        if (ratio >= T.min_normal && ratio < T.infinity && r >= T.min_normal && r < T.infinity)
        {
            const T numerator = pow(r, a - 1);
            const T denominator = pow(1 + r, a + b) * T(beta(a, b));
            const T scaledNumerator = numerator * ratio;
            const T result = scaledNumerator / denominator;
            if (numerator >= T.min_normal && numerator < T.infinity
                && scaledNumerator >= T.min_normal && scaledNumerator < T.infinity
                && denominator >= T.min_normal && denominator < T.infinity
                && result >= T.min_normal && result < T.infinity)
                return result;
        }
    }
    return exp(fLPDF(x, df1, df2));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.50.fPDF(1, 1).shouldApprox == 0.3001054;
    0.75.fPDF(1, 2).shouldApprox == 0.2532039;
    0.25.fPDF(0.5, 4).shouldApprox == 0.4904035;
    0.10.fPDF(2, 1).shouldApprox == 0.7607258;
    0.00.fPDF(1, 3).shouldApprox == double.infinity;
}

/++
Computes the F cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    df1 = degrees of freedom parameter #1
    df2 = degrees of freedom parameter #2

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution)
+/
@safe pure nothrow @nogc
T fCDF(T)(const T x, const T df1, const T df2)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (df1 > 0, "df1 must be greater than zero")
    in (df2 > 0, "df2 must be greater than zero")
{
    return fProbability!(T, false)(x, df1, df2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.50.fCDF(1, 1).shouldApprox == 0.3918266;
    0.75.fCDF(1, 2).shouldApprox == 0.522233;
    0.25.fCDF(0.5, 4).shouldApprox == 0.5183719;
    0.10.fCDF(2, 1).shouldApprox == 0.08712907;
}

/++
Computes the F complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    df1 = degrees of freedom parameter #1
    df2 = degrees of freedom parameter #2

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution)
+/
@safe pure nothrow @nogc
T fCCDF(T)(const T x, const T df1, const T df2)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (df1 > 0, "df1 must be greater than zero")
    in (df2 > 0, "df2 must be greater than zero")
{
    return fProbability!(T, true)(x, df1, df2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.50.fCCDF(1, 1).shouldApprox == 0.6081734;
    0.75.fCCDF(1, 2).shouldApprox == 0.477767;
    0.25.fCCDF(0.5, 4).shouldApprox == 0.4816281;
    0.10.fCCDF(2, 1).shouldApprox == 0.9128709;
}

/++
Computes the F inverse cumulative distribution function (InvCDF).

Params:
    p = value to evaluate InvCDF
    df1 = degrees of freedom parameter #1
    df2 = degrees of freedom parameter #2

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution)
+/
@safe pure nothrow @nogc
T fInvCDF(T)(const T p, const T df1, const T df2)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (df1 > 0, "df1 must be greater than zero")
    in (df2 > 0, "df2 must be greater than zero")
{
    import mir.stat.distribution.beta: betaInvCDF;
    import mir.math.common: log, exp;
    import mir.math.internal.log1p: log1p;

    if (p == 0)
        return 0;
    if (p == 1)
        return T.infinity;
    const T a = df1 * T(0.5), b = df2 * T(0.5);
    bool reflected = p > T(0.5);
    T q = reflected ? betaInvCDF(1-p, b, a) : betaInvCDF(p, a, b);
    if (q > T(0.5) && (reflected || 1-p < 1))
    {
        // Unequal shapes can put even a lower-half probability near one.
        // Compute the small complementary quantile instead of subtracting,
        // but only if forming 1-p preserves a probability below one.
        reflected = !reflected;
        q = reflected ? betaInvCDF(1-p, b, a) : betaInvCDF(p, a, b);
    }
    const T odds = reflected ? (1-q)/q : q/(1-q);
    const T ratio = df2 / df1;
    const T result = odds * ratio;
    if (odds >= T.min_normal && odds < T.infinity
        && ratio >= T.min_normal && ratio < T.infinity
        && result >= T.min_normal && result < T.infinity)
        return result;
    // Apply the degree ratio before a separate odds value over/underflows.
    const T logOdds = reflected ? log1p(-q)-log(q) : log(q)-log1p(-q);
    return exp(logOdds + log(df2) - log(df1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.3918266.fInvCDF(1, 1).shouldApprox == 0.50; 
    0.522233.fInvCDF(1, 2).shouldApprox == 0.75;
    0.5183719.fInvCDF(0.5, 4).shouldApprox == 0.25;
    0.08712907.fInvCDF(2, 1).shouldApprox == 0.10;
    0.0.fInvCDF(1, 1).shouldApprox == 0;
    1.0.fInvCDF(1, 1).shouldApprox == double.infinity;
}

/++
Computes the F log probability density function (LPDF).

Params:
    x = value to evaluate LPDF
    df1 = degrees of freedom parameter #1
    df2 = degrees of freedom parameter #2

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/F-distribution, F Distribution)
+/
@safe pure nothrow @nogc
T fLPDF(T)(const T x, const T df1, const T df2)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (df1 > 0, "df1 must be greater than zero")
    in (df2 > 0, "df2 must be greater than zero")
{
    import mir.math.common: log, exp, fabs;
    import mir.math.internal.log1p: log1p;
    import mir.math.internal.log_beta: logBeta;

    if (x == 0)
        return df1 < 2 ? T.infinity : df1 == 2 ? T(0) : -T.infinity;
    if (x == T.infinity)
        return -T.infinity;
    const T a = df1 * T(0.5), b = df2 * T(0.5);
    if (df1 == 2 && b > 0 && b < T.infinity)
    {
        const T ratio = x / b;
        if (ratio < T.min_normal)
            // Form the scale factor before multiplying a subnormal x;
            // x/b may round to zero even when its contribution matters.
            return -x * (1 + 1/b);
        if (ratio < T.infinity)
            return -(b + 1) * log1p(ratio);
        return -(b + 1) * (log(x) - log(b));
    }
    if (a >= 32 && b >= 32 && a < T.infinity && b < T.infinity)
        return fLargeShapeLPDF(x, a, b);

    // z=r/(1+r) is the beta argument, r=df1*x/df2. Evaluate both
    // logarithms independently: subtracting z from one loses the upper tail.
    const T degreeRatio = df1 / df2;
    const T r = degreeRatio * x;
    const bool usableRatio = degreeRatio >= T.min_normal && degreeRatio < T.infinity
        && r >= T.min_normal && r < T.infinity;
    const T logR = r >= T.min_normal && r < T.infinity
        ? log(r) : log(df1) - log(df2) + log(x);
    const T small = r >= T.min_normal && r < T.infinity
        ? (r <= 1 ? r : 1/r) : exp(-fabs(logR));
    const T correction = log1p(small);
    const T logZ = logR <= 0 ? logR - correction : -correction;
    const T logW = logR <= 0 ? -correction : -logR - correction;
    T normalization;
    const T smallShape = a < b ? a : b;
    const T largeShape = a < b ? b : a;
    if (largeShape >= 32 && largeShape < T.infinity)
    {
        // Only one shape is large here. Subtract its log-gamma terms
        // algebraically instead of forming two nearly equal large values.
        import std.mathspecial: logGamma;
        const T ratio = smallShape / largeShape;
        const T l = log1p(ratio);
        normalization = T(logGamma(smallShape)) - smallShape * log(largeShape)
            - (largeShape * l - smallShape) - (smallShape - T(0.5)) * l
            + fStirlingCorrection(1/largeShape)
            - fStirlingCorrection((1/largeShape)/(1+ratio));
    }
    else
        normalization = logBeta(a, b);
    return a * logZ + b * logW - log(x) - normalization;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    0.50.fLPDF(1, 1).shouldApprox == log(0.3001054);
    0.75.fLPDF(1, 2).shouldApprox == log(0.2532039);
    0.25.fLPDF(0.5, 4).shouldApprox == log(0.4904035);
    0.10.fLPDF(2, 1).shouldApprox == log(0.7607258);
    0.00.fLPDF(1, 3).shouldApprox == double.infinity;
}


// For large degrees, remove cancellation between the beta normalization and
// density terms by centering at x=1. Keep all arithmetic in the input type.
private @safe pure nothrow @nogc
T fLargeShapeLPDF(T)(const T x, const T a, const T b)
{
    pragma(inline, false);
    import mir.math.common: log, fabs;
    import mir.math.internal.log1p: log1p;
    import mir.stat.constant: LOGSQRT2PI;
    const T small = a < b ? a : b;
    const T large = a < b ? b : a;
    const T ratio = small / large;
    const T logSum = log1p(ratio);
    const T p = a <= b ? ratio / (1 + ratio) : 1 / (1 + ratio);
    const T q = a <= b ? 1 / (1 + ratio) : ratio / (1 + ratio);
    const T d = x - 1;
    const T denominator = 1 + p * d;
    const T za = q * d / denominator;
    const T zb = -p * d / denominator;
    T centered;
    if (fabs(za) <= T(0.5) && fabs(zb) <= T(0.5))
        centered = a * fLog1pmx(za) + b * fLog1pmx(zb);
    else
    {
        // Away from the center use logarithms of the odds, avoiding x*a.
        const T logR = log(a) - log(b) + log(x);
        import mir.math.common: exp;
        const T correction = log1p(exp(-fabs(logR)));
        const T logZ = logR <= 0 ? logR - correction : -correction;
        const T logW = logR <= 0 ? -correction : -logR - correction;
        const T logP = a <= b ? log(small) - log(large) - logSum : -logSum;
        const T logQ = a <= b ? -logSum : log(small) - log(large) - logSum;
        centered = large * ((a / large) * (logZ - logP)
            + (b / large) * (logW - logQ));
    }
    return centered - log(x) + T(0.5) * (log(small) - logSum)
        - T(LOGSQRT2PI) - fStirlingCorrection(1/a)
        - fStirlingCorrection(1/b) + fStirlingCorrection((1/large)/(1+ratio));
}

// log(1+z)-z without subtracting nearly equal values, for |z|<=1/2.
private @safe pure nothrow @nogc
T fLog1pmx(T)(const T z)
{
    const T t = z / (2 + z), t2 = t*t;
    enum terms = (T.mant_dig + 2) / 3 + 2;
    T polynomial = T(1) / (2*terms + 1);
    static foreach (j; 0 .. terms-1)
        polynomial = T(1) / (2*(terms-j-1)+1) + t2*polynomial;
    return -z*t + 2*t*t2*polynomial;
}

// Stirling correction for shape>=32; the omitted term is below 1e-24.
private @safe pure nothrow @nogc
T fStirlingCorrection(T)(const T r)
{
    const T r2 = r*r;
    return r*(T(1)/12 + r2*(-T(1)/360 + r2*(T(1)/1260
        + r2*(-T(1)/1680 + r2*(T(1)/1188 + r2*(-T(691)/360360 + r2/156))))));
}

// Density endpoints and references evaluated independently with 100-digit arithmetic.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (df; [T(0.5), T(1), T(2), T(3), T(100)])
        {
            const T expected = df < 2 ? T.infinity : df == 2 ? T(1) : T(0);
            assert(fPDF(T(0), df, T(4)) == expected);
            assert(fLPDF(T(0), df, T(4)) == (df < 2 ? T.infinity : df == 2 ? T(0) : -T.infinity));
            assert(fPDF(T.infinity, df, T(4)) == 0);
            assert(fLPDF(T.infinity, df, T(4)) == -T.infinity);
        }
        assert(approxEqual(fPDF(T(1), T(100), T(100)),
            T(1.989730934679469037453176256054261535L), T(32)*T.epsilon));
        assert(approxEqual(fLPDF(T(1), T(1e10), T(1e10)),
            T(9.900839751180610368892395415558068497L), T(16)*T.epsilon));
        assert(approxEqual(fLPDF(T(1), T(1), T(1e10)),
            T(-1.418938533254672741779496403072284307L), T(128)*T.epsilon));
        // The df1=2 density approaches one at zero; its log approaches zero.
        assert(approxEqual(fLPDF(T(1e-20L), T(2), T(4)), T(-1.5e-20L),
            T(16)*T.epsilon, T(0)));
        foreach (x; [T(0.01), T(0.5), T(1), T(2), T(100)])
            assert(approxEqual(fPDF(x, T(5), T(7)), exp(fLPDF(x, T(5), T(7))),
                T(256)*T.epsilon, T(0)));
    }}
    // Tail powers overflow in the direct expression, but the log density is finite.
    assert(approxEqual(fLPDF(1e300, 2.0, 2.0), -1381.551055796427410461721857,
        4e-15));
    assert(approxEqual(fPDF(1e100, 2.0, 2.0), 1e-200, 2e-13, 0.0));
}


// Calculate the smaller beta argument directly, retaining the upper tail
// even when df1*x overflows or the larger argument would round to one.
private @safe pure nothrow @nogc
T fProbability(T, bool complement)(const T x, const T df1, const T df2)
{
    import mir.stat.distribution.beta: betaCDF, betaCCDF;
    import mir.math.common: log, exp;
    if (x == 0)
        return complement ? T(1) : T(0);
    if (x == T.infinity)
        return complement ? T(0) : T(1);
    const T ratio = df1 / df2;
    T r = ratio*x;
    if (!(ratio >= T.min_normal && ratio < T.infinity
        && r >= T.min_normal && r < T.infinity))
        r = exp(log(df1) - log(df2) + log(x));
    const T a = df1*T(0.5), b = df2*T(0.5);
    if (r <= 1)
    {
        const T z = r/(1+r);
        static if (complement)
            return betaCCDF(z, a, b);
        else
            return betaCDF(z, a, b);
    }
    // Divide first: 1/(1+r) would discard a representable tail when r
    // overflows. The logarithmic fallback is needed only in that case.
    const T inverse = r < T.infinity ? 1/r : exp(log(df2)-log(df1)-log(x));
    const T z = inverse/(1+inverse);
    static if (complement)
        return betaCDF(z, b, a);
    else
        return betaCCDF(z, b, a);
}

// F(2,2) has CDF x/(1+x), survival 1/(1+x), and quantile p/(1-p).
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(fCDF(T(0), T(2), T(2)) == 0);
        assert(fCCDF(T(0), T(2), T(2)) == 1);
        assert(fCDF(T.infinity, T(2), T(2)) == 1);
        assert(fCCDF(T.infinity, T(2), T(2)) == 0);
        assert(approxEqual(fCCDF(T.max, T(2), T(2)), T(1)/T.max,
            T(256)*T.epsilon, T(0)));
        foreach (p; [T(0.001), T(0.25), T(0.5), T(0.75), T(0.999)])
            assert(approxEqual(fInvCDF(p, T(2), T(2)), p/(1-p), T(128)*T.epsilon, T(0)));
    }}
    assert(approxEqual(fCCDF(1e308, 4.0, 2.0), cast(double)1e-308L, 4e-13, 0.0));
    // Independent cot(pi*(1-p)/2)^2 reference at the exact binary input p.
    assert(approxEqual(fInvCDF(1-1e-10, 1.0, 1.0),
        40528466750254081239.61589107317028169, 2e-13));
    assert(approxEqual(fCCDF(40528466750254081239.61589107317028169, 1.0, 1.0),
        1.000000082740371e-10, 2e-13, 0.0));
}

// Shape-one beta identities check unequal degrees and quantile reflection.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: expm1, log, log1p;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (df; [T(0.5), T(1), T(4), T(20)])
        foreach (p; [T(0.01), T(0.25), T(0.75), T(0.99)])
        {
            const T shape = df*T(0.5);
            const T expectedLower = shape * expm1(-log1p(-p)/shape);
            const T expectedUpper = 1 / (shape * expm1(-log(p)/shape));
            assert(approxEqual(fInvCDF(p, T(2), df), expectedLower, T(512)*T.epsilon, T(0)));
            assert(approxEqual(fInvCDF(p, df, T(2)), expectedUpper, T(512)*T.epsilon, T(0)));
        }
    }}
    assert(approxEqual(fInvCDF(0.25, 20.0, 2.0),
        0.672502395887257562679242875427350516, 2e-14));
}

// A tiny lower-tail p must not be replaced by a complementary probability
// that rounds to one, even when its beta quantile is above one half.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import std.math: expm1, log;
    const double p = 1e-100;
    const double expected = 1 / (1000 * expm1(-log(p)/1000));
    assert(approxEqual(fInvCDF(p, 2000.0, 2.0), expected, 2e-14));
}
// These heavy-tail quantiles lie outside the double range; logarithmic
// odds scaling must preserve the appropriate overflow/underflow limits.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(fInvCDF(0.99, 1.0, 0.01) == double.infinity);
    assert(fInvCDF(0.01, 0.01, 1.0) == 0);
}

// Independent 100-digit references for density fallbacks away from the peak.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import std.math: nextUp;
    const double tiny = nextUp(0.0);
    assert(fLPDF(tiny, 2.0, 4.0) == -2*tiny);
    assert(approxEqual(fLPDF(tiny, 1.0, 1.0),
        371.075306074841230982910221871687758345, 4e-15));
    assert(approxEqual(fLPDF(1e308, 2.0, 0.5),
        -888.228128754107451634193619893926230543, 4e-15));
    assert(approxEqual(fLPDF(1e308, 1.0, 0.5),
        -888.325654114039056860965738764679150217, 4e-15));
    assert(approxEqual(fLPDF(0.01, 100.0, 64.0),
        -149.200721337238705029942855422958031302, 4e-15));
    assert(approxEqual(fLPDF(100.0, 64.0, 100.0),
        -158.411061709214888759771180495116846251, 4e-15));
    assert(approxEqual(fCDF(4.0, 2.0, 2.0), 0.8, 4e-15));
}
