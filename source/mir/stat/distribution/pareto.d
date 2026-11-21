/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution).

These functions describe the type-I Pareto distribution. $(D xMin) is the
positive lower support boundary and $(D alpha) is the positive shape parameter.
The forward functions require $(D x >= xMin); they do not extend below the
support. The inverse CDF accepts both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.pareto;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityLess;
import mir.internal.utility: isFloatingPoint;

/++
Computes the pareto probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D alpha / xMin)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoPDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp, log;

    const T decay = alpha * paretoLogRatio(x, xMin);
    const T factor = alpha / x;
    const T tail = exp(-decay);
    // A normal tail cannot amplify rounding in a subnormal factor. Preserve
    // direct multiplication instead of introducing logarithmic rounding error.
    // Combine logarithms when the factor overflows or the tail loses range.
    if (factor <= T.max && tail >= T.min_normal)
        return factor * tail;
    return exp(log(alpha) - log(x) - decay);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoPDF(1, 3).shouldApprox == 3;
    2.0.paretoPDF(1, 3).shouldApprox == 0.1875;
    3.0.paretoPDF(2, 4).shouldApprox == 0.2633745;
}

/// The density starts at the lower support boundary and tends to zero.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(paretoPDF(2.0, 2.0, 4.0) == 2.0);
    assert(paretoPDF(double.infinity, 2.0, 4.0) == 0.0);
}

// Subnormal normalization factors do not require logarithmic rescaling.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp;
    import mir.math.common: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = ldexp(T(1), T.max_exp - 1);
        const T unit = T.min_normal * T.epsilon;
        // At the lower support boundary the density is exactly alpha / x.
        foreach (T alpha; [T(0.5), T(1), T(3)])
            assert(paretoPDF(x, x, alpha) == alpha / x);
        assert(paretoPDF(T.max, T.max, T(1)) == T(1) / T.max);
        // Preserve the smallest subnormal and round smaller factors to zero.
        assert(paretoPDF(T(1), T(1), unit) == unit);
        assert(paretoPDF(T(2), T(2), unit) == 0);
        assert(paretoPDF(T(4), T(1), unit) == 0);
        assert(paretoPDF(T(1), T(1), T.min_normal) == T.min_normal);
        // For alpha == 1 and xMin == x / 2, the density is 1 / (2*x).
        // Form the reference without overflowing 2*x.
        const T expected = ldexp(T(1), -T.max_exp);
        assert(fabs(paretoPDF(x, x / 2, T(1)) - expected) <= 2 * unit);
        assert(paretoPDF(T.infinity, x, T(1)) == 0);
    }}

    // Exact-input, 100-digit Decimal reference for xMin / (x*x).
    const double actual = paretoPDF(1e308, 5e307, 1.0);
    const double expected = cast(double) 4.999999999999999945104681852797723516e-309L;
    assert(fabs(actual - expected) <= 2 * double.min_normal * double.epsilon);
}

/++
Computes the type-I Pareto PDF using guarded extended-precision arithmetic.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation at or above xMin, including positive infinity.
    xMin = Positive finite lower support boundary.
    alpha = Positive finite shape parameter.

Returns:
    The density at x in type T. At xMin it is alpha/xMin. At infinity it is zero.
    Near rounding midpoints, integral shapes use exact comparisons when the
    odd density products fit within eight input significand widths.
+/
@safe pure nothrow @nogc
T paretoPDF(T)(const T x, const T xMin, const T alpha)
    if (is(T == Fp!size, size_t size))
    in (validParameter(xMin), "xMin must be finite and positive")
    in (validParameter(alpha), "alpha must be finite and positive")
    in (!x.isNaN && !x.sign && (x.isInfinity || !fpProbabilityLess(x, xMin)), "x must be at least xMin")
{
    import mir.stat.internal.fp_pareto: paretoDensity;
    return paretoDensity(x, xMin, alpha);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(paretoPDF(T(2), T(1), T(3)) == T(0.1875));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(paretoPDF(R(1.5), R(1.25), R(0.75)),
            R(0.43609797474671064824310932395303468578953669162L), 8 * R.epsilon));
    {
        alias T = Fp!64;
        enum expected = T(false, -65, UInt!64.fromHexString("df483bd6c725419b"));
        assert(paretoPDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    {
        alias T = Fp!128;
        enum expected = T(false, -129, UInt!128.fromHexString("df483bd6c725419ae17ecc4045b59416"));
        assert(paretoPDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    alias T = Fp!64;
    const least = T(false, long.min, UInt!64(1));
    assert(paretoPDF(least, least, least) == T(1));
    // At the support boundary this nonintegral shape gives exactly half
    // a quantum. The guarded quotient resolves the tie directly.
    const high = T(false, long.max - 1, UInt!64(12));
    assert(paretoPDF(high, high, T(1.5)) == T(0));
    assert(paretoPDF(T(4), T(2), T(1)) == T(0.125));
    assert(paretoPDF(T(2), T(1), T(false, 70, UInt!64(1))) == T(0));
    assert(paretoPDF(T(2), least, T(2)) == T(0));
    // For t = 2^-29, (1 + 3*t/2 + 3*t^2/8)^2 exceeds (1+t)^3
    // by 1/8*t^3 + 9/64*t^4. The density is just above half a quantum.
    enum m = 29;
    enum ulong boundaryCoefficient = (1UL << (2*m+3)) + (3UL << (m+2)) + 3;
    assert(paretoPDF(T(false, -m, UInt!64((1UL << m) + 1)),
        T(false, long.min/2 - 1 - (2*m+3), UInt!64(boundaryCoefficient)), T(2)) == least);
    assert(paretoPDF(least, least, T(false, long.max - 1, UInt!64.max)).isInfinity);
    assert(paretoPDF(T(double.infinity), T(1), T(1)) == T(0));
    // With xMin = 3*x/4 and shape two the density is 9/(8*x), exactly
    // half the least quantum here. Vary xMin by one represented step.
    const x = T(false, long.max - 1, UInt!64(9));
    auto boundary = T(false, long.max - 62, UInt!64(27) << 59);
    assert(paretoPDF(x, boundary, T(2)) == T(0));
    boundary.coefficient -= 1;
    assert(paretoPDF(x, boundary, T(2)) == T(0));
    boundary.coefficient += 2;
    assert(paretoPDF(x, boundary, T(2)) == least);
    // Three half-quanta round up to the even coefficient two.
    assert(paretoPDF(T(false, long.max - 1, UInt!64(3)),
        T(false, long.max - 3, UInt!64(9)), T(2)) == T(false, long.min, UInt!64(2)));
    // At ordinary exponents the exact density is c^2 * 2^-66.
    // The odd 65-bit square falls halfway above an even coefficient.
    enum c = 5000000001UL;
    assert(paretoPDF(T(2), T(false, -32, UInt!64(c)), T(2))
        == T(false, -65, UInt!64(12500000005000000000UL)));
}

/++
Computes the pareto cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoCDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;

    return oneMinusExpNeg(alpha * paretoLogRatio(x, xMin));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoCDF(1, 3).shouldApprox == 0;
    2.0.paretoCDF(1, 3).shouldApprox == 0.875;
    3.0.paretoCDF(2, 4).shouldApprox == 0.8024691;
}

/++
Computes the type-I Pareto CDF using guarded extended-precision arithmetic.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation at or above xMin, including positive infinity.
    xMin = Positive finite lower support boundary.
    alpha = Positive finite shape parameter.

Returns:
    The lower-tail probability in type T, ranging from zero at xMin to one at infinity.
+/
@safe pure nothrow @nogc
T paretoCDF(T)(const T x, const T xMin, const T alpha)
    if (is(T == Fp!size, size_t size))
    in (validParameter(xMin), "xMin must be finite and positive")
    in (validParameter(alpha), "alpha must be finite and positive")
    in (!x.isNaN && !x.sign && (x.isInfinity || !fpProbabilityLess(x, xMin)), "x must be at least xMin")
{
    import mir.stat.internal.fp_pareto: paretoTail;
    return paretoTail!false(x, xMin, alpha);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(paretoCDF(T(2), T(1), T(3)) == T(0.875));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(paretoCDF(R(1.5), R(1.25), R(0.75)),
            R(0.12780405050657870351378135209393062842092661676L), 8 * R.epsilon));
    {
        alias T = Fp!64;
        enum expected = T(false, -66, UInt!64.fromHexString("82df10a4e36af994"));
        assert(paretoCDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    {
        alias T = Fp!128;
        enum expected = T(false, -130, UInt!128.fromHexString("82df10a4e36af9947a04cefee929afa7"));
        assert(paretoCDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    alias T = Fp!64;
    assert(paretoCDF(T(1), T(1), T(2)) == T(0));
    assert(paretoCDF(T(double.infinity), T(1), T(2)) == T(1));
    auto adjacent = T(1);
    adjacent.coefficient += 1;
    // Rounded value of 1/(2^63+1) for the next represented input above one.
    assert(paretoCDF(adjacent, T(1), T(1)) == T(false, -127, UInt!64.max - 1));
    const least = T(false, long.min, UInt!64(1));
    assert(paretoCDF(T(2), T(1), least) == least);
}

/++
Computes the pareto complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoCCDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp;

    return exp(-alpha * paretoLogRatio(x, xMin));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoCCDF(1, 3).shouldApprox == 1;
    2.0.paretoCCDF(1, 3).shouldApprox == 0.125;
    3.0.paretoCCDF(2, 4).shouldApprox == 0.1975309;
}

/++
Computes the type-I Pareto CCDF using guarded extended-precision arithmetic.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation at or above xMin, including positive infinity.
    xMin = Positive finite lower support boundary.
    alpha = Positive finite shape parameter.

Returns:
    The probability of an observation strictly greater than x, in type T. It is one at xMin and zero at infinity.
+/
@safe pure nothrow @nogc
T paretoCCDF(T)(const T x, const T xMin, const T alpha)
    if (is(T == Fp!size, size_t size))
    in (validParameter(xMin), "xMin must be finite and positive")
    in (validParameter(alpha), "alpha must be finite and positive")
    in (!x.isNaN && !x.sign && (x.isInfinity || !fpProbabilityLess(x, xMin)), "x must be at least xMin")
{
    import mir.stat.internal.fp_pareto: paretoTail;
    return paretoTail!true(x, xMin, alpha);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(paretoCCDF(T(2), T(1), T(3)) == T(0.125));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(paretoCCDF(R(1.5), R(1.25), R(0.75)),
            R(0.87219594949342129648621864790606937157907338324L), 8 * R.epsilon));
    {
        alias T = Fp!64;
        enum expected = T(false, -64, UInt!64.fromHexString("df483bd6c725419b"));
        assert(paretoCCDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    {
        alias T = Fp!128;
        enum expected = T(false, -128, UInt!128.fromHexString("df483bd6c725419ae17ecc4045b59416"));
        assert(paretoCCDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    alias T = Fp!64;
    assert(paretoCCDF(T(1), T(1), T(2)) == T(1));
    assert(paretoCCDF(T(4), T(1), T(1)) == T(0.25));
    assert(paretoCCDF(T(2), T(1), T(false, 70, UInt!64(1))) == T(0));
    // The exponent of two is fractional on either side of one.
    assert(paretoCCDF(T(2), T(1), T(0.5)) == T(false, -64, UInt!64(0xb504f333f9de6484UL)));
    assert(paretoCCDF(T(2), T(1), T(1.5)) == T(false, -65, UInt!64(0xb504f333f9de6484UL)));
    assert(paretoCCDF(T(double.infinity), T(1), T(2)) == T(0));
    // A tail far below hardware range remains positive when the CDF rounds to one.
    const large = T(false, 40000, UInt!64(1));
    assert(paretoCDF(large, T(1), T(0.5)) == T(1));
    assert(paretoCCDF(large, T(1), T(0.5)) == T(false, -20063, UInt!64(1) << 63));
    // 2^(-2^63-1) is halfway between zero and the least positive Fp.
    const n = UInt!64(1) << 63;
    assert(paretoCCDF(T(2), T(1), T(false, 0, n + 1)) == T(0));
    assert(paretoCCDF(T(2), T(1), T(false, 0, n)) == T(false, long.min, UInt!64(1)));
    alias Wide = Fp!128;
    assert(paretoCCDF(Wide(2), Wide(1), Wide(false, 0, UInt!128(n + 1))) == Wide(0));
}

/++
Computes the pareto inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D xMin)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoInvCDF(T)(const T p, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp, log;
    import mir.stat.internal.neg_log1m: negLog1m;

    if (p == 0)
        return xMin;
    const T growth = negLog1m(p) / alpha;
    const T factor = exp(growth);
    if (factor <= T.max)
        return xMin * factor;
    // A small xMin can recover a finite quantile after exp(growth) overflows.
    return exp(log(xMin) + growth);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.paretoInvCDF(1, 3).shouldApprox == 1;
    0.875.paretoInvCDF(1, 3).shouldApprox == 2;
    0.8024691.paretoInvCDF(2, 4).shouldApprox == 3;
}

/++
Computes the pareto log probability density function (LPDF).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D log(alpha) - log(xMin))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoLPDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: log;

    const T factor = alpha / x;
    const T normalization = factor >= T.min_normal && factor <= T.max
        ? log(factor) : log(alpha) - log(x);
    return normalization - alpha * paretoLogRatio(x, xMin);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    1.0.paretoLPDF(1, 3).shouldApprox == log(paretoPDF(1.0, 1, 3));
    2.0.paretoLPDF(1, 3).shouldApprox == log(paretoPDF(2.0, 1, 3));
    3.0.paretoLPDF(2, 4).shouldApprox == log(paretoPDF(3.0, 2, 4));
}

/++
Computes the type-I Pareto LPDF using guarded extended-precision arithmetic.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation at or above xMin, including positive infinity.
    xMin = Positive finite lower support boundary.
    alpha = Positive finite shape parameter.

Returns:
    The natural logarithm of the density in type T. At infinity it is negative infinity.
    For severe cancellation at integral shapes, the calculation uses exact odd
    integer products of up to eight input significand widths. For other shapes
    or larger products, guarded subtraction can lose relative precision or leave
    a residual at an exact zero.
+/
@safe pure nothrow @nogc
T paretoLPDF(T)(const T x, const T xMin, const T alpha)
    if (is(T == Fp!size, size_t size))
    in (validParameter(xMin), "xMin must be finite and positive")
    in (validParameter(alpha), "alpha must be finite and positive")
    in (!x.isNaN && !x.sign && (x.isInfinity || !fpProbabilityLess(x, xMin)), "x must be at least xMin")
{
    import mir.stat.internal.fp_pareto: paretoLogDensity;
    return paretoLogDensity(x, xMin, alpha);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(paretoLPDF(T(1.125), T(0.84375), T(2)) == T(0));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(paretoLPDF(R(1.5), R(1.25), R(0.75)),
            R(-0.82988834815541127907602064032406254297354213780L), 8 * R.epsilon));
    {
        alias T = Fp!64;
        enum expected = T(true, -64, UInt!64.fromHexString("d4739012a8b12b02"));
        assert(paretoLPDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    {
        alias T = Fp!128;
        enum expected = T(true, -128, UInt!128.fromHexString("d4739012a8b12b01b4c7e8794552e7c4"));
        assert(paretoLPDF(T(1.5), T(1.25), T(0.75)) == expected);
    }
    alias T = Fp!64;
    assert(paretoLPDF(T(2), T(2), T(2)) == T(0));
    const limit = paretoLPDF(T(double.infinity), T(2), T(2));
    assert(limit.isInfinity && limit.sign);
    // alpha = 3, x = 3*(3/4)^3, xMin = (3/4)*x give density exactly one.
    assert(paretoLPDF(T(1.265625), T(0.94921875), T(3)) == T(0));
    alias Wide = Fp!128;
    assert(paretoLPDF(Wide(1.125), Wide(0.84375), Wide(2)) == Wide(0));
    // x^9 has 576 odd coefficient bits. Exercise the guarded fallback
    // at represented inputs just below eight.
    assert(paretoLPDF(T(false, -61, UInt!64.max), T(false, -61, UInt!64.max - 2), T(8))
        == T(true, -124, UInt!64(0xf000000000000002UL)));
    assert(paretoLPDF(T(false, -63, UInt!64(0x8fffffffffffffffUL)), T(0.84375), T(2))
        == T(false, -125, UInt!64(0xaaaaaaaaaaaaaaabUL)));
    assert(paretoLPDF(T(false, -63, UInt!64(0x9000000000000001UL)), T(0.84375), T(2))
        == T(true, -125, UInt!64(0xaaaaaaaaaaaaaaaaUL)));
}

// Near the lower bound, subtraction is exact. Evaluate log(x/xMin) as
// -log(1 - (x-xMin)/x) to preserve the small relative distance using the
// existing corrected logarithm. Farther away, use a representable ratio;
// separate logarithms avoid overflow across very different scales.
private @safe pure nothrow @nogc
T paretoLogRatio(T)(const T x, const T xMin)
{
    import mir.math.common: log;
    import mir.stat.internal.neg_log1m: negLog1m;
    const T distance = x - xMin;
    if (distance <= xMin)
        return negLog1m(distance / x);
    const T ratio = x / xMin;
    if (ratio <= T.max)
        return log(ratio);
    return log(x) - log(xMin);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp, log, sqrt;
    import std.math: nextDown, nextUp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(paretoPDF(T(2), T(2), T(4)) == 2);
        assert(paretoCDF(T(2), T(2), T(4)) == 0);
        assert(paretoCCDF(T(2), T(2), T(4)) == 1);
        assert(paretoInvCDF(T(0), T(2), T(4)) == 2);
        assert(paretoInvCDF(T(1), T(2), T(4)) == T.infinity);
        assert(paretoPDF(T.infinity, T(2), T(4)) == 0);
        assert(paretoCDF(T.infinity, T(2), T(4)) == 1);
        assert(paretoCCDF(T.infinity, T(2), T(4)) == 0);
        assert(paretoLPDF(T.infinity, T(2), T(4)) == -T.infinity);

        const T large = sqrt(T.max);
        assert(approxEqual(paretoPDF(large, large, T(4)), T(4)/large, T(8)*T.epsilon, T(0)));
        assert(approxEqual(paretoLPDF(T(2), T(2), large), log(large)-log(T(2)), T(8)*T.epsilon));
        // A tiny shape makes the CDF small even well away from the boundary.
        const T tiny = T.epsilon / 16;
        assert(approxEqual(paretoCDF(T(2), T(1), tiny), tiny*log(T(2)), T(8)*T.epsilon, T(0)));
        assert(approxEqual(paretoInvCDF(tiny, T(1), tiny), exp(T(1)), T(8)*T.epsilon));
        // A large shape amplifies errors in ratios rounded near one.
        assert(approxEqual(paretoCCDF(T(1)+T.epsilon, T(1), 1/T.epsilon),
            exp(T(-1)), T(8)*T.epsilon));
        assert(approxEqual(paretoLPDF(T(1), T(1), T(1)+T.epsilon),
            T.epsilon, T(8)*T.epsilon, T(0)));
        // Both logarithm paths agree around x/xMin == 2. For alpha == 1,
        // the probabilities have simple rational references.
        foreach (x; [nextDown(T(2)), T(2), nextUp(T(2))])
        {
            assert(approxEqual(paretoCDF(x, T(1), T(1)), (x-1)/x, T(8)*T.epsilon));
            assert(approxEqual(paretoCCDF(x, T(1), T(1)), 1/x, T(8)*T.epsilon));
        }
    }}
    // Independent Decimal references using exact binary double inputs and
    // 100 decimal digits of working precision.
    assert(approxEqual(paretoPDF(1e200, 1e200, 2.0), 2e-200, 4e-15, 0.0));
    assert(approxEqual(paretoCCDF(1e200, 1e-200, 0.5), 1e-200, 2e-13, 0.0));
    assert(approxEqual(paretoInvCDF(1e-20, 1.0, 1e-20),
        2.71828182845904523537, 4e-15, 0.0));
    assert(approxEqual(paretoLPDF(2.0, 2.0, 1e20),
        45.358554679320968371, 4e-15));
    assert(approxEqual(paretoPDF(cast(double)1e-310L, cast(double)1e-311L, 300.0),
        2999999999955.543257076365, 4e-13, 0.0));
    assert(approxEqual(paretoPDF(1e-200, 1e-201, 400.0),
        3.999999999999942054888553e-198, 4e-13, 0.0));
    assert(approxEqual(paretoInvCDF(0.5, 1e-300, 0.0008),
        1.938628441591554730157366e76, 4e-13, 0.0));
}

private bool validParameter(uint bits)(Fp!bits value) @safe pure nothrow @nogc
{
    return !value.isSpecial && !value.sign && !!value.coefficient;
}
