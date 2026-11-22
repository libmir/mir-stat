/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution).

The shape and scale parameters are positive, and the support is $(D x >= 0).
The default scale is one. Shape one gives an exponential distribution with
rate $(D 1 / scale). The inverse CDF accepts both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.weibull;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

/++
Computes the Weibull probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D shape < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D shape == 1)) $(TD $(D 1 / scale)))
    $(TR $(TD $(D x == 0), $(D shape > 1)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    shape = positive shape parameter
    scale = positive scale parameter (default: one)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution)
+/
@safe pure nothrow @nogc
T weibullPDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: exp;

    // Keep the exponential distribution's endpoint as a direct reciprocal.
    if (x == 0 && shape == 1)
        return T(1) / scale;
    const T logDensity = weibullLPDF(x, shape, scale);
    // Some exp implementations overflow before the largest finite result.
    // Near that limit, evaluate half the exponent and let multiplication
    // determine whether the density overflows. Keep ordinary densities direct.
    enum T upperLog = T(T.max_exp * 0.693147180559945309417232121458176568L);
    if (logDensity > upperLog - 2)
    {
        const T root = exp(logDensity / 2);
        return root * root;
    }
    return exp(logDensity);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.weibullPDF(3.0).shouldApprox == 0;
    0.5.weibullPDF(3.0).shouldApprox == 0.6618727;
    1.0.weibullPDF(3.0).shouldApprox == 1.103638;
    1.5.weibullPDF(3.0).shouldApprox == 0.2309723;

    // Can also provide scale parameter
    0.5.weibullPDF(2.0, 3.0).shouldApprox == 0.1080672;
    1.0.weibullPDF(2.0, 3.0).shouldApprox == 0.1988532;
    1.5.weibullPDF(2.0, 3.0).shouldApprox == 0.2596003;
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.weibullPDF(1.0, 3.0).shouldApprox == 0.3333333;
}

/// At zero, the shape determines whether the density is infinite, finite, or zero.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(weibullPDF(0.0, 0.5, 2.0) == double.infinity);
    assert(weibullPDF(0.0, 1.0, 2.0) == 0.5);
    assert(weibullPDF(0.0, 2.0, 2.0) == 0.0);
}

/++
Computes the Weibull density from represented inputs with guarded precision.
The density prefactor and survival probability retain separate exponents
until their product is rounded.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    shape = Positive finite shape parameter.
    scale = Positive finite scale parameter (default: one).

Returns:
    The density in type T. At zero it is positive infinity for shape below
    one, the reciprocal scale for shape one, and zero for shape above one.
    At positive infinity the density is zero. Finite densities may overflow.
+/
@safe pure nothrow @nogc
T weibullPDF(T)(const T x, const T shape, const T scale = T(1))
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)), "x must be nonnegative")
    in (validParameter(shape), "shape must be finite and positive")
    in (validParameter(scale), "scale must be finite and positive")
{
    import mir.stat.internal.fp_weibull: weibullDensity;
    return weibullDensity(x, shape, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(weibullPDF(T(0), T(1), T(4)) == T(0.25));
    assert(weibullPDF(T(0), T(2)) == T(0));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    alias T = Fp!64;
    alias Wide = Fp!128;
    enum expected = Wide(false, -133, UInt!128.fromHexString("9dad55b76d40097a8f202e6d50714fdf"));
    assert(weibullPDF(Wide(9), Wide(1.5), Wide(4)) == expected);
    assert(approxEqual(weibullPDF(9.0, 1.5, 4.0), 0.019247691550312144518, 8 * double.epsilon));
    assert(weibullPDF(T(0), T(0.5)).isInfinity);
    assert(weibullPDF(T(double.infinity), T(0.5)) == T(0));
    assert(weibullPDF(T(2), T(128)) == T(0));
    auto largeShape = T(1);
    largeShape.exponent += 100;
    assert(weibullPDF(T(0.5), largeShape) == T(0));

    // exp(-2^63) rounds to zero alone; dividing by scale recovers the density.
    const scale = T(false, long.min + 256, UInt!64(1));
    const x = T(false, long.min + 319, UInt!64(1));
    assert(weibullCCDF(x, T(1), scale) == T(0));
    assert(weibullPDF(x, T(1), scale) ==
        T(false, -4083141060989547003L, UInt!64(0x98d3377e65eee9c8UL)));
}

/++
Computes the Weibull cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    shape = positive shape parameter
    scale = positive scale parameter (default: one)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution)
+/
@safe pure nothrow @nogc
T weibullCDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: pow;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;

    return oneMinusExpNeg(pow(x / scale, shape));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.weibullCDF(3.0).shouldApprox == 0;
    0.5.weibullCDF(3.0).shouldApprox == 0.1175031;
    1.0.weibullCDF(3.0).shouldApprox == 0.6321206;
    1.5.weibullCDF(3.0).shouldApprox == 0.9657819;

    // Can also provide scale parameter
    0.5.weibullCDF(2.0, 3.0).shouldApprox == 0.02739552;
    1.0.weibullCDF(2.0, 3.0).shouldApprox == 0.1051607;
    1.5.weibullCDF(2.0, 3.0).shouldApprox == 0.2211992;
}

/++
Computes the Weibull lower-tail probability with guarded extended precision.

Very small lower tails retain the scaled cumulative hazard through final rounding.
Rounding near a midpoint is not certified for general shapes.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    shape = Positive finite shape parameter.
    scale = Positive finite scale parameter (default: one).

Returns:
    $(D P(X <= x)) in type T, with value zero at zero and one at positive infinity.
+/
@safe pure nothrow @nogc
T weibullCDF(T)(const T x, const T shape, const T scale = T(1))
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)), "x must be nonnegative")
    in (validParameter(shape), "shape must be finite and positive")
    in (validParameter(scale), "scale must be finite and positive")
{
    import mir.stat.internal.fp_weibull: weibullTail;
    return weibullTail!false(x, shape, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(weibullCDF(T(0), T(2)) == T(0));
    assert(weibullCDF(T(double.infinity), T(2), T(3)) == T(1));
}

/++
Computes the Weibull complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    shape = positive shape parameter
    scale = positive scale parameter (default: one)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution)
+/
@safe pure nothrow @nogc
T weibullCCDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: exp, pow;

    return exp(-pow(x / scale, shape));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.weibullCCDF(3.0).shouldApprox == 1;
    0.5.weibullCCDF(3.0).shouldApprox == 0.8824969;
    1.0.weibullCCDF(3.0).shouldApprox == 0.3678794;
    1.5.weibullCCDF(3.0).shouldApprox == 0.03421812;

    // Can also provide scale parameter
    0.5.weibullCCDF(2.0, 3.0).shouldApprox == 0.9726045;
    1.0.weibullCCDF(2.0, 3.0).shouldApprox == 0.8948393;
    1.5.weibullCCDF(2.0, 3.0).shouldApprox == 0.7788008;
}

/++
Computes the Weibull strict upper-tail probability with guarded extended precision.

The survival probability is evaluated directly, including when the lower tail rounds to one.
Rounding near a midpoint is not certified for general shapes.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    shape = Positive finite shape parameter.
    scale = Positive finite scale parameter (default: one).

Returns:
    $(D P(X > x)) in type T, with value one at zero and zero at positive infinity.
+/
@safe pure nothrow @nogc
T weibullCCDF(T)(const T x, const T shape, const T scale = T(1))
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)), "x must be nonnegative")
    in (validParameter(shape), "shape must be finite and positive")
    in (validParameter(scale), "scale must be finite and positive")
{
    import mir.stat.internal.fp_weibull: weibullTail;
    return weibullTail!true(x, shape, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(weibullCCDF(T(0), T(2)) == T(1));
    assert(weibullCCDF(T(double.infinity), T(2), T(3)) == T(0));
}

// Shared dyadic inputs: the cumulative hazard is exactly 27/8.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;

    static foreach (R; AliasSeq!(float, double, real))
    {
        assert(approxEqual(weibullCDF(R(9), R(1.5), R(4)),
            R(0.9657818816883339653014449405563691854921516024381881357L), 8 * R.epsilon));
        assert(approxEqual(weibullCCDF(R(9), R(1.5), R(4)),
            R(0.03421811831166603469855505944363081450784839756181186426L), 8 * R.epsilon));
    }
    alias T = Fp!128;
    enum expectedCDF = T(false, -128, UInt!128.fromHexString("f73d7b3cebb554ce864552c109c0d0e5"));
    assert(weibullCDF(T(9), T(1.5), T(4)) == expectedCDF);
    enum expectedCCDF = T(false, -132, UInt!128.fromHexString("8c284c3144aab3179baad3ef63f2f1aa"));
    assert(weibullCCDF(T(9), T(1.5), T(4)) == expectedCCDF);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    // exp(-16384) remains representable after the CDF rounds to one.
    assert(weibullCDF(T(128), T(2)) == T(1));
    assert(weibullCCDF(T(128), T(2)) == T(false, -23701, U(0xec4bb8768ab8ab91UL)));
    auto tiny = T(1);
    tiny.exponent -= 40_000;
    auto expected = T(1);
    expected.exponent -= 20_000;
    assert(weibullCDF(tiny, T(0.5)) == expected);
    assert(weibullCCDF(tiny, T(0.5)) == T(1));
    const x = T(false, long.min, U(7));
    assert(weibullCDF(x, T(1), T(2)) == T(false, long.min, U(3)));
    // The cubed observation is exactly 62.5 least quanta; the CDF is below it.
    const cubeRoot = T(false, long.min / 3 - 1, U(5));
    assert(weibullCDF(cubeRoot, T(3)) == T(false, long.min, U(62)));

    // Adjacent observations on either side of the same represented scale.
    auto scale = T(false, 40_000, U(0x8000000000000013UL));
    auto below = scale, above = scale;
    below.coefficient -= 1;
    above.coefficient += 1;
    const shape = T(1UL << 63);
    assert(weibullCDF(below, shape, scale) == T(false, -65, U(0x9d97e11cc21ad5aeUL)));
    assert(weibullCCDF(above, shape, scale) == T(false, -67, U(0x8724bc3f102a3d05UL)));

    // The ratio 2^(2^63) lies outside Fp, but the cumulative hazard is two.
    auto high = T(1), low = T(1), smallShape = T(1);
    high.exponent += 1L << 62;
    low.exponent -= 1L << 62;
    smallShape.exponent -= 63;
    assert(weibullCDF(high, smallShape, low) == weibullCDF(T(2), T(1)));
    assert(weibullCCDF(high, smallShape, low) == weibullCCDF(T(2), T(1)));

    // The sign of log(x/scale) determines which tail vanishes.
    assert(weibullCDF(T(2), high) == T(1));
    assert(weibullCCDF(T(2), high) == T(0));
    assert(weibullCDF(T(1), high, T(2)) == T(0));
    assert(weibullCCDF(T(1), high, T(2)) == T(1));

    // Integral shapes on both sides of the bounded-power budget.
    assert(weibullCDF(T(2), T(257), T(2)) == weibullCDF(T(1), T(1)));
    assert(weibullCDF(T(2), T(128)) == T(1));
    assert(weibullCCDF(T(2), T(128)) == T(0));

    import mir.stat.distribution.exponential: exponentialCDF, exponentialCCDF;
    assert(weibullCDF(T(3), T(1), T(4)) == exponentialCDF(T(3), T(0.25)));
    assert(weibullCCDF(T(3), T(1), T(4)) == exponentialCCDF(T(3), T(0.25)));
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!32;
    const shape = T(false, -1, UInt!32(3));
    assert(weibullCDF(T(9), shape, T(4)) == T(false, -32, UInt!32(0xf73d7b3dU)));
    assert(weibullCCDF(T(9), shape, T(4)) == T(false, -36, UInt!32(0x8c284c31U)));
}

/++
Computes the Weibull inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    shape = positive shape parameter
    scale = positive scale parameter (default: one)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution)
+/
@safe pure nothrow @nogc
T weibullInvCDF(T)(const T p, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: pow;
    import mir.stat.internal.neg_log1m: negLog1m;

    return scale * pow(negLog1m(p), T(1) / shape);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    weibullInvCDF(0.0, 3).shouldApprox == 0.0;
    weibullInvCDF(0.25, 3).shouldApprox == 0.6601424;
    weibullInvCDF(0.5, 3).shouldApprox == 0.884997;
    weibullInvCDF(0.75, 3).shouldApprox == 1.115026;
    weibullInvCDF(1.0, 3).shouldApprox == double.infinity;

    // Can also provide scale parameter
    weibullInvCDF(0.2, 2, 3).shouldApprox == 1.417142;
    weibullInvCDF(0.4, 2, 3).shouldApprox == 2.144162;
    weibullInvCDF(0.6, 2, 3).shouldApprox == 2.871692;
    weibullInvCDF(0.8, 2, 3).shouldApprox == 3.805909;
}


/++
Computes the Weibull log probability density function (LPDF).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0), $(D shape < 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D x == 0), $(D shape == 1)) $(TD $(D -log(scale))))
    $(TR $(TD $(D x == 0), $(D shape > 1)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    shape = positive shape parameter
    scale = positive scale parameter (default: one)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Weibull_distribution, Weibull Distribution)
+/
@safe pure nothrow @nogc
T weibullLPDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log, exp, pow;

    if (x == 0)
        return shape < 1 ? T.infinity : shape == 1 ? -log(scale) : -T.infinity;
    if (x == T.infinity)
        return -T.infinity;

    const T ratio = x / scale;
    // Shape one is exponential. Avoid exp(log(ratio)), which can overflow
    // near T.max even when the ratio itself is representable.
    if (shape == 1)
        return -log(scale) - ratio;

    T logRatio;
    if (ratio >= T(0.5) && ratio <= 2)
    {
        // Near the scale, subtract first: rounding x / scale before taking
        // its logarithm loses differences that a large shape can amplify.
        const T delta = (x - scale) / scale;
        const T u = 1 + delta;
        // Correct log(u) for rounding in 1 + delta. If that addition rounds
        // to one, log(1 + delta) rounds to delta at this precision.
        logRatio = u == 1 ? delta : log(u) * (delta / (u - 1));
    }
    else if (ratio >= T.min_normal && ratio <= T.max)
        logRatio = log(ratio);
    else
        // Separate logarithms retain range when division overflows or loses
        // precision to subnormal rounding (including rounding to zero).
        logRatio = log(x) - log(scale);

    const T logPower = shape * logRatio;
    if (logPower == T.infinity)
        return -T.infinity;

    const T normalization = shape / x;
    const T logNormalization = normalization >= T.min_normal && normalization <= T.max
        ? log(normalization) : log(shape) - log(x);
    // With t = (x / scale)^shape, density is (shape / x) * t * exp(-t).
    // Combine its logarithms before exponentiating in the PDF: individual
    // factors may overflow or underflow although the final density is finite.

    // exp(logPower) can overflow prematurely near the largest finite power.
    // Use a small neighborhood of max_exp * log(2); the margin of two log
    // units keeps rounding of the threshold away from the overflow boundary.
    enum T upperLog = T(T.max_exp * 0.693147180559945309417232121458176568L);
    T power;
    if (logPower > upperLog - 2)
    {
        if (logPower > upperLog + 2)
            power = T.infinity;
        else if (ratio > 2)
            // Direct powers avoid rounding through log and exp. If division
            // overflowed, a representable power requires shape < 1. Taking
            // each power first retains the intermediate range in that case.
            power = ratio <= T.max ? pow(ratio, shape) : pow(x, shape) / pow(scale, shape);
        else
        {
            // Retain the corrected logarithm near the scale: a rounded ratio
            // loses information for large shapes. Splitting the exponential
            // also avoids premature overflow in extended-real exp implementations.
            const T root = exp(logPower / 2);
            power = root * root;
        }
    }
    else
        power = exp(logPower);
    return logNormalization + logPower - power;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    0.0.weibullLPDF(3.0).shouldApprox == log(0.0);
    0.5.weibullLPDF(3.0).shouldApprox == log(0.6618727);
    1.0.weibullLPDF(3.0).shouldApprox == log(1.103638);
    1.5.weibullLPDF(3.0).shouldApprox == log(0.2309723);

    // Can also provide scale parameter
    0.5.weibullLPDF(2.0, 3.0).shouldApprox == log(0.1080672);
    1.0.weibullLPDF(2.0, 3.0).shouldApprox == log(0.1988532);
    1.5.weibullLPDF(2.0, 3.0).shouldApprox == log(0.2596003);
}

// Density limits at zero and infinity, including the exponential special case.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: log;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(weibullPDF(T(0), T(0.5), T(3)) == T.infinity);
        assert(weibullLPDF(T(0), T(0.5), T(3)) == T.infinity);
        assert(weibullPDF(T(0), T(1), T(3)) == T(1) / 3);
        assert(weibullLPDF(T(0), T(1), T(3)) == -log(T(3)));
        assert(weibullPDF(T(0), T(2), T(3)) is T(0));
        assert(weibullLPDF(T(0), T(2), T(3)) == -T.infinity);
        foreach (shape; [T(0.5), T(1), T(2)])
        {
            assert(weibullPDF(T.infinity, shape, T(3)) is T(0));
            assert(weibullLPDF(T.infinity, shape, T(3)) == -T.infinity);
        }
        assert(weibullLPDF(T.max, T(1)) == -T.max);
        assert(weibullLPDF(T.max, T.max) == -T.infinity);
        assert(weibullPDF(T.max, T.max) is T(0));
    }}
}

// Preserve finite densities when ratios or individual density factors lose range.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, log;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T small = T.min_normal;
        const T large = T(1) / small;
        // Taking logarithms of extreme scales introduces error proportional
        // to their logarithm; exponentiating transfers that to relative error.
        const T tolerance = -8 * log(small) * T.epsilon;
        assert(approxEqual(weibullPDF(small, T(0.5), large), T(0.5), tolerance, T(0)));
        assert(approxEqual(weibullLPDF(small, T(0.5), large), -log(T(2)), tolerance, T(0)));
        // At x == scale, density * scale is shape / e. The normalization
        // shape / scale overflows here, but the final density does not.
        assert(approxEqual(weibullPDF(small, T(4), small) * small,
            T(1.47151776468576928638209508064584347L), tolerance, T(0)));
        // Underflow of shape / scale must not discard a finite log-density.
        assert(approxEqual(weibullLPDF(large, small, large), 2 * log(small) - 1,
            8 * T.epsilon, T(0)));
    }}

    // Independent 100-digit references for the represented double inputs.
    assert(approxEqual(weibullPDF(1e200, 0.001, 1e-200),
        2.03751860765396894e-204, 2e-12, 0.0));
    assert(approxEqual(weibullLPDF(1e200, 0.001, 1e-200),
        -469.015626272103236, 8 * double.epsilon, 0.0));
}

// Preserve adjacent observations near the scale when a large shape amplifies them.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    import mir.math.common: approxEqual, log;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T scale = T(1.5);
        const T shape = T(1) / T.epsilon;
        // As epsilon tends to zero, shape * log(x / scale) tends to +/-2/3.
        // The finite-epsilon correction is O(epsilon); these constants were
        // evaluated independently at 100 decimal digits. The log-density
        // includes log(shape), whose rounding becomes relative PDF error.
        const T tolerance = 4 * log(shape) * T.epsilon;
        assert(approxEqual(weibullPDF(nextUp(scale), shape, scale) / shape,
            T(0.185160459257156549999066633827412023L), tolerance, T(0)));
        assert(approxEqual(weibullPDF(nextDown(scale), shape, scale) / shape,
            T(0.204835329410450086687242089977651399L), tolerance, T(0)));
    }}
}

// Keep representable log-densities near the upper power boundary finite.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, isFinite;
    import mir.math.common: approxEqual, sqrt;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = nextDown(sqrt(T.max));
        assert(isFinite(weibullLPDF(x, T(2))));
        // x squared is within a few ulps of T.max; the logarithmic terms
        // are negligible on this scale. Normalize to avoid an overflowing test.
        assert(approxEqual(weibullLPDF(x, T(2)) / T.max, T(-1), 8 * T.epsilon, T(0)));
        assert(weibullPDF(x, T(2)) is T(0));

        // The ratio overflows, but its square root is approximately T.max / 4.
        assert(approxEqual(weibullLPDF(T.max / 4, T(0.5), T(4) / T.max) / T.max,
            T(-0.25), 16 * T.epsilon, T(0)));

        // At ratio two, the power is exactly 2^(max_exp - 2). The split
        // exponential uses logarithms, so its relative error scales with max_exp.
        assert(approxEqual(weibullLPDF(T(2), T(T.max_exp - 2)) / T.max,
            T(-0.25), 8 * T.max_exp * T.epsilon, T(0)));
        assert(weibullLPDF(T.max, T(2)) == -T.infinity);
    }}

    static if (real.mant_dig == 64 && real.max_exp == 16384)
    {
        // Independent 110-digit reference for these exact hexadecimal inputs.
        // An unsplit extended-real exp can overflow even this far below max.
        assert(approxEqual(weibullLPDF(0x1.99999999999a278ep-4L, 0x1p63L,
            0x1.999999999999999ap-4L) / real.max,
            -0.76078362289116266064109484385712350574L,
            8 * real.max_exp * real.epsilon, 0.0L));
    }
}

// Preserve large finite PDFs when the final exponential is near overflow.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, isFinite;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T scale = T(1.0625) / T.max;
        const T x = scale / 16;
        const T density = weibullPDF(x, T(1), scale);
        assert(isFinite(density));
        // Here x / scale is exactly 1/16. The limiting normalized density
        // (16/17)*exp(-1/16) was evaluated independently at 110 decimal digits;
        // rounding of the scale and T.max changes it only by O(epsilon).
        // The log/exp calculation contributes error proportional to max_exp.
        assert(approxEqual(density / T.max,
            T(0.88415347088327132811266901140922831485L),
            8 * T.max_exp * T.epsilon, T(0)));

        // A true overflow must remain infinite rather than being clamped.
        const T tiny = nextUp(T(0));
        assert(weibullPDF(tiny, T(1), tiny) == T.infinity);
    }}
}

// Preserve inverse-CDF values below machine epsilon for linear and square roots.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    import mir.math.common: approxEqual, sqrt;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (p; [nextUp(T(0)), T.epsilon / 16, T.epsilon])
        {
            assert(approxEqual(weibullInvCDF(p, T(1)) / p, T(1), 4 * T.epsilon, T(0)));
            assert(approxEqual(weibullInvCDF(p, T(2), T(3)) / (3 * sqrt(p)), T(1),
                8 * T.epsilon, T(0)));
        }
        assert(weibullInvCDF(T(0), T(2)) == 0);
        assert(weibullInvCDF(T(1), T(2)) == T.infinity);
    }}
}

// Preserve lower-tail CDF probabilities and round trips below machine epsilon.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T p = T.epsilon / 16;
        foreach (shape; [T(1), T(2)])
        {
            const T x = weibullInvCDF(p, shape, T(3));
            assert(x > 0);
            assert(approxEqual(weibullCDF(x, shape, T(3)) / p, T(1),
                32 * T.epsilon, T(0)));
        }
        assert(approxEqual(weibullCDF(p, T(1)) / p, T(1), 8 * T.epsilon, T(0)));
        const T smallest = nextUp(T(0));
        assert(weibullCDF(smallest, T(1)) == smallest);
        assert(weibullCDF(T(0), T(2)) == 0);
        assert(weibullCDF(T.infinity, T(2)) == 1);
    }}
}

private bool validParameter(uint bits)(Fp!bits value) @safe pure nothrow @nogc
{
    return !value.isSpecial && !value.sign && !!value.coefficient;
}
