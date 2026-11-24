/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution).

The parameters describe the normal variable $(D log(x)): $(D mean) is its
mean and $(D stdDev) is its positive standard deviation. They are not the
mean and standard deviation of $(D x). The one-argument overloads use zero
and one, respectively. Observations must be nonnegative; zero is accepted
as a limiting endpoint. The inverse CDF accepts both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

Reference tests exercise $(D Fp) significand widths of 32, 64, 128, and
256 bits. The 32-bit significand is tested on 32-bit targets.

The $(D Fp) overloads use guarded significands and separately scaled
intermediates. Rounding to the requested width occurs at the result.
Universal correct rounding is unproven. If cancellation exhausts the
working precision, a log density can evaluate to zero, or a transformed
observation can coincide with the mean at that precision. If the log
quantile cancels to zero at working precision, the inverse returns one.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.log_normal;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid;
import mir.internal.utility: isFloatingPoint;

/++
Computes the Log-normal probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

This overload uses mean zero and standard deviation one for $(D log(x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution)
+/
@safe pure nothrow @nogc
T logNormalPDF(T)(const T x)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
{
    import mir.math.common: log, exp;
    import mir.stat.distribution.normal: normalPDF, normalLPDF;
    if (x == 0 || x == T.infinity) return 0;
    const T logX = log(x);
    const T density = normalPDF(logX);
    // A rounded-away normal density can become representable after dividing
    // by a small x. Combine the logarithms before the final exponential.
    if (density < T.min_normal)
        return exp(normalLPDF(logX) - logX);
    return density / x;
}

/++
Computes the standard log-normal density with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.

Returns:
    The density in type T. Zero at zero and positive infinity.
+/
@safe pure nothrow @nogc
T logNormalPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
{
    return logNormalPDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalPDF(T(0)) == T(0));
}

/++
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Returns a density, which may exceed one; it is not a point probability.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    mean = finite mean of log(x)
    stdDev = positive finite standard deviation of log(x)
+/
@safe pure nothrow @nogc
T logNormalPDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: log, exp;
    import mir.stat.distribution.normal: normalPDF, normalLPDF;
    if (x == 0 || x == T.infinity) return 0;
    const T logX = log(x);
    const T density = normalPDF(logX, mean, stdDev);
    // Recover both underflow before division by x and overflow before
    // division by a large x. Ordinary densities retain the direct path.
    if (density < T.min_normal || density == T.infinity)
        return exp(normalLPDF(logX, mean, stdDev) - logX);
    return density / x;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalPDF(1.0).shouldApprox == 0.3989423;
    logNormalPDF(2.0).shouldApprox == 0.156874;
    logNormalPDF(3.0).shouldApprox == 0.07272826;

    // Can include location/scale
    logNormalPDF(1.0, 1, 2).shouldApprox == 0.1760327;
    logNormalPDF(2.0, 1, 2).shouldApprox == 0.09856858;
    logNormalPDF(3.0, 1, 2).shouldApprox == 0.06640961;
}

// check zero or near zero
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalPDF(0.0).shouldApprox == 0.0;
    logNormalPDF(double.min_normal * double.epsilon).shouldApprox == 0.0;
    logNormalPDF(double.min_normal).shouldApprox == 0.0;
    
    logNormalPDF(0.0, 1, 2).shouldApprox == 0.0;
    logNormalPDF(double.min_normal * double.epsilon, 1, 2).shouldApprox == 0.0;
    logNormalPDF(double.min_normal, 1, 2).shouldApprox == 0.0;
}

/// Zero and positive infinity are limiting endpoints with zero density.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(logNormalPDF(0.0) == 0);
    assert(logNormalPDF(double.infinity, 2.0, 3.0) == 0);
}

/++
Computes the location-scale log-normal density with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    mean = Finite mean of the underlying normal variable.
    stdDev = Positive finite standard deviation of the underlying normal variable.

Returns:
    The density in type T. Zero at zero and positive infinity.
+/
@safe pure nothrow @nogc
T logNormalPDF(T)(const T x, const T mean, const T stdDev)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
    in (!mean.isSpecial && !stdDev.isSpecial && !stdDev.sign && !!stdDev.coefficient)
{
    import mir.stat.internal.fp_log_normal: logNormalDensity;
    return logNormalDensity!false(x, mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalPDF(T(double.infinity), T(3), T(2)) == T(0));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    // MPFR at 1,280 bits, rounded for each represented input.
    {
        alias T = Fp!128;
        enum expected = T(false, -132, UInt!128.fromHexString("c87bd6c3dba6e14a67fc48b9a8e57d34"));
        assert(logNormalPDF(T(4), T(1), T(2)) == expected);
        assert(approxEqual(logNormalPDF(4.0, 1.0, 2.0), 0.04894622700315107, 8 * double.epsilon));
    }
    alias T = Fp!64;
    // At this scale rounding log(2) to the mean would yield zero distance.
    enum mean = T(false, -64, UInt!64.fromHexString("b17217f7d1cf79ac"));
    auto scale = T(1);
    scale.exponent -= 64;
    enum nearMean = T(false, -2, UInt!64.fromHexString("c7bf077ea61c78f0"));
    assert(logNormalPDF(T(2), mean, scale) == nearMean);
    // The Gaussian decay underflows before division by this observation.
    auto tiny = T(1);
    tiny.exponent -= 1L << 62;
    enum rescued = T(false, -8694827079744363032L, UInt!64.fromHexString("bc58382da6244d64"));
    assert(logNormalPDF(tiny, T(-3196577165595631211L), T(1)) == rescued);
    assert(logNormalPDF(T(2), T(0), T(false, long.min, UInt!64(1))) == T(0));
    enum atOne = T(false, -65, UInt!64.fromHexString("cc42299ea1b28468"));
    assert(logNormalPDF(T(1)) == atOne);
}

/++
Computes the Log-normal cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

This overload uses mean zero and standard deviation one for $(D log(x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution)
+/
@safe pure nothrow @nogc
T logNormalCDF(T)(const T x)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalCDF;
    return x.log.normalCDF;
}

/++
Computes the standard log-normal lower-tail probability with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.

Returns:
    The lower-tail probability in type T. Zero at zero and one at positive infinity.
+/
@safe pure nothrow @nogc
T logNormalCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
{
    return logNormalCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalCDF(T(0)) == T(0));
}

/++
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Returns the lower-tail probability $(D P(X <= x)).

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    mean = finite mean of log(x)
    stdDev = positive finite standard deviation of log(x)
+/
@safe pure nothrow @nogc
T logNormalCDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalCDF;
    return x.log.normalCDF(mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalCDF(0.0).shouldApprox == 0;
    logNormalCDF(1.0).shouldApprox == 0.5;
    logNormalCDF(2.0).shouldApprox == 0.7558914;
    logNormalCDF(3.0).shouldApprox == 0.8640314;

    // Can include location/scale
    logNormalCDF(0.0, 1, 2).shouldApprox == 0;
    logNormalCDF(1.0, 1, 2).shouldApprox == 0.3085375;
    logNormalCDF(2.0, 1, 2).shouldApprox == 0.439031;
    logNormalCDF(3.0, 1, 2).shouldApprox == 0.5196623;
}

/++
Computes the location-scale log-normal lower-tail probability with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    mean = Finite mean of the underlying normal variable.
    stdDev = Positive finite standard deviation of the underlying normal variable.

Returns:
    The lower-tail probability in type T. Zero at zero and one at positive infinity.
+/
@safe pure nothrow @nogc
T logNormalCDF(T)(const T x, const T mean, const T stdDev)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
    in (!mean.isSpecial && !stdDev.isSpecial && !stdDev.sign && !!stdDev.coefficient)
{
    import mir.stat.internal.fp_log_normal: logNormalTail;
    return logNormalTail!false(x, mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalCDF(T(double.infinity), T(3), T(2)) == T(1));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    // MPFR at 1,280 bits, rounded for each represented input.
    {
        alias T = Fp!128;
        enum expected = T(false, -128, UInt!128.fromHexString("939aa022469a995136af5049539a3ec5"));
        assert(logNormalCDF(T(4), T(1), T(2)) == expected);
        assert(approxEqual(logNormalCDF(4.0, 1.0, 2.0), 0.5765781482392448, 8 * double.epsilon));
    }
    alias T = Fp!64;
    // At this scale rounding log(2) to the mean would yield zero distance.
    enum mean = T(false, -64, UInt!64.fromHexString("b17217f7d1cf79ac"));
    auto scale = T(1);
    scale.exponent -= 64;
    enum nearMean = T(false, -65, UInt!64.fromHexString("d525371ddcd90795"));
    assert(logNormalCDF(T(2), mean, scale) == nearMean);
    auto extreme = T(1);
    extreme.exponent -= 64;
    enum smallTail = T(false, -1490, UInt!64.fromHexString("c708531dd01b1122"));
    assert(logNormalCDF(extreme) == smallTail);
    assert(logNormalCDF(T(1)) == T(0.5));
    // log(1+h) - h is about -h*h/2 at the next observation above one.
    enum next = T(false, -63, UInt!64.fromHexString("8000000000000001"));
    enum h = T(false, -126, UInt!64.fromHexString("8000000000000000"));
    enum hSquared = T(false, -189, UInt!64.fromHexString("8000000000000000"));
    enum adjacent = T(false, -65, UInt!64.fromHexString("9df8a1dcd538921a"));
    assert(logNormalCDF(next, h, hSquared) == adjacent);
    const negativeZero = T(true, 0, UInt!64(0));
    assert(logNormalCDF(negativeZero) == T(0));
}

/++
Computes the Log-normal complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

This overload uses mean zero and standard deviation one for $(D log(x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution)
+/
@safe pure nothrow @nogc
T logNormalCCDF(T)(const T x)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalCCDF;
    return x.log.normalCCDF;
}

/++
Computes the standard log-normal strict upper-tail probability with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.

Returns:
    The strict upper-tail probability in type T. One at zero and zero at positive infinity.
+/
@safe pure nothrow @nogc
T logNormalCCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
{
    return logNormalCCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalCCDF(T(0)) == T(1));
}

/++
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    mean = finite mean of log(x)
    stdDev = positive finite standard deviation of log(x)
+/
@safe pure nothrow @nogc
T logNormalCCDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalCCDF;
    return x.log.normalCCDF(mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalCCDF(0.0).shouldApprox == 1;
    logNormalCCDF(1.0).shouldApprox == 0.5;
    logNormalCCDF(2.0).shouldApprox == 0.2441086;
    logNormalCCDF(3.0).shouldApprox == 0.1359686;

    // Can include location/scale
    logNormalCCDF(0.0, 1, 2).shouldApprox == 1;
    logNormalCCDF(1.0, 1, 2).shouldApprox == 0.6914625;
    logNormalCCDF(2.0, 1, 2).shouldApprox == 0.560969;
    logNormalCCDF(3.0, 1, 2).shouldApprox == 0.4803377;
}

/++
Computes the location-scale log-normal strict upper-tail probability with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    mean = Finite mean of the underlying normal variable.
    stdDev = Positive finite standard deviation of the underlying normal variable.

Returns:
    The strict upper-tail probability in type T. One at zero and zero at positive infinity.
+/
@safe pure nothrow @nogc
T logNormalCCDF(T)(const T x, const T mean, const T stdDev)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
    in (!mean.isSpecial && !stdDev.isSpecial && !stdDev.sign && !!stdDev.coefficient)
{
    import mir.stat.internal.fp_log_normal: logNormalTail;
    return logNormalTail!true(x, mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalCCDF(T(double.infinity), T(3), T(2)) == T(0));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    // MPFR at 1,280 bits, rounded for each represented input.
    {
        alias T = Fp!128;
        enum expected = T(false, -129, UInt!128.fromHexString("d8cabfbb72cacd5d92a15f6d58cb8275"));
        assert(logNormalCCDF(T(4), T(1), T(2)) == expected);
        assert(approxEqual(logNormalCCDF(4.0, 1.0, 2.0), 0.4234218517607552, 8 * double.epsilon));
    }
    alias T = Fp!64;
    // At this scale rounding log(2) to the mean would yield zero distance.
    enum mean = T(false, -64, UInt!64.fromHexString("b17217f7d1cf79ac"));
    auto scale = T(1);
    scale.exponent -= 64;
    enum nearMean = T(false, -64, UInt!64.fromHexString("956d647111937c35"));
    assert(logNormalCCDF(T(2), mean, scale) == nearMean);
    auto extreme = T(1);
    extreme.exponent += 64;
    enum smallTail = T(false, -1490, UInt!64.fromHexString("c708531dd01b1122"));
    assert(logNormalCCDF(extreme) == smallTail);
    assert(logNormalCCDF(T(1)) == T(0.5));
}

/++
Computes the Log-normal inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

This overload uses mean zero and standard deviation one for $(D log(x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution)
+/
@safe pure nothrow @nogc
T logNormalInvCDF(T)(const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    out (r; r >= 0, "return must be greater than or equal to zero")
{
    import mir.math.common: exp;
    import mir.stat.distribution.normal: normalInvCDF;
    return p.normalInvCDF.exp;
}

/++
Computes the standard log-normal quantile with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    p = Cumulative probability in the closed interval from zero to one.

Returns:
    The quantile in type T. Zero at probability zero, one at one half, and positive infinity at one.
+/
@safe pure nothrow @nogc
T logNormalInvCDF(T)(const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p))
{
    return logNormalInvCDF(p, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalInvCDF(T(0.5)) == T(1));
}

/++
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Returns the quantile at cumulative probability $(D p).

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D exp(mean))))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    mean = finite mean of log(x)
    stdDev = positive finite standard deviation of log(x)
+/
@safe pure nothrow @nogc
T logNormalInvCDF(T)(const T p, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (stdDev > 0, "stdDev must be greater than zero")
    out (r; r >= 0, "return must be greater than or equal to zero")
{
    import mir.math.common: exp;
    import mir.stat.distribution.normal: normalInvCDF;
    return p.normalInvCDF(mean, stdDev).exp;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalInvCDF(0.00).shouldApprox == 0;
    logNormalInvCDF(0.25).shouldApprox == 0.5094163;
    logNormalInvCDF(0.50).shouldApprox == 1;
    logNormalInvCDF(0.75).shouldApprox == 1.963031;
    logNormalInvCDF(1.00).shouldApprox == double.infinity;

    // Can include location/scale
    logNormalInvCDF(0.00, 1, 2).shouldApprox == 0;
    logNormalInvCDF(0.25, 1, 2).shouldApprox == 0.7054076;
    logNormalInvCDF(0.50, 1, 2).shouldApprox == 2.718282;
    logNormalInvCDF(0.75, 1, 2).shouldApprox == 10.47487;
    logNormalInvCDF(1.00, 1, 2).shouldApprox == double.infinity;
}

/++
Computes the location-scale log-normal quantile with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    p = Cumulative probability in the closed interval from zero to one.
    mean = Finite mean of the underlying normal variable.
    stdDev = Positive finite standard deviation of the underlying normal variable.

Returns:
    The quantile in type T. Zero at probability zero, exp(mean) at one half, and positive infinity at one.
+/
@safe pure nothrow @nogc
T logNormalInvCDF(T)(const T p, const T mean, const T stdDev)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p))
    in (!mean.isSpecial && !stdDev.isSpecial && !stdDev.sign && !!stdDev.coefficient)
{
    import mir.stat.internal.fp_log_normal: logNormalQuantile;
    return logNormalQuantile(p, mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalInvCDF(T(1), T(3), T(2)) == T(double.infinity));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    // MPFR at 1,280 bits, rounded for each represented input.
    alias T = Fp!64;
    enum standard = T(false, -65, UInt!64.fromHexString("a20fb7947ba125bc"));
    assert(logNormalInvCDF(T(0.125)) == standard);
    enum shifted = T(false, -62, UInt!64.fromHexString("80ca47292b6ce9d8"));
    assert(logNormalInvCDF(T(0.125), T(3), T(2)) == shifted);
    enum amplified = T(false, -117, UInt!64.fromHexString("eda65451a14a78d6"));
    assert(logNormalInvCDF(T(0.125), T(0), T(32)) == amplified);
    assert(approxEqual(logNormalInvCDF(0.125, 3.0, 2.0), 2.0123460676238225, 8 * double.epsilon));
    // Rounding the standard quantile before scaling would cancel the
    // mean exactly and give one instead of the MPFR value near 0.193.
    enum mean = T(false, 4, UInt!64(0x933ea603d82b3b2aUL));
    auto scale = T(1);
    scale.exponent += 67;
    assert(logNormalInvCDF(T(0.125), mean, scale) ==
        T(false, -66, UInt!64(0xc5672ed581152ccfUL)));
    // Adjacent integer log medians straddle the subnormal boundary.
    assert(logNormalInvCDF(T(0.5), T(-6393154322601327831L), T(1)) == T(0));
    assert(logNormalInvCDF(T(0.5), T(-6393154322601327830L), T(1)) ==
        T(false, long.min, UInt!64(1)));
    assert(logNormalInvCDF(T(0.5), T(6393154322601327872L), T(1)) ==
        T(false, long.max - 2, UInt!64(0xd69f8c1de43682f5UL)));
    assert(logNormalInvCDF(T(0.5), T(6393154322601327873L), T(1)) == T(double.infinity));
    const neighbor = T(false, -65, UInt!64.max);
    assert(logNormalInvCDF(neighbor) == T(false, -64, UInt!64.max));
    auto large = T(1);
    large.exponent += 64;
    assert(logNormalInvCDF(T(0.5), large, T(1)) == T(double.infinity));
    large.sign = true;
    assert(logNormalInvCDF(T(0.5), large, T(1)) == T(0));
    assert(logNormalInvCDF(T(0)) == T(0));
}

/++
Computes the Log-normal log probability density function (LPDF).

Returns the natural logarithm of the density.

This overload uses mean zero and standard deviation one for $(D log(x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution)
+/
@safe pure nothrow @nogc
T logNormalLPDF(T)(const T x)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalLPDF;
    if (x == 0) return -T.infinity;
    const T logX = log(x);
    return normalLPDF(logX) - logX;
}

/++
Computes the standard log-normal log density with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.

Returns:
    The log density in type T. Negative infinity at zero and positive infinity.
+/
@safe pure nothrow @nogc
T logNormalLPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
{
    return logNormalLPDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalLPDF(T(0)) == T(-double.infinity));
}

/++
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Returns the natural logarithm of the density.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    mean = finite mean of log(x)
    stdDev = positive finite standard deviation of log(x)
+/
@safe pure nothrow @nogc
T logNormalLPDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.normal: normalLPDF;
    if (x == 0) return -T.infinity;
    const T logX = log(x);
    return normalLPDF(logX, mean, stdDev) - logX;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    logNormalLPDF(1.0).shouldApprox == log(0.3989423);
    logNormalLPDF(2.0).shouldApprox == log(0.156874);
    logNormalLPDF(3.0).shouldApprox == log(0.07272826);

    // Can include location/scale
    logNormalLPDF(1.0, 1, 2).shouldApprox == log(0.1760327);
    logNormalLPDF(2.0, 1, 2).shouldApprox == log(0.09856858);
    logNormalLPDF(3.0, 1, 2).shouldApprox == log(0.06640961);
}


// check zero
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logNormalLPDF(0.0).shouldApprox == -double.infinity;
    logNormalLPDF(0.0, 1, 2).shouldApprox == -double.infinity;
}

// References evaluated with mpmath at 100 decimal digits. Powers of two
// make the observations exact; compare subnormals with an absolute allowance.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: fabs, log;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        static if (T.mant_dig == 24)
        {
            enum T x = 0x1p-20;
            enum T expected = 7.760530809436573628861965896882676e-37L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T x = 0x1p-56;
            enum T expected = 1.917345496570292173741763244485091e-311L;
        }
        else
        {
            enum T x = 0x1p-218L;
            enum T expected = 1.235517764122188481314436537289697e-4893L;
        }
        const T tolerance = 16 * T.epsilon * log(x) * log(x);
        assert(fabs(logNormalPDF(x) - expected) <=
            expected * tolerance + 4 * T.min_normal * T.epsilon);
        assert(fabs(logNormalPDF(x, T(0), T(1)) - expected) <=
            expected * tolerance + 4 * T.min_normal * T.epsilon);
        assert(logNormalPDF(T.infinity) == 0);
        assert(logNormalPDF(T.infinity, T(0), T(2)) == 0);
        assert(logNormalLPDF(T.infinity) == -T.infinity);
        assert(logNormalLPDF(T.infinity, T(0), T(2)) == -T.infinity);
        assert(logNormalPDF(T(1), T(0), T.min_normal * T.epsilon) == T.infinity);
        assert(logNormalInvCDF(T(0)) == 0);
        assert(logNormalInvCDF(T(1)) == T.infinity);
    }}
    enum double expected = 1.2130856546630247693e-262;
    assert(fabs(logNormalPDF(1e-200, 0.0, 10.0) / expected - 1) < 3e-13);
}

// The normal CDF tail correction must also reach the log-normal wrappers.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: fabs;
    // mpmath, 100 decimal digits, exact power-of-two observations.
    enum double expected = 1.3649994674147508546714884699975213978e-204;
    assert(fabs(logNormalCDF(0x1p-44) / expected - 1) < 2e-13);
    assert(fabs(logNormalCCDF(0x1p44) / expected - 1) < 2e-13);
}
/++
Computes the location-scale log-normal log density with guarded extended precision.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Nonnegative observation, including positive infinity.
    mean = Finite mean of the underlying normal variable.
    stdDev = Positive finite standard deviation of the underlying normal variable.

Returns:
    The log density in type T. Negative infinity at zero and positive infinity.
+/
@safe pure nothrow @nogc
T logNormalLPDF(T)(const T x, const T mean, const T stdDev)
    if (is(T == Fp!size, size_t size))
    in (!x.isNaN && (!x.sign || (!x.isSpecial && !x.coefficient)))
    in (!mean.isSpecial && !stdDev.isSpecial && !stdDev.sign && !!stdDev.coefficient)
{
    import mir.stat.internal.fp_log_normal: logNormalDensity;
    return logNormalDensity!true(x, mean, stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logNormalLPDF(T(double.infinity), T(3), T(2)) == T(-double.infinity));
}
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    // MPFR at 1,280 bits, rounded for each represented input.
    {
        alias T = Fp!128;
        enum expected = T(true, -126, UInt!128.fromHexString("c117118b6e01dcacbae8cc76b837d21c"));
        assert(logNormalLPDF(T(4), T(1), T(2)) == expected);
        assert(approxEqual(logNormalLPDF(4.0, 1.0, 2.0), -3.017032991563637, 8 * double.epsilon));
    }
    alias T = Fp!64;
    // At this scale rounding log(2) to the mean would yield zero distance.
    enum mean = T(false, -64, UInt!64.fromHexString("b17217f7d1cf79ac"));
    auto scale = T(1);
    scale.exponent -= 64;
    enum nearMean = T(false, -58, UInt!64.fromHexString("aae871782c98a01b"));
    assert(logNormalLPDF(T(2), mean, scale) == nearMean);

    // Adjacent scales put the density on opposite sides of one. Decimal
    // references use the represented mean and round the log densities directly.
    const lowerScale = T(false, -66, UInt!64(0xcc42299ea1b28468UL));
    const upperScale = T(false, -66, UInt!64(0xcc42299ea1b28469UL));
    assert(logNormalLPDF(T(2), mean, lowerScale) ==
        T(false, -128, UInt!64(0x9e5b8dfeb3fbbd1fUL)));
    assert(logNormalLPDF(T(2), mean, upperScale) ==
        T(true, -128, UInt!64(0xa27da400ae749c47UL)));

    enum atOne = T(true, -64, UInt!64.fromHexString("eb3f8e4325f5a535"));
    assert(logNormalLPDF(T(1)) == atOne);
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!32;
    // MPFR references for the same dyadics used at 128 bits.
    enum expectedPDF = T(false, -36, UInt!32.fromHexString("c87bd6c4"));
    assert(logNormalPDF(T(4), T(1), T(2)) == expectedPDF);
    enum expectedLPDF = T(true, -30, UInt!32.fromHexString("c117118b"));
    assert(logNormalLPDF(T(4), T(1), T(2)) == expectedLPDF);
    enum expectedCDF = T(false, -32, UInt!32.fromHexString("939aa022"));
    assert(logNormalCDF(T(4), T(1), T(2)) == expectedCDF);
    enum expectedCCDF = T(false, -33, UInt!32.fromHexString("d8cabfbb"));
    assert(logNormalCCDF(T(4), T(1), T(2)) == expectedCCDF);
    auto eighth = T(1);
    eighth.exponent -= 3;
    enum expectedQuantile = T(false, -30, UInt!32.fromHexString("80ca4729"));
    assert(logNormalInvCDF(eighth, T(3), T(2)) == expectedQuantile);
}
