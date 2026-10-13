/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Log-normal_distribution, Log-normal Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.log_normal;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Log-normal probability density function (PDF).

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
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Params:
    x = value to evaluate PDF
    mean = location parameter
    stdDev = scale parameter
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

/++
Computes the Log-normal cumulative distribution function (CDF).

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
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Params:
    x = value to evaluate CDF
    mean = location parameter
    stdDev = scale parameter
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
Computes the Log-normal complementary cumulative distribution function (CCDF).

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
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Params:
    x = value to evaluate CCDF
    mean = location parameter
    stdDev = scale parameter
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
Computes the Log-normal inverse cumulative distribution function (InvCDF).

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
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Params:
    p = value to evaluate InvCDF
    mean = location parameter
    stdDev = scale parameter
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
Computes the Log-normal log probability density function (LPDF).

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
Ditto, with location and scale parameters for $(D log(x)).
These are the mean and standard deviation of the underlying normal
variable, not the mean and standard deviation of the log-normal variable.

Params:
    x = value to evaluate LPDF
    mean = location parameter
    stdDev = scale parameter
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
