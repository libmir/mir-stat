/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution).

This module uses the shape/scale parameterization of the gamma distribution. To
use the shape/rate parameterization, apply the inverse to the rate and pass it
as the scale parameter.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: Ilia Ki, John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.gamma;

import mir.internal.utility: isFloatingPoint;

/++
Computes the gamma probability density function (PDF).

`shape` values less than `1` are supported when it is a floating point type.

If `shape` is passed as a `size_t` type (or a type convertible to that), then the
PDF is calculated using the relationship with the poisson distribution (i.e.
replacing the `gamma` function with the `factorial`).

The floating-point shape overload uses a logarithmic fallback when the direct
calculation overflows or loses precision through underflow.

Params:
    x = value to evaluate PDF
    shape = shape parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution)
+/
@safe pure nothrow @nogc
T gammaPDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: exp, pow, log;
    import std.mathspecial: gamma, logGamma;
    
    if (x == 0) {
        if (shape > 1) {
            return 0;
        } else if (shape < 1) {
            return T.infinity;
        } else {
            return 1 / scale;
        }
    }

    const T x_scale = x / scale;
    const T decay = exp(-x_scale);
    const T density = decay * pow(x_scale, shape - 1) / cast(T) gamma(shape);
    // Keep the direct calculation when its intermediates retain precision.
    // A subnormal density can become significant after division by scale.
    if (x_scale >= T.min_normal && decay >= T.min_normal
        && density >= T.min_normal && density < T.infinity)
        return density / scale;

    if (x_scale == T.infinity && shape < T.infinity)
        return 0;
    // A subnormal x / scale may already have lost significant digits.
    const T logRatio = x_scale >= T.min_normal ? log(x_scale) : log(x) - log(scale);
    return exp((shape - 1) * logRatio - x_scale - cast(T) logGamma(shape) - log(scale));
}

/// ditto
@safe pure nothrow @nogc
T gammaPDF(T)(const T x, const size_t shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.stat.distribution.poisson: poissonPMF;

    if (x == 0) {
        if (shape > 1) {
            return 0;
        } else {
            return 1 / scale;
        }
    }

    return poissonPMF!"direct"(shape - 1, x / scale) / scale;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    2.0.gammaPDF(3.0).shouldApprox == 0.2706706;
    2.0.gammaPDF(3.0, 4.0).shouldApprox == 0.01895408;
    // Calling with `size_t` uses factorial function instead of gamma, but
    // produces same results
    2.0.gammaPDF(3).shouldApprox == 2.0.gammaPDF(3.0);
    2.0.gammaPDF(3, 4.0).shouldApprox == 2.0.gammaPDF(3.0, 4.0);
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should, shouldApprox;

    // check integer, x = 0
    0.0.gammaPDF(2).should == 0;
    0.0.gammaPDF(1).should == 1;
    0.0.gammaPDF(1, 4).shouldApprox == 0.25;

    // check float, x = 0
    0.0.gammaPDF(2.0).should == 0;
    0.0.gammaPDF(1.0).should == 1;
    0.0.gammaPDF(0.5).should == double.infinity;
    0.0.gammaPDF(2.0, 4.0).should == 0;
    0.0.gammaPDF(1.0, 4.0).shouldApprox == 0.25;
    0.0.gammaPDF(0.5, 4.0).should == double.infinity;

    // check integer
    0.5.gammaPDF(3).shouldApprox == 0.07581633;
    3.5.gammaPDF(4, 2.5).shouldApprox == 0.0451108;

    // check float, shape >= 1
    1.25.gammaPDF(1.0).shouldApprox == 0.2865048;
    1.25.gammaPDF(1.0, 0.5).shouldApprox == 0.16417;
    1.5.gammaPDF(2.5, 0.5).shouldApprox == 0.3892174;

    // check float, shape < 1
    0.005.gammaPDF(0.01).shouldApprox == 1.898102;
    1.5.gammaPDF(0.25).shouldApprox == 0.04540553;
    3.0.gammaPDF(0.5, 2.0).shouldApprox == 0.05139344;
}

/++
Computes the gamma cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    shape = shape parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution)
+/
@safe pure nothrow @nogc
T gammaCDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import std.mathspecial: gammaIncomplete;
    return gammaIncomplete(shape, x / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    2.0.gammaCDF(5).shouldApprox == 0.05265302;
    1.0.gammaCDF(5, 0.5).shouldApprox == 0.05265302;
}

// checking some more extreme values for shape and others
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.5.gammaCDF(2, 1.5).shouldApprox == 0.04462492;
    0.25.gammaCDF(0.5, 4).shouldApprox == 0.276326;
    0.0625.gammaCDF(0.5).shouldApprox == 0.276326;
    0.0625.gammaCDF(2).shouldApprox == 0.001873621;
    0.00007854393.gammaCDF(0.5).shouldApprox == 0.01;
    10.gammaCDF(2, 1.5).shouldApprox == 0.9902431;
    5.gammaCDF(0.5, 1.5).shouldApprox == 0.9901767;
    6.666666.gammaCDF(2).shouldApprox == 0.9902431;
    3.333333.gammaCDF(0.5).shouldApprox == 0.9901767;
}

/++
Computes the gamma complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    shape = shape parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution)
+/
@safe pure nothrow @nogc
T gammaCCDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import std.mathspecial: gammaIncompleteCompl;
    return gammaIncompleteCompl(shape, x / scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    2.0.gammaCCDF(5).shouldApprox == 0.947347;
    1.0.gammaCCDF(5, 0.5).shouldApprox == 0.947347;
}

// checking some more extreme values for shape and others
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.5.gammaCCDF(2, 1.5).shouldApprox == 0.9553751;
    0.25.gammaCCDF(0.5, 4).shouldApprox == 0.7236736;
    0.0625.gammaCCDF(0.5).shouldApprox == 0.7236736;
    0.0625.gammaCCDF(2).shouldApprox == 0.9981264;
    0.00007854393.gammaCCDF(0.5).shouldApprox == 0.99;
    10.gammaCCDF(2, 1.5).shouldApprox == 0.009756859;
    5.gammaCCDF(0.5, 1.5).shouldApprox == 0.009823275;
    6.666666.gammaCCDF(2).shouldApprox == 0.009756865;
    3.333333.gammaCCDF(0.5).shouldApprox == 0.009823278;
}

/++
Computes the gamma inverse cumulative distribution function (InvCDF).

Lower-tail probabilities are inverted directly, without first subtracting
them from one. If the unscaled quantile underflows, the scale is applied in
logarithmic form so a representable scaled result can be retained.

Very large shapes can be slow with Phobos versions before DMD 2.113, which
use a power series to evaluate the lower cumulative probability near the mean.

Params:
    p = value to evaluate InvCDF
    shape = shape parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution)
+/
@safe pure nothrow @nogc
T gammaInvCDF(T)(const T p, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import std.mathspecial: gammaIncompleteComplInverse;
    if (p >= T(0.5))
    {
        const real x = gammaIncompleteComplInverse(shape, 1 - p);
        // Small shapes can have tiny quantiles even above the median. Recover
        // these in log space before applying a scale that may restore them.
        if (x >= T.min_normal || p == 1 || !(shape < 1))
            return cast(T) (x * scale);
    }
    return cast(T) gammaLowerInvCDF(p, shape, scale);
}

// Work in real, as does the complementary inverse used for the upper half.
// Keeping the lower probability avoids losing it in the subtraction 1-p.
private @safe pure nothrow @nogc
real gammaLowerInvCDF(const real p, const real shape, const real scale)
{
    import std.math: exp, log, log1p, fabs;
    import std.mathspecial: gammaIncomplete, logGamma;
    import std.numeric: findRoot;

    if (shape == real.infinity)
        return real.nan;
    if (p == 0)
        return 0 * scale;

    // Near the mean, solve in x rather than log(x). This avoids cancellation
    // in the logarithmic derivative for large shapes. Phobos 2.113 added a
    // faster incomplete-gamma implementation here; older versions can be slow.
    if (shape > 25 && p >= gammaIncomplete(shape, 0.8L * shape))
        return findRoot((real x) => gammaIncomplete(shape, x) - p,
            0.8L * shape, shape) * scale;

    const real logP = log(p);
    const real logGammaNext = shape < 1 ? logGamma(1 + shape)
        : logGamma(shape) + log(shape);
    // P(a,x) <= x^a/Gamma(a+1), so this is a lower bound on log(x).
    // For p < 1/2 the quantile is below the mean a, giving the upper bound.
    // The upper-half fallback is only used for a < 1: a unit-scale gamma
    // with this shape is stochastically smaller than a shape-one exponential,
    // whose quantile -log(1-p) supplies an upper bound instead.
    real low = (logP + logGammaNext) / shape;
    real high = p < 0.5L ? log(shape) : log(-log1p(-p));
    real y = low;
    if (y == -real.infinity)
        return exp(y + log(scale));

    foreach (iteration; 0 .. 128)
    {
        const real x = exp(y);
        // P(a,x) = exp(a*log(x)-x)/Gamma(a+1) * sum,
        // sum = 1 + x/(a+1) + x^2/((a+1)*(a+2)) + ... .
        real term = 1;
        real sum = 1;
        bool converged;
        foreach (n; 1 .. 100_000)
        {
            term *= x / (shape + n);
            sum += term;
            if (term <= real.epsilon * sum)
            {
                converged = true;
                break;
            }
        }
        if (!converged)
            return real.nan;

        const real residual = shape * y - x - logGammaNext + log(sum) - logP;
        // d(log(P))/d(log(x)) = a/sum. Newton steps therefore remain useful
        // even when x, P, or the ordinary density would underflow.
        const real step = residual / (shape / sum);
        if (fabs(step) <= 4 * real.epsilon * (1 + fabs(y)))
            break;
        if (residual > 0)
            high = y;
        else
            low = y;
        real next = y - step;
        if (!(next > low && next < high))
            next = low + (high - low) / 2;
        if (next == y)
            break;
        y = next;
        if (iteration == 127)
            return real.nan;
    }

    const real x = exp(y);
    if (x >= real.min_normal)
        return x * scale;
    // Apply the scale before exponentiation if the unscaled quantile is
    // subnormal or zero. A large scale may restore a representable result.
    return exp(y + log(scale));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.05.gammaInvCDF(5).shouldApprox == 1.97015;
    0.05.gammaInvCDF(5, 0.5).shouldApprox == 0.9850748;
}

// checking some more extreme values for shape and others
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    0.04.gammaInvCDF(2, 1.5).shouldApprox == 0.4703589;
    0.27.gammaInvCDF(0.5, 4).shouldApprox == 0.2382233;
    0.27.gammaInvCDF(0.5).shouldApprox == 0.05955582;
    0.002.gammaInvCDF(2).shouldApprox == 0.06461886;
    0.01.gammaInvCDF(0.5).shouldApprox == 0.00007854393;
    0.99.gammaInvCDF(2, 1.5).shouldApprox == 9.957528;
    0.99.gammaInvCDF(0.5, 1.5).shouldApprox == 4.976172;
    0.99.gammaInvCDF(2).shouldApprox == 6.638352;
    0.99.gammaInvCDF(0.5).shouldApprox == 3.317448;
}

// Lower probabilities and scaled quantiles survive intermediate underflow.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log, nextDown, nextUp;
    import mir.math.common: approxEqual;
    import mir.math.constant: PI;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Shape one is exponential: -log(1-p) = p + O(p^2).
        const T p = T.epsilon * T.epsilon;
        assert(approxEqual(gammaInvCDF(p, T(1), 1 / p), T(1),
            16 * T.epsilon * (1 - log(p)), T(0)));

        // For shape 1/2, P(1/2,x) = erf(sqrt(x)), hence the small-p
        // quantile is pi*p^2/4 + O(p^4). Its unscaled value can underflow
        // even real, but the scaled result is normal in T.
        const T tinyP = 16 * T.min_normal;
        const T expected = T(PI / 4) * tinyP;
        assert(approxEqual(gammaInvCDF(tinyP, T(0.5), 1 / tinyP), expected,
            16 * T.epsilon * (1 - log(tinyP)), T(0)));
        assert(gammaInvCDF(T(0), T(2), T(3)) == 0);
        assert(gammaInvCDF(T(1), T(2), T(3)) == T.infinity);
        foreach (middle; [nextDown(T(0.5)), T(0.5), nextUp(T(0.5))])
            assert(approxEqual(gammaInvCDF(middle, T(1)), -log(1 - middle),
                32 * T.epsilon, T(0)));

        // Independent 80-decimal-digit incomplete-gamma inversion.
        assert(approxEqual(gammaInvCDF(T(1e-20L), T(200)),
            T(95.6946452213520086055421832734165936725L),
            128 * T.epsilon, T(0)));
    }}
    static foreach (T; AliasSeq!(double, real))
    {{
        // Logarithmic evaluation and input rounding are amplified in these
        // very small quantiles, especially for the nonexact shape 0.1.
        assert(approxEqual(gammaInvCDF(T(1e-20L), T(0.1L)),
            T(6.073048362407882531571435593407629104L * 1e-201L),
            4096 * T.epsilon, T(0)));
        assert(approxEqual(gammaInvCDF(T(1e-100L), T(2.5)),
            T(1.616703890291564173611661750815240370L * 1e-40L),
            512 * T.epsilon, T(0)));
    }}
}

// Tiny unscaled quantiles must remain recoverable on both sides of the median.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp, log, exp, nextDown, nextUp;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T shape = T(1) / (2 * T.max_exp);
        const T scale = ldexp(T(1), T.max_exp - 2);
        // Independent 80-digit references for the exact binary inputs.
        // At these tiny quantiles, P(a,x) = x^a/Gamma(a+1) has a
        // correction far smaller than the precision of the references.
        static if (T.max_exp == 128)
        {
            enum T median = T(4.138201387686505843786461287812758641L * 1e-40L);
            enum T upper = T(5.142350936910122516153090934962000723L * 1e-27L);
        }
        else static if (T.max_exp == 1024)
        {
            enum T median = T(7.811190683306037302813411142333638137L * 1e-310L);
            enum T upper = T(4.497757514169894067039631343546501935L * 1e-205L);
        }
        else
        {
            enum T median = T(1.179832546652321293867668997609491791L * 1e-4933L);
            enum T upper = T(1.728504354429892150970839311825448008L * 1e-3257L);
        }

        T previous = 0;
        foreach (p; [nextDown(T(0.5)), T(0.5), nextUp(T(0.5))])
        {
            const T value = gammaInvCDF(p, shape, scale);
            const T expected = median * exp(log(2 * p) / shape);
            assert(value > 0 && value >= previous);
            // Log-space error is amplified by 1/shape; the result is also
            // subnormal, so allow its final rounding error explicitly.
            assert(approxEqual(value, expected, 16 * T.epsilon / shape,
                2 * nextUp(T(0))));
            previous = value;
        }
        // A smaller shape also needs recovery well above the median.
        assert(approxEqual(gammaInvCDF(T(0.75), shape / 2, scale), upper,
            32 * T.epsilon / shape, T(0)));
    }}
}

version(mir_stat_test)
{
    // Phobos 2.113 introduced the fast large-shape incomplete-gamma path.
    // Explicit opt-in also runs these on older frontends; that can be slow.
    version(mir_stat_test_extreme_numerics)
        private enum testLargeGammaShapes = true;
    else
        private enum testLargeGammaShapes = __VERSION__ >= 2113;

    static if (testLargeGammaShapes)
    @safe pure nothrow @nogc
    unittest
    {
        import std.meta: AliasSeq;
        import std.math: sqrt;
        import mir.math.common: approxEqual;

        static foreach (T; AliasSeq!(float, double, real))
        {{
            // Independent 80-decimal-digit references.
            assert(approxEqual(gammaInvCDF(T(0.1), T(10000)),
                T(9872.060875049735778499556443787512518L),
                128 * T.epsilon, T(0)));
            assert(approxEqual(gammaInvCDF(T(1e-20L), T(1000000)),
                T(990765.903258275879702339899208327863L),
                128 * T.epsilon, T(0)));
            // At this shape, the first two normal-limit correction terms
            // locate the quantile well within one floating-point step.
            const T shape = T(1e20L);
            const T z = T(-9.262340089798407573717356977875325L);
            const T expected = shape + sqrt(shape) * z + (z * z - 1) / 3;
            assert(approxEqual(gammaInvCDF(T(1e-20L), shape), expected,
                8 * T.epsilon, T(0)));
        }}
    }
}

// confirming consistency with gammaCDF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    2.0.gammaCDF(5).gammaInvCDF(5).shouldApprox == 2;
    1.0.gammaCDF(5, 0.5).gammaInvCDF(5, 0.5).shouldApprox == 1;
    0.5.gammaCDF(2, 1.5).gammaInvCDF(2, 1.5).shouldApprox == 0.5;
    0.5.gammaCDF(2, 1.5).gammaInvCDF(2, 1.5).shouldApprox == 0.5;
    0.25.gammaCDF(0.5, 4).gammaInvCDF(0.5, 4).shouldApprox == 0.25;
    0.0625.gammaCDF(0.5).gammaInvCDF(0.5).shouldApprox == 0.0625;
    0.0625.gammaCDF(2).gammaInvCDF(2).shouldApprox == 0.0625;
    0.00007854393.gammaCDF(0.5).gammaInvCDF(0.5).shouldApprox == 0.00007854393;
    10.gammaCDF(2, 1.5).gammaInvCDF(2, 1.5).shouldApprox == 10;
    5.gammaCDF(0.5, 1.5).gammaInvCDF(0.5, 1.5).shouldApprox == 5;
    6.666666.gammaCDF(2).gammaInvCDF(2).shouldApprox == 6.666666;
    3.333333.gammaCDF(0.5).gammaInvCDF(0.5).shouldApprox == 3.333333;
}

/++
Computes the gamma log probability density function (LPDF).

`shape` values less than `1` are supported when it is a floating point type.

If `shape` is passed as a `size_t` type (or a type convertible to that), then the
LPDF is calculated using the relationship with the poisson distribution (i.e.
replacing the `logGamma` function with the `logFactorial`).

Params:
    x = value to evaluate LPDF
    shape = shape parameter
    scale = scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Gamma_distribution, Gamma Distribution)
+/
@safe pure nothrow @nogc
T gammaLPDF(T)(const T x, const T shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;
    import std.mathspecial: logGamma;

    if (x == 0) {
        if (shape > 1) {
            return -T.infinity;
        } else if (shape < 1) {
            return T.infinity;
        } else {
            return -log(scale);
        }
    }

    T x_scale = x / scale;
    return (shape - 1) * log(x_scale) - x_scale - cast(T) logGamma(shape) - log(scale);
}

/// ditto
@safe pure nothrow @nogc
T gammaLPDF(T)(const T x, const size_t shape, const T scale = 1)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (shape > 0, "shape must be greater than zero")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;
    import mir.stat.distribution.poisson: poissonLPMF;

    if (x == 0) {
        if (shape > 1) {
            return -T.infinity;
        } else {
            return -log(scale);
        } // note: shape cannot be equal to zero or less than 1 because it is size_t
    }

    return poissonLPMF(shape - 1, x / scale) - log(scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    2.0.gammaLPDF(3.0).shouldApprox == -1.306853;
    2.0.gammaLPDF(3.0, 4.0).shouldApprox == -3.965736;
    // Calling with `size_t` uses log factorial function instead of log gamma,
    // but produces same results
    2.0.gammaLPDF(3).shouldApprox == 2.0.gammaLPDF(3.0);
    2.0.gammaLPDF(3, 4.0).shouldApprox == 2.0.gammaLPDF(3.0, 4.0);
}

// test floating point version
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    for (double x = 0; x <= 10; x = x + 0.5) {
        x.gammaLPDF(5.0).exp.shouldApprox == x.gammaPDF(5.0);
        x.gammaLPDF(5.0, 1.5).exp.shouldApprox == x.gammaPDF(5.0, 1.5);
        x.gammaLPDF(1.0).exp.shouldApprox == x.gammaPDF(1.0);
        x.gammaLPDF(1.0, 1.5).exp.shouldApprox == x.gammaPDF(1.0, 1.5);
        x.gammaLPDF(0.5).exp.shouldApprox == x.gammaPDF(0.5);
        x.gammaLPDF(0.5, 1.5).exp.shouldApprox == x.gammaPDF(0.5, 1.5);
    }
}

// test size_t version
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    for (double x = 0; x <= 10; x = x + 0.5) {
        x.gammaLPDF(5).exp.shouldApprox == x.gammaPDF(5);
        x.gammaLPDF(5, 1.5).exp.shouldApprox == x.gammaPDF(5, 1.5);
        x.gammaLPDF(1).exp.shouldApprox == x.gammaPDF(1);
        x.gammaLPDF(1, 1.5).exp.shouldApprox == x.gammaPDF(1, 1.5);
    }
}

// Finite densities survive overflow and underflow in the direct formula.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp, nextUp;
    import mir.math.common: approxEqual, sqrt, exp, log;
    import mir.math.constant: PI;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Reference: 200^199 * exp(-200) / 199!, evaluated at high precision.
        // The logarithmic terms are much larger than their difference.
        assert(approxEqual(gammaPDF(T(200), T(200)),
            T(0.028197727685920821798702062339363012L), 512 * T.epsilon, T(0)));

        const T tiny = nextUp(T(0));
        const T scale = ldexp(T(1), T.max_exp - 2);
        // For shape 1/2, x/scale is negligible and the density is
        // 1/sqrt(pi*x*scale), even when the ratio rounds to zero.
        const T expected = 1 / sqrt(T(PI) * (tiny * scale));
        assert(approxEqual(gammaPDF(tiny, T(0.5), scale), expected,
            8 * T.epsilon * log(scale), T(0)));

        // The ratio need not round all the way to zero to lose precision.
        const T subnormalX = 3 * tiny;
        const T expectedSubnormal = 1 / sqrt(T(PI)) / sqrt(subnormalX) / sqrt(T(2));
        assert(approxEqual(gammaPDF(subnormalX, T(0.5), T(2)), expectedSubnormal,
            8 * T.epsilon * -log(subnormalX), T(0)));

        // Shape one is exponential. Its unscaled density underflows, but
        // dividing by a small scale restores a representable result.
        const T smallScale = ldexp(T(1), -T.max_exp / 2);
        const T z = T.max_exp;
        assert(approxEqual(gammaPDF(z * smallScale, T(1), smallScale),
            exp(-z - log(smallScale)), 8 * T.epsilon * z, T(0)));
        assert(gammaPDF(T.infinity, T(2)) == 0);
    }}
}
