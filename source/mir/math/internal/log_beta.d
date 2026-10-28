/++
Internal logarithmic beta normalization for positive finite shapes.

Keeps the direct log-gamma calculation for small normal shapes. Larger shapes
use a combined Stirling expression, so a small difference between large gamma
logarithms is not lost before the result is returned. Subnormal shapes use the
gamma recurrence to avoid an overflowing reciprocal inside logGamma.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.math.internal.log_beta;

import mir.internal.utility: isFloatingPoint;
import mir.math.internal.log_gamma: logGammaStirlingCorrection;

/++
Natural logarithm of B(alpha, beta). Intermediate arithmetic retains T's
precision except for the existing Phobos logGamma return type. Results remain
subject to rounding and may overflow when the logarithm itself exceeds T's range.
+/
package(mir)
@safe pure nothrow @nogc
T logBeta(T)(const T alpha, const T beta)
    if (isFloatingPoint!T)
{
    import std.mathspecial: logGamma;

    if (alpha < 16 && beta < 16 && alpha >= T.min_normal && beta >= T.min_normal)
        return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
    return logBetaFallback(alpha, beta);
}

private @safe pure nothrow @nogc
T logBetaFallback(T)(const T alpha, const T beta)
{
    // Keep the larger calculation out of the ordinary small-shape call site.
    pragma(inline, false);
    import std.math: log, log1p;
    import std.mathspecial: logGamma;

    // Preserve the previous behavior outside the positive finite domain.
    if (!(alpha > 0 && beta > 0 && alpha < T.infinity && beta < T.infinity))
        return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);

    const T a = alpha < beta ? alpha : beta;
    const T b = alpha < beta ? beta : alpha;
    if (b < 16)
    {
        // At least one shape is subnormal. Gamma(z)=Gamma(1+z)/z
        // moves all gamma arguments away from zero; their logs stay finite.
        return cast(T) (logGamma(1 + a) + logGamma(1 + b) - logGamma(1 + a + b))
            + log(a + b) - log(a) - log(b);
    }

    const T ratio = a / b;
    const T logRatio = log1p(ratio);
    const T inverseSum = (1 / b) / (1 + ratio);
    if (a < 16)
    {
        const T smallGamma = a < T.min_normal
            ? cast(T) logGamma(1 + a) - log(a) : cast(T) logGamma(a);
        // Combine logGamma(b)-logGamma(a+b) before evaluation. Splitting
        // (a+b-1/2)*log1p(a/b) avoids forming the possibly overflowing sum.
        return smallGamma - a * log(b) + (a - b * logRatio)
            - (a - T(0.5)) * logRatio
            + logGammaStirlingCorrection(1 / b) - logGammaStirlingCorrection(inverseSum);
    }

    // Both shapes admit Stirling expansion. a/b cannot underflow here:
    // a>=16 and b<=T.max. log(a/b) avoids subtracting two large logarithms.
    return a * log(ratio) - a * logRatio - b * logRatio
        + T(0.5) * logRatio - T(0.5) * log(a)
        + T(0.91893853320467274178032973640561764L)
        + logGammaStirlingCorrection(1 / a) + logGammaStirlingCorrection(1 / b)
        - logGammaStirlingCorrection(inverseSum);
}

// Independent references around the Stirling transition and unequal shapes.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // 120-decimal-digit references for exactly representable inputs.
        const real[3][5] references = [
            [0.125L, 16L, 1.676289181923070711366857286438451596L],
            [7.25L, 16L, -14.29950746087314117520127940803930057L],
            [16L, 16L, -22.29368078563352749923792953137939033L],
            [32L, 64L, -61.71351891820218055826666411894420236L],
            [0.5L, 1048576L, -6.359106743465463456323875047957258027L]];
        foreach (row; references)
        {
            const actual = logBeta(T(row[0]), T(row[1]));
            assert(approxEqual(actual, T(row[2]), 64 * T.epsilon, T(0)));
            assert(actual == logBeta(T(row[1]), T(row[0])));
        }
    }}
}

// Exact beta identities check huge shapes and subnormal gamma arguments.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log, nextUp, nextDown;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (b; [nextDown(T(16)), T(16), nextUp(T(16)), T(1e16), T.max / 4])
        {
            assert(approxEqual(logBeta(T(1), b), -log(b), 128 * T.epsilon, T(0)));
            assert(approxEqual(logBeta(T(2), b), -log(b) - log(b + 1),
                128 * T.epsilon, T(0)));
        }
        const T tiny = nextUp(T(0));
        assert(approxEqual(logBeta(tiny, T(1)), -log(tiny), 64 * T.epsilon, T(0)));
        // B(t,t) tends to 2/t; the correction is negligible at this scale.
        assert(approxEqual(logBeta(tiny, tiny), log(T(2)) - log(tiny),
            64 * T.epsilon, T(0)));
        assert(approxEqual(logBeta(tiny, T(16)), -log(tiny), 64 * T.epsilon, T(0)));
        const T large = T.max / 4;
        assert(approxEqual(logBeta(large, large), -large * (2 * log(T(2))),
            64 * T.epsilon, T(0)));
    }}
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    import std.mathspecial: beta;

    assert(logBeta(1.0, 2).approxEqual(log(beta(1.0, 2))));
    assert(logBeta(0.5, 4).approxEqual(log(beta(0.5, 4))));
}
