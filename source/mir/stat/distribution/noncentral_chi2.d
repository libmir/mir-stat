/++
The noncentral chi-squared distribution describes the sum of squares of $(D k)
independent unit-variance normal variables. The noncentrality $(D lambda) is
the sum of their squared means. For a normal vector with positive-definite
covariance, its squared Mahalanobis distance from a fixed target has this
distribution when $(D lambda) is the squared Mahalanobis distance between
the mean and the target.

The functions accept built-in floating-point types and positive integer
degrees of freedom. Zero noncentrality gives the central chi-squared law.
Mixture calculations use $(D real) intermediates and return NaN if their
index or iteration limit is reached.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
+/
module mir.stat.distribution.noncentral_chi2;

import mir.internal.utility: isFloatingPoint;
private import mir.stat.distribution.chi2;

version(mir_stat_test) version = mir_stat_test_noncentral_chi2;

private enum maxMixtureSteps = 16_384;

/++
Computes the natural logarithm of the noncentral chi-squared density.

Params:
    T = built-in floating-point result and argument type
    x = nonnegative observation, including positive infinity
    k = positive integer degrees of freedom
    lambda = finite, nonnegative noncentrality

Returns:
    The log density as $(D T). At zero the result is positive infinity for
    one degree of freedom, $(D -log(2) - lambda / 2) for two, and negative
    infinity otherwise. At positive infinity it is negative infinity.
+/
@safe pure nothrow @nogc
T noncentralChi2LPDF(T)(const T x, const uint k, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be nonnegative")
    in (k > 0, "degrees of freedom must be positive")
    in (lambda >= 0 && lambda < T.infinity, "noncentrality must be finite and nonnegative")
{
    return cast(T) logDensity(x, k, lambda);
}

///
version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual, log;
    assert(approxEqual(noncentralChi2LPDF(0.0, 2, 4.0), -log(2.0) - 2));
}

/++
Computes the noncentral chi-squared density.

Params:
    T = built-in floating-point result and argument type
    x = nonnegative observation, including positive infinity
    k = positive integer degrees of freedom
    lambda = finite, nonnegative noncentrality

Returns:
    The density as $(D T). At zero the result is positive infinity for one
    degree of freedom, $(D exp(-lambda / 2) / 2) for two, and zero otherwise.
    At positive infinity it is zero.
+/
@safe pure nothrow @nogc
T noncentralChi2PDF(T)(const T x, const uint k, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be nonnegative")
    in (k > 0, "degrees of freedom must be positive")
    in (lambda >= 0 && lambda < T.infinity, "noncentrality must be finite and nonnegative")
{
    import mir.math.common: exp;
    return cast(T) exp(logDensity(x, k, lambda));
}

///
version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(approxEqual(noncentralChi2PDF(2.0, 2, 2.0), 0.15425416127683552));
}

private @safe pure nothrow @nogc
real logDensity(real x, uint k, real lambda)
{
    import mir.math.common: exp, log, sqrt;
    import mir.math.constant: LN2, PI;
    import mir.stat.distribution.gamma: gammaLPDF;
    import mir.stat.distribution.poisson: poissonLPMF;
    import std.math: hypot, log1p;

    const mean = lambda / 2;
    if (mean == 0 || x == 0 || x == real.infinity)
        return chi2LPDF(x, k) - mean;

    const rootX = sqrt(x);
    const rootLambda = sqrt(lambda);
    const z = rootX * (rootLambda / 2);
    if (k == 1)
    {
        // Both signs of the shifted normal map to the same squared value.
        const delta = (x - lambda) / (rootX + rootLambda);
        return -(delta / 2) * delta + log1p(exp(-4 * z))
            - LN2 - (log(2 * PI) + log(x)) / 2;
    }

    const shape = real(k) / 2;
    // Consecutive terms have ratio z^2 / (j * (shape + j - 1)).
    // Anchor at their mode so very small densities stay on the log scale.
    const halfOffset = (shape - 1) / 2;
    const root = z == 0 ? 0 : z * (z / (hypot(z, halfOffset) + halfOffset));
    if (!(root < uint.max))
        return real.nan;
    const mode = cast(uint) root;
    const anchor = poissonLPMF(mode, mean) + gammaLPDF(x, shape + mode, 2);
    real sum = 1;
    static foreach (upward; [false, true])
    {{
        real term = 1;
        size_t j = mode;
        for (uint steps = 0; ; ++steps)
        {
            static if (upward)
            {
                ++j;
                const ratio = (z / j) * (z / (shape + j - 1));
                term *= ratio;
                if (term <= (1 - ratio) * sum * real.epsilon)
                    break;
            }
            else
            {
                if (!j) break;
                const ratio = (j / z) * ((shape + j - 1) / z);
                term *= ratio;
                --j;
                if (term <= (1 - ratio) * sum * real.epsilon)
                    break;
            }
            if (steps == maxMixtureSteps || j >= uint.max)
                return real.nan;
            sum += term;
        }
    }}
    return anchor + log(sum);
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(noncentralChi2PDF(T(0), 1, T(2)) == T.infinity);
        assert(noncentralChi2PDF(T(0), 3, T(2)) == 0);
        assert(noncentralChi2LPDF(T.infinity, 3, T(2)) == -T.infinity);
        assert(approxEqual(noncentralChi2PDF(T(3), 5, T(0)), chi2PDF(T(3), 5)));
        // x = lambda = 1600: the first Poisson weight underflows in double.
        assert(approxEqual(noncentralChi2PDF(T(1600), 1, T(1600)),
            T(0.00498677850501790847L), T(8) * T.epsilon, T(0)));
        assert(noncentralChi2LPDF(T(1), 4, T(4000)) < -1900);
        assert(noncentralChi2LPDF(T(1), 4, T(4000)) > -2000);
    }}
}

/++
Computes the lower tail of the noncentral chi-squared distribution.

Params:
    T = built-in floating-point result and argument type
    x = nonnegative observation, including positive infinity
    k = positive integer degrees of freedom
    lambda = finite, nonnegative noncentrality

Returns:
    The probability $(D P(X <= x)) as $(D T). It is zero at zero and
    one at positive infinity.
+/
@safe pure nothrow @nogc
T noncentralChi2CDF(T)(const T x, const uint k, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be nonnegative")
    in (k > 0, "degrees of freedom must be positive")
    in (lambda >= 0 && lambda < T.infinity, "noncentrality must be finite and nonnegative")
{
    return cast(T) probability(x, k, lambda, false);
}

///
version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    // A two-dimensional normal ellipse with a center one Mahalanobis unit
    // from the mean, using the 95% centered squared radius.
    assert(approxEqual(noncentralChi2CDF(5.991464547107982, 2, 1.0),
        0.8672899857674835));
}

/++
Computes the upper tail of the noncentral chi-squared distribution.

Params:
    T = built-in floating-point result and argument type
    x = nonnegative observation, including positive infinity
    k = positive integer degrees of freedom
    lambda = finite, nonnegative noncentrality

Returns:
    The probability $(D P(X > x)) as $(D T). It is one at zero and
    zero at positive infinity.
+/
@safe pure nothrow @nogc
T noncentralChi2CCDF(T)(const T x, const uint k, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be nonnegative")
    in (k > 0, "degrees of freedom must be positive")
    in (lambda >= 0 && lambda < T.infinity, "noncentrality must be finite and nonnegative")
{
    return cast(T) probability(x, k, lambda, true);
}

///
version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    // A direct upper tail retains the probability outside the interval.
    assert(approxEqual(noncentralChi2CCDF(144.0, 1, 1.0),
        1.91065957455985e-28, 2e-14, 0.0));
}

private @safe pure nothrow @nogc
real probability(real x, uint k, real lambda, bool upper)
{
    import mir.stat.distribution.gamma: gammaCDF, gammaCCDF;
    import mir.stat.distribution.poisson: poissonLPMF;
    import mir.math.common: exp;

    if (x == 0 || x == real.infinity)
        return (x == 0) == upper;
    // Markov's inequality applied to exp(X/4) bounds the upper tail by
    // exp(k*log(2)/2 + lambda/2 - x/4). Use log(2) < 1 in both
    // the moment and the underflow threshold to keep the test conservative.
    if (x / 4 > lambda / 2 + real(k) / 2
            + (2 - real.min_exp + real.mant_dig))
        return upper ? 0 : 1;
    const mean = lambda / 2;
    if (mean == 0)
        return upper ? chi2CCDF(x, k) : chi2CDF(x, k);
    if (mean >= uint.max)
        return real.nan;

    // Use the upper gamma integral above the mean and the lower one below.
    // Only subtract from one on the side with substantial probability.
    const useUpper = x > k + lambda;
    const shape = real(k) / 2;
    real tail(uint j) {
        return useUpper ? gammaCCDF(x, shape + j, 2)
                        : gammaCDF(x, shape + j, 2);
    }
    const mode = cast(uint) mean;
    const peak = exp(poissonLPMF(mode, mean));
    const centerTail = tail(mode);
    real sum = peak * centerTail;
    static foreach (upward; [false, true])
    {{
        uint j = mode;
        real weight = peak;
        real component = centerTail;
        for (uint steps = 0; ; ++steps)
        {
            static if (upward)
                const ratio = mean / (real(j) + 1);
            else
            {
                if (j == 0) break;
                const ratio = j / mean;
            }
            // Ratios decrease away from the mode. Bound all remaining
            // Poisson mass geometrically. On the side where the gamma
            // probability also decreases, its current value tightens the bound.
            weight *= ratio;
            const boundFactor = (upward != useUpper) ? component : 1;
            if (weight == 0 || (ratio < 1 &&
                weight * boundFactor <= (1 - ratio) * real.epsilon * sum))
                break;
            if (steps == maxMixtureSteps || (upward && j == uint.max))
                return real.nan;
            static if (upward) ++j;
            else --j;
            component = tail(j);
            sum += weight * component;
        }
    }}
    if (sum > 1) sum = 1;
    return upper == useUpper ? sum : 1 - sum;
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(noncentralChi2CDF(T.max, 1, T(2)) == 1);
        assert(noncentralChi2CCDF(T.max, 1, T(2)) == 0);
        assert(noncentralChi2CDF(T(0), 1, T(1)) == 0);
        assert(noncentralChi2CCDF(T.infinity, 3, T(2)) == 0);
        assert(noncentralChi2CDF(T.infinity, 1, T(1)) == 1);
        assert(noncentralChi2CCDF(T(0), 3, T(2)) == 1);
        assert(approxEqual(noncentralChi2CDF(T(3), 5, T(0)), chi2CDF(T(3), 5)));
        assert(approxEqual(noncentralChi2CCDF(T(3), 5, T(0)), chi2CCDF(T(3), 5)));
        assert(approxEqual(noncentralChi2CDF(T(1600), 1, T(1600)), T(0.5)));
    }}
}
