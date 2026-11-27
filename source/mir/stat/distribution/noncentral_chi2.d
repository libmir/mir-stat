/++
The noncentral chi-squared distribution describes the sum of squares of $(D k)
independent unit-variance normal variables. The noncentrality $(D lambda) is
the sum of their squared means. For a normal vector with positive-definite
covariance, its squared Mahalanobis distance from a fixed target has this
distribution when $(D lambda) is the squared Mahalanobis distance between
the mean and the target.

The functions accept built-in floating-point types and positive integer
degrees of freedom. Zero noncentrality gives the central chi-squared law.
$(D noncentralChi2CDF) gives the probability inside a covariance-shaped
ellipsoid when $(D x) is its squared radius. $(D noncentralChi2CCDF) gives
the probability outside. $(D noncentralChi2InvCDF) returns the squared
radius for a specified enclosed probability.

The Poisson mixture uses $(D real) intermediates. Its starting index must
be smaller than $(D uint.max), and each direction of summation allows
16,384 updates. An unresolved evaluation returns NaN, including during
quantile inversion. Accuracy also depends on the underlying gamma and
Poisson routines. The work per call can grow with noncentrality.

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
            T(0.00498677850501790847424932574917977L), T(8) * T.epsilon, T(0)));
        assert(approxEqual(noncentralChi2LPDF(T(1), 4, T(4000)),
            T(-1945.093046548164L), T(2e-14) + T.epsilon, T(0)));
        assert(approxEqual(noncentralChi2PDF(T(1600), 2, T(1600)),
            T(0.004987168234143833L), T(2e-14) + T.epsilon, T(0)));
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

// Keep tiny gamma components on the inverse comparison scale. Applying
// the scale after gammaCDF has rounded to zero loses the component.
private @safe pure nothrow @nogc
real scaledGammaLowerTail(real x, real shape, real scale)
{
    import mir.stat.distribution.gamma: gammaCDF, gammaLPDF;
    import mir.math.common: exp, log;
    const value = gammaCDF(x, shape, 2);
    if (scale == 1 || !(value < real.min_normal))
        return scale * value;

    // The lower incomplete-gamma series starts at
    // exp(-x/2) * (x/2)^^shape / Gamma(shape + 1).
    const first = exp(gammaLPDF(x, shape + 1, 2) + log(2 * scale));
    real sum = 1;
    real term = 1;
    foreach (j; 1 .. maxMixtureSteps + 1)
    {
        const ratio = (x / 2) / (shape + j);
        term *= ratio;
        if (term <= (1 - ratio) * real.epsilon * sum)
            return first * sum;
        sum += term;
    }
    return real.nan;
}

private @safe pure nothrow @nogc
real probability(real x, uint k, real lambda, bool upper, real resultScale = 1)
{
    import mir.stat.distribution.gamma: gammaCCDF;
    import mir.stat.distribution.poisson: poissonLPMF;
    import mir.math.common: exp;

    if (x == 0 || x == real.infinity)
        return (x == 0) == upper ? resultScale : 0;
    // Markov's inequality applied to exp(X/4) bounds the upper tail by
    // exp(k*log(2)/2 + lambda/2 - x/4). Use log(2) < 1 in both
    // the moment and the underflow threshold to keep the test conservative.
    if (x / 4 > lambda / 2 + real(k) / 2
            + (2 - real.min_exp + real.mant_dig))
        return upper ? 0 : resultScale;
    const mean = lambda / 2;
    if (mean == 0)
        return upper ? resultScale * chi2CCDF(x, k)
                     : scaledGammaLowerTail(x, real(k) / 2, resultScale);
    if (mean >= uint.max)
        return real.nan;

    // Use the upper gamma integral above the mean and the lower one below.
    // Only subtract from one on the side with substantial probability.
    const useUpper = x > k + lambda;
    const shape = real(k) / 2;
    real tail(uint j) {
        return useUpper ? resultScale * gammaCCDF(x, shape + j, 2)
                        : scaledGammaLowerTail(x, shape + j, resultScale);
    }
    const mode = cast(uint) mean;
    // A representable subnormal tail can consist entirely of terms below
    // the smallest positive value. Scale before multiplying by each tail.
    enum weightScale = 1 / real.epsilon;
    const peak = weightScale * exp(poissonLPMF(mode, mean));
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
            const boundFactor = (upward != useUpper) ? component : resultScale;
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
    sum /= weightScale;
    if (sum > resultScale) sum = resultScale;
    return upper == useUpper ? sum : resultScale - sum;
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

/++
Computes the noncentral chi-squared quantile at cumulative probability $(D p).

Params:
    T = built-in floating-point result and argument type
    p = lower-tail probability in the closed interval [0, 1]
    k = positive integer degrees of freedom
    lambda = finite, nonnegative noncentrality

Returns:
    The nonnegative quantile as $(D T), with zero at $(D p == 0) and
    positive infinity at $(D p == 1). An unresolved tail returns NaN.
+/
@safe pure nothrow @nogc
T noncentralChi2InvCDF(T)(const T p, const uint k, const T lambda)
    if (isFloatingPoint!T)
    in (p >= 0 && p <= 1, "probability must be in [0, 1]")
    in (k > 0, "degrees of freedom must be positive")
    in (lambda >= 0 && lambda < T.infinity, "noncentrality must be finite and nonnegative")
{
    import mir.math.common: sqrt;
    if (p == 0) return 0;
    if (p == 1) return T.infinity;
    if (lambda == 0) return chi2InvCDF(p, k);

    const upper = p > T(0.5);
    // At a subnormal lower-tail probability, compare before rounding
    // the mixture back to the public probability scale.
    const scale = p < real.min_normal ? 1 / real.epsilon : 1;
    const real target = (upper ? 1 - p : p) * scale;
    if (p < real.min_normal)
    {
        import mir.math.common: exp, log;
        import mir.math.constant: LN2;
        import std.mathspecial: logGamma;
        // Near zero, F(x) is exp(-lambda/2)*(x/2)^^shape/Gamma(shape+1)
        // times a factor between exp(-x/2) and exp(lambda*x/(4*(shape+1))).
        // Invert in log space when both corrections are below roundoff.
        const shape = real(k) / 2;
        const logX = LN2 + (log(real(p)) + real(lambda) / 2 + logGamma(shape + 1)) / shape;
        if (logX < log(real.epsilon) - log(1 + real(lambda)))
            return cast(T) exp(logX);
    }
    const center = real(k) + lambda;
    real step = sqrt(2 * (real(k) + 2 * real(lambda)));
    real low = center;
    real high = center;
    // Expand by standard deviations around the mean. A multiple of the
    // mean can be thousands of standard deviations from a central quantile.
    const centerValue = probability(center, k, lambda, upper, scale);
    if (centerValue != centerValue) return T.nan;
    const moveRight = upper ? centerValue > target : centerValue < target;
    for (;;)
    {
        if (moveRight)
            high = center + step;
        else
            low = step < center ? center - step : 0;
        const value = probability(moveRight ? high : low, k, lambda, upper, scale);
        if (value != value) return T.nan;
        if ((upper ? value > target : value < target) != moveRight)
            break;
        if (moveRight) low = high;
        else high = low;
        step *= 2;
    }

    // Bisect the exponent range while the endpoints span several powers
    // of two. Ordinary bisection then resolves the significand.
    enum smallest = real.min_normal * real.epsilon;
    for (;;)
    {
        const positiveLow = low > 0 ? low : smallest;
        const mid = positiveLow < high / 4
            ? sqrt(positiveLow) * sqrt(high)
            : low + (high - low) / 2;
        if (mid <= low || mid >= high || cast(T) low == cast(T) high)
            return cast(T) mid;
        const value = probability(mid, k, lambda, upper, scale);
        if (value != value) return T.nan;
        if (upper ? value > target : value < target)
            low = mid;
        else
            high = mid;
    }
}

///
version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    const radiusSquared = noncentralChi2InvCDF(0.95, 2, 1.0);
    assert(approxEqual(radiusSquared, 8.642203870045899));
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(noncentralChi2InvCDF(T(0), 5, T(3)) == 0);
        assert(noncentralChi2InvCDF(T(1), 5, T(3)) == T.infinity);
        assert(noncentralChi2InvCDF(T(0.4), 3, T(0)) == chi2InvCDF(T(0.4), 3));
        foreach (p; [T(1e-10), T(0.5), T(1) - T.epsilon])
        {
            const x = noncentralChi2InvCDF(p, 3, T(4));
            const actual = p > T(0.5) ? noncentralChi2CCDF(x, 3, T(4))
                                       : noncentralChi2CDF(x, 3, T(4));
            const target = p > T(0.5) ? 1 - p : p;
            assert(approxEqual(actual, target, T(1024) * T.epsilon, T(0)));
        }
    }}
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    // For X = (Z + sqrt(lambda))^^2 the median rounds to lambda here.
    assert(approxEqual(noncentralChi2InvCDF(0.5, 1, 100_000.0),
        100_000.0, 1e-11, 0.0));
    // For k = 1 the upper tail is the sum of two normal tails.
    assert(approxEqual(noncentralChi2InvCDF(1 - double.epsilon, 1, 80.0),
        291.3904503267249, 2e-15, 0.0));
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import std.math: isNaN;
    // For one degree of freedom the two normal tails give 13 minimum
    // double subnormals after rounding (the farther tail rounds to zero).
    enum minDouble = double.min_normal * double.epsilon;
    assert(noncentralChi2CCDF(48.4 * 48.4, 1, 100.0) == 13 * minDouble);
    // At k=2 and lambda=2 the density at zero is 1/(2e).
    assert(noncentralChi2InvCDF(minDouble, 2, 2.0) == 5 * minDouble);
    // Inverting the two normal tails of (Z + 40)^^2 gives an ordinary
    // positive quantile even at the smallest double probability.
    assert(approxEqual(noncentralChi2InvCDF(minDouble, 1, 1600.0),
        2.34884554236070, 1e-13, 0.0));
    // Independent reference for a lower tail far below the distribution mean.
    assert(approxEqual(noncentralChi2CDF(700.0, 5, 1600.0),
        1.9075094934193678e-42, 5e-13, 0.0));
}

version(mir_stat_test_noncentral_chi2)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: log, sqrt;
    import std.math: isNaN;

    // The density series is too broad to resolve within the work limit.
    assert(isNaN(noncentralChi2LPDF(1e8, 2, 1e8)));
    // The lower-tail mixture reaches its work limit before it can bound
    // the omitted Poisson mass on the side toward zero.
    assert(isNaN(noncentralChi2CDF(1.0, 2, 1e7)));
    // Near a large gamma shape, the lower-tail series can converge slowly
    // even at the normal/subnormal boundary. Choose the distance in units
    // of sqrt(shape) for the platform's real exponent range.
    const shape = real(uint.max);
    const distance = sqrt(-2 * shape * log(real.min_normal));
    assert(isNaN(scaledGammaLowerTail(2 * (shape - distance), shape, 1 / real.epsilon)));
    assert(isNaN(noncentralChi2LPDF(1e20, 2, 1e20)));
    assert(isNaN(noncentralChi2CDF(1e20, 2, 1e20)));
    assert(isNaN(noncentralChi2InvCDF(0.5, 2, 1e20)));
}
