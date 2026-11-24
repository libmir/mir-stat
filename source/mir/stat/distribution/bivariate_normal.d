/++
Bivariate normal distribution with standard normal marginals and correlation $(D rho).

$(LREF bivariateStandardNormalCDF) evaluates a lower-orthant probability.
$(LREF bivariateStandardNormalPDF) and $(LREF bivariateStandardNormalLPDF)
evaluate the joint density and its natural logarithm. At perfect correlation
the distribution is supported on a line and only the CDF is defined.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.bivariate_normal;

private import mir.internal.utility: isFloatingPoint;
private import mir.math.common: exp, fabs, log, sqrt;
private import mir.math.constant: PI;
private import mir.stat.constant: LOGSQRT2PI;
private import mir.stat.distribution.normal: normalCDF;

@safe pure nothrow @nogc:

/++
Evaluate the joint log density of two standardized normal observations.

Params:
    x = First standardized observation.
    y = Second standardized observation.
    rho = Correlation, strictly between -1 and 1.

Returns:
    Natural log density in the input type. An infinite coordinate gives
    negative infinity. A NaN coordinate propagates.
+/
T bivariateStandardNormalLPDF(T)(const T x, const T y, const T rho)
    if (isFloatingPoint!T)
    in (rho > -1 && rho < 1)
{
    if (x != x || y != y) return T.nan;
    if (fabs(x) == T.infinity || fabs(y) == T.infinity) return -T.infinity;
    // Rotate onto the sum and difference axes. Halve before adding so
    // an intermediate sum or square does not overflow unnecessarily.
    const u = x * T(0.5) + y * T(0.5);
    const v = x * T(0.5) - y * T(0.5);
    const logScale = T(2 * LOGSQRT2PI) + T(0.5) * log((1 - rho) * (1 + rho));
    return -u * (u / (1 + rho)) - v * (v / (1 - rho)) - logScale;
}

/// The joint density at the mean increases with positive correlation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    assert(fabs(bivariateStandardNormalLPDF(0.0, 0.0, 0.5) + 1.694036030183455) < 1e-14);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.math: isNaN;
    // Squaring x before applying the half factor would overflow.
    assert(bivariateStandardNormalLPDF(0x1p512, 0.0, 0.0) == -0x1p1023);
    // Dividing by sqrt(1.5) first rounds the normalized coordinate to 2^512.
    enum x = 0x1.3988e1409212ep+512;
    assert(bivariateStandardNormalLPDF(x, x, 0.5) == -0x1.ffffffffffffep+1023);
    assert(bivariateStandardNormalLPDF(x, -x, -0.5) == -0x1.ffffffffffffep+1023);
    enum next = 0x1.3988e1409212fp+512;
    assert(bivariateStandardNormalLPDF(next, next, 0.5) == -double.infinity);
    assert(bivariateStandardNormalLPDF(double.infinity, 0.0, 0.5) == -double.infinity);
    assert(isNaN(bivariateStandardNormalLPDF(double.infinity, double.nan, 0.0)));
}

/++
Evaluate the joint density of two standardized normal observations.

Params:
    x = First standardized observation.
    y = Second standardized observation.
    rho = Correlation, strictly between -1 and 1.

Returns:
    Joint density in the input type. The result can exceed one.
    An infinite coordinate gives zero. A NaN coordinate propagates.
+/
T bivariateStandardNormalPDF(T)(const T x, const T y, const T rho)
    if (isFloatingPoint!T)
    in (rho > -1 && rho < 1)
{
    return exp(bivariateStandardNormalLPDF(x, y, rho));
}

/// Independent standardized coordinates have the product of their marginal densities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    import mir.stat.distribution.normal: normalPDF;
    assert(fabs(bivariateStandardNormalPDF(1.0, -1.0, 0.0) -
        normalPDF(1.0) * normalPDF(-1.0)) < 1e-15);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.covariance_views: covarianceView;
    import mir.stat.distribution.multivariate_normal: tryMultivariateNormalLPDFWithWorkspace;
    double[2] x = [1, -1], mean = [0, 0], work;
    double[4] factorWork;
    foreach (rho; [-0.75, 0.0, 0.9])
    {
        double[4] covariance = [1, rho, rho, 1];
        auto outcome = tryMultivariateNormalLPDFWithWorkspace(x[], mean[],
            covarianceView(covariance[].sliced(2, 2)), work[], factorWork[].sliced(2, 2));
        assert(outcome.failedPivot == 0);
        const reference = outcome.logDensity;
        assert(fabs(bivariateStandardNormalLPDF(x[0], x[1], rho) - reference) < 1e-14);
        assert(fabs(bivariateStandardNormalPDF(x[0], x[1], rho) / exp(reference) - 1) < 1e-14);
    }
}

/++
Evaluate the probability that $(D X <= x) and $(D Y <= y), where X and Y are
jointly normal with standard normal marginals and correlation $(D rho).
For correlations strictly between -1 and
1 the calculation uses deterministic one-dimensional quadrature of
$(LINK2 https://doi.org/10.1093/biomet/41.3-4.351, Plackett's correlation derivative).
The refinement uses a fixed internal budget and returns a probability estimate.

Params:
    x = Upper threshold for the first coordinate, possibly infinite.
    y = Upper threshold for the second coordinate, possibly infinite.
    rho = Correlation from -1 through 1, inclusive.

Returns:
    Lower-orthant probability in the input type. A negative infinite threshold
    gives zero and a positive infinite threshold removes that coordinate's
    restriction. At either endpoint of the correlation interval the probability
    reduces to a univariate normal probability. A NaN threshold propagates.
    Very small probabilities may underflow.
+/
T bivariateStandardNormalCDF(T)(const T x, const T y, const T rho)
    if (isFloatingPoint!T)
    in (rho >= -1 && rho <= 1)
{
    if (x != x || y != y) return T.nan;
    if (x == -T.infinity || y == -T.infinity) return 0;
    if (rho == 1) return normalCDF(x < y ? x : y);
    if (x == T.infinity) return normalCDF(y);
    if (y == T.infinity) return normalCDF(x);

    // Accumulate float probabilities in double, including marginal tails.
    static if (is(T == float)) alias W = double;
    else alias W = T;
    const W h = x, k = y, r = rho;
    if (rho == 0) return cast(T)(normalCDF(h) * normalCDF(k));
    const positive = r > 0;
    const W base = positive ? normalCDF(h) * normalCDF(k) : oppositeCorrelationMass(h, k);
    if (rho == -1) return cast(T)base;

    // Integrate Plackett's correlation derivative starting at independence
    // or at rho = -1, so the correction is always positive. With
    // |rho| = cos(theta), u = tan(theta/2) removes the singular Jacobian.
    const W ar = fabs(r);
    const W edge = sqrt((1 - ar) / (1 + ar));
    const W lo = positive ? edge : 0;
    const W hi = positive ? 1 : edge;
    const W a = fabs(h * W(0.5) - (positive ? k : -k) * W(0.5));
    const W b = fabs(h * W(0.5) + (positive ? k : -k) * W(0.5));
    W integrand(W u)
    {
        if (u == 0 && a != 0) return 0;
        const W p = a == 0 ? 0 : a / u;
        const W q = b * u;
        return exp(-W(0.5) * (a * a + b * b + p * p + q * q)) / (1 + u * u);
    }
    // Split at the maximum of the exponential, so a narrow interior peak
    // is visible to the endpoint-clustered quadrature on both sides.
    const W peak = b > 0 ? sqrt(a / b) : hi;
    W integral;
    if (peak > lo && peak < hi)
        integral = integrate!integrand(lo, peak) + integrate!integrand(peak, hi);
    else
        integral = integrate!integrand(lo, hi);
    const W value = base + integral / W(PI);
    return cast(T)(value < 1 ? value : 1);
}

/// With correlation one half, both coordinates are below their means one third of the time.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    assert(fabs(bivariateStandardNormalCDF(0.0, 0.0, 0.5) - 1.0 / 3) < 1e-14);
}

/// Negate both thresholds to find the probability that both readings exceed them.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    // P(X > 1, Y > 2) for two standard normal readings with correlation 0.5.
    const double x = 1, y = 2, rho = 0.5;
    const bothAbove = bivariateStandardNormalCDF(-x, -y, rho);
    assert(fabs(bothAbove - 0.0132662170105167) < 1e-15);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: asin, isNaN;
    foreach (T; AliasSeq!(float, double, real))
    {
        const T inf = T.infinity;
        alias cdf = bivariateStandardNormalCDF!T;
        assert(cdf(-inf, 1, -1) == 0);
        assert(cdf(inf, 1, 0.25) == normalCDF(T(1)));
        assert(isNaN(cdf(-inf, T.nan, 0)));
        assert(cdf(1, -1, 1) == normalCDF(T(-1)));
        assert(cdf(1, -1, -1) == 0);
        assert(cdf(-1, 2, -1) == cdf(2, -1, -1));
        assert(fabs(cdf(2, -1, -1) - (normalCDF(T(-1)) - normalCDF(T(-2)))) < 4 * T.epsilon);
        assert(fabs(cdf(0, 0, T(-0.95)) - (0.25L + asin(T(-0.95)) / (2 * PI))) < 8 * T.epsilon);
        assert(fabs(bivariateStandardNormalPDF(T(0), T(0), T(0)) - 1 / (2 * PI)) < T.epsilon);
    }
    // References from an independent integral of the conditional normal CDF.
    // The positive-correlation integral has an interior peak.
    assert(fabs(bivariateStandardNormalCDF(-1.0, -2.0, 0.75) / 0.01969626510877306 - 1) < 2e-14);
    // This far tail exercises the estimate at the refinement limit in double precision.
    assert(fabs(bivariateStandardNormalCDF(-4.0, -0.5, -0.99) / 6.558642537464e-226 - 1) < 5e-13);
    assert(fabs(bivariateStandardNormalCDF(-9.0, -9.0, -0.6) / 4.428735282213935e-92 - 1) < 2e-13);
    assert(fabs(bivariateStandardNormalCDF(0.97, -1.0, -0.999999) / 5.879080566503871e-105 - 1) < 2e-13);
    // At perfect negative correlation the resulting interval may be much
    // narrower than the spacing between representable marginal CDF values.
    const width = double.epsilon;
    assert(fabs(bivariateStandardNormalCDF(1 + width, -1.0, -1.0) / width -
        exp(-0.5 - double(LOGSQRT2PI))) < 2e-16);
}

private:

// At rho = -1, the event is -y <= X <= x. Reflect right tails before
// differencing. For narrow intervals integrate the density to avoid losing
// significant digits when the two marginal probabilities nearly coincide.
T oppositeCorrelationMass(T)(const T x, const T y)
{
    if (x <= -y) return 0;
    const width = x + y;
    if (width * (1 + fabs(x)) < T(0.5))
    {
        T density(T t) { return exp((-T(0.5) * t) * t - T(LOGSQRT2PI)); }
        return integrate!density(-y, x);
    }
    if (x > y) return normalCDF(y) - normalCDF(-x);
    return normalCDF(x) - normalCDF(-y);
}

// Double-exponential quadrature on a finite interval. Evaluate symmetrically
// from the two endpoints; forming 1 - t would lose the nearest right nodes.
T integrate(alias f, T)(const T lo, const T hi)
{
    import std.math: cosh, sinh;
    const width = hi - lo;
    if (width == 0) return 0;
    T sum = T(PI / 4) * f(lo + width / 2);
    T previous = 0;
    foreach (level; 0 .. 9)
    {
        const T step = T(1) / (1 << level);
        T extra = 0;
        for (int j = 1; j * step <= 4; j += level ? 2 : 1)
        {
            const T s = j * step;
            const T e = exp(-T(PI) * sinh(s));
            const T den = 1 + e;
            const T offset = width * (e / den);
            extra += (T(PI) * cosh(s) * e / (den * den)) *
                (f(lo + offset) + f(hi - offset));
        }
        sum += extra;
        const T current = step * sum;
        if (level >= 4 && fabs(current - previous) <= 8 * T.epsilon * current)
            return width * current;
        previous = current;
    }
    return width * previous;
}
