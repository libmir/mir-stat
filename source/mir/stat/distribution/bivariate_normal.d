/++
Bivariate normal distribution with correlation $(D rho).

$(LREF bivariateNormalPDF), $(LREF bivariateNormalLPDF), and $(LREF bivariateNormalCDF)
take two coordinates in the original measurement units, two marginal means, two
standard deviations, and a correlation. For standardized coordinates use the
scalar functions below.

$(LREF bivariateStandardNormalCDF) evaluates a lower-orthant probability.
$(LREF bivariateStandardNormalPDF) and $(LREF bivariateStandardNormalLPDF)
evaluate the joint density and its natural logarithm.
$(LREF bivariateStandardNormalRectangleProbability) accepts lower and upper
bounds for each standardized coordinate. $(LREF bivariateNormalRectangleProbability)
accepts bounds in the original measurement units. At perfect correlation the
distribution is supported on a line and interval probabilities reduce to univariate calculations.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.bivariate_normal;

private import mir.internal.utility: isFloatingPoint;
private import mir.math.common: exp, fabs, log, sqrt;
private import mir.math.constant: PI;
private import mir.stat.constant: LOGSQRT2PI;
private import mir.stat.internal.linearAlgebra.types: Element, vector;
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

/++
Evaluate the natural log of the joint density for a bivariate normal distribution.
The vectors may independently be D arrays or one-dimensional Mir slices,
including strided views. Coordinate order is the same in all three vectors.
The correlation is converted to their common floating-point element type.

Params:
    point = Two observed coordinates in the original measurement units.
    mean = Two finite marginal means with the same floating-point element type as point.
    stdDev = Two positive finite marginal standard deviations with the same element type as point.
    rho = Correlation strictly between -1 and 1.

Returns:
    Natural log density in the element type of point. An infinite coordinate gives
    negative infinity. A NaN coordinate propagates.
+/
Element!P bivariateNormalLPDF(P, M, S, R)(scope auto ref P point, scope auto ref M mean,
    scope auto ref S stdDev, const R rho)
    if (bivariateVectors!(P, M, S) && is(R : Element!P))
{
    const z = standardizedPair(point, mean, stdDev);
    return bivariateStandardNormalLPDF!(Element!P)(z[0], z[1], rho) - (log(stdDev[0]) + log(stdDev[1]));
}

/// At the mean, changing the measurement units adjusts the log density by the log scales.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs, log;
    double[2] mean = [100, 50], stdDev = [10, 5];
    assert(fabs(bivariateNormalLPDF(mean, mean, stdDev, 0.4) -
        (bivariateStandardNormalLPDF(0.0, 0.0, 0.4) - log(50.0))) < 1e-14);
}

/++
Evaluate the joint density for a bivariate normal distribution.
The vectors may independently be D arrays or one-dimensional Mir slices,
including strided views. Coordinate order is the same in all three vectors.
The correlation is converted to their common floating-point element type.

Params:
    point = Two observed coordinates in the original measurement units.
    mean = Two finite marginal means with the same floating-point element type as point.
    stdDev = Two positive finite marginal standard deviations with the same element type as point.
    rho = Correlation strictly between -1 and 1.

Returns:
    Joint density in the element type of point. The result may exceed one.
    An infinite coordinate gives zero. A NaN coordinate propagates.
+/
Element!P bivariateNormalPDF(P, M, S, R)(scope auto ref P point, scope auto ref M mean,
    scope auto ref S stdDev, const R rho)
    if (bivariateVectors!(P, M, S) && is(R : Element!P))
{
    return exp(bivariateNormalLPDF(point, mean, stdDev, rho));
}

/// Independent measurements have a joint density equal to the product of their marginal densities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    import mir.stat.distribution.normal: normalPDF;
    double[2] point = [110, 45], mean = [100, 50], stdDev = [10, 5];
    const density = bivariateNormalPDF(point, mean[], stdDev, 0.0);
    assert(fabs(density / (normalPDF(110.0, 100.0, 10.0) * normalPDF(45.0, 50.0, 5.0)) - 1) < 1e-14);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;
    foreach (T; AliasSeq!(float, double, real))
    {
        // Select each vector from a separate column of the same matrix.
        const T[6] data = [5, 1, 2, 2, 5, 3];
        auto rows = data[].sliced(2, 3);
        auto point = rows[0 .. 2, 0], mean = rows[0 .. 2, 1], stdDev = rows[0 .. 2, 2];
        static assert(!__traits(compiles, bivariateNormalLPDF(rows, mean, stdDev, 0)));
        static assert(!__traits(compiles, bivariateNormalPDF(point, mean, int[2].init, 0)));
        const rho = T(-0.5);
        const expected = bivariateStandardNormalLPDF(T(2), T(-1), rho) - log(T(6));
        const actual = bivariateNormalLPDF(point, mean, stdDev, rho);
        static assert(is(typeof(actual) == const T));
        assert(fabs(actual - expected) < 8 * T.epsilon);
        assert(fabs(bivariateNormalPDF(point, mean, stdDev, rho) / exp(expected) - 1) < 16 * T.epsilon);
    }

    // The standardized density underflows, but the density in the original units is finite.
    double[2] scales = [0x1p-600, 0x1p-600], origin = [0, 0], tail = [40 * scales[0], 0];
    const expectedLog = -800 - 2 * double(LOGSQRT2PI) + 1200 * log(2.0);
    assert(fabs(bivariateNormalPDF(tail, origin, scales, 0) / exp(expectedLog) - 1) < 3e-13);
    tail[0] = double.infinity;
    assert(bivariateNormalLPDF(tail, origin, scales, 0) == -double.infinity);
    tail[1] = double.nan;
    import std.math: isNaN;
    assert(isNaN(bivariateNormalPDF(tail, origin, scales, 0)));
}

/++
Evaluate the probability that both coordinates are at or below their thresholds for a bivariate normal distribution.
The vectors may independently be D arrays or one-dimensional Mir slices,
including strided views. Coordinate order is the same in all three vectors.
The correlation is converted to their common floating-point element type.

Params:
    point = Two upper thresholds in the original measurement units.
    mean = Two finite marginal means with the same floating-point element type as point.
    stdDev = Two positive finite marginal standard deviations with the same element type as point.
    rho = Correlation from -1 through 1, inclusive.

Returns:
    Lower-orthant probability in the element type of point. A negative infinite threshold
    gives zero and a positive infinite threshold removes that coordinate's restriction.
    A NaN threshold propagates. As in $(LREF bivariateStandardNormalCDF), the result
    is a deterministic estimate and very small probabilities may underflow.
+/
Element!P bivariateNormalCDF(P, M, S, R)(scope auto ref P point, scope auto ref M mean,
    scope auto ref S stdDev, const R rho)
    if (bivariateVectors!(P, M, S) && is(R : Element!P))
{
    const z = standardizedPair(point, mean, stdDev);
    return bivariateStandardNormalCDF!(Element!P)(z[0], z[1], rho);
}

/// Each column holds two readings. Select one column directly to evaluate both thresholds.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    double[4] data = [90, 110, 55, 45];
    auto readings = data[].sliced(2, 2);
    double[2] mean = [100, 50], stdDev = [10, 5];
    auto probability = bivariateNormalCDF(readings[0 .. 2, 1], mean[], stdDev, 0.4);
    assert(probability == bivariateStandardNormalCDF(1.0, -1.0, 0.4));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    foreach (T; AliasSeq!(float, double, real))
    {
        // The first centered difference overflows, but the standardized threshold is two.
        T[2] point = [T.max, 1], mean = [-T.max, 4], stdDev = [T.max, 3];
        foreach (rho; [T(-1), T(0), T(0.75), T(1)])
            assert(bivariateNormalCDF(point[], mean, stdDev[], rho) == bivariateStandardNormalCDF(T(2), T(-1), rho));
        point[0] = T.infinity;
        assert(bivariateNormalCDF(point, mean, stdDev, 1) == normalCDF(T(-1)));
        point[0] = -T.infinity;
        assert(bivariateNormalCDF(point, mean, stdDev, -1) == 0);
        point[1] = T.nan;
        assert(isNaN(bivariateNormalCDF(point, mean, stdDev, 0)));
    }
}

/++
Evaluate the probability that each standardized coordinate lies between its
lower and upper bounds. Both vectors may be D arrays or one-dimensional Mir
slices, including strided views. The calculation is deterministic.

Params:
    lower = Two lower bounds in coordinate order.
    upper = Two corresponding upper bounds with the same floating-point
        element type. Each must be at least its lower bound. Either bound may be infinite.
    rho = Correlation from -1 through 1, inclusive.

Returns:
    Rectangle probability in the element type of the bounds. A zero-width
    coordinate gives zero and a NaN bound propagates. Perfect correlation
    reduces to a univariate interval. Quadrature uses a fixed refinement budget
    and returns its best estimate. Very small probabilities may underflow.
+/
Element!L bivariateStandardNormalRectangleProbability(L, U, R)(
    scope auto ref L lower, scope auto ref U upper, const R rho)
    if (vector!L && vector!U && is(Element!L == Element!U) && is(R : Element!L))
    in (lower.length == 2 && upper.length == 2)
    in (cast(Element!L)rho >= -1 && cast(Element!L)rho <= 1)
{
    import mir.stat.internal.bivariate_normal_rectangle: rectangleEstimate;
    alias T = Element!L;
    static if (is(T == float)) alias W = double;
    else alias W = T;
    return cast(T)rectangleEstimate!W(lower[0], upper[0], lower[1], upper[1], cast(T)rho).probability;
}

/// Both independent readings must be within one standard deviation of their means.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    double[2] lower = [-1, -1], upper = [1, 1];
    const p = bivariateStandardNormalRectangleProbability(lower, upper[], 0);
    assert(fabs(p - 0.4660649426743923) < 1e-15);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.ndslice.slice: sliced;
    foreach (T; AliasSeq!(float, double, real))
    {
        alias prob = bivariateStandardNormalRectangleProbability;
        const T[4] bounds = [-1, 0.75, -0.5, 1.25];
        auto table = bounds[].sliced(2, 2);
        const p = prob(table[0 .. 2, 0], table[0 .. 2, 1], T(0.4));
        static assert(is(typeof(p) == const T));
        assert(fabs(p - T(0.371616434006933143761L)) < 32 * T.epsilon);

        T[2] lo = [-100000, -100000], hi = [100000, 100000];
        assert(fabs(prob(lo, hi, T(0.5)) - 1) <= 8 * T.epsilon);
        lo = [T(0), T(-2)];
        hi = [T(1), T(-1)];
        assert(prob(lo, hi, T(-1)) == 0);
        lo[0] = hi[0];
        assert(prob(lo, hi, 0) == 0);
        lo[1] = T.nan;
        assert(isNaN(prob(lo, hi, 0)));
    }
    alias prob = bivariateStandardNormalRectangleProbability;
    // Adjacent endpoints in each coordinate can coincide after rotation.
    double[2] lo = [-2, 9], hi = [-0x1.fffffffffffffp0, 0x1.2000000000001p3];
    assert(fabs(prob(lo, hi, 0.5) / 1.093214828440718926555e-61 - 1) < 1e-13);
    lo = [0, 5];
    hi = [0x1p-51, 0x1.4000000000001p2];
    assert(fabs(prob(lo, hi, 0.5) / 4.18811475425843043874e-39 - 1) < 1e-13);
    lo = [0, 1];
    hi = [1, 2];
    assert(fabs(prob(lo, hi, 1 - 0x1p-52) / 2.034267221349542553e-9 - 1) < 1e-13);
    lo = [-double.infinity, -double.infinity];
    assert(prob(lo, hi, 0.5) == bivariateStandardNormalCDF(1.0, 2.0, 0.5));
    lo = [8, 8];
    hi = [9, 9];
    assert(fabs(prob(lo, hi, 0) / 3.8686309932399678e-31 - 1) < 1e-14);
    lo = [-0.5, 2];
    hi = [1, double.infinity];
    assert(fabs(prob(lo, hi, 0.6) / 0.006711394883858296 - 1) < 1e-14);
    lo = [-double.infinity, 2];
    hi = [double.infinity, 3];
    assert(prob(lo, hi, -0.5) == normalCDF(-2.0) - normalCDF(-3.0));
    lo = [0, -1.5];
    hi = [1, -0.5];
    assert(fabs(prob(lo, hi, -1) - 0.14988228479452984) < 1e-15);

    // At independence the joint mass is about 0.539 subnormal units.
    // Rounding the first marginal before multiplying can give zero instead.
    const unit = double.min_normal * double.epsilon;
    lo = [0, -double.infinity];
    hi = [3 * unit, -0.125];
    assert(prob(lo, hi, 0) == unit);
    hi[0] = 2 * unit;
    assert(prob(lo, hi, 0) == 0);

    // Independent integration gives 0.604 of a double subnormal unit in
    // the first rectangle. In the second the reference rounds to zero.
    lo[] = 33.25;
    hi[] = 33.5;
    assert(prob(lo, hi, 0.5) == double.min_normal * double.epsilon);
    lo[] = 33.275;
    hi[] = 33.525;
    assert(prob(lo, hi, 0.5) == 0);

    // The intervals are close to the mean, but separated by 38.1
    // conditional standard deviations. This exercises the scaled retry.
    const rho = 0.9999;
    lo = [0, 1 + 38.1 * sqrt(1 - rho * rho)];
    hi = [1, 2];
    assert(prob(lo, hi, rho) == 89 * (double.min_normal * double.epsilon));

    static if (real.max_exp > double.max_exp)
    {
        // This rectangle lies beyond the double-precision quadrature cutoff.
        real[2] lower = [40, -40.25L], upper = [40.25L, -40];
        import std.math: ldexp;
        const reference = ldexp(0.610170627348307500172L, -4631);
        assert(fabs(prob(lower, upper, 0.5L) / reference - 1) < 3e-16L);
        static if (real.mant_dig == 64 && real.min_exp == -16381)
        {
            // Corresponding boundary in the 80-bit extended format.
            lower[] = 130.70L;
            upper[] = 130.95L;
            assert(prob(lower, upper, 0.5L) == real.min_normal * real.epsilon);
            lower[] = 130.71L;
            upper[] = 130.96L;
            assert(prob(lower, upper, 0.5L) == 0);
        }
    }
}

/++
Evaluate the probability that both measurements lie within their respective
intervals, expressed in the original measurement units. The four vectors
may independently be D arrays or one-dimensional Mir slices, including
strided views. Use the same coordinate order and floating-point element
type for all four vectors.

Params:
    lower = Two lower measurement limits, possibly infinite.
    upper = Two upper measurement limits, each at least its corresponding lower limit.
    mean = Two finite marginal means.
    stdDev = Two positive finite marginal standard deviations.
    rho = Correlation from -1 through 1, inclusive, converted to the element type.

Returns:
    Joint rectangle probability in the element type of the bounds. A NaN bound
    propagates, and an interval of zero width gives zero. Perfect correlation
    reduces to a univariate interval. As for $(LREF bivariateStandardNormalRectangleProbability),
    quadrature is deterministic with a fixed refinement budget and very small
    probabilities may underflow.
+/
Element!L bivariateNormalRectangleProbability(L, U, M, S, R)(scope auto ref L lower,
    scope auto ref U upper, scope auto ref M mean, scope auto ref S stdDev, const R rho)
    if (bivariateVectors!(L, M, S) && vector!U && is(Element!L == Element!U) && is(R : Element!L))
    in (lower.length == 2 && upper.length == 2 && mean.length == 2 && stdDev.length == 2)
    in (cast(Element!L)rho >= -1 && cast(Element!L)rho <= 1)
{
    import mir.stat.internal.bivariate_normal_rectangle: originalRectangleProbability;
    alias T = Element!L;
    static if (is(T == float)) alias W = double;
    else alias W = T;
    return cast(T)originalRectangleProbability!W(lower, upper, mean, stdDev, cast(T)rho);
}

/// A part passes inspection when both its length and its width fall within tolerance.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    double[2] lower = [9, 4], upper = [11, 6], mean = [10, 5], stdDev = [1, 0.5];
    const pass = bivariateNormalRectangleProbability(lower, upper[], mean, stdDev, 0);
    assert(fabs(pass - 0.6516269400855773) < 1e-14);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    import mir.ndslice.slice: sliced;
    import std.math: isNaN;
    import std.meta: AliasSeq;
    alias prob = bivariateNormalRectangleProbability;
    foreach (T; AliasSeq!(float, double, real))
    {
        const T[8] data = [8, 11, 10, 2, 2, 8, 5, 3];
        auto table = data[].sliced(2, 4);
        T[2] zl = [-1, -1], zu = [0.5, 1];
        foreach (rho; [T(-1), T(-0.999), T(0), T(0.75), T(1)])
            assert(prob(table[0 .. 2, 0], table[0 .. 2, 1], table[0 .. 2, 2], table[0 .. 2, 3], rho) ==
                bivariateStandardNormalRectangleProbability(zl, zu, rho));

        T[2] lo = [0, -T.infinity], hi = [T(0x1p-80), T.infinity];
        T[2] mean = [-1, 0], sd = [1, 1];
        // Both standardized endpoints round to one. The original width survives.
        const expected = hi[0] * T(0.241970724519143349798L);
        foreach (rho; [T(-1), T(0), T(0.5), T(1)])
            assert(fabs(prob(lo, hi, mean, sd, rho) / expected - 1) < 8 * T.epsilon);

        // Both coordinate intervals collapse after standardization.
        lo[1] = 0;
        hi[1] = 2 * hi[0];
        mean[1] = -2;
        sd[1] = 2;
        static if (T.max_exp > 128)
        {
            const reference = hi[0] * hi[0] * T(0.09435389770895923001736674973854711436L);
            assert(fabs(prob(lo, hi, mean, sd, 0.5) / reference - 1) < 32 * T.epsilon);
        }

        lo = [-T.max, -T.infinity];
        hi = [T.max, T.infinity];
        mean = [T(0), T(0)];
        sd = [T.max, T(1)];
        assert(fabs(prob(lo, hi, mean, sd, 0.5) - T(0.68268949213708589717L)) < 8 * T.epsilon);
        lo[0] = T.infinity;
        hi[0] = T.infinity;
        assert(prob(lo, hi, mean, sd, 1) == 0);
        hi[1] = T.nan;
        assert(isNaN(prob(lo, hi, mean, sd, -1)));
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: fabs;
    import std.meta: AliasSeq;
    foreach (T; AliasSeq!(float, double, real))
    {
        // In standardized units both intervals have endpoints that round to one.
        // Their intersection is half the first interval, or a single point.
        const h = T(0x1p-80);
        T[2] lower = [T(0), h], upper = [h, 3 * h];
        T[2] mean = [-1, -2], sd = [1, 2];
        const halfMass = h * T(0.241970724519143349798L) / 2;
        alias prob = bivariateNormalRectangleProbability;
        assert(fabs(prob(lower, upper, mean, sd, 1) / halfMass - 1) < 8 * T.epsilon);
        lower[1] = 2 * h;
        assert(prob(lower, upper, mean, sd, 1) == 0);

        // With a common mean and scale the original bounds give the overlap directly.
        T[2] commonLower = [T(0), h / 2], commonUpper = [h, 2 * h];
        T[2] commonMean = [-1, -1], commonSd = [1, 1];
        assert(fabs(prob(commonLower, commonUpper, commonMean, commonSd, 1) / halfMass - 1) < 8 * T.epsilon);

        // Reflect the second coordinate to obtain the same half overlap.
        lower[1] = -3 * h;
        upper[1] = -h;
        mean[1] = 2;
        assert(fabs(prob(lower, upper, mean, sd, -1) / halfMass - 1) < 8 * T.epsilon);
    }
}

private enum bivariateVectors(P, M, S) = vector!P && vector!M && vector!S &&
    is(Element!P == Element!M) && is(Element!P == Element!S);

private Element!P[2] standardizedPair(P, M, S)(scope auto ref P point, scope auto ref M mean,
    scope auto ref S stdDev)
    in (point.length == 2, "point must contain two coordinates")
    in (mean.length == 2, "mean must contain two coordinates")
    in (stdDev.length == 2, "stdDev must contain two standard deviations")
{
    import mir.stat.internal.standardize: standardize;
    alias T = Element!P;
    T[2] z;
    foreach (i; 0 .. 2)
    {
        assert(fabs(mean[i]) <= T.max, "means must be finite");
        assert(stdDev[i] > 0 && stdDev[i] <= T.max, "standard deviations must be positive and finite");
        z[i] = standardize(point[i], mean[i], stdDev[i]);
    }
    return z;
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
