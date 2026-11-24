/++
Bivariate normal distribution with standard normal marginals and correlation $(D rho).

$(LREF bivariateStandardNormalPDF) and $(LREF bivariateStandardNormalLPDF)
evaluate the joint density and its natural logarithm.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.distribution.bivariate_normal;

private import mir.internal.utility: isFloatingPoint;
private import mir.math.common: exp, fabs, log, sqrt;
private import mir.stat.constant: LOGSQRT2PI;

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
