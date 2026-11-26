/++
Randomized integration of an ordered multivariate normal rectangle.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.normal_rectangle_integral;

private import mir.stat.distribution.normal : normalCDF, normalInvCDF;
private import mir.stat.internal.bivariate_normal_rectangle : normalIntervalMass;
private import mir.stat.internal.linearAlgebra.types : Element, matrix;
private import mir.stat.internal.standardize : standardize;
private import std.algorithm : max, min, clamp;
private import std.math : abs, isFinite, sqrt;

@safe pure nothrow @nogc:

version (mir_stat_test) version = mir_stat_test_rectangle_integral;
version (mir_stat_test_blas) version = mir_stat_test_rectangle_integral;

private immutable uint[7] bases = [2u, 3, 5, 7, 11, 13, 17];
private import mir.stat.internal.normal_rectangle_order : rectangleOrderMaxDimension;
package(mir.stat) enum rectangleMaxDimension = rectangleOrderMaxDimension;
static assert(bases.length + 1 == rectangleMaxDimension);
private enum shiftCount = 16;
private enum groupSize = shiftCount / 2;
package(mir.stat) enum rectangleMinEvaluations = 64 * shiftCount;
private alias Coordinates = double[rectangleMaxDimension - 1];

package(mir.stat) enum RectangleIntegralStatus { unresolved, budgetExhausted, toleranceMet }

package(mir.stat) struct RectangleIntegralResult
{
    RectangleIntegralStatus status;
    size_t evaluations;
    double probability;
    double estimatedError;
}

// Fraction obtained by reflecting the base-b digits of index across the radix point.
private double radicalInverse(size_t index, uint base)
{
    double place = 1.0 / base;
    double value = 0;
    while (index)
    {
        value += (index % base) * place;
        index /= base;
        place /= base;
    }
    return value;
}

// Select a truncated-normal quantile in [a,b]. Work with the smaller of the
// two tail probabilities, including for intervals reaching positive infinity.
private double sampleInterval(double a, double b, double mass, double u)
{
    const left = normalCDF(a);
    const right = normalCDF(-b);
    if (mass < 0x1p-26 * min(left, right))
    {
        // The CDF at an endpoint can absorb most or all of the interval mass.
        // Invert the local mass instead. Both endpoints here are finite.
        double lo = a, hi = b;
        const target = mass * u;
        foreach (_; 0 .. double.mant_dig + 1)
        {
            const mid = lo + (hi - lo) / 2;
            if (mid == lo || mid == hi) break;
            if (normalIntervalMass(a, mid, mid - a) < target) lo = mid;
            else hi = mid;
        }
        return lo + (hi - lo) / 2;
    }
    const lowerTail = left + u * mass;
    const upperTail = right + (1 - u) * mass;
    const q = max(min(lowerTail, upperTail), double.min_normal * double.epsilon);
    const z = normalInvCDF(q);
    return clamp(lowerTail <= upperTail ? z : -z, a, b);
}

private double rectangleWeight(F)(scope const(double)[] lower, scope const(double)[] upper,
    scope auto ref F factor, ref const Coordinates point, ref const Coordinates shift)
{
    Coordinates z;
    double product = 1;
    foreach (i; 0 .. lower.length)
    {
        double center = 0;
        foreach (j; 0 .. i) center += factor[i, j] * z[j];
        if (!isFinite(center)) return double.nan;
        const sd = factor[i, i];
        const a = standardize(lower[i], center, sd);
        const b = standardize(upper[i], center, sd);
        if (!(a < b)) return double.nan;
        const width = standardize(upper[i], lower[i], sd);
        const mass = normalIntervalMass(a, b, width);
        if (!(mass >= 0 && mass <= 1)) return double.nan;
        product *= mass;
        if (product == 0) break;
        if (i + 1 < lower.length)
        {
            import std.math : floor;
            const shifted = point[i] + shift[i];
            double u = shifted - floor(shifted);
            // Reflect tail intervals so small probabilities use the lower-tail inverse.
            if (a >= 0) z[i] = -sampleInterval(-b, -a, mass, u);
            else z[i] = sampleInterval(a, b, mass, u);
        }
    }
    return product;
}

// Three standard errors per eight-shift group. Normalize deviations before
// squaring to preserve the error estimate for small nonzero probabilities.
private double groupEstimate(scope const(double)[] sums, size_t count, out double error)
{
    double total = 0;
    foreach (sum; sums) total += sum;
    const mean = total / (groupSize * count);
    double largest = 0;
    foreach (sum; sums) largest = max(largest, abs(sum / count - mean));
    error = 0;
    if (largest > 0)
    {
        double squares = 0;
        foreach (sum; sums)
        {
            const scaled = (sum / count - mean) / largest;
            squares += scaled * scaled;
        }
        error = largest * (3 * sqrt(squares / (groupSize * (groupSize - 1))));
    }
    return mean;
}

version (mir_stat_test_rectangle_integral)
@safe pure nothrow @nogc unittest
{
    // The spread is representable, but directly squaring the deviations underflows.
    double[8] sums = [1e-200, 3e-200, 1e-200, 3e-200, 1e-200, 3e-200, 1e-200, 3e-200];
    double error;
    const mean = groupEstimate(sums[], 1, error);
    assert(abs(mean / 2e-200 - 1) < 1e-14);
    assert(abs(error / (3e-200 / sqrt(7.0)) - 1) < 1e-14);
}

/++
Integrate a positive-width rectangle after successful conditional ordering.
Each point contributes a product of conditional normal interval masses.
Sixteen independently shifted Halton sequences form two confirmation groups.

Params:
    lower = Ordered lower bounds relative to the mean, with 3 through 8 coordinates.
    upper = Corresponding strictly greater upper bounds. Infinite endpoints are allowed.
    factor = Lower Cholesky factor in the same coordinate order.
    seed = Seed for local random shifts. Repeating the inputs and seed in the
        same build reproduces the estimate and stopping decision.
    absoluteTolerance = Finite nonnegative absolute probability tolerance.
    relativeTolerance = Finite nonnegative tolerance relative to the estimate.
    maxEvaluations = Maximum number of integrand evaluations, at least 1024.
        Check after 64 points per shift, then quadruple the cumulative point
        count or use the remaining complete group of 16 evaluations.

Returns:
    Estimate, estimated absolute error, actual evaluation count and status.
    Tolerance is met when both group errors and the difference between the
    group means fit within the larger requested tolerance. This statistical
    error estimate is not a guaranteed bound. Budget exhaustion retains the
    last estimate. An unresolved computation has no usable error estimate.
+/
package(mir.stat) RectangleIntegralResult integrateNormalRectangle(F)(
    scope const(double)[] lower, scope const(double)[] upper, scope auto ref F factor,
    ulong seed, double absoluteTolerance, double relativeTolerance, size_t maxEvaluations)
    if (matrix!F && is(Element!F == double))
{
    import std.random : Mt19937_64;

    const n = lower.length;
    assert(n >= 3 && n <= rectangleMaxDimension && upper.length == n);
    assert(factor.length!0 == n && factor.length!1 == n);
    assert(isFinite(absoluteTolerance) && absoluteTolerance >= 0);
    assert(isFinite(relativeTolerance) && relativeTolerance >= 0);
    assert(maxEvaluations >= rectangleMinEvaluations);
    foreach (i; 0 .. n) assert(lower[i] < upper[i] && factor[i, i] > 0 && isFinite(factor[i, i]));

    auto rng = Mt19937_64(seed);
    Coordinates[shiftCount] shifts;
    foreach (ref shift; shifts)
        foreach (j; 0 .. n - 1)
        {
            shift[j] = (rng.front >> 11) * 0x1p-53;
            rng.popFront();
        }

    double[shiftCount] sums = 0;
    size_t previous = 0;
    size_t count = rectangleMinEvaluations / shiftCount;
    const limit = maxEvaluations / shiftCount;
    alias Status = RectangleIntegralStatus;
    while (true)
    {
        foreach (i; previous .. count)
        {
            Coordinates point;
            foreach (j; 0 .. n - 1) point[j] = radicalInverse(i + 1, bases[j]);
            foreach (s, ref shift; shifts)
            {
                const weight = rectangleWeight(lower, upper, factor, point, shift);
                if (!isFinite(weight))
                    return RectangleIntegralResult(Status.unresolved, i * shiftCount + s + 1);
                sums[s] += weight;
            }
        }
        double e1, e2;
        const p1 = groupEstimate(sums[0 .. groupSize], count, e1);
        const p2 = groupEstimate(sums[groupSize .. $], count, e2);
        const estimate = (p1 + p2) / 2;
        const error = max(e1, e2, abs(p1 - p2));
        if (estimate >= double.min_normal && error <= max(absoluteTolerance, relativeTolerance * estimate))
            return RectangleIntegralResult(Status.toleranceMet, count * shiftCount, estimate, error);
        if (count == limit)
        {
            if (estimate < double.min_normal)
                return RectangleIntegralResult(Status.unresolved, count * shiftCount, estimate);
            return RectangleIntegralResult(Status.budgetExhausted, count * shiftCount, estimate, error);
        }
        previous = count;
        count = min(count * 4, limit);
    }
}


version (mir_stat_test_rectangle_integral)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice : sliced;
    import mir.stat.covariance_views : covarianceView;
    import mir.stat.internal.normal_rectangle_order : orderNormalRectangle, RectangleOrderStatus;

    // At correlation 0.9999 the first two coordinates are overwhelmingly
    // inside [-8,8] conditional on the third being in [3.5,3.55].
    double[9] covariance = [1,.9999,.9999, .9999,1,.9999, .9999,.9999,1], storage;
    auto factor = storage[].sliced(3, 3);
    double[3] lower = [-8,-8,3.5], upper = [8,8,3.55], typical;
    size_t[3] order;
    assert(orderNormalRectangle(covarianceView(covariance[].sliced(3, 3)),
        lower[], upper[], order[], factor, typical[]) == RectangleOrderStatus.success);
    assert(order[0] == 2);
    const r = integrateNormalRectangle(lower[], upper[], factor, 5489, 0, 0.01, 16384);
    assert(r.status == RectangleIntegralStatus.toleranceMet);
    assert(r.evaluations == rectangleMinEvaluations);
    assert(abs(r.probability / 4.001350339989166e-5 - 1) < 1e-12);
    assert(r.estimatedError <= 0.01 * r.probability);
}

version (mir_stat_test_rectangle_integral)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice : sliced;
    import mir.stat.covariance_views : covarianceView;
    import mir.stat.internal.normal_rectangle_order : orderNormalRectangle, RectangleOrderStatus;

    double[9] cov = [1,.3,-.5, .3,1,.4, -.5,.4,1], storage;
    auto factor = storage[].sliced(3, 3);
    double[3] lower = -double.infinity, upper = 0, typical;
    size_t[3] order;
    assert(orderNormalRectangle(covarianceView(cov[].sliced(3, 3)),
        lower[], upper[], order[], factor, typical[]) == RectangleOrderStatus.success);
    auto run(ulong seed, size_t budget)
    {
        return integrateNormalRectangle(lower[], upper[], factor, seed, 0, 0, budget);
    }
    const r = run(187, 2063);
    assert(r.status == RectangleIntegralStatus.budgetExhausted && r.evaluations == 2048);
    // Exact trivariate orthant probability from the sum of correlation arcsines.
    assert(abs(r.probability - 0.14032747444711657) < 0.002 && r.estimatedError > 0);
    assert(run(187, 2048) == r);
    assert(run(188, 2048).probability != r.probability);
}

version (mir_stat_test_rectangle_integral)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice : sliced;

    double[9] identity = [1,0,0, 0,1,0, 0,0,1];
    auto factor = identity[].sliced(3, 3);
    double[3] lower = 24, upper = 25;
    auto run()
    {
        return integrateNormalRectangle(lower[], upper[], factor, 0, 1e-5, 0, 1024);
    }
    const underflow = run();
    assert(underflow.status == RectangleIntegralStatus.unresolved && underflow.evaluations == 1024);
    assert(underflow.probability == 0 && !isFinite(underflow.estimatedError));

    // CDF endpoint subtraction alone would erase this positive interval.
    lower = [-1e-18, -double.infinity, -double.infinity];
    upper = [2e-18, double.infinity, double.infinity];
    const narrow = run();
    assert(narrow.status == RectangleIntegralStatus.toleranceMet);
    assert(abs(narrow.probability / 1.196826841204298e-18 - 1) < 1e-14);
    auto x = sampleInterval(-1e-18, 2e-18, narrow.probability, 0.75);
    assert(abs(x - 1.25e-18) < 1e-32);

    // The two conditional endpoints coincide in floating point on the first sample.
    lower = [10.0, 0.0, -1.0];
    upper = [10.1, 1e-18, 1.0];
    identity[3] = 0.5;
    const lost = run();
    assert(lost.status == RectangleIntegralStatus.unresolved && lost.evaluations == 1);
    assert(!isFinite(lost.probability) && !isFinite(lost.estimatedError));
}

version (mir_stat_test_rectangle_integral)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice : sliced;
    import mir.stat.covariance_views : covarianceView;
    import mir.stat.internal.normal_rectangle_order : orderNormalRectangle, RectangleOrderStatus;

    // Markov chain with alternating signs. Independent one-dimensional
    // transition quadratures give the reference at both 64 and 100 nodes.
    double[7] correlation = [.81, -.72, .62, -.86, .77, .69, -.74];
    double[64] raw, work;
    double[8] lower, upper, typical;
    size_t[8] order;
    auto covariance = raw[].sliced(8, 8);
    auto factor = work[].sliced(8, 8);
    foreach (i; 0 .. 8)
    {
        lower[i] = -1.15 + 0.07 * i;
        upper[i] = 0.9 + 0.035 * i;
        covariance[i, i] = 1;
        foreach (j; i + 1 .. 8) covariance[i, j] = covariance[i, j - 1] * correlation[j - 1];
    }
    assert(orderNormalRectangle(covarianceView(covariance), lower[], upper[], order[],
        factor, typical[]) == RectangleOrderStatus.success);
    const r = integrateNormalRectangle(lower[], upper[], factor, 89071, 0, 0.01, 16384);
    assert(r.status == RectangleIntegralStatus.toleranceMet && r.evaluations <= 16384);
    assert(abs(r.probability / 0.11786570377215408 - 1) < 0.01);
}
