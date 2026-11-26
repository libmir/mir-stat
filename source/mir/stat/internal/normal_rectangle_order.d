/++
Conditional ordering of coordinates for multivariate normal rectangle integration.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.normal_rectangle_order;

private import mir.stat.covariance_views: CovarianceRepresentation, isCovarianceMetric, TrianglePolicy;
private import mir.stat.internal.linearAlgebra.types: Element, matrix;
private import mir.stat.internal.standardize: standardize;
private import std.math: isFinite;

@safe pure nothrow @nogc:

package(mir.stat) enum RectangleOrderStatus
{
    success,
    invalidFactor,      /// A residual variance or factor entry failed its numerical check.
    unresolvedInterval, /// The conditional interval or its ordering score could not be resolved.
}

// For a standard normal restricted to [a, b], E[Z] = (phi(a) - phi(b)) / mass.
// Factor out the larger density to avoid cancellation for nearby endpoints.
private double truncatedMean(double a, double b, double width, double mass)
{
    import mir.math.common: fabs;
    import mir.stat.distribution.normal: normalPDF;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    import std.algorithm: clamp;

    if (fabs(a) == fabs(b)) return 0;
    if (width * (1 + fabs(a)) < 0x1p-26) return a + width / 2;
    const difference = width * fabs(a / 2 + b / 2);
    const fraction = oneMinusExpNeg(difference);
    const numerator = fabs(a) < fabs(b) ? normalPDF(a) : -normalPDF(b);
    return clamp(numerator * fraction / mass, a, b);
}

/++
Choose the next Cholesky pivot by the smallest conditional normal interval
mass, evaluated at representative truncated means for preceding coordinates.
Scores within eight relative machine epsilons retain the first candidate in
the current order. Only the labeled covariance triangle is read.

Params:
    covariance = Dense positive-definite double covariance view.
    lower = Lower bounds relative to the mean, permuted in place.
    upper = Upper bounds relative to the mean, each strictly greater than its lower bound, permuted in place.
        Either endpoint may be infinite.
    order = Output original coordinate index at each permuted position.
    factor = Output lower Cholesky factor of the permuted covariance, with
        zeros above the diagonal.
    typical = Scratch vector for the representative standard normal coordinates.

All vectors have one entry per dimension. The factor has the same square shape
as the covariance. All writable buffers must be disjoint from each other and
from the covariance.

Returns:
    Success, a numerical factorization failure, or an unresolved conditional
    interval. A positive-width interval with an unrepresentable score cannot
    establish a zero rectangle probability. Use the outputs only on success.
+/
package(mir.stat) RectangleOrderStatus orderNormalRectangle(M, F)(scope auto ref M covariance,
    scope double[] lower, scope double[] upper, scope size_t[] order,
    scope auto ref F factor, scope double[] typical)
    if (isCovarianceMetric!M && M.form == CovarianceRepresentation.covariance &&
        is(Element!(typeof(M.init.storage)) == double) && matrix!F && is(Element!F == double))
{
    import mir.math.common: sqrt;
    import mir.stat.internal.bivariate_normal_rectangle: normalIntervalMass;
    import std.algorithm: swap;

    alias Status = RectangleOrderStatus;
    const n = order.length;
    assert(lower.length == n && upper.length == n && typical.length == n);
    assert(covariance.storage.length!0 == n && covariance.storage.length!1 == n);
    assert(factor.length!0 == n && factor.length!1 == n);
    foreach (i; 0 .. n)
    {
        assert(lower[i] < upper[i]);
        order[i] = i;
        foreach (j; i + 1 .. n) factor[i, j] = 0;
    }

    foreach (k; 0 .. n)
    {
        size_t best = k;
        double bestMass = 0, diagonal = 0;
        foreach (j; k .. n)
        {
            const index = order[j];
            double variance = covariance.storage[index, index];
            double center = 0;
            foreach (h; 0 .. k)
            {
                variance -= factor[j, h] * factor[j, h];
                center += factor[j, h] * typical[h];
            }
            if (!(variance > 0) || !isFinite(variance)) return Status.invalidFactor;
            if (!isFinite(center)) return Status.unresolvedInterval;
            const sd = sqrt(variance);
            const a = standardize(lower[j], center, sd);
            const b = standardize(upper[j], center, sd);
            const width = standardize(upper[j], lower[j], sd);
            if (!(a < b)) return Status.unresolvedInterval;
            const mass = normalIntervalMass(a, b, width);
            if (!(mass >= double.min_normal && mass <= 1)) return Status.unresolvedInterval;
            if (j == k || bestMass - mass > 8 * double.epsilon * bestMass)
            {
                best = j;
                bestMass = mass;
                diagonal = sd;
                typical[k] = truncatedMean(a, b, width, mass);
            }
        }
        if (best != k)
        {
            swap(order[k], order[best]);
            swap(lower[k], lower[best]);
            swap(upper[k], upper[best]);
            foreach (h; 0 .. k) swap(factor[k, h], factor[best, h]);
        }
        factor[k, k] = diagonal;
        foreach (j; k + 1 .. n)
        {
            size_t i = order[j], pivot = order[k];
            static if (M.triangle == TrianglePolicy.upper)
            {
                if (i > pivot) swap(i, pivot);
            }
            else if (i < pivot) swap(i, pivot);
            double value = covariance.storage[i, pivot];
            foreach (h; 0 .. k) value -= factor[j, h] * factor[k, h];
            value /= diagonal;
            if (!isFinite(value)) return Status.invalidFactor;
            factor[j, k] = value;
        }
    }
    return Status.success;
}


version (mir_stat_test) version = mir_stat_test_rectangle_order;
version (mir_stat_test_blas) version = mir_stat_test_rectangle_order;

version (mir_stat_test_rectangle_order)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.covariance_views: covarianceView;
    import std.math: abs;

    // Once coordinate 2 has been selected, its positive conditional mean
    // makes the negative interval on coordinate 1 rarer than coordinate 0.
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    {{
        double[9] raw = [1, 0.1, 0.1, 0.1, 1, 0.9, 0.1, 0.9, 1];
        double[9] work;
        double[3] lower = [-0.6, -2.0, 1.0], upper = [0.6, 0.0, 1.1], typical;
        const originalLower = lower, originalUpper = upper;
        auto input = raw[].sliced(3, 3);
        auto factor = work[].sliced(3, 3);
        foreach (i; 0 .. 3)
            foreach (j; 0 .. 3)
                if (triangle == TrianglePolicy.upper ? i > j : i < j) input[i, j] = double.nan;
        size_t[3] order;
        auto metric = covarianceView!triangle(input);
        assert(orderNormalRectangle(metric, lower[], upper[], order[], factor, typical[]) ==
            RectangleOrderStatus.success);
        assert(order == [2, 1, 0]);
        foreach (i; 0 .. 3)
        {
            assert(lower[i] == originalLower[order[i]] && upper[i] == originalUpper[order[i]]);
            foreach (j; 0 .. 3)
            {
                if (j > i) assert(factor[i, j] == 0);
                double sum = 0;
                foreach (h; 0 .. 3) sum += factor[i, h] * factor[j, h];
                auto expected = input[order[i], order[j]];
                if (expected != expected) expected = input[order[j], order[i]];
                assert(abs(sum - expected) < 1e-14);
            }
        }
    }}
}

version (mir_stat_test_rectangle_order)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.covariance_views: covarianceView;

    import mir.stat.internal.bivariate_normal_rectangle: normalIntervalMass;

    double[4] raw = [1, 0, 0, 1], work;
    double[2] lower = [0, 0], upper = [1, 1 - 6 * double.epsilon], typical;
    size_t[2] order;
    auto metric = covarianceView(raw[].sliced(2, 2));
    auto factor = work[].sliced(2, 2);
    auto run()
    {
        return orderNormalRectangle(metric, lower[], upper[], order[], factor, typical[]);
    }
    // The second interval is slightly smaller; its score falls within the tie tolerance.
    assert(normalIntervalMass(0.0, upper[1], upper[1]) < normalIntervalMass(0.0, upper[0], upper[0]));
    assert(run() == RectangleOrderStatus.success && order == [0, 1]);
    upper[1] = 0.99;
    assert(run() == RectangleOrderStatus.success && order == [1, 0]);

    lower = [-double.infinity, -1e-18];
    upper = [double.infinity, 2e-18];
    assert(run() == RectangleOrderStatus.success && order == [1, 0]);
    assert(typical[0] > 0 && typical[0] < 2e-18 && typical[1] == 0);
    lower[0] = 40;
    upper[0] = 41;
    assert(run() == RectangleOrderStatus.unresolvedInterval);

    // The first pivot has a nonzero representative mean. Subtracting the
    // conditional center then rounds the second pair of bounds together.
    raw[1] = 0.5;
    lower = [10.0, 0.0];
    upper = [10.01, 1e-17];
    assert(run() == RectangleOrderStatus.unresolvedInterval);

    lower[] = -double.infinity;
    upper[] = double.infinity;
    raw[1] = 1;
    assert(run() == RectangleOrderStatus.invalidFactor);
    auto empty = raw[0 .. 0].sliced(0, 0);
    assert(orderNormalRectangle(covarianceView(empty), lower[0 .. 0], upper[0 .. 0],
        order[0 .. 0], empty, typical[0 .. 0]) == RectangleOrderStatus.success);
}
