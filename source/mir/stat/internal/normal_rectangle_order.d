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

package(mir.stat) enum rectangleOrderMaxDimension = 8;

package(mir.stat) enum RectangleOrderStatus
{
    success,
    invalidFactor,      /// A residual variance or factor entry failed its numerical check.
    unresolvedInterval, /// The conditional interval or its ordering score could not be resolved.
}

// A Schur pivot is the ratio of two successive principal minors. Evaluate
// an ambiguous minor exactly on an integer grid using fraction-free Bareiss
// elimination. Keep its binary exponent separate when converting to double.
private double principalMinor(M)(scope auto ref M covariance, scope const(size_t)[] indices,
    out int exponent)
{
    import mir.bignum.integer: BigInt;
    import std.math: frexp, ldexp;

    enum nmax = rectangleOrderMaxDimension;
    struct Entry { long coefficient; int power; }
    Entry[nmax][nmax] entries = void;
    int grid = int.max, highest = int.min;
    foreach (r, i; indices)
        foreach (c; r .. indices.length)
        {
            const j = indices[c];
            static if (M.triangle == TrianglePolicy.upper)
                const x = i <= j ? covariance.storage[i, j] : covariance.storage[j, i];
            else
                const x = i >= j ? covariance.storage[i, j] : covariance.storage[j, i];
            int e;
            const coefficient = cast(long)ldexp(frexp(x, e), double.mant_dig);
            const power = e - double.mant_dig;
            entries[r][c] = Entry(coefficient, power);
            if (coefficient && power < grid) grid = power;
            if (coefficient && e > highest) highest = e;
        }

    double evaluate(uint words)()
    {
        alias Integer = BigInt!words;
        alias Product = BigInt!(2 * words + 1);
        Integer[nmax][nmax] a;
        const n = indices.length;
        foreach (r; 0 .. n)
            foreach (c; r .. n)
            {
                const entry = entries[r][c];
                a[r][c] = entry.coefficient;
                if (entry.coefficient) a[r][c] <<= entry.power - grid;
            }
        Integer previous = 1;
        foreach (k; 0 .. n - 1)
        {
            const ref pivot = a[k][k];
            // Every leading principal minor must be positive.
            if (!pivot.length || pivot.sign) return 0;
            foreach (i; k + 1 .. n)
                foreach (j; i .. n)
                {
                    Product value, cross;
                    if (a[i][j].length) value.multiply(pivot, a[i][j]);
                    if (a[k][i].length && a[k][j].length)
                    {
                        cross.multiply(a[k][i], a[k][j]);
                        value -= cross;
                    }
                    if (k && value.length) value /= previous;
                    a[i][j] = value;
                }
            previous = pivot;
        }
        auto det = a[n - 1][n - 1];
        if (!det.length || det.sign) return 0;
        const shift = det.bitLength > double.mant_dig ? det.bitLength - double.mant_dig : 0;
        det >>= shift;
        exponent = cast(int)n * grid + cast(int)shift;
        return cast(double)cast(long)det;
    }
    // Each minor sums at most 8! products (16 extra bits). For double
    // inputs the full exponent grid runs from -1126 through 1024.
    if (indices.length * (highest - grid) + 16 <= 640) return evaluate!10();
    return evaluate!((nmax * 2150 + 16 + 63) / 64)();
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
    covariance = Dense positive-definite double covariance view of dimension at most eight.
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
    import std.math: frexp, ldexp, fma;

    alias Status = RectangleOrderStatus;
    const n = order.length;
    assert(lower.length == n && upper.length == n && typical.length == n);
    assert(covariance.storage.length!0 == n && covariance.storage.length!1 == n);
    assert(factor.length!0 == n && factor.length!1 == n);
    assert(n <= rectangleOrderMaxDimension);
    // Scale each coordinate by a power of two. Products in the Schur
    // complement then stay in range even for widely different variances.
    double[rectangleOrderMaxDimension][rectangleOrderMaxDimension] residual = void;
    int[rectangleOrderMaxDimension] exponent;
    foreach (i; 0 .. n)
    {
        const variance = covariance.storage[i, i];
        if (!(variance > 0) || !isFinite(variance)) return Status.invalidFactor;
        frexp(variance, exponent[i]);
        exponent[i] /= 2;
        foreach (j; 0 .. i + 1)
        {
            static if (M.triangle == TrianglePolicy.upper) const entry = covariance.storage[j, i];
            else const entry = covariance.storage[i, j];
            const value = ldexp(entry, -exponent[i] - exponent[j]);
            if (!isFinite(value)) return Status.invalidFactor;
            residual[i][j] = residual[j][i] = value;
        }
        assert(lower[i] < upper[i]);
        order[i] = i;
        foreach (j; i + 1 .. n) factor[i, j] = 0;
    }

    bool smallPivotUsed;
    const pivotThreshold = 64 * n * double.epsilon;
    foreach (k; 0 .. n)
    {
        size_t best = k;
        double bestMass = 0, diagonal = 0;
        double previousMinor = 0;
        int previousExponent;
        foreach (j; k .. n)
        {
            const index = order[j];
            auto variance = residual[index][index];
            // A second Schur update can amplify rounding from the first.
            // Check small pivots against the original covariance entries.
            if (k >= 2 && (smallPivotUsed || variance <= pivotThreshold))
            {
                if (!previousMinor)
                    previousMinor = principalMinor(covariance, order[0 .. k], previousExponent);
                size_t[rectangleOrderMaxDimension] minorIndices;
                minorIndices[0 .. k] = order[0 .. k];
                minorIndices[k] = index;
                int e;
                const minor = principalMinor(covariance, minorIndices[0 .. k + 1], e);
                if (!(minor > 0 && previousMinor > 0)) return Status.invalidFactor;
                variance = ldexp(minor / previousMinor, e - previousExponent - 2 * exponent[index]);
                residual[index][index] = variance;
            }
            double center = 0;
            foreach (h; 0 .. k)
                center += factor[j, h] * typical[h];
            if (!(variance > 0) || !isFinite(variance)) return Status.invalidFactor;
            if (!isFinite(center)) return Status.unresolvedInterval;
            const sd = ldexp(sqrt(variance), exponent[index]);
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
        const pivot = order[k];
        const p = residual[pivot][pivot];
        smallPivotUsed |= p <= pivotThreshold;
        const root = sqrt(p);
        int shift;
        if (p < 0x1p-500)
        {
            frexp(p, shift);
            shift /= 2;
        }
        const scaledPivot = shift ? ldexp(p, -2 * shift) : p;
        foreach (j; k + 1 .. n)
        {
            const i = order[j];
            const x = residual[i][pivot];
            const scaledX = shift ? ldexp(x, -shift) : x;
            factor[j, k] = ldexp(x / root, exponent[i]);
            if (!isFinite(factor[j, k])) return Status.invalidFactor;
            foreach (h; k + 1 .. j + 1)
            {
                const t = order[h];
                const y = residual[t][pivot];
                // Compute (p*a - x*y)/p before rounding either Cholesky
                // entry. Identical rows have an exactly zero residual.
                const scaledY = shift ? ldexp(y, -shift) : y;
                const xy = scaledX * scaledY;
                const value = (fma(scaledPivot, residual[i][t], -xy) +
                    fma(-scaledX, scaledY, xy)) / scaledPivot;
                residual[i][t] = residual[t][i] = value;
            }
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

version (mir_stat_test_rectangle_order)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.covariance_views: covarianceView;

    // The first two axes have very different scales and perfect anticorrelation.
    static foreach (triangle; [TrianglePolicy.upper, TrianglePolicy.lower])
    {{
        double[9] cov = [0x1p901, -2, 0, -2, 0x1p-899, 0, 0, 0, 1], work;
        double[3] lower = -double.infinity, upper = double.infinity, typical;
        size_t[3] order;
        auto metric = covarianceView!triangle(cov[].sliced(3, 3));
        auto factor = work[].sliced(3, 3);
        auto run()
        {
            return orderNormalRectangle(metric, lower[], upper[], order[], factor, typical[]);
        }
        assert(run() == RectangleOrderStatus.invalidFactor);
        cov[4] += 0x1p-951;
        assert(run() == RectangleOrderStatus.success);
        // The exact residual variance is the one-ulp perturbation.
        import std.math: abs;
        assert(factor[1, 1] > 0 && abs(factor[1, 1] * factor[1, 1] / 0x1p-951 - 1) <= double.epsilon);
    }}
}
