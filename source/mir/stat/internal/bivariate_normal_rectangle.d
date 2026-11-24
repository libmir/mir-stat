/++
Numerical integration for standard bivariate normal rectangle probabilities.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.bivariate_normal_rectangle;

private import mir.math.common: exp, fabs, sqrt;
private import mir.stat.distribution.normal: normalCDF;
private import std.algorithm: min, max, sort;
private import std.math: fma;

@safe pure nothrow @nogc:

// Keep the budget status available while evaluating a public reporting contract.
package(mir.stat) struct RectangleEstimate(T)
{
    T probability;
    bool budgetExhausted;
}

// Scale each factor by 2^256 so even the smallest representable joint mass
// has room for panel subdivision and summation before rounding.
private enum factorScale = 0x1p256;
private enum tailThreshold(T) = T.max_exp <= 1024 ? 26 : 104;

private T scaledGaussian(T)(const T x)
{
    const halfExp = exp((-T(0.25) * x) * x);
    return (halfExp * T(factorScale)) * halfExp;
}

private T density(bool scaled, T)(const T x)
{
    enum invSqrt2Pi = T(0.398942280401432677939946059934381868L);
    static if (scaled)
        return scaledGaussian(x) * invSqrt2Pi;
    else
        return exp((-T(0.5) * x) * x) * invSqrt2Pi;
}

private T marginalCDF(bool scaled, T)(const T x)
{
    static if (scaled)
    {
        import mir.math.constant: SQRT1_2;
        import mir.stat.distribution.normal: erfce;
        if (x <= -tailThreshold!T)
            return scaledGaussian(x) * (T(0.5) * erfce(-x * T(SQRT1_2)));
        return normalCDF(x) * T(factorScale);
    }
    else
        return normalCDF(x);
}

// Eight-point Gauss-Legendre quadrature on [0, 1].
private T average(alias f, T)()
{
    static immutable T[4] nodes = [T(0.183434642495649804939476142360184L),
        T(0.525532409916328985817739049189246L), T(0.796666477413626739591553936475831L),
        T(0.960289856497536231683560868569473L)];
    static immutable T[4] weights = [T(0.362683783378361982965150449277195L),
        T(0.313706645877887287337962201986601L), T(0.222381034453374470544355994426240L),
        T(0.101228536290376259152531354309962L)];
    T sum = 0;
    foreach (i, node; nodes)
        sum += weights[i] * (f((1 - node) / 2) + f((1 + node) / 2));
    return sum / 2;
}

// Reflection and axis swaps must preserve the independently computed width.
private struct Interval(T)
{
    T lower, upper, width;

    Interval reflected() const @safe pure nothrow @nogc
    {
        return Interval(-upper, -lower, width);
    }

    T mass(bool scaled = false)() const @safe pure nothrow @nogc
    {
        return interval!scaled(lower, upper, width);
    }
}

// The width may be known more accurately than the difference of the transformed endpoints.
private T interval(bool scaled = false, T)(const T lo, const T hi, const T width)
{
    if (width <= 0) return 0;
    if (width * (1 + fabs(lo)) < T(0.125))
    {
        T integrand(T u) { return density!scaled(fma(width, u, lo)); }
        return width * average!(integrand, T)();
    }
    return lo >= 0 ? marginalCDF!scaled(-lo) - marginalCDF!scaled(-hi) :
        marginalCDF!scaled(hi) - marginalCDF!scaled(lo);
}

private RectangleEstimate!T integrate(alias f, T)(scope const(T)[] cuts)
{
    struct Panel { T lo, hi, mass, error; }
    T rule(T lo, T hi)
    {
        T mapped(T u) { return f(fma(hi - lo, u, lo)); }
        return (hi - lo) * average!(mapped, T)();
    }
    Panel panel(T lo, T hi)
    {
        const mid = lo + (hi - lo) / 2;
        const fine = rule(lo, mid) + rule(mid, hi);
        return Panel(lo, hi, fine, fabs(fine - rule(lo, hi)));
    }

    Panel[128] panels = void;
    size_t count;
    foreach (i; 1 .. cuts.length)
        if (cuts[i - 1] < cuts[i])
            panels[count++] = panel(cuts[i - 1], cuts[i]);

    for (;;)
    {
        T sum = 0, error = 0;
        size_t worst;
        foreach (i, ref p; panels[0 .. count])
        {
            sum += p.mass;
            error += p.error;
            if (p.error > panels[worst].error) worst = i;
        }
        // Compare the sum of the panel errors with the mass of the whole
        // rectangle, so negligible tails cannot dominate the refinement.
        const done = error <= 224 * T.epsilon * sum;
        if (done || count == panels.length)
            return RectangleEstimate!T(sum, !done);
        const p = panels[worst];
        const mid = p.lo + (p.hi - p.lo) / 2;
        panels[worst] = panel(p.lo, mid);
        panels[count++] = panel(mid, p.hi);
    }
}

package(mir.stat) T originalRectangleProbability(T, L, U, M, S)(scope auto ref L lower,
    scope auto ref U upper, scope auto ref M mean, scope auto ref S stdDev, T r)
{
    import mir.stat.internal.standardize: standardize;
    Interval!T[2] axes;
    foreach (i; 0 .. 2)
    {
        const T a = lower[i], b = upper[i], m = mean[i], s = stdDev[i];
        assert(fabs(m) <= T.max, "means must be finite");
        assert(s > 0 && s <= T.max, "standard deviations must be positive and finite");
        assert(!(b < a), "lower bounds must be at or below the corresponding upper bounds");
        axes[i] = Interval!T(standardize(a, m, s), standardize(b, m, s),
            a == b ? 0 : standardize(b, a, s));
    }

    return rectangleEstimate(axes[0], axes[1], r).probability;
}

package(mir.stat) RectangleEstimate!T rectangleEstimate(T)(T a, T b, T c, T d, T r)
{
    return rectangleEstimate(Interval!T(a, b, b - a), Interval!T(c, d, d - c), r);
}

private RectangleEstimate!T rectangleEstimate(T)(Interval!T x, Interval!T y, T r)
{
    const a = x.lower, b = x.upper, c = y.lower, d = y.upper;
    import mir.stat.distribution.bivariate_normal: bivariateStandardNormalCDF;
    alias Result = RectangleEstimate!T;
    enum inf = T.infinity;
    if (a != a || b != b || c != c || d != d) return Result(T.nan);
    assert(a <= b && c <= d, "Lower bounds must be at or below the corresponding upper bounds");
    if (x.width == 0 || y.width == 0 || a == inf || b == -inf || c == inf || d == -inf) return Result(0);
    if (a == -inf && b == inf) return Result(y.mass);
    if (c == -inf && d == inf) return Result(x.mass);
    if (r == 0)
    {
        const product = x.mass * y.mass;
        if (product >= T.min_normal) return Result(product);
        // Keep the marginals above the subnormal range until the final rounding.
        return Result((x.mass!true * y.mass!true) * T(1 / (factorScale * factorScale)));
    }
    if (r < 0) return rectangleEstimate(x, y.reflected, -r);
    if (r == 1)
    {
        const lo = max(a, c), hi = min(b, d);
        return Result(interval(lo, hi, hi - lo));
    }
    if ((a == -inf || b == inf) && (c == -inf || d == inf))
        return Result(bivariateStandardNormalCDF(a == -inf ? b : -a, c == -inf ? d : -c,
            (a == -inf) == (c == -inf) ? r : -r));

    if (x.width > y.width) return rectangleEstimate(y, x, r);
    if (max(a, -b, c, -d) <= tailThreshold!T)
    {
        auto result = quadrature!false(x, y, r);
        if (result.probability >= T.min_normal)
        {
            result.probability = min(T(1), result.probability);
            return result;
        }
    }
    // A tail interval can select scaling immediately. Near-singular
    // correlations can also produce tiny masses with ordinary bounds.
    auto result = quadrature!true(x, y, r);
    result.probability = min(T(1), result.probability * T(1 / (factorScale * factorScale)));
    return result;
}

private RectangleEstimate!T quadrature(bool scaled, T)(Interval!T x, Interval!T y, T r)
{
    const a = x.lower, b = x.upper, c = y.lower, d = y.upper;
    // Integrate across the shorter side in original coordinates when rotating
    // could erase the width. Use [0, 1] even if the endpoints are adjacent.
    const sigma = sqrt((1 - r) * (1 + r));
    if (x.width <= sigma / 8)
    {
        T conditional(T u)
        {
            const delta = x.width * u;
            const lo = fma(-r, delta, fma(-r, a, c)) / sigma;
            const hi = fma(-r, delta, fma(-r, a, d)) / sigma;
            return density!scaled(a + delta) * interval!scaled(lo, hi, y.width / sigma);
        }
        static immutable T[2] unit = [0, 1];
        auto result = integrate!conditional(unit[]);
        result.probability *= x.width;
        return result;
    }

    // X = A*U + B*V, Y = A*U - B*V with independent standard U and V.
    // Each rectangle corner can change which bound limits U.
    const A = sqrt((1 + r) / 2), B = sqrt((1 - r) / 2);
    T rotated(T v)
    {
        const lo = max(a - B * v, c + B * v) / A;
        const hi = min(b - B * v, d + B * v) / A;
        const width = min(x.width, y.width, fma(-2 * B, v, b - c), fma(2 * B, v, d - a)) / A;
        return density!scaled(v) * interval!scaled(lo, hi, width);
    }
    // Normal mass beyond these endpoints is below the smallest subnormal.
    enum limit = T.max_exp <= 1024 ? 40 : 160;
    const left = max(T(-limit), (a - d) / (2 * B));
    const right = min(T(limit), (b - c) / (2 * B));
    if (left >= right) return RectangleEstimate!T(0);

    T[24] cuts = void;
    size_t n;
    void split(T cut) { if (left < cut && cut < right) cuts[n++] = cut; }
    cuts[n++] = left;
    cuts[n++] = right;
    split(0);
    for (int k = 2; k < limit; k *= 2)
    {
        split(-k);
        split(k);
    }
    foreach (xBound; [a, b])
        foreach (yBound; [c, d])
            split((xBound - yBound) / (2 * B));
    sort(cuts[0 .. n]);
    return integrate!rotated(cuts[0 .. n]);
}
