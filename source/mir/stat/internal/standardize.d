/++
Internal helpers for location-scale standardization.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.standardize;

// Scalar callers can inspect the actual difference. Lazy views instead use
// FloatingScale to prepare their normalization once for every observation.
// Divide first only when subtracting finite opposite-signed values overflows.
// In that case the divisions cannot cancel infinities of the same sign.
package(mir.stat)
@safe pure nothrow @nogc
T standardize(T)(const T x, const T location, const T scale)
{
    import mir.math.common: fabs;
    const T delta = x - location;
    if (fabs(delta) == T.infinity && fabs(x) <= T.max && fabs(location) <= T.max)
        return x / scale - location / scale;
    return delta / scale;
}

// Normalize once instead of branching on overflow at every element. The center
// cutoff lies just below half an ULP at F.max: smaller centers cannot make a
// finite observation overflow on subtraction under round-to-nearest arithmetic.
package(mir.stat) struct FloatingScale(F, R)
{
    F center;
    R divisor;
    // Both possible factors are exact in double. Using it for real avoids an
    // extended-precision field/load, without changing the arithmetic type.
    static if (is(F == real))
        double factor;
    else
        F factor;

    void prepare(F m, R d) @safe pure nothrow @nogc
    {
        import std.math: abs, isFinite;
        // Halving must not round a subnormal divisor or erase it. This also
        // permits a divisor <= 1 when R is wider than the subtraction type.
        factor = isFinite(m) && abs(m) > F.max * (F.epsilon / 4) &&
            d >= R.min_normal * 2 ? 0.5f : 1.0f;
        center = m * factor;
        divisor = d * factor;
    }

    R opCall(E)(E value) const @safe pure nothrow @nogc
    {
        return (cast(F) value * factor - center) / divisor;
    }
}

// Logarithm of the absolute distance between finite values, even when
// their subtraction overflows. Equal inputs return negative infinity.
package(mir.stat)
@safe pure nothrow @nogc
T logDistance(T)(const T x, const T mean)
{
    import mir.math.common: fabs, log;
    import mir.math.internal.log1p: log1p;
    const T distance = fabs(x - mean);
    if (distance <= T.max)
        return log(distance);
    const T a = fabs(x), b = fabs(mean);
    return a > b ? log(a) + log1p(b / a) : log(b) + log1p(a / b);
}
