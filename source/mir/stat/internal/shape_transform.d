/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.shape_transform;

import mir.internal.utility: isFloatingPoint;

/++
Evaluates $(D log(1 + xi * s) / xi), with limit $(D s) at zero shape.
The caller handles support endpoints before calling this function.

Params:
    s = standardized observation
    xi = shape parameter

Implementation_Notes:
For float and double precision, $(D log1p) is used when
$(D abs(xi * s) < 1.0 / 8). Extended precision uses it throughout.
Below epsilon the first-order limit avoids product underflow; its relative
truncation error is approximately $(D abs(xi * s) / 2).
An overflowing positive product is evaluated by adding logarithms.
DMD 2.102 uses a compensated logarithm to avoid its broken $(D log1p).
+/
package(mir.stat)
@safe pure nothrow @nogc
T log1pScaled(T)(const T s, const T xi)
    if (isFloatingPoint!T)
{
    import mir.math.common: fabs, log;
    import std.math: log1p;

    if (xi == 0)
        return s;
    const T u = xi * s;
    if (fabs(u) < T.epsilon)
        return s;
    if (u == T.infinity)
        return (log(fabs(xi)) + log(fabs(s))) / xi;
    const T v = 1 + u;
    static if (__VERSION__ == 2102)
        // The tiny-product branch ensures v - 1 is nonzero.
        return s * (log(v) / (v - 1));
    else static if (T.mant_dig > 53)
        return log1p(u) / xi;
    else
        return fabs(u) < T(0.125) ? log1p(u) / xi : log(v) / xi;
}

/++
Evaluates $(D (exp(xi * t) - 1) / xi), with limit $(D t) at zero shape.

Params:
    t = transformed probability before applying the shape
    xi = shape parameter

Implementation_Notes:
Using $(D expm1) retains small nonzero shapes. Below epsilon the first-order
limit also preserves products which underflow. If the exponential overflows
before division by a large shape, the quotient is evaluated in log space.
+/
package(mir.stat)
@safe pure nothrow @nogc
T expm1Scaled(T)(const T t, const T xi)
    if (isFloatingPoint!T)
{
    import mir.math.common: fabs, exp, log;
    import std.math: expm1;

    if (xi == 0)
        return t;
    const T u = xi * t;
    if (fabs(u) < T.epsilon)
        return t;
    const T e = expm1(u);
    if (e == T.infinity)
        return (xi > 0 ? T(1) : T(-1)) * exp(u - log(fabs(xi)));
    return e / xi;
}

// Limits, independent values, and intermediate overflow in the scaled transforms.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, log;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T ln2 = T(0.69314718055994530941723212145817656808L);
        assert(log1pScaled(T.infinity, T(0)) == T.infinity);
        assert(expm1Scaled(T.infinity, T(0)) == T.infinity);
        assert(expm1Scaled(-T.infinity, T(1)) == -1);
        assert(expm1Scaled(T.infinity, T(-1)) == 1);
        assert(log1pScaled(T.min_normal, T.min_normal) == T.min_normal);
        assert(expm1Scaled(T.min_normal, T.min_normal) == T.min_normal);
        assert(approxEqual(log1pScaled(T(1), T(1)), ln2, 8 * T.epsilon, T(0)));
        assert(approxEqual(log1pScaled(T(1), T(-.5)), 2 * ln2, 8 * T.epsilon, T(0)));
        assert(approxEqual(expm1Scaled(T(1), T(1)),
            T(1.71828182845904523536028747135266249776L), 8 * T.epsilon, T(0)));
        assert(approxEqual(expm1Scaled(T(1), T(-1)),
            T(0.63212055882855767840447622983853913255L), 8 * T.epsilon, T(0)));

        // exp(xi*t) overflows, but dividing by xi gives approximately two.
        const T logMax = log(T.max);
        const T t = (logMax + ln2) / T.max;
        assert(approxEqual(expm1Scaled(t, T.max), T(2),
            8 * T.epsilon * logMax, T(0)));
    }}
}
