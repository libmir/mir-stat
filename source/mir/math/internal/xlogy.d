/++
Internal logarithmic products for entropy and distribution calculations.

A zero multiplier contributes positive zero without evaluating the logarithm.
This preserves endpoint terms such as $(D 0 * log(0)) in log probabilities.
Callers remain responsible for validating their distribution parameters;
these helpers do not use a zero multiplier to establish that an input is valid.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.math.internal.xlogy;

import mir.internal.utility: isFloatingPoint;

/++
Evaluates $(D x * log(x)) for nonnegative inputs.

Both signs of zero return positive zero. Positive infinity returns infinity.
Negative inputs and NaN fail the assertion when assertions are enabled.

Returns:
    $(D x * log(x)), with the limiting value zero at $(D x == 0).
+/
package(mir)
@safe pure nothrow @nogc
T xlog(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: log;

    assert(x >= 0, "xlog: x must be greater than or equal to zero");
    return x ? x * log(x) : 0;
}

/++
Evaluates $(D x * log(y)), treating a zero multiplier as a zero contribution.

Both signs of zero for $(D x) return positive zero for every $(D y), including
NaN and values outside the logarithm's domain. Otherwise the ordinary logarithm
and multiplication rules apply: for example, a positive finite multiplier with
$(D y == 0) returns negative infinity, and an infinite multiplier with
$(D y == 1) returns NaN. The multiplier may be negative.
+/
package(mir)
@safe pure nothrow @nogc
T xlogy(T)(const T x, const T y)
    if (isFloatingPoint!T)
{
    import mir.math.common: log;

    if (x == 0)
        return 0;
    else
        return x * log(y);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(xlogy(0.0, 2) == 0);
    assert(xlogy(0.5, 4).approxEqual(0.5 * log(4.0)));
}

/++
Evaluates $(D x * log(1 + y)), retaining small increments through $(D log1p).

Both signs of zero for $(D x) return positive zero for every $(D y), including
NaN and values below -1. Otherwise the ordinary $(D log1p) and multiplication
rules apply: for example, a positive finite multiplier with $(D y == -1)
returns negative infinity, and an infinite multiplier with $(D y == 0)
returns NaN. The multiplier may be negative.
+/
package(mir)
@safe pure nothrow @nogc
T xlog1py(T)(const T x, const T y)
    if (isFloatingPoint!T)
{
    import mir.math.internal.log1p: log1p;

    if (x == 0)
        return 0;
    else
        return x * log1p(y);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(xlog1py(0.0, 2) == 0);
    assert(xlog1py(0.5, 4).approxEqual(0.5 * log(1 + 4.0)));
}

// Zero contributions suppress logarithm evaluation, including invalid arguments.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: signbit;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (zero; [T(0), -T(0)])
        {
            const self = xlog(zero);
            assert(self == 0);
            assert(!signbit(self));
            foreach (y; [T(0), T(-1), T(-2), T(1), T.infinity, -T.infinity, T.nan])
            {
                const ordinary = xlogy(zero, y);
                const increment = xlog1py(zero, y);
                assert(ordinary == 0);
                assert(increment == 0);
                assert(!signbit(ordinary));
                assert(!signbit(increment));
            }
        }
    }}
}

// Nonzero multipliers retain logarithm-domain and infinity arithmetic.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T twiceLn2 = T(1.386294361119890618834464242916353136L);
        assert(xlog(T(1)) == 0);
        assert(xlog(T.infinity) == T.infinity);
        assert(approxEqual(xlog(T(2)), twiceLn2, 4 * T.epsilon, T(0)));
        assert(approxEqual(xlogy(T(-2), T(2)), -twiceLn2, 4 * T.epsilon, T(0)));
        assert(approxEqual(xlog1py(T(-2), T(1)), -twiceLn2, 4 * T.epsilon, T(0)));
        assert(xlogy(T(2), T(0)) == -T.infinity);
        assert(xlogy(T(-2), T(0)) == T.infinity);
        assert(xlog1py(T(2), T(-1)) == -T.infinity);
        assert(xlog1py(T(-2), T(-1)) == T.infinity);
        assert(xlogy(T(2), T.infinity) == T.infinity);
        assert(xlog1py(T(2), T.infinity) == T.infinity);
        assert(isNaN(xlogy(T(1), T(-1))));
        assert(isNaN(xlog1py(T(1), T(-2))));
        assert(isNaN(xlogy(T(1), T.nan)));
        assert(isNaN(xlog1py(T(1), T.nan)));
        assert(isNaN(xlogy(T.nan, T(1))));
        assert(isNaN(xlog1py(T.nan, T(0))));
        assert(isNaN(xlogy(T.infinity, T(1))));
        assert(isNaN(xlog1py(T.infinity, T(0))));
    }}
}

// A large multiplier must not amplify an increment already lost by log1p.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {
        assert(approxEqual(xlog1py(1 / T.epsilon, T.epsilon / 16),
            T(1) / 16, 4 * T.epsilon, T(0)));
        assert(approxEqual(xlog1py(1 / T.epsilon, -T.epsilon / 16),
            -T(1) / 16, 4 * T.epsilon, T(0)));
    }
}
