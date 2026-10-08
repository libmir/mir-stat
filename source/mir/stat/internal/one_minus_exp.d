/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.one_minus_exp;

import mir.internal.utility: isFloatingPoint;

/++
Evaluates $(D 1 - exp(-t)) without losing tiny inputs to subtraction rounding.

Params:
    t = value whose negation is the exponent; nonnegative for a CDF

Returns:
    $(D 1 - exp(-t)), including zero for zero and one for positive infinity.

Implementation_Notes:
    Below $(D T.epsilon) in magnitude, returns $(D t) directly. The omitted
    quadratic term has relative magnitude approximately $(D abs(t) / 2).
    This also preserves subnormal inputs that some $(D expm1) implementations
    round to zero.

    For float and double precision, uses $(D expm1) below $(D 0.5) and
    ordinary $(D exp) above it, where subtraction no longer causes severe
    cancellation. Extended precision uses $(D expm1) outside the tiny-input
    region. These choices favor performance in the tested LDC builds.
    Precision follows $(D T.mant_dig), since $(D real) varies between targets.
+/
package(mir.stat)
@safe pure nothrow @nogc
T oneMinusExpNeg(T)(const T t)
    if (isFloatingPoint!T)
{
    import std.math: expm1, fabs;

    if (fabs(t) < T.epsilon)
        return t;

    static if (T.mant_dig > 53)
        return -expm1(-t);
    else
    {
        import mir.math.common: exp;
        return t < T(0.5) ? -expm1(-t) : 1 - exp(-t);
    }
}

// Independent reference values, tiny inputs, and calculation boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, nextUp, isNaN;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(oneMinusExpNeg(T(0)) == 0);
        assert(oneMinusExpNeg(T.infinity) == 1);
        assert(oneMinusExpNeg(-T.infinity) == -T.infinity);
        assert(isNaN(oneMinusExpNeg(T.nan)));

        foreach (t; [nextUp(T(0)), T.min_normal, T.epsilon / 16])
        {
            assert(oneMinusExpNeg(t) == t);
            assert(oneMinusExpNeg(-t) == -t);
        }

        const T[2][4] cases = [
            [T(0.5), T(0.39346934028736657639620046500881954656L)],
            [T(1), T(0.63212055882855767840447622983853913255L)],
            [T(2), T(0.86466471676338730810600050502751559659L)],
            [T(-1), T(-1.71828182845904523536028747135266249776L)]];
        foreach (entry; cases)
            assert(approxEqual(oneMinusExpNeg(entry[0]) / entry[1], T(1),
                8 * T.epsilon, T(0)));

        foreach (t; [nextDown(T.epsilon), T.epsilon, nextUp(T.epsilon)])
        {
            const T expected = t - t * t / 2;
            assert(approxEqual(oneMinusExpNeg(t) / expected, T(1),
                4 * T.epsilon, T(0)));
        }

        foreach (boundary; [T.epsilon, T(0.5)])
        {
            const T low = oneMinusExpNeg(nextDown(boundary));
            const T middle = oneMinusExpNeg(boundary);
            const T high = oneMinusExpNeg(nextUp(boundary));
            assert(low <= middle && middle <= high);
            assert(approxEqual(low / middle, T(1), 8 * T.epsilon, T(0)));
            assert(approxEqual(high / middle, T(1), 8 * T.epsilon, T(0)));
        }
    }}
}
