/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.neg_log1m;

import mir.internal.utility: isFloatingPoint;

/++
Evaluates $(D -log(1 - p)) without losing tiny probabilities to subtraction rounding.

Params:
    p = probability in [0, 1], validated by the calling distribution function

Returns:
    The negative logarithm of the complementary probability; zero for
    $(D p == 0) and positive infinity for $(D p == 1).

Implementation_Notes:
    For float and double precision, uses $(D log1p) below $(D T.epsilon)
    and ordinary $(D log) with a rounding correction above that cutoff.
    This favors performance in the tested LDC builds. Extended precision
    uses $(D log1p) directly. Precision is determined by $(D T.mant_dig),
    since $(D real) varies between targets. DMD 2.102 uses the compatibility
    path described below instead of its problematic $(D log1p).
+/
package(mir.stat)
@safe pure nothrow @nogc
T negLog1m(T)(const T p)
    if (isFloatingPoint!T)
{
    import mir.math.common: log;

    static if (__VERSION__ == 2102)
    {
        // mir.math.internal.log1p deliberately avoids this frontend's broken
        // log1p implementation. Its log(1 + x) fallback also loses tiny p.
        // Here -log(1 - p) = p + O(p*p), so returning p below epsilon has
        // relative truncation error below approximately epsilon / 2.
        if (p < T.epsilon)
            return p;
    }
    else
    {
        import mir.math.internal.log1p: log1p;

        static if (T.mant_dig > 53)
            return -log1p(-p);
        else if (p < T.epsilon)
            return -log1p(-p);
    }

    static if (__VERSION__ == 2102 || T.mant_dig <= 53)
    {
        const T u = 1 - p;
        // Subtraction is exact for p in [0.5, 1]. This also handles p == 1.
        if (p >= T(0.5))
            return -log(u);

        // Rounding u can make 1 - u differ from the original p. Since
        // -log(1 - p) is approximately proportional to p for small p, this
        // ratio compensates for the rounding. It is a numerical correction,
        // not an exact identity. The cutoff above ensures 1 - u is nonzero.
        return -log(u) * (p / (1 - u));
    }
}

// Tiny probabilities, known values, and the two calculation boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, nextUp;
    import mir.math.common: approxEqual;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(negLog1m(T(0)) == 0);
        assert(negLog1m(T(1)) == T.infinity);
        const T smallest = nextUp(T(0));
        assert(negLog1m(smallest) == smallest);

        foreach (p; [T.min_normal, T.epsilon / 16, nextDown(T.epsilon),
                     T.epsilon, nextUp(T.epsilon)])
        {
            const T expected = p + p * p / 2;
            assert(approxEqual(negLog1m(p) / expected, T(1), 4 * T.epsilon, T(0)));
        }

        const T[2][3] cases = [
            [T(0.125), T(0.13353139262452262314634362093134997459L)],
            [T(0.5), T(0.69314718055994530941723212145817656808L)],
            [T(0.875), T(2.07944154167983592825169636437452970423L)]];
        foreach (entry; cases)
            assert(approxEqual(negLog1m(entry[0]) / entry[1], T(1),
                4 * T.epsilon, T(0)));

        foreach (boundary; [T.epsilon, T(0.5)])
        {
            const T low = negLog1m(nextDown(boundary));
            const T middle = negLog1m(boundary);
            const T high = negLog1m(nextUp(boundary));
            assert(low <= middle && middle <= high);
            assert(approxEqual(low / middle, T(1), 4 * T.epsilon, T(0)));
            assert(approxEqual(high / middle, T(1), 4 * T.epsilon, T(0)));
        }

        // 1 - nextDown(1) is exactly epsilon / 2 in binary arithmetic.
        const T expectedTail = T(T.mant_dig) * T(0.69314718055994530941723212145817656808L);
        assert(approxEqual(negLog1m(nextDown(T(1))) / expectedTail, T(1),
            4 * T.epsilon, T(0)));
    }}
}
