/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.math.internal.log1p;


// There are issues with the v2.102.2 point release for `log1p` producing the
// wrong results. Keep a corrected-log fallback for 2.102 as a whole.
static if (__VERSION__ != 2102) {
    public import std.math.exponential: log1p;
} else {
    import std.math.exponential: log;
    package(mir)
    @safe pure nothrow @nogc
    float log1p(const float x) {
        return correctedLog1p!float(x);
    }

    package(mir)
    @safe pure nothrow @nogc
    double log1p(const double x) {
        return correctedLog1p!double(x);
    }

    package(mir)
    @safe pure nothrow @nogc
    real log1p(const real x) {
        return correctedLog1p!real(x);
    }

    private @safe pure nothrow @nogc
    T correctedLog1p(T)(const T x)
    {
        const T u = 1 + x;
        // When addition loses x entirely, log(1+x) rounds to x at T's
        // precision. Returning x also preserves the sign of zero.
        if (u == 1)
            return x;
        // Avoid infinity/infinity in the correction for positive infinity.
        if (u == T.infinity)
            return log(u);
        // u-1 is the increment retained by rounded addition. Rescale the
        // logarithm by the original increment to compensate for that error.
        return log(u) * (x / (u - 1));
    }
    
    version(mir_stat_test)
    @safe pure nothrow @nogc
    unittest
    {
        import std.math.exponential: log;
        import mir.test: shouldApprox;
        float x1 = -0.0125;
        double x2 = x1;
        real x3 = x1;
        x1.log1p.shouldApprox == -0.0125787822;
        x2.log1p.shouldApprox == -0.0125787822;
        x3.log1p.shouldApprox == -0.0125787822;
    }
}

// The compatibility path must retain small increments and logarithm endpoints.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN, nextUp, signbit;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (x; [T.epsilon / 16, T.min_normal, nextUp(T(0))])
        {
            assert(T(log1p(x)) == x);
            assert(T(log1p(-x)) == -x);
        }
        assert(log1p(T(0)) == 0);
        assert(!signbit(log1p(T(0))));
        assert(signbit(log1p(-T(0))));
        assert(log1p(T(-1)) == -T.infinity);
        assert(log1p(T.infinity) == T.infinity);
        assert(isNaN(log1p(T(-2))));
        assert(isNaN(log1p(T.nan)));
        assert(approxEqual(T(log1p(T(0.125))),
            T(0.1177830356563834545387941094705217051L), 4 * T.epsilon, T(0)));
        assert(approxEqual(T(log1p(T(-0.125))),
            T(-0.1335313926245226231463436209313499746L), 4 * T.epsilon, T(0)));
    }}
}
