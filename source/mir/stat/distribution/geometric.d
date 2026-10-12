/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.geometric;

import mir.internal.utility: isFloatingPoint;

// Tiny probabilities and counts whose successor is not an integer size_t.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        const T tiny = T.epsilon / 16;
        assert(geometricCDF(size_t(0), tiny) == tiny);
        assert(approxEqual(geometricCDF(size_t(1), tiny), 2 * tiny, 8 * T.epsilon, T(0)));
        assert(geometricCDF(size_t(0), nextUp(T(0))) == nextUp(T(0)));
        foreach (k; [size_t(0), size_t(64), size_t(65), size_t.max]) {
            assert(geometricPMF(k, T(0)) == 0);
            assert(geometricPMF(k, T(1)) == (k == 0 ? 1 : 0));
            assert(geometricCDF(k, T(0)) == 0);
            assert(geometricCDF(k, T(1)) == 1);
            assert(geometricCCDF(k, T(0)) == 1);
            assert(geometricCCDF(k, T(1)) == 0);
        }
        assert(geometricCDF(size_t.max, T(0.5)) == 1);
        assert(geometricCCDF(size_t.max, T(0.5)) == 0);
        foreach (x; [cast(T)size_t.max, T.infinity]) {
            assert(geometricCDF(x, T(0.5)) == 1);
            assert(geometricCCDF(x, T(0.5)) == 0);
            assert(geometricCDF(x, T(0)) == 0);
            assert(geometricCCDF(x, T(0)) == 1);
            assert(geometricCDF(x, T(1)) == 1);
            assert(geometricCCDF(x, T(1)) == 0);
        }
    }}
    static if (size_t.sizeof == 8) {
        // Independent 100-digit references for exact binary p and k.
        assert(approxEqual(geometricPMF(1UL << 60, 0x1p-60),
            3.190845514646648713140259429307883234771e-19, 8 * double.epsilon, 0.0));
        assert(approxEqual(geometricCCDF(size_t.max, 0x1p-64),
            0.3678794411714423215855522278535, 8 * double.epsilon, 0.0));
    }
}

/++
Computes the geometric probability mass function (PMF) for the number of
failures before the first success.

Params:
    k = value to evaluate PMF
    p = `true` probability

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricPMF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow, exp;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (k == 0)
        return p;
    if (p == 0 || p == 1)
        return 0;
    // Keep the short power calculation for ordinary counts. At large counts,
    // rounding 1-p before exponentiation can noticeably change the probability.
    if (k <= 64)
        return pow(1 - p, k) * p;
    return exp(-cast(T)k * negLog1m(p)) * p;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.geometricPMF(0.5).shouldApprox == 0.5;
    1.geometricPMF(0.5).shouldApprox == 0.25;
    2.geometricPMF(0.25).shouldApprox == 0.140625;
}

/++
Computes the geometric cumulative density function (CDF).

Params:
    k = value to evaluate CDF
    p = `true` probability

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricCDF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.internal.neg_log1m: negLog1m;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    if (k == 0 || p == 0 || p == 1)
        return p;
    // Convert before adding: k+1 need not fit in size_t.
    return oneMinusExpNeg((cast(T)k + 1) * negLog1m(p));
}

/// ditto
@safe pure @nogc nothrow
T geometricCDF(T)(const T x, const T p)
    if (isFloatingPoint!T)
    in (x >= -1, "x must be larger than or equal to -1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    if (x < 0)
        return 0;
    // Avoid converting infinity or a count outside size_t to an integer.
    if (x >= cast(T)size_t.max) {
        import mir.math.common: floor;
        import mir.stat.internal.neg_log1m: negLog1m;
        import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
        if (p == 0 || p == 1)
            return p;
        return oneMinusExpNeg((floor(x) + 1) * negLog1m(p));
    }
    return geometricCDF!T(cast(const size_t) x, p);  
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    geometricCDF(-1.0, 0.5).shouldApprox == 0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCDF(0.5).shouldApprox == 0.5;
    1.geometricCDF(0.5).shouldApprox == 0.75;
    2.geometricCDF(0.5).shouldApprox == 0.875;

    geometricCDF(-1.0, 0.25).shouldApprox == 0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCDF(0.25).shouldApprox == 0.25;
    1.geometricCDF(0.25).shouldApprox == 0.4375;
    2.geometricCDF(0.25).shouldApprox == 0.578125;
    2.5.geometricCDF(0.25).shouldApprox == 0.578125;
}

/++
Computes the geometric complementary cumulative density function (CCDF).

Params:
    k = value to evaluate CCDF
    p = `true` probability

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricCCDF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: exp;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (k == 0 || p == 0 || p == 1)
        return 1 - p;
    return exp(-(cast(T)k + 1) * negLog1m(p));
}

/// ditto
@safe pure @nogc nothrow
T geometricCCDF(T)(const T x, const T p)
    if (isFloatingPoint!T)
    in (x >= -1, "x must be larger than or equal to -1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    if (x < 0)
        return 1;
    if (x >= cast(T)size_t.max) {
        import mir.math.common: floor, exp;
        import mir.stat.internal.neg_log1m: negLog1m;
        if (p == 0 || p == 1)
            return 1 - p;
        return exp(-(floor(x) + 1) * negLog1m(p));
    }
    return geometricCCDF!T(cast(const size_t) x, p);  
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    geometricCCDF(-1.0, 0.5).shouldApprox == 1.0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCCDF(0.5).shouldApprox == 0.5;
    1.geometricCCDF(0.5).shouldApprox == 0.25;
    2.geometricCCDF(0.5).shouldApprox == 0.125;

    geometricCCDF(-1.0, 0.25).shouldApprox == 1.0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCCDF(0.25).shouldApprox == 0.75;
    1.geometricCCDF(0.25).shouldApprox == 0.5625;
    2.geometricCCDF(0.25).shouldApprox == 0.421875;
    2.5.geometricCCDF(0.25).shouldApprox == 0.421875;
}

/++
Computes the geometric inverse cumulative distribution function (InvCDF).

For interior probabilities, returns the smallest nonnegative count whose CDF
is at least `q`. Retains the boundary conventions `q = 0` returns -1 (except
at `p = 1`, where it returns zero), and `q = 1` returns infinity. For `p = 0`,
positive queries also return infinity. The floating return type cannot represent
every integer at large counts. Close comparisons use bounded exact binary
arithmetic where possible; larger cases retain floating-point accuracy.

Params:
    q = value to evaluate InvCDF
    p = `true` probability

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricInvCDF(T)(const T q, const T p)
    if (isFloatingPoint!T)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: ceil;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (q == 1) {
        return T.infinity;
    }
    if (p == 1) {
        return 0;
    }
    if (q == 0)
        return -1;
    if (p == 0)
        return T.infinity;
    const real lp = negLog1m(cast(real)p);
    const real lq = negLog1m(cast(real)q);
    real guess = ceil(lq / lp) - 1;
    if (guess < 0)
        guess = 0;
    // At counts beyond consecutive-integer precision, stepping by one may
    // leave the value unchanged. Return the representable approximation there.
    if (guess + 1 == guess || guess == real.infinity)
        return cast(T)guess;
    while (guess > 0 && geometricQuantileReached(guess - 1, p, q, lp, lq))
        --guess;
    while (!geometricQuantileReached(guess, p, q, lp, lq)) {
        const real next = guess + 1;
        if (next == guess)
            break;
        guess = next;
    }
    return cast(T)guess;
}

// Comparing logarithms avoids reconstructing a CDF close to one. Near a
// boundary, do not snap approximately equal values to an integer: adjacent
// queries can lie on opposite sides of the exact CDF step.
private @safe pure nothrow @nogc
bool geometricQuantileReached(T)(real k, T p, T q, real lp, real lq)
{
    import std.math: fabs, frexp, ldexp;
    import mir.bignum.integer: BigInt;
    const real value = (k + 1) * lp;
    if (k < 8192 && fabs(value - lq) <= 32 * real.epsilon * (value + lq)) {
        static if (T.mant_dig <= 64) {
            // For p=a/2^b, compare (2^b-a)^(k+1)/2^(b*(k+1))
            // directly with 1-q. Fixed storage keeps the fallback @nogc.
            alias Integer = BigInt!256;
            static void split(T x, out ulong a, out int b) {
                int e;
                a = cast(ulong)ldexp(frexp(x, e), T.mant_dig);
                b = T.mant_dig - e;
                while (a && !(a & 1)) { a >>= 1; --b; }
            }
            static void shift(ref Integer x, int bits) {
                while (bits >= 32) { x *= 1UL << 32; bits -= 32; }
                if (bits) x *= 1UL << bits;
            }
            ulong a, c;
            int b, d;
            split(p, a, b);
            split(q, c, d);
            const size_t m = cast(size_t)k + 1;
            if (b > 0 && b <= 8192 && d > 0 && d <= 8192 && m <= 8192 / b) {
                Integer base = Integer(1);
                shift(base, b);
                Integer probabilityNumerator = Integer(a);
                base -= probabilityNumerator;
                Integer tail = Integer(1);
                size_t exponent = m;
                while (exponent) {
                    if (exponent & 1) tail *= base;
                    exponent >>= 1;
                    if (exponent) base *= base;
                }
                Integer target = Integer(1);
                shift(target, d);
                Integer queryNumerator = Integer(c);
                target -= queryNumerator;
                const int tailExponent = cast(int)m * b;
                if (d > tailExponent) shift(tail, d - tailExponent);
                else shift(target, tailExponent - d);
                return tail <= target;
            }
        }
    }
    return value >= lq;
}

// Exact dyadic boundaries and queries on either side of them.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        foreach (k, q; [T(0.5), T(0.75), T(0.875)]) {
            assert(geometricInvCDF(nextDown(q), T(0.5)) == k);
            assert(geometricInvCDF(q, T(0.5)) == k);
            assert(geometricInvCDF(nextUp(q), T(0.5)) == k + 1);
        }
        assert(geometricInvCDF(T(0.5), T(0)) == T.infinity);
        assert(geometricInvCDF(T(0), T(0.5)) == -1);
        assert(geometricInvCDF(T(0), T(1)) == 0);
        assert(geometricInvCDF(T(1), T(1)) == T.infinity);
        assert(geometricInvCDF(T(1e-20L), T(0.5)) == 0);
        assert(approxEqual(geometricInvCDF(T(0.5), T(1e-20L)),
            T(6.9314718055994530941e19L), 8 * T.epsilon, T(0)));
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    0.geometricInvCDF(0.5).should == -1;
    0.5.geometricInvCDF(0.5).should == 0;
    0.75.geometricInvCDF(0.5).should == 1;
    0.875.geometricInvCDF(0.5).should == 2;
    0.95.geometricInvCDF(0.5).should == 4;

    0.geometricInvCDF(0.25).should == -1;
    0.25.geometricInvCDF(0.25).should == 0;
    0.4375.geometricInvCDF(0.25).should == 1;
    0.578125.geometricInvCDF(0.25).should == 2;
    0.95.geometricInvCDF(0.25).should == 10;

    0.5.geometricInvCDF(1).should == 0;
    1.geometricInvCDF(0.5).should == double.infinity;
}

/++
Computes the geometric log probability density function (LPMF).

Params:
    k = value to evaluate LPMF
    p = `true` probability

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricLPMF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;
    import mir.math.internal.xlogy: xlog1py;
    return xlog1py(k, -p) + log(p);   
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    0.geometricLPMF(0.5).exp.shouldApprox == 0.geometricPMF(0.5);
    1.geometricLPMF(0.5).exp.shouldApprox == 1.geometricPMF(0.5);
    2.geometricLPMF(0.25).exp.shouldApprox == 2.geometricPMF(0.25);
}
