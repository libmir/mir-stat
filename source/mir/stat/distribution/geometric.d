/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution).

The count is the number of failures before the first success in independent
trials with success probability $(D p). Thus $(D k == 0) means the first trial
succeeds; it does not mean that no trials were performed. The usual
distribution has $(D 0 < p <= 1). The API also accepts $(D p == 0) as a
limiting case with no success and zero mass at every finite count.

PMF accepts built-in floating-point and $(D Fp) probabilities. The other
functions use built-in floating-point probabilities. PMF and LPMF take a $(D size_t) count. CDF and CCDF additionally
accept a floating-point threshold $(D x >= -1), using the integer part of
nonnegative finite thresholds. Inverse CDF returns a floating-point count
and has explicit endpoint conventions described below.

The tables assume valid parameters except where an invalid input is listed.
Finite expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts does
not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.geometric;

import mir.internal.utility: isFloatingPoint;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid;

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

Returns the point probability $(D P(K == k)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D p)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D 0)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate PMF
    p = per-trial success probability in $(D [0, 1])

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

/++
Extended-range and extended-precision PMF for an $(D Fp) probability.

Evaluates the represented probability without hardware-floating-point conversions.
Tested widths are $(D Fp!64), $(D Fp!128), and $(D Fp!256). The complement and
power retain 192 extra working bits, with a separate intermediate exponent and
final subnormal rounding. This reduces rounding error but does not promise
universally correct rounding. Counts retain their full $(D size_t) width and
power evaluation costs $(D O(log(k + 1))) multiplications.
+/
@safe pure nothrow @nogc
T geometricPMF(T)(const size_t k, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpGeometric;
    return fpGeometric(k, p);
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
Computes the geometric cumulative distribution function (CDF).

Returns the inclusive lower-tail probability $(D P(K <= k)), or $(D P(K <= x))
for a floating-point threshold. The interval $(D -1 <= x && x < 0) is accepted
and lies below the first outcome.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) or $(D 0 <= x && x < 1)) $(TD $(D p)))
    $(TR $(TD $(D -1 <= x && x < 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0) and $(D x >= 0) (or any $(D k))) $(TD $(D 0), including $(D x == T.infinity)))
    $(TR $(TD $(D p > 0) and $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1) and $(D x >= 0) (or any $(D k))) $(TD $(D 1)))
    $(TR $(TD $(D x < -1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate CDF
    x = floating-point threshold, at least -1
    p = per-trial success probability in $(D [0, 1])

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
Computes the geometric complementary cumulative distribution function (CCDF).

Returns the strict upper-tail probability $(D P(K > k)), or $(D P(K > x))
for a floating-point threshold. This excludes the mass at the threshold.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) or $(D 0 <= x && x < 1)) $(TD $(D 1 - p)))
    $(TR $(TD $(D -1 <= x && x < 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 0) and $(D x >= 0) (or any $(D k))) $(TD $(D 1), including $(D x == T.infinity)))
    $(TR $(TD $(D p > 0) and $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D x >= 0) (or any $(D k))) $(TD $(D 0)))
    $(TR $(TD $(D x < -1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate CCDF
    x = floating-point threshold, at least -1
    p = per-trial success probability in $(D [0, 1])

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

/// Count failures, and include the whole CDF step for a fractional threshold.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(geometricPMF(0, 0.5) == 0.5); // Success on the first trial.
    assert(geometricPMF(1, 0.5) == 0.25); // One failure, then success.
    assert(approxEqual(geometricCDF(1.5, 0.5), 0.75));
    assert(approxEqual(geometricCCDF(1.5, 0.5), 0.25));
}

/++
Computes the geometric inverse cumulative distribution function (InvCDF).

For $(D 0 < q && q < 1) and $(D 0 < p && p < 1), returns the smallest
nonnegative count whose CDF reaches $(D q). The floating return type cannot
represent every integer at large counts. Close comparisons use bounded exact
binary arithmetic where possible; larger cases retain floating-point accuracy.

The $(D q == 1) convention takes precedence over $(D p == 1). In particular,
$(D geometricInvCDF(1.0, 1.0)) returns infinity even though all finite mass
is at zero. The returned $(D -1) at the lower endpoint is a boundary convention,
not a possible number of failures.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 1)) $(TD $(D T.infinity), including when $(D p == 1)))
    $(TR $(TD $(D q < 1) and $(D p == 1)) $(TD $(D 0)))
    $(TR $(TD $(D q == 0) and $(D p < 1)) $(TD $(D -1)))
    $(TR $(TD $(D 0 < q && q < 1) and $(D p == 0)) $(TD $(D T.infinity)))
    $(TR $(TD $(D q < 0), $(D q > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    q = cumulative probability in $(D [0, 1])
    p = per-trial success probability in $(D [0, 1])

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
    // Keep ordinary double logarithms in double; widen float only. The exact
    // fallback below resolves ambiguous steps without widening every call.
    static if (is(T == float)) alias F = double;
    else alias F = T;
    const F lp = negLog1m(cast(F)p);
    const F lq = negLog1m(cast(F)q);
    F guess = ceil(lq / lp) - 1;
    if (guess < 0)
        guess = 0;
    // At counts beyond consecutive-integer precision, stepping by one may
    // leave the value unchanged. Return the representable approximation there.
    if (guess + 1 == guess || guess == F.infinity)
        return cast(T)guess;
    while (guess > 0 && geometricQuantileReached(guess - 1, p, q, lp, lq))
        --guess;
    while (guess + 1 != guess && !geometricQuantileReached(guess, p, q, lp, lq))
        ++guess;
    return cast(T)guess;
}

/// The inverse CDF has explicit conventions at probability zero and one.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(geometricInvCDF(0.0, 0.5) == -1);
    assert(geometricInvCDF(0.75, 0.5) == 1);
    assert(geometricInvCDF(1.0, 0.5) == double.infinity);
    assert(geometricInvCDF(0.0, 1.0) == 0);
    assert(geometricInvCDF(1.0, 1.0) == double.infinity);
}

// Comparing logarithms avoids reconstructing a CDF close to one. Near a
// boundary, do not snap approximately equal values to an integer: adjacent
// queries can lie on opposite sides of the exact CDF step.
private @safe pure nothrow @nogc
bool geometricQuantileReached(T, F)(F k, T p, T q, F lp, F lq)
{
    import std.math: fabs, frexp, ldexp;
    import mir.bignum.integer: BigInt;
    const F value = (k + 1) * lp;
    if (k < 8192 && fabs(value - lq) <= 32 * F.epsilon * (value + lq)) {
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
        // Dividing the smallest positive query by -log(1-p) may underflow
        // before ceil. The first count still reaches this positive query.
        assert(geometricInvCDF(nextUp(T(0)), T(0.99)) == 0);
        // The next query above CDF(0)=p belongs to count 1. Rounded logarithms
        // can initially suggest count 0, requiring an upward correction.
        const T probability = T(0.0003);
        assert(geometricInvCDF(nextUp(probability), probability) == 1);
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
Computes the geometric log probability mass function (LPMF).

Returns the natural logarithm of $(D P(K == k)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) and $(D p > 0)) $(TD $(D log(p))))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 1) and $(D k == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate LPMF
    p = per-trial success probability in $(D [0, 1])

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

// BEGIN geometric references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.distribution.negative_binomial: negativeBinomialPMF;
    {
        alias T = Fp!64;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -65, UInt!64.fromHexString("8000000000000000"))),
            Reference(17UL, T(false, -64, UInt!64.fromHexString("c000000000000000")), T(false, -98, UInt!64.fromHexString("c000000000000000"))),
            Reference(1000UL, T(false, -66, UInt!64.fromHexString("8000000000000000")), T(false, -259, UInt!64.fromHexString("a3b3890f36f74447"))),
            Reference(18446744073709551615UL, T(false, -128, UInt!64.fromHexString("8000000000000000")), T(false, -129, UInt!64.fromHexString("9b4597e37cb04ff4"))),
            Reference(18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -129, UInt!64.fromHexString("bc5ab1b16779be36"))),
            Reference(1152921504606846976UL, T(false, -123, UInt!64.fromHexString("8000000000000000")), T(false, -125, UInt!64.fromHexString("bc5ab1b16779be30"))),
            Reference(7UL, T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -512, UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(0UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(1000UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(65UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -128, UInt!64.fromHexString("ffffffffffffffbf"))),
            Reference(100UL, T(false, -73, UInt!64.fromHexString("8000000000000000")), T(false, -74, UInt!64.fromHexString("e82bbc2a781ed0fb"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(17UL, T(false, -128, UInt!128.fromHexString("c0000000000000000000000000000000")), T(false, -162, UInt!128.fromHexString("c0000000000000000000000000000000"))),
            Reference(1000UL, T(false, -130, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -323, UInt!128.fromHexString("a3b3890f36f74446be82c7214803d5bf"))),
            Reference(18446744073709551615UL, T(false, -256, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -257, UInt!128.fromHexString("ffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -193, UInt!128.fromHexString("bc5ab1b16779be35d3eae7ddd466d137"))),
            Reference(1152921504606846976UL, T(false, -187, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -189, UInt!128.fromHexString("bc5ab1b16779be2f92e80179e4dc0049"))),
            Reference(7UL, T(false, -128, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")), T(false, -1024, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(0UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1000UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(65UL, T(false, -255, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -256, UInt!128.fromHexString("ffffffffffffffffffffffffffffffbf"))),
            Reference(100UL, T(false, -137, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -138, UInt!128.fromHexString("e82bbc2a781ed0fadd053012bc81d36d"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(17UL, T(false, -256, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000")), T(false, -290, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1000UL, T(false, -258, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -451, UInt!256.fromHexString("a3b3890f36f74446be82c7214803d5bea9f6b40967456fb173e93441f785bbe5"))),
            Reference(18446744073709551615UL, T(false, -512, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -513, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -321, UInt!256.fromHexString("bc5ab1b16779be35d3eae7ddd466d136a6fe9c010eb169994b346efec1bf3cab"))),
            Reference(1152921504606846976UL, T(false, -315, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -317, UInt!256.fromHexString("bc5ab1b16779be2f92e80179e4dc0048c9b3e38c2c2c6680ed787ae2afa36e19"))),
            Reference(7UL, T(false, -256, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")), T(false, -2048, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(0UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1000UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(65UL, T(false, -511, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -512, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffbf"))),
            Reference(100UL, T(false, -265, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -266, UInt!256.fromHexString("e82bbc2a781ed0fadd053012bc81d36d283f7cae01c8ed11916fddc996a18599"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
}
// END geometric references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.distribution.negative_binomial: negativeBinomialPMF;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (k; [size_t(0), size_t(1), size_t.max]) {
            assert(geometricPMF(k, T(0)) == T(0));
            assert(geometricPMF(k, T(1)) == T(k == 0 ? 1 : 0));
            auto p = T(false, long.min, UInt!bits(1));
            assert(geometricPMF(k, p) == p);
            assert(negativeBinomialPMF(k, 1, p) == p);
        }
        static if (size_t.sizeof == 8) {
            const p = T(0.5);
            assert(geometricPMF(cast(size_t)long.max, p) == T(false, long.min, UInt!bits(1)));
            assert(geometricPMF(cast(size_t)long.max + 1, p) == T(0));
            assert(geometricPMF(size_t.max, p) == T(0));
        }
        static assert(is(typeof(geometricPMF(1, T(0.5))) == T));
    }}
}

version(mir_stat_test)
@system unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        foreach (p; [-1.0, 2.0, double.infinity, -double.infinity, double.nan])
            assertThrown!AssertError(geometricPMF(1, Fp!128(p)));
    }
}

/// The probability type selects the extended-precision PMF.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    const probability = Fp!128(0.25);
    const mass = geometricPMF(2, probability);
    assert(mass == Fp!128(0.140625));
}

// The tiny-probability shortcut must agree with guarded evaluation on either
// side of its exponent threshold, including nonnormalized coefficients.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability: fpProbabilityLess;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (e; [-2L * bits - 67, -2L * bits - 66, -2L * bits - 65]) {
            auto p = T(1);
            p.exponent = e;
            assert(geometricPMF(size_t.max, p) == p);
            const raw = T(false, e, UInt!bits(1));
            const actual = geometricPMF(size_t.max, raw);
            assert(!fpProbabilityLess(actual, raw) && !fpProbabilityLess(raw, actual));
        }
    }}
}
