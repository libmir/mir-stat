/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_hypergeometric_tails;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.bignum.integer: BigInt;
import mir.stat.distribution.hypergeometric: HypergeometricTailResult, HypergeometricTailStatus,
    HypergeometricInverseResult, HypergeometricInverseStatus;
import mir.stat.internal.fp_probability: fpProbabilityLess;

// n <= capacity - 1 after complementing and exchanging the margins.
// C(N,n) <= N^n; an adjacent numerator may temporarily acquire two
// extra size_t factors. Final division also needs room for the result bits.
private struct ExactTails(uint bits, uint capacity)
{
    alias T = Fp!bits;
    alias Integer = BigInt!(capacity + 2 + (bits + 63) / 64);

    @safe pure nothrow @nogc:

    private static void multiply(ref Integer value, size_t factor)
    {
        const overflow = value.opOpAssign!"*"(factor);
        assert(!overflow);
    }

    private static void divideExactly(ref Integer value, size_t divisor)
    {
        if (divisor <= uint.max) {
            const remainder = value.opOpAssign!"/"(cast(uint) divisor);
            assert(!remainder);
            value.normalize();
        } else {
            auto wideDivisor = BigInt!1(divisor);
            Integer remainder;
            value.divMod(wideDivisor, value, remainder);
            assert(!remainder.length);
        }
    }

    private static void choose(ref Integer value, size_t population, size_t count)
    {
        if (count > population - count) count = population - count;
        value = 1;
        foreach (i; 1 .. count + 1) {
            multiply(value, population - count + i);
            divideExactly(value, i);
        }
    }

    // Normalize the rational before division. The quotient has exactly
    // bits significant bits; its remainder resolves even an exact midpoint.
    private static T roundedRatio(ref const Integer numerator, ref const Integer denominator)
    {
        assert(numerator.length && numerator < denominator);
        const offset = denominator.bitLength - numerator.bitLength;
        Integer quotient = numerator;
        quotient <<= offset;
        const extra = bits - 1 + (quotient < denominator);
        quotient <<= extra;
        Integer remainder;
        quotient.divMod(denominator, quotient, remainder);
        remainder <<= 1;
        UInt!(bits + 64) coefficient;
        coefficient.data[0 .. quotient.length] = quotient.data[0 .. quotient.length];
        if (remainder > denominator || (remainder == denominator && coefficient.bt(0)))
            coefficient += 1;
        auto result = T(coefficient);
        result.exponent -= cast(long)(offset + extra);
        return result;
    }

    private static void advance(ref Integer term, size_t i, size_t N, size_t K, size_t n)
    {
        multiply(term, K - i);
        multiply(term, n - i);
        divideExactly(term, i + 1);
        divideExactly(term, (N - K) - (n - i) + 1);
    }

    private static void retreat(ref Integer term, size_t i, size_t N, size_t K, size_t n)
    {
        multiply(term, i);
        multiply(term, (N - K) - (n - i));
        divideExactly(term, K - i + 1);
        divideExactly(term, n - i + 1);
    }

    // The smallest cumulative integer numerator meeting q is ceil(q*D).
    // For positive products, ceil(x / 2^s) = 1 + ((x-1) >> s).
    private static void targetRank(ref Integer target, ref const Integer denominator, T q)
    {
        BigInt!(capacity + 1) count;
        count = denominator;
        auto coefficient = BigInt!((bits + 63)/64)(q.coefficient);
        target.multiply(count, coefficient);
        Integer one = Integer(1);
        target -= one;
        const shift = 0UL - cast(ulong)q.exponent;
        if (shift >= target.bitLength) {
            target = 1;
            return;
        }
        target >>= cast(size_t)shift;
        target += one;
    }

    static size_t quantile(T q, size_t N, size_t K, size_t n, bool flipped,
        size_t lower, size_t upper)
    {
        Integer denominator, rank, term, sum;
        choose(denominator,N,n);
        targetRank(rank,denominator,q);
        enum half = T(false,-1,UInt!bits(1));
        const fromUpper = fpProbabilityLess(half,q);
        if (fromUpper) {
            // Count rank from the opposite end with the same inclusive test.
            denominator -= rank;
            Integer one = Integer(1);
            denominator += one;
            rank = denominator;
        }
        const descend = flipped != fromUpper;
        choose(term,descend ? K : N-K,n);
        sum = term;
        size_t offset;
        while (sum < rank) {
            assert(offset < n);
            if (descend) retreat(term,n-offset,N,K,n);
            else advance(term,offset,N,K,n);
            const overflow = sum.opOpAssign!"+"(term);
            assert(!overflow);
            ++offset;
        }
        return fromUpper ? upper-offset : lower+offset;
    }

    static HypergeometricTailResult!T evaluate(size_t k, size_t N, size_t K, size_t n, bool flipped)
    {
        Integer denominator, term, sum;
        choose(denominator, N, n);
        // The canonical support is [0,n]. Start at the nearer endpoint and
        // update exact counts of samples by the adjacent hypergeometric ratio.
        const lower = k + 1 <= n - k;
        if (lower) {
            choose(term, N - K, n);
            sum = term;
            foreach (i; 0 .. k) {
                advance(term,i,N,K,n);
                const overflow = sum.opOpAssign!"+"(term);
                assert(!overflow);
            }
        } else {
            choose(term, K, n);
            sum = term;
            for (size_t i = n; i > k + 1; --i) {
                retreat(term,i,N,K,n);
                const overflow = sum.opOpAssign!"+"(term);
                assert(!overflow);
            }
        }

        // Both tails have positive mass here. Complement the integer count
        // before rounding, so a CDF rounded to one can coexist with a tiny CCDF.
        assert(sum.length && sum < denominator);
        term = denominator;
        const overflow = term.opOpAssign!"-"(sum);
        assert(!overflow);
        const direct = roundedRatio(sum, denominator);
        const complement = roundedRatio(term, denominator);
        return lower != flipped
            ? HypergeometricTailResult!T(HypergeometricTailStatus.found, direct, complement)
            : HypergeometricTailResult!T(HypergeometricTailStatus.found, complement, direct);
    }
}

package(mir.stat) @safe pure nothrow @nogc
HypergeometricTailResult!(Fp!bits) exactHypergeometricTails(uint bits)(
    size_t k, size_t N, size_t K, size_t n, size_t maxSupport)
{
    alias T = Fp!bits;
    alias Result = HypergeometricTailResult!T;
    if (k == (K < n ? K : n))
        return Result(HypergeometricTailStatus.found, T(1), T(0));

    const lower = n > N-K ? n-(N-K) : 0;
    const upper = K < n ? K : n;
    const flipped = canonicalize(N,K,n);
    k = flipped ? upper-k-1 : k-lower;
    if (n >= maxSupport) return Result.init;
    if (n < 64) return ExactTails!(bits,64).evaluate(k,N,K,n,flipped);
    if (n < 256) return ExactTails!(bits,256).evaluate(k,N,K,n,flipped);
    // Enforce the fixed workspace capacity even when caller contracts are disabled.
    if (n < 1024) return ExactTails!(bits,1024).evaluate(k,N,K,n,flipped);
    return Result.init;
}

// After these symmetries the support is [0,n]. An odd number of
// reflections reverses its ordering; transposition preserves it.
private @safe pure nothrow @nogc
bool canonicalize(size_t N, ref size_t K, ref size_t n)
{
    bool flipped;
    if (n > N-n) { n = N-n; flipped = !flipped; }
    if (K > N-K) { K = N-K; flipped = !flipped; }
    if (n > K) {
        const temp = n;
        n = K;
        K = temp;
    }
    return flipped;
}

package(mir.stat) @safe pure nothrow @nogc
HypergeometricInverseResult exactHypergeometricInverse(uint bits)(
    Fp!bits q, size_t N, size_t K, size_t n, size_t maxSupport)
{
    alias S = HypergeometricInverseStatus;
    if (!q.coefficient) return HypergeometricInverseResult(S.found,0);
    const lower = n > N-K ? n-(N-K) : 0;
    const upper = K < n ? K : n;
    if (lower == upper || !fpProbabilityLess(q,Fp!bits(1)))
        return HypergeometricInverseResult(S.found,upper);
    const flipped = canonicalize(N,K,n);
    if (n >= maxSupport || n >= 1024) return HypergeometricInverseResult.init;
    size_t count;
    if (n < 64) count = ExactTails!(bits,64).quantile(q,N,K,n,flipped,lower,upper);
    else if (n < 256) count = ExactTails!(bits,256).quantile(q,N,K,n,flipped,lower,upper);
    else count = ExactTails!(bits,1024).quantile(q,N,K,n,flipped,lower,upper);
    return HypergeometricInverseResult(S.found,count);
}

// Bypass the public budget precondition to exercise the dispatcher's own limit.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(exactHypergeometricInverse(Fp!64(0.5),2048,1024,1024,1025).status
        == HypergeometricInverseStatus.supportLimit);
    alias evaluate = exactHypergeometricTails!64;
    assert(evaluate(0, 2048, 1024, 1024, 1025).status == HypergeometricTailStatus.supportLimit);
    assert(evaluate(0, size_t.max, 1280, 1280, size_t.max).status == HypergeometricTailStatus.supportLimit);
}

// Synthetic rationals put the remainder exactly at a midpoint. Exercise
// both parities, the carry to one, and the two sides of that boundary.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias E = ExactTails!(64, 64);
    alias B = E.Integer;
    alias T = Fp!64;
    B one = B(1), denominator = B(1);
    denominator <<= 65;
    B a = B(1);
    a <<= 64;
    a += one;
    assert(E.roundedRatio(a,denominator) == T(0.5));
    a += one;
    a += one;
    assert(E.roundedRatio(a,denominator) == T(false,-64,UInt!64(0x8000_0000_0000_0002UL)));
    a = denominator;
    a -= one;
    assert(E.roundedRatio(a,denominator) == T(1));
    denominator <<= 1;
    a <<= 1;
    a -= one;
    assert(E.roundedRatio(a,denominator) == T(false,-64,UInt!64(ulong.max)));
    a += one;
    a += one;
    assert(E.roundedRatio(a,denominator) == T(1));
}
