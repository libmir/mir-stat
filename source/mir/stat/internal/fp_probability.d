/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.fp_probability;

import mir.bignum.fp: Fp;

// C(a+b,a), with separate counts so their sum need not fit size_t.
// Products round at the requested precision; the final integer division has
// 64 guard bits. This is not a correctly-rounded coefficient guarantee.
// Cost is O(min(a,b)); intermediate products must fit Fp's exponent range.
package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpProbabilityCoefficient(uint bits)(size_t a, size_t b)
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!bits;
    if (a <= 64 && b <= 64 - a) {
        import mir.combinatorics: binomial;
        return T(binomial(a + b, a));
    }
    const size_t smaller = a < b ? a : b;
    const size_t larger = a < b ? b : a;
    T numeratorProduct = T(1);
    T denominatorProduct = T(1);
    foreach (i; 0 .. smaller) {
        UInt!128 numerator = UInt!128(larger);
        numerator += i;
        numerator += 1;
        numeratorProduct *= T(numerator);
        denominatorProduct *= T(i + 1);
    }
    // Avoid Fp division, which currently converts through hardware real.
    UInt!(2 * bits + 64) quotient = numeratorProduct.coefficient.toSize!(2 * bits + 64);
    quotient <<= bits + 64;
    quotient /= denominatorProduct.coefficient;
    T coefficient = T(quotient);
    coefficient.exponent += numeratorProduct.exponent - denominatorProduct.exponent - cast(long)bits - 64;
    return coefficient;
}

// Validate the represented value, including probabilities that round to 0 or
// 1 in a hardware type. Fp's special exponent denotes infinity or NaN.
package(mir.stat) @safe pure nothrow @nogc
bool fpProbabilityValid(uint bits)(Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    if (p.exponent == long.max)
        return false;
    if (!p.coefficient)
        return true;
    if (p.sign || p.exponent > 0)
        return false;
    if (p.exponent <= -cast(long)bits)
        return true;
    UInt!bits one = UInt!bits(1);
    one <<= cast(size_t)-p.exponent;
    return p.coefficient <= one;
}

// Fp does not supply subtraction. Form 1-p exactly as an integer at p's
// binary exponent, then let Fp's integer constructor round to its precision.
// The temporary accommodates cancellation near one and guard bits near zero.
package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpProbabilityComplement(uint bits)(Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    if (!p.coefficient || p.exponent < -2 * cast(long)bits)
        return Fp!bits(1); // p is less than half the spacing immediately below 1.
    UInt!(2 * bits + 64) difference = UInt!(2 * bits + 64)(1);
    difference <<= cast(size_t)-p.exponent;
    difference -= p.coefficient;
    auto result = Fp!bits(difference);
    result.exponent += p.exponent;
    return result;
}

// Compare nonnegative finite values without narrowing or assuming normalized
// coefficients. Align only when exponents differ by fewer than bits; otherwise
// even the smallest nonzero coefficient at the larger exponent is greater.
package(mir.stat) @safe pure nothrow @nogc
bool fpProbabilityLess(uint bits)(Fp!bits a, Fp!bits b)
{
    import mir.bignum.fixed: UInt;
    if (!a.coefficient)
        return !!b.coefficient;
    if (!b.coefficient)
        return false;
    if (a.exponent == b.exponent)
        return a.coefficient < b.coefficient;
    if (a.exponent > b.exponent)
    {
        const gap = cast(ulong)a.exponent - cast(ulong)b.exponent;
        if (gap >= bits)
            return false;
        auto aligned = UInt!(2 * bits)(a.coefficient);
        aligned <<= cast(size_t)gap;
        return aligned < UInt!(2 * bits)(b.coefficient);
    }
    const gap = cast(ulong)b.exponent - cast(ulong)a.exponent;
    if (gap >= bits)
        return true;
    auto aligned = UInt!(2 * bits)(b.coefficient);
    aligned <<= cast(size_t)gap;
    return UInt!(2 * bits)(a.coefficient) < aligned;
}

// p*(1-p)^k with guard precision and a separate intermediate exponent.
package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpGeometric(uint bits)(size_t k, Fp!bits p)
{
    if (!p.coefficient)
        return Fp!bits(0);
    if (!k)
        return p;
    // p < 2^(-bits-66), hence k*p < 2^(-bits-2) for size_t <= 64 bits.
    // Bernoulli's inequality bounds the relative correction by k*p: less
    // than half an output ulp, also for subnormal p. Avoid widening at the floor.
    if (p.exponent < -2 * cast(long)bits - 66)
        return p;
    import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;
    // 64 bits absorb count amplification; 128 more retain small corrections
    // near output midpoints and guard intermediate rounding.
    alias W = Fp!(bits + 192);
    alias S = ProbabilityScaled!(bits + 192);
    const complement = fpProbabilityComplement(cast(W)p);
    if (!complement.coefficient)
        return Fp!bits(0);
    // A represented p < 1 has 1-p >= 2^-bits. Even size_t.max powers
    // therefore have exponent magnitude below bits*2^64, well within UInt!128.
    return S.from(complement).power(k).multiply(S.from(p)).finish!bits;
}

// Widen only probabilities away from the exponent floor. Smaller probabilities
// have a complement indistinguishable from one at the working precision.
private @safe pure nothrow @nogc
Fp!workBits probabilityComplementWide(uint workBits, uint bits)(Fp!bits p)
{
    alias W = Fp!workBits;
    if (p.exponent < -2 * cast(long)workBits)
        return W(1);
    return fpProbabilityComplement(cast(W)p);
}

package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpGeometricTail(bool upper, uint bits)(size_t k, Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;
    alias T = Fp!bits;
    alias W = Fp!(bits + 192);
    alias S = ProbabilityScaled!(bits + 192);
    if (!p.coefficient)
        return T(upper ? 1 : 0);
    if (!k && !upper)
        return p;
    const q = probabilityComplementWide!(bits + 192)(p);
    if (!q.coefficient)
        return T(upper ? 0 : 1);
    const one = S.from(W(1));
    auto power = S.from(q);
    static if (upper)
        return power.power(k).multiply(power).finish!bits;
    else {
        // Start with one trial and append k trials in binary blocks. A block
        // stores q^m and sum(q^i, i=0..m-1). Doubling uses only positive terms.
        S sum = one;
        auto accumulatedPower = power;
        S blockSum = one;
        size_t count = k;
        while (count) {
            if (count & 1) {
                sum = sum.add(accumulatedPower.multiply(blockSum));
                accumulatedPower = accumulatedPower.multiply(power);
            }
            count >>= 1;
            if (count) {
                blockSum = blockSum.multiply(one.add(power));
                power = power.multiply(power);
            }
        }
        auto result = S.from(p).multiply(sum);
        // If the complement rounds to one, n*p is exact at working precision.
        // The true CDF is slightly smaller for n>1. Preserve that direction
        // at final midpoints instead of rounding the polynomial upward.
        if (k && q == W(1))
            result.fraction.coefficient -= 1;
        return result.finish!bits;
    }
}

package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpNegativeBinomial(uint bits)(size_t k, size_t r, Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;
    alias T = Fp!bits;
    alias W = Fp!(bits + 192);
    alias S = ProbabilityScaled!(bits + 192);
    if (r == 1)
        return fpGeometric(k, p);
    if (!p.coefficient)
        return T(0);
    // For p this small and r>=2, C(k+r-1,r-1) <= (k+r)^(r-1).
    // k+r < 2^65, so the mass is below p^r * 2^(65*(r-1)),
    // hence below half the smallest quantum. Avoid huge exponent products.
    if (p.exponent < long.min / 2 - cast(long)bits - 66)
        return T(0);
    const probability = S.from(p);
    if (!k)
        return probability.power(r).finish!bits;
    const q = probabilityComplementWide!(bits + 192)(p);
    if (!q.coefficient)
        return T(0);
    const smaller = k < r - 1 ? k : r - 1;
    const larger = k < r - 1 ? r - 1 : k;
    auto numerator = S.from(W(1));
    auto denominator = numerator;
    foreach (i; 0 .. smaller) {
        auto factor = UInt!128(larger);
        factor += i;
        factor += 1;
        numerator = numerator.multiply(S.from(W(factor)));
        denominator = denominator.multiply(S.from(W(i + 1)));
    }
    auto result = numerator.divide(denominator)
        .multiply(probability.power(r)).multiply(S.from(q).power(k));
    if (q == W(1))
        result.fraction.coefficient -= 1;
    return result.finish!bits;
}
