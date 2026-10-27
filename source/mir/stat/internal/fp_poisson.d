/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

// Poisson intermediates can exceed Fp's exponent range even when the final
// mass fits. Keep normalized significands and a separate, biased exponent.
// This is private to this calculation, not a general Fp arithmetic API.
private enum exponentBias = UInt!128.fromHexString("80000000000000000000000000000000");

private @safe pure nothrow @nogc
void addExponent(ref UInt!128 exponent, long amount)
{
    if (amount >= 0)
        exponent += cast(ulong)amount;
    else
        exponent -= 0UL - cast(ulong)amount;
}

private struct PoissonScaled(uint bits)
{
    Fp!bits fraction;
    UInt!128 exponent = exponentBias;

    @safe pure nothrow @nogc:

    static PoissonScaled from(uint inputBits)(Fp!inputBits value)
    {
        PoissonScaled result;
        result.fraction = Fp!bits(value.coefficient);
        addExponent(result.exponent, value.exponent);
        result.normalize();
        return result;
    }

    void normalize()
    {
        addExponent(exponent, fraction.exponent + bits);
        fraction.exponent = -cast(long)bits;
    }

    PoissonScaled multiply(PoissonScaled rhs) const
    {
        PoissonScaled result;
        result.fraction = fraction * rhs.fraction;
        result.exponent = exponent + rhs.exponent - exponentBias;
        result.normalize();
        return result;
    }

    PoissonScaled divide(PoissonScaled rhs) const
    {
        auto quotient = fraction.coefficient.toSize!(2 * bits + 64);
        quotient <<= bits + 64;
        quotient /= rhs.fraction.coefficient;
        PoissonScaled result;
        result.fraction = Fp!bits(quotient);
        result.fraction.exponent -= bits + 64;
        result.exponent = exponent - rhs.exponent + exponentBias;
        result.normalize();
        return result;
    }

    PoissonScaled power(size_t count) const
    {
        auto result = from(Fp!bits(1));
        PoissonScaled factor = this;
        while (count)
        {
            if (count & 1)
                result = result.multiply(factor);
            count >>= 1;
            if (count)
                factor = factor.multiply(factor);
        }
        return result;
    }

    Fp!resultBits finish(uint resultBits)() const
    {
        auto minimum = exponentBias;
        addExponent(minimum, long.min);
        // Round subnormal results directly at their final exponent, avoiding
        // both a premature Fp underflow and double rounding through resultBits.
        if (exponent < minimum + resultBits)
        {
            const distance = minimum + bits - exponent;
            if (distance > UInt!128(bits))
                return Fp!resultBits(0);
            const shift = cast(size_t)distance;
            UInt!bits coefficient;
            UInt!bits remainder = fraction.coefficient;
            if (shift < bits)
            {
                coefficient = fraction.coefficient >> shift;
                remainder -= coefficient << shift;
            }
            auto half = UInt!bits(1);
            half <<= shift - 1;
            if (remainder > half || (remainder == half && coefficient.bt(0)))
                coefficient += 1;
            if (!coefficient)
                return Fp!resultBits(0);
            return Fp!resultBits(false, long.min, coefficient.toSize!resultBits);
        }
        auto result = cast(Fp!resultBits)fraction;
        UInt!128 finalExponent = exponent;
        addExponent(finalExponent, result.exponent);
        result.exponent = cast(long)cast(ulong)(finalExponent - exponentBias);
        return result;
    }
}

// exp(-rate), using a fixed-point alternating series for x <= 1/2 followed
// by at most 67 squarings. The caller supplies 192 extra bits: 67 cover
// error amplification in squaring, 64 cover full-width powers/products,
// and the remaining bits absorb series truncation and intermediate rounding.
private @safe pure nothrow @nogc
PoissonScaled!bits poissonDecay(uint bits)(PoissonScaled!bits rate)
{
    uint squarings;
    if (rate.exponent >= exponentBias)
        squarings = cast(uint)(rate.exponent - exponentBias) + 1;
    auto shift = exponentBias + squarings - rate.exponent;
    UInt!(bits + 64) x;
    if (shift < UInt!128(bits))
        x = rate.fraction.coefficient.toSize!(bits + 64) >> cast(size_t)shift;
    auto term = UInt!(bits + 64)(1);
    term <<= bits;
    auto sum = term;
    // A positive rate below the fixed-point grid still makes exp(-rate) < 1.
    // Retain that direction with one working ulp rather than replacing the
    // exponential by exactly one and rounding polynomial midpoints upward.
    if (!x)
        sum -= 1;
    for (uint n = 1; ; ++n)
    {
        auto product = term.toSize!(2 * bits + 128);
        product *= x;
        product >>= bits;
        product /= UInt!64(n);
        term = product.toSize!(bits + 64);
        if (!term)
            break;
        if (n & 1)
            sum -= term;
        else
            sum += term;
    }
    auto value = Fp!bits(sum);
    value.exponent -= bits;
    auto result = PoissonScaled!bits.from(value);
    foreach (i; 0 .. squarings)
        result = result.multiply(result);
    return result;
}

package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpPoisson(uint bits)(size_t k, Fp!bits lambda)
{
    alias W = Fp!(bits + 192);
    alias Scaled = PoissonScaled!(bits + 192);
    if (lambda.exponent == long.max)
        return Fp!bits(0); // The public contract permits only positive infinity.
    const rate = Scaled.from(lambda);
    // For lambda >= 2^66 and k <= size_t.max <= 2^64-1, k <= lambda/4.
    // k! >= (k/e)^k gives log(P) <= -lambda/4, below Fp's exponent floor.
    if (rate.exponent >= exponentBias + 67)
        return Fp!bits(0);
    // For lambda < 1, P <= lambda^k <= 2^(k * ceil(log2(lambda))).
    // This also bounds the negative exponents used below before multiplication.
    if (k && rate.exponent < exponentBias)
    {
        auto magnitude = exponentBias - rate.exponent;
        magnitude *= UInt!64(k);
        auto floor = UInt!128(1);
        floor <<= 63;
        floor += 1;
        if (magnitude >= floor)
            return Fp!bits(0);
    }
    auto result = poissonDecay(rate);
    if (k)
    {
        result = result.multiply(rate.power(k));
        // Batch exact machine-word products, as in factorial, but retain the
        // wider exponent separately and use the requested working precision.
        import mir.checkedint: mulu;
        auto factorial = Scaled.from(W(1));
        ulong product = 1;
        foreach (i; 0 .. k)
        {
            bool overflow;
            auto next = mulu(product, cast(ulong)i + 1, overflow);
            if (overflow)
            {
                factorial = factorial.multiply(Scaled.from(W(product)));
                product = cast(ulong)i + 1;
            }
            else
                product = next;
        }
        factorial = factorial.multiply(Scaled.from(W(product)));
        result = result.divide(factorial);
    }
    return result.finish!bits;
}

// Exact halfway cases exercise final subnormal rounding independently of
// the transcendental reference cases in the public module.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias S = PoissonScaled!256;
    auto half = S.from(Fp!256(1));
    half.exponent = exponentBias;
    addExponent(half.exponent, long.min);
    assert(half.finish!64 == Fp!64(0)); // half of the smallest subnormal
    half.fraction.coefficient += 1;
    assert(half.finish!64.coefficient == 1);

    auto oneAndHalf = S.from(Fp!256(1.5));
    oneAndHalf.exponent = exponentBias;
    addExponent(oneAndHalf.exponent, long.min + 1);
    assert(oneAndHalf.finish!64.coefficient == 2); // ties to even
}
