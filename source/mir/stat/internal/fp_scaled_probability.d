/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_scaled_probability;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

// Internal positive probability arithmetic. Inputs must be finite and nonzero.
// Keep normalized significands and a separate biased exponent; callers must
// bound exponent accumulation within UInt!128 and final results to at most one.
// This is not a general Fp arithmetic API.
package(mir.stat) enum exponentBias = UInt!128.fromHexString("80000000000000000000000000000000");

package(mir.stat) @safe pure nothrow @nogc
void addExponent(ref UInt!128 exponent, long amount)
{
    if (amount >= 0)
        exponent += cast(ulong)amount;
    else
        exponent -= 0UL - cast(ulong)amount;
}

package(mir.stat) struct ProbabilityScaled(uint bits)
{
    Fp!bits fraction;
    UInt!128 exponent = exponentBias;

    @safe pure nothrow @nogc:

    static ProbabilityScaled from(uint inputBits)(Fp!inputBits value)
    {
        ProbabilityScaled result;
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

    ProbabilityScaled multiply(ProbabilityScaled rhs) const
    {
        ProbabilityScaled result;
        result.fraction = fraction * rhs.fraction;
        result.exponent = exponent + rhs.exponent - exponentBias;
        result.normalize();
        return result;
    }

    ProbabilityScaled divide(ProbabilityScaled rhs) const
    {
        auto quotient = fraction.coefficient.toSize!(2 * bits + 64);
        quotient <<= bits + 64;
        quotient /= rhs.fraction.coefficient;
        ProbabilityScaled result;
        result.fraction = Fp!bits(quotient);
        result.fraction.exponent -= bits + 64;
        result.exponent = exponent - rhs.exponent + exponentBias;
        result.normalize();
        return result;
    }

    ProbabilityScaled power(size_t count) const
    {
        auto result = from(Fp!bits(1));
        ProbabilityScaled factor = this;
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

// Exact halfway cases exercise final subnormal rounding independently of
// the transcendental reference cases in the public module.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias S = ProbabilityScaled!256;
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
