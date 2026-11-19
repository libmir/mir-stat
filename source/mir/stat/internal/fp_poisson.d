/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias;

import mir.stat.internal.fp_scaled_probability: scaledDecay;

package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpPoisson(uint bits)(size_t k, Fp!bits lambda)
{
    alias W = Fp!(bits + 192);
    alias Scaled = ProbabilityScaled!(bits + 192);
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
    auto result = scaledDecay(rate);
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
