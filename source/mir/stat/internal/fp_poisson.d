/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias;

// exp(-rate), using a fixed-point alternating series for x <= 1/2 followed
// by at most 67 squarings. The caller supplies 192 extra bits: 67 cover
// error amplification in squaring, 64 cover full-width powers/products,
// and the remaining bits absorb series truncation and intermediate rounding.
private @safe pure nothrow @nogc
ProbabilityScaled!bits poissonDecay(uint bits)(ProbabilityScaled!bits rate)
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
    auto result = ProbabilityScaled!bits.from(value);
    foreach (i; 0 .. squarings)
        result = result.multiply(result);
    return result;
}

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
