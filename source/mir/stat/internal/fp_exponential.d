/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_exponential;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, addExponent;

@safe pure nothrow @nogc:

// finish handles the exponent floor. Check the narrowed significand's
// exponent before finish converts the separately tracked exponent to long.
private Fp!bits finish(uint bits, uint work)(ProbabilityScaled!work value)
{
    if (!value.fraction.coefficient)
        return Fp!bits(0);
    auto exponent = value.exponent;
    addExponent(exponent, (cast(Fp!bits)value.fraction).exponent);
    auto ceiling = exponentBias;
    addExponent(ceiling, long.max);
    if (exponent >= ceiling)
        return Fp!bits(false, long.max, UInt!bits(0));
    return value.finish!bits;
}

package(mir.stat) Fp!bits exponentialQuantile(uint bits)(Fp!bits p, Fp!bits rate)
{
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    if (!p.coefficient)
        return Fp!bits(0);
    // At p = 2^(1-bits), p + p^2/2 is an output midpoint. Keep
    // the cubic term, about 2*bits places below p, through division.
    enum logBits = bits <= 192 ? bits : 2 * bits;
    alias Log = ProbabilityLog!logBits;
    alias S = Log.S;

    // For very small p the log kernel returns p. The omitted correction
    // is strictly positive. A dyadic p/rate can land exactly on a final
    // midpoint, including half the smallest subnormal. Retain its side.
    const top = bits - p.coefficient.ctlz;
    if (p.exponent <= -cast(long)logBits - 256 - top)
    {
        // A nonzero distance from a binary rounding boundary for p/rate
        // can cost another rate coefficient's worth of bits to distinguish.
        alias Wide = ProbabilityScaled!(2 * bits + 192);
        auto quotient = Wide.from(p).divide(Wide.from(rate));
        if (quotient.fraction.coefficient != typeof(quotient.fraction.coefficient).max)
            quotient.fraction.coefficient += 1;
        return finish!bits(quotient);
    }
    // This branch is far above the exponent floor, so widening is exact.
    static if (logBits == bits)
        auto logarithm = Log.complement(p);
    else
        auto logarithm = Log.complement(cast(Fp!logBits)p);
    return finish!bits(logarithm.divide(S.from(rate)));
}

// Keep the full coefficient product and both input exponents. For t >=
// 2^65, even the largest finite rate times exp(-t) rounds to zero.
private auto rateProduct(uint work, uint bits)(Fp!bits x, Fp!bits rate)
{
    alias S = ProbabilityScaled!work;
    return S.from(x).multiply(S.from(rate));
}

package(mir.stat) Fp!bits exponentialDecay(bool density, uint bits)(Fp!bits x, Fp!bits rate)
{
    import mir.stat.internal.fp_scaled_probability: scaledDecay;
    alias T = Fp!bits;
    if (x.isInfinity)
        return T(0);
    if (!x.coefficient)
        return density ? rate : T(1);
    const t = rateProduct!(2 * bits + 192)(x, rate);
    if (t.exponent >= exponentBias + 66)
        return T(0);
    auto result = scaledDecay(t);
    static if (density)
        result = result.multiply(typeof(result).from(rate));
    return finish!bits(result);
}

package(mir.stat) Fp!bits exponentialLower(uint bits)(Fp!bits x, Fp!bits rate)
{
    import mir.stat.internal.fp_probability: fpProbabilityComplement;
    import mir.stat.internal.fp_scaled_probability: scaledDecay;
    alias T = Fp!bits;
    if (x.isInfinity)
        return T(1);
    if (!x.coefficient)
        return T(0);
    enum work = 4 * bits + 192;
    auto t = rateProduct!work(x, rate);
    if (t.exponent + (2 * bits + 4) <= exponentBias)
    {
        // t has at most 2*bits significant bits. The O(t^2) correction
        // cannot cross a rounding boundary except when t is on it.
        // Take its working predecessor to resolve that case downward.
        if (t.fraction.coefficient == (UInt!work(1) << (work - 1)))
        {
            t.fraction.coefficient = UInt!work.max;
            t.exponent -= 1;
        }
        else
            t.fraction.coefficient -= 1;
        return finish!bits(t);
    }
    if (t.exponent >= exponentBias + 66)
        return T(1);
    // At most 2*bits+5 leading places cancel at the lower transition.
    // Complement the guarded decay before rounding to the requested width.
    auto decay = scaledDecay(t);
    if (decay.exponent + bits + 1 <= exponentBias)
        return T(1);
    decay.fraction.exponent -= cast(long)cast(ulong)(exponentBias - decay.exponent);
    return cast(T) fpProbabilityComplement(decay.fraction);
}

package(mir.stat) Fp!bits exponentialLogDensity(uint bits)(Fp!bits x, Fp!bits rate)
{
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    alias T = Fp!bits;
    if (x.isInfinity)
        return T(true,long.max,UInt!bits(0));

    alias Log = ProbabilityLog!(2 * bits + 64);
    enum work = 2 * bits + 256;
    alias S = ProbabilityScaled!work;
    Log.Context context;
    const logRate = Log.logDyadic(rate.coefficient, rate.exponent, context);
    if (!x.coefficient)
    {
        auto result = finish!bits(S.from(logRate));
        result.sign = logRate.sign;
        return result;
    }
    const product = rateProduct!work(x, rate);
    if (!logRate.coefficient)
    {
        auto result = finish!bits(product);
        result.sign = true;
        return result;
    }

    const logMagnitude = S.from(logRate);
    const productLarger = product.exponent > logMagnitude.exponent ||
        (product.exponent == logMagnitude.exponent &&
         product.fraction.coefficient >= logMagnitude.fraction.coefficient);
    auto larger = productLarger ? product : logMagnitude;
    const smaller = productLarger ? logMagnitude : product;
    const gap = larger.exponent - smaller.exponent;
    // Align integer significands before subtraction so cancellation exposes
    // the guarded low bits. For a remote smaller operand keep a sticky bit
    // to distinguish either side of a final rounding midpoint.
    alias I = UInt!(2 * work + 64);
    auto magnitude = larger.fraction.coefficient.toSize!(2 * work + 64);
    magnitude <<= work;
    auto term = I(1);
    if (gap <= UInt!128(work))
        term = smaller.fraction.coefficient.toSize!(2 * work + 64) << (work - cast(uint)gap);
    if (logRate.sign)
        magnitude += term;
    else
        magnitude -= term;
    // Keep all aligned bits until the destination rounds the signed result.
    auto total = ProbabilityScaled!(2 * work + 64).from(Fp!(2 * work + 64)(magnitude));
    total.exponent += larger.exponent - exponentBias - 2 * work;
    auto result = finish!bits(total);
    result.sign = !!magnitude && (logRate.sign || productLarger);
    return result;
}
