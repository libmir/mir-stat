/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_location_scale;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityLess;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias;

@safe pure nothrow @nogc:

// Construct a normalized half even when a double significand would not fit.
package(mir.stat) Fp!bits half(uint bits)()
{
    auto value = Fp!bits(1);
    value.exponent -= 1;
    return value;
}

// Align the two coefficients at the smaller exponent before taking the
// absolute difference. Across a larger gap even a full-width low coefficient
// contributes less than a working ulp relative to a one-bit high coefficient.
package(mir.stat) auto distance(uint work, uint bits)(Fp!bits x, Fp!bits location)
{
    alias S = ProbabilityScaled!work;
    if (!x.coefficient)
        return S.from(location);
    if (!location.coefficient)
        return S.from(x);
    const high = x.exponent >= location.exponent ? x : location;
    const low = x.exponent >= location.exponent ? location : x;
    const gap = cast(ulong)high.exponent - cast(ulong)low.exponent;
    if (gap >= work + bits)
        return S.from(high);
    // The exact sum fits even when both inputs have full coefficients.
    enum capacity = work + 2 * bits + 64;
    const a = high.coefficient.toSize!capacity << cast(size_t)gap;
    const b = low.coefficient.toSize!capacity;
    const c = high.sign != low.sign ? a + b : a >= b ? a - b : b - a;
    return S.from(Fp!capacity(false, low.exponent, c));
}

// Compare represented values, including signed zero and unnormalized coefficients.
package(mir.stat) int position(uint bits)(Fp!bits x, Fp!bits location)
{
    if (x.sign != location.sign)
    {
        if (!x.coefficient && !location.coefficient)
            return 0;
        return x.sign ? -1 : 1;
    }
    if (fpProbabilityLess(x, location))
        return x.sign ? 1 : -1;
    if (fpProbabilityLess(location, x))
        return x.sign ? -1 : 1;
    return 0;
}

// Add a signed, separately scaled offset to a finite dyadic base.
// Align normalized fractions on a local exponent grid. Beyond the alignment
// window only the side of a rounding midpoint can depend on the smaller term.
package(mir.stat) Fp!bits addOffset(uint bits, uint baseBits, uint work)(Fp!baseBits base, ProbabilityScaled!work offset, bool negative)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_scaled_probability: addExponent;
    alias T = Fp!bits;
    alias W = Fp!(2 * work + 64);
    alias S = ProbabilityScaled!(2 * work + 64);
    enum width = 2 * work + 64;
    auto magnitude = offset.fraction.coefficient.toSize!width;
    auto exponent = offset.exponent;
    if (base.coefficient)
    {
        const scaledBase = ProbabilityScaled!work.from(base);
        auto baseMagnitude = scaledBase.fraction.coefficient.toSize!width;
        const gap = exponent > scaledBase.exponent ? exponent - scaledBase.exponent : scaledBase.exponent - exponent;
        const shift = gap > UInt!128(work) ? work + 1 : cast(uint)gap;
        if (scaledBase.exponent > exponent)
        {
            baseMagnitude <<= shift;
            exponent = scaledBase.exponent - shift;
        }
        else
        {
            magnitude <<= shift;
            exponent -= shift;
        }
        if (base.sign == negative)
            magnitude += baseMagnitude;
        else if (magnitude >= baseMagnitude)
            magnitude -= baseMagnitude;
        else
        {
            magnitude = baseMagnitude - magnitude;
            negative = base.sign;
        }
    }
    if (!magnitude)
        return T(0);
    auto value = S.from(W(magnitude));
    value.exponent += exponent - exponentBias - work;
    auto narrowedExponent = value.exponent;
    addExponent(narrowedExponent, (cast(T)value.fraction).exponent);
    if (narrowedExponent >= exponentBias + cast(ulong)long.max)
        return T(negative, long.max, UInt!bits(0));
    auto result = value.finish!bits;
    result.sign = negative;
    return result;
}

// At the resolution limit the guarded terms coincide. The result is +0.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias S = ProbabilityScaled!384;
    const peak = Fp!384(1);
    const residual = addOffset!64(peak, S.from(peak), true);
    assert(!residual.coefficient && !residual.sign);
}
