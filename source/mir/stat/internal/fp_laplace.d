/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_laplace;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

// Construct a normalized half even when a double significand would not fit.
private Fp!bits half(uint bits)()
{
    auto value = Fp!bits(1);
    value.exponent -= 1;
    return value;
}

// Near the median, the first two decay terms can reach a rounding midpoint.
// Keep the cubic term (of order 2^(-3*bits)) when deciding the final side.
private alias Decay(uint bits) = ProbabilityScaled!(3 * bits + 192);

// Align the two coefficients at the smaller exponent before taking the
// absolute difference. Across a larger gap even a full-width low coefficient
// contributes less than a working ulp relative to a one-bit high coefficient.
private auto distance(uint work, uint bits)(Fp!bits x, Fp!bits location)
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
private int position(uint bits)(Fp!bits x, Fp!bits location)
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

// For distinct finite points, retain the separate exponent until the caller
// has applied its normalization. At z >= 2^65 even division by the smallest
// scale (2^long.min) leaves less than half the smallest output quantum.
private auto halfDecay(uint bits, uint work)(Fp!bits x, Fp!bits location, ProbabilityScaled!work scale)
{
    alias S = ProbabilityScaled!work;
    const z = distance!work(x, location).divide(scale);
    if (z.exponent >= exponentBias + 66)
        return S(Fp!work(0));
    auto tail = scaledDecay(z);
    tail.exponent -= 1;
    return tail;
}

package(mir.stat) Fp!bits laplaceDensity(uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(0);
    alias S = Decay!bits;
    const divisor = S.from(scale);
    const numerator = position(x, location) == 0 ? S.from(half!bits) : halfDecay(x, location, divisor);
    if (!numerator.fraction.coefficient)
        return T(0);
    // The peak 0.5 / 2^long.min is representable for every supported width.
    return numerator.divide(divisor).finish!bits;
}

package(mir.stat) Fp!bits laplaceTail(bool upper, uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(x.sign == upper ? 1 : 0);
    const side = position(x, location);
    if (!side)
        return half!bits;
    const smallTail = upper != (side < 0);
    alias S = Decay!bits;
    auto tail = halfDecay(x, location, S.from(scale));
    if (!tail.fraction.coefficient)
        return T(smallTail ? 0 : 1);
    if (smallTail)
        return tail.finish!bits;
    if (tail.exponent + bits + 1 <= exponentBias)
        return T(1);
    tail.fraction.exponent -= cast(long)cast(ulong)(exponentBias - tail.exponent);
    return cast(T) fpProbabilityComplement(tail.fraction);
}

// Add a signed, separately scaled offset to a finite dyadic base.
// Align normalized fractions on a local exponent grid. Beyond the alignment
// window only the side of a rounding midpoint can depend on the smaller term.
private Fp!bits addOffset(uint bits, uint baseBits, uint work)(Fp!baseBits base, ProbabilityScaled!work offset, bool negative)
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

package(mir.stat) Fp!bits laplaceLogDensity(uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(true, long.max, UInt!bits(0));

    // Supply the exact coefficient and exponent of 2*scale separately.
    // logDyadic accepts long.max as an integer exponent even though an Fp
    // with that exponent would encode a special value.
    // With 2*scale = 1-h, two dyadic coordinates can match the first
    // two terms of -(1-h)*log(1-h), leaving a cubic residual. Allow two
    // input widths for cancellation in addition to the result significand.
    alias Log = ProbabilityLog!(3 * bits);
    Log.Context context;
    auto peak = Log.logDyadic(scale.coefficient, scale.exponent + 1, context);
    peak.sign = peak.coefficient && !peak.sign;
    if (!position(x, location))
        return cast(T)peak;
    enum work = 3 * bits + 192;
    const z = distance!work(x, location).divide(ProbabilityScaled!work.from(scale));
    return addOffset!bits(peak, z, true);
}

package(mir.stat) Fp!bits laplaceQuantile(uint bits)(Fp!bits p, Fp!bits location, Fp!bits scale)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    alias T = Fp!bits;
    if (!p.coefficient)
        return T(true, long.max, UInt!bits(0));
    const side = position(p, half!bits);
    if (!side)
        return location;

    // Above the median, 1-p is an exact dyadic at the input width.
    const tail = side < 0 ? p : fpProbabilityComplement(p);
    if (!tail.coefficient)
        return T(false, long.max, UInt!bits(0));
    alias Log = ProbabilityLog!(3 * bits);
    Log.Context context;
    const logTail = Log.logDyadic(tail.coefficient, tail.exponent + 1, context);
    alias S = Log.S;
    const offset = S.from(logTail).multiply(S.from(scale));
    return addOffset!bits(location, offset, side < 0);
}
