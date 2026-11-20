/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_logistic;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, half, position;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

// Near one half, the cubic term of the logistic tail can decide rounding
// after its linear term reaches a midpoint. Keep it through the division.
private alias Decay(uint bits) = ProbabilityScaled!(3 * bits + 192);

// At z >= 2^65 even division by the smallest finite scale cannot rescue
// a representable density. A zero fraction marks this saturated decay.
private auto decay(uint work)(ProbabilityScaled!work z)
{
    if (z.exponent >= exponentBias + 66)
        return ProbabilityScaled!work(Fp!work(0));
    return scaledDecay(z);
}

package(mir.stat) Fp!bits logisticTail(bool upper, uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(x.sign == upper ? 1 : 0);
    const side = position(x, location);
    if (!side)
        return half!bits;
    const small = upper != (side < 0);
    alias S = Decay!bits;
    const z = distance!(3 * bits + 192)(x, location).divide(S.from(scale));
    const t = decay(z);
    if (!t.fraction.coefficient)
        return T(small ? 0 : 1);
    const one = S.from(T(1));
    const numerator = small ? t : one;
    return numerator.divide(one.add(t)).finish!bits;
}

package(mir.stat) Fp!bits logisticDensity(uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(0);
    alias S = Decay!bits;
    const divisor = S.from(scale);
    const one = S.from(T(1));
    if (!position(x, location))
    {
        S quarter = one;
        quarter.exponent -= 2;
        return quarter.divide(divisor).finish!bits;
    }
    const z = distance!(3 * bits + 192)(x, location).divide(divisor);
    const t = decay(z);
    if (!t.fraction.coefficient)
        return T(0);
    const denominator = one.add(t);
    return t.divide(denominator.multiply(denominator).multiply(divisor)).finish!bits;
}

// 2*log(cosh(z/2)) = a - a^2/6 + O(a^3), where a = z^2/4.
// This branch is used only for z < 2^(-bits-64). Retain the side of
// the leading dyadic term even when the quadratic correction is remote.
private auto smallLogShape(uint work)(ProbabilityScaled!work z)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability: fpProbabilityComplement;
    alias S = ProbabilityScaled!work;
    alias W = Fp!work;
    auto a = z.multiply(z);
    a.exponent -= 2;
    const k = a.divide(S.from(W(6)));
    W factor;
    if (k.exponent + work <= exponentBias)
        factor = W(false, -cast(long)work, UInt!work.max);
    else
    {
        W fraction = k.fraction;
        fraction.exponent -= cast(long)cast(ulong)(exponentBias - k.exponent);
        factor = fpProbabilityComplement(fraction);
    }
    return a.multiply(S.from(factor));
}

package(mir.stat) Fp!bits logisticLogDensity(uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(true, long.max, UInt!bits(0));

    alias Log = ProbabilityLog!(4 * bits + 128);
    alias S = Log.S;
    enum work = 4 * bits + 320;
    Log.Context context;
    const centered = position(x, location) == 0;
    S z;
    if (!centered)
    {
        z = distance!work(x, location).divide(S.from(scale));
        // For z above this cutoff, 2*log(1+exp(-z)) is below the
        // working grid. Its sign can still resolve a midpoint at scale one.
        if (z.exponent > S.from(T(2 * work)).exponent)
        {
            auto peak = Log.logDyadic(scale.coefficient, scale.exponent, context);
            peak.sign = !peak.sign;
            if (!peak.coefficient)
            {
                peak = Fp!work(-1);
                peak.exponent -= 2 * work;
            }
            return addOffset!bits(peak, z, true);
        }
    }

    // Factor out the peak to preserve a small log density near scale 1/4.
    // Shift the coefficient so even the largest finite scale admits 4*scale.
    const fourScale = scale.coefficient.toSize!(bits + 64) << 2;
    auto peak = Log.logDyadic(fourScale, scale.exponent, context);
    peak.sign = peak.coefficient && !peak.sign;
    if (centered)
        return cast(T)peak;
    if (z.exponent + bits + 64 <= exponentBias)
        return addOffset!bits(peak, smallLogShape(z), true);

    // (1+t)^2/(4*t) = cosh(z/2)^2. Its exponent fits long here.
    const t = scaledDecay(z);
    const denominator = S.from(T(1)).add(t);
    auto shape = denominator.multiply(denominator).divide(t);
    const e = cast(long)cast(ulong)(shape.exponent - exponentBias) - work - 2;
    const correction = Log.logDyadic(shape.fraction.coefficient, e, context);
    return addOffset!bits(peak, S.from(correction), true);
}
