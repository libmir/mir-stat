/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_log_normal;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset;
import mir.stat.internal.fp_pi: guardedPi;
import mir.stat.internal.fp_scaled_probability: exponentBias, scaledDecay, scaledSqrt;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

// Keep log(x) at working precision across subtraction, including when
// mean is the input-width rounding of log(x) and the scale is tiny.
private auto standardized(uint bits)(Fp!bits x, Fp!bits mean, Fp!bits stdDev)
{
    alias B = ShapeTransform!bits;
    B.L.Context ctx;
    const logX = B.L.logDyadic(x.coefficient, x.exponent, ctx);
    auto offset = B.term(mean);
    offset.negative = !offset.negative;
    if (logX.coefficient)
        offset = B.combine(B.Term(B.S.from(logX), logX.sign, true), offset);
    if (offset.present)
        offset.value = offset.value.divide(B.S.from(stdDev));
    return offset;
}

package(mir.stat) Fp!bits logNormalDensity(bool logarithmic, uint bits)
    (Fp!bits x, Fp!bits mean, Fp!bits stdDev)
{
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (!x.coefficient || x.isInfinity)
        return logarithmic ? T(true, long.max, UInt!bits(0)) : T(0);

    const z = standardized(x, mean, stdDev);
    S q;
    if (z.present)
    {
        q = z.value.multiply(z.value);
        q.exponent -= 1;
        static if (!logarithmic)
        {
            // Even with x and stdDev at 2^long.min, q >= 2^65
            // puts the final density below half the least subnormal.
            if (q.exponent >= exponentBias + 66)
                return T(0);
        }
    }
    auto twicePi = S.from(guardedPi!(B.work)());
    twicePi.exponent += 1;
    const denominator = S.from(x).multiply(S.from(stdDev)).multiply(scaledSqrt(twicePi, false));
    static if (logarithmic)
    {
        auto peak = B.logarithm(denominator);
        if (peak.coefficient) peak.sign = !peak.sign;
        return z.present ? addOffset!bits(peak, q, true) : cast(T)peak;
    }
    else
    {
        const decay = z.present ? scaledDecay(q) : S.from(T(1));
        return addOffset!bits(T(0), decay.divide(denominator), false);
    }
}

package(mir.stat) Fp!bits logNormalTail(bool upper, uint bits)
    (Fp!bits x, Fp!bits mean, Fp!bits stdDev)
{
    import mir.stat.internal.fp_location_scale: half;
    import mir.stat.internal.fp_normal: standardNormalTail;
    alias T = Fp!bits;
    if (x.isInfinity) return T(upper ? 0 : 1);
    if (!x.coefficient) return T(upper ? 1 : 0);
    const z = standardized(x, mean, stdDev);
    if (!z.present) return half!bits();
    return standardNormalTail!bits(z.value, upper != z.negative);
}

package(mir.stat) Fp!bits logNormalQuantile(uint bits)(Fp!bits p, Fp!bits mean, Fp!bits stdDev)
{
    import mir.stat.internal.fp_location_scale: half, position;
    import mir.stat.internal.fp_normal: standardNormalQuantileMagnitude;
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (!p.coefficient) return T(0);
    const infinity = T(false, long.max, UInt!bits(0));
    if (position(p, T(1)) == 0) return infinity;

    auto logX = B.term(mean);
    const side = position(p, half!bits());
    if (side)
    {
        const z = standardNormalQuantileMagnitude!bits(p, side < 0);
        logX = B.combine(logX, B.Term(z.multiply(S.from(stdDev)), side < 0, true));
    }
    if (!logX.present) return T(1);
    // Beyond this magnitude the exponential is outside either end of
    // the Fp range, including subnormal values and the coefficient width.
    if (logX.value.exponent >= exponentBias + 65)
        return logX.negative ? T(0) : infinity;
    const decay = scaledDecay(logX.value);
    const value = logX.negative ? decay : S.from(T(1)).divide(decay);
    return addOffset!bits(T(0), value, false);
}
