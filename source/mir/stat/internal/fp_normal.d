/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_normal;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_pi: guardedPi;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay, scaledSqrt;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

package(mir.stat) @safe pure nothrow @nogc
Fp!bits normalDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits mean, Fp!bits stdDev)
{
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return logarithmic ? T(true, long.max, UInt!bits(0)) : T(0);

    const scale = S.from(stdDev);
    const centered = position(x, mean) == 0;
    S q;
    if (!centered)
    {
        const z = distance!(B.work)(x, mean).divide(scale);
        q = z.multiply(z);
        q.exponent -= 1;
        static if (!logarithmic)
        {
            // At q >= 2^65, exp(-q) / (sqrt(2*pi) * 2^long.min)
            // is below half the least subnormal even at the smallest scale.
            if (q.exponent >= exponentBias + 66)
                return T(0);
        }
    }

    auto twicePi = S.from(guardedPi!(B.work)());
    twicePi.exponent += 1;
    const denominator = scale.multiply(scaledSqrt(twicePi, false));
    static if (logarithmic)
    {
        // A single logarithm preserves cancellation when the peak is near one.
        auto peakLog = B.logarithm(denominator);
        if (peakLog.coefficient)
            peakLog.sign = !peakLog.sign;
        return centered ? cast(T)peakLog : addOffset!bits(peakLog, q, true);
    }
    else
    {
        // Keep the decay scaled until division: a small stdDev can rescue
        // a density even when the standard density would round to zero.
        const decay = centered ? S.from(T(1)) : scaledDecay(q);
        return decay.divide(denominator).finish!bits;
    }
}

package(mir.stat) @safe pure nothrow @nogc
Fp!bits normalTail(bool upper, uint bits)(Fp!bits x, Fp!bits mean, Fp!bits stdDev)
{
    import mir.stat.internal.fp_erf: erfPositive, erfcxPositive;
    import mir.stat.internal.fp_location_scale: half;
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (x.isNaN) return x;
    if (x.isInfinity) return T(x.sign == upper ? 1 : 0);
    const side = position(x, mean);
    if (!side) return half!bits();
    const small = upper == (side > 0);
    const z = distance!(B.work)(x, mean).divide(S.from(stdDev));
    if (z.exponent + bits + 3 < exponentBias)
        return half!bits();
    auto q = z.multiply(z);
    q.exponent -= 1;
    if (q.exponent >= exponentBias + 65)
        return T(small ? 0 : 1);
    const t = workingValue(scaledSqrt(q, false));
    if (z.exponent <= exponentBias)
    {
        auto displacement = S.from(erfPositive(t));
        displacement.exponent -= 1;
        return addOffset!bits(half!bits(), displacement, small);
    }
    auto tail = scaledDecay(q).multiply(S.from(erfcxPositive(t)));
    tail.exponent -= 1;
    return small ? tail.finish!bits : addOffset!bits(T(1), tail, true);
}

// Callers bound the unbiased exponent before materializing the working value.
private @safe pure nothrow @nogc
Fp!work workingValue(uint work)(ProbabilityScaled!work value)
{
    auto result = value.fraction;
    result.exponent += cast(long)cast(ulong)(value.exponent - exponentBias);
    return result;
}
