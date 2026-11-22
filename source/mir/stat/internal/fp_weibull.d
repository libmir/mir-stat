/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_weibull;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;
import mir.stat.internal.fp_location_scale: addOffset;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;
import mir.stat.internal.fp_probability_log: ProbabilityLog, logRatio;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

// Callers bound the absolute argument before range reduction. Keep the
// exponential scaled so subsequent products can recover an in-range result.
private S expMagnitude(S)(S argument, bool negative)
{
    const decay = scaledDecay(argument);
    return negative ? decay : S.from(Fp!64(1)).divide(decay);
}

// A negative omitted correction decides an exact rounding midpoint. A
// power of two is on the output grid or rounds to zero at the floor.
private auto belowMidpoint(uint work)(ProbabilityScaled!work value)
{
    if (value.fraction.coefficient.cttz != work - 1)
        value.fraction.coefficient -= 1;
    return value;
}

// Bound each integer power's odd significand to 2*bits+128 bits.
// The exponent stays separate even when the powered value is outside Fp.
private uint rationalShape(uint bits)(Fp!bits shape, Fp!bits x, Fp!bits scale)
{
    const zeros = shape.coefficient.cttz;
    if (shape.exponent < -cast(long)zeros)
        return 0;
    const shift = cast(ulong)shape.exponent + zeros;
    if (shift >= 32)
        return 0;
    const xBits = bits - x.coefficient.ctlz - x.coefficient.cttz;
    const scaleBits = bits - scale.coefficient.ctlz - scale.coefficient.cttz;
    const limit = (2 * bits + 128) / (xBits > scaleBits ? xBits : scaleBits);
    const odd = shape.coefficient >> zeros;
    if (odd > (limit >> cast(uint)shift))
        return 0;
    return cast(uint)odd << cast(uint)shift;
}

package(mir.stat) Fp!bits weibullTail(bool upper, uint bits)(Fp!bits x, Fp!bits shape, Fp!bits scale)
{
    alias T = Fp!bits;
    alias L = ProbabilityLog!(3 * bits + 64);
    alias S = L.S;

    if (x.isInfinity)
        return T(upper ? 0 : 1);
    if (!x.coefficient)
        return T(upper ? 1 : 0);

    const one = S.from(T(1));
    const k = S.from(shape);
    S hazard = one;
    const count = rationalShape(shape, x, scale);
    if (count)
    {
        const numerator = S.from(x).power(count);
        const denominator = S.from(scale).power(count);
        hazard = numerator.divide(denominator);
        static if (!upper)
        {
            // A rational hazard is separated from a noncoincident output
            // midpoint by more than 2^(-bits-denominatorBits-2) relatively.
            const denominatorBits = count * (bits - scale.coefficient.ctlz - scale.coefficient.cttz);
            if (hazard.exponent + (bits + denominatorBits + 8) <= exponentBias)
                return belowMidpoint(hazard).finish!bits;
        }
    }
    else
    {
        L.Context context;
        const logarithm = logRatio!bits(x, scale, context);
        if (logarithm.coefficient)
        {
            const magnitude = S.from(logarithm).multiply(k);
            // For log(hazard) >= 64 the upper tail is below half the
            // least quantum. A large negative log instead suppresses
            // the hazard itself; keep its wide exponent until then.
            const limit = logarithm.sign ? 66 : 7;
            if (magnitude.exponent >= exponentBias + limit)
                return T(logarithm.sign == upper ? 1 : 0);
            hazard = expMagnitude(magnitude, logarithm.sign);
        }
    }

    if (hazard.exponent >= exponentBias + 66)
        return T(upper ? 0 : 1);
    if (hazard.exponent + bits + 64 <= exponentBias)
    {
        static if (upper)
            return T(1);
        else
        {
            // 1-exp(-h) = h*(1-h/2+O(h^2)). Retain the first correction
            // without subtracting two probabilities close to one.
            enum work = 3 * bits + 256;
            auto halfHazard = hazard;
            halfHazard.exponent -= 1;
            const factor = addOffset!work(Fp!work(1), halfHazard, true);
            return hazard.multiply(S.from(factor)).finish!bits;
        }
    }
    auto survival = scaledDecay(hazard);
    static if (upper)
        return survival.finish!bits;
    else
    {
        if (survival.exponent + bits + 1 <= exponentBias)
            return T(1);
        survival.fraction.exponent += cast(long)cast(ulong)(survival.exponent - exponentBias);
        return cast(T)fpProbabilityComplement(survival.fraction);
    }
}

package(mir.stat) Fp!bits weibullDensity(uint bits)(Fp!bits x, Fp!bits shape, Fp!bits scale)
{
    alias T = Fp!bits;
    alias L = ProbabilityLog!(3 * bits + 64);
    alias S = L.S;
    const one = S.from(T(1));
    const k = S.from(shape);
    const divisor = S.from(scale);
    if (x.isInfinity)
        return T(0);
    if (!x.coefficient)
    {
        if (k == one)
            return addOffset!bits(T(0), one.divide(divisor), false);
        return fpProbabilityLess(shape, T(1)) ? T(false, long.max, UInt!bits(0)) : T(0);
    }

    const obs = S.from(x);
    S hazard, prefactor;
    const count = rationalShape(shape, x, scale);
    if (count)
    {
        // Form k*x^(k-1) and scale^k before dividing so an exact unit
        // prefactor survives in the log density, even for very small hazards.
        const previous = obs.power(count - 1);
        const denominator = divisor.power(count);
        hazard = previous.multiply(obs).divide(denominator);
        prefactor = previous.multiply(k).divide(denominator);
    }
    else
    {
        L.Context context;
        const logarithm = logRatio!bits(x, scale, context);
        hazard = one;
        if (logarithm.coefficient)
        {
            const magnitude = S.from(logarithm).multiply(k);
            // Neither shape/x nor its reciprocal can offset exponential
            // decay once |log(hazard)| reaches 2^65.
            if (magnitude.exponent >= exponentBias + 66)
                return T(0);
            hazard = expMagnitude(magnitude, logarithm.sign);
        }
        prefactor = k.divide(obs).multiply(hazard);
    }
    if (hazard.exponent >= exponentBias + 66)
        return T(0);
    // Below this threshold the decay can change the rounded rational
    // prefactor only at a midpoint, where the correction is negative.
    if (count && hazard.exponent + (3 * bits + 136) <= exponentBias)
        return addOffset!bits(T(0), belowMidpoint(prefactor), false);
    return addOffset!bits(T(0), prefactor.multiply(scaledDecay(hazard)), false);
}
