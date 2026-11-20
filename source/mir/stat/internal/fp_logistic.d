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
