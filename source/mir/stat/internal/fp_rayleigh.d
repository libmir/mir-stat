/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_rayleigh;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

// Square the input coefficients exactly before dividing. The separate
// exponents accommodate the squared ratio even at opposite exponent limits.
private auto halfSquaredRatio(uint work)(ProbabilityScaled!work x, ProbabilityScaled!work scaleSquared)
{
    auto q = x.multiply(x).divide(scaleSquared);
    q.exponent -= 1;
    return q;
}

package(mir.stat) Fp!bits rayleighUpper(uint bits)(Fp!bits x, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(0);
    if (!x.coefficient)
        return T(1);

    enum work = 5 * bits + 256;
    alias S = ProbabilityScaled!work;
    const divisor = S.from(scale);
    const q = halfSquaredRatio(S.from(x), divisor.multiply(divisor));
    if (q.exponent >= exponentBias + 66)
        return T(0);
    return scaledDecay(q).finish!bits;
}
