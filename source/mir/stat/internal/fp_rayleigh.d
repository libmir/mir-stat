/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_rayleigh;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityComplement;
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

// At a midpoint the first omitted correction to both the CDF and PDF is
// negative. Step below the leading term on the working grid before narrowing.
private auto predecessor(uint work)(ProbabilityScaled!work value)
{
    if (value.fraction.coefficient == (UInt!work(1) << (work - 1)))
    {
        value.fraction.coefficient = UInt!work.max;
        value.exponent -= 1;
    }
    else
        value.fraction.coefficient -= 1;
    return value;
}

package(mir.stat) Fp!bits rayleighTail(bool upper, uint bits)(Fp!bits x, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(upper ? 0 : 1);
    if (!x.coefficient)
        return T(upper ? 1 : 0);

    enum work = 5 * bits + 256;
    alias S = ProbabilityScaled!work;
    const divisor = S.from(scale);
    const q = halfSquaredRatio(S.from(x), divisor.multiply(divisor));
    if (q.exponent >= exponentBias + 66)
        return T(upper ? 0 : 1);
    static if (!upper)
    {
        // The denominator of q contains at most 2*bits coefficient bits.
        // Comparing with a bits-wide midpoint costs up to another bits+1.
        if (q.exponent + (3 * bits + 8) <= exponentBias)
            return predecessor(q).finish!bits;
    }
    auto decay = scaledDecay(q);
    static if (upper)
        return decay.finish!bits;
    else
    {
        if (decay.exponent + bits + 1 <= exponentBias)
            return T(1);
        decay.fraction.exponent -= cast(long)cast(ulong)(exponentBias - decay.exponent);
        return cast(T)fpProbabilityComplement(decay.fraction);
    }
}
