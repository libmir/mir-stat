/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_pareto;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset;
import mir.stat.internal.fp_probability: fpProbabilityLess;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

private alias Log(uint bits) = ProbabilityLog!(3 * bits + 64);

// Use one logarithm near equality. Elsewhere subtract the input logarithms,
// since the exponent of their quotient need not fit in a signed long.
private auto logRatio(uint bits)(Fp!bits a, Fp!bits b, ref Log!bits.Context context)
{
    alias L = Log!bits;
    alias S = L.S;
    enum work = 3 * bits + 256;
    const ratio = S.from(a).divide(S.from(b));
    if (ratio.exponent >= exponentBias - 1 && ratio.exponent <= exponentBias + 2)
    {
        const e = cast(long)cast(ulong)(ratio.exponent - exponentBias) - work;
        return L.logDyadic(ratio.fraction.coefficient, e, context);
    }
    const first = L.logDyadic(a.coefficient, a.exponent, context);
    const second = L.logDyadic(b.coefficient, b.exponent, context);
    return second.coefficient ? addOffset!work(first, S.from(second), !second.sign) : first;
}

// Recognize an integral binary exponent after range reduction. Callers supply
// values below 2^67. For division, a nonintegral quotient of the input dyadics
// here differs from an integer by at least 2^(-bits-67).
private bool integralPower(uint work)(ProbabilityScaled!work power, out UInt!128 n)
{
    if (power.exponent <= exponentBias || power.exponent > exponentBias + 67)
        return false;
    const shift = work - cast(uint)(power.exponent - exponentBias);
    if (power.fraction.coefficient.cttz < shift)
        return false;
    n = (power.fraction.coefficient >> shift).toSize!128;
    return true;
}

private auto survival(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape, Log!bits.S rate)
{
    alias S = Log!bits.S;
    const a = S.from(x);
    const b = S.from(lower);
    if (a.fraction.coefficient == b.fraction.coefficient)
    {
        const count = Fp!128(a.exponent - b.exponent);
        UInt!128 n;
        if (integralPower(S.from(count).multiply(S.from(shape)), n))
        {
            auto result = S.from(Fp!bits(1));
            result.exponent -= n;
            return result;
        }
    }
    return scaledDecay(rate);
}

package(mir.stat) Fp!bits paretoUpper(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape)
{
    alias T = Fp!bits;
    alias S = Log!bits.S;
    if (x.isInfinity)
        return T(0);
    if (!fpProbabilityLess(lower, x))
        return T(1);
    if (S.from(shape) == S.from(T(1)))
        return S.from(lower).divide(S.from(x)).finish!bits;
    Log!bits.Context context;
    const rate = S.from(logRatio!bits(x, lower, context)).multiply(S.from(shape));
    if (rate.exponent >= exponentBias + 66)
        return T(0);
    return survival!bits(x, lower, shape, rate).finish!bits;
}
