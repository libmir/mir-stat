/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_rayleigh;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
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

package(mir.stat) Fp!bits rayleighDensity(uint bits)(Fp!bits x, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity || !x.coefficient)
        return T(0);
    alias S = ProbabilityScaled!(5 * bits + 256);
    const obs = S.from(x);
    const divisor = S.from(scale);
    const square = divisor.multiply(divisor);
    const q = halfSquaredRatio(obs, square);
    // Above 2^65 even the smallest scale cannot rescue z*exp(-z*z/2).
    if (q.exponent >= exponentBias + 66)
        return T(0);
    const prefactor = obs.divide(square);
    const density = q.exponent + (3 * bits + 8) <= exponentBias
        ? predecessor(prefactor) : prefactor.multiply(scaledDecay(q));
    // The peak is exp(-1/2)/scale, below the Fp ceiling even at the least scale.
    return density.finish!bits;
}

package(mir.stat) Fp!bits rayleighLogDensity(uint bits)(Fp!bits x, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity || !x.coefficient)
        return T(true, long.max, UInt!bits(0));
    enum logBits = 3 * bits + 64;
    enum work = logBits + 192;
    alias Log = ProbabilityLog!logBits;
    alias S = Log.S;
    const obs = S.from(x);
    const divisor = S.from(scale);
    const square = divisor.multiply(divisor);
    const ratio = obs.divide(square);
    Log.Context context;
    Fp!work logRatio;
    if (ratio.exponent >= exponentBias - 1 && ratio.exponent <= exponentBias + 1)
    {
        // Near x == scale^2 take a single log, including exact log(1).
        const e = cast(long)cast(ulong)(ratio.exponent - exponentBias) - work;
        logRatio = Log.logDyadic(ratio.fraction.coefficient, e, context);
    }
    else
    {
        // The exponent of x/scale^2 need not fit in a signed long.
        logRatio = Log.logDyadic(x.coefficient, x.exponent, context);
        const logScale = Log.logDyadic(scale.coefficient, scale.exponent, context);
        if (logScale.coefficient)
        {
            auto term = S.from(logScale);
            term.exponent += 1;
            logRatio = addOffset!work(logRatio, term, !logScale.sign);
        }
    }
    return addOffset!bits(logRatio, halfSquaredRatio(obs, square), true);
}

package(mir.stat) Fp!bits rayleighQuantile(uint bits)(Fp!bits p, Fp!bits scale)
{
    import mir.stat.internal.uint_sqrt: isqrtRem;
    alias T = Fp!bits;
    if (!p.coefficient)
        return T(0);
    if (!fpProbabilityLess(p, T(1)))
        return T(false, long.max, UInt!bits(0));

    enum logBits = 3 * bits + 64;
    enum work = logBits + 192;
    alias Log = ProbabilityLog!logBits;
    alias S = Log.S;
    const tiny = S.from(p).exponent + (3 * bits + 8) <= exponentBias;
    // For tiny p the leading squared quantile 2*p*scale^2 is an exact
    // dyadic with at most 3*bits significant bits. The positive log
    // correction can decide a midpoint. Avoid widening p at its floor.
    const logarithm = tiny ? S.from(p) : Log.complement(cast(Fp!logBits)p);
    const divisor = S.from(scale);
    auto squared = logarithm.multiply(divisor.multiply(divisor));
    squared.exponent += 1;

    // Make the binary exponent even and extract a work-bit root. A sticky
    // low bit preserves the side of a midpoint through the final narrowing.
    const odd = squared.exponent.bt(0);
    alias I = UInt!(2 * work);
    const radicand = squared.fraction.coefficient.toSize!(2 * work) << (work - odd);
    I remainder;
    auto coefficient = isqrtRem(radicand, remainder).toSize!work;
    if (remainder || tiny)
        coefficient |= 1u;
    S result;
    result.fraction = Fp!work(false, -cast(long)work, coefficient);
    result.exponent = (squared.exponent >> 1) + (exponentBias >> 1);
    if (odd)
        result.exponent += 1;
    return addOffset!bits(T(0), result, false);
}
