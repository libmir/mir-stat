/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_pareto;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;
import mir.stat.internal.fp_probability_log: ProbabilityLog, logRatio;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, addExponent, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

private alias Log(uint bits) = ProbabilityLog!(3 * bits + 64);

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

package(mir.stat) Fp!bits paretoTail(bool upper, uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape)
{
    alias T = Fp!bits;
    alias S = Log!bits.S;
    if (x.isInfinity)
        return T(upper ? 0 : 1);
    if (!fpProbabilityLess(lower, x))
        return T(upper ? 1 : 0);
    if (S.from(shape) == S.from(T(1)))
    {
        static if (upper)
            return S.from(lower).divide(S.from(x)).finish!bits;
        else
            return distance!(3 * bits + 256)(x, lower).divide(S.from(x)).finish!bits;
    }
    Log!bits.Context context;
    const rate = S.from(logRatio!bits(x, lower, context)).multiply(S.from(shape));
    if (rate.exponent >= exponentBias + 66)
        return T(upper ? 0 : 1);
    static if (!upper)
        if (rate.exponent + bits + 64 <= exponentBias)
            return rate.finish!bits;
    auto tail = survival!bits(x, lower, shape, rate);
    static if (upper)
        return tail.finish!bits;
    else
    {
        if (tail.exponent + bits + 1 <= exponentBias)
            return T(1);
        tail.fraction.exponent += cast(long)cast(ulong)(tail.exponent - exponentBias);
        return cast(T)fpProbabilityComplement(tail.fraction);
    }
}

// Scratch has one extra limb so that a product whose bit length is ambiguous
// by one can be checked without instantiating a double-budget integer.
private bool multiplyWithin(uint capacity)(ref UInt!(capacity + 64) a, UInt!(capacity + 64) b)
{
    enum width = capacity + 64;
    if ((width - a.ctlz) + (width - b.ctlz) > capacity + 1)
        return false;
    a *= b;
    return a.ctlz >= 64;
}

private bool multiplyPower(uint capacity)(ref UInt!(capacity + 64) a, UInt!(capacity + 64) factor, uint n)
{
    while (n)
    {
        if ((n & 1) && !multiplyWithin!capacity(a, factor))
            return false;
        n >>= 1;
        if (n && !multiplyWithin!capacity(factor, factor))
            return false;
    }
    return true;
}

// For an integral shape, density = shape * lower^shape / x^(shape+1).
// The caller compares nearby values, so only the low word of the binary
// exponent is needed after subtracting the exponent of the comparison value.
private bool densityProducts(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape,
    out UInt!(8 * bits + 64) numerator, out UInt!(8 * bits + 64) denominator, out ulong exponent)
{
    enum capacity = 8 * bits;
    const ts = shape.coefficient.cttz;
    if (shape.exponent < -cast(long)ts)
        return false;
    const shapeShift = cast(ulong)shape.exponent + ts;
    const tx = x.coefficient.cttz;
    const tl = lower.coefficient.cttz;
    numerator = (shape.coefficient >> ts).toSize!(capacity + 64);
    const base = (lower.coefficient >> tl).toSize!(capacity + 64);
    denominator = (x.coefficient >> tx).toSize!(capacity + 64);
    // Two unit odd bases contribute only a binary exponent, even for
    // shapes too large to materialize as machine integers.
    const n = shapeShift < 64 ? cast(ulong)numerator << cast(uint)shapeShift : 0UL;
    if (base != 1 || denominator != 1)
    {
        if (shapeShift >= 32 || numerator > (cast(ulong)capacity >> cast(uint)shapeShift))
            return false;
        const count = cast(uint)n;
        if (!multiplyPower!capacity(numerator, base, count)
            || !multiplyPower!capacity(denominator, denominator, count))
            return false;
    }
    const xe = cast(ulong)x.exponent + tx;
    const le = cast(ulong)lower.exponent + tl;
    exponent = shapeShift - xe + n * (le - xe);
    return true;
}

// Compare the rational density to the nearest output midpoint only when the
// guarded result is within 2^(-bits-16) relative distance of that midpoint.
private bool roundDensityMidpoint(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape,
    Log!bits.S value, out Fp!bits result)
{
    enum work = 3 * bits + 256;
    alias S = Log!bits.S;
    auto minimum = exponentBias;
    addExponent(minimum, long.min);
    // Just below half the least quantum, the midpoint lies one binade above.
    if (value.exponent + 1 < minimum)
        return false;
    const shift = value.exponent < minimum + bits
        ? cast(uint)(minimum + work - value.exponent) : work - bits;
    const coefficient = value.fraction.coefficient.toSize!(work + 64);
    auto units = coefficient >> shift;
    const odd = (units << 1) + 1;
    const middle = odd << (shift - 1);
    const gap = coefficient >= middle ? coefficient - middle : middle - coefficient;
    if (gap > (coefficient >> (bits + 16)))
        return false;

    UInt!(8 * bits + 64) numerator, denominator;
    ulong exponent;
    if (!densityProducts!bits(x, lower, shape, numerator, denominator, exponent))
        return false;
    // The odd midpoint factor has at most bits+1 bits. Since both sides
    // are already close, aligning them costs at most one further bit.
    enum width = 9 * bits + 64;
    auto left = numerator.toSize!width;
    auto right = denominator.toSize!width;
    right *= odd.toSize!width;
    const grid = value.exponent - work + shift;
    const alignment = cast(long)(exponent - cast(ulong)(grid - exponentBias) + 1);
    if (alignment >= 0)
        left <<= cast(uint)alignment;
    else
        right <<= cast(uint)-alignment;
    if (left > right || (left == right && units.bt(0)))
        units += 1;
    if (!units)
        result = Fp!bits(0);
    else
    {
        auto rounded = S.from(Fp!(work + 64)(units));
        rounded.exponent += grid - exponentBias;
        result = addOffset!bits(Fp!bits(0), rounded, false);
    }
    return true;
}

package(mir.stat) Fp!bits paretoDensity(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape)
{
    alias T = Fp!bits;
    alias S = Log!bits.S;
    if (x.isInfinity)
        return T(0);
    const obs = S.from(x);
    auto value = S.from(shape).divide(obs);
    if (fpProbabilityLess(lower, x))
    {
        if (S.from(shape) == S.from(T(1)))
            value = S.from(lower).divide(obs.multiply(obs));
        else
        {
            Log!bits.Context context;
            const rate = S.from(logRatio!bits(x, lower, context)).multiply(S.from(shape));
            // Above 2^65, even the largest shape divided by the least x
            // cannot rescue the density from underflow.
            if (rate.exponent >= exponentBias + 66)
                return T(0);
            value = value.multiply(survival!bits(x, lower, shape, rate));
        }
    }
    T rounded;
    if (roundDensityMidpoint!bits(x, lower, shape, value, rounded))
        return rounded;
    return addOffset!bits(T(0), value, false);
}

package(mir.stat) Fp!bits paretoQuantile(uint bits)(Fp!bits p, Fp!bits lower, Fp!bits shape)
{
    alias T = Fp!bits;
    alias L = Log!bits;
    alias S = L.S;
    alias W = Fp!(3 * bits + 64);
    if (!p.coefficient)
        return lower;
    if (!fpProbabilityLess(p, T(1)))
        return T(false, long.max, UInt!bits(0));
    const probability = S.from(p);
    const tiny = probability.exponent + 3 * bits + 256 < exponentBias;
    if (S.from(shape) == S.from(T(1)))
    {
        if (tiny)
            return lower;
        const remaining = fpProbabilityComplement(cast(W)p);
        return addOffset!bits(T(0), S.from(lower).divide(S.from(remaining)), false);
    }
    // A tiny p can still give substantial growth after division by shape.
    const growth = (tiny ? probability : L.complement(cast(W)p)).divide(S.from(shape));
    if (growth.exponent >= exponentBias + 66)
        return T(false, long.max, UInt!bits(0));
    auto value = S.from(lower);
    // A dyadic complement below one can be a power of two only for p >= 1/2.
    if (probability.exponent >= exponentBias)
    {
        const remaining = S.from(fpProbabilityComplement(cast(W)p));
        if (remaining.fraction.coefficient == S.from(T(1)).fraction.coefficient)
        {
            const count = Fp!128(exponentBias + 1 - remaining.exponent);
            UInt!128 n;
            if (integralPower(S.from(count).divide(S.from(shape)), n))
            {
                value.exponent += n;
                return addOffset!bits(T(0), value, false);
            }
        }
    }
    return addOffset!bits(T(0), value.divide(scaledDecay(growth)), false);
}

// Near unit density take -log(1-difference/larger) to retain the sign.
private bool exactResidual(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape, out Fp!bits result)
{
    enum capacity = 8 * bits;
    alias L = Log!bits;
    alias S = L.S;
    UInt!(capacity + 64) numerator, denominator;
    ulong exponent;
    if (!densityProducts!bits(x, lower, shape, numerator, denominator, exponent))
        return false;
    const alignment = cast(long)exponent;
    // The caller has |log(density)| < 1/8. With both odd products below
    // 2^capacity, an alignment larger than capacity would put the density
    // outside [1/2, 2].
    if (alignment > 0)
        numerator <<= cast(uint)alignment;
    else if (alignment < 0)
        denominator <<= cast(uint)-alignment;
    if (numerator == denominator)
    {
        result = Fp!bits(0);
        return true;
    }
    const negative = numerator < denominator;
    const larger = negative ? denominator : numerator;
    const difference = negative ? denominator - numerator : numerator - denominator;
    alias W = Fp!(capacity + 64);
    const relative = S.from(W(difference)).divide(S.from(W(larger))).finish!(3 * bits + 64);
    result = L.complement(relative).finish!bits;
    result.sign = negative;
    return true;
}

package(mir.stat) Fp!bits paretoLogDensity(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits shape)
{
    alias T = Fp!bits;
    alias L = Log!bits;
    alias S = L.S;
    if (x.isInfinity)
        return T(true, long.max, UInt!bits(0));
    L.Context context;
    const prefactor = logRatio!bits(shape, x, context);
    if (!fpProbabilityLess(lower, x))
        return cast(T)prefactor;
    const rate = S.from(logRatio!bits(x, lower, context)).multiply(S.from(shape));
    const result = addOffset!bits(prefactor, rate, true);
    if (prefactor.coefficient && !prefactor.sign && !result.isSpecial)
    {
        const magnitude = result.coefficient ? S.from(result).exponent : UInt!128(0);
        if (magnitude < exponentBias - 2 && magnitude + bits <= rate.exponent)
        {
            T exact;
            if (exactResidual(x, lower, shape, exact))
                return exact;
        }
    }
    return result;
}

// The odd denominators straddle the eight-width limit by one bit.
// Both densities are close to one; the ninth power is the limiting product.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    enum c = 133432608295961803UL;
    const lower = T(false, -54, UInt!64(c - 2));
    T residual;
    assert(exactResidual(T(false, -54, UInt!64(c)), lower, T(8), residual));
    assert(!exactResidual(T(false, -54, UInt!64(c + 2)), lower, T(8), residual));
    // Unit density with a nonintegral shape, then near-unit densities
    // whose integer powers exceed the product budget at different stages.
    assert(!exactResidual(T(1.5), T(1.5), T(1.5), residual));
    const belowOne = T(false, -64, UInt!64.max);
    foreach (shape; [64, 513])
    {
        auto near = T(shape);
        near.coefficient -= 1;
        assert(!exactResidual(T(shape), near, T(shape), residual));
    }
    // At shape one and x=1 the log-density is exactly log(lower).
    assert(exactResidual(T(1), belowOne, T(1), residual));
    assert(residual == T(true, -127, UInt!64(1) << 63));
}
