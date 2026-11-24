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

package(mir.stat) @safe pure nothrow @nogc
Fp!bits normalQuantile(uint bits)(Fp!bits p, Fp!bits mean, Fp!bits stdDev)
{
    import mir.stat.internal.fp_location_scale: half;
    import mir.stat.internal.fp_probability: fpProbabilityComplement;
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (!p.coefficient || position(p, T(1)) == 0)
        return T(!p.coefficient, long.max, UInt!bits(0));
    const median = half!bits();
    const side = position(p, median);
    if (!side) return mean;

    const tail = side < 0 ? p : fpProbabilityComplement(p);
    T quarter = median;
    quarter.exponent -= 1;
    const central = position(tail, quarter) >= 0;
    S target;
    if (central)
    {
        target = distance!(B.work)(p, median);
        target.exponent += 1; // erf(t) = 2 * abs(p - 1/2)
    }
    else
    {
        auto twiceTail = S.from(tail);
        twiceTail.exponent += 1;
        target = S.from(B.logarithm(twiceTail)); // magnitude of log(2*tail)
    }
    const t = inverseMagnitude!bits(target, central);
    const z = t.multiply(scaledSqrt(S.from(T(2)), false));
    return addOffset!bits(mean, z.multiply(S.from(stdDev)), side < 0);
}

// Solve for t = abs(z)/sqrt(2). In the outer tail use
// -log(erfc(t)) = t*t - log(erfcx(t)), even at the Fp exponent floor.
private @safe pure nothrow @nogc
auto inverseMagnitude(uint bits)(ShapeTransform!bits.S target, bool central)
{
    import mir.stat.internal.fp_erf: erfPositive, erfcxPositive;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    alias W = Fp!(B.work);
    auto halfSqrtPi = scaledSqrt(S.from(guardedPi!(B.work)()), false);
    halfSqrtPi.exponent -= 1;
    W low = W(0);
    W high = central ? W(1) : workingValue(scaledSqrt(target, false));
    W t = central ? workingValue(target.multiply(halfSqrtPi)) : high;
    bool refining = true;
    for (size_t iteration = 0; refining; ++iteration)
    {
        const value = S.from(t);
        const square = value.multiply(value);
        S model, inverseSlope;
        if (central)
        {
            model = S.from(erfPositive(t));
            inverseSlope = halfSqrtPi.divide(scaledDecay(square));
        }
        else
        {
            const erfcx = S.from(erfcxPositive(t));
            model = square.add(S.from(B.logarithm(erfcx)));
            inverseSlope = halfSqrtPi.multiply(erfcx);
        }
        const residual = B.combine(B.Term(model, false, true), B.Term(target, true, true));
        if (!residual.present) break;
        const step = residual.value.multiply(inverseSlope);
        // Leave room for the large logarithm in the outer-tail residual.
        // The retained precision also guards subsequent location cancellation.
        if (step.exponent + B.work - 72 < value.exponent)
            break;
        if (residual.negative) low = t;
        else high = t;
        refining = advanceQuantile(t, low, high, step, !residual.negative, iteration < 16);
    }
    return S.from(t);
}

// Prefer an interior Newton step while the iteration budget permits.
// Otherwise bisect; an endpoint midpoint exhausts the working grid.
private @safe pure nothrow @nogc
bool advanceQuantile(uint work)(ref Fp!work t, Fp!work low, Fp!work high,
    ProbabilityScaled!work step, bool downward, bool tryNewton)
{
    if (tryNewton)
    {
        const candidate = addOffset!work(t, step, downward);
        if (position(candidate, low) > 0 && position(candidate, high) < 0)
        {
            t = candidate;
            return true;
        }
    }
    auto halfGap = distance!work(high, low);
    halfGap.exponent -= 1;
    t = addOffset!work(low, halfGap, false);
    return position(t, low) != 0 && position(t, high) != 0;
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias W = Fp!64;
    alias S = ProbabilityScaled!64;
    const step = S.from(W(1));
    W t = W(4);
    // A Newton step outside [4, 8] is replaced by its midpoint.
    assert(advanceQuantile(t, W(4), W(8), step, true, true));
    assert(t == W(6));
    // Once the budget is spent, bisect even when the Newton step is interior.
    assert(advanceQuantile(t, W(2), W(6), step, true, false));
    assert(t == W(4));
    // Adjacent working values round their midpoint to the even endpoint.
    W a = W(1), b = a, c = a;
    b.coefficient += 1;
    c.coefficient += 2;
    t = b;
    assert(!advanceQuantile(t, a, b, step, true, false));
    assert(t == a);
    t = b;
    assert(!advanceQuantile(t, b, c, step, true, false));
    assert(t == c);
}
