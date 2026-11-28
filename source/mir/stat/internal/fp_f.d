/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_f;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_beta: BetaIntegral;
import mir.stat.internal.fp_location_scale: addOffset, position;
import mir.stat.internal.fp_scaled_probability: exponentBias;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

private struct FTransform(uint bits)
{
    alias B = ShapeTransform!bits;
    alias S = B.S;
    alias T = Fp!bits;
    S a, b, one;

    this(T df1, T df2)
    {
        a = S.from(df1);
        b = S.from(df2);
        --a.exponent;
        --b.exponent;
        one = S.from(T(1));
    }

    // Inverting a unit beta shape gives exp(t)-1 in the numerator
    // or denominator. Keep t and the degree factor on separate exponents.
    T unitQuantile(T p, bool numeratorUnit) const
    {
        import mir.stat.internal.fp_probability_log: ProbabilityLog;
        import mir.stat.internal.fp_scaled_probability: scaledDecay;
        import mir.stat.internal.fp_location_scale: distance;
        const probability = S.from(p);
        if (a == one && b == one)
        {
            const complement = distance!(B.work)(T(1), p);
            return addOffset!bits(T(0), probability.divide(complement), false);
        }
        const logTarget = numeratorUnit ? ProbabilityLog!bits.complement(p) : ProbabilityLog!bits.probability(p);
        auto magnitude = S.from(logTarget.fraction);
        magnitude.exponent += logTarget.exponent - exponentBias;
        const shape = numeratorUnit ? b : a;
        const rate = magnitude.divide(shape);
        if (rate.exponent >= exponentBias + 66)
            return numeratorUnit ? T(false, long.max, UInt!bits(0)) : T(0);
        S growth;
        if (rate.exponent + B.work / 2 < exponentBias)
            growth = rate;
        else
        {
            const decay = scaledDecay(rate);
            const difference = B.combine(B.Term(one, false, true), B.Term(decay, true, true));
            growth = difference.value.divide(decay);
        }
        const product = shape.multiply(growth);
        return addOffset!bits(T(0), numeratorUnit ? product : one.divide(product), false);
    }

    alias Point = BetaIntegral!bits.Point;

    // Divide the smaller weight by the larger before adding one. Both
    // coordinates and both negative logarithms retain their own exponents.
    Point coordinates(S x) const
    {
        const ax = a.multiply(x);
        const reflected = ax.exponent > b.exponent ||
            (ax.exponent == b.exponent && ax.fraction.coefficient > b.fraction.coefficient);
        const ratio = reflected ? b.divide(ax) : ax.divide(b);
        const sum = one.add(ratio);
        const small = ratio.divide(sum);
        const large = one.divide(sum);
        const logSmall = S.from(B.logarithm(small));
        // log(1+r) = r*(1+O(r)). Use the linear term below the
        // threshold where forming 1+r would lose half the working bits.
        const logLarge = ratio.exponent + B.work / 2 < exponentBias
            ? ratio : S.from(B.logarithm(sum));
        return reflected ? Point(large, small, logLarge, logSmall)
                         : Point(small, large, logSmall, logLarge);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    alias F = FTransform!64;
    alias S = F.S;
    alias U = UInt!64;
    const transform = F(T(2), T(2));
    const huge = T(false, 2000, U(1));
    const tiny = T(false, -2000, U(1));
    const half = T(0.5);
    assert(transform.unitQuantile(half, true) == T(1));
    assert(F(T(2), T(4)).unitQuantile(T(0.75), true) == T(2));
    assert(F(T(4), T(2)).unitQuantile(T(0.25), false) == half);
    const floor = T(false, long.min, U(1));
    assert(F(T(2), T(4)).unitQuantile(floor, true) == floor);
    assert(F(T(2), floor).unitQuantile(half, true).isInfinity);
    assert(F(floor, T(2)).unitQuantile(half, false) == T(0));
    const right = transform.coordinates(S.from(huge));
    const left = transform.coordinates(S.from(tiny));
    assert(right.x == left.y && right.y == left.x);
    assert(right.negLogX == left.negLogY && right.negLogY == left.negLogX);
    assert(right.y.finish!64 == S.from(tiny).finish!64);
    assert(right.negLogX.finish!64 == right.y.finish!64);
}

package(mir.stat) Fp!bits fTail(bool upper, uint bits)(Fp!bits x, Fp!bits df1, Fp!bits df2)
{
    alias T = Fp!bits;
    alias I = BetaIntegral!bits;
    if (x.isInfinity) return T(upper ? 0 : 1);
    if (!x.coefficient) return T(upper ? 1 : 0);
    const f = FTransform!bits(df1, df2);
    const obs = I.S.from(x);
    if (f.a == f.b && obs == f.one)
    {
        import mir.stat.internal.fp_location_scale: half;
        return half!bits();
    }
    // At x=1 the transformed point is exactly the beta mean. With
    // extremely concentrated unequal shapes, rounding that point can
    // move it many standard deviations before integration begins.
    if (obs == f.one && f.a.exponent > exponentBias + I.B.work &&
        f.b.exponent > exponentBias + I.B.work)
        return T(false, long.max, UInt!bits(1));
    const p = f.coordinates(obs);
    auto integral = I(f.one, 4096);
    if (f.a == f.one && f.b == f.one)
    {
        I.S tail = upper ? p.y : p.x;
        // The denominator 1+r is strictly greater than one even when
        // r is below the working grid. Keep that side of a midpoint.
        if (tail.exponent + I.B.work < exponentBias)
            tail = I.B.withCorrection(tail, false);
        return tail.finish!bits;
    }
    if (f.a == f.one || f.b == f.one)
    {
        const reflected = f.a == f.one;
        const shape = reflected ? f.b : f.a;
        if (shape == I.S.from(T(2)))
        {
            const base = reflected ? p.y : p.x;
            const other = reflected ? p.x : p.y;
            if (upper != reflected)
                return other.multiply(f.one.add(base)).finish!bits;
            // When the odds are tiny, base^2 = odds^2/(1+odds)^2
            // lies strictly below odds^2. Keep that side of a midpoint.
            const ax = f.a.multiply(obs);
            const weight = reflected ? f.b : ax;
            const opposite = reflected ? ax : f.b;
            const odds = weight.divide(opposite);
            if (odds.exponent + 3 * bits + 8 < exponentBias)
                return I.B.withCorrection(odds.multiply(odds), false).finish!bits;
            const z = weight.divide(weight.add(opposite));
            return z.multiply(z).finish!bits;
        }
        const rate = reflected ? f.b.multiply(p.negLogY) : f.a.multiply(p.negLogX);
        if (upper != reflected) return integral.oneMinusDecay(rate).finish!bits;
        const tail = integral.fromLog(I.neg(rate));
        return tail.present ? tail.value.finish!bits : T(0);
    }
    I.Term tail;
    bool reflected;
    if (!integral.directTail(f.a, f.b, p, reflected, tail))
        return T(false, long.max, UInt!bits(1));
    const complement = upper != reflected;
    return tail.present ? addOffset!bits(T(complement ? 1 : 0), tail.value, complement) : T(complement ? 1 : 0);
}
