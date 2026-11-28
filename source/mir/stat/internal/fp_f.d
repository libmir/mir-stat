/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_f;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
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
    struct Coordinates
    {
        S lower, upper, negLogLower, negLogUpper;
    }

    // Divide the smaller weight by the larger before adding one. Both
    // coordinates and both negative logarithms retain their own exponents.
    Coordinates coordinates(S x) const
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
        return reflected ? Coordinates(large, small, logLarge, logSmall)
                         : Coordinates(small, large, logSmall, logLarge);
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
    assert(right.lower == left.upper && right.upper == left.lower);
    assert(right.negLogLower == left.negLogUpper && right.negLogUpper == left.negLogLower);
    assert(right.upper.finish!64 == S.from(tiny).finish!64);
    assert(right.negLogLower.finish!64 == right.upper.finish!64);
}
