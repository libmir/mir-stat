/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_gev;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_scaled_probability: exponentBias, scaledDecay;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

package(mir.stat) bool gevSupported(uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    if (!xi.coefficient)
        return true;
    if (x.isNaN)
        return false;
    if (x.isInfinity)
        return x.sign == xi.sign;
    return !ShapeTransform!bits.residual(x, mu, sigma, xi).negative;
}

private struct Evaluation(uint bits)
{
    alias B = ShapeTransform!bits;
    alias T = B.T;
    alias S = B.S;
    alias Term = B.Term;

    // h = log(1+xi*z)/xi has the sign of x-mu, including the Gumbel limit.
    // The GEV CDF is exp(-a), where a = exp(-h). A missing magnitude is
    // zero; infiniteA means that a exceeds even the output exponent range.
    struct Transform
    {
        Term h;
        Term a;
        bool infiniteA;
    }

    static Transform transform(T x, T mu, T sigma, T xi, Term numerator)
    {
        const side = position(x, mu);
        const one = S.from(T(1));
        if (!side)
            return Transform(Term.init, Term(one, false, true));
        const h = Term(B.log1pScaledMagnitude(x, mu, sigma, xi, numerator), side < 0, true);
        if (!position(xi, T(-1)))
            return Transform(h, Term(numerator.value.divide(S.from(sigma)), false, true));
        if (!position(xi, T(1)))
            return Transform(h, Term(S.from(sigma).divide(numerator.value), false, true));
        // Beyond this bound exp(|h|) exceeds the Fp exponent range even
        // after division by any finite scale or multiplication by |xi|.
        if (h.value.exponent >= exponentBias + 66)
            return Transform(h, Term.init, h.negative);
        const decay = scaledDecay(h.value);
        return Transform(h, Term(h.negative ? one.divide(decay) : decay, false, true));
    }

    // An output midpoint has at most bits+1 significant bits, including
    // at the subnormal floor. This filter also admits some exact outputs;
    // one working low bit cannot change their rounded value.
    static bool mayBeMidpoint(S magnitude)
    {
        return magnitude.fraction.coefficient.cttz >= B.work - bits - 1;
    }

    // For shape one, write x-mu = d-l when x and mu have the same sign.
    // If m*d = sigma exactly, a = m/(1 + m*(1-l/sigma)).
    // The tail exceeds m iff l/sigma > 1 + 1/m - 1/(-log(1-m))
    // = 3/2 + m/12 + O(m^2). For 0<m<1/2 the excess over 3/2 is
    // positive and less than m (by the log series). In the tiny-tail
    // window m < 2^(-bits-2); distinct represented l and 3*sigma/2
    // near equality differ by at least sigma*2^(-bits-2).
    static bool aboveUnitShapeMidpoint(S m, T x, T mu, T sigma)
    {
        if (!mayBeMidpoint(m) || x.sign != mu.sign || !x.coefficient || !mu.coefficient)
            return false;
        const d = S.from(x.sign ? mu : x);
        const scale = S.from(sigma);
        if (m.multiply(d) != scale)
            return false;
        auto threshold = scale.multiply(S.from(T(3)));
        threshold.exponent -= 1;
        const small = S.from(x.sign ? x : mu);
        return small.exponent > threshold.exponent ||
            (small.exponent == threshold.exponent &&
             small.fraction.coefficient > threshold.fraction.coefficient);
    }

    static T tail(bool upper)(T x, T mu, T sigma, T xi)
    {
        if (x.isNaN)
            return x;
        if (x.isInfinity)
            return T(x.sign == upper ? 1 : 0);
        const numerator = xi.coefficient ? B.residual(x, mu, sigma, xi) : B.term(sigma);
        if (!numerator.present)
            return T(xi.sign == upper ? 0 : 1);
        const v = transform(x, mu, sigma, xi, numerator);
        if (v.infiniteA || (v.a.present && v.a.value.exponent >= exponentBias + 66))
            return T(upper ? 1 : 0);
        if (!v.a.present)
            return T(upper ? 0 : 1);
        if (v.a.value.exponent + B.work / 2 <= exponentBias)
        {
            static if (upper)
            {
                const above = !position(xi, T(1)) &&
                    aboveUnitShapeMidpoint(v.a.value, x, mu, sigma);
                return B.withCorrection(v.a.value, above).finish!bits;
            }
            else
                return T(1);
        }
        const lower = scaledDecay(v.a.value);
        static if (upper)
            return addOffset!bits(T(1), lower, true);
        else
            return lower.finish!bits;
    }

}

package(mir.stat) auto gevTail(bool upper, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.tail!upper(x, mu, sigma, xi);
}
