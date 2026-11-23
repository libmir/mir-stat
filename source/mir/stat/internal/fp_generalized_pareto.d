/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_generalized_pareto;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_shape_transform: ShapeTransform;
import mir.stat.internal.fp_scaled_probability: exponentBias, scaledDecay;

@safe pure nothrow @nogc:

private struct Evaluation(uint bits)
{
    alias B = ShapeTransform!bits;
    alias T = B.T;
    alias S = B.S;
    alias L = B.L;
    alias Term = B.Term;
    enum work = B.work;

    static T tail(bool upper)(T x, T mu, T sigma, T xi)
    {
        if (x.isInfinity) return T(upper ? 0 : 1);
        if (!position(x, mu)) return T(upper ? 1 : 0);
        auto numerator = xi.coefficient ? B.residual(x, mu, sigma, xi) : B.term(sigma);
        if (!numerator.present) return T(upper ? 0 : 1);
        if (position(xi, T(-1)) == 0)
        {
            const part = upper ? numerator.value : distance!work(x, mu);
            return part.divide(S.from(sigma)).finish!bits;
        }
        const h = B.log1pScaledMagnitude(x, mu, sigma, xi, numerator);
        if (h.exponent >= exponentBias + 66) return T(upper ? 0 : 1);
        if (h.exponent + work / 2 <= exponentBias)
        {
            static if (upper) return T(1);
            else
            {
                // At zero location z is a single dyadic ratio. Its odd
                // denominator has at most bits bits. In this window a
                // non-midpoint stays on its side of the rounding boundary.
                // At a midpoint F(z) - z has the sign of -(1+xi).
                if (!mu.coefficient && h.exponent + (2 * bits + 64) <= exponentBias &&
                    (!xi.coefficient || S.from(xi).multiply(h).exponent + (2 * bits + 64) <= exponentBias))
                    return B.withCorrection(h, position(xi, T(-1)) < 0).finish!bits;
                return h.finish!bits;
            }
        }
        const survival = scaledDecay(h);
        static if (upper) return survival.finish!bits;
        else return addOffset!bits(T(1), survival, true);
    }

    // The density is exp(-(1+xi)*h)/sigma. The sign changes at xi=-1;
    // construct the distance from -1 before multiplication to retain it.
    static T density(bool logarithmic)(T x, T mu, T sigma, T xi)
    {
        const minusInfinity = T(true, long.max, UInt!bits(0));
        if (x.isInfinity) return logarithmic ? minusInfinity : T(0);
        const shapeOrder = position(xi, T(-1));
        const scale = S.from(sigma);
        const one = S.from(T(1));
        static if (logarithmic)
        {
            // Use the original exponent directly, including at the Fp floor.
            L.Context context;
            auto peak = L.logDyadic(sigma.coefficient, sigma.exponent, context);
            if (peak.coefficient) peak.sign = !peak.sign;
        }
        if (!shapeOrder || !position(x, mu))
        {
            static if (logarithmic) return cast(T)peak;
            else return addOffset!bits(T(0), one.divide(scale), false);
        }
        const numerator = xi.coefficient ? B.residual(x, mu, sigma, xi) : B.term(sigma);
        if (!numerator.present)
            return shapeOrder < 0 ? T(false, long.max, UInt!bits(0))
                : logarithmic ? minusInfinity : T(0);
        const h = B.log1pScaledMagnitude(x, mu, sigma, xi, numerator);
        const magnitude = h.multiply(distance!work(xi, T(-1)));
        static if (logarithmic)
            return addOffset!bits(peak, magnitude, shapeOrder > 0);
        else
        {
            if (magnitude.exponent >= exponentBias + 66)
                return shapeOrder > 0 ? T(0) : T(false, long.max, UInt!bits(0));
            if (magnitude.exponent + (2 * bits + 64) <= exponentBias)
                return addOffset!bits(T(0), B.withCorrection(one.divide(scale), shapeOrder < 0), false);
            const decay = scaledDecay(magnitude);
            const value = shapeOrder > 0 ? decay.divide(scale) : one.divide(decay.multiply(scale));
            return addOffset!bits(T(0), value, false);
        }
    }

    // For negative shapes the upper endpoint can cancel against mu.
    // Combine mu*|xi| + sigma before subtracting the residual tail;
    // a zero endpoint then leaves the entire residual available.
    static T fromEndpoint(T mu, S scale, S k, Term remainder)
    {
        auto base = B.term(mu);
        if (base.present) base.value = base.value.multiply(k);
        auto numerator = B.combine(base, Term(scale, false, true));
        if (remainder.present)
            numerator = B.combine(numerator, Term(scale.multiply(remainder.value), true, true));
        return B.finish(numerator, k);
    }

    static T quantile(T p, T mu, T sigma, T xi)
    {
        if (!p.coefficient) return mu;
        const scale = S.from(sigma);
        const one = S.from(T(1));
        if (!position(p, T(1)))
        {
            if (xi.sign && xi.coefficient)
                return fromEndpoint(mu, scale, S.from(xi), Term.init);
            return T(false, long.max, UInt!bits(0));
        }
        const prob = S.from(p);
        const linear = scale.multiply(prob);
        if (!position(xi, T(-1)))
            return addOffset!bits(mu, linear, false);
        const k = xi.coefficient ? S.from(xi) : one;
        if (prob.exponent + work / 2 <= exponentBias &&
            k.multiply(prob).exponent + work / 2 <= exponentBias)
        {
            // Q = mu + sigma*p + sigma*p^2*(1+xi)/2 + ...
            // Keep the second term when the location cancels the first.
            const base = B.combine(B.term(mu), Term(linear, false, true));
            auto correction = linear.multiply(prob).multiply(distance!work(xi, T(-1)));
            correction.exponent -= 1;
            const negative = position(xi, T(-1)) < 0;
            if (base.present && base.value.exponent > correction.exponent + work)
                return B.finish(Term(B.withCorrection(base.value, negative == base.negative), base.negative, true), one);
            return B.finish(B.combine(base, Term(correction, negative, true)), one);
        }
        // At shape one the offset is the rational sigma*p/(1-p).
        // Keep the location in its numerator through the final division.
        if (!position(xi, T(1)))
        {
            const q = B.combine(Term(one, false, true), Term(prob, true, true)).value;
            auto base = B.term(mu);
            if (base.present) base.value = base.value.multiply(q);
            return B.finish(B.combine(base, Term(linear, false, true)), q);
        }
        // Widening an input at the exponent floor can discard it. When p
        // is tiny, the relative correction to the complement log is O(p).
        const h = prob.exponent + work / 2 <= exponentBias ? prob
            : L.complement(cast(Fp!(3 * bits + 64))p);
        if (!xi.coefficient)
            return addOffset!bits(mu, scale.multiply(h), false);
        const power = h.multiply(k);
        if (power.exponent + work / 2 <= exponentBias)
            return addOffset!bits(mu, scale.multiply(h), false);
        if (power.exponent >= exponentBias + 66)
        {
            if (xi.sign) return fromEndpoint(mu, scale, k, Term.init);
            return T(false, long.max, UInt!bits(0));
        }
        const decay = scaledDecay(power);
        if (xi.sign && power.exponent >= exponentBias + 1)
            return fromEndpoint(mu, scale, k, Term(decay, false, true));
        const difference = B.combine(Term(one, false, true), Term(decay, true, true));
        auto offset = difference.value.multiply(scale).divide(k);
        if (!xi.sign) offset = offset.divide(decay);
        return addOffset!bits(mu, offset, false);
    }
}

package(mir.stat) bool supported(uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    if (x.isNaN) return false;
    if (x.isInfinity) return !x.sign && (!xi.sign || !xi.coefficient);
    if (position(x, mu) < 0) return false;
    if (!xi.sign || !xi.coefficient) return true;
    return !ShapeTransform!bits.residual(x, mu, sigma, xi).negative;
}

package(mir.stat) auto paretoTail(bool upper, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.tail!upper(x, mu, sigma, xi);
}

package(mir.stat) auto paretoDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.density!logarithmic(x, mu, sigma, xi);
}

package(mir.stat) auto paretoQuantile(uint bits)(Fp!bits p, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.quantile(p, mu, sigma, xi);
}
