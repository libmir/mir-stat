/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_generalized_pareto;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

private struct Evaluation(uint bits)
{
    alias T = Fp!bits;
    alias L = ProbabilityLog!(3 * bits + 64);
    enum work = 3 * bits + 256;
    alias S = L.S;

    struct Term
    {
        S value;
        bool negative;
        bool present;
    }

    static Term term(T x)
    {
        return x.coefficient ? Term(S.from(x), x.sign, true) : Term.init;
    }

    // Carry the exponent separately when subtracting nearly equal dyadics.
    // The larger term selects the local grid, even at the Fp exponent floor.
    static Term combine(Term a, Term b)
    {
        if (!a.present) return b;
        if (!b.present) return a;
        if (a.negative == b.negative)
            return Term(a.value.add(b.value), a.negative, true);
        if (a.value.exponent < b.value.exponent ||
            (a.value.exponent == b.value.exponent &&
             a.value.fraction.coefficient < b.value.fraction.coefficient))
            return combine(b, a);
        const gap = a.value.exponent - b.value.exponent;
        if (gap > UInt!128(work))
            return a;
        auto c = a.value.fraction.coefficient.toSize!(2 * work);
        c <<= cast(size_t)gap;
        c -= b.value.fraction.coefficient.toSize!(2 * work);
        if (!c) return Term.init;
        auto value = S.from(Fp!(2 * work)(c));
        value.exponent += b.value.exponent - exponentBias - work;
        return Term(value, a.negative, true);
    }

    // sigma + xi*x - xi*mu. Each product has at most 2*bits significant
    // bits. Sum the two leading terms first so exact cancellation exposes
    // the third, even across an arbitrarily large exponent gap.
    static Term residual(T x, T mu, T sigma, T xi)
    {
        const k = S.from(xi);
        auto a = term(x);
        auto b = term(mu);
        auto c = term(sigma);
        if (a.present) {
            a.value = a.value.multiply(k);
            a.negative ^= xi.sign;
        }
        if (b.present) {
            b.value = b.value.multiply(k);
            b.negative ^= !xi.sign;
        }
        if (!a.present || (b.present && a.value.exponent < b.value.exponent))
        {
            const tmp = a; a = b; b = tmp;
        }
        if (!b.present || b.value.exponent < c.value.exponent)
            return combine(combine(a, c), b);
        return combine(combine(a, b), c);
    }

    static auto logarithm(S value)
    {
        L.Context ctx;
        const neg = value.exponent < exponentBias;
        const count = neg ? exponentBias - value.exponent : value.exponent - exponentBias;
        if (count <= (1UL << 62))
        {
            const e = cast(long)cast(ulong)count;
            return L.logDyadic(value.fraction.coefficient, (neg ? -e : e) - work, ctx);
        }
        const frac = L.logDyadic(value.fraction.coefficient, -cast(long)work, ctx);
        const ln2 = L.logDyadic(UInt!64(1), 1, ctx);
        return addOffset!work(frac, S.from(Fp!128(count)).multiply(S.from(ln2)), neg);
    }

    static S hazard(T x, T mu, T sigma, T xi, Term numerator)
    {
        const scale = S.from(sigma);
        const z = distance!work(x, mu).divide(scale);
        if (!xi.coefficient)
            return z;
        const k = S.from(xi);
        if (k.multiply(z).exponent + work / 2 <= exponentBias)
            return z; // log1p(u)/u = 1 + O(u)
        return S.from(logarithm(numerator.value.divide(scale))).divide(k);
    }

    static T tail(bool upper)(T x, T mu, T sigma, T xi)
    {
        if (x.isInfinity) return T(upper ? 0 : 1);
        if (!position(x, mu)) return T(upper ? 1 : 0);
        auto numerator = xi.coefficient ? residual(x, mu, sigma, xi) : term(sigma);
        if (!numerator.present) return T(upper ? 0 : 1);
        if (position(xi, T(-1)) == 0)
        {
            const part = upper ? numerator.value : distance!work(x, mu);
            return part.divide(S.from(sigma)).finish!bits;
        }
        const h = hazard(x, mu, sigma, xi, numerator);
        if (h.exponent >= exponentBias + 66) return T(upper ? 0 : 1);
        if (h.exponent + work / 2 <= exponentBias)
        {
            static if (upper) return T(1);
            else return h.finish!bits;
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
        const numerator = xi.coefficient ? residual(x, mu, sigma, xi) : term(sigma);
        if (!numerator.present)
            return shapeOrder < 0 ? T(false, long.max, UInt!bits(0))
                : logarithmic ? minusInfinity : T(0);
        const h = hazard(x, mu, sigma, xi, numerator);
        const magnitude = h.multiply(distance!work(xi, T(-1)));
        static if (logarithmic)
            return addOffset!bits(peak, magnitude, shapeOrder > 0);
        else
        {
            if (magnitude.exponent >= exponentBias + 66)
                return shapeOrder > 0 ? T(0) : T(false, long.max, UInt!bits(0));
            const decay = magnitude.exponent + work <= exponentBias ? one : scaledDecay(magnitude);
            const value = shapeOrder > 0 ? decay.divide(scale) : one.divide(decay.multiply(scale));
            return addOffset!bits(T(0), value, false);
        }
    }

}

package(mir.stat) bool supported(uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    if (x.isNaN) return false;
    if (x.isInfinity) return !x.sign && (!xi.sign || !xi.coefficient);
    if (position(x, mu) < 0) return false;
    if (!xi.sign || !xi.coefficient) return true;
    return !Evaluation!bits.residual(x, mu, sigma, xi).negative;
}

package(mir.stat) auto paretoTail(bool upper, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.tail!upper(x, mu, sigma, xi);
}

package(mir.stat) auto paretoDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits sigma, Fp!bits xi)
{
    return Evaluation!bits.density!logarithmic(x, mu, sigma, xi);
}
