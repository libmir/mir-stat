/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_shape_transform;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: exponentBias;

@safe pure nothrow @nogc:

package(mir.stat) struct ShapeTransform(uint bits)
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
        return combineThree(a, b, c);
    }

    static Term combineThree(Term a, Term b, Term c)
    {
        if (!c.present)
            return combine(a, b);
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

    // Magnitude of log(1+xi*z)/xi at a nonzero standardized distance.
    // The signed transform has the sign of x-mu. Both signs use the exact support test.
    static S log1pScaledMagnitude(T x, T mu, T sigma, T xi, Term numerator)
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

    // A nonzero correction below the working grid can still decide an
    // output midpoint. Set an odd low bit on the correct side, including
    // just below a power of two, then normalize on the separate exponent.
    static S withCorrection(S value, bool positive)
    {
        auto c = value.fraction.coefficient;
        if (!c.bt(0))
        {
            if (positive) c += 1;
            else c -= 1;
        }
        auto result = S.from(Fp!work(c));
        result.exponent += value.exponent - exponentBias - work;
        return result;
    }

    static T finish(Term result, S divisor)
    {
        if (!result.present) return T(0);
        return addOffset!bits(T(0), result.value.divide(divisor), result.negative);
    }
}
