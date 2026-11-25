/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_gamma;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;
import mir.stat.internal.fp_location_scale: addOffset, position;
import mir.stat.internal.fp_log_gamma: PositiveLogGamma;
import mir.stat.internal.fp_scaled_probability: exponentBias;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

private struct GammaDensity(uint bits)
{
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    alias Term = B.Term;
    alias G = PositiveLogGamma!bits;
    enum work = B.work;
    alias W = Fp!work;

    static Term term(uint n)(Fp!n v, bool subtract = false)
    {
        return v.coefficient ? Term(S.from(v), v.sign != subtract, true) : Term.init;
    }

    static S widen(G.S source)
    {
        auto value = S.from(source.fraction);
        value.exponent += source.exponent - exponentBias;
        return value;
    }

    alias P = ShapeTransform!(2 * bits);

    // For integer n, form x^(n-1) / ((n-1)! * scale^n) with exact
    // products and a separate exponent. Bound n to keep the factorial
    // in UInt!128, and reserve precision for division near one.
    static bool integerPrefactor(T x, T shape, T scale, out P.S value)
    {
        const zeros = shape.coefficient.cttz;
        if (shape.exponent < -cast(long)zeros) return false;
        const shift = cast(ulong)shape.exponent + zeros;
        const odd = shape.coefficient >> zeros;
        if (shift > 5 || odd > (32u >> cast(uint)shift)) return false;
        const n = cast(uint)odd << cast(uint)shift;
        if (n < 2) return false;

        UInt!128 factorial = UInt!128(1);
        foreach (k; 2 .. n) factorial *= k;
        const factorialBits = 128 - factorial.ctlz - factorial.cttz;
        const xBits = bits - x.coefficient.ctlz - x.coefficient.cttz;
        const scaleBits = bits - scale.coefficient.ctlz - scale.coefficient.cttz;
        enum budget = P.work - bits - 64;
        if ((n - 1) * xBits > budget || n * scaleBits + factorialBits > budget)
            return false;
        const numerator = P.S.from(x).power(n - 1);
        const denominator = P.S.from(scale).power(n).multiply(P.S.from(Fp!128(factorial)));
        value = numerator.divide(denominator);
        return true;
    }

    // Positive r-1-log(r). Keep the quadratic term on its own exponent
    // when r is near one. The input delta is the signed relative residual.
    static S deviance(S r, Term delta)
    {
        if (delta.value.exponent >= exponentBias)
        {
            const l = term(B.logarithm(r), true);
            return B.combine(delta, l).value;
        }
        // d-log(1+d) = d^2*(1/2 - d/3 + d^2/4 - ...), |d| < 1/2.
        const one = S.from(W(1));
        S series = one;
        series.exponent -= 1;
        auto power = delta.value;
        for (uint n = 3; power.exponent + work >= exponentBias; ++n)
        {
            const next = power.divide(S.from(W(n)));
            series = B.combine(Term(series, false, true),
                Term(next, !delta.negative && (n & 1) != 0, true)).value;
            power = power.multiply(delta.value);
        }
        return delta.value.multiply(delta.value).multiply(series);
    }

    static Term logDensity(T x, T shape, T scale)
    {
        const a = S.from(shape);
        const s = S.from(scale);
        const one = S.from(W(1));
        if (position(shape, T(1)) == 0)
        {
            const base = term(B.logarithm(s), true);
            return x.coefficient ? B.combine(base, Term(S.from(x).divide(s), true, true)) : base;
        }
        const observation = S.from(x);
        const z = observation.divide(s);
        if (a.exponent > exponentBias + G.shift)
        {
            import mir.stat.internal.fp_pi: guardedPi;
            // x and a*s have at most 2*bits coefficient bits. Subtract
            // before division so a rounded quotient cannot hide a residual.
            const mean = a.multiply(s);
            auto delta = B.combine(Term(observation, false, true), Term(mean, true, true));
            Term loss;
            if (delta.present)
            {
                delta.value = delta.value.divide(mean);
                loss = Term(a.multiply(deviance(observation.divide(mean), delta)), true, true);
            }
            // log f = -a*(r-1-log r) - log(2*pi*x^2/a)/2 - R(a).
            auto normalizer = observation.multiply(observation).divide(a)
                .multiply(S.from(guardedPi!work()));
            normalizer.exponent += 1;
            auto peak = term(B.logarithm(normalizer), true);
            if (peak.present) peak.value.exponent -= 1;
            const correction = widen(G.correction(G.S.from(shape)));
            return B.combineThree(peak, loss, Term(correction, true, true));
        }

        if (a.exponent <= exponentBias - 1)
        {
            // log(a/x) + a*log(z) - z - log Gamma(1+a). Combining the
            // first ratio directly preserves its exact zero when x == a.
            auto weighted = term(B.logarithm(z));
            if (weighted.present) weighted.value = weighted.value.multiply(a);
            const residual = B.combineThree(weighted, Term(z, true, true),
                Term(widen(G.smallIncrementMagnitude(shape)), false, true));
            return B.combine(term(B.logarithm(a.divide(observation))), residual);
        }
        auto weighted = term(B.logarithm(z));
        if (weighted.present)
        {
            const shift = B.combine(Term(a, false, true), Term(one, true, true));
            weighted.value = weighted.value.multiply(shift.value);
            weighted.negative ^= shift.negative;
        }
        const base = B.combineThree(weighted, Term(z, true, true), term(B.logarithm(s), true));
        return B.combine(base, term(G.guardedLogarithm(shape), true));
    }
}

package(mir.stat) Fp!bits gammaDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits shape, Fp!bits scale)
{
    import mir.stat.internal.fp_scaled_probability: scaledDecay;
    alias T = Fp!bits;
    alias D = GammaDensity!bits;
    alias S = D.S;
    const infinity = T(false, long.max, UInt!bits(0));
    const minusInfinity = T(true, long.max, UInt!bits(0));
    if (x.isInfinity)
        return logarithmic ? minusInfinity : T(0);
    const side = position(shape, T(1));
    if (!x.coefficient && side != 0)
        return side < 0 ? infinity : logarithmic ? minusInfinity : T(0);

    alias P = D.P;
    P.S prefactor;
    if (side != 0 && D.integerPrefactor(x, shape, scale, prefactor))
    {
        const z = P.S.from(x).divide(P.S.from(scale));
        static if (logarithmic)
        {
            const base = P.logarithm(prefactor);
            return addOffset!bits(base, z, true);
        }
        else
        {
            // For n <= 32 and z >= 2^^65 even z^(n-1)*exp(-z/2) < 1.
            if (z.exponent >= exponentBias + 66) return T(0);
            // Cross-multiplying by a target midpoint uses at most
            // budget + bits + 1 significant bits in the denominator
            // times its midpoint coefficient. At this smaller decay,
            // only an exact midpoint can change the rounded result.
            if (z.exponent + P.work - 16 <= exponentBias)
                prefactor = P.withCorrection(prefactor, false);
            else
                prefactor = prefactor.multiply(scaledDecay(z));
            return addOffset!bits(T(0), prefactor, false);
        }
    }
    static if (!logarithmic)
    {
        if (side == 0)
        {
            // Divide the scaled decay directly, including the exact
            // reciprocal-scale limit at zero.
            const s = S.from(scale);
            auto decay = S.from(T(1));
            if (x.coefficient)
            {
                const z = S.from(x).divide(s);
                if (z.exponent >= exponentBias + 66) return T(0);
                decay = scaledDecay(z);
            }
            return addOffset!bits(T(0), decay.divide(s), false);
        }
    }

    const result = D.logDensity(x, shape, scale);
    if (!result.present) return T(logarithmic ? 0 : 1);
    static if (logarithmic)
        return addOffset!bits(T(0), result.value, result.negative);
    else
    {
        // At |log(f)| >= 2^65 the final value is outside the Fp range.
        if (result.value.exponent >= exponentBias + 66)
            return result.negative ? T(0) : infinity;
        auto density = scaledDecay(result.value);
        if (!result.negative) density = S.from(T(1)).divide(density);
        return addOffset!bits(T(0), density, false);
    }
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!32;
    alias U = UInt!32;
    const x = T(false, 124, U(1));
    const shape = T(false, 93, U(0x80000001));
    const scale = T(false, -31, U(0x7fffffff));
    assert(gammaDensity!true(x, shape, scale) == T(true, -26, U(0xb1938571)));
    assert(gammaDensity!false(x, shape, scale) == T(false, -96, U(0xf7c72c72)));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!512;
    alias U = UInt!512;
    const x = T(false, 300, U(1));
    enum expected = U.fromHexString(
        "cc42299ea1b284687e59e2805d5c717fa7053e60a24e0db1808e369004c73a5d" ~
        "3bf158aa88f96b16106d7a69610928a3fd7d27ccf4d1f61f514704ab2449c3ec");
    assert(gammaDensity!false(x, x, T(1)) == T(false, -663, expected));
}
