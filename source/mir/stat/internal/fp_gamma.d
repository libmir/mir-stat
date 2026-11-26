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

private enum gammaTermLimit = 32768;

// Regularized incomplete gamma for a positive shape. Work on normalized
// ratios so x/scale can cross either end of the public exponent range.
private struct GammaIntegral(uint bits)
{
    alias D = GammaDensity!bits;
    alias B = D.B;
    alias S = B.S;
    alias T = Fp!bits;
    alias W = D.W;
    alias Term = B.Term;
    enum work = D.work;
    uint termLimit;

    S a, one;
    Term negLogGamma1p;
    Term stirlingTerm;
    bool small;
    uint integerShape;

    @safe pure nothrow @nogc:

    this(T shape, uint limit = gammaTermLimit)
    {
        termLimit = limit;
        a = S.from(shape);
        one = S.from(W(1));
        small = a.exponent <= exponentBias - 1;
        if (a.exponent > exponentBias && a.exponent <= exponentBias + 6)
        {
            const shift = work - cast(uint)(a.exponent - exponentBias);
            if (a.fraction.coefficient.cttz >= shift)
                integerShape = cast(uint)(a.fraction.coefficient >> shift);
        }
        if (small)
            negLogGamma1p = Term(D.widen(D.G.smallIncrementMagnitude(shape)), false, true);
        else if (a.exponent <= exponentBias + 16)
            negLogGamma1p = D.term(D.G.logGamma1p(shape), true);
        else
            stirlingTerm = Term(D.widen(D.G.correction(D.G.S.from(shape))), true, true);
    }

    // Inputs larger than 2^65 in magnitude are outside the exp range
    // needed by both tails. Callers establish the sign before this point.
    static S exponential(Term t)
    {
        import mir.stat.internal.fp_scaled_probability: scaledDecay;
        const unit = S.from(W(1));
        if (!t.present) return unit;
        const decay = scaledDecay(t.value);
        return t.negative ? decay : unit.divide(decay);
    }

    static int compare(S x, S y)
    {
        if (x.exponent != y.exponent)
            return x.exponent < y.exponent ? -1 : 1;
        return x.fraction.coefficient.opCmp(y.fraction.coefficient);
    }

    // Evaluate the upper fraction by successive numerator/denominator
    // convergents. Separate exponents avoid rescaling the recurrence.
    bool upperFactor(S z, out S factor)
    {
        const diff = B.combine(Term(z, false, true), Term(a, true, true));
        auto b = diff.present ? diff.value.add(one) : one;
        auto numerator = one;
        auto denominator = b;
        S previousNumerator;
        auto previousDenominator = one;
        auto last = one.divide(b);
        foreach (uint i; 1 .. termLimit + 1)
        {
            const n = S.from(W(i));
            auto c = B.combine(Term(a, false, true), Term(n, true, true));
            if (!c.present) { factor = last; return true; }
            c.value = c.value.multiply(n);
            b = b.add(S.from(W(2)));
            const top = i == 1 ? b : B.combine(Term(b.multiply(numerator), false, true),
                Term(c.value.multiply(previousNumerator), c.negative, true)).value;
            const bottom = B.combine(Term(b.multiply(denominator), false, true),
                Term(c.value.multiply(previousDenominator), c.negative, true)).value;
            const next = top.divide(bottom);
            const change = B.combine(Term(last, true, true), Term(next, false, true));
            if (!change.present || change.value.exponent + bits + 96 < next.exponent)
            {
                factor = next;
                return true;
            }
            previousNumerator = numerator;
            previousDenominator = denominator;
            numerator = top;
            denominator = bottom;
            last = next;
        }
        return false;
    }

    static Term product(Term x, Term y, bool negate = false)
    {
        if (!x.present || !y.present) return Term.init;
        return Term(x.value.multiply(y.value), (x.negative != y.negative) != negate, true);
    }

    // Taylor coefficients on one unit interval satisfy
    // (n+1)c[n+1] = (slope - n*grid)c[n] - curvature*c[n-1].
    bool segmentIntegral(Term slope, Term curvature, Term grid, out Term nextValue, out Term segment)
    {
        Term previous;
        auto coefficient = Term(one, false, true);
        nextValue = coefficient;
        segment = coefficient;
        uint negligible;
        const terms = termLimit < work ? termLimit : work;
        foreach (uint n; 1 .. terms + 1)
        {
            auto next = B.combine(product(slope, coefficient), product(curvature, previous, true));
            if (next.present) next.value = next.value.divide(S.from(W(n)));
            nextValue = B.combine(nextValue, next);
            previous = coefficient;
            coefficient = next;
            if (next.present) next.value = next.value.divide(S.from(W(n+1)));
            segment = B.combine(segment, next);
            // A single Taylor coefficient may vanish inside the series.
            if (!coefficient.present || coefficient.value.exponent + bits + 96 < exponentBias)
            {
                if (++negligible == 2)
                    return nextValue.present && !nextValue.negative && segment.present && !segment.negative;
            }
            else negligible = 0;
            slope = B.combine(slope, Term(grid.value, !grid.negative, true));
        }
        return false;
    }

    // Integrate outwards from z, in units of h=z/(sqrt(a)+|z-a|).
    // The gamma density relative to its value at z satisfies
    // (1+k*t) f'=(c-d*t) f, k=+/-h/z, c=k*(a-1-z), d=h*h/z.
    // This grid resolves the peak even when adding h to z would round
    // straight back to z. It also follows the short decay length far out.
    bool transition(S z, bool upper, out Term tail)
    {
        import mir.stat.internal.fp_pi: guardedPi;
        import mir.stat.internal.fp_scaled_probability: scaledSqrt;
        const root = scaledSqrt(a, false);
        auto delta = B.combine(Term(z, false, true), Term(a, true, true));
        S widthDivisor = root;
        Term loss;
        if (delta.present)
        {
            widthDivisor = root.add(delta.value);
            auto relative = delta;
            relative.value = relative.value.divide(a);
            loss = Term(a.multiply(D.deviance(z.divide(a), relative)), true, true);
            if (loss.value.exponent >= exponentBias + 66) return true;
        }
        auto k = Term(one.divide(widthDivisor), !upper, true);
        const d = z.multiply(k.value).multiply(k.value);
        const c = product(k, B.combine(delta, Term(one, false, true)), true);
        auto endpoint = one;
        Term area;
        foreach (uint j; 0 .. 4 * (bits + 128))
        {
            const index = Term(S.from(W(j ? j : 1)), false, j != 0);
            const divisor = B.combine(Term(one, false, true), product(k, index));
            if (!divisor.present || divisor.negative) return false;
            auto slope = B.combine(c, product(Term(d, false, true), index, true));
            if (slope.present) slope.value = slope.value.divide(divisor.value);
            const curvature = Term(d.divide(divisor.value), false, true);
            const grid = Term(k.value.divide(divisor.value), k.negative, true);
            Term nextValue, segment;
            if (!segmentIntegral(slope, curvature, grid, nextValue, segment)) return false;
            area = B.combine(area, Term(endpoint.multiply(segment.value), false, true));
            endpoint = endpoint.multiply(nextValue.value);
            // For a > 1 the density is log concave. Once the outward
            // logarithmic slope exceeds 1/4, the remaining area is at
            // most four times the endpoint density in these units.
            if (endpoint.exponent + bits + 100 < area.value.exponent)
            {
                const endIndex = Term(S.from(W(j + 1)), false, true);
                const endSlope = B.combine(c, product(Term(d, false, true), endIndex, true));
                const endDivisor = B.combine(Term(one, false, true), product(k, endIndex));
                if (endDivisor.present && !endDivisor.negative && endSlope.present &&
                    endSlope.negative && endSlope.value.exponent + 1 >= endDivisor.value.exponent)
                    break;
            }
            if (j + 1 == 4 * (bits + 128)) return false;
        }
        auto pi2 = S.from(guardedPi!work());
        pi2.exponent += 1;
        tail = Term(exponential(B.combine(loss, stirlingTerm)).multiply(root)
            .divide(widthDivisor).divide(scaledSqrt(pi2, false)).multiply(area.value), false, true);
        return true;
    }

    // Return the directly evaluated tail and its orientation. A zero is
    // encoded by an absent term. False denotes an exhausted iteration budget.
    bool evaluate(S z, out bool upper, out Term tail)
    {
        upper = compare(z, a) >= 0 && compare(z, one) >= 0;
        // P(n,z) = z^n/n! * (1 - nz/(n+1) + O(z^2)). For a short
        // dyadic ratio and a tiny z the only ambiguous rounding case is
        // a coincident midpoint. Retain the strict downward correction.
        if (integerShape && integerShape <= 32 && z.exponent + work < exponentBias &&
            integerShape * (work - z.fraction.coefficient.cttz) <= work - bits - 128)
        {
            UInt!128 factorial = UInt!128(1);
            foreach (uint n; 2 .. integerShape + 1) factorial *= n;
            const lead = z.power(integerShape).divide(S.from(Fp!128(factorial)));
            tail = Term(B.withCorrection(lead, false), false, true);
            return true;
        }
        if (stirlingTerm.present) return transition(z, upper, tail);
        auto weighted = D.term(B.logarithm(z));
        if (weighted.present) weighted.value = weighted.value.multiply(a);
        const power = B.combine(weighted, negLogGamma1p);
        if (small && compare(z, one) < 0 && (!power.present ||
            power.value.exponent < exponentBias))
        {
            // DLMF 8.7.1, alternating form. Isolate the constant so Q
            // remains relative-accurate even as shape approaches zero.
            upper = true;
            auto sum = one.divide(a.add(one));
            auto factorialTerm = one;
            foreach (uint n; 2 .. work + 2)
            {
                factorialTerm = factorialTerm.multiply(z).divide(S.from(W(n)));
                const next = factorialTerm.divide(a.add(S.from(W(n))));
                if (next.exponent + bits + 96 < sum.exponent) break;
                sum = B.combine(Term(sum, false, true), Term(next, !(n & 1), true)).value;
            }
            const e = exponential(power);
            auto base = B.combine(Term(one, false, true), Term(e, true, true));
            if (power.present && power.value.exponent + bits + 96 < exponentBias)
                base = Term(power.value, !power.negative, true);
            tail = B.combine(base, Term(e.multiply(a).multiply(z).multiply(sum), false, true));
            return true;
        }

        auto logFactor = B.combine(power, Term(z, true, true));
        if (logFactor.present && logFactor.negative &&
            logFactor.value.exponent >= exponentBias + 66) return true;
        auto factor = exponential(logFactor);
        if (upper)
        {
            S fraction;
            if (!upperFactor(z, fraction)) return false;
            factor = factor.multiply(a).multiply(fraction);
        }
        else
        {
            // DLMF 8.7.1, positive series. Ratios decrease with n.
            auto sum = one;
            auto term = one;
            foreach (uint n; 1 .. termLimit + 1)
            {
                const ratio = z.divide(a.add(S.from(W(n))));
                term = term.multiply(ratio);
                sum = sum.add(term);
                if (term.exponent + bits + 96 < sum.exponent)
                {
                    factor = factor.multiply(sum);
                    tail = Term(factor, false, true);
                    return true;
                }
            }
            return false;
        }
        tail = Term(factor, false, true);
        return true;
    }
}

package(mir.stat) Fp!bits gammaTail(bool upper, uint bits)(Fp!bits x, Fp!bits shape,
    Fp!bits scale, uint termLimit = gammaTermLimit)
{
    import mir.stat.internal.fp_weibull: weibullTail;
    alias T = Fp!bits;
    alias I = GammaIntegral!bits;
    if (x.isInfinity) return T(upper ? 0 : 1);
    if (!x.coefficient) return T(upper ? 1 : 0);
    if (position(shape, T(1)) == 0) return weibullTail!upper(x, shape, scale);
    auto integral = I(shape, termLimit);
    I.Term tail;
    bool directUpper;
    if (!integral.evaluate(I.S.from(x).divide(I.S.from(scale)), directUpper, tail))
        return T(false, long.max, UInt!bits(1));
    const complement = upper != directUpper;
    if (!tail.present) return T(complement ? 1 : 0);
    return addOffset!bits(T(complement ? 1 : 0), tail.value, complement);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    // Exhaust the lower series, upper fraction and local density expansion.
    // No partially converged value may escape from any of the three paths.
    alias T = Fp!64;
    assert(gammaTail!false(T(1), T(2), T(1), 1).isNaN);
    assert(gammaTail!true(T(2), T(1.5), T(1), 1).isNaN);
    assert(gammaTail!true(T(65536), T(65536), T(1), 1).isNaN);
}

// Consecutive ranks enumerate positive dyadics on the output grid, with
// the first binade sharing the subnormal grid. Include one virtual point
// past the largest finite value to locate the overflow rounding midpoint.
private struct GammaGrid(uint bits)
{
    alias S = GammaIntegral!bits.S;
    alias Rank = UInt!(bits + 128);
    enum block = Rank(1) << (bits - 1);

    static S value(Rank rank) @safe pure nothrow @nogc
    {
        const binade = rank >> (bits - 1);
        auto coefficient = (rank & (block - 1)).toSize!bits;
        if (binade) coefficient |= UInt!bits(1) << (bits - 1);
        auto result = S.from(Fp!bits(coefficient));
        result.exponent -= 1UL << 63;
        if (binade) result.exponent += binade.toSize!128 - 1;
        return result;
    }
}

package(mir.stat) Fp!bits gammaQuantile(uint bits)(Fp!bits p, Fp!bits shape,
    Fp!bits scale, uint termLimit = gammaTermLimit)
{
    import mir.stat.internal.fp_weibull: weibullQuantile;
    alias T = Fp!bits;
    alias I = GammaIntegral!bits;
    alias Grid = GammaGrid!bits;
    alias Rank = Grid.Rank;
    const infinity = T(false, long.max, UInt!bits(0));
    if (!p.coefficient) return T(0);
    if (position(p, T(1)) == 0) return infinity;
    if (position(shape, T(1)) == 0) return weibullQuantile(p, shape, scale);

    auto integral = I(shape, termLimit);
    const divisor = I.S.from(scale);
    const lowerTarget = I.S.from(p);
    const upperTarget = I.B.combine(I.Term(integral.one, false, true),
        I.Term(lowerTarget, true, true)).value;
    const median = position(p, T(false, -1, UInt!bits(1))) == 0;
    const mean = integral.a.multiply(divisor);
    Rank low;
    const end = (Rank(ulong.max) + 1) << (bits - 1);
    Rank high = end;
    while (low < high)
    {
        const rank = (low + high) >> 1;
        auto midpoint = Grid.value(rank + 1);
        if (rank) midpoint = midpoint.add(Grid.value(rank));
        midpoint.exponent -= 1;
        // The gamma median is strictly below the mean. At an exact
        // mean midpoint this decides the side even when the skewness
        // correction to one half is smaller than the integration error.
        if (median && I.compare(midpoint, mean) == 0)
        {
            high = rank;
            continue;
        }
        I.Term tail;
        bool upper;
        if (!integral.evaluate(midpoint.divide(divisor), upper, tail))
            return T(false, long.max, UInt!bits(1));
        const cmp = tail.present ? I.compare(tail.value, upper ? upperTarget : lowerTarget) : -1;
        if ((upper ? -cmp : cmp) < 0 || (!cmp && rank.bt(0)))
            low = rank + 1;
        else
            high = rank;
    }
    if (!low) return T(0);
    if (low == end) return infinity;
    return addOffset!bits(T(0), Grid.value(low), false);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    assert(gammaQuantile(T(0.5), T(2), T(1), 1).isNaN);
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
    const half = T(false, -1, U(1));
    const threeQuarters = T(false, -2, U(3));
    assert(gammaTail!false(half, threeQuarters, T(1)) == T(false, -32, U(0x8726e2ed)));
    assert(gammaTail!true(half, threeQuarters, T(1)) == T(false, -33, U(0xf1b23a26)));
    assert(gammaQuantile(T(false, -4, U(15)), T(false, -4, U(1)), T(1)) ==
        T(false, -33, U(0x883babd5)));
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
    enum lower = U.fromHexString(
        "cb849b65036bd619c0482a825720333b5e8ffa52c85906d7dc359fa011cb154f" ~
        "2e4c2f6ab72228b706bb5321ee7f629ee5edf6984d8fc2f3d9f44b7e165540c4");
    assert(gammaTail!false(T(0.5), T(1.5), T(1)) == T(false, -514, lower));
    enum center = U.fromHexString(
        "ffbbe9f1bfa03393c02bfe82011f2ad721888a35ebfba984461779f5b0b0ed1d" ~
        "32b7be83af51276c9e729f38f5033f458383dfb27790104902b085201b2ad10e");
    assert(gammaTail!true(T(65536), T(65536), T(1)) == T(false, -513, center));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    alias U = UInt!64;
    // MPFR's upper incomplete gamma at each pair of bounding midpoints
    // brackets the exact target for three adjacent probability inputs.
    const T[3] probabilities = [T(false, -65, U.max), T(0.5), T(false, -64, U(0x8000000000000001UL))];
    foreach (i, probability; probabilities)
        assert(gammaQuantile(probability, T(2.5), T(1)) == T(false, -62, U(10033784061509306440UL + i)));
    assert(gammaQuantile(T(0.9375), T(65536), T(1)) ==
        T(false, -47, U(9278707942925074154UL)));
    // Invert (1+x)*exp(-x) = 1/2 independently by MPFR bisection.
    alias V = Fp!128;
    enum expected = UInt!128.fromHexString("d6d412fcbb1593e94926a69af3a8f66c");
    assert(gammaQuantile(V(0.5), V(2), V(1)) == V(false, -127, expected));
}
