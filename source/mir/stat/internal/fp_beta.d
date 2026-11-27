/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_beta;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_log_gamma: PositiveLogGamma;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

private struct BetaIntegral(uint bits)
{
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    alias Term = B.Term;
    alias W = Fp!(B.work);
    alias G = PositiveLogGamma!bits;

    S one;
    uint limit;
    Term inverseBetaLog;
    bool reuseNormalizer;

    struct Point
    {
        S x, y;
        S negLogX, negLogY;
        Point reflect() const { return Point(y, x, negLogY, negLogX); }
    }

    static Term signedValue(uint n)(Fp!n v, bool subtract = false)
    {
        return v.coefficient ? Term(S.from(v), v.sign != subtract, true) : Term.init;
    }

    static Term pos(S v) { return Term(v, false, true); }
    static Term neg(S v) { return Term(v, true, true); }

    static Term times(Term v, S factor)
    {
        if (v.present) v.value = v.value.multiply(factor);
        return v;
    }

    static int compare(S a, S b)
    {
        if (a.exponent != b.exponent) return a.exponent < b.exponent ? -1 : 1;
        return a.fraction.coefficient.opCmp(b.fraction.coefficient);
    }

    static S widen(uint n)(ProbabilityScaled!n v)
    {
        auto result = S.from(v.fraction);
        result.exponent += v.exponent - exponentBias;
        return result;
    }

    Point point(T x)
    {
        const X = S.from(x);
        const Y = distance!(B.work)(T(1), x);
        // Keep -log(1-x) on its own exponent even if y rounds to one.
        return Point(X, Y, S.from(B.logarithm(X)), widen(ProbabilityLog!bits.complement(x)));
    }

    // 1-exp(-t). Below the working grid retain the scale of t.
    S oneMinusDecay(S t)
    {
        if (t.exponent + B.work < exponentBias) return t;
        if (t.exponent >= exponentBias + 66) return one;
        if (t.exponent + B.work / 2 < exponentBias)
        {
            auto halfSquare = t.multiply(t);
            --halfSquare.exponent;
            return B.combine(pos(t), neg(halfSquare)).value;
        }
        return B.combine(pos(one), neg(scaledDecay(t))).value;
    }

    T powerTail(T shape, Point p, bool complement, bool exactBase)
    {
        const k = S.from(shape);
        const linear = k.multiply(p.y);
        if (p.y.exponent + 2 * bits + 8 < exponentBias &&
            linear.exponent + 2 * bits + 8 < exponentBias)
        {
            // 1-(1-y)^k = k*y - k*(k-1)*y^2/2 + ... . The product
            // is exact; the correction can only decide a target midpoint.
            if (!complement) return T(1);
            return B.withCorrection(linear, compare(k, one) < 0).finish!bits;
        }
        if (exactBase && k.exponent > exponentBias && k.exponent <= exponentBias + 12)
        {
            const shift = B.work - cast(uint)(k.exponent - exponentBias);
            if (k.fraction.coefficient.cttz >= shift)
            {
                const n = cast(uint)(k.fraction.coefficient >> shift);
                if (n * (B.work - p.x.fraction.coefficient.cttz) <= B.work)
                    return addOffset!bits(T(complement ? 1 : 0), p.x.power(n), complement);
            }
        }
        const rate = k.multiply(p.negLogX);
        if (complement) return oneMinusDecay(rate).finish!bits;
        if (rate.exponent >= exponentBias + 66) return T(0);
        return scaledDecay(rate).finish!bits;
    }

    // (a+b)*x-a. Use the smaller coordinate, whose displacement from
    // the endpoint is exact, before aligning the three dyadic products.
    static Term residual(S a, S b, Point p)
    {
        if (compare(p.x, p.y) <= 0)
            return B.combineThree(pos(a.multiply(p.x)), pos(b.multiply(p.x)), neg(a));
        return B.combineThree(pos(b), neg(a.multiply(p.y)), neg(b.multiply(p.y)));
    }

    // u-log(1+u), where ratio=1+u. The quadratic series avoids
    // subtracting nearly equal terms at the mean.
    S deviance(Term u, S ratio)
    {
        if (u.value.exponent >= exponentBias)
            return B.combine(u, signedValue(B.logarithm(ratio), true)).value;
        auto series = one;
        --series.exponent;
        S power = u.value;
        for (uint k = 3; power.exponent + B.work >= exponentBias; ++k)
        {
            series = B.combine(pos(series), Term(power.divide(S.from(W(k))),
                !u.negative && (k & 1) != 0, true)).value;
            power = power.multiply(u.value);
        }
        return u.value.multiply(u.value).multiply(series);
    }

    // log(x^a * y^b / B(a,b)) relative to its peak at x=a/(a+b).
    Term centered(S a, S b, Point p)
    {
        import mir.stat.internal.fp_pi: guardedPi;
        const sum = a.add(b);
        auto pi2 = S.from(guardedPi!(B.work)());
        ++pi2.exponent;
        auto peak = signedValue(B.logarithm(a.multiply(b).divide(sum.multiply(pi2))));
        if (peak.present) --peak.value.exponent;
        auto ga = G.S.from(a.fraction), gb = G.S.from(b.fraction);
        ga.exponent += a.exponent - exponentBias;
        gb.exponent += b.exponent - exponentBias;
        peak = B.combine(peak, pos(widen(G.correction(ga.add(gb)))));
        peak = B.combine(peak, neg(widen(G.correction(ga))));
        peak = B.combine(peak, neg(widen(G.correction(gb))));
        const delta = residual(a, b, p);
        if (!delta.present) return peak;
        const u = Term(delta.value.divide(a), delta.negative, true);
        const v = Term(delta.value.divide(b), !delta.negative, true);
        return B.combineThree(peak, neg(a.multiply(deviance(u, p.x.multiply(sum).divide(a)))),
            neg(b.multiply(deviance(v, p.y.multiply(sum).divide(b)))));
    }

    // For either Lentz denominator z, consecutive coefficients q1 and q2
    // give z' = (r + q1/z)/(1 + q1/z), where r = 1 + q2. The product
    // of the two denominators in the pair is r + q1/z. Return an absent
    // product if the update is singular or loses too many guard bits.
    Term updatePair(ref Term z, Term q1, S r)
    {
        const t = Term(q1.value.divide(z.value), q1.negative != z.negative, true);
        const first = B.combine(pos(one), t);
        const product = B.combine(pos(r), t);
        if (!first.present || !product.present) return Term.init;
        // Reject cancellation that consumes the guard bits needed for the
        // convergence test, including when the fraction terminates exactly.
        enum spare = B.work - bits - 96;
        if (t.negative && (first.value.exponent + spare < one.exponent ||
            product.value.exponent + spare < r.exponent)) return Term.init;
        z = Term(product.value.divide(first.value), product.negative != first.negative, true);
        return product;
    }

    // Modified Lentz updates for DLMF 8.17.22. A zero denominator
    // or an exhausted budget leaves the tail unresolved.
    bool fraction(S a, S b, Point p, out S factor)
    {
        const ap1 = a.add(one);
        // 1-(a+b)*x/(a+1) = (1-((a+b)*x-a))/(a+1).
        auto delta = residual(a, b, p);
        delta.negative = !delta.negative;
        const initial = B.combine(pos(one), delta);
        if (!initial.present || initial.negative) return false;
        auto d = pos(initial.value.divide(ap1));
        auto c = pos(one);
        auto h = pos(ap1.divide(initial.value));
        for (uint j = 0; j < limit; ++j)
        {
            const m = S.from(W(j + 1));
            const bm = B.combine(pos(b), neg(m));
            if (!bm.present)
            {
                factor = h.value;
                return !h.negative;
            }
            const even = m.add(m);
            const base = a.add(even);
            // Do not form a+2m-1 by subtracting one from a rounded sum.
            const prev = a.add(S.from(W(2UL * j + 1)));
            const q1 = Term(bm.value.multiply(m).multiply(p.x).divide(prev.multiply(base)),
                bm.negative, true);
            // Expand 1+q2 before rounding. All numerator terms are
            // positive, even when x rounds to one at working precision:
            // (a+2m)(a+2m+1) - (a+m)(a+b+m)x
            // = (a+m)(initial + 2m + m*y) + m(m+1).
            const r = a.add(m).multiply(initial.value.add(even).add(m.multiply(p.y)))
                .add(m.multiply(m.add(one))).divide(base.multiply(base.add(one)));
            const numerator = updatePair(c, q1, r);
            if (!numerator.present) return false;
            const denominator = updatePair(d, q1, r);
            if (!denominator.present) return false;
            h = Term(h.value.multiply(numerator.value.divide(denominator.value)),
                h.negative != (numerator.negative != denominator.negative), true);
            if (c.negative != d.negative || h.negative) continue;
            const diff = B.combine(pos(one), neg(c.value.divide(d.value)));
            if (!diff.present || diff.value.exponent + bits + 96 < exponentBias)
            {
                factor = h.value;
                return true;
            }
        }
        return false;
    }

    // Raise small shapes before using the centered formula. From
    // B(a+1,b) = a*B(a,b)/(a+b), each increment contributes
    // log(a/(a+b)) - log(x) to the original logarithmic factor.
    uint raise(ref S a, S b, ref S numerator, ref S denominator)
    {
        uint n;
        while (a.exponent <= exponentBias + G.shift)
        {
            numerator = numerator.multiply(a);
            denominator = denominator.multiply(a.add(b));
            a = a.add(one);
            ++n;
        }
        return n;
    }

    Term front(S a, S b, Point p)
    {
        if (reuseNormalizer)
            return B.combineThree(inverseBetaLog, neg(a.multiply(p.negLogX)), neg(b.multiply(p.negLogY)));
        auto numerator = one, denominator = one;
        const n = raise(a, b, numerator, denominator);
        const m = raise(b, a, numerator, denominator);
        auto result = centered(a, b, p);
        if (n || m)
        {
            result = B.combine(result, signedValue(B.logarithm(numerator.divide(denominator))));
            if (n) result = B.combine(result, pos(S.from(W(n)).multiply(p.negLogX)));
            if (m) result = B.combine(result, pos(S.from(W(m)).multiply(p.negLogY)));
        }
        return result;
    }

    // Recover -log B once from the prefactor at exactly 1/2. Bound both
    // shapes so removing their log(1/2) terms costs at most 17 bits here.
    // Large shapes retain the centered evaluation at every search point.
    void prepareQuantile(T a, T b)
    {
        import mir.stat.internal.fp_location_scale: half;
        const A = S.from(a), Bshape = S.from(b);
        if (A.exponent > exponentBias + 16 || Bshape.exponent > exponentBias + 16) return;
        const center = point(half!bits());
        inverseBetaLog = B.combine(front(A, Bshape, center), pos(A.add(Bshape).multiply(center.negLogX)));
        reuseNormalizer = true;
    }

    bool raw(T a, T b, Point p, out Term logTail)
    {
        const A = S.from(a), Bshape = S.from(b);
        S factor;
        if (!fraction(A, Bshape, p, factor)) return false;
        logTail = B.combine(front(A, Bshape, p), signedValue(B.logarithm(factor.divide(A))));
        return true;
    }

    // Integrate from x to a dyadic c with b*c <= 1 and add the direct
    // upper tail at c. This keeps a tiny upper tail when 1-x rounds to one.
    bool anchored(T a, T b, Point p, out Term tail)
    {
        const A = S.from(a), bs = S.from(b);
        if (compare(A, one) > 0) return false;
        auto c = T(1);
        --c.exponent;
        if (bs.exponent > exponentBias + 1)
        {
            const inv = one.divide(bs);
            if (inv.exponent < exponentBias - (cast(ulong)long.max - bits)) return false;
            c.exponent += cast(long)cast(ulong)(inv.exponent - exponentBias);
        }
        const C = S.from(c);
        if (compare(p.x, C) >= 0) return false;
        const anchor = point(c);
        Term atC;
        if (!raw(b, a, anchor.reflect(), atC) || !atC.present || !atC.negative) return false;
        const ratioLog = S.from(B.logarithm(C.divide(p.x)));
        auto area = pos(oneMinusDecay(A.multiply(ratioLog)).divide(A));
        auto coefficient = pos(one);
        bool converged;
        for (uint k = 0; k < limit; ++k)
        {
            const n = S.from(W(k + 1));
            const diff = B.combine(pos(n), neg(bs));
            if (!diff.present)
            {
                converged = true;
                break;
            }
            coefficient = Term(coefficient.value.multiply(diff.value).multiply(C).divide(n),
                coefficient.negative != diff.negative, true);
            const an = A.add(n);
            const term = times(coefficient, oneMinusDecay(an.multiply(ratioLog)).divide(an));
            area = B.combine(area, term);
            if (area.present && term.value.exponent + bits + 96 < area.value.exponent)
            {
                converged = true;
                break;
            }
        }
        if (!converged || !area.present || area.negative) return false;
        const interval = B.combineThree(signedValue(B.logarithm(area.value)),
            pos(bs.multiply(anchor.negLogY)), front(A, bs, anchor));
        if (interval.present && !interval.negative) return false;
        tail = B.combine(fromLog(atC), fromLog(interval));
        return true;
    }

    Term fromLog(Term v)
    {
        if (!v.present) return pos(one);
        if (v.value.exponent >= exponentBias + 66) return Term.init;
        return pos(scaledDecay(v.value));
    }

    T evaluate(bool upper)(T x, T a, T b)
    {
        const p = point(x);
        if (position(b, T(1)) == 0) return powerTail(a, p, upper, true);
        if (position(a, T(1)) == 0)
        {
            const exactBase = x.exponent >= -cast(long)B.work - cast(long)x.coefficient.cttz;
            return powerTail(b, p.reflect(), !upper, exactBase);
        }
        Term tail;
        bool reflected;
        if (!directTail(a, b, p, reflected, tail)) return T(false, long.max, UInt!bits(1));
        const complement = reflected != upper;
        if (!tail.present) return T(complement ? 1 : 0);
        return addOffset!bits(T(complement ? 1 : 0), tail.value, complement);
    }

    // Keep the directly computed tail on its own exponent for comparisons
    // with a target probability before output rounding.
    bool directTail(T a, T b, Point p, out bool reflected, out Term tail)
    {
        // Compare x*(a+b+2) with a+1 without overflowing the shape sum.
        const side = B.combine(residual(S.from(a), S.from(b), p), B.combine(pos(p.x), neg(p.y)));
        reflected = side.present && !side.negative;
        T first = reflected ? b : a, second = reflected ? a : b;
        Point q = reflected ? p.reflect() : p;
        Term l;
        if (!raw(first, second, q, l)) return false;
        // Across the center, complementing the guarded result loses at most
        // three bits when the other probability is >= 1/8. Outside that
        // interval evaluate the smaller tail directly before narrowing.
        const switchLog = S.from(B.logarithm(S.from(W(8)).divide(S.from(W(7)))));
        if (!l.present || !l.negative || compare(l.value, switchLog) < 0)
        {
            if (!anchored(first, second, q, tail))
            {
                if (!raw(second, first, q.reflect(), l) || !l.present || !l.negative) return false;
                tail = fromLog(l);
            }
            reflected = !reflected;
        }
        else tail = fromLog(l);
        return !tail.present || compare(tail.value, one) <= 0;
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias I = BetaIntegral!64;
    const one = I.S.from(I.W(1));
    auto integral = I(one, 4096);
    auto z = I.pos(one);
    const half = I.S.from(I.W(0.5));
    assert(!integral.updatePair(z, I.neg(one), half).present); // first denominator is zero
    assert(!integral.updatePair(z, I.neg(half), half).present); // product is zero
    I.S nearHalf = half;
    nearHalf.fraction.coefficient += 1;
    assert(!integral.updatePair(z, I.neg(half), nearHalf).present);
    assert(I.compare(z.value, one) == 0 && !z.negative);
}

// Use the same centered normalization as the tails. The density divides
// the beta prefactor by x*(1-x) before the final logarithm or exponential
// is rounded to the caller's precision.
package(mir.stat) Fp!bits betaDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits a, Fp!bits b)
{
    alias T = Fp!bits;
    alias I = BetaIntegral!bits;
    alias S = I.S;
    alias B = I.B;
    const unit = T(1);
    if (!x.coefficient || position(x, unit) == 0)
    {
        const side = position(!x.coefficient ? a : b, unit);
        if (side < 0 || (logarithmic && side > 0))
            return T(side > 0, long.max, UInt!bits(0));
        if (side > 0) return T(0);
        const height = !x.coefficient ? b : a;
        static if (logarithmic) return cast(T)B.logarithm(S.from(height));
        else return height;
    }

    auto integral = I(S.from(unit), 4096);
    const p = integral.point(x);
    I.Term result;
    if (position(a, unit) == 0 || position(b, unit) == 0)
    {
        const reflected = position(a, unit) == 0;
        const k = S.from(reflected ? b : a);
        const q = reflected ? p.reflect() : p;
        const slope = B.combine(I.pos(k), I.neg(integral.one));
        if (!slope.present) return T(logarithmic ? 0 : 1);
        // With shape two the density is exactly twice the base. Near its
        // zero crossing log(2*base) keeps the displacement from one.
        if (I.compare(k, S.from(T(2))) == 0)
        {
            S height = q.x;
            ++height.exponent;
            static if (logarithmic) return cast(T)B.logarithm(height);
            else return height.finish!bits;
        }
        result = B.combine(I.signedValue(B.logarithm(k)),
            I.Term(slope.value.multiply(q.negLogX),
                !slope.negative, true));
    }
    else
    {
        result = B.combineThree(integral.front(S.from(a), S.from(b), p),
            I.pos(p.negLogX), I.pos(p.negLogY));
        // The Stirling and recurrence terms have bounded working precision.
        // A residual too close to zero cannot resolve the log density.
        if (!result.present || result.value.exponent + bits + 48 < exponentBias)
            return logarithmic ? T(false, long.max, UInt!bits(1)) : unit;
    }
    if (!result.present) return T(logarithmic ? 0 : 1);
    static if (logarithmic)
        return addOffset!bits(T(0), result.value, result.negative);
    else
    {
        if (result.value.exponent >= exponentBias + 66)
            return result.negative ? T(0) : T(false, long.max, UInt!bits(0));
        auto height = scaledDecay(result.value);
        if (!result.negative) height = integral.one.divide(height);
        return addOffset!bits(T(0), height, false);
    }
}

// Search rounding boundaries on the output grid, including the half-step
// below its smallest subnormal. Ranks advance by one between adjacent Fp
// values. The first binade uses the same spacing as the subnormal grid.
package(mir.stat) Fp!bits betaQuantile(uint bits)(Fp!bits p, Fp!bits a, Fp!bits b, uint limit = 4096)
{
    import mir.stat.internal.fp_location_scale: half;
    alias T = Fp!bits;
    alias I = BetaIntegral!bits;
    alias B = I.B;
    alias S = I.S;
    alias Rank = UInt!(bits+128);
    const unit = T(1);
    const nan = T(false, long.max, UInt!bits(1));
    if (!p.coefficient || position(p, unit) == 0) return p;
    if (position(a, unit) == 0 && position(b, unit) == 0) return p;
    if (position(a, b) == 0 && position(p, half!bits()) == 0) return half!bits();

    const one = S.from(unit);
    auto integral = I(one, limit);
    const target = integral.point(p);
    if (position(b, unit) == 0)
    {
        if (position(a, T(2)) == 0)
        {
            import mir.stat.internal.fp_scaled_probability: scaledSqrt;
            return scaledSqrt(target.x, false).finish!bits;
        }
        const rate = target.negLogX.divide(S.from(a));
        if (rate.exponent >= exponentBias + 66) return T(0);
        return scaledDecay(rate).finish!bits;
    }
    if (position(a, unit) == 0)
    {
        const k = S.from(b);
        const linear = target.x.divide(k);
        if (target.x.exponent + 2 * bits + 8 < exponentBias &&
            linear.exponent + 2 * bits + 8 < exponentBias)
        {
            // 1-(1-p)^(1/b) = p/b + (b-1)*(p/b)^2/2 + ...
            return B.withCorrection(linear, I.compare(k, one) > 0).finish!bits;
        }
        return integral.oneMinusDecay(target.negLogY.divide(k)).finish!bits;
    }

    integral.prepareQuantile(a, b);
    enum block = Rank(1) << (bits - 1);
    S grid(Rank rank, bool midpoint)
    {
        const binade = rank >> (bits - 1);
        auto coefficient = (rank & (block - 1)).toSize!(bits + 64);
        if (binade) coefficient |= block.toSize!(bits + 64);
        if (midpoint) coefficient = (coefficient << 1) + 1;
        auto value = S.from(Fp!(bits + 64)(coefficient));
        value.exponent -= 1UL << 63;
        if (binade) value.exponent += binade.toSize!128 - 1;
        if (midpoint) --value.exponent;
        return value;
    }

    // Compare the tail evaluated at the exact rounding midpoint. A target
    // indistinguishable at guard precision cannot select an output.
    bool below(Rank rank, out bool lower)
    {
        const x = grid(rank, true);
        const y = B.combine(I.pos(one), I.neg(x)).value;
        const negLogY = x.exponent + B.work / 2 < exponentBias ? x : S.from(B.logarithm(y));
        const point = I.Point(x, y, S.from(B.logarithm(x)), negLogY);
        I.Term tail;
        bool upper;
        if (!integral.directTail(a, b, point, upper, tail)) return false;
        const goal = upper ? target.y : target.x;
        const diff = B.combine(tail, I.neg(goal));
        if (!diff.present || diff.value.exponent + bits + 64 < goal.exponent) return false;
        lower = diff.negative != upper;
        return true;
    }

    auto high = (Rank(1UL << 63) + 2 - bits) << (bits - 1);
    Rank low;
    Rank stride = block;
    // Find the relevant exponent interval before bisecting significands.
    while (high > stride)
    {
        const rank = high - stride;
        bool lower;
        if (!below(rank, lower)) return nan;
        if (lower)
        {
            low = rank + 1;
            break;
        }
        high = rank;
        stride <<= 1;
    }
    while (low < high)
    {
        const rank = (low + high) >> 1;
        bool lower;
        if (!below(rank, lower)) return nan;
        if (lower) low = rank + 1;
        else high = rank;
    }
    return low ? grid(low, false).finish!bits : T(0);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    assert(betaQuantile(T(0.25), T(3), T(5), 0).isNaN);
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!32;
    alias U = UInt!32;
    const quarter = T(false, -2, U(1));
    assert(betaDensity!false(quarter, T(3), T(5)) == T(false, -30, U(0x84e40000)));
    assert(betaQuantile(T(false, -1, U(1)), T(2), T(3)) == T(false, -33, U(0xc57e1575)));
}

// Guarded regularized lower or strict upper beta tail for positive finite
// shapes and x in [0,1]. Each continued fraction allows at most limit pairs
// of updates; the small-input interval series allows limit terms. Exhaustion
// or a singular intermediate returns NaN. Very concentrated shapes near the
// mean can exceed these limits. Correct rounding is unproven.
package(mir.stat) Fp!bits betaTail(bool upper, uint bits)(Fp!bits x, Fp!bits a, Fp!bits b,
    uint limit = 4096)
    in ((!x.sign || !x.coefficient) && x.exponent != long.max && position(x, Fp!bits(1)) <= 0)
    in (!a.sign && !!a.coefficient && a.exponent != long.max)
    in (!b.sign && !!b.coefficient && b.exponent != long.max)
{
    alias T = Fp!bits;
    if (!x.coefficient) return T(upper ? 1 : 0);
    if (position(x, T(1)) == 0) return T(upper ? 0 : 1);
    if (position(a, T(1)) == 0 && position(b, T(1)) == 0)
    {
        import mir.stat.internal.fp_probability: fpProbabilityComplement;
        return upper ? fpProbabilityComplement(x) : x;
    }
    import mir.stat.internal.fp_location_scale: half;
    if (position(a, b) == 0 && position(x, half!bits()) == 0) return half!bits();
    alias I = BetaIntegral!bits;
    auto integral = I(I.S.from(I.W(1)), limit);
    return integral.evaluate!upper(x, a, b);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!64;
    alias U = UInt!64;
    const one = T(1);
    const half = T(0.5);
    const floor = T(false, long.min, U(1));
    assert(betaTail!true(T(0), T(2), half) == one);
    assert(betaTail!false(one, T(2), half) == one);
    assert(betaTail!true(one, T(2), half) == T(0));
    assert(betaTail!false(T(0), T(2), half) == T(0));
    assert(betaTail!false(floor, one, one) == floor);
    assert(betaTail!false(floor, one, half) == floor); // just above the midpoint
    assert(betaTail!true(T(0.25), one, T(2)) == T(0.5625));
    assert(betaTail!false(T(0.25), T(2), one) == T(0.0625));
    assert(betaTail!false(half, T(4096), T(4096), 0) == half);
    assert(betaTail!false(half, T(3), T(7), 0).isNaN);
    assert(betaTail!true(T(0.25), T(4096), T(12288), 1).isNaN);

    // For Beta(1/2,1/2) the CDF at 1/4 is exactly 1/3.
    assert(betaTail!false(T(0.25), half, half) == T(false, -65, U(0xaaaaaaaaaaaaaaabUL)));
    import mir.stat.distribution.beta: betaCDF;
    import mir.math.common: approxEqual;
    assert(approxEqual(betaCDF(0.25, 0.5, 0.5), 1.0 / 3, 32 * double.epsilon, 0.0));

    // Exact binomial sums at the mean and in a directly evaluated upper tail.
    assert(betaTail!false(T(0.25), T(4096), T(12288)) ==
        T(false, -64, U(0x804e9eafb1327e1cUL)));
    assert(betaTail!true(T(false, -64, U.max), T(3), T(7)) ==
        T(false, -506, U(0x8fffffffffffffffUL)));

    // For Beta(2,b), Q = (1-x)^b * (1+b*x). At x=4/b both tails
    // round to the same references for these two shapes. The reflected
    // coefficient q2 is close to -1 in both cases.
    foreach (e; [400, 448])
    {
        const b = T(false, e, U(1));
        const x = T(false, -e, U(4));
        assert(betaTail!true(x, T(2), b) == T(false, -67, U(0xbb8d59314c618cafUL)));
        assert(betaTail!false(x, T(2), b) == T(false, -64, U(0xe88e54d9d673ce6aUL)));
    }

    // Reference from the finite binomial expansion for integer b=5.
    const tiny = T(false, -1800, U(1));
    const x = T(false, -1000, U(1));
    assert(betaTail!true(x, tiny, T(5)) == T(false, -1854, U(0xacc41612ad8f477cUL)));
    // At the integration anchor itself the same finite expansion checks
    // the continued fraction for the opposite tail.
    assert(betaTail!true(T(0.125), tiny, T(5)) == T(false, -1865, U(0xe74f3a79958784b1UL)));
    // Nonterminating interval series; see the b=3/2 integral below.
    assert(betaTail!true(x, tiny, T(1.5)) == T(false, -1854, U(0xad222473fecd848fUL)));
    // Q(1/2; a, 1) = 1-exp(-a*log(2)) with a=2^-300.
    assert(betaTail!true(half, T(false, -300, U(1)), one) ==
        T(false, -364, U(0xb17217f7d1cf79acUL)));
    // Although 1-x rounds to one, (1-x)^(1/x) rounds to exp(-1).
    assert(betaTail!true(x, one, T(false, 1000, U(1))) ==
        T(false, -65, U(0xbc5ab1b16779be35UL)));
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!32;
    alias U = UInt!32;
    const quarter = T(false, -2, U(1));
    assert(betaTail!false(quarter, T(3), T(5)) == T(false, -34, U(0xf9700000)));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!256;
    alias U = UInt!256;
    // For b=3/2 the limit of Q/a as a tends to zero is
    // log((1+s)^2/x)-2*s, s=sqrt(1-x). At a=2^-1800 the relative
    // correction O(a*|log(x)|) is below the precision of this reference.
    const x = T(false, -1000, U(1));
    const a = T(false, -1800, U(1));
    enum coefficient = U.fromHexString(
        "ad222473fecd848e990d4f3c3fdd3e627d0e09351f226e438f95cf5b142acdcc");
    assert(betaTail!true(x, a, T(1.5)) == T(false, -2046, coefficient));
}
