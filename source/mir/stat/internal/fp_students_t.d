/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_students_t;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_beta: BetaIntegral;
import mir.stat.internal.fp_location_scale: addOffset, distance, half, position;
import mir.stat.internal.fp_scaled_probability: exponentBias;

@safe pure nothrow @nogc:

// For z = |x-location|/scale, twice the small tail is the regularized
// beta integral with weights nu and z^2 and shapes nu/2 and 1/2.
private struct StudentIntegral(uint bits)
{
    alias I = BetaIntegral!bits;
    alias T = Fp!bits;
    alias S = I.S;
    alias B = I.B;
    I integral;
    S degrees, a, b;

    this(T nu)
    {
        integral = I(S.from(T(1)), 4096);
        degrees = a = S.from(nu);
        --a.exponent;
        b = integral.one;
        --b.exponent;
    }

    I.Point point(S z) const
    {
        return I.weightedPoint(degrees, z.multiply(z), integral.one);
    }

    // A reflected evaluation returns the mass between -z and z.
    bool betaMass(S z, out bool central, out I.Term mass)
    {
        return integral.directTail(a, b, point(z), central, mass);
    }
}

// With two degrees of freedom, F(x) = (1 + d/sqrt(d^2+2*s^2))/2,
// where d=x-location and s=scale. The same exact comparison rounds a
// probability or a quantile, including when d cannot fit on a single grid.
private struct StudentTwo(uint bits)
{
    import mir.stat.internal.fp_shape_transform: ShapeTransform;

    alias T = Fp!bits;
    alias B = ShapeTransform!(2 * bits);
    alias S = B.S;
    alias Term = B.Term;
    alias Rank = UInt!(bits + 128);
    enum block = Rank(1) << (bits - 1);
    enum end = (Rank(ulong.max) + 1) << (bits - 1);

    static Term term(T x)
    {
        return x.coefficient ? Term(S.from(x), x.sign, true) : Term.init;
    }

    static Term product(Term a, Term b)
    {
        return a.present && b.present
            ? Term(a.value.multiply(b.value), a.negative != b.negative, true) : Term.init;
    }

    // Each of the nine initial terms has at most 4*(bits+1) significant
    // bits. If the two highest exponents differ by more than four, the
    // largest magnitude exceeds the sum of all eight others. Otherwise
    // merging the leading pair costs at most five extra bits. All eight
    // merges therefore fit exactly in the 6*bits+256 working bits.
    static int sumSign(Term[9] terms)
    {
        size_t count;
        foreach (t; terms)
            if (t.present) terms[count++] = t;
        while (count > 1)
        {
            foreach (i; 0 .. 2)
            {
                size_t largest = i;
                foreach (j; i + 1 .. count)
                    if (terms[j].value.exponent > terms[largest].value.exponent)
                        largest = j;
                const tmp = terms[largest];
                terms[largest] = terms[i];
                terms[i] = tmp;
            }
            if (terms[0].value.exponent > terms[1].value.exponent + 4)
                break;
            const merged = B.combine(terms[0], terms[1]);
            terms[1] = terms[--count];
            terms[0] = merged.present ? merged : terms[--count];
        }
        return !count ? 0 : terms[0].negative ? -1 : 1;
    }

    Term minusLocation, scaleSquared, halfValue;

    this(T location, T scale)
    {
        minusLocation = term(location);
        minusLocation.negative = !minusLocation.negative;
        scaleSquared = product(term(scale), term(scale));
        halfValue = term(half!bits());
    }

    // Sign of F(x)-p. Once d and p-1/2 have the same sign, compare
    // 2*p*(1-p)*(d^2+2*s^2) with s^2. Expand both differences so even
    // a location or probability far below the working grid participates.
    int compare(Term x, Term p) const
    {
        Term minusHalf = halfValue;
        minusHalf.negative = true;
        const centered = B.combine(p, minusHalf);
        const d = B.combine(x, minusLocation);
        const dSide = !d.present ? 0 : d.negative ? -1 : 1;
        const pSide = !centered.present ? 0 : centered.negative ? -1 : 1;
        if (dSide != pSide) return dSide < pSide ? -1 : 1;
        if (!dSide) return 0;

        auto twiceP = p;
        ++twiceP.value.exponent;
        auto minusTwicePSquared = product(twiceP, p);
        minusTwicePSquared.negative = true;
        auto cross = product(x, minusLocation);
        ++cross.value.exponent;
        Term twiceScaleSquared = scaleSquared;
        ++twiceScaleSquared.value.exponent;
        Term[9] terms;
        foreach (i, factor; [product(x, x), cross, product(minusLocation, minusLocation), twiceScaleSquared])
        {
            terms[2 * i] = product(factor, twiceP);
            terms[2 * i + 1] = product(factor, minusTwicePSquared);
        }
        terms[8] = scaleSquared;
        terms[8].negative = true;
        return dSide * sumSign(terms);
    }

    // Rank zero is zero. The virtual rank end is one ulp above the
    // largest finite value and supplies the overflow midpoint.
    static S grid(Rank rank, bool midpoint)
    {
        const binade = rank >> (bits - 1);
        auto c = (rank & (block - 1)).toSize!(bits + 64);
        if (binade) c |= block.toSize!(bits + 64);
        if (midpoint) c = (c << 1) + 1;
        auto v = S.from(Fp!(bits + 64)(c));
        v.exponent -= 1UL << 63;
        if (binade) v.exponent += binade.toSize!128 - 1;
        if (midpoint) --v.exponent;
        return v;
    }

    T rounded(bool inverse)(Term fixed, Rank high, bool negative) const
    {
        Rank low;
        while (low < high)
        {
            const rank = (low + high) >> 1;
            const mid = Term(grid(rank, true), negative, true);
            static if (inverse)
            {
                const cmp = compare(mid, fixed);
                const side = negative ? cmp : -cmp;
            }
            else const side = compare(fixed, mid);
            if (side > 0 || (!side && rank.bt(0))) low = rank + 1;
            else high = rank;
        }
        if (low == end) return T(negative, long.max, UInt!bits(0));
        auto result = low ? grid(low, false).finish!bits : T(0);
        result.sign = negative;
        return result;
    }

    static T tail(bool upper)(T x, T location, T scale)
    {
        // Reflection makes the strict upper tail a direct lower tail.
        static if (upper)
        {
            x.sign = !x.sign;
            location.sign = !location.sign;
        }
        const t = StudentTwo(location, scale);
        const observation = term(x);
        const oneRank = (Rank(1UL << 63) + 2 - bits) << (bits - 1);
        return t.rounded!false(observation, oneRank, false);
    }

    static T quantile(T p, T location, T scale)
    {
        const t = StudentTwo(location, scale);
        const probability = term(p);
        const atZero = t.compare(Term.init, probability);
        if (!atZero) return T(0);
        const negative = atZero > 0;
        return t.rounded!true(probability, end, negative);
    }
}



package(mir.stat) Fp!bits studentTail(bool upper, uint bits)(Fp!bits x, Fp!bits nu,
    Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    alias C = StudentIntegral!bits;
    if (x.isNaN) return x;
    if (x.isInfinity) return T(x.sign == upper ? 1 : 0);
    const side = position(x, location);
    if (!side) return half!bits();
    if (!position(nu, T(2))) return StudentTwo!bits.tail!upper(x, location, scale);
    auto t = C(nu);
    const z = distance!(C.B.work)(x, location).divide(C.S.from(scale));
    bool central;
    C.I.Term mass;
    if (!t.betaMass(z, central, mass)) return T(false, long.max, UInt!bits(1));
    const small = upper == (side > 0);
    const base = central ? half!bits() : T(small ? 0 : 1);
    if (!mass.present) return base;
    --mass.value.exponent;
    return addOffset!bits(base, mass.value, central == small);
}

package(mir.stat) Fp!bits studentDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits nu,
    Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    alias C = StudentIntegral!bits;
    alias I = C.I;
    if (x.isNaN) return x;
    if (x.isInfinity) return logarithmic ? T(true, long.max, UInt!bits(0)) : T(0);
    auto t = C(nu);
    const one = t.integral.one;
    const anchor = t.point(one);
    // At |z|=1 the beta prefactor equals the standard t density.
    // Recover the modal height there, then apply the scale and decay.
    const power = t.a.add(t.b);
    auto height = C.B.combineThree(t.integral.front(t.a, t.b, anchor),
        I.pos(power.multiply(anchor.negLogX)), I.signedValue(C.B.logarithm(C.S.from(scale)), true));
    if (position(x, location))
    {
        const z = distance!(C.B.work)(x, location).divide(C.S.from(scale));
        height = C.B.combine(height, I.neg(power.multiply(t.point(z).negLogX)));
    }
    if (!height.present || height.value.exponent + bits + 48 < exponentBias)
        return logarithmic ? T(false, long.max, UInt!bits(1)) : T(1);
    return t.integral.finishDensity!logarithmic(height);
}

package(mir.stat) Fp!bits studentQuantile(uint bits)(Fp!bits p, Fp!bits nu,
    Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    alias C = StudentIntegral!bits;
    alias I = C.I;
    alias S = C.S;
    const one = T(1);
    if (!p.coefficient || !position(p, one)) return T(!p.coefficient, long.max, UInt!bits(0));
    const midpoint = half!bits();
    const side = position(p, midpoint);
    if (!side) return location;
    if (!position(nu, T(2))) return StudentTwo!bits.quantile(p, location, scale);
    auto t = C(nu);
    auto tail = side < 0 ? S.from(p) : distance!(C.B.work)(one, p);
    auto center = distance!(C.B.work)(midpoint, p);
    ++tail.exponent;
    ++center.exponent;
    const multiplier = S.from(scale);
    T finish(S z) { return addOffset!bits(location, z.multiply(multiplier), side < 0); }

    t.integral.prepareQuantile(t.a, t.b);

    // Compare like quantities: a direct two-sided tail or a direct
    // central mass. Both targets retain the input probability's bits.
    bool below(S z, out bool lower)
    {
        I.Term mass;
        bool central;
        if (!t.betaMass(z, central, mass)) return false;
        const goal = central ? center : tail;
        const diff = C.B.combine(mass, I.neg(goal));
        if (!diff.present || diff.value.exponent + bits + 80 < goal.exponent) return false;
        lower = diff.negative == central;
        return true;
    }

    const nan = T(false, long.max, UInt!bits(1));
    const unit = t.integral.one;
    S lo = unit, hi = unit;
    bool upward;
    if (!below(unit, upward)) return nan;
    // Two finite Fp magnitudes and their ratio span fewer than 2^65
    // binades. Past that bound the affine result saturates or is location.
    for (uint k = 0; k <= 65; ++k)
    {
        const step = UInt!128(1) << k;
        S trial = unit;
        if (upward) trial.exponent += step;
        else trial.exponent -= step;

        bool lower;
        if (!below(trial, lower)) return nan;
        if (lower) lo = trial;
        else hi = trial;
        if (lower != upward) break;
        if (k == 65) return finish(trial);
    }
    // Bisect binades before significands. Stop once both ends round to
    // the same location-scale output, including cancellation at zero.
    for (;;)
    {
        const left = finish(lo), right = finish(hi);
        if (left == right) return left;
        S trial;
        if (hi.exponent - lo.exponent > UInt!128(1))
        {
            trial = unit;
            trial.exponent = lo.exponent + ((hi.exponent - lo.exponent) >> 1);
        }
        else
        {
            trial = lo.add(hi);
            --trial.exponent;
            if (trial == lo || trial == hi) return nan;
        }
        bool lower;
        if (!below(trial, lower)) return nan;
        if (lower) lo = trial;
        else hi = trial;
    }
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!32;
    alias U = UInt!32;
    const adjacent = T(false, -31, U(0x80000001u));
    const scale = T(false, -30, U(1));
    assert(studentTail!true(adjacent, T(2), T(1), scale) ==
        T(false, -33, U(0xaaaaaaabu)));
}
