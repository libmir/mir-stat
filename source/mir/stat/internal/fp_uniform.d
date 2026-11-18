/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_uniform;

import mir.bignum.fixed: UInt, extendedMul;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityLess;
import mir.stat.internal.fp_scaled_probability: addExponent, exponentBias;

@safe pure nothrow @nogc:

// Compare finite represented values, including nonnormalized coefficients and signed zeros.
private int comparePoints(uint bits)(Fp!bits a, Fp!bits b)
{
    const an = a.sign && !!a.coefficient;
    const bn = b.sign && !!b.coefficient;
    if (an != bn)
        return an ? -1 : 1;
    const order = fpProbabilityLess(a, b) ? -1 : fpProbabilityLess(b, a) ? 1 : 0;
    return an ? -order : order;
}

package(mir.stat) bool uniformBoundsValid(uint bits)(Fp!bits lower, Fp!bits upper)
{
    return !lower.isSpecial && !upper.isSpecial && comparePoints(lower, upper) < 0;
}

package(mir.stat) bool uniformPointValid(uint bits)(Fp!bits x, Fp!bits lower, Fp!bits upper)
{
    return uniformBoundsValid(lower, upper) && !x.isSpecial &&
        comparePoints(lower, x) <= 0 && comparePoints(x, upper) <= 0;
}

private UInt!128 biased(long exponent)
{
    auto e = exponentBias;
    addExponent(e, exponent);
    return e;
}

// A comparison with a candidate uniform tail expands into at most four
// signed dyadics. Exponents are biased so even products at opposite Fp
// range limits can be ordered and aligned without signed overflow.
private struct Term(uint capacity)
{
    UInt!capacity coefficient;
    UInt!128 exponent;
    bool negative;

    @safe pure nothrow @nogc:

    static Term point(uint bits)(Fp!bits x, bool subtract)
    {
        return Term(x.coefficient.toSize!capacity, biased(x.exponent), x.sign != subtract);
    }

    static Term product(uint bits, size_t n)(Fp!bits x, UInt!n c, UInt!128 e, bool subtract)
    {
        addExponent(e, x.exponent);
        return Term(extendedMul(x.coefficient, c).toSize!capacity, e, x.sign != subtract);
    }

    UInt!128 top() const
    {
        return exponent + (capacity - coefficient.ctlz);
    }
}

// Merge the two largest magnitudes until the sign is known. When their
// leading bits differ by at least three places the other three terms
// together cannot reverse the sign. Otherwise alignment grows the wider
// coefficient by at most two bits plus one carry per merge. Callers allow
// at least nine spare bits beyond the widest initial term.
private int sumSign(uint capacity, size_t n)(Term!capacity[n] terms)
{
    static assert(n <= 4);
    size_t count;
    foreach (term; terms)
        if (term.coefficient)
            terms[count++] = term;
    while (count > 1)
    {
        foreach (i; 0 .. 2)
        {
            size_t largest = i;
            foreach (j; i + 1 .. count)
                if (terms[j].top > terms[largest].top)
                    largest = j;
            const tmp = terms[i];
            terms[i] = terms[largest];
            terms[largest] = tmp;
        }
        auto a = terms[0], b = terms[1];
        if (a.top > b.top + 2)
            return a.negative ? -1 : 1;
        if (a.exponent > b.exponent)
        {
            a.coefficient <<= cast(size_t)(a.exponent - b.exponent);
            a.exponent = b.exponent;
        }
        else
            b.coefficient <<= cast(size_t)(b.exponent - a.exponent);
        if (a.negative == b.negative)
            a.coefficient += b.coefficient;
        else if (a.coefficient >= b.coefficient)
            a.coefficient -= b.coefficient;
        else
        {
            a.coefficient = b.coefficient - a.coefficient;
            a.negative = b.negative;
        }
        terms[1] = terms[--count];
        terms[0] = a;
        if (!a.coefficient)
            terms[0] = terms[--count];
    }
    return count ? (terms[0].negative ? -1 : 1) : 0;
}

// compare(c,e) orders the nonnegative exact result against c * 2^(e-bias).
// Find its output binade, then its coefficient on the normal or subnormal
// grid. A final exact midpoint comparison also handles rounding to zero.
private Fp!bits roundUniform(uint bits, alias compare)()
{
    alias C = UInt!(bits + 64);
    auto lo = biased(long.min);
    auto hi = biased(long.max - 1);
    const leading = C(1) << (bits - 1);
    while (lo < hi)
    {
        const mid = lo + ((hi - lo + 1) >> 1);
        if (compare(leading, mid) >= 0)
            lo = mid;
        else
            hi = mid - 1;
    }
    C c;
    foreach_reverse (i; 0 .. bits)
    {
        const next = c | (C(1) << i);
        const order = compare(next, lo);
        if (order >= 0)
            c = next;
        if (order == 0)
            return Fp!bits(false, cast(long)cast(ulong)(lo - exponentBias), c.toSize!bits);
    }
    const midpoint = compare((c << 1) + 1, lo - 1);
    if (midpoint > 0 || (midpoint == 0 && c.bt(0)))
        c += 1;
    if (!c)
        return Fp!bits(0);
    if (c.bt(bits))
    {
        c >>= 1;
        lo += 1;
        // Tails are at most one and quantiles lie between the finite bounds.
        // A positive width is at least 2^long.min, so its reciprocal fits.
        // Log density rounds either a tiny displacement or the full width
        // in a wider type that can hold twice the largest input magnitude.
        assert(lo < biased(long.max));
    }
    return Fp!bits(false, cast(long)cast(ulong)(lo - exponentBias), c.toSize!bits);
}

// Evaluate (end-start)/(upper-lower) by comparing its exact numerator
// to a candidate times the exact denominator. Neither difference is
// materialized on an exponent-sized integer grid.
package(mir.stat) Fp!bits uniformRatio(uint bits)(Fp!bits start, Fp!bits end,
    Fp!bits lower, Fp!bits upper)
{
    if (comparePoints(start, end) == 0)
        return Fp!bits(0);
    if (comparePoints(start, lower) == 0 && comparePoints(end, upper) == 0)
        return Fp!bits(1);
    alias TermType = Term!(2 * bits + 64);
    int compare(UInt!(bits + 64) c, UInt!128 e)
    {
        return sumSign([TermType.point(end, false), TermType.point(start, true),
            TermType.product(upper, c, e, true), TermType.product(lower, c, e, false)]);
    }
    return roundUniform!(bits, compare)();
}

// Compare |width-1| with a candidate without rounding the width near one.
private int deviationOrder(uint bits, size_t n)(Fp!bits lower, Fp!bits upper,
    bool belowOne, UInt!n c, UInt!128 e)
{
    alias D = Term!(bits + n + 64);
    const order = sumSign([D.point(upper, false), D.point(lower, true),
        D.point(Fp!bits(1), true), D(c.toSize!(bits + n + 64), e, !belowOne)]);
    return belowOne ? -order : order;
}

package(mir.stat) Fp!bits uniformLogDensity(uint bits)(Fp!bits lower, Fp!bits upper)
{
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    alias T = Fp!bits;
    alias C = UInt!(bits + 64);
    const side = deviationOrder(lower, upper, false, C(0), exponentBias);
    if (!side)
        return T(0);
    const belowOne = side < 0;
    if (deviationOrder(lower, upper, belowOne, C(1), biased(-long(bits)-32)) <= 0)
    {
        // For |d| <= 2^(-bits-32), log(1+d) differs from d by at most
        // d^2/(2*(1-|d|)). Round the displacement directly on the output
        // grid, retaining the sign even when 1+d rounds to one.
        int compare(C c, UInt!128 e)
        {
            return deviationOrder(lower, upper, belowOne, c, e);
        }
        auto result = roundUniform!(bits, compare)();
        result.sign = !belowOne;
        return result;
    }

    // At the transition the subtraction implicit in log(width) can cost
    // bits+32 significant bits. Allow that loss plus 96 guard bits. The
    // wider Fp also accommodates a full width above the input type's ceiling.
    enum work = 2 * bits + 128;
    alias D = Term!(work + 128);
    int compareWidth(UInt!(work + 64) c, UInt!128 e)
    {
        return sumSign([D.point(upper, false), D.point(lower, true),
            D(c.toSize!(work + 128), e, true)]);
    }
    const width = roundUniform!(work, compareWidth)();
    auto result = cast(T) ProbabilityLog!work.logPositive(width);
    result.sign = !belowOne;
    return result;
}

// Compare lower + p*upper - p*lower with a signed candidate. Retaining
// the two full products lets cancellation expose a much smaller quantile.
package(mir.stat) Fp!bits uniformQuantile(uint bits)(Fp!bits p, Fp!bits lower, Fp!bits upper)
{
    alias T = Fp!bits;
    if (!p.coefficient)
        return lower;
    if (comparePoints(p, T(1)) == 0)
        return upper;
    enum capacity = 2 * bits + 64;
    alias D = Term!capacity;
    alias C = UInt!(bits + 64);
    const offset = D.point(lower, false);
    const e = biased(p.exponent);
    const high = D.product(upper, p.coefficient, e, false);
    const low = D.product(lower, p.coefficient, e, true);
    const side = sumSign([offset, high, low]);
    if (!side)
        return T(0);
    const negative = side < 0;
    int compare(C c, UInt!128 exponent)
    {
        const order = sumSign([offset, high, low, D(c.toSize!capacity, exponent, !negative)]);
        return negative ? -order : order;
    }
    auto result = roundUniform!(bits, compare)();
    result.sign = negative;
    return result;
}
