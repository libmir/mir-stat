/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson_tails;

import mir.bignum.fixed: UInt;
import mir.bignum.integer: BigInt;
import mir.bignum.fp: Fp;
import mir.stat.distribution.poisson: PoissonTailResult, PoissonTailStatus,
    PoissonInverseResult, PoissonInverseStatus;
import mir.stat.internal.fp_probability: fpProbabilityLess;

// Directed fixed-point bounds for the unnormalized masses lambda^j/j!.
// Every upward rounding adds less than one grid unit. Since lambda < 1024,
// each propagated error is bounded by exp(1024), as is the mass series.
// With at most 1024 steps, sums are below (2^precision+1024)*exp(1024).
// The geometric remainder multiplies by at most 1025; allow another factor
// of two for sums. This needs fewer than precision + 1500 bits. The 2048-bit
// allowance also covers normalization, and the extra output bits cover both
// multiplication by the rate coefficient and exact rounding of the ratios.
private struct PoissonIntervals(uint bits)
{
    alias T = Fp!bits;
    alias Result = PoissonTailResult!T;
    enum precision = bits + 128;
    alias Integer = BigInt!((precision + bits + 2048 + 63) / 64);

    @safe pure nothrow @nogc:

    private static void add(ref Integer a, ref const Integer b)
    {
        const overflow = a.opOpAssign!"+"(b);
        assert(!overflow);
    }

    private static void multiply(ref Integer a, size_t b)
    {
        const overflow = a.opOpAssign!"*"(b);
        assert(!overflow);
    }

    // Return whether division discarded a nonzero remainder.
    private static bool divide(ref Integer a, uint b)
    {
        const remainder = a.opOpAssign!"/"(b);
        a.normalize();
        return remainder != 0;
    }

    // Exact rational rounding, including bounds at zero and one.
    private static T roundedRatio(ref const Integer numerator, ref const Integer denominator)
    {
        if (!numerator.length) return T(0);
        if (numerator == denominator) return T(1);
        assert(numerator < denominator);
        const offset = denominator.bitLength - numerator.bitLength;
        Integer quotient = numerator;
        quotient <<= offset;
        const extra = bits - 1 + (quotient < denominator);
        quotient <<= extra;
        Integer remainder;
        quotient.divMod(denominator, quotient, remainder);
        remainder <<= 1;
        UInt!(bits + 64) coefficient;
        coefficient.data[0 .. quotient.length] = quotient.data[0 .. quotient.length];
        if (remainder > denominator || (remainder == denominator && coefficient.bt(0)))
            coefficient += 1;
        auto result = T(coefficient);
        result.exponent -= cast(long)(offset + extra);
        return result;
    }

    private static void advance(ref Integer value,
        ref const BigInt!((bits + 63) / 64) coefficient, long exponent, uint divisor, bool up)
    {
        BigInt!((precision + 2048 + 63) / 64) weight;
        assert(value.bitLength <= precision + 2048);
        weight = value;
        Integer product;
        product.multiply(weight, coefficient);
        bool sticky;
        if (exponent >= 0) {
            // lambda <1024, so a positive binary exponent is at most 9.
            assert(exponent < 10);
            product <<= cast(size_t)exponent;
            sticky = divide(product, divisor);
        } else {
            // floor(floor(a/d)/2^s) = floor(a/(d*2^s)). Retain both
            // sources of discarded bits to obtain the upward bound too.
            sticky = divide(product, divisor);
            const shift = 0UL - cast(ulong)exponent;
            if (shift >= product.bitLength) {
                sticky = sticky || product.length != 0;
                product = 0;
            } else {
                Integer original = product;
                product >>= cast(size_t)shift;
                Integer restored = product;
                restored <<= cast(size_t)shift;
                sticky = sticky || original != restored;
            }
        }
        if (up && sticky) {
            Integer one = Integer(1);
            add(product, one);
        }
        value = product;
    }

    // Integer n reaches q*d exactly when n >= ceil(q*d). Keeping the
    // target as an integer avoids rounding a CDF before the comparison.
    private static Integer targetRank(ref const Integer denominator, T q)
    {
        BigInt!((precision + 2048 + 63) / 64) weight;
        weight = denominator;
        auto coefficient = BigInt!((bits + 63) / 64)(q.coefficient);
        Integer rank;
        rank.multiply(weight, coefficient);
        Integer one = Integer(1);
        rank -= one;
        const shift = 0UL - cast(ulong)q.exponent;
        if (shift >= rank.bitLength)
            return one;
        rank >>= cast(size_t)shift;
        add(rank, one);
        return rank;
    }

    static PoissonInverseResult quantile(T q, T lambda, uint ceiling, size_t limit)
    {
        alias R = PoissonInverseResult;
        assert(limit && limit <= 1024 && ceiling <= limit);
        auto coefficient = BigInt!((bits + 63) / 64)(lambda.coefficient);
        Integer lo = Integer(1);
        lo <<= precision;
        Integer hi = lo;
        Integer totalLo, totalHi;
        foreach (m; 0 .. limit) {
            add(totalLo, lo);
            add(totalHi, hi);
            advance(lo, coefficient, lambda.exponent, cast(uint)m + 1, false);
            advance(hi, coefficient, lambda.exponent, cast(uint)m + 1, true);
        }
        // hi bounds the first omitted mass. All later successive ratios
        // are at most ceil(lambda)/(limit+1).
        multiply(hi, limit + 1);
        if (divide(hi, cast(uint)(limit + 1 - ceiling))) {
            Integer one = Integer(1);
            add(hi, one);
        }
        add(totalHi, hi);
        const reachedRank = targetRank(totalHi, q);
        const belowRank = targetRank(totalLo, q);

        // A second pass avoids storing a table of interval weights. Before
        // advancing k, prove CDF(k)<q; return only once CDF(k)>=q is proven.
        lo = 1;
        lo <<= precision;
        hi = lo;
        Integer sumLo, sumHi;
        for (size_t k = 0; ; ++k) {
            add(sumLo, lo);
            add(sumHi, hi);
            if (sumLo >= reachedRank)
                return R(PoissonInverseStatus.found, k);
            if (sumHi >= belowRank || k + 1 == limit)
                return R.init;
            advance(lo, coefficient, lambda.exponent, cast(uint)k + 1, false);
            advance(hi, coefficient, lambda.exponent, cast(uint)k + 1, true);
        }
    }

    static Result evaluate(size_t k, T lambda, uint ceiling, size_t limit)
    {
        assert(ceiling > 0 && ceiling <= 1024 && limit <= 1024 && k < limit);
        auto coefficient = BigInt!((bits + 63) / 64)(lambda.coefficient);
        Integer lo = Integer(1);
        lo <<= precision;
        Integer hi = lo;
        Integer lowerLo, lowerHi, upperLo, upperHi;
        foreach (m; 0 .. limit) {
            if (m <= k) {
                add(lowerLo, lo);
                add(lowerHi, hi);
            } else {
                add(upperLo, lo);
                add(upperHi, hi);
            }
            advance(lo, coefficient, lambda.exponent, cast(uint)m + 1, false);
            advance(hi, coefficient, lambda.exponent, cast(uint)m + 1, true);
            if (m < k || m + 2 <= ceiling)
                continue;

            // After the next mass, all successive ratios are at most
            // ceil(lambda)/(m+2). Sum the resulting geometric upper bound.
            Integer remainder = hi;
            multiply(remainder, m + 2);
            if (divide(remainder, cast(uint)(m + 2 - ceiling))) {
                Integer one = Integer(1);
                add(remainder, one);
            }
            Integer upper = upperHi;
            add(upper, remainder);
            Integer lowerDenominator = lowerLo;
            add(lowerDenominator, upper);
            Integer upperDenominator = lowerHi;
            add(upperDenominator, upperLo);

            // A/(A+B) increases with A and decreases with B. Keep that
            // dependency when bounding the normalized tails, independently.
            const cdfLo = roundedRatio(lowerLo, lowerDenominator);
            const cdfHi = roundedRatio(lowerHi, upperDenominator);
            const ccdfLo = roundedRatio(upperLo, upperDenominator);
            const ccdfHi = roundedRatio(upper, lowerDenominator);
            if (cdfLo == cdfHi && ccdfLo == ccdfHi)
                return Result(PoissonTailStatus.found, cdfLo, ccdfLo);
        }
        return Result.init;
    }
}

package(mir.stat) @safe pure nothrow @nogc
PoissonTailResult!(Fp!bits) boundedPoissonTails(uint bits)(size_t k, Fp!bits lambda, size_t maxTerms)
{
    alias T = Fp!bits;
    alias Result = PoissonTailResult!T;
    if (lambda.exponent == long.max)
        return Result(PoissonTailStatus.found, T(0), T(1));
    // For 0 < lambda <= 2^(-bits-1), exp(-lambda) rounds to one:
    // 0 < 1-exp(-lambda) < lambda <= half the spacing below one.
    // Also 0 < lambda-(1-exp(-lambda)) < lambda^2/2. The spacing
    // immediately below a represented lambda is at least lambda*2^-bits
    // (including powers of two and exponent-floor values). Thus the strict
    // upper tail rounds to lambda. No series terms or fixed-point grid are
    // needed, even when lambda is close to the Fp exponent floor.
    enum smallRate = T(false, -cast(long)bits - 1, UInt!bits(1));
    if (!k && !fpProbabilityLess(smallRate, lambda)) {
        // Normalize without shifting the exponent below its floor. Raw Fp
        // constructors can supply equivalent, unnormalized coefficients.
        const room = cast(ulong)lambda.exponent - cast(ulong)long.min;
        const leading = lambda.coefficient.ctlz;
        const shift = room < leading ? cast(size_t)room : leading;
        lambda.coefficient <<= shift;
        lambda.exponent -= cast(long)shift;
        return Result(PoissonTailStatus.found, T(1), lambda);
    }
    const limit = maxTerms < 1024 ? maxTerms : 1024;
    if (k >= limit || !fpProbabilityLess(lambda, T(limit)))
        return Result.init;

    const ceiling = poissonRateCeiling(lambda);
    return PoissonIntervals!bits.evaluate(k, lambda, ceiling, limit);
}

// Called only for positive finite rates below 1024.
private @safe pure nothrow @nogc
uint poissonRateCeiling(uint bits)(Fp!bits lambda)
{
    // Compute ceil(lambda) from the represented value, including unnormalized
    // inputs and exponents too small to negate or narrow to size_t.
    auto coefficient = lambda.coefficient;
    uint ceiling;
    if (lambda.exponent >= 0) {
        coefficient <<= cast(size_t)lambda.exponent;
        ceiling = cast(uint)coefficient;
    } else {
        const shift = 0UL - cast(ulong)lambda.exponent;
        if (shift >= bits)
            ceiling = 1;
        else {
            const remainder = coefficient.cttz < shift;
            coefficient >>= cast(size_t)shift;
            ceiling = cast(uint)coefficient + remainder;
        }
    }
    return ceiling;
}

package(mir.stat) @safe pure nothrow @nogc
PoissonInverseResult boundedPoissonInverse(uint bits)(Fp!bits q, Fp!bits lambda, size_t maxTerms)
{
    alias T = Fp!bits;
    alias Result = PoissonInverseResult;
    alias Status = PoissonInverseStatus;
    if (!q.coefficient)
        return Result(Status.found, 0);
    if (!fpProbabilityLess(q, T(1)) || lambda.exponent == long.max)
        return Result(Status.found, size_t.max);
    const limit = maxTerms < 1024 ? maxTerms : 1024;
    if (!limit || !fpProbabilityLess(lambda, T(limit)))
        return Result.init;
    return PoissonIntervals!bits.quantile(q, lambda, poissonRateCeiling(lambda), limit);
}
