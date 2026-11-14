/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson_mode;

import mir.bignum.fp: Fp;
import mir.bignum.integer: BigInt;
import mir.stat.distribution.poisson: PoissonTailResult, PoissonTailStatus,
    PoissonInverseResult, PoissonInverseStatus;
import mir.stat.internal.fp_poisson_tails: roundedPoissonRatio;

// Relative masses start at 1 at floor(lambda). Both outward ratios are <= 1,
// so every weight, even rounded upward, is <= S = 2^(bits+128).
package(mir.stat) struct PoissonMode(uint bits)
{
    private:
    alias T = Fp!bits;
    enum precision = bits + 128;
    // The visited sum is <= 1024*S. Each omitted-mass bound is at most
    // S*(mode+2), since the first omitted ratio is <= 1 and the geometric
    // denominator is >= 1/(mode+2). Allow two such bounds and upward rounding.
    // A total or prefix thus fits in precision+67 bits (size_t <= 64 bits).
    enum weightLimbs = (precision + 67 + 63) / 64;
    alias Weight = BigInt!weightLimbs;
    // Write lambda = numerator / 2^shift. For lambda >= 1, shift < bits.
    // numerator fits either the Fp coefficient or a size_t (integer rate).
    // A recurrence divisor may additionally include one size_t index.
    enum factorLimbs = (bits + 64 + 63) / 64;
    alias Factor = BigInt!factorLimbs;
    alias Integer = BigInt!(weightLimbs + factorLimbs);

    struct Interval { Integer lo, hi; }
    Integer numerator;
    size_t shift, mode, first, last;
    Interval left, right, lower, upper;

    @safe pure nothrow @nogc:

    static void add(ref Integer sum, ref const Integer value)
    {
        const overflow = sum.opOpAssign!"+"(value);
        assert(!overflow);
    }

    static void multiplyIndex(ref Integer value, size_t factor)
    {
        const overflow = value.opOpAssign!"*"(factor);
        assert(!overflow);
    }

    static Integer product(ref const Integer weight, ref const Integer factor)
    {
        Weight x;
        Factor y;
        x = weight;
        y = factor;
        Integer result;
        result.multiply(x, y);
        return result;
    }

    static void divide(ref Integer value, ref const Integer divisor, bool roundUp)
    {
        bool discarded;
        if (divisor.length == 1 && !(divisor.data[0] & (divisor.data[0] - 1))) {
            // Exact binary scaling, including division by one.
            discarded = value.length && (value.data[0] & (divisor.data[0] - 1));
            value >>= divisor.bitLength - 1;
        } else {
            Integer remainder;
            value.divMod(divisor, value, remainder);
            discarded = remainder.length != 0;
        }
        if (roundUp && discarded) {
            Integer one = 1;
            add(value, one);
        }
    }

    static void scale(ref Interval weight, ref const Integer factor, ref const Integer divisor)
    {
        weight.lo = product(weight.lo, factor);
        weight.hi = product(weight.hi, factor);
        divide(weight.lo, divisor, false);
        divide(weight.hi, divisor, true);
    }

    Integer scaledIndex(size_t j) const
    {
        Integer value = j;
        value <<= shift;
        return value;
    }

    void down(ref Interval weight, size_t j) const
    {
        auto factor = scaledIndex(j);
        scale(weight, factor, numerator);
    }

    void up(ref Interval weight, size_t j) const
    {
        auto divisor = scaledIndex(j + 1);
        scale(weight, numerator, divisor);
    }

    static void accumulate(ref Interval sum, ref const Interval weight)
    {
        add(sum.lo, weight.lo);
        add(sum.hi, weight.hi);
    }

    static Interval unit()
    {
        Integer value = 1;
        value <<= precision;
        return Interval(value, value);
    }

    bool initialize(T rate, size_t limit)
    {
        assert(limit && limit <= 1024);
        numerator = Integer(rate.coefficient);
        auto whole = numerator;
        if (rate.exponent >= 0) {
            if (rate.exponent >= size_t.sizeof * 8 ||
                whole.bitLength > size_t.sizeof * 8 - rate.exponent)
                return false;
            whole <<= cast(size_t)rate.exponent;
            numerator = whole;
        } else {
            const distance = 0UL - cast(ulong)rate.exponent;
            assert(distance < whole.bitLength); // caller requires rate >= limit >= 1
            shift = cast(size_t)distance;
            whole >>= shift;
            if (whole.bitLength > size_t.sizeof * 8) return false;
        }
        mode = whole.data[0];
        // At most limit-1 steps to the right; the remainder uses last+2.
        if (mode > size_t.max - limit - 1) return false;
        first = last = mode;
        left = right = unit();
        return true;
    }

    bool sumWindow(size_t k, size_t limit)
    {
        // Reject counts even the furthest possible window cannot reach.
        if ((k < mode ? mode - k : k - mode) >= limit) return false;
        accumulate(k < mode ? upper : lower, left);
        foreach (i; 1 .. limit) {
            // Favor the heavier endpoint. Certification depends only on the
            // bounds for the resulting window, so ties can favor either side.
            if (first && (left.hi > right.hi ||
                (left.hi == right.hi && mode - first <= last - mode))) {
                down(left, first);
                --first;
                accumulate(first <= k ? lower : upper, left);
            } else {
                up(right, last);
                ++last;
                accumulate(last <= k ? lower : upper, right);
            }
        }
        if (k < first || k > last) return false;

        // At each end, bound the first omitted term and its successors by
        // a geometric series. Moving outward only decreases successive ratios.
        if (first) {
            // w[first] * first / (lambda-first+1).
            auto factor = scaledIndex(first);
            auto remainder = product(left.hi, factor);
            auto divisor = numerator;
            auto previous = scaledIndex(first - 1);
            divisor -= previous;
            divide(remainder, divisor, true);
            add(lower.hi, remainder);
        }
        // w[last] * lambda*(last+2) / ((last+1)*(last+2-lambda)).
        // The numerator before division fits precision + max(bits,64) + 65
        // bits; the denominator fits bits + 128 bits. Both fit Integer.
        auto remainder = product(right.hi, numerator);
        multiplyIndex(remainder, last + 2);
        auto divisor = scaledIndex(last + 2);
        divisor -= numerator;
        multiplyIndex(divisor, last + 1);
        divide(remainder, divisor, true);
        add(upper.hi, remainder);
        return true;
    }

    // ceil(total*q) = 1 + floor((total*coefficient - 1) / 2^(-exponent)).
    static Integer target(ref const Integer total, T q)
    {
        Integer coefficient = Integer(q.coefficient);
        auto rank = product(total, coefficient);
        Integer one = 1;
        rank -= one;
        const distance = 0UL - cast(ulong)q.exponent;
        if (distance >= rank.bitLength) return one;
        rank >>= cast(size_t)distance;
        add(rank, one);
        return rank;
    }

    public:
    static PoissonTailResult!T tails(size_t k, T rate, size_t limit)
    {
        alias Result = PoissonTailResult!T;
        PoissonMode window;
        if (!window.initialize(rate, limit) || !window.sumWindow(k, limit))
            return Result.init;

        // A/(A+B) increases with A and decreases with B. Complement the
        // interval endpoints before rounding either probability.
        auto d1 = window.lower.lo, d2 = window.lower.hi;
        add(d1, window.upper.hi);
        add(d2, window.upper.lo);
        const cdf = roundedPoissonRatio!bits(window.lower.lo, d1);
        const ccdf = roundedPoissonRatio!bits(window.upper.lo, d2);
        if (cdf != roundedPoissonRatio!bits(window.lower.hi, d2) ||
            ccdf != roundedPoissonRatio!bits(window.upper.hi, d1))
            return Result.init;
        return Result(PoissonTailStatus.found, cdf, ccdf);
    }

    static PoissonInverseResult inverse(T q, T rate, size_t limit)
    {
        alias Result = PoissonInverseResult;
        enum found = PoissonInverseStatus.found;
        PoissonMode window;
        if (!window.initialize(rate, limit)) return Result.init;
        const covered = window.sumWindow(window.mode, limit);
        assert(covered);

        auto total = window.lower;
        accumulate(total, window.upper);
        const reached = target(total.hi, q);
        const below = target(total.lo, q);
        auto prefix = window.lower, weight = unit();
        size_t k = window.mode;
        // Replay from the mode using the same outward bounds. For a move
        // left, subtract the upper weight from the lower prefix and vice versa.
        if (prefix.lo >= reached) {
            for (;;) {
                prefix.hi -= weight.lo;
                if (prefix.hi < below) return Result(found, k);
                if (k == window.first || prefix.lo < weight.hi) return Result.init;
                prefix.lo -= weight.hi;
                if (prefix.lo < reached) return Result.init;
                window.down(weight, k);
                --k;
            }
        }
        while (prefix.hi < below && k < window.last) {
            window.up(weight, k);
            ++k;
            accumulate(prefix, weight);
            if (prefix.lo >= reached) return Result(found, k);
        }
        return Result.init;
    }
}
