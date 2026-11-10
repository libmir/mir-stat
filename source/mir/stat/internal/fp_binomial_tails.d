/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_binomial_tails;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.bignum.integer: BigInt;
import mir.stat.distribution.binomial: BinomialTailResult, BinomialTailStatus;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;

// Round n*p approached strictly from below. Callers establish that the
// correction is smaller than half a coefficient unit and half an output ulp.
package(mir.stat) Fp!bits roundedBinomialProductBelow(uint bits)(size_t n, Fp!bits p)
    @safe pure nothrow @nogc
{
    alias T = Fp!bits;
    auto product = p.coefficient.toSize!(bits + 64);
    product *= UInt!(bits + 64)(n);
    const used = bits + 64 - product.ctlz;
    if (used > bits) {
        const shift = used - bits;
        const remainder = product - ((product >> shift) << shift);
        if (remainder == (UInt!(bits + 64)(1) << (shift - 1)))
            product -= 1;
    }
    auto result = T(product);
    if (result.exponent < 0 && p.exponent < long.min - result.exponent) {
        product <<= cast(size_t)(p.exponent - long.min);
        return T(false, long.min, product.toSize!bits);
    }
    result.exponent += p.exponent;
    return result;
}

private struct ExactTails(uint bits, uint capacity)
{
    alias T = Fp!bits;
    // 2^capacity needs one extra bit. An adjacent numerator may also
    // acquire a size_t factor before division by its binomial index.
    alias Integer = BigInt!(capacity / 64 + 2);

    @safe pure nothrow @nogc:

    private static T rounded(ref const Integer numerator, size_t scale)
    {
        const length = numerator.bitLength;
        const shift = length > bits ? length - bits : 0;
        Integer quotient = numerator;
        quotient >>= shift;
        UInt!(bits + 64) coefficient;
        version (LittleEndian)
            coefficient.data[0 .. quotient.length] = quotient.data[0 .. quotient.length];
        else
            foreach (i; 0 .. quotient.length)
                coefficient.data[$ - 1 - i] = quotient.data[i];

        if (shift) {
            enum wordBits = size_t.sizeof * 8;
            const halfBit = shift - 1;
            const index = halfBit / wordBits;
            const mask = size_t(1) << (halfBit % wordBits);
            const word = numerator.data[index];
            if (word & mask) {
                bool roundUp = (word & (mask - 1)) != 0 || coefficient.bt(0);
                foreach (limb; numerator.data[0 .. index])
                    roundUp |= limb != 0;
                if (roundUp)
                    coefficient += 1;
            }
        }
        auto result = T(coefficient);
        result.exponent -= cast(long)(scale - shift);
        return result;
    }

    static BinomialTailResult!T evaluate(size_t k, size_t n, UInt!bits odd, size_t s)
    {
        Integer a;
        // The reduced numerator is smaller than 2^s, even if the input
        // coefficient has more words than the selected workspace.
        enum wordBits = size_t.sizeof * 8;
        const words = (s + wordBits - 1) / wordBits;
        a.length = cast(uint)(words < odd.data.length ? words : odd.data.length);
        version (LittleEndian)
            a.data[0 .. a.length] = odd.data[0 .. a.length];
        else
            foreach (i; 0 .. a.length)
                a.data[i] = odd.data[$ - 1 - i];
        a.normalize();
        Integer b = Integer(1);
        b <<= s;
        b -= a;
        const lower = k + 1 <= n - k;
        Integer factor = lower ? b : a;
        Integer term = Integer(1);
        for (size_t power = n; power; ) {
            if (power & 1)
                term *= factor;
            power >>= 1;
            if (power)
                factor *= factor;
        }

        Integer sum = term;
        const terms = lower ? k + 1 : n - k;
        foreach (j; 1 .. terms) {
            // Remove one endpoint factor before advancing C(n,j-1) to C(n,j).
            if (lower) {
                term /= b;
                term *= a;
            } else {
                term /= a;
                term *= b;
            }
            const carry = term.opOpAssign!"*"(n - j + 1);
            assert(!carry);
            const remainder = term.opOpAssign!"/"(cast(uint)j);
            assert(!remainder);
            term.normalize();
            const overflow = sum.opOpAssign!"+"(term);
            assert(!overflow);
        }
        const scale = s * n;
        term = 1;
        term <<= scale;
        term -= sum;
        const direct = rounded(sum, scale);
        const complement = rounded(term, scale);
        return BinomialTailResult!T(BinomialTailStatus.found,
            lower ? direct : complement, lower ? complement : direct);
    }
}

package(mir.stat) BinomialTailResult!(Fp!bits) exactBinomialTails(uint bits)(
    size_t k, size_t n, Fp!bits p, size_t maxTerms, size_t maxScale)
    @safe pure nothrow @nogc
{
    alias T = Fp!bits;
    alias Result = BinomialTailResult!T;
    enum found = BinomialTailStatus.found;
    if (k == n || !p.coefficient)
        return Result(found, T(1), T(0));
    if (!fpProbabilityLess(p, T(1)))
        return Result(found, T(0), T(1));
    if (n == 1)
        return Result(found, fpProbabilityComplement(p), p);

    // P(X >= 2) <= C(n,2)*p^2. Below the exponent floor it rounds to zero.
    if (k && p.exponent < long.min / 2 - cast(long)bits - 64)
        return Result(found, T(1), T(0));
    // 0 < n*p - P(X > 0) <= C(n,2)*p^2. With n < 2^64 this bound
    // is less than half a unit of the input coefficient at this exponent.
    if (!k && p.exponent < -2 * cast(long)bits - 130)
        return Result(found, T(1), roundedBinomialProductBelow(n, p));

    const trailing = p.coefficient.cttz;
    const odd = p.coefficient >> trailing;
    const s = (0UL - cast(ulong)p.exponent) - trailing;
    if (k == n - 1 && odd == UInt!bits(1)) {
        // The strict tail is 2^(-s*n). At half the smallest quantum, zero wins.
        enum floorScale = 1UL << 63;
        if (s > floorScale / n)
            return Result(found, T(1), T(0));
        const scale = s * n;
        const exponent = scale == floorScale ? long.min : -cast(long)scale;
        // Normalize as far as the exponent floor permits.
        const shift = floorScale - scale < bits - 1
            ? cast(size_t)(floorScale - scale) : bits - 1;
        const upper = T(false, exponent - cast(long)shift, UInt!bits(1) << shift);
        return Result(found, fpProbabilityComplement(upper), upper);
    }

    const terms = k + 1 < n - k ? k + 1 : n - k;
    if (terms > maxTerms || terms > 1024 || s > maxScale / n || s > 65536 / n)
        return Result.init;
    const scale = cast(size_t)s * n;
    if (scale <= 4096)
        return ExactTails!(bits, 4096).evaluate(k, n, odd, cast(size_t)s);
    if (scale <= 16384)
        return ExactTails!(bits, 16384).evaluate(k, n, odd, cast(size_t)s);
    return ExactTails!(bits, 65536).evaluate(k, n, odd, cast(size_t)s);
}
