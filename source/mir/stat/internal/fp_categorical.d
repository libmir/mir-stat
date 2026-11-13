/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_categorical;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

// Sum nonnegative finite represented probabilities.
// Visit distinct input exponents in ascending order without allocating or
// modifying the input. Cost is O(n * number of distinct exponents).
package(mir.stat) struct CategoricalSum(uint bits)
{
    Fp!bits value;
    // Sign of rounded value minus the exact sum; zero denotes exact equality.
    int rounding;
}

package(mir.stat) CategoricalSum!bits categoricalSum(uint bits, R)(scope const R p,
    size_t first, size_t last) @safe pure nothrow @nogc
{
    enum guard = bits + 64;
    enum width = 2 * bits + 128;
    alias Integer = UInt!width;
    alias T = Fp!bits;
    long exponent = long.max;
    foreach (i; first .. last)
        if (p[i].coefficient && p[i].exponent < exponent)
            exponent = p[i].exponent;
    if (exponent == long.max)
        return CategoricalSum!bits(T(0), 0);
    Integer sum;
    bool sticky;
    while (true) {
        long next = long.max;
        foreach (i; first .. last) {
            if (!p[i].coefficient) continue;
            if (p[i].exponent == exponent) {
                auto term = p[i].coefficient.toSize!width;
                term <<= guard;
                sum += term;
            } else if (p[i].exponent > exponent && p[i].exponent < next) {
                next = p[i].exponent;
            }
        }
        if (next == long.max) break;
        const gap = cast(ulong)next - cast(ulong)exponent;
        if (gap >= width) {
            sticky |= !!sum;
            sum = Integer(0);
        } else {
            sticky |= !!sum && sum.cttz < gap;
            sum >>= cast(size_t)gap;
        }
        exponent = next;
    }
    // Future terms always align to integer units of the accumulator. Any
    // discarded fraction remains strictly below one unit; it cannot carry.
    // At most size_t.max coefficients fit in width, including guard bits.
    const used = width - sum.ctlz;
    size_t shift = used > bits ? used - bits : 0;
    if (shift < guard && exponent < long.min + cast(long)(guard - shift))
        shift = guard - cast(size_t)(cast(ulong)exponent - cast(ulong)long.min);
    auto coefficient = sum >> shift;
    bool remainder = sticky || (shift && sum.cttz < shift);
    bool up;
    if (shift && sum.bt(shift - 1))
        up = sticky || sum.cttz < shift - 1 || coefficient.bt(0);
    if (up) coefficient += 1;
    if (width - coefficient.ctlz > bits) {
        coefficient >>= 1;
        ++shift;
    }
    const resultExponent = shift >= guard
        ? exponent + cast(long)(shift - guard)
        : exponent - cast(long)(guard - shift);
    return CategoricalSum!bits(T(false, resultExponent, coefficient.toSize!bits),
        remainder ? (up ? 1 : -1) : 0);
}

// Compare the exact sum to a represented threshold, including rounded ties.
private int compare(uint bits)(CategoricalSum!bits sum, Fp!bits threshold)
    @safe pure nothrow @nogc
{
    import mir.stat.internal.fp_probability: fpProbabilityLess;
    if (fpProbabilityLess(sum.value, threshold)) return -1;
    if (fpProbabilityLess(threshold, sum.value)) return 1;
    return -sum.rounding;
}

package(mir.stat) bool categoricalValid(uint bits, R)(scope const R p)
    @safe pure nothrow @nogc
{
    import mir.stat.internal.fp_probability: fpProbabilityValid;
    if (!p.length) return false;
    foreach (i; 0 .. p.length)
        if (!fpProbabilityValid(p[i])) return false;
    // Match approxEqual(total, 1): both defaults are exactly 2^-20.
    enum lower = Fp!bits(false, -20, UInt!bits((1u << 20) - 1));
    enum upper = Fp!bits(false, -20, UInt!bits((1u << 20) + 1));
    const total = categoricalSum!bits(p, 0, p.length);
    return compare(total, lower) >= 0 && compare(total, upper) <= 0;
}

package(mir.stat) size_t categoricalInverse(uint bits, R)(Fp!bits q, scope const R p)
    @safe pure nothrow @nogc
{
    if (!q.coefficient) return 0;
    size_t lower, upper = p.length;
    // Nonnegative exact prefix sums are monotone, including zero categories.
    // A sum rounding upward to q has NOT reached the represented query.
    while (lower < upper) {
        const middle = lower + (upper - lower) / 2;
        const prefix = categoricalSum!bits(p, 0, middle + 1);
        const reached = compare(prefix, q) >= 0;
        if (reached) upper = middle;
        else lower = middle + 1;
    }
    if (lower < p.length) return lower;
    // Match the supplied-total endpoint convention: never select a trailing
    // zero merely because a total slightly below one did not reach q.
    foreach_reverse (i; 0 .. p.length)
        if (p[i].coefficient) return i;
    assert(0, "a valid categorical distribution has positive total");
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64, 128]) {{
        alias T = Fp!bits;
        alias U = UInt!bits;
        enum half = T(false, -cast(long)bits, U(1) << (bits - 1));
        enum below = T(false, -cast(long)bits - 1, U.max);
        enum quarterGap = T(false, -cast(long)bits - 2, U(1));
        T[4] p = [below, quarterGap, half, quarterGap];
        const prefix = categoricalSum!bits(p[], 0, 2);
        assert(prefix.value == half && prefix.rounding == 1);
        const total = categoricalSum!bits(p[], 0, 4);
        assert(total.value == T(1) && total.rounding == 0);
        assert(categoricalInverse(half, p[]) == 2);
        assert(categoricalInverse(below, p[]) == 0);
        assert(categoricalInverse(T(1), p[]) == 3);
        T[3] zeros = [T(0), T(1), T(0)];
        assert(categoricalInverse(T(0), zeros[]) == 0);
        assert(categoricalInverse(half, zeros[]) == 1);
        assert(categoricalInverse(T(1), zeros[]) == 1);
        T[2] floor = [T(false, long.min, U(1)), T(false, long.min, U(2))];
        const small = categoricalSum!bits(floor[], 0, 2);
        assert(small.value == T(false, long.min, U(3)) && small.rounding == 0);
        T[2] distant = [half, T(false, long.min, U(1))];
        const rounded = categoricalSum!bits(distant[], 0, 2);
        assert(rounded.value == half && rounded.rounding == -1);
        assert(categoricalInverse(half, distant[]) == 0);
        assert(categoricalSum!bits(p[], 0, 0).value == T(0));
    }}
}
