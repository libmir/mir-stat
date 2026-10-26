/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution).

The count $(D k) is the number of successes in $(D n) independent trials with
success probability $(D p). Counts use $(D size_t), with $(D k <= n), and
$(D p) must be finite and in $(D [0, 1]). Forward functions reject counts
outside this interval rather than extending their tails beyond the support.

Only PMF has an $(D Fp) overload, also available as $(LREF fp_binomialPMF).
CDF, CCDF, inverse CDF, and LPMF accept built-in floating-point probabilities.
LPMF has no algorithm selector. Normal and Poisson approximations trade
accuracy for speed and do not promise the direct algorithm's interior
probabilities, although their point masses and boundary returns are handled
explicitly.

Tables assume valid parameters except where an invalid input is listed.
Finite expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts does
not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.binomial;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid;
import mir.internal.utility: isFloatingPoint;
import mir.stat.distribution.poisson: PoissonAlgo;

/++
Algorithms used to calculate binomial distribution.

`BinomialAlgo.direct` can be more time-consuming for large values of the number
of events (`k`) or the number of trials (`n`). Additional algorithms are
provided to the user to choose the trade-off between running time and accuracy.

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
enum BinomialAlgo {
    /++
    Direct
    +/
    direct,
    /++
    Approximates binomial distribution with normal distribution. Generally a better approximation when
    `n > 20` and `p` is far from 0 or 1, but a variety of rules of thumb can help determine when it is 
    appropriate to use.
    +/
    approxNormal,
    /++
    Approximates binomial distribution with normal distribution (including continuity correction). More 
    accurate than `BinomialAlgo.approxNormal`.
    +/
    approxNormalContinuityCorrection,
    /++
    Approximates binomial distribution with poisson distribution (also requires specifying poissonAlgo). 
    Generally a better approximation when `n >= 20` and `p <= 0.05` or when `n >= 100` and `np <= 10`.
    +/
    approxPoisson
}

private
@safe pure nothrow @nogc
T binomialPMFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.direct)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: exp, pow;
    import mir.combinatorics: binomial;

    // All coefficients through n=64 fit in ulong. Near an endpoint, also try
    // the short direct product, checking the coefficient's overflow sentinel.
    // Limit trial counts too: raising rounded (1-p) to a large power can
    // amplify its rounding error even when the coefficient fits.
    if (n <= 64 || (n <= 1024 && (k < 32 || n - k < 32)))
    {
        const ulong coefficient = binomial(n, k);
        const T successes = pow(p, k);
        const T failures = pow(1 - p, n - k);
        if (coefficient != 0 && successes >= T.min_normal && failures >= T.min_normal)
            return coefficient * successes * failures;
    }
    return cast(T) exp(binomialLogMass(k, n, cast(real) p));
}

// The correction to Stirling's log-factorial formula, used only for n>=32.
// The first omitted term is below 1e-24 at that boundary.
private @safe pure nothrow @nogc
real binomialStirlingCorrection(real n)
{
    const real inv = 1 / n;
    const real inv2 = inv * inv;
    return inv * (1.0L / 12 + inv2 * (-1.0L / 360
        + inv2 * (1.0L / 1260 + inv2 * (-1.0L / 1680
        + inv2 * (1.0L / 1188 + inv2 * (-691.0L / 360360 + inv2 / 156))))));
}

// x*log(x/mean)+mean-x is small near x=mean. Evaluate a convergent
// series there instead of subtracting terms much larger than the answer.
private @safe pure nothrow @nogc
real binomialDeviance(real x, real mean)
{
    import mir.math.common: fabs, log;
    const real delta = x - mean;
    if (fabs(delta) < x * 0.5L)
    {
        const real v = delta / (x + mean);
        const real vv = v * v;
        real sum = delta * v;
        real term = 2 * x * v;
        // |v|<1/3. Stop once the next term no longer changes the sum.
        for (uint j = 1; j < 64; ++j)
        {
            term *= vv;
            const real next = sum + term / (2 * j + 1);
            if (next == sum)
                break;
            sum = next;
        }
        return sum;
    }
    // Separate logs also avoid overflow in x/mean for very small means.
    return x * (log(x) - log(mean)) + mean - x;
}

// Shared by the PMF fallback and large-count LPMF. Counts stay in size_t:
// narrowing k before exploiting symmetry can silently change the distribution.
private @safe pure nothrow @nogc
real binomialLogMass(size_t k, size_t n, real p)
{
    return binomialLogMassCounts(k, n - k, p);
}

// Separate counts also support negative-binomial masses when their sum exceeds
// size_t. Form the total in floating point only when integer addition would wrap.
package(mir.stat) @safe pure nothrow @nogc
real binomialLogMassCounts(size_t k, size_t failureCount, real p)
{
    import mir.math.common: log;
    import std.math: log1p;
    import std.math.constants: PI;
    if (p == 0)
        return k == 0 ? 0 : -real.infinity;
    if (p == 1)
        return failureCount == 0 ? 0 : -real.infinity;
    if (k == 0)
        return cast(real) failureCount * log1p(-p);
    if (failureCount == 0)
        return cast(real) k * log(p);

    const bool fits = k <= size_t.max - failureCount;
    const real total = fits ? cast(real)(k + failureCount) : cast(real)k + failureCount;
    const size_t smaller = k < failureCount ? k : failureCount;
    if (smaller < 32)
    {
        // Only a few factors are needed near either endpoint. This avoids
        // subtracting almost equal log-factorials when n is very large.
        real logCoefficient = 0;
        foreach (i; 0 .. smaller)
            logCoefficient += log(fits ? cast(real)(k + failureCount - i) : total - i)
                - log(cast(real)(i + 1));
        return logCoefficient + cast(real) k * log(p)
            + cast(real)failureCount * log1p(-p);
    }

    // Center the log mass on the expected successes and failures. The usual
    // log(coefficient)+k*log(p)+(n-k)*log(1-p) subtracts large terms near the
    // mode; this form keeps the small deviations separate from normalization.
    const real successes = k;
    const real failures = failureCount;
    return binomialStirlingCorrection(total)
        - binomialStirlingCorrection(successes) - binomialStirlingCorrection(failures)
        - binomialDeviance(successes, total * p)
        - binomialDeviance(failures, total * (1 - p))
        - 0.5L * (log(2 * PI) + log(successes) + log(failures) - log(total));
}

private
@safe pure nothrow @nogc
T binomialPMFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.approxNormal)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalPDF;

    return normalPDF(k, n * p, sqrt(n * p * (1 - p)));
}

private
@safe pure nothrow @nogc
T binomialPMFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    return normalCDF(cast(T) k + 0.5, n * p, sqrt(n * p * (1 - p))) - normalCDF(cast(T) k - 0.5, n * p, sqrt(n * p * (1 - p)));
}

private
@safe pure nothrow @nogc
T binomialPMFImpl(T, BinomialAlgo binomialAlgo, PoissonAlgo poissonAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.approxPoisson)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.poisson: poissonPMF;

    return poissonPMF!poissonAlgo(k, n * p);
}

/++
Computes the binomial probability mass function (PMF).

The direct algorithm retains its coefficient-and-powers calculation for small
trial counts when the intermediate values retain precision. Otherwise it uses
a logarithmic calculation, avoiding coefficient overflow and premature underflow.

Additional algorithms may be provided for calculating PMF that allow trading off
time and accuracy. If `approxPoisson` is provided, the default is `PoissonAlgo.gamma`

All algorithms return the exact point mass when `n = 0`, `p = 0`, or `p = 1`.

Returns the point probability $(D P(X == k)). The table applies to all
$(D BinomialAlgo) choices.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D n == 0) (so $(D k == 0))) $(TD $(D 1)))
    $(TR $(TD $(D p == 0)) $(TD $(D 1) at $(D k == 0); otherwise $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D 1) at $(D k == n); otherwise $(D 0)))
    $(TR $(TD $(D k > n)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    binomialAlgo = algorithm for calculating PMF (default: BinomialAlgo.direct)
    poissonAlgo = algorithm for poisson approximation (default: PoissonAlgo.gamma)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
@safe pure nothrow @nogc
template binomialPMF(BinomialAlgo binomialAlgo = BinomialAlgo.direct,
                     PoissonAlgo poissonAlgo = PoissonAlgo.gamma)
{
    /++
    Params:
    k = value to evaluate PMF (e.g. number of "heads")
    n = number of trials
    p = per-trial success probability in $(D [0, 1])
    +/
    T binomialPMF(T)(const size_t k, const size_t n, const T p)
        if (isFloatingPoint!T)
        in (k <= n, "k must be less than or equal to n")
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
    {
        static if (binomialAlgo != BinomialAlgo.direct) {
            // A point mass has no variance and needs no approximation.
            if (n == 0 || p == 0)
                return k == 0 ? 1 : 0;
            if (p == 1)
                return k == n ? 1 : 0;
        }
        static if (binomialAlgo != BinomialAlgo.approxPoisson)
            return binomialPMFImpl!(T, binomialAlgo)(k, n, p);
        else
            return binomialPMFImpl!(T, binomialAlgo, poissonAlgo)(k, n, p);
    }

    /++
    Extended-range and extended-precision PMF; see $(LREF fp_binomialPMF)
    for supported widths, rounding, and range limitations.
    Only the direct algorithm accepts an $(D Fp) probability.
    +/
    T binomialPMF(T)(const size_t k, const size_t n, const T p)
        if (is(T == Fp!size, size_t size) && binomialAlgo == BinomialAlgo.direct)
    {
        return fp_binomialPMF(k, n, p);
    }
}

/// ditto
@safe pure nothrow @nogc
template binomialPMF(string binomialAlgo, string poissonAlgo = "gamma")
{
    mixin("alias binomialPMF = .binomialPMF!(BinomialAlgo." ~ binomialAlgo ~ ", PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, pow;

    assert(4.binomialPMF(6, 2.0 / 3).approxEqual(15.0 * pow(2.0 / 3, 4) * pow(1.0 / 3, 2)));
    // For large values of `n` with `p` not too extreme, can approximate with normal distribution
    assert(550_000.binomialPMF!"approxNormal"(1_000_000, 0.55).approxEqual(0.0008019042));
    // Or closer with continuity correction
    assert(550_000.binomialPMF!"approxNormalContinuityCorrection"(1_000_000, 0.55).approxEqual(0.000801904));
    // Poisson approximation is better when `p` is low
    assert(10_000.binomialPMF!"approxPoisson"(1_000_000, 0.01).approxEqual(0.00398939));
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, pow;
    import mir.combinatorics: binomial;

    for (size_t i; i <= 5; i++) {
        assert(i.binomialPMF(5, 0.25).approxEqual(binomial(5, i) * pow(0.25, i) * pow(0.75, 5 - i)));
        assert(i.binomialPMF(5, 0.50).approxEqual(binomial(5, i) * pow(0.50, 5)));
        assert(i.binomialPMF(5, 0.75).approxEqual(binomial(5, i) * pow(0.75, i) * pow(0.25, 5 - i)));
    }
}

// Coefficient overflow must not erase ordinary central probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Independent 100-digit references; probabilities here are exactly 1/2.
        assert(approxEqual(binomialPMF(100, 200, T(0.5)),
            T(0.0563484790092564222472452641429001498014L), 64 * T.epsilon, T(0)));
        assert(approxEqual(binomialPMF(1000, 2000, T(0.5)),
            T(0.01783901114585432073023806723820746438468L), 64 * T.epsilon, T(0)));
        foreach (n; [size_t(0), size_t(65), size_t.max])
        {
            assert(binomialPMF(0, n, T(0)) == 1);
            assert(binomialPMF(n, n, T(1)) == 1);
            if (n != 0)
            {
                assert(binomialPMF(1, n, T(0)) == 0);
                assert(binomialPMF(0, n, T(1)) == 0);
            }
        }
    }}
    // p^32 is subnormal, but multiplying by the coefficient restores a normal
    // result. Reject the imprecise direct intermediate, not just a zero result.
    assert(approxEqual(binomialPMF(32, 64, 1e-10),
        1.832624135078195428602296825521737138147e-302,
        512 * double.epsilon, 0.0));
    static if (size_t.sizeof == 8)
    {
        // 1-p rounds to 1 in double, but its large power is measurably below 1.
        assert(approxEqual(binomialPMF(0, 1UL << 40, 0x1p-60),
            0.9999990463261383409563257953301640312839, 16 * double.epsilon, 0.0));
        assert(approxEqual(binomialPMF(1, 1UL << 40, 0x1p-60),
            0.0000009536734069119819086300723134406561437, 64 * double.epsilon, 0.0));
    }
}

// test BinomialAlgo.approxNormal / approxNormalContinuityCorrection / approxPoisson
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.stat.distribution.normal: normalCDF, normalPDF;
    import mir.stat.distribution.poisson: poissonPMF;
    
    for (size_t i; i < 5; i++) {
        assert(i.binomialPMF!"approxNormal"(5, 0.75).approxEqual(normalPDF(i, 5.0 * 0.75, sqrt(5.0 * 0.75 * 0.25))));
        assert(i.binomialPMF!"approxNormalContinuityCorrection"(5, 0.75).approxEqual(normalCDF(i + 0.5, 5.0 * 0.75, sqrt(5.0 * 0.75 * 0.25)) - normalCDF(i - 0.5, 5.0 * 0.75, sqrt(5.0 * 0.75 * 0.25))));
        assert(i.binomialPMF!"approxPoisson"(5, 0.75).approxEqual(poissonPMF!"gamma"(i, 5 * 0.75)));
        assert(i.binomialPMF!("approxPoisson", "direct")(5, 0.75).approxEqual(poissonPMF(i, 5 * 0.75)));
    }
}

/++
Computes the binomial PMF with extended-range and extended-precision $(D Fp)
arithmetic. Equivalent to $(D binomialPMF(k, n, p)) for an $(D Fp) probability;
this entry point is retained for compatibility.

Probability validation, complements, coefficients, and powers avoid conversions
through hardware floating point. Tested widths are $(D Fp!64), $(D Fp!128),
and $(D Fp!256). Complements, coefficient products, and powers use 128 extra
significand bits, followed by one conversion to the requested result width.
This reserves 64 bits for error amplification by full-width counts and another
64 guard bits. The coefficient ratio uses integer division with a further 64
guard bits. Intermediate operations still round: this substantially reduces
accumulated error, but does not guarantee a correctly rounded result for every
input, especially near rounding midpoints. Greater width does not make the
calculation exact.

For probabilities so small that $(D (n*p)^^2) is below half the smallest
$(D Fp) quantum, a separate path avoids exponent-floor widening and multiplication:
it returns one at zero successes, rounds an exact integer product for one
success, and returns zero for larger success counts. Midpoint rounding accounts
for the strictly negative correction to $(D n*p).

Both counts retain their full $(D size_t) range. Coefficient construction takes
$(D O(min(k, n-k))) iterations; powers take logarithmic time. Large counts are
practical near the endpoints, while central masses with enormous counts are
expensive. Intermediate products must fit the finite $(D Fp) exponent range.
The underlying multiplication can underflow near its exponent floor before
normalization, even when the final mathematical result is representable.
Use $(LREF binomialLPMF) when only a hardware-precision log probability is needed.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D n == 0) and $(D k == 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 0)) $(TD Unit mass at $(D k == 0)))
    $(TR $(TD $(D p == 1)) $(TD Unit mass at $(D k == n)))
    $(TR $(TD $(D k > n)) $(TD Contract violation))
    $(TR $(TD Invalid or nonfinite $(D p)) $(TD Contract violation))
)

Params:
    k = value to evaluate PMF (e.g. number of "heads")
    n = number of trials
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
@safe pure nothrow @nogc
T fp_binomialPMF(T)(const size_t k, const size_t n, const T p)
    if (is(T == Fp!size, size_t size))
    in (k <= n, "k must be less than or equal to n")
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.math.internal.fp_powi: fp_powi;
    import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityCoefficient;
    enum bits = typeof(p.coefficient).sizeof * 8;

    if (n == 0)
        return T(1);
    if (!p.coefficient)
        return T(k == 0 ? 1 : 0);
    // Preserve the one-trial identity even at the exponent floor.
    if (n == 1 && k == 1)
        return p;
    // Here n*p < 2^(long.min/2-1), even for a full-width size_t count.
    // Thus (n*p)^2 < 2^(long.min-2), below half the smallest Fp quantum.
    // Avoid widening or multiplying a probability near the exponent floor.
    if (p.exponent < long.min / 2 - cast(long)bits - 64) {
        if (k == 0)
            return T(1);
        if (k > 1)
            return T(0);
        import mir.bignum.fixed: UInt;
        // The k=1 mass is n*p minus a positive correction smaller than that
        // quantum. Round the exact product, choosing the lower neighbor at
        // a midpoint because (1-p)^(n-1) is strictly less than one for n>1.
        auto product = p.coefficient.toSize!(bits + 64);
        product *= UInt!(bits + 64)(n);
        const usedBits = bits + 64 - product.ctlz;
        if (usedBits > bits) {
            const discarded = usedBits - bits;
            auto remainder = product;
            remainder <<= bits + 64 - discarded;
            auto halfway = UInt!(bits + 64)(1);
            halfway <<= bits + 63;
            if (remainder == halfway)
                product -= 1;
        }
        T result = T(product);
        if (result.exponent < 0 && p.exponent < long.min - result.exponent) {
            // Normalization would cross the exponent floor. The exact product
            // fits the output coefficient here; retain it as a subnormal.
            product <<= cast(size_t)(p.exponent - long.min);
            return T(false, long.min, product.toSize!bits);
        }
        result.exponent += p.exponent;
        return result;
    }
    // Keep the extra precision through the entire expression, including the
    // complement: rounding 1-p at the output width can lose count-amplified bits.
    enum workBits = bits + 128;
    alias W = Fp!workBits;
    const W probability = cast(W)p;
    if (k == n)
        return cast(T)fp_powi(probability, k);
    const W complement = fpProbabilityComplement(probability);
    if (!complement.coefficient)
        return T(0);
    if (k == 0)
        return cast(T)fp_powi(complement, n);
    return cast(T)(fpProbabilityCoefficient!workBits(k, n - k)
        * fp_powi(probability, k) * fp_powi(complement, n - k));
}

// Tiny probabilities must not assert during widening or lose a representable
// k=1 mass. Compare exact integer products without Fp multiplication.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (offset; [0L, 64L, 127L, 128L, 255L, 512L]) {
            T p = T(1);
            p.exponent = long.min + offset;
            T expected = p;
            ++expected.exponent; // 2*p, with a correction too small to round.
            const mass = binomialPMF(1, 2, p);
            assert(mass.coefficient == expected.coefficient);
            assert(mass.exponent == expected.exponent);
            assert(cast(double)binomialPMF(0, size_t.max, p) == 1);
            assert(!binomialPMF(2, size_t.max, p).coefficient);
        }
        // Full-width counts, including the largest exponent entering this path.
        foreach (exponent; [long.min, long.min / 2 - cast(long)bits - 65]) {
            T p = T(1);
            p.exponent = exponent;
            T expected = T(size_t.max);
            expected.exponent += exponent + bits - 1;
            const mass = binomialPMF(1, size_t.max, p);
            assert(mass.coefficient == expected.coefficient);
            assert(mass.exponent == expected.exponent);
            assert(!binomialPMF(2, size_t.max, p).coefficient);
        }
        // A subnormal whose normalization would cross long.min.
        const T subnormal = T(false, long.min, UInt!bits(1));
        const T doubled = binomialPMF(1, 2, subnormal);
        assert(doubled.exponent == long.min && doubled.coefficient == UInt!bits(2));
        // 3*m ends in binary ...11: discarding its low bit is a midpoint
        // whose upper neighbor is even. The negative correction selects down.
        UInt!bits m = UInt!bits(1);
        m <<= bits - 1;
        m += 1;
        const T midpoint = T(false, long.min + 512, m);
        auto exact = m.toSize!(bits + 64);
        exact *= UInt!(bits + 64)(3);
        exact >>= 1;
        const T rounded = binomialPMF(1, 3, midpoint);
        assert(rounded.coefficient == exact.toSize!bits);
        assert(rounded.exponent == midpoint.exponent + 1);
    }}
}

/// The probability type selects the extended-precision implementation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    auto mass = binomialPMF(1, 2, Fp!128(0.5));
    static assert(is(typeof(mass) == Fp!128));
    assert(cast(double)mass == 0.5);
    static assert(is(typeof(binomialPMF(1, 2, 0.5)) == double));
    static assert(is(typeof(binomialPMF!"direct"(1, 2, Fp!128(0.5))) == Fp!128));
    static assert(!__traits(compiles, binomialPMF!"approxNormal"(1, 2, Fp!128(0.5))));
}

// Check contracts against the represented probability, including values which
// would round into [0,1] if validated through hardware floating point.
version(mir_stat_test)
@system unittest // Catching AssertError is @system; the PMF itself remains @safe.
{
    // -unittest keeps test assertions enabled even with -release, while
    // function contracts can be disabled independently. Probe the latter
    // before supplying out-of-contract inputs to the PMF.
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (!contractsEnabled)
        return;

    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    UInt!128 allOnes;
    allOnes.data[] = size_t.max;
    foreach (p; [T(true, -2000, allOnes), T(false, -127, allOnes),
                 T(double.infinity), T(double.nan)]) {
        assertThrown!AssertError(binomialPMF(1, 2, p));
        assertThrown!AssertError(fp_binomialPMF(1, 2, p));
    }
    assertThrown!AssertError(binomialPMF(3, 2, T(0.5)));
}

// Exact integer/dyadic references compare the significand, not a double cast.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        UInt!bits allOnes;
        allOnes.data[] = size_t.max;
        const T p = T(false, -cast(long)bits, allOnes); // 1 - 2^-bits
        const T mass = binomialPMF(1, 2, p); // 2*p*(1-p)
        assert(mass.coefficient == p.coefficient);
        assert(mass.exponent == 1 - 2 * cast(long)bits);
        assert(!mass.sign);
        const T legacy = fp_binomialPMF(1, 2, p);
        assert(legacy.coefficient == mass.coefficient && legacy.exponent == mass.exponent);

        // C(14,4)*(1/4)^4*(3/4)^10 = 59108049/2^28 exactly.
        T expected = T(59108049);
        expected.exponent -= 28;
        const T dyadic = binomialPMF(4, 14, T(0.25));
        assert(dyadic.coefficient == expected.coefficient && dyadic.exponent == expected.exponent);

        // C(200,100)/2^200. Integer reference obtained by exact combinatorics.
        enum exact = UInt!256("90548514656103281165404177077484163874504589675413336841320");
        T centralExpected = T(exact);
        centralExpected.exponent -= 200;
        const T central = binomialPMF(100, 200, T(0.5));
        assert(central.exponent == centralExpected.exponent);
        assert(central.coefficient == centralExpected.coefficient);

        foreach (probability; [T(0), T(1), T(0.5)])
            assert(cast(double)binomialPMF(0, 0, probability) == 1);
        assert(cast(double)binomialPMF(0, size_t.max, T(0)) == 1);
        assert(cast(double)binomialPMF(size_t.max, size_t.max, T(1)) == 1);
        assert(!binomialPMF(1, 2, T(0)).coefficient);
        assert(!binomialPMF(1, 2, T(1)).coefficient);

        T tiny = T(1);
        tiny.exponent = -2000;
        const identity = binomialPMF(1, 1, tiny);
        assert(identity.coefficient == tiny.coefficient && identity.exponent == tiny.exponent);
        tiny.exponent = long.min;
        const floorIdentity = binomialPMF(1, 1, tiny);
        assert(floorIdentity.coefficient == tiny.coefficient && floorIdentity.exponent == tiny.exponent);
        // An exact tail that underflows double, but retains its Fp significand.
        T tailExpected = T(1);
        tailExpected.exponent -= 2000;
        const tail = binomialPMF(0, 2000, T(0.5));
        assert(tail.coefficient == tailExpected.coefficient && tail.exponent == tailExpected.exponent);

        static if (size_t.sizeof == 8) {
            // k exceeds uint.max; symmetry makes the coefficient cheap.
            enum size_t n = 4_294_967_298UL;
            T largeExpected = T(n);
            largeExpected.exponent -= n;
            foreach (k; [size_t(1), n - 1]) {
                const large = binomialPMF(k, n, T(0.5));
                assert(large.coefficient == largeExpected.coefficient);
                assert(large.exponent == largeExpected.exponent);
            }
        }
    }}
}

// BEGIN exact binomial references (tools/generate_fp_binomial_reference.py)
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    struct Reference { size_t k, n; string input; int scale; string coefficient; long exponent; }
    {
        alias T = Fp!64;
        static foreach (c; [
            Reference(17, 70, "6148914691236517205", 64, "16733388697873911521", -69),
            Reference(23, 200, "3602879701896397", 55, "10230625927903023421", -67),
            Reference(500, 1000, "1", 1, "14890222546908570619", -69),
            Reference(1, 200, "18446744073709551615", 64, "14411518807585587199", -12792),
            Reference(199, 200, "18446744073709551615", 64, "14411518807585587045", -120),
            Reference(0, 10000, "1", 65, "18446744073709546616", -64),
            Reference(0, 1, "1", 65, "9223372036854775808", -63),
            Reference(10000, 10000, "18446744073709551615", 64, "18446744073709541616", -64),
            Reference(73, 211, "9223372036854788153", 64, "10609862016950127203", -82)
        ]) {{
            enum input = UInt!64(c.input);
            enum expected = UInt!64(c.coefficient);
            T p = T(input);
            p.exponent -= c.scale;
            const actual = binomialPMF(c.k, c.n, p);
            assert(!actual.sign);
            assert(actual.exponent == c.exponent);
            assert(actual.coefficient == expected);
        }}
    }
    {
        alias T = Fp!128;
        static foreach (c; [
            Reference(17, 70, "113427455640312821154458202477256070485", 128, "308676538795583967883201033362326941131", -133),
            Reference(23, 200, "3602879701896397", 55, "188721738205884379767494546975556810049", -131),
            Reference(500, 1000, "1", 1, "274676124523402021009046115289447739942", -133),
            Reference(1, 200, "340282366920938463463374607431768211455", 128, "265845599156983174580761412056068915199", -25592),
            Reference(199, 200, "340282366920938463463374607431768211455", 128, "265845599156983174580761412056068915045", -248),
            Reference(0, 10000, "1", 129, "340282366920938463463374607431768206456", -128),
            Reference(0, 1, "1", 129, "170141183460469231731687303715884105728", -127),
            Reference(10000, 10000, "340282366920938463463374607431768211455", 128, "340282366920938463463374607431768201456", -128),
            Reference(73, 211, "170141183460469231731687303715884118073", 128, "195717409284067856491917390569949726042", -146)
        ]) {{
            enum input = UInt!128(c.input);
            enum expected = UInt!128(c.coefficient);
            T p = T(input);
            p.exponent -= c.scale;
            const actual = binomialPMF(c.k, c.n, p);
            assert(!actual.sign);
            assert(actual.exponent == c.exponent);
            assert(actual.coefficient == expected);
        }}
    }
    {
        alias T = Fp!256;
        static foreach (c; [
            Reference(17, 70, "38597363079105398474523661669562635951089994888546854679819194669304376546645", 256, "105037183234324200288561499843774761503107551648074383090466587478109325941529", -261),
            Reference(23, 200, "3602879701896397", 55, "64218679766132039475189973950717947873717832431603277521197531359611118099977", -259),
            Reference(500, 1000, "1", 1, "93467441789493670146796325644877583094735607217692095428279460550916380027012", -261),
            Reference(1, 200, "115792089237316195423570985008687907853269984665640564039457584007913129639935", 256, "90462569716653277674664832038037428010367175520031690655826237506182132531199", -51192),
            Reference(199, 200, "115792089237316195423570985008687907853269984665640564039457584007913129639935", 256, "90462569716653277674664832038037428010367175520031690655826237506182132531045", -504),
            Reference(0, 10000, "1", 257, "115792089237316195423570985008687907853269984665640564039457584007913129634936", -256),
            Reference(0, 1, "1", 257, "57896044618658097711785492504343953926634992332820282019728792003956564819968", -255),
            Reference(10000, 10000, "115792089237316195423570985008687907853269984665640564039457584007913129639935", 256, "115792089237316195423570985008687907853269984665640564039457584007913129629936", -256),
            Reference(73, 211, "57896044618658097711785492504343953926634992332820282019728792003956564832313", 256, "66599183278816666490719969379627519090704857053655328512939303056219464395352", -274)
        ]) {{
            enum input = UInt!256(c.input);
            enum expected = UInt!256(c.coefficient);
            T p = T(input);
            p.exponent -= c.scale;
            const actual = binomialPMF(c.k, c.n, p);
            assert(!actual.sign);
            assert(actual.exponent == c.exponent);
            assert(actual.coefficient == expected);
        }}
    }
}
// END exact binomial references

/// fp_binomialPMF preserves the range of small probabilities for large `n`.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.math.common: approxEqual;

    assert(1.fp_binomialPMF(1_000_000, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(1, 1_000_000, 0.75)));
}

// more values to test
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.math.common: approxEqual;

    enum size_t val = 1_000_000;

    assert(0.fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(0, val + 5, 0.75)));
    assert(1.fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(1, val + 5, 0.75)));
    assert(2.fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(2, val + 5, 0.75)));
    assert(5.fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(5, val + 5, 0.75)));
    assert((val / 2).fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(val / 2, val + 5, 0.75)));
    assert((val - 5).fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(val - 5, val + 5, 0.75)));
    assert((val - 2).fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(val - 2, val + 5, 0.75)));
    assert((val - 1).fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(val - 1, val + 5, 0.75)));
    assert((val - 0).fp_binomialPMF(val + 5, Fp!128(0.75)).fp_log!double.approxEqual(binomialLPMF(val, val + 5, 0.75)));
}

// using Fp!128
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.conv: to;
    import mir.math.common: approxEqual;

    for (size_t i; i <= 5; i++) {
        assert(i.fp_binomialPMF(5, Fp!128(0.50)).to!double.approxEqual(binomialPMF(i, 5, 0.50)));
        assert(i.fp_binomialPMF(5, Fp!128(0.75)).to!double.approxEqual(binomialPMF(i, 5, 0.75)));
    }
}

/*
Evaluate the tail on the side of the mean containing k. This avoids forming a
small probability by subtracting an almost-one probability, and keeps p intact
instead of reflecting it through the rounded value of 1-p.

For n <= 64, start at the boundary mass and walk away from the mean using the
ratio of adjacent masses. All coefficients fit in ulong in this range, and a
short recurrence avoids repeated powers. Accumulate float inputs in double.
Larger counts retain direct PMF summation; no incomplete-beta fallback is used.
*/
private @safe pure nothrow @nogc
T binomialDirectTail(bool upper, T)(size_t k, size_t n, T p)
{
    import mir.math.common: exp;
    import std.math: log1p, expm1;

    if (k == n || p == 0)
        return upper ? 0 : 1;
    if (p == 1)
        return upper ? 1 : 0;
    if (k == 0) {
        const real logMass = cast(real)n * log1p(-cast(real)p);
        static if (upper)
            return cast(T) -expm1(logMass);
        else
            return cast(T) exp(logMass);
    }

    static if (is(T == float))
        alias F = double;
    else
        alias F = T;

    const bool lower = k < cast(real)n * p;
    size_t j = lower ? k : k + 1;
    F sum = 0;
    if (n <= 64) {
        const F probability = p;
        const F complement = 1 - probability;
        F term = binomialPMFImpl!(F, BinomialAlgo.direct)(j, n, probability);
        sum = term;
        if (lower) {
            const F ratio = complement / probability;
            while (j > 0) {
                term *= cast(F)j / (n - j + 1) * ratio;
                sum += term;
                --j;
            }
        } else {
            const F ratio = probability / complement;
            while (j < n) {
                term *= cast(F)(n - j) / (j + 1) * ratio;
                sum += term;
                ++j;
            }
        }
    } else {
        // Test the endpoint before incrementing: n + 1 can overflow size_t.
        while (true) {
            const F mass = binomialPMFImpl!(F, BinomialAlgo.direct)(j, n, cast(F)p);
            sum += mass;
            // Stop explicitly at the support boundary before estimating the
            // remaining tail or stepping an unsigned index past its endpoint.
            if (lower ? j == 0 : j == n)
                break;
            // Ratios decrease as we move away from the mean. The geometric
            // series bounds all remaining masses, avoiding a long walk through
            // negligible terms for a large n and a strongly skewed distribution.
            const real ratio = lower
                ? (cast(real)j / (cast(real)(n - j) + 1)) * ((1 - cast(real)p) / p)
                : (cast(real)(n - j) / (cast(real)j + 1)) * (p / (1 - cast(real)p));
            if (ratio < 1 && cast(real)mass * ratio <= F.epsilon * sum * (1 - ratio))
                break;
            if (lower) {
                --j;
            } else {
                ++j;
            }
        }
    }
    return cast(T)(upper == lower ? 1 - sum : sum);
}

private
@safe pure nothrow @nogc
T binomialCDFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.direct)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    return binomialDirectTail!(false, T)(k, n, p);
}

private
@safe pure nothrow @nogc
T binomialCDFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T &&
        (binomialAlgo == BinomialAlgo.approxNormal || 
         binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection))
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    T l = k;
    static if (binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCDF(l, n * p, sqrt(n * p * (1 - p)));
}

private
@safe pure nothrow @nogc
T binomialCDFImpl(T, BinomialAlgo binomialAlgo, PoissonAlgo poissonAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.approxPoisson)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.poisson: poissonCDF;

    return poissonCDF!poissonAlgo(k, n * p);
}

/++
Computes the binomial cumulative distribution function (CDF).

Additional algorithms may be provided for calculating CDF that allow trading off
time and accuracy. If `approxPoisson` is provided, the default is `PoissonAlgo.gamma`

All algorithms handle `p = 0` and `p = 1` exactly and return one at `k = n`,
including when `n = 0`. Interior approximations are not renormalized.

Returns the inclusive lower-tail probability $(D P(X <= k)). The boundary
returns below apply to every algorithm choice.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == n)) $(TD $(D 1)))
    $(TR $(TD $(D p == 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1) and $(D k < n)) $(TD $(D 0)))
    $(TR $(TD $(D k > n)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    binomialAlgo = algorithm for calculating CDF (default: BinomialAlgo.direct)
    poissonAlgo = algorithm for poisson approximation (default: PoissonAlgo.gamma)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
@safe pure nothrow @nogc
template binomialCDF(BinomialAlgo binomialAlgo = BinomialAlgo.direct,
                     PoissonAlgo poissonAlgo = PoissonAlgo.gamma)
{
    /++
    Params:
    k = value to evaluate CDF (e.g. number of "heads")
    n = number of trials
    p = per-trial success probability in $(D [0, 1])
    +/
    T binomialCDF(T)(const size_t k, const size_t n, const T p)
        if (isFloatingPoint!T)
        in (k <= n, "k must be less than or equal to n")
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
    {
        static if (binomialAlgo != BinomialAlgo.direct) {
            if (k == n || p == 0)
                return 1;
            if (p == 1)
                return 0;
        }
        static if (binomialAlgo != BinomialAlgo.approxPoisson)
            return binomialCDFImpl!(T, binomialAlgo)(k, n, p);
        else
            return binomialCDFImpl!(T, binomialAlgo, poissonAlgo)(k, n, p);
    }
}

/// ditto
@safe pure nothrow @nogc
template binomialCDF(string binomialAlgo, string poissonAlgo = "gamma")
{
    mixin("alias binomialCDF = .binomialCDF!(BinomialAlgo." ~ binomialAlgo ~ ", PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, pow;

    assert(4.binomialCDF(6, 2.0 / 3).approxEqual(binomialPMF(0, 6, 2.0 / 3) + binomialPMF(1, 6, 2.0 / 3) + binomialPMF(2, 6, 2.0 / 3) + binomialPMF(3, 6, 2.0 / 3) + binomialPMF(4, 6, 2.0 / 3)));
    // For large values of `n` with `p` not too extreme, can approximate with normal distribution
    assert(550_000.binomialCDF!"approxNormal"(1_000_000, 0.55).approxEqual(0.5));
    // Or closer with continuity correction
    assert(550_000.binomialCDF!"approxNormalContinuityCorrection"(1_000_000, 0.55).approxEqual(0.500401));
    // Poisson approximation is better when `p` is low
    assert(10_000.binomialCDF!"approxPoisson"(1_000_000, 0.01).approxEqual(0.5026596));
}

// test multiple direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    
    static double sumOfbinomialPMFs(T)(size_t k, size_t n, T p) {
        double result = 0.0;
        for (size_t i; i <= k; i++) {
            result += binomialPMF(i, n, p);
        }
        return result;
    }

    // n = 5
    for (size_t i; i <= 5; i++) {
        assert(i.binomialCDF(5, 0.25).approxEqual(sumOfbinomialPMFs(i, 5, 0.25)));
        assert(i.binomialCDF(5, 0.50).approxEqual(sumOfbinomialPMFs(i, 5, 0.50)));
        assert(i.binomialCDF(5, 0.75).approxEqual(sumOfbinomialPMFs(i, 5, 0.75)));
    }

    // n = 6
    for (size_t i; i <= 6; i++) {
        assert(i.binomialCDF(6, 0.25).approxEqual(sumOfbinomialPMFs(i, 6, 0.25)));
        assert(i.binomialCDF(6, 0.5).approxEqual(sumOfbinomialPMFs(i, 6, 0.5)));
        assert(i.binomialCDF(6, 0.75).approxEqual(sumOfbinomialPMFs(i, 6, 0.75)));
    }
}

// test BinomialAlgo.approxNormal / approxNormalContinuityCorrection / approxPoisson
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.stat.distribution.normal: normalCDF;
    import mir.stat.distribution.poisson: poissonCDF;
    
    for (size_t i; i < 5; i++) {
        assert(i.binomialCDF!"approxNormal"(5, 0.75).approxEqual(normalCDF(i, 5.0 * 0.75, sqrt(5.0 * 0.75 * 0.25))));
        assert(i.binomialCDF!"approxNormalContinuityCorrection"(5, 0.75).approxEqual(normalCDF(i + 0.5, 5.0 * 0.75, sqrt(5.0 * 0.75 * 0.25))));
        assert(i.binomialCDF!"approxPoisson"(5, 0.75).approxEqual(poissonCDF!"gamma"(i, 5 * 0.75)));
        assert(i.binomialCDF!("approxPoisson", "direct")(5, 0.75).approxEqual(poissonCDF(i, 5 * 0.75)));
    }
}

private
@safe pure nothrow @nogc
T binomialCCDFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.direct)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    return binomialDirectTail!(true, T)(k, n, p);
}

private
@safe pure nothrow @nogc
T binomialCCDFImpl(T, BinomialAlgo binomialAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T &&
        (binomialAlgo == BinomialAlgo.approxNormal || 
         binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection))
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCCDF;

    T l = k;
    static if (binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCCDF(l, n * p, sqrt(n * p * (1 - p)));
}

private
@safe pure nothrow @nogc
T binomialCCDFImpl(T, BinomialAlgo binomialAlgo, PoissonAlgo poissonAlgo)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.approxPoisson)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.poisson: poissonCCDF;

    return poissonCCDF!poissonAlgo(k, n * p);
}

/++
Computes the binomial complementary cumulative distribution function (CCDF).

Additional algorithms may be provided for calculating CCDF that allow trading off
time and accuracy. If `approxPoisson` is provided, the default is `PoissonAlgo.gamma`

All algorithms handle `p = 0` and `p = 1` exactly and return zero at `k = n`,
including when `n = 0`. Interior approximations are not renormalized.

Returns the strict upper-tail probability $(D P(X > k)). The boundary
returns below apply to every algorithm choice.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == n)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k < n)) $(TD $(D 1)))
    $(TR $(TD $(D k > n)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    binomialAlgo = algorithm for calculating CCDF (default: BinomialAlgo.direct)
    poissonAlgo = algorithm for poisson approximation (default: PoissonAlgo.gamma)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
@safe pure nothrow @nogc
template binomialCCDF(BinomialAlgo binomialAlgo = BinomialAlgo.direct,
                      PoissonAlgo poissonAlgo = PoissonAlgo.gamma)
{
    /++
    Params:
    k = value to evaluate CCDF (e.g. number of "heads")
    n = number of trials
    p = per-trial success probability in $(D [0, 1])
    +/
    T binomialCCDF(T)(const size_t k, const size_t n, const T p)
        if (isFloatingPoint!T)
        in (k <= n, "k must be less than or equal to n")
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
    {
        static if (binomialAlgo != BinomialAlgo.direct) {
            if (k == n || p == 0)
                return 0;
            if (p == 1)
                return 1;
        }
        static if (binomialAlgo != BinomialAlgo.approxPoisson)
            return binomialCCDFImpl!(T, binomialAlgo)(k, n, p);
        else
            return binomialCCDFImpl!(T, binomialAlgo, poissonAlgo)(k, n, p);
    }
}

/// ditto
@safe pure nothrow @nogc
template binomialCCDF(string binomialAlgo, string poissonAlgo = "gamma")
{
    mixin("alias binomialCCDF = .binomialCCDF!(BinomialAlgo." ~ binomialAlgo ~ ", PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, pow;

    assert(4.binomialCCDF(6, 2.0 / 3).approxEqual(binomialPMF(5, 6, 2.0 / 3) + binomialPMF(6, 6, 2.0 / 3)));
    // For large values of `n` with `p` not too extreme, can approximate with normal distribution
    assert(550_000.binomialCCDF!"approxNormal"(1_000_000, 0.55).approxEqual(0.5));
    // Or closer with continuity correction
    assert(550_000.binomialCCDF!"approxNormalContinuityCorrection"(1_000_000, 0.55).approxEqual(0.499599));
    // Poisson approximation is better when `p` is low
    assert(10_000.binomialCCDF!"approxPoisson"(1_000_000, 0.01).approxEqual(0.4973404));
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    // n = 5
    for (size_t i; i <= 5; i++) {
        assert(i.binomialCCDF(5, 0.25).approxEqual(1 - binomialCDF(i, 5, 0.25)));
        assert(i.binomialCCDF(5, 0.50).approxEqual(1 - binomialCDF(i, 5, 0.50)));
        assert(i.binomialCCDF(5, 0.75).approxEqual(1 - binomialCDF(i, 5, 0.75)));
    }

    // n = 6
    for (size_t i; i <= 6; i++) {
        assert(i.binomialCCDF(6, 0.25).approxEqual(1 - binomialCDF(i, 6, 0.25)));
        assert(i.binomialCCDF(6, 0.5).approxEqual(1 - binomialCDF(i, 6, 0.5)));
        assert(i.binomialCCDF(6, 0.75).approxEqual(1 - binomialCDF(i, 6, 0.75)));
    }
}

// test approxNormal / approxNormalContinuityCorrection / approxPoisson
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    for (size_t i; i <= 5; i++) {
        assert(i.binomialCCDF!"approxNormal"(5, 0.25).approxEqual(1 - binomialCDF!"approxNormal"(i, 5, 0.25)));
        assert(i.binomialCCDF!"approxNormalContinuityCorrection"(5, 0.25).approxEqual(1 - binomialCDF!"approxNormalContinuityCorrection"(i, 5, 0.25)));
        assert(i.binomialCCDF!"approxPoisson"(5, 0.25).approxEqual(1 - binomialCDF!"approxPoisson"(i, 5, 0.25)));
        assert(i.binomialCCDF!("approxPoisson", "direct")(5, 0.25).approxEqual(1 - binomialCDF!("approxPoisson", "direct")(i, 5, 0.25)));
    }
}

/*
For a close quantile comparison, floating-point rounding can turn an exact
equality into a crossing. Binary floating-point inputs are exact fractions:
p = a / 2^b. Compute integer probability numerators over the common denominator
2^(b*n), then compare with the exact fraction represented by q.

Limit the denominator to 8192 bits. The twice-as-wide stack storage also holds
intermediate products before division. This is a bounded fallback, not an
arbitrary-precision quantile algorithm. Unsupported inputs retain the precision
of the floating-point CDF/CCDF calculation.
*/
package(mir.stat) @safe pure nothrow @nogc
bool binomialExactQuantileReached(T)(size_t k, size_t n, T p, T q,
    out bool supported, bool complement = false)
{
    import mir.bignum.integer: BigInt;
    import std.math: frexp, ldexp;

    enum maxBits = 8192;
    alias Integer = BigInt!(2 * maxBits / 64);
    supported = false;
    static if (T.mant_dig > 64) {
        return false;
    } else {
        static void split(T value, out ulong numerator, out int exponent) {
            int e;
            const T fraction = frexp(value, e);
            numerator = cast(ulong) ldexp(fraction, T.mant_dig);
            exponent = T.mant_dig - e;
            while (numerator && !(numerator & 1)) {
                numerator >>= 1;
                --exponent;
            }
        }
        static void shift(ref Integer value, int bits) {
            while (bits >= 32) {
                value *= 1UL << 32;
                bits -= 32;
            }
            if (bits)
                value *= 1UL << bits;
        }
        static Integer power(Integer value, size_t exponent) {
            Integer result = Integer(1);
            while (exponent) {
                if (exponent & 1)
                    result *= value;
                exponent >>= 1;
                if (exponent)
                    value *= value;
            }
            return result;
        }

        ulong a, qNumerator;
        int b, qExponent;
        split(p, a, b);
        split(q, qNumerator, qExponent);
        if (b <= 0 || b > maxBits || n > maxBits / cast(uint)b || qExponent > maxBits)
            return false;
        supported = true;

        Integer failureNumerator = Integer(1);
        shift(failureNumerator, b);
        Integer numerator = Integer(a);
        failureNumerator -= numerator; // 1-p = (2^b-a) / 2^b.
        Integer term = power(failureNumerator, n);
        Integer sum = Integer(0);
        foreach (j; 0 .. k + 1) {
            sum += term;
            if (j < k) {
                // Adjacent binomial numerators: C(n,j) * a^j * (2^b-a)^(n-j).
                // Both divisions are exact, including division by j+1 first.
                term *= n - j;
                term *= a;
                term /= cast(uint)(j + 1);
                // Scalar division leaves leading zero limbs in the buffer.
                term.normalize;
                term /= failureNumerator;
            }
        }
        Integer target = Integer(qNumerator);
        // Negative-binomial CDFs are binomial upper tails. Form 1-q in
        // integer arithmetic rather than rounding away a tiny query.
        if (complement) {
            Integer one = Integer(1);
            shift(one, qExponent);
            one -= target;
            target = one;
        }
        const int exponent = cast(int)n * b;
        if (qExponent > exponent)
            shift(sum, qExponent - exponent);
        else
            shift(target, exponent - qExponent);
        return complement ? sum <= target : sum >= target;
    }
}

private @safe pure nothrow @nogc
bool binomialQuantileReached(T)(size_t k, size_t n, T p, T q, bool upper, T value)
{
    import std.math: abs;

    const T target = upper ? 1 - q : q;
    // A conservative screen for close comparisons, not a tolerance that changes
    // the quantile inequality. Exact equality belongs to the lower integer.
    // Subnormal tails have less relative precision. Adding min_normal supplies
    // an absolute allowance based on the smallest subnormal spacing, so rounding
    // cannot erase the screen precisely where an exact comparison is needed.
    if (n <= 8192 && abs(value - target) <=
        128 * (n + 1) * T.epsilon * (abs(value) + target + T.min_normal)) {
        bool supported;
        const bool reached = binomialExactQuantileReached(k, n, p, q, supported);
        if (supported)
            return reached;
        return upper
            ? binomialCCDF(k, n, cast(real)p) <= 1 - cast(real)q
            : binomialCDF(k, n, cast(real)p) >= q;
    }
    return upper ? value <= target : value >= target;
}

private
@safe pure nothrow @nogc
size_t binomialInvCDFImpl(T, BinomialAlgo binomialAlgo)(const T q, const size_t n, const T p)
    if (isFloatingPoint!T && binomialAlgo == BinomialAlgo.direct)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import std.math: abs;

    if (q == 0 || p == 0 || n == 0) {
        return 0;
    } else if (q == 1 || p == 1) {
        return n;
    }

    static if (is(T == float))
        alias F = double;
    else
        alias F = T;
    const F probability = p;
    const F quantile = q;
    const bool upper = q > 0.5;
    const F target = upper ? 1 - quantile : quantile;
    bool reached(size_t k, F value) {
        return binomialQuantileReached(k, n, probability, quantile, upper, value);
    }
    F evaluate(size_t k) {
        return upper ? binomialCCDF(k, n, probability) : binomialCDF(k, n, probability);
    }

    // Short distributions need only one endpoint mass and an adjacent-mass
    // recurrence. Do not seed it with an underflowed or subnormal probability.
    if (n <= 64) {
        size_t k = upper ? n : 0;
        F term = binomialPMF(k, n, probability);
        if (term >= F.min_normal) {
            F sum = upper ? 0 : term;
            if (upper) {
                while (k > 0) {
                    const F next = sum + term;
                    if (!reached(k - 1, next))
                        return k;
                    sum = next;
                    term *= cast(F)k / (n - k + 1) * ((1 - probability) / probability);
                    --k;
                }
            } else {
                while (k < n && !reached(k, sum)) {
                    term *= cast(F)(n - k) / (k + 1) * (probability / (1 - probability));
                    sum += term;
                    ++k;
                }
            }
            return k;
        }
    }

    size_t guess = p > 0.5 ? n : 0;
    if ((n > 20 && (probability > 0.25 && probability < 0.75)) ||
        (n * probability > 9 * (1 - probability) && n * (1 - probability) > 9 * probability) ||
        (n * probability >= 5 && n * (1 - probability) >= 5)) {
        guess = binomialInvCDFImpl!(F, BinomialAlgo.approxNormalContinuityCorrection)(quantile, n, probability);
    } else if ((n >= 20 && probability <= 0.05) ||
               (n >= 100 && n * probability <= 10)) {
        guess = binomialInvCDFImpl!(F, BinomialAlgo.approxPoisson, PoissonAlgo.approxNormalContinuityCorrection)(quantile, n, probability);
    }
    F value = evaluate(guess);
    if (reached(guess, value)) {
        while (guess > 0) {
            const F mass = binomialPMF(guess, n, probability);
            F next = upper ? value + mass : value - mass;
            // Re-evaluate after substantial cancellation or a close comparison
            // instead of propagating subtraction error through later steps.
            if (next < value * 0.5 || abs(next - target) <= 64 * F.epsilon * (abs(value) + mass + target))
                next = evaluate(guess - 1);
            if (!reached(guess - 1, next))
                break;
            --guess;
            value = next;
        }
    } else {
        while (guess < n) {
            ++guess;
            const F mass = binomialPMF(guess, n, probability);
            F next = upper ? value - mass : value + mass;
            if (next < value * 0.5 || abs(next - target) <= 64 * F.epsilon * (abs(value) + mass + target))
                next = evaluate(guess);
            value = next;
            if (reached(guess, value))
                break;
        }
    }
    return guess;
}

private
@safe pure nothrow @nogc
size_t binomialInvCDFImpl(T, BinomialAlgo binomialAlgo)(const T q, const size_t n, const T p)
    if (isFloatingPoint!T &&
        (binomialAlgo == BinomialAlgo.approxNormal || 
         binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection))
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalCDF, normalInvCDF;

    T mu = n * p;
    T std = sqrt(mu * (1 - p));

    // Handles case where q is small or large, better than just using probLowerBound = 0 or probUpperBound = 0
    T probLowerBound = 0;
    T probUpperBound = 1;
    T lowerValue = 0;
    T upperValue = n;
    static if (binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection) {
        lowerValue += 0.5;
        upperValue -= 0.5;
    }
    probLowerBound = normalCDF(lowerValue, mu, std);
    probUpperBound = normalCDF(upperValue, mu, std);
    if (q <= probLowerBound) {
        return 0;
    } else if (q >= probUpperBound) {
        return n;
    }

    auto result = normalInvCDF(q, mu, std);
    static if (binomialAlgo == BinomialAlgo.approxNormalContinuityCorrection) {
        result = result - 0.5;
    }
    // Check in floating point before converting: large counts can round up
    // beyond size_t.max, and a negative approximation must not wrap unsigned.
    if (result <= 0)
        return 0;
    if (result >= n)
        return n;
    return cast(size_t) floor(result);
}

private
@safe pure nothrow @nogc
size_t binomialInvCDFImpl(T, BinomialAlgo binomialAlgo, PoissonAlgo poissonAlgo)(const T q, const size_t n, const T p)
    if (isFloatingPoint!T &&
        binomialAlgo == BinomialAlgo.approxPoisson && poissonAlgo != PoissonAlgo.gamma)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.poisson: poissonInvCDF;

    const size_t result = poissonInvCDF!poissonAlgo(q, n * p);
    // A Poisson distribution is unbounded; the binomial support ends at n.
    return result < n ? result : n;
}

/++
Computes the binomial inverse cumulative distribution function (InvCDF).

All algorithms return zero for `q = 0`, `p = 0`, or `n = 0`; otherwise
`q = 1` or `p = 1` returns `n`. Approximate quantiles are limited to `[0, n]`.

The direct algorithm seeks the smallest `k` whose CDF is at least `q`.
Upper-half probabilities are compared using the CCDF
so that subtracting a small tail from one does not lose the comparison.
Close comparisons use exact integer arithmetic when the binary probability
fractions fit a bounded workspace. Outside that bound, the result remains
subject to the accuracy of the floating-point tail calculation.

Additional algorithms may be provided for calculating InvCDF that allow trading off
time and accuracy. If `approxPoisson` is provided, the default is
`PoissonAlgo.direct`, which is different from `binomialPMF` and `binomialCDF`
`PoissonAlgo.gamma` is not supported.

Zero queries and the point mass at zero take precedence over upper-endpoint
returns, for every algorithm choice. A zero query returns zero even when
$(D p == 1) puts all mass at $(D n).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 0), $(D p == 0), or $(D n == 0)) $(TD $(D 0)))
    $(TR $(TD $(D q == 1) with $(D p > 0) and $(D n > 0)) $(TD $(D n)))
    $(TR $(TD $(D p == 1) and $(D q > 0)) $(TD $(D n)))
    $(TR $(TD $(D q < 0), $(D q > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    binomialAlgo = algorithm for calculating InvCDF (default: BinomialAlgo.direct)
    poissonAlgo = algorithm for poisson approximation (default: PoissonAlgo.direct)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
@safe pure nothrow @nogc
template binomialInvCDF(BinomialAlgo binomialAlgo = BinomialAlgo.direct,
                        PoissonAlgo poissonAlgo = PoissonAlgo.direct)
    if (poissonAlgo != PoissonAlgo.gamma)
{
    /++
    Params:
    q = cumulative probability in $(D [0, 1])
    n = number of trials
    p = per-trial success probability in $(D [0, 1])
    +/
    size_t binomialInvCDF(T)(const T q, const size_t n, const T p)
        if (isFloatingPoint!T)
        in (q >= 0, "q must be greater than or equal to 0")
        in (q <= 1, "q must be less than or equal to 1")
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
    {
        static if (binomialAlgo != BinomialAlgo.direct) {
            if (q == 0 || n == 0 || p == 0)
                return 0;
            if (q == 1 || p == 1)
                return n;
        }
        static if (binomialAlgo != BinomialAlgo.approxPoisson)
            return binomialInvCDFImpl!(T, binomialAlgo)(q, n, p);
        else
            return binomialInvCDFImpl!(T, binomialAlgo, poissonAlgo)(q, n, p);
    }
}

/// ditto
@safe pure nothrow @nogc
template binomialInvCDF(string binomialAlgo, string poissonAlgo = "direct")
{
    mixin("alias binomialInvCDF = .binomialInvCDF!(BinomialAlgo." ~ binomialAlgo ~ ", PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.15.binomialInvCDF(6, 2.0 / 3) == 3);
    // For large values of `n` with `p` not too extreme, can approximate with normal distribution
    assert(0.5.binomialInvCDF!"approxNormal"(1_000_000, 0.55) == 550_000);
    // Or closer with continuity correction
    assert(0.500401.binomialInvCDF!"approxNormalContinuityCorrection"(1_000_000, 0.55) == 550_000);
    // Poisson approximation is better when `p` is low
    // The Poisson CDF at 10,000 is approximately 0.5026595812190076.
    assert(0.5026595.binomialInvCDF!"approxPoisson"(1_000_000, 0.01) == 10_000);
    assert(0.5026596.binomialInvCDF!"approxPoisson"(1_000_000, 0.01) == 10_001);
}

// test BinomialAlgo.direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.binomialInvCDF(5, 0.6) == 0);
    assert(1.binomialInvCDF(5, 0.6) == 5);
    for (double x = 0.05; x < 1; x = x + 0.05) {
        size_t value = x.binomialInvCDF(5, 0.6);
        assert(value.binomialCDF(5, 0.6) >= x);
        assert((value - 1).binomialCDF(5, 0.6) < x);
    }
}

// Degenerate distributions and bounded searches at the support endpoints.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real)) {{
        foreach (n; [size_t(0), size_t(20), size_t.max]) {
            assert(binomialInvCDF(T(0.5), n, T(0)) == 0);
            assert(binomialInvCDF(T(1), n, T(0)) == 0);
            assert(binomialInvCDF(T(0), n, T(1)) == 0);
            assert(binomialInvCDF(T(0.5), n, T(1)) == n);
            assert(binomialInvCDF(T(1), n, T(1)) == n);
        }
        assert(binomialInvCDF(T(0.5), 0, T(0.5)) == 0);
        assert(binomialInvCDF(T(1), 100, T(0.5)) == 100);
        // The endpoint mass used by the short recurrence underflows here.
        assert(binomialInvCDF(T(0.9), 64, T(1e-8L)) == 0);
        assert(binomialInvCDF(T(0.1), 64, T(1) - T(1e-8L)) == 64);
    }}
}

// Exact ties belong to the lower integer, rather than the next observation.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown, nextUp;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real)) {{
        // Binomial(5, 1/4) has exact CDF numerators over 4^5 = 1024.
        // These references come from integer binomial coefficients, not CDF().
        foreach (k, numerator; [243, 648, 918, 1008, 1023]) {
            const T q = T(numerator) / 1024;
            assert(binomialInvCDF(nextDown(q), 5, T(0.25)) == k);
            assert(binomialInvCDF(q, 5, T(0.25)) == k);
            assert(binomialInvCDF(nextUp(q), 5, T(0.25)) == k + 1);
        }
        // Symmetry gives an exact half at the middle of odd-sized support.
        // Check both sides of the short-distribution cutoff.
        foreach (n; [size_t(63), size_t(65)]) {
            assert(binomialInvCDF(nextDown(T(0.5)), n, T(0.5)) == n / 2);
            assert(binomialInvCDF(T(0.5), n, T(0.5)) == n / 2);
            assert(binomialInvCDF(nextUp(T(0.5)), n, T(0.5)) == n / 2 + 1);
        }
    }}
}

// Tiny upper tails must not be reconstructed by subtracting a rounded CDF.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown;

    // For Binomial(64, 1/2), P(X>61)=2081/2^64 and P(X>62)=65/2^64.
    // The target tail is 2^-53, between these two exact probabilities.
    assert(binomialInvCDF(nextDown(1.0), 64, 0.5) == 62);
    // Independently checked by summing masses with 100-decimal precision for
    // the exact binary value of 0.01f. The old search could step past n.
    assert(binomialInvCDF(nextDown(1.0f), 200, 0.01f) == 13);
}

// Subnormal lower tails still distinguish queries on either side of a CDF step.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown, nextUp;

    // Exact integer sums of binomial coefficients over 2^1500 put CDF(79)
    // strictly between nextDown(q) and q, and CDF(80) above nextUp(q).
    // A purely relative comparison screen could underflow and return 79 for q.
    const double q = 0x0.0000000010cbep-1022;
    assert(binomialInvCDF(nextDown(q), 1500, 0.5) == 79);
    assert(binomialInvCDF(q, 1500, 0.5) == 80);
    assert(binomialInvCDF(nextUp(q), 1500, 0.5) == 80);
}

// Exact comparisons have a fixed arithmetic budget and never silently truncate.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    bool supported;
    assert(!binomialExactQuantileReached(0, 8192, 0.5, 0.5, supported));
    assert(supported);
    assert(!binomialExactQuantileReached(0, 8193, 0.5, 0.5, supported));
    assert(!supported);
    // A probability with a long binary denominator can exceed the same budget
    // with far fewer trials. The ordinary floating-point comparison remains.
    assert(!binomialExactQuantileReached(60, 200, 0.3, 0.5, supported));
    assert(!supported);
    assert(binomialInvCDF(0.5, 200, 0.3) == 60);
    // For the exact double value of p=0.3, a 100-decimal reference gives
    // CDF(60) = 0.5348091761606989177271863599018766395... .
    // Nearby queries exercise the floating-point fallback outside the budget.
    assert(binomialInvCDF(0.5348091761606, 200, 0.3) == 60);
    assert(binomialInvCDF(0.5348091761608, 200, 0.3) == 61);
    // The lower-half fallback must distinguish either side of CDF(59) too:
    // 0.4733474593659296238141304129738303866... for the same binary p.
    assert(binomialInvCDF(0.4733474593658, 200, 0.3) == 59);
    assert(binomialInvCDF(0.4733474593660, 200, 0.3) == 60);
}

// A skewed distribution can start at n and require several downward steps.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    // Exact rational binomial sums for the binary double p=0.95 establish
    // these quantiles. Exercise both lower-CDF and upper-CCDF searches.
    assert(binomialInvCDF(0.1, 65, 0.95) == 59);
    assert(binomialInvCDF(0.6, 65, 0.95) == 62);
}

// test Binomial.direct, alternate guess paths
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    static immutable int[] ns =    [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.05; x < 1; x = x + 0.05) {
            value = x.binomialInvCDF(ns[i], ps[i]);
            assert(value.binomialCDF(ns[i], ps[i]) >= x);
            if (value > 1)
                assert((value - 1).binomialCDF(ns[i], ps[i]) < x);
        }
    }
}

// test Binomial.direct, detailed alternate guess paths
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    static immutable int[] ns =    [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.01; x < 1; x = x + 0.01) {
            value = x.binomialInvCDF(ns[i], ps[i]);
            assert(value.binomialCDF(ns[i], ps[i]) >= x);
            if (value > 1)
                assert((value - 1).binomialCDF(ns[i], ps[i]) < x);
        }
    }
}

// test Binomial.approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalInvCDF;

    assert(0.binomialInvCDF!"approxNormal"(1000, 0.55) == 0);
    assert(0.binomialInvCDF!"approxNormalContinuityCorrection"(1000, 0.55) == 0);
    assert(1.binomialInvCDF!"approxNormal"(1000, 0.55) == 1_000);
    assert(1.binomialInvCDF!"approxNormalContinuityCorrection"(1000, 0.55) == 1_000);
    double checkValue;
    for (double x = 0.05; x < 1; x = x + 0.05) {
        checkValue = normalInvCDF(x, 1_000 * 0.55, sqrt(1000 * 0.55 * 0.45));
        assert(x.binomialInvCDF!"approxNormal"(1_000, 0.55) == floor(checkValue));
        assert(x.binomialInvCDF!"approxNormalContinuityCorrection"(1_000, 0.55) == floor(checkValue - 0.5));
    }
}

// Approximation choices preserve point masses and the finite binomial support.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown;

    static foreach (T; AliasSeq!(float, double, real)) {{
        static foreach (algo; ["approxNormal", "approxNormalContinuityCorrection", "approxPoisson"]) {{
            assert(binomialPMF!algo(0, 0, T(0.3)) == 1);
            assert(binomialCDF!algo(0, 0, T(0.3)) == 1);
            assert(binomialCCDF!algo(0, 0, T(0.3)) == 0);
            assert(binomialInvCDF!algo(T(0.5), 0, T(0.3)) == 0);
            foreach (n; [size_t(5), size_t.max]) {
                assert(binomialPMF!algo(0, n, T(0)) == 1);
                assert(binomialPMF!algo(1, n, T(0)) == 0);
                assert(binomialPMF!algo(n, n, T(1)) == 1);
                assert(binomialPMF!algo(n - 1, n, T(1)) == 0);
                assert(binomialCDF!algo(0, n, T(0)) == 1);
                assert(binomialCDF!algo(n - 1, n, T(1)) == 0);
                assert(binomialCCDF!algo(0, n, T(0)) == 0);
                assert(binomialCCDF!algo(n - 1, n, T(1)) == 1);
                foreach (p; [T(0), T(0.25), T(1)]) {
                    assert(binomialCDF!algo(n, n, p) == 1);
                    assert(binomialCCDF!algo(n, n, p) == 0);
                    assert(binomialInvCDF!algo(T(0), n, p) == 0);
                }
                assert(binomialInvCDF!algo(T(1), n, T(0.25)) == n);
                assert(binomialInvCDF!algo(T(1), n, T(0)) == 0);
                assert(binomialInvCDF!algo(T(0.5), n, T(0)) == 0);
                assert(binomialInvCDF!algo(T(0.5), n, T(1)) == n);
            }
            assert(binomialInvCDF!algo(nextDown(T(1)), 20, T(0.25)) <= 20);
        }}
    }}
}

// Rounding near the normal lower cutoff must not wrap a negative count.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextUp, sqrt;
    import std.meta: AliasSeq;
    import mir.stat.distribution.normal: normalCDF;

    static foreach (T; AliasSeq!(float, double, real)) {{
        static foreach (algo; ["approxNormal", "approxNormalContinuityCorrection"]) {{
            foreach (n; [size_t(2), size_t(5)]) {
                foreach (p; [T(0.25), T(0.5), T(0.75)]) {
                    const T lower = algo == "approxNormal" ? 0 : 0.5;
                    const T q = nextUp(normalCDF(lower, n * p, sqrt(n * p * (1 - p))));
                    // Just above this cutoff, the continuous result (after
                    // continuity correction, if any) is near zero: floor to 0.
                    assert(binomialInvCDF!algo(q, n, p) == 0);
                }
            }
        }}
    }}
    // Here the lower-cutoff CDF underflows to zero. The normal quantile of
    // the smallest positive double is still negative and must be clipped.
    assert(binomialInvCDF!"approxNormal"(nextUp(0.0), 1000, 0.5) == 0);
    assert(binomialInvCDF!"approxNormalContinuityCorrection"(nextUp(0.0), 1000, 0.5) == 0);
    // A small positive query below a nonzero lower cutoff also returns zero.
    assert(binomialInvCDF!"approxNormal"(0.01, 2, 0.25) == 0);
    assert(binomialInvCDF!"approxNormalContinuityCorrection"(0.01, 2, 0.25) == 0);
}

// A query below the normal upper cutoff can still round back to that endpoint.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown, sqrt;
    import std.meta: AliasSeq;
    import mir.stat.distribution.normal: normalCDF;

    static foreach (T; AliasSeq!(float, double, real)) {{
        const T p = T(0.99);
        const T mu = 2 * p;
        const T deviation = sqrt(mu * (1 - p));
        const T q = nextDown(normalCDF(T(2), mu, deviation));
        const size_t result = binomialInvCDF!"approxNormal"(q, 2, p);
        // The continuous quantile is just below 2. Depending on rounding,
        // flooring gives 1 or the endpoint guard returns 2.
        assert(result == 1 || result == 2);
    }}
}

// The Poisson sub-algorithm must not change endpoint handling or return k > n.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown;

    static foreach (algo; ["direct", "gamma", "approxNormal", "approxNormalContinuityCorrection"]) {
        assert(binomialPMF!("approxPoisson", algo)(0, 20, 0.0) == 1);
        assert(binomialCDF!("approxPoisson", algo)(0, 20, 0.0) == 1);
        assert(binomialCCDF!("approxPoisson", algo)(0, 20, 0.0) == 0);
        static if (algo != "gamma") {
            assert(binomialInvCDF!("approxPoisson", algo)(0.5, 20, 0.0) == 0);
            assert(binomialInvCDF!("approxPoisson", algo)(1.0, 20, 0.25) == 20);
            assert(binomialInvCDF!("approxPoisson", algo)(nextDown(1.0), 20, 0.25) == 20);
        }
    }
}

// test Binomial.approxPoisson
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.distribution.poisson: poissonInvCDF;

    assert(0.binomialInvCDF!"approxPoisson"(20, 0.25) == 0);
    for (double x = 0.05; x < 1; x = x + 0.05) {
        assert(x.binomialInvCDF!"approxPoisson"(20, 0.25) == poissonInvCDF(x, 20 * 0.25));
        assert(x.binomialInvCDF!("approxPoisson", "approxNormalContinuityCorrection")(1_000, 0.55) == poissonInvCDF!"approxNormalContinuityCorrection"(x, 1000 * 0.55));
    }
}

/// Count successes in a fixed number of trials and distinguish the two tails.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(binomialPMF(1, 2, 0.5) == 0.5);
    assert(approxEqual(binomialCDF(1, 2, 0.5), 0.75));
    assert(approxEqual(binomialCCDF(1, 2, 0.5), 0.25));
    assert(binomialInvCDF(0.75, 2, 0.5) == 1);
    assert(binomialInvCDF(0.0, 2, 1.0) == 0);
    assert(binomialInvCDF(1.0, 2, 1.0) == 2);
}

/++
Computes the binomial log probability mass function (LPMF).

Large counts use a centered calculation to avoid cancellation near the mode.
Counts retain their full `size_t` width, including near either endpoint.

Returns the natural logarithm of $(D P(X == k)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D n == 0) (so $(D k == 0))) $(TD $(D 0)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0) at $(D k == 0); otherwise $(D -T.infinity)))
    $(TR $(TD $(D p == 1)) $(TD $(D 0) at $(D k == n); otherwise $(D -T.infinity)))
    $(TR $(TD $(D k > n)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = value to evaluate LPMF (e.g. number of "heads")
    n = number of trials
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Binomial_distribution, Binomial Distribution)
+/
T binomialLPMF(T)(const size_t k, const size_t n, const T p)
    if (isFloatingPoint!T)
    in (k <= n, "k must be less than or equal to n")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.internal.xlogy: xlogy, xlog1py;
    import mir.math.internal.log_binomial: logBinomialCoefficient;

    // Keep the existing result-type arithmetic for ordinary small counts.
    if (n <= 64)
        return logBinomialCoefficient(n, cast(const uint) k)
            + xlogy(k, p) + xlog1py((n - k), -p);
    return cast(T) binomialLogMass(k, n, cast(real) p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;

    for (size_t i; i <= 5; i++) {
        assert(i.binomialLPMF(5, 0.5).exp.approxEqual(binomialPMF(i, 5, 0.5)));
        assert(i.binomialLPMF(5, 0.75).exp.approxEqual(binomialPMF(i, 5, 0.75)));
    }
}

// Check centered log masses, helper transitions, and full-width observation counts.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(approxEqual(binomialLPMF(31, 65, T(0.5)),
            T(-2.385031079372598900724141513641103501569L), 128 * T.epsilon, T(0)));
        assert(approxEqual(binomialLPMF(32, 65, T(0.5)),
            T(-2.324406457556164058143535381600683238283L), 64 * T.epsilon, T(0)));
        assert(approxEqual(binomialLPMF(1000, 2000, T(0.5)),
            T(-4.026367582410560281327866338545531330554L), 64 * T.epsilon, T(0)));
        foreach (p; [T(0.125), T(0.25), T(0.5), T(0.75), T(0.875)])
        {
            assert(approxEqual(binomialLPMF(32, 64, p), binomialLPMF(32, 64, 1-p),
                128 * T.epsilon, T(0)));
            assert(approxEqual(exp(binomialLPMF(32, 65, p)), binomialPMF(32, 65, p),
                128 * T.epsilon, T(0)));
        }
        assert(binomialLPMF(0, size_t.max, T(0)) == 0);
        assert(binomialLPMF(size_t.max, size_t.max, T(1)) == 0);
        assert(binomialLPMF(1, size_t.max, T(0)) == -T.infinity);
        assert(binomialLPMF(0, size_t.max, T(1)) == -T.infinity);
    }}
    static if (size_t.sizeof == 8)
    {
        // The probability is the exact binary double used for the reference.
        const double p = 0.99999999883584678173065185546875;
        assert(approxEqual(binomialLPMF(4_294_967_296UL, 4_294_967_301UL, p),
            -1.740302180029467515858762710331235197724, 128 * double.epsilon, 0.0));
        assert(approxEqual(binomialLPMF(500_000_000_000UL, 1_000_000_000_000UL, 0.5),
            -14.04130191060925153647104634305362631735, 32 * double.epsilon, 0.0));
    }
}

/// Accurate values for large values of `n`
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.math.common: approxEqual;

    assert(1.binomialLPMF(1_000_000, 0.75).approxEqual(fp_binomialPMF(1, 1_000_000, Fp!128(0.75)).fp_log!double));
}

// more values to test
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.math.common: approxEqual;

    enum size_t val = 1_000_000;

    assert(0.binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(0, val + 5, Fp!128(0.75)).fp_log!double));
    assert(1.binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(1, val + 5, Fp!128(0.75)).fp_log!double));
    assert(2.binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(2, val + 5, Fp!128(0.75)).fp_log!double));
    assert(5.binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(5, val + 5, Fp!128(0.75)).fp_log!double));
    assert((val / 2).binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(val / 2, val + 5, Fp!128(0.75)).fp_log!double));
    assert((val - 5).binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(val - 5, val + 5, Fp!128(0.75)).fp_log!double));
    assert((val - 2).binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(val - 2, val + 5, Fp!128(0.75)).fp_log!double));
    assert((val - 1).binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(val - 1, val + 5, Fp!128(0.75)).fp_log!double));
    assert((val - 0).binomialLPMF(val + 5, 0.75).approxEqual(fp_binomialPMF(val, val + 5, Fp!128(0.75)).fp_log!double));
}

// Direct tails: degenerate distributions, cancellation, and recurrence boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: abs;

    static foreach (T; AliasSeq!(float, double, real)) {{
        foreach (n; [size_t(0), size_t(10), size_t.max]) {
            assert(binomialCDF(n, n, T(0)) == 1);
            assert(binomialCCDF(n, n, T(1)) == 0);
            if (n != 0) {
                assert(binomialCDF(n - 1, n, T(0)) == 1);
                assert(binomialCCDF(n - 1, n, T(0)) == 0);
                assert(binomialCDF(n - 1, n, T(1)) == 0);
                assert(binomialCCDF(n - 1, n, T(1)) == 1);
            }
        }
        const T tiny = T(1e-20L);
        // The quadratic correction is negligible at this input precision.
        assert(abs(binomialCCDF(0, 10, tiny) / (10 * tiny) - 1) < 16 * T.epsilon);
        assert(abs(binomialCCDF(1, 10, tiny) / (45 * tiny * tiny) - 1) <
            (is(T == float) ? T(1e-4) : T(1e-13)));

        // Check bounds, monotonicity, and complement agreement across n=64/65.
        const T tolerance = is(T == float) ? T(2e-6) : T(2e-13);
        foreach (n; [size_t(64), size_t(65)]) {
            T previous = 0;
            foreach (k; 0 .. n + 1) {
                const T c = binomialCDF(k, n, T(0.25));
                const T s = binomialCCDF(k, n, T(0.25));
                assert(c >= 0 && c <= 1 && s >= 0 && s <= 1);
                assert(c + tolerance >= previous);
                assert(abs(c + s - 1) <= tolerance);
                previous = c;
            }
        }
        // Independent finite binomial sums evaluated with mpmath at 100 digits.
        assert(abs(binomialCCDF(32, 64, T(0.25)) /
            T(0.00000458614953167524842111188594734398354L) - 1) < tolerance);
        assert(abs(binomialCCDF(32, 65, T(0.25)) /
            T(0.00000708105513203867121233649993116407543L) - 1) < tolerance);
        assert(abs(binomialCDF(31, 64, T(0.75)) /
            T(0.00000458614953167524842111188594734398354L) - 1) < tolerance);
        // The old reflected calculation loses this small upper tail.
        assert(abs(binomialCCDF(1, 65, tiny) / (2080 * tiny * tiny) - 1) <
            (is(T == float) ? T(1e-4) : T(1e-12)));
    }}
    // A large, sparse distribution should not require traversing n masses.
    assert(abs(binomialCCDF(1, 1_000_000, 1e-20) / 4.999995e-29 - 1) < 1e-12);
    // Only one upper-tail mass is needed, even when n+1 is unrepresentable.
    const double p = 1 - double.epsilon / 2;
    const double expected = binomialPMF(size_t.max, size_t.max, p);
    assert(binomialCCDF(size_t.max - 1, size_t.max, p) == expected);
}
