/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution).

There are multiple alternative formulations of the Negative Binomial Distribution. The
formulation in this module counts failures before the `r`th success. Thus
`k = 0` means that the first `r` trials all succeed, and `r = 1` is geometric.

The required success count $(D r) is a positive $(D size_t), not a real-valued
shape parameter. The per-trial success probability $(D p) must be finite and
in $(D [0, 1]). At $(D p == 1), there are no failures. At $(D p == 0), success
never occurs: the API uses zero mass and zero CDF at every finite count, with
CCDF equal to one.

PMF and LPMF accept $(D Fp) probabilities. $(LREF negativeBinomialTailsChecked)
computes independently rounded Fp tails with explicit work limits.
$(LREF negativeBinomialInvCDFChecked) provides an exact checked Fp count inverse. The separate
CDF, CCDF, and inverse CDF use built-in floating-point probabilities. The inverse returns $(D size_t), using
$(D size_t.max) for an unbounded quantile or one outside the count range.
Its endpoint conventions differ from $(D geometricInvCDF), even for $(D r == 1).

The tables assume valid parameters except where an invalid input is listed.
Finite expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts does
not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.negative_binomial;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityComplement;
import mir.internal.utility: isFloatingPoint;

/++
Computes the negative binomial probability mass function (PMF).

Returns the point probability $(D P(K == k)). The count $(D k) excludes the
$(D r) successes; the stopping trial is therefore trial $(D k + r).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D p) raised to $(D r)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k == 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D 0)))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])


See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
pragma(inline, true)
T negativeBinomialPMF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow, exp;
    import mir.combinatorics: binomial;

    // Preserve the inexpensive product only when its coefficient is known to
    // fit. Test by subtraction so k+r-1 cannot wrap before the decision.
    if (k <= 64 && r - 1 <= 64 - k) {
        const ulong coefficient = binomial(k + r - 1, r - 1);
        const T failures = pow(1 - p, k);
        const T successes = pow(p, r);
        if (failures >= T.min_normal && successes >= T.min_normal)
            return coefficient * failures * successes;
    }
    if (p == 0)
        return 0;
    if (p == 1)
        return k == 0 ? 1 : 0;
    if (k == 0)
        return pow(p, r);
    return cast(T)exp(negativeBinomialLogMass(k, r, cast(real)p));
}

/// Zero failures means that every trial up to the required success count succeeds.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.stat.distribution.geometric: geometricPMF;
    // Two successes with no failures: p squared.
    assert(negativeBinomialPMF(0, 2, 0.5) == 0.25);
    // One required success recovers the geometric count convention.
    assert(negativeBinomialPMF(1, 1, 0.5) == geometricPMF(1, 0.5));
}

private @safe pure nothrow @nogc
real negativeBinomialLogMass(size_t k, size_t r, real p)
{
    import mir.stat.distribution.binomial: binomialLogMassCounts;
    import mir.math.common: log;
    import std.math: log1p;
    if (p == 0)
        return -real.infinity;
    if (p == 1)
        return k == 0 ? 0 : -real.infinity;
    if (k == 0)
        return cast(real)r * log(p);
    // NB(k,r,p) = BinomialMass(r,k+r,p) * r/(k+r). Separate counts
    // avoid overflowing k+r, and log1p preserves a small ratio k/r.
    return binomialLogMassCounts(r, k, p) - log1p(cast(real)k / r);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialPMF(6, 3.0 / 4).shouldApprox == 0.0875988;
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0877915;
    1.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.175583;
    2.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.2048468;
    3.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.1820861;
    4.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.1365645;
    5.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.09104303;
    6.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.05563741;
    7.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0317928;
    8.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0172211;
    9.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.008929461;
    10.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.00446473;
}

/++
Computes the negative binomial probability mass function (PMF) directly with extended
floating point types (e.g. `Fp!128`), which provides additional accuracy for
extreme values of `k`, `r`, or `p`. Complements and coefficient ratios retain
the supplied `Fp` precision rather than converting through hardware floating
point. Complements, powers, and coefficient products retain 192 extra working
bits. Separate intermediate exponents preserve representable final masses across
intermediate overflow and underflow, with subnormal rounding at the final
exponent. This is guarded arithmetic, not a universal correct-rounding guarantee.
The $(D r == 1) case shares the geometric PMF implementation.

The coefficient requires `O(min(k, r - 1))` iterations. Powers use logarithmic
time in their exponents, so a large count is practical when the other count is
small. Central probabilities with two enormous counts remain expensive.

The success and failure counts have the same interpretation as in the built-in
floating-point overload. Extended precision does not change the support or
endpoint conventions.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D p) raised to $(D r)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k == 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D 0)))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialPMF(T)(const size_t k, const size_t r, const T p)
    if (is(T == Fp!size, size_t size))
    in (r > 0, "number of successes must be larger than zero")
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpNegativeBinomial;
    return fpNegativeBinomial(k, r, p);
}

/// The probability type selects the extended-precision implementation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;

    auto probability = Fp!128(0.5);
    auto mass = negativeBinomialPMF(1, 2, probability);
    static assert(is(typeof(mass) == Fp!128));
    assert(cast(double)mass == 0.25);
    static assert(is(typeof(negativeBinomialPMF(1, 2, 0.5)) == double));
}

/++
Compatibility name for the `Fp` overload of $(LREF negativeBinomialPMF).
Existing calls remain supported; new code can use `negativeBinomialPMF` with
either a built-in floating-point probability or an `Fp` probability.

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = success probability as an `Fp` value
+/
@safe pure nothrow @nogc
T fp_negativeBinomialPMF(T)(const size_t k, const size_t r, const T p)
    if (is(T == Fp!size, size_t size))
{
    return negativeBinomialPMF(k, r, p);
}

// Exact dyadic probabilities exercise precision unavailable through double.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        UInt!bits allOnes;
        allOnes.data[] = size_t.max;
        const T p = T(false, -cast(long)bits, allOnes); // 1 - 2^-bits
        const T mass = fp_negativeBinomialPMF(1, 1, p);
        assert(mass.coefficient == p.coefficient);
        assert(mass.exponent == -2 * cast(long)bits);
        assert(!mass.sign);
        // C(14,10)*(1/4)^5*(3/4)^10 = 59108049/2^30 exactly.
        T expected = T(59108049);
        expected.exponent -= 30;
        const T dyadic = fp_negativeBinomialPMF(10, 5, T(0.25));
        assert(dyadic.coefficient == expected.coefficient);
        assert(dyadic.exponent == expected.exponent);
        // C(200,100)/2^201 tests the large-coefficient path. Compare at the
        // requested precision, so conversion through real cannot pass.
        enum exactCoefficient = UInt!256("90548514656103281165404177077484163874504589675413336841320");
        T centralExpected = T(exactCoefficient);
        centralExpected.exponent -= 201;
        const T central = fp_negativeBinomialPMF(100, 101, T(0.5));
        assert(central.exponent == centralExpected.exponent);
        assert(central.coefficient >= centralExpected.coefficient - UInt!bits(16));
        assert(central.coefficient <= centralExpected.coefficient + UInt!bits(16));
        assert(cast(double)fp_negativeBinomialPMF(1, 2, T(0.5)) == 0.25);
        assert(cast(double)fp_negativeBinomialPMF(0, size_t.max, T(1)) == 1);
        assert(!fp_negativeBinomialPMF(1, size_t.max, T(1)).coefficient);
        assert(!fp_negativeBinomialPMF(0, 2, T(0)).coefficient);
        // With no failures and one required success, the mass is p itself,
        // even at the Fp exponent floor where multiplication can underflow.
        auto smallestExponent = T(1);
        smallestExponent.exponent = long.min;
        const identityMass = negativeBinomialPMF(0, 1, smallestExponent);
        assert(identityMass.coefficient == smallestExponent.coefficient);
        assert(identityMass.exponent == smallestExponent.exponent);
        assert(!identityMass.sign);
        assert(fpProbabilityValid(T(0)) && fpProbabilityValid(T(1)));
        assert(!fpProbabilityValid(T(-0.5)) && !fpProbabilityValid(T(2)));
        assert(!fpProbabilityValid(T(double.infinity)));
        // A negative value too small to survive a double conversion is invalid.
        assert(!fpProbabilityValid(T(true, -2000, p.coefficient)));
        // This coefficient/exponent represents a value just above 1.
        assert(!fpProbabilityValid(T(false, 1 - cast(long)bits, allOnes)));
        const T tiny = T(false, -3 * cast(long)bits, p.coefficient);
        assert(cast(double)fpProbabilityComplement(tiny) == 1);
        T spacing = T(1);
        spacing.exponent -= bits;
        const T predecessor = fpProbabilityComplement(spacing);
        assert(predecessor.coefficient == allOnes && predecessor.exponent == -cast(long)bits);
        --spacing.exponent; // Exactly halfway between 1 and its predecessor.
        const T halfway = fpProbabilityComplement(spacing);
        assert(halfway.coefficient == T(1).coefficient && halfway.exponent == T(1).exponent);
        static if (size_t.sizeof == 8) {
            // k=1 makes the coefficient exactly r. The old uint conversion
            // erased it when r-1 was 2^32. Reference uses exact binary p.
            assert(approxEqual(cast(double)fp_negativeBinomialPMF(1,
                4_294_967_297UL, T(1 - 0x1p-32)),
                0.36787944112861551803393984760514035666, 1e-12, 0.0));
            // k+r-1 needs 65 bits. Underflow of this astronomically small mass
            // is expected; wrapping the coefficient input or asserting is not.
            assert(!fp_negativeBinomialPMF(2, size_t.max, T(0.5)).coefficient);
        }
    }}
}

/// Extended precision provides accurate values for large counts.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    1.fp_negativeBinomialPMF(1_000_000, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(1, 1_000_000, 0.75);
}


// test more values
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    0.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(0, val + 5, 0.75);
    1.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(1, val + 5, 0.75);
    2.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(2, val + 5, 0.75);
    5.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(5, val + 5, 0.75);
    (val / 2).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val / 2, val + 5, 0.75);
    (val - 5).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 5, val + 5, 0.75);
    (val - 2).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 2, val + 5, 0.75);
    (val - 1).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 1, val + 5, 0.75);
    (val - 0).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val, val + 5, 0.75);
}

// using Fp!128
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.conv: to;
    import mir.test: shouldApprox;

    for (size_t i; i <= 5; i++) {
        i.fp_negativeBinomialPMF(5, Fp!128(0.50)).to!double.shouldApprox == negativeBinomialPMF(i, 5, 0.50);
        i.fp_negativeBinomialPMF(5, Fp!128(0.75)).to!double.shouldApprox == negativeBinomialPMF(i, 5, 0.75);
    }
}

/// Outcome of $(LREF negativeBinomialTailsChecked).
enum NegativeBinomialTailStatus
{
    /// The calculation exceeds a work limit. The tail fields are unused.
    workLimit,
    /// Both tail fields contain their independently rounded probabilities.
    found,
}

/// Checked Fp probabilities. Read the tails only when status is found.
struct NegativeBinomialTailResult(T)
    if (is(T == Fp!bits, uint bits))
{
    /// Default initialization denotes an unresolved result.
    NegativeBinomialTailStatus status = NegativeBinomialTailStatus.workLimit;
    /// Probability of at most k failures before the rth success.
    T cdf;
    /// Probability of strictly more than k failures before the rth success.
    T ccdf;
}

/++
Computes the negative binomial CDF and strict CCDF for the represented $(D Fp)
probability, independently rounded to nearest with ties to even.

At most k failures before the rth success is equivalent to at least r successes
in the first k+r trials. The calculation therefore exchanges the checked binomial
tails at count r-1, using the original success probability without rounding its
complement.

Use this function when extended-precision tails with a correct-rounding guarantee
are required. The checked calculation reports $(D found) only when both tails
are resolved; read neither probability field otherwise. Each tail is rounded
independently, preserving a small tail even when the other rounds to one.

For built-in floating-point types, use $(LREF negativeBinomialCDF) and
$(LREF negativeBinomialCCDF) separately. Calling the CCDF avoids subtracting an
already-rounded CDF from one. Those functions retain their documented numerical
and approximation behavior; simply returning their results as a pair would not
provide this checked function's rounding guarantee.

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = finite success probability in [0,1]
    maxTerms = maximum number of integer weights to sum (default 256, ceiling 1024)
    maxScale = maximum exponent of the common power-of-two denominator
        (default 16384, ceiling 65536)

Returns:
    At p equal to zero, returns zero CDF and unit CCDF. At p equal to one,
    returns unit CDF and zero CCDF. These endpoints ignore the work budgets.

    Other queries require k+r to fit size_t. The general calculation sums
    min(r,k+1) binomial weights. For p=a/2^^s in lowest terms, the common
    denominator is 2^^((k+r)*s). Returns $(D workLimit) when the count sum
    overflows or either budget is exceeded. Certain extreme tails can be
    resolved even with zero budgets. Workspace is fixed-capacity stack storage,
    as in $(REF binomialTailsChecked, mir,stat,distribution,binomial).
+/
@safe pure nothrow @nogc
NegativeBinomialTailResult!T negativeBinomialTailsChecked(T)(size_t k, size_t r, T p,
    size_t maxTerms = 256, size_t maxScale = 16384)
    if (is(T == Fp!bits, uint bits))
    in (r > 0, "number of successes must be larger than zero")
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpProbabilityLess;
    import mir.stat.distribution.binomial: binomialTailsChecked, BinomialTailStatus;
    alias Result = NegativeBinomialTailResult!T;
    enum found = NegativeBinomialTailStatus.found;
    if (!p.coefficient)
        return Result(found, T(0), T(1));
    if (!fpProbabilityLess(p, T(1)))
        return Result(found, T(1), T(0));
    // Guard before addition, including when contracts are disabled.
    if (k > size_t.max - r)
        return Result.init;
    const tails = binomialTailsChecked(r - 1, k + r, p, maxTerms, maxScale);
    if (tails.status != BinomialTailStatus.found)
        return Result.init;
    return Result(found, tails.ccdf, tails.cdf);
}

/// Retain a small strict upper tail even when the rounded CDF is one.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    const result = negativeBinomialTailsChecked(99, 1, T(0.5));
    assert(result.status == NegativeBinomialTailStatus.found && result.cdf == T(1));
    assert(result.ccdf == T(false, -163, UInt!64(1UL << 63)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias R = NegativeBinomialTailResult!T;
    alias S = NegativeBinomialTailStatus;
    alias evaluate = negativeBinomialTailsChecked!T;
    assert(R.init.status == S.workLimit);
    const reference = evaluate(4, 5, T(0.1));
    assert(reference == R(S.found,
        T(false, -74, UInt!64(16829002987672901589UL)), T(false, -64, UInt!64(18430309500479402298UL))));

    assert(evaluate(size_t.max, size_t.max, T(0), 0, 0) == R(S.found, T(0), T(1)));
    assert(evaluate(size_t.max, size_t.max, T(1), 0, 0) == R(S.found, T(1), T(0)));
    assert(evaluate(0, 1, T(false, -3, UInt!64(8)), 0, 0) == R(S.found, T(1), T(0)));
    assert(evaluate(size_t.max, 1, T(0.5)).status == S.workLimit);
    assert(evaluate(1, size_t.max, T(0.5)).status == S.workLimit);
    assert(evaluate(0, size_t.max, T(false, long.min, UInt!64(1)), 0, 0)
        == R(S.found, T(0), T(1)));
    assert(evaluate(2, 3, T(0.5)) == R(S.found, T(0.5), T(0.5)));
    assert(evaluate(1, 3, T(0.25), 2, 8) == R(S.found, T(13.0/256), T(243.0/256)));
    assert(evaluate(1, 3, T(0.25), 1, 8).status == S.workLimit);
    assert(evaluate(1, 3, T(0.25), 2, 7).status == S.workLimit);
    assert(evaluate(1, 3, T(false, -8, UInt!64(64)), 2, 8) == evaluate(1, 3, T(0.25)));
    assert(evaluate(0, 3, T(0.25)) == R(S.found, T(1.0/64), T(63.0/64)));
    assert(evaluate(3, 1, T(0.25)) == R(S.found, T(175.0/256), T(81.0/256)));
    assert(evaluate(0, 1, T(0.25), 0, 0) == R(S.found, T(0.25), T(0.75)));
    const quantum = T(false, long.min, UInt!64(1));
    assert(evaluate(0, 1, quantum, 0, 0) == R(S.found, quantum, T(1)));
    assert(evaluate(0, 2, quantum, 0, 0) == R(S.found, T(0), T(1)));
    // The complement of 2^-65 is exactly halfway below one at64 bits.
    assert(evaluate(0, 65, T(0.5), 0, 0) ==
        R(S.found, T(false, -128, UInt!64(1UL << 63)), T(1)));
    const smallLower = evaluate(0, 100, T(0.5));
    assert(smallLower == R(S.found, T(false, -163, UInt!64(1UL << 63)), T(1)));
    assert(evaluate(1, 3, T(false, -40000, UInt!64(3)), size_t.max, size_t.max).status == S.workLimit);
    assert(evaluate(1024, 1025, T(0.5), size_t.max, size_t.max).status == S.workLimit);
}

version(mir_stat_test)
unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        alias T = Fp!64;
        assertThrown!AssertError(negativeBinomialTailsChecked(0, 0, T(0.5)));
        foreach (p; [-1.0, 2.0, double.infinity, double.nan])
            assertThrown!AssertError(negativeBinomialTailsChecked(0, 1, T(p)));
    }
}

version(mir_stat_test)
version(X86)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [32, 96]) {{
        alias T = Fp!bits;
        T p = T(UInt!bits(1));
        p.exponent -= 2;
        T lower = T(UInt!bits(13));
        lower.exponent -= 8;
        T upper = T(UInt!bits(243));
        upper.exponent -= 8;
        const result = negativeBinomialTailsChecked(1, 3, p);
        assert(result.status == NegativeBinomialTailStatus.found);
        assert(result.cdf == lower && result.ccdf == upper);
    }}
}

// Direct negative-binomial PMF sums evaluated as exact rational numbers.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!128;
        enum p = T(false, -128, UInt!128.fromHexString("80000000000000000000000000000001"));
        enum lower = T(false, -128, UInt!128.fromHexString("80000000000000000000000000000002"));
        enum upper = T(false, -129, UInt!128.fromHexString("fffffffffffffffffffffffffffffffb"));
        const result = negativeBinomialTailsChecked(4, 5, p);
        assert(result.status == NegativeBinomialTailStatus.found);
        assert(result.cdf == lower && result.ccdf == upper);
    }
    {
        alias T = Fp!256;
        enum p = T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001"));
        enum lower = T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000002"));
        enum upper = T(false, -257, UInt!256.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffb"));
        const result = negativeBinomialTailsChecked(4, 5, p);
        assert(result.status == NegativeBinomialTailStatus.found);
        assert(result.cdf == lower && result.ccdf == upper);
    }
    {
        alias T = Fp!512;
        enum p = T(false, -512, UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"));
        enum lower = T(false, -512, UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002"));
        enum upper = T(false, -513, UInt!512.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffb"));
        const result = negativeBinomialTailsChecked(4, 5, p);
        assert(result.status == NegativeBinomialTailStatus.found);
        assert(result.cdf == lower && result.ccdf == upper);
    }
}

/++
Computes the negative binomial cumulative distribution function (CDF).

Returns the inclusive lower-tail probability $(D P(K <= k)). For $(D r == 1),
this is the geometric CDF.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D p) raised to $(D r)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D 1)))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialCDF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow;
    import std.mathspecial: betaIncomplete;

    if (p == 0 || p == 1)
        return p;
    if (r == 1) {
        import mir.stat.distribution.geometric: geometricCDF;
        return geometricCDF(k, p);
    }
    if (k == 0) {
        return pow(p, r);
    }
    if (useNegativeBinomialCountTail(k, r, p)) {
        import mir.stat.distribution.binomial: binomialCCDF;
        return binomialCCDF(r - 1, k + r, p);
    }
    // betaIncomplete takes real shapes. Convert counts directly to its working
    // type rather than rounding through T, and add after conversion to avoid
    // wrapping the largest size_t count.
    return betaIncomplete(cast(real)r, cast(real)k + 1, p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialCDF(6, 3.0 / 4).shouldApprox == 0.9218731;
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    
    static double sumOfnegativeBinomialPMFs(T)(size_t k, size_t r, T p) {
        double result = 0.0;
        for (size_t i; i <= k; i++) {
            result += negativeBinomialPMF(i, r, p);
        }
        return result;
    }

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialCDF(5, 0.25).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.25);
        i.negativeBinomialCDF(5, 0.50).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.50);
        i.negativeBinomialCDF(5, 0.75).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.75);

        i.negativeBinomialCDF(6, 0.25).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.25);
        i.negativeBinomialCDF(6, 0.5).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.5);
        i.negativeBinomialCDF(6, 0.75).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.75);
    }
}

/++
Computes the negative binomial complementary cumulative distribution function (CCDF).

Returns the strict upper-tail probability $(D P(K > k)). For $(D r == 1),
this is the geometric CCDF.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D 1) minus $(D p) raised to $(D r)))
    $(TR $(TD $(D p == 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1)) $(TD $(D 0)))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialCCDF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    import std.mathspecial: betaIncomplete;

    if (p == 0 || p == 1)
        return 1 - p;
    if (r == 1) {
        import mir.stat.distribution.geometric: geometricCCDF;
        return geometricCCDF(k, p);
    }
    if (k == 0) {
        return oneMinusExpNeg(-cast(T)r * log(p));
    }
    if (useNegativeBinomialCountTail(k, r, p)) {
        import mir.stat.distribution.binomial: binomialCDF;
        return binomialCDF(r - 1, k + r, p);
    }
    // Preserve count precision in the existing real-valued beta backend.
    return betaIncomplete(cast(real)k + 1, cast(real)r, 1 - p);
}

// For tiny p, rounding 1-p before incomplete-beta evaluation can move a
// large-count quantile by several integers. Below sqrt(epsilon), that input
// error can exceed sqrt(epsilon) relative to p. Sum the equivalent binomial
// tail instead, retaining p itself. Limit r so this fallback stays inexpensive;
// the binomial tail walks away from its mean and stops at negligible terms.
private @safe pure nothrow @nogc
bool useNegativeBinomialCountTail(T)(size_t k, size_t r, T p)
{
    import mir.math.common: sqrt;
    return p < sqrt(T.epsilon) && r <= 64 && k <= size_t.max - r;
}

// Lower tails must not disappear through subtraction, or count addition wrap.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown;
    import mir.math.common: approxEqual;
    // With p=1/2 and k=r-1 the beta shapes are equal. Counts beyond float's
    // consecutive-integer precision must still reach the backend unchanged.
    // Compare input types: older Phobos versions have an independent large-
    // shape error here, so they do not reliably return the exact value 1/2.
    assert(negativeBinomialCDF(size_t(16_777_217), size_t(16_777_218), 0.5f)
        == cast(float)negativeBinomialCDF(size_t(16_777_217), size_t(16_777_218), 0.5L));
    assert(negativeBinomialCCDF(size_t(16_777_217), size_t(16_777_218), 0.5f)
        == cast(float)negativeBinomialCCDF(size_t(16_777_217), size_t(16_777_218), 0.5L));
    static foreach (T; AliasSeq!(float, double, real)) {{
        const T p = T.epsilon / 16;
        assert(approxEqual(negativeBinomialCDF(1, 1, p), 2 * p, 16 * T.epsilon, T(0)));
        // P(K<=1) = p^2*(3-2p) for r=2.
        assert(approxEqual(negativeBinomialCDF(1, 2, p), p*p*(3-2*p),
            128 * T.epsilon, T(0)));
        const T nearOne = nextDown(T(1));
        const T failure = 1 - nearOne;
        assert(approxEqual(negativeBinomialCCDF(0, 2, nearOne),
            failure * (2 - failure), 16 * T.epsilon, T(0)));
        foreach (r; [size_t(1), size_t(6)]) {
            foreach (k; [size_t(0), size_t(1), size_t.max]) {
                assert(negativeBinomialCDF(k, r, T(0)) == 0);
                assert(negativeBinomialCDF(k, r, T(1)) == 1);
                assert(negativeBinomialCCDF(k, r, T(0)) == 1);
                assert(negativeBinomialCCDF(k, r, T(1)) == 0);
            }
        }
        assert(negativeBinomialCDF(size_t.max, 1, T(0.5)) == 1);
        assert(negativeBinomialCCDF(size_t.max, 1, T(0.5)) == 0);
    }}
    static if (size_t.sizeof == 8) {
        // Independent 100-digit finite binomial-tail sum for binary p=1e-8.
        enum double cdf = 0.550000000007872962468353187594624191;
        assert(approxEqual(negativeBinomialCDF(2529614601UL, 25, 1e-8),
            cdf, 256 * double.epsilon, 0.0));
        assert(approxEqual(negativeBinomialCCDF(2529614601UL, 25, 1e-8),
            1 - cdf, 256 * double.epsilon, 0.0));
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialCCDF(6, 3.0 / 4).shouldApprox == 0.07812691;
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialCCDF(5, 0.25).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.25);
        i.negativeBinomialCCDF(5, 0.50).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.50);
        i.negativeBinomialCCDF(5, 0.75).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.75);

        i.negativeBinomialCCDF(6, 0.25).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.25);
        i.negativeBinomialCCDF(6, 0.5).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.5);
        i.negativeBinomialCCDF(6, 0.75).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.75);
    }
}

/// Outcome of $(LREF negativeBinomialInvCDFChecked).
enum NegativeBinomialInverseStatus
{
    /// The exact calculation exceeds a work limit. The count field is unused.
    workLimit,
    /// The count satisfies the quantile or endpoint convention.
    found,
}

/// Read count when status is found. Default initialization is unresolved.
struct NegativeBinomialInverseResult
{
    /// Outcome of the query.
    NegativeBinomialInverseStatus status = NegativeBinomialInverseStatus.workLimit;
    /// Quantile or endpoint value when found; unused otherwise.
    size_t count;
}

/++
Computes a checked negative binomial inverse CDF for represented Fp probabilities.
For interior probabilities, returns the least failure count whose exact CDF
reaches q. Cumulative integer weights are compared before probabilities are rounded.

Params:
    q = finite cumulative probability in [0,1]
    r = positive number of successes required before stopping
    p = finite success probability in [0,1], with the same Fp width as q
    maxTerms = maximum number of failure counts visited from zero (default 256, ceiling 1024)
    maxScale = maximum exponent of the common power-of-two denominator
        (default 16384, ceiling 65536)

Returns:
    Zero when q is zero or p is one. Otherwise q equal to one or p equal to
    zero returns size_t.max. These endpoint conventions match the existing
    count inverse and ignore the work budgets.

    For p=a/2^^s in lowest terms, the cumulative probability at k failures
    has denominator 2^^((r+k)*s). The scan starts at zero and updates exact
    integer weights and the denominator until the query is reached. Returns
    $(D workLimit) when either budget is exhausted, without an approximate
    count. Fixed-capacity workspace is allocated on the stack.
+/
@safe pure nothrow @nogc
NegativeBinomialInverseResult negativeBinomialInvCDFChecked(uint bits)(
    Fp!bits q, size_t r, Fp!bits p, size_t maxTerms = 256, size_t maxScale = 16384)
    in (r > 0, "number of successes must be larger than zero")
    in (fpProbabilityValid(q), "q must be a finite probability in [0,1]")
    in (fpProbabilityValid(p), "p must be a finite probability in [0,1]")
{
    import mir.stat.internal.fp_binomial_tails: exactNegativeBinomialInverse;
    return exactNegativeBinomialInverse(q, r, p, maxTerms, maxScale);
}

/// An exact CDF jump and its next represented query select different counts.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    const q = T(0.5);
    T above = q;
    above.coefficient += 1;
    const tie = negativeBinomialInvCDFChecked(q, 3, T(0.5));
    const next = negativeBinomialInvCDFChecked(above, 3, T(0.5));
    assert(tie.status == NegativeBinomialInverseStatus.found && tie.count == 2);
    assert(next.status == NegativeBinomialInverseStatus.found && next.count == 3);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias R = NegativeBinomialInverseResult;
    alias S = NegativeBinomialInverseStatus;
    alias evaluate = negativeBinomialInvCDFChecked!64;
    assert(R.init.status == S.workLimit);
    assert(evaluate(T(0), size_t.max, T(0), 0, 0) == R(S.found, 0));
    assert(evaluate(T(1), size_t.max, T(1), 0, 0) == R(S.found, 0));
    assert(evaluate(T(0.5), 3, T(0), 0, 0) == R(S.found, size_t.max));
    assert(evaluate(T(1), 3, T(0.5), 0, 0) == R(S.found, size_t.max));
    assert(evaluate(T(0.5), 3, T(false, -3, UInt!64(8)), 0, 0) == R(S.found, 0));
    assert(evaluate(T(false, -3, UInt!64(8)), 3, T(0.5), 0, 0) == R(S.found, size_t.max));
    assert(evaluate(T(0.5), 3, T(0.5), 0).status == S.workLimit);
    assert(evaluate(T(0.5), 3, T(0.5), 3, 5) == R(S.found, 2));
    assert(evaluate(T(0.5), 3, T(0.5), 2, 5).status == S.workLimit);
    assert(evaluate(T(0.5), 3, T(0.5), 3, 4).status == S.workLimit);
    assert(evaluate(T(0.5), 3, T(0.5), 3, 2).status == S.workLimit);
    assert(evaluate(T(0.5), 3, T(false, -3, UInt!64(4)), 3, 5) == R(S.found, 2));
    assert(evaluate(T(false, -3, UInt!64(4)), 3, T(0.5)) == R(S.found, 2));
    assert(evaluate(T(0.5), size_t.max, T(0.5), size_t.max, size_t.max).status == S.workLimit);
    assert(evaluate(T(0.5), 3, T(false, long.min, UInt!64(1))).status == S.workLimit);
    assert(evaluate(T(false, long.min, UInt!64(1)), 3, T(0.5)) == R(S.found, 0));
    // q has many more fractional bits than the exact denominator.
    T tiny = T(1); tiny.exponent -= 100;
    assert(evaluate(tiny, 3, T(0.5)) == R(S.found, 0));
    T nearOne = T(false, -64, UInt!64(ulong.max));
    assert(evaluate(nearOne, 1, T(0.5), 64) == R(S.found, 63));
    assert(evaluate(nearOne, 1, T(0.5), 63).status == S.workLimit);
    assert(evaluate(T(0.5), 1, T(1.0/2048), size_t.max, size_t.max).status == S.workLimit);
    foreach (r; [4097, 16385, 65536]) {
        const q = T(false, -r, UInt!64(1));
        assert(evaluate(q, r, T(0.5), 1, r) == R(S.found, 0));
        assert(evaluate(q, r, T(0.5), 1, r - 1).status == S.workLimit);
    }
    assert(evaluate(T(false, -65537, UInt!64(1)), 65537, T(0.5), size_t.max, size_t.max).status == S.workLimit);
}

version(mir_stat_test)
unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        alias T = Fp!64;
        assertThrown!AssertError(negativeBinomialInvCDFChecked(T(0.5), 0, T(0.5)));
        foreach (value; [-1.0, 2.0, double.infinity, double.nan]) {
            assertThrown!AssertError(negativeBinomialInvCDFChecked(T(value), 1, T(0.5)));
            assertThrown!AssertError(negativeBinomialInvCDFChecked(T(0.5), 1, T(value)));
        }
    }
}

// Adjacent queries bracketing an exact rational CDF for a full-width input.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!128;
        enum p = T(false, -128, UInt!128.fromHexString("80000000000000000000000000000001"));
        enum q = T(false, -128, UInt!128.fromHexString("c6000000000000000000000000000001"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 4);
    }
    {
        alias T = Fp!128;
        enum p = T(false, -128, UInt!128.fromHexString("80000000000000000000000000000001"));
        enum q = T(false, -128, UInt!128.fromHexString("c6000000000000000000000000000002"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 5);
    }
    {
        alias T = Fp!256;
        enum p = T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001"));
        enum q = T(false, -256, UInt!256.fromHexString("c600000000000000000000000000000000000000000000000000000000000001"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 4);
    }
    {
        alias T = Fp!256;
        enum p = T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001"));
        enum q = T(false, -256, UInt!256.fromHexString("c600000000000000000000000000000000000000000000000000000000000002"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 5);
    }
    {
        alias T = Fp!512;
        enum p = T(false, -512, UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"));
        enum q = T(false, -512, UInt!512.fromHexString("c6000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 4);
    }
    {
        alias T = Fp!512;
        enum p = T(false, -512, UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"));
        enum q = T(false, -512, UInt!512.fromHexString("c6000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000002"));
        const result = negativeBinomialInvCDFChecked(q, 3, p);
        assert(result.status == NegativeBinomialInverseStatus.found && result.count == 5);
    }
}

/++
Computes the negative binomial inverse cumulative distribution function (InvCDF).

For $(D 0 < q && q < 1) and $(D 0 < p && p < 1), returns the smallest
representable count whose CDF reaches $(D q), or $(D size_t.max) when the
quantile is beyond the count range. Close comparisons use bounded exact
binomial arithmetic when the binary input fractions fit its workspace;
other interior results remain subject to incomplete-beta tail accuracy.

Zero queries and the point mass at $(D p == 1) take precedence over the
$(D size_t.max) convention. That sentinel cannot distinguish an exact maximum
count from an unbounded or out-of-range quantile.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 0)) $(TD $(D 0), including when $(D p == 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D 0), including when $(D q == 1)))
    $(TR $(TD $(D q == 1) and $(D p < 1)) $(TD $(D size_t.max)))
    $(TR $(TD $(D q > 0) and $(D p == 0)) $(TD $(D size_t.max)))
    $(TR $(TD $(D q < 0), $(D q > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    q = cumulative probability in $(D [0, 1])
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
size_t negativeBinomialInvCDF(T)(const T q, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: sqrt, floor;
    import mir.stat.distribution.normal: normalInvCDF;
    import mir.stat.distribution.geometric: geometricInvCDF;

    if (q == 0 || p == 1)
        return 0;
    if (q == 1 || p == 0)
        return size_t.max;
    if (r == 1) {
        const real value = geometricInvCDF(cast(real)q, cast(real)p);
        return value >= cast(real)size_t.max ? size_t.max : cast(size_t)value;
    }

    static if (is(T == float)) alias F = double;
    else alias F = T;
    const F probability = p;
    const F query = q;
    const bool upper = query > 0.5;
    const F target = upper ? 1 - query : query;
    bool reached(size_t k) {
        import std.math: abs;
        import mir.stat.distribution.binomial: binomialExactQuantileReached;
        const F value = upper ? negativeBinomialCCDF(k, r, probability)
                              : negativeBinomialCDF(k, r, probability);
        if (k < 8192 && r <= 8192 - k && abs(value - target) <=
            128 * (k + r + 1) * F.epsilon * (abs(value) + target + F.min_normal)) {
            // K<=k is equivalent to at least r successes in k+r trials.
            // Reuse the bounded exact comparison at ambiguous CDF steps.
            bool supported;
            const bool exact = binomialExactQuantileReached(r - 1, k + r,
                probability, query, supported, true);
            if (supported)
                return exact;
        }
        return upper ? value <= target : value >= target;
    }

    const F mu = r * (1 - probability) / probability;
    const F root = sqrt(r * (1 - probability));
    const F sigma = root / probability;
    const F z = normalInvCDF(query);
    F estimate;
    if (r > 20 && probability > 0.25 && probability < 0.75)
        estimate = mu + sigma * z - 0.5;
    else
        estimate = mu + sigma * (z + ((2 - probability) / root) * (z*z - 1) / 6);
    // Clamp before conversion, including an indeterminate extreme estimate.
    // The estimate only chooses a starting point; tail comparisons decide.
    size_t guess = !(estimate > 0) ? 0
        : estimate >= cast(F)size_t.max ? size_t.max : cast(size_t)floor(estimate);
    size_t low, high;
    size_t step = 1;
    if (reached(guess)) {
        high = guess;
        while (true) {
            if (high == 0)
                return 0;
            const size_t next = step < high ? high - step : 0;
            if (!reached(next)) {
                low = next;
                break;
            }
            high = next;
            step = step <= size_t.max / 2 ? step * 2 : size_t.max;
        }
    } else {
        low = guess;
        while (true) {
            // Saturate an unrepresentable quantile rather than wrapping a count.
            if (low == size_t.max)
                return size_t.max;
            const size_t next = step < size_t.max - low ? low + step : size_t.max;
            if (reached(next)) {
                high = next;
                break;
            }
            low = next;
            step = step <= size_t.max / 2 ? step * 2 : size_t.max;
        }
    }
    // Invariant: CDF(low)<q<=CDF(high), with upper queries compared as tails.
    // Bisect without adding endpoints, which could overflow.
    while (high - low > 1) {
        const size_t middle = low + (high - low) / 2;
        if (reached(middle)) high = middle;
        else low = middle;
    }
    return high;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    0.9.negativeBinomialInvCDF(6, 3.0 / 4).should == 4;
}

// Endpoint precedence, negative initial estimates, and exact CDF steps.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    static foreach (T; AliasSeq!(float, double, real)) {{
        assert(negativeBinomialInvCDF(T(0), 5, T(0)) == 0);
        assert(negativeBinomialInvCDF(T(0.5), 5, T(0)) == size_t.max);
        assert(negativeBinomialInvCDF(T(0.5), 5, T(1)) == 0);
        assert(negativeBinomialInvCDF(T(1), 5, T(1)) == 0);
        assert(negativeBinomialInvCDF(T(1), 5, T(0.5)) == size_t.max);
        assert(negativeBinomialInvCDF(T(1e-20L), 25, T(0.5)) == 0);
        assert(negativeBinomialInvCDF(T(0.5), 1, T(1e-30L)) == size_t.max);
        assert(negativeBinomialInvCDF(T(0.5), 25, T(0.5)) == 24);
        // For r=2, p=1/2, CDF(4)=57/64 exactly.
        const T q = T(57) / 64;
        assert(negativeBinomialInvCDF(nextDown(q), 2, T(0.5)) == 4);
        assert(negativeBinomialInvCDF(q, 2, T(0.5)) == 4);
        assert(negativeBinomialInvCDF(nextUp(q), 2, T(0.5)) == 5);
        assert(negativeBinomialInvCDF(nextUp(T(0.75)), 1, T(0.5)) == 2);
    }}
    // The binary double 0.0001 is slightly larger than the square of the
    // binary double 0.01: an exact rational comparison puts this query at 1.
    assert(negativeBinomialInvCDF(0.0001, 2, 0.01) == 1);
    // Independent 100-digit references for a skewed, wider distribution.
    assert(negativeBinomialInvCDF(0.5, 100, 0.01) == 9867);
    static if (size_t.sizeof == 8) {
        // 100-digit binomial-tail summation for the exact binary p and q.
        // CDF(k-1)=0.5499999992233198939..., CDF(k)=0.5500000000078729624....
        assert(negativeBinomialInvCDF(0.55, 25, 1e-8) == 2529614601UL);
    }
    assert(negativeBinomialInvCDF(0.5, 6, 1e-30) == size_t.max);
}

//
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    0.negativeBinomialInvCDF(5, 0.6).should == 0;
    1.negativeBinomialInvCDF(5, 0.6).should == size_t.max;

    for (double x = 0.05; x < 1; x = x + 0.05) {
        size_t value = x.negativeBinomialInvCDF(5, 0.6);
        value.negativeBinomialCDF(5, 0.6).should!"a >= b"(x);
        if (value > 0) {
            (value - 1).negativeBinomialCDF(5, 0.6).should!"a < b"(x);
        }
    }
}

// alternate guess paths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    static immutable size_t[] ns = [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.05; x < 1; x = x + 0.05) {
            value = x.negativeBinomialInvCDF(ns[i], ps[i]);
            negativeBinomialCDF(value, ns[i], ps[i]).should!"a >= b"(x);
            if (value > 0) {
                negativeBinomialCDF(value - 1, ns[i], ps[i]).should!"a < b"(x);
            }
        }
    }
}

// more detailed guess paths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    static immutable size_t[] ns = [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.01; x < 1; x = x + 0.01) {
            value = x.negativeBinomialInvCDF(ns[i], ps[i]);
            negativeBinomialCDF(value, ns[i], ps[i]).should!"a >= b"(x);
            if (value > 0) {
                negativeBinomialCDF(value - 1, ns[i], ps[i]).should!"a < b"(x);
            }
        }
    }
}

/// Inverse endpoints use an integer sentinel, except for the deterministic outcome.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(negativeBinomialInvCDF(0.0, 2, 0.5) == 0);
    assert(negativeBinomialInvCDF(1.0, 2, 0.5) == size_t.max);
    assert(negativeBinomialInvCDF(0.5, 2, 0.0) == size_t.max);
    assert(negativeBinomialInvCDF(1.0, 2, 1.0) == 0);
}

/++
Computes the negative binomial log probability mass function (LPMF).

Returns the natural logarithm of $(D P(K == k)).

The $(D Fp) overload evaluates the log mass directly with guarded count and
probability logarithms. Both counts may span the full $(D size_t) domain
(subject to $(D r > 0)), including when their sum exceeds $(D size_t.max).
The centered calculation preserves accuracy near the mode and can return a
finite log mass when the PMF underflows. Universal correct rounding is not
guaranteed.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) and $(D p > 0)) $(TD $(D r * log(p))))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 1) and $(D k == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D r == 0)) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures before stopping
    r = positive number of successes required before stopping
    p = per-trial success probability in $(D [0, 1])

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialLPMF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.internal.xlogy: xlogy, xlog1py;
    import mir.math.internal.log_binomial: logBinomialCoefficient;

    if (k <= 64 && r - 1 <= 64 - k)
        return logBinomialCoefficient(k + r - 1, cast(uint)(r - 1))
            + xlog1py(k, -p) + xlogy(r, p);
    return cast(T)negativeBinomialLogMass(k, r, cast(real)p);
}

/// ditto
@safe pure nothrow @nogc
T negativeBinomialLPMF(T)(const size_t k, const size_t r, const T p)
    if (is(T == Fp!bits, uint bits))
    in (r > 0, "number of successes must be larger than zero")
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.distribution.geometric: geometricLPMF;
    import mir.stat.distribution.binomial: binomialLPMF;
    import mir.stat.internal.fp_binomial_log: BinomialLog;
    import mir.stat.internal.fp_probability: fpProbabilityLess;

    if (r == 1) return geometricLPMF(k,p);
    if (k == 0) return binomialLPMF(r,r,p); // All r trials succeed.
    if (!p.coefficient || !fpProbabilityLess(p,T(1))) return T(-double.infinity);
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    return BinomialLog!bits.negativeMass(k,r,p);
}

/// With one required success, the log mass agrees with the geometric distribution.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    import mir.stat.distribution.geometric: geometricLPMF;
    auto p = Fp!128(0.25);
    assert(negativeBinomialLPMF(7,1,p) == geometricLPMF(7,p));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!128;
    assert(negativeBinomialLPMF(0,3,T(1)) == T(0));
    assert(negativeBinomialLPMF(2,3,T(1)) == T(-double.infinity));
    assert(negativeBinomialLPMF(0,3,T(0)) == T(-double.infinity));
    assert(negativeBinomialLPMF(2,3,T(0)) == T(-double.infinity));
    debug
    {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        assertThrown!AssertError(negativeBinomialLPMF(0,0,T(1)));
        foreach (invalid; [T(-0.25),T(2),T(double.infinity),T(double.nan)])
            assertThrown!AssertError(negativeBinomialLPMF(0,1,invalid));
    }
}

// BEGIN public negative binomial log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { ulong k; ulong successes; T p; T value; }
        enum references = [
            Reference(0UL,3UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,1,UInt!64.fromHexString("851591f9dd5b9b41"))),
            Reference(0UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,-64,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(2UL,3UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,1,UInt!64.fromHexString("851591f9dd5b9b40"))),
            Reference(7UL,12UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,-55,UInt!64.fromHexString("9614bb226a60fe2d"))),
            Reference(57UL,19UL,T(false,-4,UInt!64.fromHexString("3")),T(true,-61,UInt!64.fromHexString("91572451b6b80573"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("bb91dcc9ca4664bc"))),
            Reference(18446744073709551615UL,2UL,T(false,-64,UInt!64.fromHexString("1")),T(true,-58,UInt!64.fromHexString("b57217f7d1cf79ac"))),
            Reference(18446744073709551615UL,18446744073709551615UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("bb91dcc9ca4664bc"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max || r.successes > size_t.max) continue;
            assert(negativeBinomialLPMF(cast(size_t)r.k,cast(size_t)r.successes,r.p) == r.value);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; ulong successes; T p; T value; }
        enum references = [
            Reference(0UL,3UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-63,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903"))),
            Reference(0UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-192,UInt!128.fromHexString("ffffffffffffffff0000000000000000"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!128.fromHexString("ffffffffffffffff")),T(true,-127,UInt!128.fromHexString("80000000000000004000000000000000"))),
            Reference(2UL,3UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-63,UInt!128.fromHexString("851591f9dd5b9b3ff21266e0b1f62772"))),
            Reference(7UL,12UL,T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-118,UInt!128.fromHexString("98ac480da0fb44517d5de15135ecc1bf"))),
            Reference(57UL,19UL,T(false,-4,UInt!128.fromHexString("3")),T(true,-125,UInt!128.fromHexString("91572451b6b80572f96dede89956efeb"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("bb91dcc9ca4664bc16b126e66e238b8b"))),
            Reference(18446744073709551615UL,2UL,T(false,-64,UInt!128.fromHexString("1")),T(true,-122,UInt!128.fromHexString("b57217f7d1cf79abc7e3b39803f2f6af"))),
            Reference(18446744073709551615UL,18446744073709551615UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("bb91dcc9ca4664bc1ab126e66e238b8b"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max || r.successes > size_t.max) continue;
            assert(negativeBinomialLPMF(cast(size_t)r.k,cast(size_t)r.successes,r.p) == r.value);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; ulong successes; T p; T value; }
        enum references = [
            Reference(0UL,3UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-191,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1"))),
            Reference(0UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-448,UInt!256.fromHexString("ffffffffffffffff000000000000000000000000000000000000000000000000"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!256.fromHexString("ffffffffffffffff")),T(true,-255,UInt!256.fromHexString("800000000000000040000000000000002aaaaaaaaaaaaaaacaaaaaaaaaaaaaab"))),
            Reference(2UL,3UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-191,UInt!256.fromHexString("851591f9dd5b9b3ff21266e0b1f6277239b7d1e6515b4eefa8060e8493839992"))),
            Reference(7UL,12UL,T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-245,UInt!256.fromHexString("99f80e833c486763e7028f3b1cb0accc48a50c280a3785bc3512c03432fa9f02"))),
            Reference(57UL,19UL,T(false,-4,UInt!256.fromHexString("3")),T(true,-253,UInt!256.fromHexString("91572451b6b80572f96dede89956efeb5f60e2980e256227a708056045ed31a8"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("bb91dcc9ca4664bc16b126e66e238b8b167494af74fd4560a8762b7c3e71a16a"))),
            Reference(18446744073709551615UL,2UL,T(false,-64,UInt!256.fromHexString("1")),T(true,-250,UInt!256.fromHexString("b57217f7d1cf79abc7e3b39803f2f6af4048987bc7ee0b82df0d175b8baafa2c"))),
            Reference(18446744073709551615UL,18446744073709551615UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("bb91dcc9ca4664bc1ab126e66e238b8b1b7494af74fd4560aecb80d193c6f6c0"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max || r.successes > size_t.max) continue;
            assert(negativeBinomialLPMF(cast(size_t)r.k,cast(size_t)r.successes,r.p) == r.value);
        }
    }
    {
        alias T = Fp!512;
        struct Reference { ulong k; ulong successes; T p; T value; }
        enum references = [
            Reference(0UL,3UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-447,UInt!512.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba0edca58985270c172402ffe3c783b944cb1e302a850ea18f31d816c5ecd0f4a1c"))),
            Reference(0UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-960,UInt!512.fromHexString("ffffffffffffffff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!512.fromHexString("ffffffffffffffff")),T(true,-511,UInt!512.fromHexString("800000000000000040000000000000002aaaaaaaaaaaaaaacaaaaaaaaaaaaaaac4444444444444445999999999999999abe2be2be2be2be2ce2be2be2be2be2c"))),
            Reference(2UL,3UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-447,UInt!512.fromHexString("851591f9dd5b9b3ff21266e0b1f6277239b7d1e6515b4eefa8060e849383999209f7d57563f6a995f70801c33ae271d85edf5d91299fc3fff5bc0ac05d1f5b29"))),
            Reference(7UL,12UL,T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-500,UInt!512.fromHexString("9a9df1be09eef8ed1bd4e6301012a252d0bcf374d73e929206ef1a5226981cf40fdaa6750dacba06b81aee05052ddc02059403544f6636fb54f4321bedd1750b"))),
            Reference(57UL,19UL,T(false,-4,UInt!512.fromHexString("3")),T(true,-509,UInt!512.fromHexString("91572451b6b80572f96dede89956efeb5f60e2980e256227a708056045ed31a83e8a1bf2f6d706df8ab0a7457bff898e50f950119c696a7a1c29b0c145c570d5"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("bb91dcc9ca4664bc16b126e66e238b8b167494af74fd4560a8762b7c3e71a16a22cb7f8f994e7a2b539f4287ff4cf30f44ca928d955bace0484d13c105ad00ba"))),
            Reference(18446744073709551615UL,2UL,T(false,-64,UInt!512.fromHexString("1")),T(true,-506,UInt!512.fromHexString("b57217f7d1cf79abc7e3b39803f2f6af4048987bc7ee0b82df0d175b8baafa2be78542ed3ab87965223ffda5f5a4c5bb97c0f75a538145d67069844da3b2dcb8"))),
            Reference(18446744073709551615UL,18446744073709551615UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("bb91dcc9ca4664bc1ab126e66e238b8b1b7494af74fd4560aecb80d193c6f6bf8040d4e4eea3cf80b38164aa216f153174c75f5a622879ad277c29b264c2f219"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max || r.successes > size_t.max) continue;
            assert(negativeBinomialLPMF(cast(size_t)r.k,cast(size_t)r.successes,r.p) == r.value);
        }
    }
}
// END public negative binomial log references

// Independent high-precision references and full-width count boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        assert(approxEqual(negativeBinomialPMF(100, 100, T(0.5)),
            T(0.0281742395046282111236226320714500749007L), 128 * T.epsilon, T(0)));
        // No failures means r consecutive successes, including beyond the
        // small-coefficient path. At p=1/2 the mass is exactly 2^-r.
        assert(negativeBinomialPMF(0, 100, T(0.5)) == T(0x1p-100L));
        assert(approxEqual(negativeBinomialLPMF(0, 100, T(0.5)),
            T(-69.31471805599453094172321214581765680755L), 8 * T.epsilon, T(0)));
        assert(negativeBinomialLPMF(1, 100, T(0)) == -T.infinity);
        foreach (r; [size_t(1), size_t(100), size_t.max]) {
            assert(negativeBinomialPMF(0, r, T(1)) == 1);
            assert(negativeBinomialPMF(1, r, T(1)) == 0);
            assert(negativeBinomialPMF(0, r, T(0)) == 0);
            assert(negativeBinomialLPMF(1, r, T(1)) == -T.infinity);
        }
        // Their sum exceeds size_t, but the log mass is still finite.
        assert(negativeBinomialLPMF(size_t.max, size_t.max, T(0.5)) < 0);
        assert(negativeBinomialLPMF(size_t.max, size_t.max, T(0.5)) > -100);
    }}
    static if (size_t.sizeof == 8) {
        assert(approxEqual(negativeBinomialLPMF(1, 4_294_967_297UL, 1 - 0x1p-32),
            -1.00000000011641532189921495928864740589, 256 * double.epsilon, 0.0));
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    4.negativeBinomialLPMF(6, 3.0 / 4).exp.shouldApprox == 4.negativeBinomialPMF(6, 3.0 / 4);
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialLPMF(5, 0.5).exp.shouldApprox == negativeBinomialPMF(i, 5, 0.5);
        i.negativeBinomialLPMF(5, 0.75).exp.shouldApprox == negativeBinomialPMF(i, 5, 0.75);
    }
}

/// Accurate values for large values of `n`
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    1.negativeBinomialLPMF(1_000_000, 0.75).shouldApprox == fp_negativeBinomialPMF(1, 1_000_000, Fp!128(0.75)).fp_log!double;
}

// testing more values
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    0.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(0, val + 5, Fp!128(0.75)).fp_log!double;
    1.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(1, val + 5, Fp!128(0.75)).fp_log!double;
    2.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(2, val + 5, Fp!128(0.75)).fp_log!double;
    5.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(5, val + 5, Fp!128(0.75)).fp_log!double;
    (val / 2).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val / 2, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 5).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 5, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 2).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 2, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 1).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 1, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 0).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val, val + 5, Fp!128(0.75)).fp_log!double;
}

// BEGIN negative-binomial references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { ulong k, r; T p, mass; }
        enum references = [
            Reference(0UL, 2UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, 2UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("8000000000000000"))),
            Reference(10UL, 5UL, T(false, -65, UInt!64.fromHexString("8000000000000000")), T(false, -68L, UInt!64.fromHexString("e17ab44000000000"))),
            Reference(100UL, 101UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -69L, UInt!64.fromHexString("e6cda9a862b57059"))),
            Reference(1000UL, 3UL, T(false, -67, UInt!64.fromHexString("8000000000000000")), T(false, -150L, UInt!64.fromHexString("e2fd971892ff2d9e"))),
            Reference(1UL, 2UL, T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -127L, UInt!64.fromHexString("fffffffffffffffe"))),
            Reference(2UL, 3UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -3361L, UInt!64.fromHexString("c000000000000000"))),
            Reference(18446744073709551615UL, 2UL, T(false, -128, UInt!64.fromHexString("8000000000000000")), T(false, -130L, UInt!64.fromHexString("9b4597e37cb04ff4"))),
            Reference(18446744073709551615UL, 3UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -130L, UInt!64.fromHexString("bc5ab1b16779be37"))),
            Reference(1UL, 18446744073709551615UL, T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be35"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max || c.r > size_t.max) continue;
            assert(negativeBinomialPMF(cast(size_t)c.k, cast(size_t)c.r, c.p) == c.mass);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k, r; T p, mass; }
        enum references = [
            Reference(0UL, 2UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, 2UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(10UL, 5UL, T(false, -129, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -132L, UInt!128.fromHexString("e17ab440000000000000000000000000"))),
            Reference(100UL, 101UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -133L, UInt!128.fromHexString("e6cda9a862b570591145badcc1f49f12"))),
            Reference(1000UL, 3UL, T(false, -131, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -214L, UInt!128.fromHexString("e2fd971892ff2d9e4f0f3393d5b17c52"))),
            Reference(1UL, 2UL, T(false, -128, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")), T(false, -255L, UInt!128.fromHexString("fffffffffffffffffffffffffffffffe"))),
            Reference(2UL, 3UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -3425L, UInt!128.fromHexString("c0000000000000000000000000000000"))),
            Reference(18446744073709551615UL, 2UL, T(false, -256, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -322L, UInt!128.fromHexString("ffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, 3UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -194L, UInt!128.fromHexString("bc5ab1b16779be369045998f3be08f6c"))),
            Reference(1UL, 18446744073709551615UL, T(false, -128, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")), T(false, -192L, UInt!128.fromHexString("fffffffffffffffe0000000000000002"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max || c.r > size_t.max) continue;
            assert(negativeBinomialPMF(cast(size_t)c.k, cast(size_t)c.r, c.p) == c.mass);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k, r; T p, mass; }
        enum references = [
            Reference(0UL, 2UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, 2UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(10UL, 5UL, T(false, -257, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -260L, UInt!256.fromHexString("e17ab44000000000000000000000000000000000000000000000000000000000"))),
            Reference(100UL, 101UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -261L, UInt!256.fromHexString("e6cda9a862b570591145badcc1f49f11a32fdc37906d95c68000000000000000"))),
            Reference(1000UL, 3UL, T(false, -259, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -342L, UInt!256.fromHexString("e2fd971892ff2d9e4f0f3393d5b17c51b3d31749987e3ce0a3480461d2e89df6"))),
            Reference(1UL, 2UL, T(false, -256, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")), T(false, -511L, UInt!256.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"))),
            Reference(2UL, 3UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -3553L, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL, 2UL, T(false, -512, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -706L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, 3UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -322L, UInt!256.fromHexString("bc5ab1b16779be369045998f3be08f6c7ae983dee3183acff2330affd070a645"))),
            Reference(1UL, 18446744073709551615UL, T(false, -256, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")), T(false, -448L, UInt!256.fromHexString("fffffffffffffffeffffffffffffffffffffffffffffffff0000000000000002"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max || c.r > size_t.max) continue;
            assert(negativeBinomialPMF(cast(size_t)c.k, cast(size_t)c.r, c.p) == c.mass);
        }
    }
}
// END negative-binomial references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        const p = T(false,long.min/2,UInt!bits(1));
        assert(negativeBinomialPMF(0,2,p) == T(false,long.min,UInt!bits(1)));
        assert(negativeBinomialPMF(2,2,p) == T(false,long.min,UInt!bits(3)));
        const smaller = T(false,long.min/2-1,UInt!bits(1));
        assert(negativeBinomialPMF(1,2,smaller) == T(0));
        assert(negativeBinomialPMF(2,2,smaller) == T(false,long.min,UInt!bits(1)));
        assert(negativeBinomialPMF(5,2,smaller) == T(false,long.min,UInt!bits(1)));
        assert(negativeBinomialPMF(size_t.max,2,T(false,long.min,UInt!bits(1))) == T(0));
    }}
}

// Shared dyadic reference cases exercise scalar and checked APIs together.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: fabs, log;
    enum uint[3][4] references = [[4,4,12],[4,8,8],[3,11,5],[2,13,3]];
    static foreach (T; AliasSeq!(float, double, real, Fp!64)) {{
        foreach (k, row; references) {
            const expectedMass = row[0] / 16.0;
            const expectedCDF = row[1] / 16.0;
            const expectedCCDF = row[2] / 16.0;
            const q = T(expectedCDF);
            enum tolerance = is(T == float) ? 2e-6 : 2e-14;
            assert(fabs(cast(double)negativeBinomialPMF(k, 2, T(0.5)) - expectedMass) <= tolerance);
            assert(fabs(cast(double)negativeBinomialLPMF(k, 2, T(0.5)) - log(expectedMass)) <= tolerance);
            static if (isFloatingPoint!T) {
                assert(fabs(cast(double)negativeBinomialCDF(k, 2, T(0.5)) - expectedCDF) <= tolerance);
                assert(fabs(cast(double)negativeBinomialCCDF(k, 2, T(0.5)) - expectedCCDF) <= tolerance);
                assert(negativeBinomialInvCDF(q, 2, T(0.5)) == k);
            } else {
                const tails = negativeBinomialTailsChecked(k, 2, T(0.5));
                assert(tails.status == NegativeBinomialTailStatus.found);
                assert(tails.cdf == T(expectedCDF) && tails.ccdf == T(expectedCCDF));
                const inverse = negativeBinomialInvCDFChecked(q, 2, T(0.5));
                assert(inverse.status == NegativeBinomialInverseStatus.found && inverse.count == k);
            }
        }
    }}
}
