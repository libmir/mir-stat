/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution).

The count $(D k) is the number of events in an observation interval, and
$(D lambda) is the expected number of events in that interval. Counts use
$(D size_t). Forward functions and count-returning inverse CDFs require
$(D lambda > 0): a zero rate is rejected even though it is a mathematical
limiting case. Tables use a finite positive rate unless stated otherwise.

The direct PMF and LPMF accept $(D Fp) rates. The Fp PMF is also available
as $(LREF fp_poissonPMF). CDF, CCDF, and inverse functions use built-in
floating-point values. Normal approximations need not match the direct or
gamma algorithms at finite counts.

$(D poissonInvCDF!(PoissonAlgo.gamma)) has a different purpose from the other
inverse algorithms: given a count and a cumulative probability, it solves
for the rate. The other choices solve for a count at a supplied rate.

Tables assume valid parameters except where an invalid input is listed.
Finite expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts does
not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.poisson;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

/++
Algorithms used to calculate poisson distribution.

`PoissonAlgo.direct` can be more time-consuming for large values of the number
of events (`k`) or the rate of occurrences (`lambda`). Additional algorithms are
provided to the user to choose the trade-off between running time and accuracy.

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
enum PoissonAlgo {
    /++
    Direct
    +/
    direct,
    /++
    Gamma Incomplete function
    +/
    gamma,
    /++
    Approximates poisson distribution with normal distribution. Generally a better approximation when
    `lambda > 1000`.
    +/
    approxNormal,
    /++
    Approximates poisson distribution with normal distribution (including continuity correction). More 
    accurate than `PoissonAlgo.approxNormal`. Generally a better approximation when `lambda > 10`.
    +/
    approxNormalContinuityCorrection
}

private
@safe pure nothrow @nogc
T poissonPMFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.direct)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: exp, pow;
    import mir.math.numeric: factorial;

    // Keep the direct calculation while the intermediate values retain precision.
    // Bound the factorial work as well as avoiding an integer wrap at k + 1.
    if (k < 2048)
    {
        const T decay = exp(-lambda);
        const T mass = decay * pow(lambda, k) / cast(T) factorial(k);
        if (decay >= T.min_normal && mass >= T.min_normal && mass < T.infinity)
            return mass;
    }
    return cast(T) exp(poissonLogMass(k, cast(real) lambda));
}

// Shared logarithmic calculation for the PMF fallback and large-count LPMF.
// Keep intermediate calculations in real until the final result conversion.
private @safe pure nothrow @nogc
real poissonLogMass(size_t k, real lambda)
{
    import mir.math.common: log, fabs;
    import mir.math.internal.log_binomial: logFactorial;
    import std.math.constants: PI;

    if (k == 0)
        return -lambda;
    if (lambda == real.infinity)
        return -real.infinity;
    if (k < 32)
        return k * log(lambda) - lambda - logFactorial!real(k);

    // Center Stirling's formula on lambda=k. Subtracting k*log(k) terms
    // separately loses the whole log mass for large counts near the mode.
    const real n = k;
    const real delta = n - lambda;
    real deviance;
    if (fabs(delta) < n * 0.5L)
    {
        // k*log(k/lambda)+lambda-k, using log((1+v)/(1-v))
        // = 2*(v+v^3/3+...). Here |v| < 1/3, so the series converges quickly.
        const real v = delta / (n + lambda);
        const real vv = v * v;
        deviance = delta * v;
        real term = 2 * n * v;
        for (uint j = 1; j < 64; ++j)
        {
            term *= vv;
            const real next = deviance + term / (2 * j + 1);
            if (next == deviance)
                break;
            deviance = next;
        }
    }
    else
    {
        const real ratio = n / lambda;
        const real logRatio = ratio > 0 && ratio < real.infinity
            ? log(ratio) : log(n) - log(lambda);
        deviance = n * logRatio + (lambda - n);
    }
    // Stirling correction through n^-13. At n>=32 the first omitted term
    // is below 1e-24, smaller than the precision of supported real types.
    const real inv = 1 / n;
    const real inv2 = inv * inv;
    const real correction = inv * (1.0L / 12 + inv2 * (-1.0L / 360
        + inv2 * (1.0L / 1260 + inv2 * (-1.0L / 1680
        + inv2 * (1.0L / 1188 + inv2 * (-691.0L / 360360 + inv2 / 156))))));
    return -deviance - 0.5L * (log(2 * PI) + log(n)) - correction;
}

private
@safe pure nothrow @nogc
T poissonPMFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.gamma)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: exp;
    import std.mathspecial: gammaIncompleteCompl;

    // Zero events have probability exp(-lambda); the gamma difference below
    // would call Phobos with a zero shape, outside its supported domain.
    if (k == 0)
        return exp(-lambda);

    const real upper = gammaIncompleteCompl(cast(real) k + 1, lambda);
    const real mass = upper - gammaIncompleteCompl(cast(real) k, lambda);
    // Adjacent CDF values can both round to one in the upper tail. Keep their
    // difference only when cancellation costs at most three significant bits.
    if (mass >= upper * 0.125L && mass >= T.min_normal)
        return cast(T) mass;
    return poissonPMFImpl!(T, PoissonAlgo.direct)(k, lambda);
}

private
@safe pure nothrow @nogc
T poissonPMFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.approxNormal)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalPDF;

    return normalPDF(k, lambda, sqrt(lambda));
}

private
@safe pure nothrow @nogc
T poissonPMFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    return normalCDF(cast(T) k + 0.5, lambda, sqrt(lambda)) - normalCDF(cast(T) k - 0.5, lambda, sqrt(lambda));
}

/++
Computes the poisson probability mass function (PMF).

The direct algorithm uses a logarithmic fallback when intermediate powers,
factorials, or exponentials overflow or underflow.
The gamma algorithm falls back to the direct algorithm when subtracting adjacent
CDF values would lose precision.

Returns $(D P(X == k)). The zero-count formula below applies to the direct
and gamma algorithms, not to normal approximations.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) (direct or gamma)) $(TD $(D exp(-lambda))))
    $(TR $(TD $(D lambda == T.infinity) (direct)) $(TD $(D 0)))
    $(TR $(TD $(D lambda <= 0) or NaN) $(TD Contract violation))
)

Params:
    poissonAlgo = algorithm for calculating PMF (default: PoissonAlgo.direct)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
template poissonPMF(PoissonAlgo poissonAlgo = PoissonAlgo.direct)
{
    /++
    Params:
        k = value to evaluate PMF (e.g. number of events)
        lambda = positive expected event count in the observation interval
    +/
    T poissonPMF(T)(const size_t k, const T lambda)
        if (isFloatingPoint!T)
        in (lambda > 0, "lambda must be greater than or equal to 0")
    {
        return poissonPMFImpl!(T, poissonAlgo)(k, lambda);
    }

    /++ Extended-range and extended-precision PMF; see $(LREF fp_poissonPMF). +/
    T poissonPMF(T)(const size_t k, const T lambda)
        if (is(T == Fp!size, size_t size) && poissonAlgo == PoissonAlgo.direct)
    {
        return fp_poissonPMF(k, lambda);
    }
}

/// ditto
@safe pure nothrow @nogc
template poissonPMF(string poissonAlgo)
{
    mixin("alias poissonPMF = .poissonPMF!(PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;
    
    assert(3.poissonPMF(6.0).approxEqual(exp(-6.0) * 216 / 6));
    // Can compute directly with differences of upper incomplete gamma function
    assert(3.poissonPMF!"gamma"(6.0).approxEqual(poissonPMF(3, 6.0)));
    // For large values of k or lambda, can approximate with normal distribution
    assert(1_000_000.poissonPMF!"approxNormal"(1_000_000.0).approxEqual(poissonPMF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
    // Or closer with continuity correction
    assert(1_000_000.poissonPMF!"approxNormalContinuityCorrection"(1_000_000.0).approxEqual(poissonPMF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
}

// test PoissonAlgo.direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;

    assert(0.poissonPMF(5.0).approxEqual(exp(-5.0)));
    assert(1.poissonPMF(5.0).approxEqual(exp(-5.0) * 5));
    assert(2.poissonPMF(5.0).approxEqual(exp(-5.0) * 25 / 2));
    assert(3.poissonPMF(5.0).approxEqual(exp(-5.0) * 125 / 6));
    assert(4.poissonPMF(5.0).approxEqual(exp(-5.0) * 625 / 24));
    assert(5.poissonPMF(5.0).approxEqual(exp(-5.0) * 3125 / 120));
    assert(6.poissonPMF(5.0).approxEqual(exp(-5.0) * 15625 / 720));
    assert(7.poissonPMF(5.0).approxEqual(exp(-5.0) * 78125 / 5040));
    assert(8.poissonPMF(5.0).approxEqual(exp(-5.0) * 390625 / 40320));
    assert(9.poissonPMF(5.0).approxEqual(exp(-5.0) * 1953125 / 362880));
    assert(10.poissonPMF(5.0).approxEqual(exp(-5.0) * 9765625 / 3628800));
}

// Direct PMFs must survive overflow and underflow of their intermediate values.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // Independent 100-digit references, rounded to the result type.
        assert(approxEqual(poissonPMF(40, T(40)),
            T(0.062947039423592203188230813788406L), 512 * T.epsilon, T(0)));
        assert(approxEqual(poissonPMF(200, T(200)),
            T(0.028197727685920821798702062339363L), 2048 * T.epsilon, T(0)));
        assert(approxEqual(poissonPMF(1000, T(1000)),
            T(0.012614611348721499718036936474579L), 8192 * T.epsilon, T(0)));
        assert(approxEqual(poissonPMF(3000, T(3000)),
            T(0.00728345388297452668151904477178226L), 32768 * T.epsilon, T(0)));
        assert(poissonPMF(size_t.max, T(5)) == 0);
        assert(poissonPMF(1, T.infinity) == 0);
    }}
    import mir.math.ieee: nextUp;
    // Zero events also use the logarithmic fallback after exponential underflow.
    assert(approxEqual(poissonPMF(0, 740.0), 85 * nextUp(0.0),
        0.0, nextUp(0.0)));
    assert(approxEqual(poissonPMF(1, 740.0), 62738 * nextUp(0.0),
        0.0, nextUp(0.0)));
}

// test PoissonAlgo.gamma
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    for (size_t i; i < 20; i++) {
        assert(i.poissonPMF!"gamma"(5.0).approxEqual(poissonPMF(i, 5.0)));
    }
}

// Differences of adjacent CDF values must not erase a representable mass.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(approxEqual(poissonPMF!"gamma"(30, T(1)),
            T(1.3869009421120462916399604485452e-33L), 128 * T.epsilon, T(0)));
        assert(poissonPMF!"gamma"(size_t.max, T(5)) == 0);
        foreach (k; 1 .. 35)
            assert(approxEqual(poissonPMF!"gamma"(k, T(5)), poissonPMF(k, T(5)),
                512 * T.epsilon, T(0)));
    }}
}

// The gamma algorithm must handle zero events without a zero-shape gamma call.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    {
        foreach (lambda; [T(0.125), T(1), T(5), T(80)])
        {
            // P(0) = exp(-lambda), and P(1) = lambda * P(0).
            const expected = exp(-lambda);
            assert(approxEqual(poissonPMF!"gamma"(0, lambda), expected,
                T(16) * T.epsilon, T(0)));
            assert(approxEqual(poissonPMF!"gamma"(1, lambda), lambda * expected,
                T(128) * T.epsilon, T(0)));
        }
    }
}

// test PoissonAlgo.approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.stat.distribution.normal: normalCDF, normalPDF;
    for (size_t i; i < 20; i++) {
        assert(i.poissonPMF!"approxNormal"(5.0).approxEqual(normalPDF(i, 5.0, sqrt(5.0))));
        assert(i.poissonPMF!"approxNormalContinuityCorrection"(5.0).approxEqual(normalCDF(i + 0.5, 5.0, sqrt(5.0)) - normalCDF(i - 0.5, 5.0, sqrt(5.0))));
    }
}

/++
Computes the Poisson PMF with extended range and precision. Prefer the
$(D poissonPMF(k, lambda)) overload for an $(D Fp) rate; this name is retained
for compatibility. Only the direct algorithm accepts $(D Fp).

Validation and evaluation use the represented rate without conversion through
hardware floating point. Tested widths are $(D Fp!64), $(D Fp!128), and
$(D Fp!256). The exponential uses an integer fixed-point series and repeated
squaring, with 192 extra working bits. Powers, factorial products, and integer
division retain that working precision. A separate wide exponent keeps
intermediate overflow or underflow from discarding a representable final mass.

The guard bits reduce accumulated rounding error; they do not guarantee correct
rounding for every input. Final subnormal results have fewer significant bits,
and masses below the representable range round to zero. Counts retain their
full $(D size_t) width. Factorial construction costs $(D O(k)) except for
proven underflow shortcuts, so very large counts remain impractical even when
their probabilities are representable. This implementation favors accuracy
over the speed of hardware-floating-point PMFs. Use $(LREF poissonLPMF)
to compute a log probability directly at Fp precision, including when the
PMF would underflow.

Finite positive rates below the hardware exponent range are accepted. A zero
rate remains a contract violation, consistently with the existing Poisson API.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD Extended-precision approximation to $(D exp(-lambda))))
    $(TR $(TD $(D lambda == +infinity)) $(TD $(D 0)))
    $(TR $(TD $(D lambda <= 0) or NaN) $(TD Contract violation))
)

Params:
    k = number of events at which to evaluate PMF
    lambda = positive expected event count in the observation interval

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
T fp_poissonPMF(T)(const size_t k, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (!lambda.sign && (lambda.exponent == long.max ? !lambda.coefficient : !!lambda.coefficient), "lambda must be positive and not NaN")
{
    import mir.stat.internal.fp_poisson: fpPoisson;
    return fpPoisson(k, lambda);
}

/// Evaluate the PMF at the rate's precision, or retain the compatibility name.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    const rate = Fp!128(5);
    const mass = poissonPMF(3, rate);
    static assert(is(typeof(mass) == const(Fp!128)));
    assert(mass == fp_poissonPMF(3, rate));
    assert(mass == poissonPMF!"direct"(3, rate));
    static assert(is(typeof(poissonPMF(3, 5.0)) == double));
    static assert(!__traits(compiles, poissonPMF!"gamma"(3, rate)));
    static assert(!__traits(compiles, poissonPMF!"approxNormal"(3, rate)));
    static assert(!__traits(compiles, poissonPMF!"approxNormalContinuityCorrection"(3, rate)));
}

// Extreme represented rates must not be narrowed or widened through hardware
// floating point or through an Fp conversion at its exponent floor.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (offset; [0L, 1L, 64L, 128L, 256L])
        {
            auto rate = T(1);
            rate.exponent = long.min + offset;
            assert(poissonPMF(0, rate) == T(1));
            assert(poissonPMF(1, rate) == rate);
            assert(!poissonPMF(2, rate).coefficient);
            assert(!poissonPMF(size_t.max, rate).coefficient);
        }
        auto rate = T(false, long.min, UInt!bits(1));
        assert(poissonPMF(1, rate) == rate);
        assert(!poissonPMF(2, rate).coefficient);
        assert(!poissonPMF(size_t.max, T(0x1p66)).coefficient);
        assert(!poissonPMF(1, T(double.infinity)).coefficient);
        auto huge = T(1);
        huge.exponent = long.max - 1;
        assert(!poissonPMF(0, huge).coefficient);
        static if (size_t.sizeof == 8)
            assert(!poissonPMF(size_t.max, T(0.25)).coefficient);
    }}
}

version(mir_stat_test)
@system unittest // Catching AssertError is @system; PMF retains its attributes.
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled)
    {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        foreach (rate; [0.0, -0.0, -1.0, -double.infinity, double.nan])
        {
            assertThrown!AssertError(poissonPMF(1, Fp!128(rate)));
            assertThrown!AssertError(fp_poissonPMF(1, Fp!128(rate)));
        }
    }
}

// BEGIN Poisson references (tools/generate_fp_poisson_reference.py)
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { size_t k; T rate, expected; }
        enum references = [
            Reference(0, T(false, -63, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be35"))),
            Reference(1, T(false, -63, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be35"))),
            Reference(100, T(false, -63, UInt!64.fromHexString("8000000000000000")), T(false, -590L, UInt!64.fromHexString("ddad2b7e93d473c4"))),
            Reference(1000, T(false, -54, UInt!64.fromHexString("fa00000000000000")), T(false, -70L, UInt!64.fromHexString("cead83cc72d71abb"))),
            Reference(3000, T(false, -52, UInt!64.fromHexString("bb80000000000000")), T(false, -71L, UInt!64.fromHexString("eeaa0a1d59b22a13"))),
            Reference(0, T(false, -63, UInt!64.fromHexString("8000000000000008")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be2a"))),
            Reference(0, T(false, -163, UInt!64.fromHexString("8000000000000000")), T(false, -63L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1, T(false, -163, UInt!64.fromHexString("8000000000000000")), T(false, -163L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163L, UInt!64.fromHexString("8000000000000000"))),
            Reference(2, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -2264L, UInt!64.fromHexString("8000000000000000"))),
            Reference(3, T(false, -1242, UInt!64.fromHexString("ee09a40000000000")), T(false, -3600L, UInt!64.fromHexString("8934259e9d39c2a9"))),
            Reference(0, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -64L, UInt!64.fromHexString("9b4597e37cb04ff4"))),
            Reference(0, T(false, -65, UInt!64.fromHexString("ffffffffffffffe0")), T(false, -64L, UInt!64.fromHexString("9b4597e37cb04ffe"))),
            Reference(0, T(false, -64, UInt!64.fromHexString("8000000000000010")), T(false, -64L, UInt!64.fromHexString("9b4597e37cb04fea"))),
            Reference(0, T(false, -33, UInt!64.fromHexString("8000000000000000")), T(false, -1549082068L, UInt!64.fromHexString("9fccfa2e1a711b26"))),
            Reference(0, T(false, -3, UInt!64.fromHexString("8000000000000000")), T(false, -1663314137230540375L, UInt!64.fromHexString("b9133b37813f1c4e"))),
            Reference(1, T(false, -3, UInt!64.fromHexString("8000000000000000")), T(false, -1663314137230540315L, UInt!64.fromHexString("b9133b37813f1c4e"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("8000000000000000")), T(false, -6653256548922161309L, UInt!64.fromHexString("8bdce1c394ea050d"))),
            Reference(10, T(false, -1, UInt!64.fromHexString("8000000000000000")), T(false, -6653256548922160711L, UInt!64.fromHexString("a1a8a24453317d32"))),
            Reference(0, T(false, 3, UInt!64.fromHexString("8000000000000000")), T(false, long.min, UInt!64.fromHexString("0000000000000000"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf7950")), T(false, -9223372036854775805L, UInt!64.fromHexString("9435d139b89382d3"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf7953")), T(false, -9223372036854775807L, UInt!64.fromHexString("8447dae5e2df8b34"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf797f")), T(false, long.min, UInt!64.fromHexString("000000013cf65cf9"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf79aa")), T(false, long.min, UInt!64.fromHexString("0000000000000002"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf79ac")), T(false, long.min, UInt!64.fromHexString("0000000000000001"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf79ad")), T(false, long.min, UInt!64.fromHexString("0000000000000001"))),
            Reference(0, T(false, -1, UInt!64.fromHexString("b17217f7d1cf79af")), T(false, long.min, UInt!64.fromHexString("0000000000000000"))),
            Reference(12, T(false, -1, UInt!64.fromHexString("b17217f7d1cf7ac1")), T(false, -9223372036854775351L, UInt!64.fromHexString("e74c82bc9bd5e553"))),
        ];
        foreach (c; references)
        {
            const actual = poissonPMF(c.k, c.rate);
            assert(actual.exponent == c.expected.exponent || !actual.coefficient && !c.expected.coefficient);
            assert(actual.coefficient == c.expected.coefficient);
            assert(!actual.sign);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { size_t k; T rate, expected; }
        enum references = [
            Reference(0, T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129L, UInt!128.fromHexString("bc5ab1b16779be3575bd8f0520a9f21c"))),
            Reference(1, T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129L, UInt!128.fromHexString("bc5ab1b16779be3575bd8f0520a9f21c"))),
            Reference(100, T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -654L, UInt!128.fromHexString("ddad2b7e93d473c451e40df7ae2509c6"))),
            Reference(1000, T(false, -118, UInt!128.fromHexString("fa000000000000000000000000000000")), T(false, -134L, UInt!128.fromHexString("cead83cc72d71abad1fb7ec0740a7a0c"))),
            Reference(3000, T(false, -116, UInt!128.fromHexString("bb800000000000000000000000000000")), T(false, -135L, UInt!128.fromHexString("eeaa0a1d59b22a128ff1e16851de9436"))),
            Reference(0, T(false, -127, UInt!128.fromHexString("80000000000000080000000000000000")), T(false, -129L, UInt!128.fromHexString("bc5ab1b16779be29b01273eea90e0f23"))),
            Reference(0, T(false, -227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("fffffffffffffffffffffffff0000000"))),
            Reference(1, T(false, -227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -228L, UInt!128.fromHexString("fffffffffffffffffffffffff0000000"))),
            Reference(1, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(2, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -2328L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(3, T(false, -1306, UInt!128.fromHexString("ee09a400000000000000000000000000")), T(false, -3664L, UInt!128.fromHexString("8934259e9d39c2a98000000000000000"))),
            Reference(0, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("9b4597e37cb04ff3d675a35530cdd768"))),
            Reference(0, T(false, -129, UInt!128.fromHexString("ffffffffffffffe00000000000000000")), T(false, -128L, UInt!128.fromHexString("9b4597e37cb04ffd8acf218cfbd2d6f3"))),
            Reference(0, T(false, -128, UInt!128.fromHexString("80000000000000100000000000000000")), T(false, -128L, UInt!128.fromHexString("9b4597e37cb04fea221c251d65c8d878"))),
            Reference(0, T(false, -97, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1549082132L, UInt!128.fromHexString("9fccfa2e1a711b266363f1eb17b09354"))),
            Reference(0, T(false, -67, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1663314137230540439L, UInt!128.fromHexString("b9133b37813f1c4e2581835c7c261ea0"))),
            Reference(1, T(false, -67, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1663314137230540379L, UInt!128.fromHexString("b9133b37813f1c4e2581835c7c261ea0"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -6653256548922161373L, UInt!128.fromHexString("8bdce1c394ea050c85bf445c7c28d4ec"))),
            Reference(10, T(false, -65, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -6653256548922160775L, UInt!128.fromHexString("a1a8a24453317d31ac3c0d0f3e4ae61b"))),
            Reference(0, T(false, -61, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, long.min, UInt!128.fromHexString("00000000000000000000000000000000"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf78f792035be6ed3b6436")), T(false, -9223372036854775805L, UInt!128.fromHexString("80000000000000000d6c37309bf58b0c"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5")), T(false, -9223372036854775807L, UInt!128.fromHexString("8000000000000000174feac89fe881bb"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf795310d7b7af1c3620ca")), T(false, long.min, UInt!128.fromHexString("00000000000000010000000000000000"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf79aa66ff83a860540358")), T(false, long.min, UInt!128.fromHexString("00000000000000000000000000000002"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")), T(false, long.min, UInt!128.fromHexString("00000000000000000000000000000001"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf79ad2cc7e387a791ea07")), T(false, long.min, UInt!128.fromHexString("00000000000000000000000000000000"))),
            Reference(0, T(false, -65, UInt!128.fromHexString("b17217f7d1cf79ae8fac13774b30dd5e")), T(false, long.min, UInt!128.fromHexString("00000000000000000000000000000000"))),
            Reference(12, T(false, -65, UInt!128.fromHexString("b17217f7d1cf7ac10c2926cfd821131b")), T(false, -9223372036854775415L, UInt!128.fromHexString("e1deb287e14c3fa099a337f10a5c54ec"))),
        ];
        foreach (c; references)
        {
            const actual = poissonPMF(c.k, c.rate);
            assert(actual.exponent == c.expected.exponent || !actual.coefficient && !c.expected.coefficient);
            assert(actual.coefficient == c.expected.coefficient);
            assert(!actual.sign);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { size_t k; T rate, expected; }
        enum references = [
            Reference(0, T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("bc5ab1b16779be3575bd8f0520a9f21bb5300b556ad8ee66604973a14a0fb5db"))),
            Reference(1, T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("bc5ab1b16779be3575bd8f0520a9f21bb5300b556ad8ee66604973a14a0fb5db"))),
            Reference(100, T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -782L, UInt!256.fromHexString("ddad2b7e93d473c451e40df7ae2509c67d8f62b878fd9f3b21933f0b04d43603"))),
            Reference(1000, T(false, -246, UInt!256.fromHexString("fa00000000000000000000000000000000000000000000000000000000000000")), T(false, -262L, UInt!256.fromHexString("cead83cc72d71abad1fb7ec0740a7a0c02ebf7ad6e9e5adb0c84cd0a5ac33e23"))),
            Reference(3000, T(false, -244, UInt!256.fromHexString("bb80000000000000000000000000000000000000000000000000000000000000")), T(false, -263L, UInt!256.fromHexString("eeaa0a1d59b22a128ff1e16851de94362cd015e4175e3463ff668440090fd8da"))),
            Reference(0, T(false, -255, UInt!256.fromHexString("8000000000000008000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("bc5ab1b16779be29b01273eea90e0f2286aff3b71d18e56fa4e11271acd40144"))),
            Reference(0, T(false, -355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("fffffffffffffffffffffffff000000000000000000000000080000000000000"))),
            Reference(1, T(false, -355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -356L, UInt!256.fromHexString("fffffffffffffffffffffffff000000000000000000000000080000000000000"))),
            Reference(1, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(2, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -2456L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(3, T(false, -1434, UInt!256.fromHexString("ee09a40000000000000000000000000000000000000000000000000000000000")), T(false, -3792L, UInt!256.fromHexString("8934259e9d39c2a9800000000000000000000000000000000000000000000000"))),
            Reference(0, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("9b4597e37cb04ff3d675a35530cdd767e347bf8ad0e80abbce4ae95861014318"))),
            Reference(0, T(false, -257, UInt!256.fromHexString("ffffffffffffffe0000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("9b4597e37cb04ffd8acf218cfbd2d6f2ed6de69c35ed7cc34c82eb3fac98337b"))),
            Reference(0, T(false, -256, UInt!256.fromHexString("8000000000000010000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("9b4597e37cb04fea221c251d65c8d8781eb97bf61c328c8ac5b63ca1e341c78a"))),
            Reference(0, T(false, -225, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1549082260L, UInt!256.fromHexString("9fccfa2e1a711b266363f1eb17b09354103b7d520c8aeca78d3495bddcbc068c"))),
            Reference(0, T(false, -195, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1663314137230540567L, UInt!256.fromHexString("b9133b37813f1c4e2581835c7c261ea04294a8ee48c49a3696d47b083bdc8b41"))),
            Reference(1, T(false, -195, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1663314137230540507L, UInt!256.fromHexString("b9133b37813f1c4e2581835c7c261ea04294a8ee48c49a3696d47b083bdc8b41"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -6653256548922161501L, UInt!256.fromHexString("8bdce1c394ea050c85bf445c7c28d4ec5e44467dfd326c29a9677547eefabad7"))),
            Reference(10, T(false, -193, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -6653256548922160903L, UInt!256.fromHexString("a1a8a24453317d31ac3c0d0f3e4ae61a8a529037c1aafe1338de4e4ec5e096a6"))),
            Reference(0, T(false, -189, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf78461feb64151dc1b86c51fd44be7cdf7cee9fb9bddc8fdbc662")), T(false, -9223372036854775806L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffd2aab0cd74a5b60d"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18")), T(false, long.min, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffe6c4df848bfbaa64"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf78fa57cbbbc634794ae55d3fab227fa206ec96c9f0e8f2f4cca2")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000100000000000000000000000000000000"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf79aa66ff83a860540357ad2bdbf66ab2c8cf0826910ea6798dd1")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000000000000000000000000000000000002"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000000000000000000000000000000000001"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf79ad2cc7e387a791ea06d4baaa567a7ea38c0bf39da870dc6687")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(0, T(false, -193, UInt!256.fromHexString("b17217f7d1cf79ae8fac13774b30dd5e68821186826490ea8dda23f5560dd2e2")), T(false, long.min, UInt!256.fromHexString("0000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(12, T(false, -193, UInt!256.fromHexString("b17217f7d1cf7ac10c2926cfd821131ab4bbe0ac9e3a28030626036e9a47a154")), T(false, -9223372036854775543L, UInt!256.fromHexString("e1deb287e14c3fa0bad7647d6c740fcfbffa87b55e9dc3df3eac8a65dfe6eacc"))),
        ];
        foreach (c; references)
        {
            const actual = poissonPMF(c.k, c.rate);
            assert(actual.exponent == c.expected.exponent || !actual.coefficient && !c.expected.coefficient);
            assert(actual.coefficient == c.expected.coefficient);
            assert(!actual.sign);
        }
    }
}
// END Poisson references

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp;
    import mir.conv: to;
    import mir.math.common: approxEqual, exp;

    for (size_t i; i <= 10; i++) {
        assert(i.fp_poissonPMF(Fp!128(5.0)).to!double.approxEqual(poissonPMF(i, 5.0)));
    }
}

// Extended storage must preserve the exponential before applying the power.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    import mir.bignum.fp: fp_log;
    const central = fp_poissonPMF(1000, Fp!128(1000));
    assert(approxEqual(cast(double) central, 0.012614611348721499718036936474579,
        1e-12, 0.0));
    const tiny = fp_poissonPMF(0, Fp!128(1000));
    assert(tiny.coefficient != 0);
    assert(approxEqual(fp_log!double(tiny), -1000.0, 1e-14, 0.0));
    // A finite extended rate can exceed the hardware exponent range.
    const hugeRate = Fp!128(double.max) * Fp!128(2);
    assert(fp_poissonPMF(0, hugeRate).coefficient == 0);
    foreach (rate; [511.0, 512.0, 513.0, 1024.0, 2000.0])
    {
        const zero = fp_poissonPMF(0, Fp!128(rate));
        const one = fp_poissonPMF(1, Fp!128(rate));
        assert(approxEqual(cast(double)(one / zero), rate, 1e-14, 0.0));
    }
}

private
@safe pure nothrow @nogc
T poissonCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.direct)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: exp;
    // The unscaled sum grows like exp(lambda). Use the incomplete gamma
    // function if this sum or its final rescaling loses range or precision.
    if (k < 2048)
    {
        T output = 1;
        T multiplier = 1;
        for (size_t i = 1; i <= k; i++) {
            multiplier *= (lambda / i);
            output += multiplier;
        }
        const T decay = exp(-lambda);
        const T result = output * decay;
        if (decay >= T.min_normal && result >= T.min_normal && result <= 1)
            return result;
    }
    return poissonCDFImpl!(T, PoissonAlgo.gamma)(k, lambda);
}

private
@safe pure nothrow @nogc
T poissonCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.gamma)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import std.mathspecial: gammaIncompleteCompl;
    // Convert before adding one: k may be size_t.max.
    return cast(T) gammaIncompleteCompl(cast(real) k + 1, lambda);
}

private
@safe pure nothrow @nogc
T poissonCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && 
        (poissonAlgo == PoissonAlgo.approxNormal || 
         poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection))
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    T l = k;
    static if (poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCDF(l, lambda, sqrt(lambda));
}

/++
Computes the poisson cumulative distribution function (CDF).

The direct algorithm falls back to the incomplete gamma function when its
unscaled sum cannot be evaluated reliably, or the count would require a long sum.

Returns the inclusive lower-tail probability $(D P(X <= k)). The table
specifies direct and gamma evaluation; normal approximations can differ.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D exp(-lambda))))
    $(TR $(TD $(D lambda <= 0) or NaN) $(TD Contract violation))
)

Params:
    poissonAlgo = algorithm for calculating CDF (default: PoissonAlgo.direct)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
template poissonCDF(PoissonAlgo poissonAlgo = PoissonAlgo.direct)
{
    /++
    Params:
        k = value to evaluate CDF (e.g. number of events)
        lambda = positive expected event count in the observation interval
    +/
    T poissonCDF(T)(const size_t k, const T lambda)
        if (isFloatingPoint!T)
        in (lambda > 0, "lambda must be greater than or equal to 0")
    {
        return poissonCDFImpl!(T, poissonAlgo)(k, lambda);
    }
}

/// ditto
@safe pure nothrow @nogc
template poissonCDF(string poissonAlgo)
{
    mixin("alias poissonCDF = .poissonCDF!(PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    
    assert(3.poissonCDF(6.0).approxEqual(poissonPMF(0, 6.0) + poissonPMF(1, 6.0) + poissonPMF(2, 6.0) + poissonPMF(3, 6.0)));
    // Can compute directly with upper incomplete gamma function
    assert(3.poissonCDF!"gamma"(6.0).approxEqual(poissonCDF(3, 6.0)));
    // For large values of k or lambda, can approximate with normal distribution
    assert(1_000_000.poissonCDF!"approxNormal"(1_000_000.0).approxEqual(poissonCDF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
    // Or closer with continuity correction
    assert(1_000_000.poissonCDF!"approxNormalContinuityCorrection"(1_000_000.0).approxEqual(poissonCDF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
}

// test PoissonAlgo.direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    
    static double sumOfPoissonPMFs(size_t k, double lambda) {
        double output = 0;
        for (size_t i; i < (k + 1); i++) {
            output += poissonPMF(i, lambda);
        }
        return output;
    }
    
    for (size_t i; i < 20; i++) {
        assert(i.poissonCDF(5.0).approxEqual(sumOfPoissonPMFs(i, 5.0)));
    }
}

// test PoissonAlgo.gamma
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    for (size_t i; i < 20; i++) {
        assert(i.poissonCDF!"gamma"(5.0).approxEqual(poissonCDF(i, 5.0)));
    }
}

// test PoissonAlgo.approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.stat.distribution.normal: normalCDF;
    for (size_t i; i < 20; i++) {
        assert(i.poissonCDF!"approxNormal"(5.0).approxEqual(normalCDF(i, 5.0, sqrt(5.0))));
        assert(i.poissonCDF!"approxNormalContinuityCorrection"(5.0).approxEqual(normalCDF(i + 0.5, 5.0, sqrt(5.0))));
    }
}

private
@safe pure nothrow @nogc
T poissonCCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.direct)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    const T cdf = poissonCDFImpl!(T, poissonAlgo)(k, lambda);
    // Keep subtraction when the remaining tail is at least 1/8.
    // Subtraction preserves the CDF's absolute error, but that error is larger
    // relative to a small tail probability. For smaller tails, evaluate the
    // upper tail directly instead of subtracting a value near one.
    return cdf <= T(0.875) ? T(1) - cdf
        : poissonCCDFImpl!(T, PoissonAlgo.gamma)(k, lambda);
}

private
@safe pure nothrow @nogc
T poissonCCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.gamma)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import std.mathspecial: gammaIncomplete;
    return cast(T) gammaIncomplete(cast(real) k + 1, lambda);
}

private
@safe pure nothrow @nogc
T poissonCCDFImpl(T, PoissonAlgo poissonAlgo)(const size_t k, const T lambda)
    if (isFloatingPoint!T && 
        (poissonAlgo == PoissonAlgo.approxNormal || 
         poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection))
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCCDF;

    T l = k;
    static if (poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCCDF(l, lambda, sqrt(lambda));
}

/++
Computes the poisson complementary cumulative distribution function (CCDF).

Returns the probability of strictly more than `k` events. The direct algorithm
uses the incomplete gamma function for small upper tails to avoid cancellation.

The table specifies direct and gamma evaluation; normal approximations can
differ at finite counts.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D 1 - exp(-lambda))))
    $(TR $(TD $(D lambda <= 0) or NaN) $(TD Contract violation))
)

Params:
    poissonAlgo = algorithm for calculating CCDF (default: PoissonAlgo.direct)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
template poissonCCDF(PoissonAlgo poissonAlgo = PoissonAlgo.direct)
{
    /++
    Params:
        k = value to evaluate CCDF (e.g. number of events)
        lambda = positive expected event count in the observation interval
    +/
    T poissonCCDF(T)(const size_t k, const T lambda)
        if (isFloatingPoint!T)
        in (lambda > 0, "lambda must be greater than or equal to 0")
    {
        return poissonCCDFImpl!(T, poissonAlgo)(k, lambda);
    }
}

/// ditto
@safe pure nothrow @nogc
template poissonCCDF(string poissonAlgo)
{
    mixin("alias poissonCCDF = .poissonCCDF!(PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(3.poissonCCDF(6.0).approxEqual(1.0 - (poissonPMF(0, 6.0) + poissonPMF(1, 6.0) + poissonPMF(2, 6.0) + poissonPMF(3, 6.0))));
    // Can compute directly with upper incomplete gamma function
    assert(3.poissonCCDF!"gamma"(6.0).approxEqual(poissonCCDF(3, 6.0)));
    // For large values of k or lambda, can approximate with normal distribution
    assert(1_000_000.poissonCCDF!"approxNormal"(1_000_000.0).approxEqual(poissonCCDF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
    // Or closer with continuity correction
    assert(1_000_000.poissonCCDF!"approxNormalContinuityCorrection"(1_000_000.0).approxEqual(poissonCCDF!"gamma"(1_000_000, 1_000_000.0), 10e-3));
}

// test PoissonAlgo.direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    for (size_t i; i < 20; i++) {
        assert(i.poissonCCDF(5.0).approxEqual(1.0 - poissonCDF(i, 5.0)));
    }
}

// test PoissonAlgo.gamma
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    for (size_t i; i < 20; i++) {
        assert(i.poissonCCDF!"gamma"(5.0).approxEqual(poissonCCDF(i, 5.0)));
    }
}

// test PoissonAlgo.approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, sqrt;
    import mir.stat.distribution.normal: normalCCDF;
    for (size_t i; i < 20; i++) {
        assert(i.poissonCCDF!"approxNormal"(5.0).approxEqual(normalCCDF(i, 5.0, sqrt(5.0))));
        assert(i.poissonCCDF!"approxNormalContinuityCorrection"(5.0).approxEqual(normalCCDF(i + 0.5, 5.0, sqrt(5.0))));
    }
}

// Tail calculations must remain bounded and retain small probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(approxEqual(poissonCDF(1000, T(1000)),
            T(0.50840936716850599121425909287246L), 2048 * T.epsilon, T(0)));
        assert(approxEqual(poissonCCDF(20, T(1)),
            T(7.542625077205278476019579100834e-21L), 128 * T.epsilon, T(0)));
        assert(poissonCDF(size_t.max, T(5)) == 1);
        assert(poissonCCDF(size_t.max, T(5)) == 0);
        assert(poissonCDF!"gamma"(size_t.max, T(5)) == 1);
        assert(poissonCCDF!"gamma"(size_t.max, T(5)) == 0);
        assert(poissonCDF(3, T.infinity) == 0);
        assert(poissonCCDF(3, T.infinity) == 1);
        foreach (k; 0 .. 20)
        {
            assert(approxEqual(poissonCDF(k, T(5)) + poissonCCDF(k, T(5)),
                T(1), 32 * T.epsilon));
        }
    }}
}

private
@safe pure nothrow @nogc
size_t poissonInvCDFImpl(T, PoissonAlgo poissonAlgo)(const T p, const T lambda)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.direct)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    if (p == 0) {
        return 0;
    }

    if (p == 1 || lambda == T.infinity)
        return size_t.max;

    // Compare the smaller tail. Repeatedly adding/subtracting masses can stop
    // changing the cumulative value, or accumulate enough error to miss a bin.
    bool reached(size_t k)
    {
        return p <= T(0.5) ? poissonCDF(k, lambda) >= p
            : poissonCCDF(k, lambda) <= 1 - p;
    }

    size_t low = 0;
    size_t high = lambda > 16
        ? poissonInvCDFImpl!(T, PoissonAlgo.approxNormalContinuityCorrection)(p, lambda)
        : 0;
    size_t step = 1;
    while (!reached(high))
    {
        if (high == size_t.max)
            return high; // The quantile does not fit in the result type.
        low = high + 1;
        high += step < size_t.max - high ? step : size_t.max - high;
        step = step <= size_t.max / 2 ? step * 2 : size_t.max;
    }
    // A bounded binary search also handles an initial guess above the answer,
    // without decrementing an unsigned counter past zero.
    while (low < high)
    {
        const mid = low + (high - low) / 2;
        if (reached(mid))
            high = mid;
        else
            low = mid + 1;
    }
    return low;
}

private
@safe pure nothrow @nogc
T poissonInvCDFImpl(T, PoissonAlgo poissonAlgo)(const T p, const size_t k)
    if (isFloatingPoint!T && poissonAlgo == PoissonAlgo.gamma)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import std.mathspecial: gammaIncompleteComplInverse;

    if (p == 0) {
        return T.infinity;
    }
    if (p == 1)
        return 0;
    return gammaIncompleteComplInverse(cast(real) k + 1, p);
}

private
@safe pure nothrow @nogc
size_t poissonInvCDFImpl(T, PoissonAlgo poissonAlgo)(const T p, const T lambda)
    if (isFloatingPoint!T && 
        (poissonAlgo == PoissonAlgo.approxNormal || 
         poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection))
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalCDF, normalInvCDF;

    if (p == 0)
        return 0;
    if (p == 1 || lambda == T.infinity)
        return size_t.max;

    T std = sqrt(lambda);

    // Handles case where p is small better than just using pLowerBound = 0
    T pLowerBound = 0;
    T lowerValue = 0;
    static if (poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection)
        lowerValue += 0.5;
    pLowerBound = normalCDF(lowerValue, lambda, std);
    if (p <= pLowerBound) { 
        return 0;
    }

    auto result = normalInvCDF(p, lambda, std);
    static if (poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection) {
        result = result - 0.5;
    }
    if (result <= 0)
        return 0;
    if (result >= cast(real) size_t.max)
        return size_t.max;
    return cast(size_t) floor(result);
}

/++
Computes the poisson inverse cumulative distribution function (InvCDF).

For algorithms `PoissonAlgo.direct`, `PoissonAlgo.approxNormal`, and 
`PoissonAlgo.approxNormalContinuityCorrection`, the inverse CDF returns the 
number of events (`k`) given the probability (`p`) and rate of occurrence
(`lambda`). For the `PoissonAlgo.gamma` algorithm, the inverse CDF returns the rate
of occurrence (`lambda`) given the probability (`p`) and the number of events (`k`).

For `PoissonAlgo.direct`, if the value of `lambda` is larger than 16, then an
initial guess is made based on `PoissonAlgo.approxNormalContinuityCorrection`.

Count-returning algorithms return `0` at `p = 0` and `size_t.max` at `p = 1`,
where no finite Poisson quantile exists. `size_t.max` also denotes a quantile
outside the result type's range. The gamma algorithm instead solves for the
rate and returns positive infinity at `p = 0` and zero at `p = 1`.

Count results use $(D size_t.max) as a sentinel; this cannot distinguish an
exact maximum count from an unbounded or out-of-range quantile. The gamma
algorithm instead returns a floating-point rate.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0) (count algorithms)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) (count algorithms)) $(TD $(D size_t.max)))
    $(TR $(TD $(D lambda == T.infinity) and $(D p > 0) (count algorithms)) $(TD $(D size_t.max)))
    $(TR $(TD $(D p == 0) (gamma rate inversion)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p == 1) (gamma rate inversion)) $(TD $(D 0)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D lambda <= 0) or NaN (count algorithms)) $(TD Contract violation))
)

Params:
    poissonAlgo = algorithm for calculating InvCDF (default: PoissonAlgo.direct)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
template poissonInvCDF(PoissonAlgo poissonAlgo = PoissonAlgo.direct)
    if (poissonAlgo == PoissonAlgo.direct ||
        poissonAlgo == PoissonAlgo.approxNormal ||
        poissonAlgo == PoissonAlgo.approxNormalContinuityCorrection)
{
    /++
    Params:
        p = cumulative probability in $(D [0, 1])
        lambda = positive expected event count in the observation interval
    +/
    size_t poissonInvCDF(T)(const T p, const T lambda)
        if (isFloatingPoint!T)
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
        in (lambda > 0, "lambda must be greater than or equal to 0")
    {
        return poissonInvCDFImpl!(T, poissonAlgo)(p, lambda);
    }
}

/// ditto
@safe pure nothrow @nogc
template poissonInvCDF(PoissonAlgo poissonAlgo)
    if (poissonAlgo == PoissonAlgo.gamma)
{
    /++
    Params:
        p = cumulative probability in $(D [0, 1])
        k = number of events
    +/
    T poissonInvCDF(T)(const T p, const size_t k)
        if (isFloatingPoint!T)
        in (p >= 0, "p must be greater than or equal to 0")
        in (p <= 1, "p must be less than or equal to 1")
    {
        return poissonInvCDFImpl!(T, poissonAlgo)(p, k);
    }
}

/// ditto
@safe pure nothrow @nogc
template poissonInvCDF(string poissonAlgo)
{
    mixin("alias poissonInvCDF = .poissonInvCDF!(PoissonAlgo." ~ poissonAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    
    assert(0.15.poissonInvCDF(6.0) == 3);
    // Passing `gamma` returns the rate of occurnece
    assert(0.151204.poissonInvCDF!"gamma"(3).approxEqual(6));
    // For large values of k or lambda, can approximate with normal distribution
    assert(0.5.poissonInvCDF!"approxNormal"(1_000_000.0) == 1_000_000);
    // Or closer with continuity correction
    assert(0.5009974.poissonInvCDF!"approxNormalContinuityCorrection"(1_000_000.0) == 1_000_002);
}

// test PoissonAlgo.direct
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.poissonInvCDF(5.0) == 0);
    for (double x = 0.05; x < 1; x = x + 0.05) {
        size_t value = x.poissonInvCDF(5.0);
        assert(value.poissonCDF(5.0) >= x);
        assert((value - 1).poissonCDF(5.0) < x);
    }
}

// Quantile endpoints and bracketing must not depend on cumulative roundoff.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.ieee: nextDown, nextUp;
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(poissonInvCDF(T(0.5), T(1000)) == 1000);
        // Independently bracketed using 100-digit Poisson sums.
        assert(poissonInvCDF(T(0.1), T(1000)) == 960);
        assert(poissonInvCDF(T(0.9), T(1000)) == 1041);
        assert(poissonInvCDF(T(0), T(5)) == 0);
        assert(poissonInvCDF(T(1), T(5)) == size_t.max);
        assert(poissonInvCDF!"approxNormal"(T(1), T(5)) == size_t.max);
        assert(poissonInvCDF!"approxNormalContinuityCorrection"(T(1), T(5)) == size_t.max);
        assert(poissonInvCDF!"gamma"(T(1), size_t(5)) == 0);
        assert(poissonInvCDF!"gamma"(T(0), size_t(5)) == T.infinity);
        assert(poissonInvCDF(T(0.5), T.infinity) == size_t.max);
        // Finite rates can also put the median beyond the count return type.
        const hugeRate = cast(T) size_t.max * 4;
        assert(poissonInvCDF(T(0.5), hugeRate) == size_t.max);
        assert(poissonInvCDF!"approxNormal"(T(0.5), hugeRate) == size_t.max);
        assert(poissonInvCDF!"approxNormalContinuityCorrection"(T(0.5), hugeRate)
            == size_t.max);
        // Just above the lower cutoff, inverse-normal rounding can put the
        // continuity-corrected result at or slightly below zero.
        foreach (rate; [T(5), T(10)])
        {
            const p = nextUp(normalCDF(T(0.5), rate, sqrt(rate)));
            assert(poissonInvCDF!"approxNormalContinuityCorrection"(p, rate) == 0);
        }
        foreach (lambda; [T(1), T(5), T(20), T(1000)])
            foreach (p; [T(1e-9), T(0.1), T(0.5), T(0.9), nextDown(T(1))])
            {
                const k = poissonInvCDF(p, lambda);
                if (p <= T(0.5))
                {
                    assert(poissonCDF(k, lambda) >= p);
                    assert(k == 0 || poissonCDF(k - 1, lambda) < p);
                }
                else
                {
                    assert(poissonCCDF(k, lambda) <= 1 - p);
                    assert(k == 0 || poissonCCDF(k - 1, lambda) > 1 - p);
                }
            }
    }}
    assert(poissonInvCDF(nextDown(1.0), 5.0) == 32);
}

// test PoissonAlgo.direct, large lambda branch, check small p
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    double x = 1.0e-9;
    assert(x.poissonInvCDF(20.0) == 0);
}

// test PoissonAlgo.direct, large lambda branch
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.poissonInvCDF(25.0) == 0);
    for (double x = 0.01; x < 1; x = x + 0.01) {
        size_t value = x.poissonInvCDF(25.0);
        assert(value.poissonCDF(25.0) >= x);
        assert((value - 1).poissonCDF(25.0) < x);
    }
}

// test PoissonAlgo.gamma (note the difference in how it is tested)
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.0.poissonInvCDF!"gamma"(5) == double.infinity);
    for (double x = 0.05; x < 1; x = x + 0.05) {
        for (size_t i; i < 10; i++) {
            assert(poissonCDF!"gamma"(i, poissonInvCDF!"gamma"(x, i)).approxEqual(x));
        }
    }
}

// test PoissonAlgo.approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalInvCDF;

    assert(0.poissonInvCDF!"approxNormal"(5.0) == 0);
    assert(0.poissonInvCDF!"approxNormalContinuityCorrection"(5.0) == 0);
    double checkValue;
    for (double x = 0.05; x < 1; x = x + 0.05) {
        checkValue = normalInvCDF(x, 5.0, sqrt(5.0));
        assert(x.poissonInvCDF!"approxNormal"(5.0) == floor(checkValue));
        assert(x.poissonInvCDF!"approxNormalContinuityCorrection"(5.0) == floor(checkValue - 0.5));
    }
}

/// Count inversion and gamma rate inversion answer different questions.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual, log;
    // At a rate of one, the median event count is one.
    assert(poissonInvCDF(0.5, 1.0) == 1);
    // With zero events, CDF = exp(-lambda); solve exp(-lambda) = 1/2.
    const rate = poissonInvCDF!(PoissonAlgo.gamma)(0.5, size_t(0));
    assert(approxEqual(rate, log(2.0)));
    assert(poissonInvCDF(1.0, 1.0) == size_t.max);
    assert(poissonInvCDF!(PoissonAlgo.gamma)(1.0, size_t(0)) == 0);
}

/++
Computes the poisson log probability mass function (LPMF).

For large counts, the calculation centers the logarithm around the mean rather
than subtracting large log-factorial and power terms. This preserves accuracy
when the count and rate are close.

The $(D Fp) overload returns the same Fp type and retains the full $(D size_t)
count. Guarded logarithms, Stirling corrections, and a centered deviance series
avoid conversion through hardware floating point. Finite rates throughout the
Fp exponent range are accepted; $(D k == 0) returns the exact negated rate.
This is guarded evaluation, not a universal correctly-rounded guarantee.

Returns the natural logarithm of $(D P(X == k)), which can remain finite
when the PMF underflows to zero.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D -lambda)))
    $(TR $(TD $(D lambda == +infinity)) $(TD Negative infinity))
    $(TR $(TD $(D lambda <= 0) or NaN) $(TD Contract violation))
)

Params:
    k = number of events at which to evaluate LPMF
    lambda = positive expected event count in the observation interval

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution)
+/
@safe pure nothrow @nogc
T poissonLPMF(T)(const size_t k, const T lambda)
    if (isFloatingPoint!T)
    in (lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: log;
    import mir.math.internal.log_binomial: logFactorial;

    if (k == 0 || lambda == T.infinity)
        return -lambda;
    // Small counts do not need the centered expansion. Preserve their original
    // result-type calculation rather than paying for extended-precision logs.
    if (k < 32)
        return k * log(lambda) - (logFactorial!T(k) + lambda);
    return cast(T) poissonLogMass(k, cast(real) lambda);
}

/// ditto
@safe pure nothrow @nogc
T poissonLPMF(T)(const size_t k, const T lambda)
    if (is(T == Fp!bits, uint bits))
    in (!lambda.sign && (lambda.exponent == long.max ? !lambda.coefficient : !!lambda.coefficient),
        "lambda must be positive and not NaN")
{
    if (!k || lambda.exponent == long.max)
    {
        T result = lambda;
        result.sign = true;
        return result;
    }
    import mir.stat.internal.fp_poisson_log: PoissonLog;
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    return PoissonLog!bits.mass(k,lambda);
}

/// Compute the log mass directly when the PMF underflows.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    auto rate = Fp!128(1);
    rate.exponent += 66;
    assert(!poissonPMF(0,rate).coefficient);
    auto expected = rate;
    expected.sign = true;
    assert(poissonLPMF(0,rate) == expected);
}

// Large central counts must not lose the log mass to cancellation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(approxEqual(poissonLPMF(32, T(32)),
            T(-2.6544105665237876717458292327029558L), 32 * T.epsilon, T(0)));
        assert(poissonLPMF(0, T.infinity) == -T.infinity);
        assert(poissonLPMF(1, T.infinity) == -T.infinity);
        foreach (lambda; [T(16), T(16.001), T(31), T(32), T(33), T(47.999), T(48)])
            assert(approxEqual(exp(poissonLPMF(32, lambda)), poissonPMF(32, lambda),
                512 * T.epsilon, T(0)));
    }}
    static if (size_t.sizeof == 8)
    {
        assert(approxEqual(poissonLPMF(1_000_000_000_000UL, 1e12),
            -14.734449091169030179221611797845136, 8 * double.epsilon, 0.0));
        assert(poissonLPMF(size_t.max, 5.0) < 0);
    }
    // Forming k/lambda overflows, although its logarithm is finite.
    // log(32!) = 81.55795945611503717850296866601120668710... (100-digit reference).
    import mir.math.common: log;
    const real tinyRate = real.min_normal;
    const real expected = 32 * log(tinyRate)
        - 81.55795945611503717850296866601120668710L;
    assert(approxEqual(poissonLPMF(32, tinyRate), expected,
        32 * real.epsilon, 0.0L));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;

    for (size_t i; i <= 10; i++) {
        assert(i.poissonLPMF(5.0).exp.approxEqual(poissonPMF(i, 5.0)));
    }
}
// BEGIN public Poisson log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,0,UInt!64.fromHexString("1")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(3UL,T(false,0,UInt!64.fromHexString("5")),T(true,-63,UInt!64.fromHexString("fb5230930bc33f89"))),
            Reference(32UL,T(false,0,UInt!64.fromHexString("20")),T(true,-62,UInt!64.fromHexString("a9e1dcdb5818955f"))),
            Reference(1000000000000UL,T(false,0,UInt!64.fromHexString("e8d4a51000")),T(true,-60,UInt!64.fromHexString("ebc04db0b25dd3c2"))),
            Reference(18446744073709551615UL,T(false,0,UInt!64.fromHexString("ffffffffffffffff")),T(true,-59,UInt!64.fromHexString("b8cc1469eaff26d5"))),
            Reference(1UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!64.fromHexString("1")),T(true,11,UInt!64.fromHexString("b2b4fc27c173189e"))),
            Reference(1UL,T(false,66,UInt!64.fromHexString("1")),T(true,2,UInt!64.fromHexString("fffffffffffffff5"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!64.fromHexString("ffffffffffffffff")),T(true,long.max-1,UInt!64.fromHexString("ffffffffffffffff"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max) continue;
            assert(poissonLPMF(cast(size_t)r.k,r.rate) == r.value);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,0,UInt!128.fromHexString("1")),T(true,-127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(3UL,T(false,0,UInt!128.fromHexString("5")),T(true,-127,UInt!128.fromHexString("fb5230930bc33f8967cb28d579916ec4"))),
            Reference(32UL,T(false,0,UInt!128.fromHexString("20")),T(true,-126,UInt!128.fromHexString("a9e1dcdb5818955e92b49e080ddd5f40"))),
            Reference(1000000000000UL,T(false,0,UInt!128.fromHexString("e8d4a51000")),T(true,-124,UInt!128.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e9"))),
            Reference(18446744073709551615UL,T(false,0,UInt!128.fromHexString("ffffffffffffffff")),T(true,-123,UInt!128.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b"))),
            Reference(1UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!128.fromHexString("1")),T(true,-53,UInt!128.fromHexString("b2b4fc27c173189e7185af291a33c099"))),
            Reference(1UL,T(false,66,UInt!128.fromHexString("1")),T(true,-62,UInt!128.fromHexString("fffffffffffffff490257486fa20a86d"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max) continue;
            assert(poissonLPMF(cast(size_t)r.k,r.rate) == r.value);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,0,UInt!256.fromHexString("1")),T(true,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(3UL,T(false,0,UInt!256.fromHexString("5")),T(true,-255,UInt!256.fromHexString("fb5230930bc33f8967cb28d579916ec4119119dea2eddcfc1c416f2034597e6f"))),
            Reference(32UL,T(false,0,UInt!256.fromHexString("20")),T(true,-254,UInt!256.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d5"))),
            Reference(1000000000000UL,T(false,0,UInt!256.fromHexString("e8d4a51000")),T(true,-252,UInt!256.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d4"))),
            Reference(18446744073709551615UL,T(false,0,UInt!256.fromHexString("ffffffffffffffff")),T(true,-251,UInt!256.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7c"))),
            Reference(1UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!256.fromHexString("1")),T(true,-181,UInt!256.fromHexString("b2b4fc27c173189e7185af291a33c098d3ba7fb8966604a6b1d475a6b95bdebb"))),
            Reference(1UL,T(false,66,UInt!256.fromHexString("1")),T(true,-190,UInt!256.fromHexString("fffffffffffffff490257486fa20a86d7cd2ecb3bed719b450522c059d284211"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max) continue;
            assert(poissonLPMF(cast(size_t)r.k,r.rate) == r.value);
        }
    }
    {
        alias T = Fp!512;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,0,UInt!512.fromHexString("1")),T(true,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(3UL,T(false,0,UInt!512.fromHexString("5")),T(true,-511,UInt!512.fromHexString("fb5230930bc33f8967cb28d579916ec4119119dea2eddcfc1c416f2034597e6e9fa29c9bb84a432c08bedb0f0d872d1b813e8e5d8b7ccfb750ed8fb299874a76"))),
            Reference(32UL,T(false,0,UInt!512.fromHexString("20")),T(true,-510,UInt!512.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d4aca06d9e0af2a50055bdb513df118bee2b5b8e220154933855515720464d755d"))),
            Reference(1000000000000UL,T(false,0,UInt!512.fromHexString("e8d4a51000")),T(true,-508,UInt!512.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d42f8ea980296e0afc5e6fc996cf24c22ebbe42027c3ace73635a7dddc47e8327d"))),
            Reference(18446744073709551615UL,T(false,0,UInt!512.fromHexString("ffffffffffffffff")),T(true,-507,UInt!512.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7bcaab31a0562ab4b79085ee0c1f21dab00f9521c95da0c0fa93e58bc1b03c4728"))),
            Reference(1UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-449,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!512.fromHexString("1")),T(true,-437,UInt!512.fromHexString("b2b4fc27c173189e7185af291a33c098d3ba7fb8966604a6b1d475a6b95bdebb24f5569d7b2f7e3b35e4dcf4adbacfc200a55053c651b4550b86a04b37b633ee"))),
            Reference(1UL,T(false,66,UInt!512.fromHexString("1")),T(true,-446,UInt!512.fromHexString("fffffffffffffff490257486fa20a86d7cd2ecb3bed719b450522c059d2842109a27fe997f7ae02b909c62e8ea4f602e7be026cdaae14168b67dc5890be1db1b"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references) {
            if (r.k > size_t.max) continue;
            assert(poissonLPMF(cast(size_t)r.k,r.rate) == r.value);
        }
    }
}
// END public Poisson log references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    foreach (bits; AliasSeq!(64u,128u,256u,512u))
    {
        alias T = Fp!bits;
        const rate = T(1);
        static assert(is(typeof(poissonLPMF(1,rate)) == T));
        assert(poissonLPMF(1,rate) == T(-1));
        // The zero-count result preserves even raw and subnormal inputs.
        UInt!bits ones;
        ones.data[] = size_t.max;
        foreach (value; [T(false,-3,UInt!bits(3)),
                         T(false,long.min,UInt!bits(1)),
                         T(false,long.max-1,ones),rate])
        {
            const result = poissonLPMF(0,value);
            assert(result.sign);
            assert(result.coefficient == value.coefficient);
            assert(result.exponent == value.exponent);
        }
        foreach (k; [size_t(0),size_t(1),size_t.max])
            assert(poissonLPMF(k,T(double.infinity)) == T(-double.infinity));
        debug
        {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            foreach (bad; [T(0),T(-0.0),T(-1),T(-double.infinity),T(double.nan)])
            {
                assertThrown!AssertError(poissonLPMF(0,bad));
                assertThrown!AssertError(poissonLPMF(1,bad));
            }
        }
    }
    foreach (T; AliasSeq!(float,double,real))
        static assert(is(typeof(poissonLPMF(1,T(1))) == T));
}
