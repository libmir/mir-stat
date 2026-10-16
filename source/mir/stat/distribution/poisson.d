/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Poisson_distribution, Poisson Distribution).

The count $(D k) is the number of events in an observation interval, and
$(D lambda) is the expected number of events in that interval. Counts use
$(D size_t). Forward functions and count-returning inverse CDFs require
$(D lambda > 0): a zero rate is rejected even though it is a mathematical
limiting case. Tables use a finite positive rate unless stated otherwise.

Ordinary functions use built-in floating-point values. $(LREF fp_poissonPMF)
is a separate $(D Fp) implementation with extended exponent range, not a
full-precision replacement for every calculation. Normal approximations
need not match the direct or gamma algorithms at finite counts.

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
Computes the poisson probability mass function (PMF) directly with extended 
floating point types (e.g. `Fp!128`), which provides additional accuracy for
large values of `lambda` or `k`. 

The exponential is seeded in `double` precision, then squared in the extended
type when necessary to preserve its exponent range. This avoids premature
underflow; it does not provide full coefficient-width transcendental accuracy.

The rate is converted to $(D double) for validation and the exponential seed.
A positive rate too small to remain positive after that conversion is rejected.
If the conversion produces positive infinity, the function returns zero.
Thus an $(D Fp) result does not imply that all rate precision or range is retained.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD Extended-range approximation to $(D exp(-lambda))))
    $(TR $(TD $(D cast(double)lambda == double.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D cast(double)lambda <= 0) or NaN) $(TD Contract violation))
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
    in (cast(double) lambda > 0, "lambda must be greater than or equal to 0")
{
    import mir.math.common: exp;
    import mir.math.internal.fp_powi: fp_powi;
    import mir.math.numeric: factorial;

    double rate = cast(double) lambda;
    if (rate == double.infinity)
        return T(0);
    // exp(-rate) can underflow in double even when the final mass is ordinary.
    // Halving is exact; start with a normal exponential and perform the remaining
    // squarings in Fp, whose exponent range can retain the very small factor.
    uint squarings;
    while (rate > 512)
    {
        rate *= 0.5;
        ++squarings;
    }
    T decay = T(exp(-rate));
    foreach (i; 0 .. squarings)
        decay *= decay;
    if (k == 0)
        return decay;
    return decay * fp_powi(lambda, k) / factorial(k);
}

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
    // A finite extended rate can exceed the double exponential seed's range.
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

Returns the natural logarithm of $(D P(X == k)), which can remain finite
when the PMF underflows to zero.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D -lambda)))
    $(TR $(TD $(D lambda == T.infinity)) $(TD $(D -T.infinity)))
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
