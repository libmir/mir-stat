/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution).

There are multiple alternative parameterizations of the Hypergeometric Distribution.
The formulation in this module measures the number of draws (`k`) with a
specific feature in `n` total draws without replacement from a population of
size `N` such that `K` of these have the feature of interest.

Hypergeometric distribution functions can utilize different algorithms. The 
default is `HypergeometricAlgo.direct`, which can be more time-consuming for
large values of the parameters. Additional algorithms are provided to the user
to choose the trade-off between running time and accuracy.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.hypergeometric;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

/++
Algorithms used to calculate hypergeometric distribution.

`HypergeometricAlgo.direct` can be more time-consuming for large values of the
parameters. Additional algorithms are provided to the user to choose the
trade-off between running time and accuracy.
+/
enum HypergeometricAlgo {
    /++
    Direct
    +/
    direct,
    /++
    Approximates hypergeometric distribution with binomial distribution.
    +/
    approxBinomial,
    /++
    Approximates hypergeometric distribution with poisson distribution (uses gamma approximation, except for inverse CDF).
    +/
    approxPoisson,
    /++
    Approximates hypergeometric distribution with normal distribution.
    +/
    approxNormal,
    /++
    Approximates hypergeometric distribution with normal distribution (including continuity correction).
    +/
    approxNormalContinuityCorrection,
}

private
@safe pure @nogc nothrow
T hypergeometricPMFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.direct)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.combinatorics: binomial;
    const numerator = binomial(K, k) * binomial(N - K, n - k);
    const denominator = binomial(N, n);
    if (denominator != 0)
        return cast(T) numerator / denominator;
    return hypergeometricLargePMF!T(k, N, K, n);
}

private
@safe pure @nogc nothrow
T hypergeometricPMFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxBinomial)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.binomial: binomialPMF;

    return binomialPMF(k, n, cast(T) K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricPMFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxPoisson)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.poisson: poissonPMF, PoissonAlgo;

    return poissonPMF!(PoissonAlgo.gamma)(k, cast(T) n * K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricPMFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && 
        hypergeometricAlgo == HypergeometricAlgo.approxNormal)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalPDF;

    T l = k;
    return normalPDF(l, (cast(T) n * K) / N, sqrt(n * (cast(T) K / N) * (cast(T) (N - K) / N) * (cast(T) (N - n) / (N - 1))));
}

private
@safe pure @nogc nothrow
T hypergeometricPMFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && 
        hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    T location = (cast(T) n * K) / N;
    T scale = sqrt(n * (cast(T) K / N) * (cast(T) (N - K) / N) * (cast(T) (N - n) / (N - 1)));
    return normalCDF(k + 0.5, location, scale) - normalCDF(k - 0.5, location, scale);
}

/++
Computes the hypergeometric probability mass function (PMF).

The direct algorithm uses integer coefficients where they fit, and extended
precision otherwise. Request `hypergeometricPMF!(Fp!128)` to retain an extended
precision result instead of converting to a built-in floating-point type.

Additional algorithms may be provided for calculating PMF that allow trading off
time and accuracy. If `approxPoisson` is provided, `PoissonAlgo.gamma` is assumed.

Params:
    k = value to evaluate PMF (e.g. number of correct draws of object of interest)
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
T hypergeometricPMF(T, HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    if (n == 0 || K == 0 || n == N || K == N)
        return 1;
    return hypergeometricPMFImpl!(T, hypergeometricAlgo)(k, N, K, n);
}

/// ditto
template hypergeometricPMF(HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
{
    alias hypergeometricPMF = hypergeometricPMF!(double, hypergeometricAlgo);
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricPMF(T, string hypergeometricAlgo)
{
    mixin("alias hypergeometricPMF = .hypergeometricPMF!(T, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricPMF(string hypergeometricAlgo)
{
    mixin("alias hypergeometricPMF = .hypergeometricPMF!(double, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricPMF(7, 4, 3).shouldApprox == 0.02857143;
    1.hypergeometricPMF(7, 4, 3).shouldApprox == 0.3428571;
    2.hypergeometricPMF(7, 4, 3).shouldApprox == 0.5142857;
    3.hypergeometricPMF(7, 4, 3).shouldApprox == 0.1142857;

    // can also provide a template argument to change output type
    static assert(is(typeof(hypergeometricPMF!float(3, 7, 4, 3)) == float));
}

// Check n=0 condition
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricPMF(7, 4, 0).shouldApprox == 1.0;
}

/// Alternate algorithms
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: exp;

    // Can approximate hypergeometric with binomial distribution
    20.hypergeometricPMF!"approxBinomial"(750_000, 250_000, 50).shouldApprox == exp(hypergeometricLPMF(20, 750_000, 250_000, 50));
    // Can approximate hypergeometric with poisson distribution
    20.hypergeometricPMF!"approxPoisson"(100_000, 100, 5_000).shouldApprox == exp(hypergeometricLPMF(20, 100_000, 100, 5_000));
    // Can approximate hypergeometric with normal distribution
    3_500.hypergeometricPMF!"approxNormal"(10_000, 7_500, 5_000).shouldApprox == exp(hypergeometricLPMF(3_500, 10_000, 7_500, 5_000));
    // Can approximate hypergeometric with normal distribution (with continuity correction)
    3_500.hypergeometricPMF!"approxNormalContinuityCorrection"(10_000, 7_500, 5_000).shouldApprox == exp(hypergeometricLPMF(3_500, 10_000, 7_500, 5_000));
}

/++
Compatibility name for the extended-precision $(LREF hypergeometricPMF)
overload. New code can request `hypergeometricPMF!(Fp!128)` explicitly.
The default precision of this wrapper remains `Fp!128`.

Params:
    k = value to evaluate PMF (e.g. number of correct draws of object of interest)
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
T fp_hypergeometricPMF(T = Fp!128)(const size_t k, const size_t N, const size_t K, const size_t n)
    if (is(T == Fp!size, size_t size))
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    return hypergeometricPMF!T(k, N, K, n);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;
    size_t N = val + 5;
    size_t K = val / 2;
    size_t n = val / 100;
    0.fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(0, N, K, n);
    1.fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(1, N, K, n);
    2.fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(2, N, K, n);
    5.fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(5, N, K, n);
    (n / 2).fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(n / 2, N, K, n);
    (n - 5).fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(n - 5, N, K, n);
    (n - 2).fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(n - 2, N, K, n);
    (n - 1).fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(n - 1, N, K, n);
    n.fp_hypergeometricPMF(N, K, n).fp_log!double.shouldApprox == hypergeometricLPMF(n, N, K, n);
}

/++
Computes the direct PMF at the explicitly requested `Fp` precision.
Coefficient products and division retain extended precision; counts are not
narrowed to `uint`. Cost grows with the smaller side of each binomial
coefficient, so very large central counts may still be expensive.

Params:
    k = number of successes drawn
    N = population size
    K = successes in the population
    n = number of draws
+/
@safe pure nothrow @nogc
T hypergeometricPMF(T)(const size_t k, const size_t N, const size_t K, const size_t n)
    if (is(T == Fp!bits, uint bits))
    in (K <= N)
    in (n <= N)
    in (k <= n)
    in (k <= K)
    in (n - k <= N - K)
{
    // Extra coefficient bits absorb intermediate rounding. Do not use Fp's
    // division operator: it currently converts through hardware real.
    static if (is(T == Fp!bits, uint bits)) {
        const numerator = hypergeometricCoefficient!(bits + 64)(K, k) *
            hypergeometricCoefficient!(bits + 64)(N - K, n - k);
        const denominator = hypergeometricCoefficient!(bits + 64)(N, n);
        return cast(T) hypergeometricDivide(numerator, denominator);
    }
}

private @safe pure nothrow @nogc
Fp!bits hypergeometricDivide(uint bits)(Fp!bits numerator, Fp!bits denominator)
{
    import mir.bignum.fixed: UInt;
    UInt!(2 * bits + 64) quotient = numerator.coefficient.toSize!(2 * bits + 64);
    quotient <<= bits + 64;
    quotient /= denominator.coefficient;
    auto result = Fp!bits(quotient);
    result.exponent += numerator.exponent - denominator.exponent - cast(long)bits - 64;
    return result;
}

private @safe pure nothrow @nogc
Fp!bits hypergeometricCoefficient(uint bits)(size_t n, size_t k)
{
    import mir.math.numeric: factorial;
    if (k > n - k) k = n - k;
    if (k == 0) return Fp!bits(1);
    return hypergeometricDivide(factorial!bits(k, n - k + 1), factorial!bits(k));
}

/// Select extended precision explicitly, or retain the compatibility name.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        auto mass = hypergeometricPMF!T(1, 4, 2, 2);
        static assert(is(typeof(mass) == T));
        assert(cast(double)mass == 2.0 / 3);
        assert(mass == fp_hypergeometricPMF!T(1, 4, 2, 2));
        // 1/(2^32+1) = (2^32-1)/(2^64-1). Its repeating binary
        // expansion exercises precision beyond hardware floating point.
        static if (size_t.sizeof == 8) {
            import mir.bignum.fixed: UInt;
            const result = hypergeometricPMF!T(0x100000000UL, 0x100000001UL,
                0x100000000UL, 0x100000000UL);
            UInt!(bits + 128) scaled = UInt!(bits + 128)(1);
            scaled <<= bits + 96;
            scaled /= UInt!64(0x100000001UL);
            auto expected = T(scaled);
            expected.exponent -= bits + 96;
            assert(result.exponent == expected.exponent);
            assert(result.coefficient == expected.coefficient);
        }
    }}
}

// Exact integer combinations give the rounded 128-bit coefficient below.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    enum expected = UInt!128("215582029535793405898658859699468210548");
    const mass = hypergeometricPMF!(Fp!128)(25, 100, 50, 50);
    assert(mass.exponent == -130);
    assert(mass.coefficient == expected);
}

// Keep extended-precision machinery out of the ordinary integer fast path.
private @safe pure nothrow @nogc
T hypergeometricLargePMF(T)(size_t k, size_t N, size_t K, size_t n)
{
    pragma(inline, false);
    return hypergeometricToFloat!T(hypergeometricPMF!(Fp!128)(k, N, K, n));
}

// Normalize before conversion so a very small Fp exponent is not narrowed
// inside a floating-point cast. ldexp performs the final subnormal scaling.
private @safe pure nothrow @nogc
T hypergeometricToFloat(T)(Fp!128 value)
{
    import std.math: ldexp;
    const exponent = value.exponent + 128;
    if (exponent < T.min_exp - T.mant_dig)
        return 0;
    value.exponent = -128;
    // Some runtime ldexp implementations truncate subnormal results. Keep
    // both factors normal and let hardware multiplication round the result.
    if (exponent < T.min_exp)
        return ldexp(cast(T)value, T.min_exp) * ldexp(T(1), cast(int)exponent - T.min_exp);
    return ldexp(cast(T)value, cast(int)exponent);
}

private @safe pure nothrow @nogc
T hypergeometricTail(T, bool upper)(size_t k, size_t N, size_t K, size_t n)
{
    import mir.combinatorics: binomial;
    const lower = n > N - K ? n - (N - K) : 0;
    const last = n < K ? n : K;
    static if (upper) {
        if (k == last) return 0;
        ++k;
    }
    const denominator = binomial(N, n);
    T sum = 0;
    if (denominator != 0) {
        // Every numerator is at most the denominator, so its product fits too.
        static if (upper) {
            for (size_t i = k;; ++i) {
                sum += cast(T)(binomial(K, i) * binomial(N - K, n - i)) / denominator;
                if (i == last) break;
            }
        } else {
            for (size_t i = lower;; ++i) {
                sum += cast(T)(binomial(K, i) * binomial(N - K, n - i)) / denominator;
                if (i == k) break;
            }
        }
    } else {
        // Relative terms shrink as we move away from the mean. Keep the
        // initial mass in Fp until after summation: an individual mass can
        // underflow even when the sum of the tail is representable.
        const initial = hypergeometricPMF!(Fp!128)(k, N, K, n);
        T term = 1;
        sum = 1;
        size_t i = k;
        while (true) {
            static if (upper) {
                if (i == last) break;
                term *= (cast(T)(K - i) / (i + 1)) *
                    (cast(T)(n - i) / ((N - K) - (n - i) + 1));
                ++i;
            } else {
                if (i == lower) break;
                term *= (cast(T)i / (K - i + 1)) *
                    (cast(T)((N - K) - (n - i)) / (n - i + 1));
                --i;
            }
            const next = sum + term;
            if (next == sum) break;
            sum = next;
        }
        sum = hypergeometricToFloat!T(initial * Fp!128(sum));
    }
    return sum > 1 ? T(1) : sum;
}

private
@safe pure @nogc nothrow
T hypergeometricCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.direct)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    // Sum the tail on its own side of the mean. Using n/2 instead can
    // subtract a tiny probability from a CDF already rounded to one.
    if (k < cast(T)n * (cast(T)K / N))
        return hypergeometricTail!(T, false)(k, N, K, n);
    return 1 - hypergeometricTail!(T, true)(k, N, K, n);
}

private
@safe pure @nogc nothrow
T hypergeometricCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxBinomial)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.binomial: binomialCDF;

    return binomialCDF(k, n, cast(T) K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxPoisson)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.poisson: poissonCDF, PoissonAlgo;

    return poissonCDF!(PoissonAlgo.gamma)(k, cast(T) n * K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && 
        (hypergeometricAlgo == HypergeometricAlgo.approxNormal || 
         hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection))
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCDF;

    T l = k;
    static if (hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCDF(l, (cast(T) n * K) / N, sqrt(n * (cast(T) K / N) * (cast(T) (N - K) / N) * (cast(T) (N - n) / (N - 1))));
}

/++
Computes the hypergeometric cumulative distribution function (CDF).

Additional algorithms may be provided for calculating CDF that allow trading off
time and accuracy. If `approxPoisson` is provided, `PoissonAlgo.gamma` is assumed.

Setting `hypergeometricAlgo = HypergeometricAlgo.direct` results in direct
summation being used, which can result in significant slowdowns for large values
of `k`. 

Params:
    k = value to evaluate CDF (e.g. number of correct draws of object of interest)
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
T hypergeometricCDF(T, HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    if (k == n || k == K || n == 0)
        return 1;
    return hypergeometricCDFImpl!(T, hypergeometricAlgo)(k, N, K, n);
}

/// ditto
template hypergeometricCDF(HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
{
    alias hypergeometricCDF = hypergeometricCDF!(double, hypergeometricAlgo);
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricCDF(T, string hypergeometricAlgo)
{
    mixin("alias hypergeometricCDF = .hypergeometricCDF!(T, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricCDF(string hypergeometricAlgo)
{
    mixin("alias hypergeometricCDF = .hypergeometricCDF!(double, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricCDF(7, 4, 3).shouldApprox == 0.02857143;
    1.hypergeometricCDF(7, 4, 3).shouldApprox == 0.3714286;
    2.hypergeometricCDF(7, 4, 3).shouldApprox == 0.8857143;
    3.hypergeometricCDF(7, 4, 3).shouldApprox == 1.0;

    // can also provide a template argument to change output type
    static assert(is(typeof(hypergeometricCDF!float(3, 7, 4, 3)) == float));
}

// Check n=0 condition
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricCDF(7, 4, 0).shouldApprox == 1.0;
}

/// Alternate algorithms
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    // Can approximate hypergeometric with binomial distribution
    20.hypergeometricCDF!"approxBinomial"(750_000, 250_000, 50).shouldApprox(1e-2) == 0.8740839;
    // Can approximate hypergeometric with poisson distribution
    8.hypergeometricCDF!"approxPoisson"(100_000, 100, 5_000).shouldApprox(1e-2) == 0.9370063;
    // Can approximate hypergeometric with normal distribution
    3_750.hypergeometricCDF!"approxNormal"(10_000, 7_500, 5_000).shouldApprox(2e-2) == 0.5092122;
    // Can approximate hypergeometric with normal distribution
    3_750.hypergeometricCDF!"approxNormalContinuityCorrection"(10_000, 7_500, 5_000).shouldApprox(1e-2) == 0.5092122;
}

private
@safe pure @nogc nothrow
T hypergeometricCCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.direct)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    if (k < cast(T)n * (cast(T)K / N))
        return 1 - hypergeometricTail!(T, false)(k, N, K, n);
    return hypergeometricTail!(T, true)(k, N, K, n);
}

private
@safe pure @nogc nothrow
T hypergeometricCCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxBinomial)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.binomial: binomialCCDF;

    return binomialCCDF(k, n, cast(T) K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricCCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxPoisson)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.stat.distribution.poisson: poissonCCDF, PoissonAlgo;

    return poissonCCDF!(PoissonAlgo.gamma)(k, cast(T) n * K / N);
}

private
@safe pure @nogc nothrow
T hypergeometricCCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && 
        (hypergeometricAlgo == HypergeometricAlgo.approxNormal || 
         hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection))
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.normal: normalCCDF;

    T l = k;
    static if (hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection) {
        l += 0.5;
    }
    return normalCCDF(l, (cast(T) n * K) / N, sqrt(n * (cast(T) K / N) * (cast(T) (N - K) / N) * (cast(T) (N - n) / (N - 1))));
}

/++
Computes the hypergeometric complementary cumulative distribution function (CCDF).

Additional algorithms may be provided for calculating CCDF that allow trading off
time and accuracy. If `approxPoisson` is provided, `PoissonAlgo.gamma` is assumed.

Setting `hypergeometricAlgo = HypergeometricAlgo.direct` results in direct
summation being used, which can result in significant slowdowns for large values
of `k`. 

Params:
    k = value to evaluate CCDF (e.g. number of correct draws of object of interest)
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
T hypergeometricCCDF(T, HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
        (const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    if (k == n || k == K || n == 0)
        return 0;
    return hypergeometricCCDFImpl!(T, hypergeometricAlgo)(k, N, K, n);
}

/// ditto
template hypergeometricCCDF(HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
{
    alias hypergeometricCCDF = hypergeometricCCDF!(double, hypergeometricAlgo);
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricCCDF(T, string hypergeometricAlgo)
{
    mixin("alias hypergeometricCCDF = .hypergeometricCCDF!(T, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricCCDF(string hypergeometricAlgo)
{
    mixin("alias hypergeometricCCDF = .hypergeometricCCDF!(double, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricCCDF(7, 4, 3).shouldApprox == 0.9714286;
    1.hypergeometricCCDF(7, 4, 3).shouldApprox == 0.6285714;
    2.hypergeometricCCDF(7, 4, 3).shouldApprox == 0.1142857;
    3.hypergeometricCCDF(7, 4, 3).shouldApprox == 0.0;

    // can also provide a template argument to change output type
    static assert(is(typeof(hypergeometricCCDF!float(3, 7, 4, 3)) == float));
}

// Check n=0 condition
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.hypergeometricCCDF(7, 4, 0).shouldApprox == 0.0;
}

/// Alternate algorithms
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: exp;

    // Can approximate hypergeometric with binomial distribution
    20.hypergeometricCCDF!"approxBinomial"(750_000, 250_000, 50).shouldApprox(1e-2) == 0.1259161;
    // Can approximate hypergeometric with poisson distribution
    8.hypergeometricCCDF!"approxPoisson"(100_000, 100, 5_000).shouldApprox(1e-1) == 0.0629937;
    // Can approximate hypergeometric with normal distribution
    3_750.hypergeometricCCDF!"approxNormal"(10_000, 7_500, 5_000).shouldApprox(2e-2) == 0.4907878;
    // Can approximate hypergeometric with normal distribution
    3_750.hypergeometricCCDF!"approxNormalContinuityCorrection"(10_000, 7_500, 5_000).shouldApprox(1e-2) == 0.4907878;
}

private
@safe pure nothrow @nogc
size_t hypergeometricInvCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const T p, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.direct)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    const lower = n > N - K ? n - (N - K) : 0;
    const upper = n < K ? n : K;
    size_t guess = lower;
    if (n < 0.1 * N) {
        if ((cast(T) n * K > 9 * cast(T)(N - K)) &&
            (cast(T) n * (N - K) > 9 * cast(T) K)) {
            guess = hypergeometricInvCDFImpl!(T, HypergeometricAlgo.approxNormalContinuityCorrection)(p, N, K, n);
        } else if (K < 0.1 * N) {
            guess = hypergeometricInvCDFImpl!(T, HypergeometricAlgo.approxPoisson)(p, N, K, n);
        } else {
            guess = hypergeometricInvCDFImpl!(T, HypergeometricAlgo.approxBinomial)(p, N, K, n);
        }
    }
    if (guess < lower) guess = lower;
    if (guess > upper) guess = upper;
    T cdfGuess = hypergeometricCDF!(T, hypergeometricAlgo)(guess, N, K, n);
    if (p <= cdfGuess) {
        while (guess > lower) {
            const previous = cdfGuess - hypergeometricPMF!(T, hypergeometricAlgo)(guess, N, K, n);
            if (p > previous) break;
            --guess;
            cdfGuess = previous;
        }
    } else {
        while (guess < upper && p > cdfGuess) {
            ++guess;
            cdfGuess += hypergeometricPMF!(T, hypergeometricAlgo)(guess, N, K, n);
        }
    }
    return guess;
}

private
@safe pure @nogc nothrow
size_t hypergeometricInvCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const T p, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxBinomial)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.binomial: binomialInvCDF;

    return binomialInvCDF(p, n, cast(T) K / N);
}

private
@safe pure @nogc nothrow
size_t hypergeometricInvCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const T p, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && hypergeometricAlgo == HypergeometricAlgo.approxPoisson)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.distribution.poisson: poissonInvCDF, PoissonAlgo;

    // Using PoissonAlgo.direct because PoissonAlgo.gamma does not return the
    // same result
    return poissonInvCDF!(PoissonAlgo.direct)(p, cast(T) n * K / N);
}

private
@safe pure @nogc nothrow
size_t hypergeometricInvCDFImpl(T, HypergeometricAlgo hypergeometricAlgo)
        (const T p, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T && 
        (hypergeometricAlgo == HypergeometricAlgo.approxNormal || 
         hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection))
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalCDF, normalInvCDF;

    T location = (cast(T) n * K) / N;
    T scale = sqrt(n * (cast(T) K / N) * (cast(T) (N - K) / N) * (cast(T) (N - n) / (N - 1)));

    // Handles case where p is small or large, better than just using pLowerBound = 0 or pUpperBound = 1
    T pLowerBound = 0;
    T pUpperBound = 1;
    T lowerValue = 0;
    T upperValue = K;
    static if (hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection) {
        lowerValue += 0.5;
        upperValue -= 0.5;
    }
    pLowerBound = normalCDF(lowerValue, location, scale);
    pUpperBound = normalCDF(upperValue, location, scale);
    if (p <= pLowerBound) {
        return 0;
    } else if (p >= pUpperBound) {
        return K;
    }

    auto result = normalInvCDF(p, location, scale);
    static if (hypergeometricAlgo == HypergeometricAlgo.approxNormalContinuityCorrection) {
        result = result - 0.5;
    }
    return cast(size_t) floor(result);
}

/++
Computes the hypergeometric inverse cumulative distribution function (InvCDF).

For positive probabilities, results lie between `max(0, n - (N - K))` and
`min(n, K)`. At `p = 0`, this function retains its convention of returning zero
even when the smallest attainable number of successes is positive.

Additional algorithms may be provided for calculating InvCDF that allow trading off
time and accuracy. If `approxPoisson` is provided, `PoissonAlgo.direct` is assumed.
This is different from other functions that use `PoissonAlgo.gamma` since in this
case it does not provide the same result.

Setting `hypergeometricAlgo = HypergeometricAlgo.direct` results in direct
summation being used, which can result in significant slowdowns for large values
of `k`. 

Params:
    p = value to evaluate InvCDF
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
size_t hypergeometricInvCDF(T, HypergeometricAlgo hypergeometricAlgo = HypergeometricAlgo.direct)
        (const T p, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    // Preserve the existing p=0 convention; positive probabilities are
    // bounded by the actual support, which need not begin at zero.
    if (n == 0 || p == 0)
        return 0;
    const lower = n > N - K ? n - (N - K) : 0;
    const upper = n < K ? n : K;
    if (lower == upper || p == 1)
        return upper;
    const result = hypergeometricInvCDFImpl!(T, hypergeometricAlgo)(p, N, K, n);
    return result < lower ? lower : result > upper ? upper : result;
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricInvCDF(T, string hypergeometricAlgo)
{
    mixin("alias hypergeometricInvCDF = .hypergeometricInvCDF!(T, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

/// ditto
@safe pure nothrow @nogc
template hypergeometricInvCDF(string hypergeometricAlgo)
{
    mixin("alias hypergeometricInvCDF = .hypergeometricInvCDF!(double, HypergeometricAlgo." ~ hypergeometricAlgo ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    0.0.hypergeometricInvCDF(40, 15, 20).should == 0;
    0.1.hypergeometricInvCDF(40, 15, 20).should == 6;
    0.2.hypergeometricInvCDF(40, 15, 20).should == 6;
    0.3.hypergeometricInvCDF(40, 15, 20).should == 7;
    0.4.hypergeometricInvCDF(40, 15, 20).should == 7;
    0.5.hypergeometricInvCDF(40, 15, 20).should == 7;
    0.6.hypergeometricInvCDF(40, 15, 20).should == 8;
    0.7.hypergeometricInvCDF(40, 15, 20).should == 8;
    0.8.hypergeometricInvCDF(40, 15, 20).should == 9;
    0.9.hypergeometricInvCDF(40, 15, 20).should == 9;
    1.0.hypergeometricInvCDF(40, 15, 20).should == 15;
}

// Check n=0 condition
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    0.0.hypergeometricInvCDF(7, 4, 0).should == 0;
    0.5.hypergeometricInvCDF(7, 4, 0).should == 0;
    1.0.hypergeometricInvCDF(7, 4, 0).should == 0;
}

/// Alternate algorithms
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: exp;

    // Can approximate hypergeometric with binomial distribution
    0.5.hypergeometricInvCDF!"approxBinomial"(750_000, 250_000, 50).shouldApprox!double == 17;
    // Can approximate hypergeometric with poisson distribution
    0.4.hypergeometricInvCDF!"approxPoisson"(100_000, 100, 5_000).shouldApprox!double == 4;
    // Can approximate hypergeometric with normal distribution
    0.6.hypergeometricInvCDF!"approxNormal"(10_000, 7_500, 5_000).shouldApprox!double == 3755;
    // Can approximate hypergeometric with normal distribution
    0.6.hypergeometricInvCDF!"approxNormalContinuityCorrection"(10_000, 7_500, 5_000).shouldApprox!double(1) == 3755;
}

// test approxNormal / approxNormalContinuityCorrection
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalInvCDF;
    import mir.test: should;

    size_t N = 50;
    size_t K = 15;
    size_t n = 25;
    double location = cast(double) (n * K) / N;
    double scale = sqrt(n * (cast(double) K / N) * (cast(double) (N - K) / N) * (cast(double) (N - n) / (N - 1)));
    0.0000001.hypergeometricInvCDF!"approxNormal"(N, K, n).should == 0;
    0.0000001.hypergeometricInvCDF!"approxNormalContinuityCorrection"(N, K, n).should == 0;
    0.9999999.hypergeometricInvCDF!"approxNormal"(N, K, n).should == K;
    0.9999999.hypergeometricInvCDF!"approxNormalContinuityCorrection"(N, K, n).should == K;
    double checkValue;
    for (double x = 0.05; x < 1; x = x + 0.05) {
        checkValue = normalInvCDF(x, location, scale);
        x.hypergeometricInvCDF!"approxNormal"(N, K, n).should == floor(checkValue);
        x.hypergeometricInvCDF!"approxNormalContinuityCorrection"(N, K, n).should == floor(checkValue - 0.5);
    }
}

// test alternate direct guess paths
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    static immutable size_t[] Ns = [120, 100, 200];
    static immutable size_t[] Ks = [ 50,  50,   9];
    static immutable size_t[] ns = [ 10,   9,   1];

    size_t value;
    for (size_t i; i < Ns.length; i++) {
        for (double x = 0.01; x < 1; x = x + 0.01) {
            value = x.hypergeometricInvCDF(Ns[i], Ks[i], ns[i]);
            assert(value.hypergeometricCDF(Ns[i], Ks[i], ns[i]) >= x);
            if (value > 1)
                assert((value - 1).hypergeometricCDF(Ns[i], Ks[i], ns[i]) < x);
        }
    }
}

/++
Computes the hypergeometric log probability mass function (LPMF).

Params:
    k = value to evaluate LPMF (e.g. number of correct draws of object of interest)
    N = total population size
    K = number of relevant objects in population
    n = number of draws

See_also: $(LINK2 https://en.wikipedia.org/wiki/Hypergeometric_distribution, Hypergeometric Distribution)
+/
@safe pure @nogc nothrow
T hypergeometricLPMF(T = double)(const size_t k, const size_t N, const size_t K, const size_t n)
    if (isFloatingPoint!T)
    in (K <= N, "K must be less than or equal to N")
    in (n <= N, "n must be less than or equal to N")
    in (k <= n, "n - k must be greater than or equal to zero")
    in (k <= K, "k must be less than or equal to K")
    in (n - k <= N - K, "`N - K` must be greater than or equal to `n - k`")
{
    import mir.math.internal.log_binomial: logBinomialCoefficient;
    
    return logBinomialCoefficient(K, cast(const uint) k) + 
           logBinomialCoefficient(N - K, cast(const uint) (n - k)) - 
           logBinomialCoefficient(N, cast(const uint) n);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    0.hypergeometricLPMF(7, 4, 3).shouldApprox == log(0.02857143);
    1.hypergeometricLPMF(7, 4, 3).shouldApprox == log(0.3428571);
    2.hypergeometricLPMF(7, 4, 3).shouldApprox == log(0.5142857);
    3.hypergeometricLPMF(7, 4, 3).shouldApprox == log(0.1142857);
}

// Actual support and deterministic draws apply to every approximation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (algo; __traits(allMembers, HypergeometricAlgo)) {{
        enum a = __traits(getMember, HypergeometricAlgo, algo);
        assert(hypergeometricInvCDF!(double, a)(1.0, 10, 8, 4) == 4);
        assert(hypergeometricPMF!a(3, 10, 3, 10) == 1);
        assert(hypergeometricCDF!a(3, 10, 3, 10) == 1);
        assert(hypergeometricCCDF!a(3, 10, 3, 10) == 0);
        assert(hypergeometricInvCDF!(double, a)(0.5, 10, 3, 10) == 3);
    }}
    assert(hypergeometricInvCDF(0.5, 10, 8, 4) == 3);
    assert(hypergeometricInvCDF(0.01, 10, 8, 4) == 2);
    assert(hypergeometricInvCDF(0.99, 10, 8, 4) == 4);
    assert(hypergeometricInvCDF(0.0, 10, 8, 4) == 0);
    assert(hypergeometricPMF(1, size_t.max, 1, 1) > 0);
}

// Coefficients exceed ulong even though the probabilities are ordinary.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.test: shouldApprox;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real)) {
        hypergeometricPMF!T(25, 100, 50, 50).shouldApprox == T(0.15838466116133042726L);
        hypergeometricCDF!T(25, 100, 50, 50).shouldApprox == T(0.57919233058066521363L);
        hypergeometricCCDF!T(25, 100, 50, 50).shouldApprox == T(0.42080766941933478637L);
        assert(hypergeometricPMF!T(0, 20000, 10000, 10000) == 0);
        assert(hypergeometricCDF!T(0, 20000, 10000, 10000) == 0);
    }
    assert(hypergeometricPMF(99, 2000, 1000, 1000) == 0x0.0000000000005p-1022);
    assert(hypergeometricCDF(99, 2000, 1000, 1000) == 0x0.0000000000005p-1022);
    assert(hypergeometricCCDF(900, 2000, 1000, 1000) == 0x0.0000000000005p-1022);
    static if (size_t.sizeof == 8) {
        hypergeometricCCDF(0, 1_000_000_000_000_000_000UL, 1, 1).shouldApprox == 1e-18;
    }
}
