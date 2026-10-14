/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.bernoulli;

import mir.internal.utility: isFloatingPoint;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityComplement, fpProbabilityLess;

/++
Computes the bernoulli probability mass function (PMF).

Accepts built-in floating-point probabilities and $(D Fp) probabilities.
The latter retain their supplied precision without conversion to a hardware float.

Params:
    x = value to evaluate PMF
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution)
+/
@safe pure nothrow @nogc
T bernoulliPMF(T)(const bool x, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    return p * x + (1 - p) * (1 - x);
}

/// ditto
@safe pure nothrow @nogc
T bernoulliPMF(T)(const bool x, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    return x ? p : fpProbabilityComplement(p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(true.bernoulliPMF(0.5) == 0.5);
    assert(false.bernoulliPMF(0.5) == 0.5);

    assert(true.bernoulliPMF(0.7).approxEqual(0.7));
    assert(false.bernoulliPMF(0.7).approxEqual(0.3));
}

/// Preserve probabilities smaller than hardware floating-point types can represent.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;

    auto p = Fp!128(1);
    p.exponent -= 2000; // Exactly 2^-2000.
    const mass = bernoulliPMF(true, p);
    assert(mass.coefficient == p.coefficient && mass.exponent == p.exponent);
    const tail = bernoulliCCDF(false, p);
    assert(tail.coefficient == p.coefficient && tail.exponent == p.exponent);
}

/++
Computes the bernoulli cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution)
+/
@safe pure nothrow @nogc
T bernoulliCDF(T)(const bool x, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    return x + (1 - x) * (1 - p);
}

/// ditto
@safe pure nothrow @nogc
T bernoulliCDF(T)(const bool x, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    return x ? T(1) : fpProbabilityComplement(p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(true.bernoulliCDF(0.5) == 1);
    assert(false.bernoulliCDF(0.5) == 0.5);

    assert(true.bernoulliCDF(0.7) == 1);
    assert(false.bernoulliCDF(0.7).approxEqual(0.3));
}

/++
Computes the bernoulli complementary cumulative distribution function (CCDF).

Params:
    x = value to evaluate CCDF
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution)
+/
@safe pure nothrow @nogc
T bernoulliCCDF(T)(const bool x, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    return (1 - x) * p;
}

/// ditto
@safe pure nothrow @nogc
T bernoulliCCDF(T)(const bool x, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    return x ? T(0) : p;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(true.bernoulliCCDF(0.5) == 0);
    assert(false.bernoulliCCDF(0.5) == 0.5);

    assert(true.bernoulliCCDF(0.7) == 0);
    assert(false.bernoulliCCDF(0.7).approxEqual(0.7));
}

/++
Computes the bernoulli inverse cumulative distribution function (InvCDF).

For $(D q < 1), returns $(D false) when $(D q <= 1 - p), and $(D true) otherwise.
At $(D q == 1), returns $(D true) whenever $(D p > 0).
At $(D q == 0), returns $(D false), including when $(D p == 1).

The $(D Fp) overload compares the represented probabilities without conversion
to a hardware type. Its decision is equivalent to an exact $(D q + p > 1),
avoiding rounding of a tiny probability out of the complementary threshold.

Params:
    q = value to evaluate InvCDF
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution)
+/
@safe pure nothrow @nogc
bool bernoulliInvCDF(T)(const T q, const T p)
    if (isFloatingPoint!T)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    // The lower outcome has probability 1 - p; equality selects false.
    // Preserve the upper endpoint even when 1 - p rounds to one.
    return q == 1 ? p > 0 : q > 1 - p;
}

/// ditto
@safe pure nothrow @nogc
bool bernoulliInvCDF(T)(const T q, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(q), "q must be a finite probability in [0, 1]")
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    // The decision is q + p > 1. Complement the larger operand so tiny
    // probabilities cannot disappear in 1 - tiny. If the larger is at least
    // 1/2, its complement is exact at the supplied precision. If both are
    // below 1/2, the comparison is false even if the complement rounds.
    if (fpProbabilityLess(q, p))
        return fpProbabilityLess(fpProbabilityComplement(p), q);
    return fpProbabilityLess(fpProbabilityComplement(q), p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.25.bernoulliInvCDF(0.5) == false);
    assert(0.5.bernoulliInvCDF(0.5) == false);
    assert(0.75.bernoulliInvCDF(0.5) == true);

    assert(0.3.bernoulliInvCDF(0.7) == false);
    assert(0.7.bernoulliInvCDF(0.7) == true);
    assert(0.9.bernoulliInvCDF(0.7) == true);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(0.0.bernoulliInvCDF(0.5) == false);
    assert(1.0.bernoulliInvCDF(0.5) == true);
    assert(0.0.bernoulliInvCDF(0.7) == false);
    assert(1.0.bernoulliInvCDF(0.7) == true);
}

/// Compute quantiles from Fp probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;

    const p = Fp!128(0.75);
    assert(!bernoulliInvCDF(Fp!128(0.25), p));
    assert(bernoulliInvCDF(Fp!128(0.5), p));
}

/++
Computes the bernoulli log probability mass function (LPMF).

Params:
    x = value to evaluate LPMF
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution)
+/
@safe pure nothrow @nogc
T bernoulliLPMF(T)(const bool x, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.internal.xlogy: xlogy, xlog1py;

    return xlogy(x, p) + xlog1py(1 - x, -p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(true.bernoulliLPMF(0.5).approxEqual(log(bernoulliPMF(true, 0.5))));
    assert(false.bernoulliLPMF(0.5).approxEqual(log(bernoulliPMF(false, 0.5))));

    assert(true.bernoulliLPMF(0.7).approxEqual(log(bernoulliPMF(true, 0.7))));
    assert(false.bernoulliLPMF(0.7).approxEqual(log(bernoulliPMF(false, 0.7))));
}

// Asymmetric probabilities expose a swapped inverse-CDF threshold.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {
        static foreach (p; [T(0), T(0.25), T(0.75), T(1)])
        {
            assert(bernoulliPMF(false, p) == 1 - p);
            assert(bernoulliPMF(true, p) == p);
            assert(bernoulliCDF(false, p) == 1 - p);
            assert(bernoulliCDF(true, p) == 1);
            assert(bernoulliCCDF(false, p) == p);
            assert(bernoulliCCDF(true, p) == 0);
            assert(!bernoulliInvCDF(T(0), p));
            assert(!bernoulliInvCDF(T(1 - p), p));
            assert(bernoulliInvCDF(T(1), p) == (p != 0));
        }
        assert(bernoulliInvCDF(T(1), T.min_normal));
        assert(!bernoulliInvCDF(T(1) - T.epsilon, T.min_normal));
        assert(bernoulliInvCDF(T(0.5), T(0.75)));
        assert(!bernoulliInvCDF(T(0.5), T(0.25)));
        assert(bernoulliInvCDF(T(0.25) + T.epsilon, T(0.75)));
        assert(!bernoulliInvCDF(T(0.25) - T.epsilon, T(0.75)));
        assert(bernoulliLPMF(false, T(0)) == 0);
        assert(bernoulliLPMF(true, T(0)) == -T.infinity);
        assert(bernoulliLPMF(false, T(1)) == -T.infinity);
        assert(bernoulliLPMF(true, T(1)) == 0);
    }
}

// Exact complements near one must not pass through double or real.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        UInt!bits ones;
        ones.data[] = size_t.max;
        const p = T(false, -cast(long)bits, ones); // 1 - 2^-bits.
        auto expected = T(1);
        expected.exponent -= bits;
        const mass = bernoulliPMF(false, p);
        const cdf = bernoulliCDF(false, p);
        assert(mass.coefficient == expected.coefficient && mass.exponent == expected.exponent);
        assert(cdf.coefficient == expected.coefficient && cdf.exponent == expected.exponent);
        assert(cast(double)bernoulliCDF(true, p) == 1);
        assert(!bernoulliCCDF(true, p).coefficient);
        assert(!bernoulliPMF(false, T(1)).coefficient);
        assert(!bernoulliPMF(true, T(0)).coefficient);
        assert(cast(double)bernoulliPMF(false, T(0)) == 1);
        assert(cast(double)bernoulliPMF(true, T(1)) == 1);
        auto tiny = T(1);
        tiny.exponent = long.min;
        const identity = bernoulliPMF(true, tiny);
        assert(identity.coefficient == tiny.coefficient && identity.exponent == tiny.exponent);
        assert(cast(double)bernoulliCDF(false, tiny) == 1);
    }}
}

// Compare exact dyadic probabilities on both sides of the inverse threshold.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        static foreach (p; [0.0, 0.25, 0.75, 1.0])
        {
            assert(!bernoulliInvCDF(T(0), T(p)));
            assert(!bernoulliInvCDF(T(1 - p), T(p)));
            assert(bernoulliInvCDF(T(1), T(p)) == (p != 0));
        }
        foreach (i; 0 .. 33)
            foreach (j; 0 .. 33)
                assert(bernoulliInvCDF(T(i / 32.0), T(j / 32.0)) == (i + j > 32));
        auto halfAbove = T(0.5);
        halfAbove.coefficient += UInt!bits(1);
        auto halfBelow = T(0.5);
        halfBelow.coefficient -= UInt!bits(1); // Deliberately unnormalized.
        assert(bernoulliInvCDF(T(0.5), halfAbove));
        assert(bernoulliInvCDF(halfAbove, T(0.5)));
        assert(!bernoulliInvCDF(T(0.5), halfBelow));
        assert(!bernoulliInvCDF(halfBelow, T(0.5)));
        assert(!bernoulliInvCDF(halfAbove, halfBelow)); // Sum is exactly one.
        auto tiny = T(1);
        tiny.exponent = long.min;
        assert(bernoulliInvCDF(T(1), tiny));
        assert(bernoulliInvCDF(tiny, T(1)));
        assert(!bernoulliInvCDF(tiny, tiny));
        // Equivalent representations compare equally, even without normalization.
        const a = T(false, -2, UInt!bits(1));
        const b = T(false, -3, UInt!bits(2));
        assert(!fpProbabilityLess(a, b) && !fpProbabilityLess(b, a));
        assert(fpProbabilityLess(a, T(false, -3, UInt!bits(3))));
        assert(!fpProbabilityLess(T(false, -3, UInt!bits(3)), a));
    }}
}