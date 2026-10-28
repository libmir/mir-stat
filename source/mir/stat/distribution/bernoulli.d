/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Bernoulli_distribution, Bernoulli Distribution).

The outcomes are $(D false) (zero) and $(D true) (one), with probabilities
$(D 1 - p) and $(D p). The parameter $(D p) must be finite and in $(D [0, 1]);
both endpoints describe a deterministic outcome.

PMF, CDF, CCDF, and inverse CDF accept built-in floating-point or $(D Fp)
probabilities. LPMF also accepts both types and returns a natural logarithm.
The inverse CDF returns $(D bool).

Finite expressions are subject to floating-point rounding. Contract violations
are not prescribed return values: disabling contracts does not make an invalid
input supported.

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

Returns the point probability $(D P(X == x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == false)) $(TD $(D 1 - p)))
    $(TR $(TD $(D x == true)) $(TD $(D p)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PMF
    p = finite probability of $(D true), in $(D [0, 1])

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

Returns $(D P(X <= x)), ordering $(D false) before $(D true).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == false)) $(TD $(D 1 - p)))
    $(TR $(TD $(D x == true)) $(TD $(D 1)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    p = finite probability of $(D true), in $(D [0, 1])

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

Returns the strict upper-tail probability $(D P(X > x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == false)) $(TD $(D p)))
    $(TR $(TD $(D x == true)) $(TD $(D 0)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    p = finite probability of $(D true), in $(D [0, 1])

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

For $(D q < 1), equality at the lower outcome's cumulative probability
selects $(D false); the upper endpoint follows the rule in the table.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 0)) $(TD $(D false), even when $(D p == 1)))
    $(TR $(TD $(D p == 0)) $(TD $(D false) for every valid $(D q)))
    $(TR $(TD $(D q == 1) and $(D p > 0)) $(TD $(D true)))
    $(TR $(TD $(D p == 1) and $(D q > 0)) $(TD $(D true)))
    $(TR $(TD $(D p) or $(D q) outside $(D [0, 1]) or NaN) $(TD Contract violation))
)

Params:
    q = cumulative probability in $(D [0, 1])
    p = finite probability of $(D true), in $(D [0, 1])

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

/// A deterministic outcome still uses the lower endpoint convention at zero probability.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(bernoulliPMF(true, 1.0) == 1);
    assert(bernoulliCDF(false, 1.0) == 0);
    assert(bernoulliCCDF(false, 1.0) == 1);
    assert(!bernoulliInvCDF(0.0, 1.0));
    assert(bernoulliInvCDF(0.5, 1.0));
}

/++
Computes the bernoulli log probability mass function (LPMF).

Returns the natural logarithm of $(D P(X == x)).

The $(D Fp) overload evaluates logarithms directly from the represented probability,
without conversion to a hardware float or first rounding $(D 1 - p).
It uses guarded integer series, retaining 192 extra bits before final rounding;
this is not a universal correctly-rounded guarantee. Tiny probabilities retain
nonzero $(D log(1 - p)) results, including at the Fp exponent floor.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0), $(D x == false); or $(D p == 1), $(D x == true)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0), $(D x == true); or $(D p == 1), $(D x == false)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPMF
    p = finite probability of $(D true), in $(D [0, 1])

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

/// ditto
@safe pure nothrow @nogc
T bernoulliLPMF(T)(const bool x, const T p)
    if (is(T == Fp!bits, uint bits))
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    if (!p.coefficient) return T(x ? -double.infinity : 0);
    if (!fpProbabilityLess(p,T(1))) return T(x ? 0 : -double.infinity);
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    alias L = ProbabilityLog!bits;
    auto result = (x ? L.probability(p) : L.complement(p)).finish!bits;
    result.sign = true;
    return result;
}

/// Evaluate log probabilities without first rounding the complement to one.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    auto p = Fp!128(1);
    p.exponent -= 2000;
    auto expected = p;
    expected.sign = true;
    assert(bernoulliLPMF(false,p) == expected);
    assert(bernoulliLPMF(true,p).sign);
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

// BEGIN bernoulli probability log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!64.fromHexString("1")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf79ac")),T(true,long.min,UInt!64.fromHexString("1"))),
            Reference(T(false,-2000,UInt!64.fromHexString("1")),T(true,-53,UInt!64.fromHexString("ad496b6802e49cd2")),T(true,-2063,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-385,UInt!64.fromHexString("1")),T(true,-55,UInt!64.fromHexString("856e4b05d94482fe")),T(true,-448,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-384,UInt!64.fromHexString("1")),T(true,-55,UInt!64.fromHexString("851591f9dd5b9b41")),T(true,-447,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-64,UInt!64.fromHexString("1")),T(true,-58,UInt!64.fromHexString("b17217f7d1cf79ac")),T(true,-127,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-1,UInt!64.fromHexString("1")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ac")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,long.min,UInt!64.fromHexString("3")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf79aa")),T(true,long.min,UInt!64.fromHexString("3"))),
            Reference(T(false,-2,UInt!64.fromHexString("3")),T(true,-65,UInt!64.fromHexString("934b1089a6dc93c2")),T(true,-63,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,-195,UInt!64.fromHexString("155d3680a0b3f28b")),T(true,-57,UInt!64.fromHexString("ba924d3daa964b98")),T(true,-198,UInt!64.fromHexString("aae9b405059f9458"))),
            Reference(T(false,-277,UInt!64.fromHexString("2dfcf807a025849a")),T(true,-56,UInt!64.fromHexString("955b6db83dc9254e")),T(true,-279,UInt!64.fromHexString("b7f3e01e80961268"))),
            Reference(T(false,-349,UInt!64.fromHexString("47da373f7dfb4d2f")),T(true,-56,UInt!64.fromHexString("c6d14852b454feba")),T(true,-350,UInt!64.fromHexString("8fb46e7efbf69a5e"))),
            Reference(T(false,-111,UInt!64.fromHexString("6ce38d733ed6ff1b")),T(true,-58,UInt!64.fromHexString("85bb2748be588940")),T(true,-112,UInt!64.fromHexString("d9c71ae67dae5ad8"))),
            Reference(T(false,-97,UInt!64.fromHexString("739145488dc18dc8")),T(true,-59,UInt!64.fromHexString("bd5a7b03c7b4674d")),T(true,-98,UInt!64.fromHexString("e7228a91359906e7"))),
            Reference(T(false,-64,UInt!64.fromHexString("7fffffffffffffff")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ae")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(T(false,long.min,UInt!64.fromHexString("8000000000000000")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf7954")),T(true,long.min,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-64,UInt!64.fromHexString("8000000000000001")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79aa")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ae"))),
            Reference(T(false,-245,UInt!64.fromHexString("c3590e78d844c8c3")),T(true,-57,UInt!64.fromHexString("fb75c72b86861549")),T(true,-245,UInt!64.fromHexString("c3590e78d844c8c3"))),
            Reference(T(false,-182,UInt!64.fromHexString("c4283a1bd25a8240")),T(true,-57,UInt!64.fromHexString("a41d81821f3b6956")),T(true,-182,UInt!64.fromHexString("c4283a1bd25a8240"))),
            Reference(T(false,-203,UInt!64.fromHexString("dbc71f0e08eae614")),T(true,-57,UInt!64.fromHexString("c100026a481f7aa7")),T(true,-203,UInt!64.fromHexString("dbc71f0e08eae614"))),
            Reference(T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf7953")),T(true,long.min,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-385,UInt!64.fromHexString("ffffffffffffffff")),T(true,-56,UInt!64.fromHexString("de80100dbe152790")),T(true,-385,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-384,UInt!64.fromHexString("ffffffffffffffff")),T(true,-56,UInt!64.fromHexString("ddce9df5c6435817")),T(true,-384,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,-127,UInt!64.fromHexString("8000000000000000")),T(true,-58,UInt!64.fromHexString("b17217f7d1cf79ac"))),
        ];
        foreach (r; references) {
            assert(bernoulliLPMF(true,r.p) == r.logP);
            assert(bernoulliLPMF(false,r.p) == r.logQ);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!128.fromHexString("1")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(true,long.min,UInt!128.fromHexString("1"))),
            Reference(T(false,-2000,UInt!128.fromHexString("1")),T(true,-117,UInt!128.fromHexString("ad496b6802e49cd1c3285d6273db44e7")),T(true,-2127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-513,UInt!128.fromHexString("1")),T(true,-119,UInt!128.fromHexString("b1cad103cdb861689fc8a571cff4f02b")),T(true,-640,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-512,UInt!128.fromHexString("1")),T(true,-119,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(true,-639,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-128,UInt!128.fromHexString("1")),T(true,-121,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(true,-255,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-1,UInt!128.fromHexString("1")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,long.min,UInt!128.fromHexString("3")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf79a9976664426391a3c2")),T(true,long.min,UInt!128.fromHexString("3"))),
            Reference(T(false,-2,UInt!128.fromHexString("3")),T(true,-129,UInt!128.fromHexString("934b1089a6dc93c1df5bb3b60554e152")),T(true,-127,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-382,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff")),T(true,-120,UInt!128.fromHexString("b64bbf32e5853dc6a1241a3a00126877")),T(true,-390,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff00"))),
            Reference(T(false,-191,UInt!128.fromHexString("a80121e07c0fd2678904cf45e2c018")),T(true,-122,UInt!128.fromHexString("c689de4538e16ebd0e27b39d9833320e")),T(true,-199,UInt!128.fromHexString("a80121e07c0fd267897310c25aab2961"))),
            Reference(T(false,-360,UInt!128.fromHexString("64f2572158cb0b0e40975b1d17ec17bd")),T(true,-120,UInt!128.fromHexString("a1bda0903fb3ed35ce48f971d1474472")),T(true,-361,UInt!128.fromHexString("c9e4ae42b196161c812eb63a2fd82f7a"))),
            Reference(T(false,-128,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6b1")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6ad"))),
            Reference(T(false,long.min,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fbbaafebb5d8183e3d")),T(true,long.min,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-128,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6ad")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6b1"))),
            Reference(T(false,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb")),T(true,-121,UInt!128.fromHexString("e8c44510834cce7c54fe08dccac9d95f")),T(true,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb"))),
            Reference(T(false,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24")),T(true,-120,UInt!128.fromHexString("a2c394eebd064edee58f98f5a40c10f3")),T(true,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24"))),
            Reference(T(false,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b0e9db0254")),T(true,-121,UInt!128.fromHexString("849dceb8353d6aebee7b02b2406be60b")),T(true,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b150084f92"))),
            Reference(T(false,-201,UInt!128.fromHexString("d4daf2ca7a55cf775dd9cd750c9be8c2")),T(true,-122,UInt!128.fromHexString("cb2321f2965d0451d33dac9c0907e374")),T(true,-201,UInt!128.fromHexString("d4daf2ca7a55cf775e060c4c650c573c"))),
            Reference(T(false,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc04caadc")),T(true,-121,UInt!128.fromHexString("91ecd8d283c2b13d18884a2d32bd0c4f")),T(true,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc07920d2"))),
            Reference(T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5")),T(true,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-119,UInt!128.fromHexString("856e4b05d94482fdad4fb88bcef8327f")),T(true,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-119,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903")),T(true,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-255,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-121,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
        ];
        foreach (r; references) {
            assert(bernoulliLPMF(true,r.p) == r.logP);
            assert(bernoulliLPMF(false,r.p) == r.logQ);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!256.fromHexString("1")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(true,long.min,UInt!256.fromHexString("1"))),
            Reference(T(false,-2000,UInt!256.fromHexString("1")),T(true,-245,UInt!256.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84f")),T(true,-2255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-769,UInt!256.fromHexString("1")),T(true,-246,UInt!256.fromHexString("8541ee7fdb500f1f425d3f9ee8f735c11c86af2d9f8f2ecfb2ec54ca7fa3265f")),T(true,-1024,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-768,UInt!256.fromHexString("1")),T(true,-246,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1")),T(true,-1023,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-256,UInt!256.fromHexString("1")),T(true,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(true,-511,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-1,UInt!256.fromHexString("1")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,long.min,UInt!256.fromHexString("3")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf79a9976664426391a3c1f8c0287c6821bcc20de491a81be9de4b")),T(true,long.min,UInt!256.fromHexString("3"))),
            Reference(T(false,-2,UInt!256.fromHexString("3")),T(true,-257,UInt!256.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d96")),T(true,-255,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-503,UInt!256.fromHexString("1e9b726592d30ba658d702c4cd40a6d64c302ae640625db7771e120788e25f79")),T(true,-248,UInt!256.fromHexString("ad54cfc4ceb558d41b83731273e0d5b69e1baeab78d4c672c565f798f4bc71bb")),T(true,-506,UInt!256.fromHexString("f4db932c96985d32c6b816266a0536b2618157320312edbbb8f0903c4712fbe5"))),
            Reference(T(false,-500,UInt!256.fromHexString("3ec4002d4882689ddd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a0739")),T(true,-248,UInt!256.fromHexString("aa889f60a17a6ac22409e5390478487377a9195c206d0f262c1ffd000e2260ce")),T(true,-502,UInt!256.fromHexString("fb1000b52209a2777504248f67cf06b7f762c7cdff5886c94eb476365a281ed0"))),
            Reference(T(false,-310,UInt!256.fromHexString("4720f4a871f2ec93ca5547d5408492cb24034b668fda7a54a82657222d4b3a8e")),T(true,-250,UInt!256.fromHexString("9ad7b0641c6f5d67d755b34e15da51116dc2e7ac2d75e04ba389c62c9517790d")),T(true,-311,UInt!256.fromHexString("8e41e950e3e5d976a1da5db70c1967f75b26a1c198a4aac4ce591fb21d3f54b4"))),
            Reference(T(false,-423,UInt!256.fromHexString("4d11226bb2db7011471dca27dde7b62d311503cec0f7abd7f1bd17a3c338aca7")),T(true,-249,UInt!256.fromHexString("e9e9839efca15d57a7d00a402ee7e8cff36f3691f107f7eb3c992d1002903be1")),T(true,-424,UInt!256.fromHexString("9a2244d765b6e0228e3b944fbbcf6c5a622a079d821dbe4f983b3c5e4793d45c"))),
            Reference(T(false,-430,UInt!256.fromHexString("7612335515ecf948ebdb260951f5a219d41440275124cbcfcbdd58ef250485fa")),T(true,-249,UInt!256.fromHexString("f2c353f4ddfc69c00ea9259be92cc6e46195964656615a6647d91df1a9880724")),T(true,-431,UInt!256.fromHexString("ec2466aa2bd9f291d7b64c12a3eb4433a828804ea24a7172ba317bc082a21cf6"))),
            Reference(T(false,-256,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2e")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2a"))),
            Reference(T(false,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf784a4897f3e4089e92730d537a4e9491450a256d50c33f700b73")),T(true,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2a")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2e"))),
            Reference(T(false,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de07984ea35c2a2dc0399d61d2294e0d")),T(true,-249,UInt!256.fromHexString("cfc182533e5dac574b7d8fee956fa9ce8b87699f0afb6e22cfa5e91111fc105e")),T(true,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de079983aae9d0063609b2a9f0ed2601"))),
            Reference(T(false,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa08801907f3462d0e84f25bca22b80357ef032e8edb2d5a5")),T(true,-250,UInt!256.fromHexString("e4e42a2e3d8a4bda61b942d2db67f21a4b7363237f2c65e966c5cde7a8f7c141")),T(true,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa088106368d15ba936381f65cba0da3a96f5267f2515c8ae"))),
            Reference(T(false,-294,UInt!256.fromHexString("eee66cfb0ccb08a554bda8b2c9fdfd2bcc24375ff6aefe360e542751a3681ab8")),T(true,-251,UInt!256.fromHexString("d345118635839daa37a4f317628449bcc9c2a201fa08dd5642cdc23397deaa14")),T(true,-294,UInt!256.fromHexString("eee66cfb0e88eb296a5c90e10dae57ee4e21e53fc6eaa4427494b698b0a02a3d"))),
            Reference(T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18")),T(true,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-247,UInt!256.fromHexString("b1cad103cdb861689fc8a571cff4f02a9893bcc805d20288a0d21de73970cfa9")),T(true,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-247,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(true,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-511,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
        ];
        foreach (r; references) {
            assert(bernoulliLPMF(true,r.p) == r.logP);
            assert(bernoulliLPMF(false,r.p) == r.logQ);
        }
    }
}
// END bernoulli probability log references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    foreach (T; AliasSeq!(Fp!64,Fp!128,Fp!256,Fp!512)) {
        const zero = T(0);
        const one = T(1);
        const infinity = T(-double.infinity);
        assert(bernoulliLPMF(false,zero) == zero);
        assert(bernoulliLPMF(true,one) == zero);
        assert(bernoulliLPMF(true,zero) == infinity);
        assert(bernoulliLPMF(false,one) == infinity);
        // Exact-one endpoints must also accept an unnormalized coefficient.
        const rawOne = T(false,0,UInt!(typeof(T.init.coefficient).sizeof * 8)(1));
        assert(bernoulliLPMF(true,rawOne) == zero);
        assert(bernoulliLPMF(false,rawOne) == infinity);
        T negativeZero = zero;
        negativeZero.sign = true;
        assert(bernoulliLPMF(false,negativeZero) == zero);
        auto p = T(0.25);
        const original = p;
        p.coefficient >>= 1;
        p.exponent += 1;
        assert(bernoulliLPMF(true,p) == bernoulliLPMF(true,original));
        assert(bernoulliLPMF(false,p) == bernoulliLPMF(false,original));
        static assert(is(typeof(bernoulliLPMF(true,one)) == T));
        static assert(!__traits(compiles,bernoulliLPMF(true,1)));
        debug {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            foreach (bad; [T(-0.5),T(1.5),T(double.infinity),T(double.nan)]) {
                assertThrown!AssertError(bernoulliLPMF(true,bad));
            }
        }
    }
}
