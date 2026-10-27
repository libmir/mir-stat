/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution).

The outcomes are the integers from $(D lower) through $(D upper), inclusive,
with equal probability. Bounds and observations use $(D size_t); negative
outcomes are not represented by this API. Forward functions require an
observation inside the interval rather than extending the tails outside it.

PMF, CDF, and CCDF accept $(D lower == upper). LPMF and inverse CDF instead
require $(D lower < upper). Forward functions return $(D double) by default.
PMF, CDF, and CCDF additionally accept an explicit $(D Fp) result type, such as
$(D uniformDiscreteCDF!(Fp!128)(x, lower, upper)). The inverse CDF accepts a
built-in floating-point probability and returns $(D size_t).

In the tables, $(D n = upper - lower + 1) denotes the mathematical number
of outcomes. The built-in implementation adds one after conversion to floating point
to avoid integer overflow for the full $(D size_t) domain. Large interval
sizes and interior quantile ranks can still round in the floating-point type.
The $(D Fp) overloads retain exact integer counts, including the full
$(D size_t) domain, and round the exact ratio once to nearest, ties to even.
Tested result widths are $(D Fp!64), $(D Fp!128), and $(D Fp!256). Each tail
is evaluated directly, so a small CCDF is not lost by subtracting a rounded CDF
from one. Work depends on the result width rather than the interval length.

Finite expressions are subject to floating-point rounding. Contract violations
are not prescribed return values: disabling contracts does not make an invalid
input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.uniform_discrete;

import mir.internal.utility: isFloatingPoint;
import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;

/++
Computes the discrete uniform probability mass function (PMF).

Returns $(D P(X == x) = 1 / n) for every outcome in the interval.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D lower <= x && x <= upper)) $(TD $(D 1 / n)))
    $(TR $(TD $(D lower == upper) and $(D x == lower)) $(TD $(D 1)))
    $(TR $(TD $(D x < lower) or $(D x > upper)) $(TD Contract violation))
)

Params:
    x = value to evaluate PMF
    lower = inclusive lower bound
    upper = inclusive upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution)
+/
@safe pure nothrow @nogc
double uniformDiscretePMF(const size_t x, const size_t lower = 0, const size_t upper = 1)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    return 1.0 / (cast(double)(upper - lower) + 1);
}

/// ditto
@safe pure nothrow @nogc
T uniformDiscretePMF(T)(const size_t x, const size_t lower = 0, const size_t upper = 1)
    if (is(T == Fp!bits, uint bits))
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    auto denominator = UInt!128(upper - lower);
    denominator += 1;
    auto numerator = UInt!128(1);
    return uniformDiscreteRatio!T(numerator, denominator);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.uniformDiscretePMF.shouldApprox == 0.5;
    2.uniformDiscretePMF(1, 3).shouldApprox == 1.0 / 3;
}

/++
Computes the discrete uniform cumulative distribution function (CDF).

Returns the inclusive lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D 1 / n)))
    $(TR $(TD $(D x == upper)) $(TD $(D 1)))
    $(TR $(TD $(D x < lower) or $(D x > upper)) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    lower = inclusive lower bound
    upper = inclusive upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution)
+/
@safe pure nothrow @nogc
double uniformDiscreteCDF(const size_t x, const size_t lower = 0, const size_t upper = 1)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    return (cast(double)(x - lower) + 1) / (cast(double)(upper - lower) + 1);
}

/// ditto
@safe pure nothrow @nogc
T uniformDiscreteCDF(T)(const size_t x, const size_t lower = 0, const size_t upper = 1)
    if (is(T == Fp!bits, uint bits))
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    auto denominator = UInt!128(upper - lower);
    denominator += 1;
    auto numerator = UInt!128(x - lower);
    numerator += 1;
    return uniformDiscreteRatio!T(numerator, denominator);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.uniformDiscreteCDF.shouldApprox == 0.5;
    1.uniformDiscreteCDF.shouldApprox == 1.0;

    1.uniformDiscreteCDF(1, 3).shouldApprox == 1.0 / 3;
    2.uniformDiscreteCDF(1, 3).shouldApprox == 2.0 / 3;
    3.uniformDiscreteCDF(1, 3).shouldApprox == 1.0;
}

/++
Computes the discrete uniform complementary cumulative distribution function (CCDF).

Returns the strict upper-tail probability $(D P(X > x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D (n - 1) / n)))
    $(TR $(TD $(D x == upper)) $(TD $(D 0)))
    $(TR $(TD $(D x < lower) or $(D x > upper)) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    lower = inclusive lower bound
    upper = inclusive upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution)
+/
@safe pure nothrow @nogc
double uniformDiscreteCCDF(const size_t x, const size_t lower = 0, const size_t upper = 1)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    return cast(double)(upper - x) / (cast(double)(upper - lower) + 1);
}

/// ditto
@safe pure nothrow @nogc
T uniformDiscreteCCDF(T)(const size_t x, const size_t lower = 0, const size_t upper = 1)
    if (is(T == Fp!bits, uint bits))
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower <= upper, "lower must be less than or equal to upper")
{
    auto denominator = UInt!128(upper - lower);
    denominator += 1;
    auto numerator = UInt!128(upper - x);
    return uniformDiscreteRatio!T(numerator, denominator);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.uniformDiscreteCCDF.shouldApprox == 0.5;
    1.uniformDiscreteCCDF.shouldApprox == 0.0;

    1.uniformDiscreteCCDF(1, 3).shouldApprox == 2.0 / 3;
    2.uniformDiscreteCCDF(1, 3).shouldApprox == 1.0 / 3;
    3.uniformDiscreteCCDF(1, 3).shouldApprox == 0.0;
}

/++
Computes the discrete uniform inverse cumulative distribution function (InvCDF).

Selects the first outcome whose cumulative probability reaches $(D p).
Interior ranks are computed in $(D T) and are subject to rounding.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D lower)))
    $(TR $(TD $(D p == 1)) $(TD $(D upper)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D lower >= upper)) $(TD Contract violation))
)

Params:
    p = cumulative probability in $(D [0, 1])
    lower = inclusive lower bound
    upper = inclusive upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution)
+/
@safe pure nothrow @nogc
size_t uniformDiscreteInvCDF(T)(const T p, const size_t lower = 0, const size_t upper = 1)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (lower < upper, "lower must be less than upper")
{
    import mir.math.common: ceil;

    if (p == 0)
        return lower;
    if (p == 1)
        return upper;
    // Subtract while integral, then add one in floating point so even the
    // complete size_t domain has a representable interval size.
    const n = cast(T)(upper - lower) + 1;
    const rank = ceil(p * n);
    if (rank <= 1) {
        return lower;
    }
    // Rounding may reach n before p reaches one. Clamp before the integer cast.
    if (rank >= n)
        return upper;
    return lower + (cast(size_t) rank - 1);
}

///.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    0.0.uniformDiscreteInvCDF.should == 0;
    0.5.uniformDiscreteInvCDF.should == 0;
    1.0.uniformDiscreteInvCDF.should == 1;

    0.0.uniformDiscreteInvCDF(1, 3).should == 1;
    0.2.uniformDiscreteInvCDF(1, 3).should == 1;
    (1.0 / 3).uniformDiscreteInvCDF(1, 3).should == 1;
    0.5.uniformDiscreteInvCDF(1, 3).should == 2;
    (2.0 / 3).uniformDiscreteInvCDF(1, 3).should == 2;
    1.0.uniformDiscreteInvCDF(1, 3).should == 3;
}

/// The CDF includes the selected integer; the CCDF counts only larger integers.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    // Four equally likely outcomes: 10, 11, 12, and 13.
    assert(uniformDiscretePMF(11, 10, 13) == 0.25);
    assert(uniformDiscreteCDF(11, 10, 13) == 0.5);
    assert(uniformDiscreteCCDF(11, 10, 13) == 0.5);
    assert(uniformDiscreteInvCDF(0.5, 10, 13) == 11);
}

// Interior probabilities select the first value whose CDF reaches p.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    foreach (lower; 0 .. 4)
    foreach (n; 2 .. 9)
    foreach (j; 1 .. 100)
    {
        const p = j / 100.0;
        const upper = lower + n - 1;
        const x = uniformDiscreteInvCDF(p, lower, upper);
        assert(uniformDiscreteCDF(x, lower, upper) >= p);
        if (x > lower)
            assert(uniformDiscreteCDF(x - 1, lower, upper) < p);
    }
    assert(uniformDiscreteInvCDF(0.25, 10, 13) == 10);
    assert(uniformDiscreteInvCDF(0.25 + double.epsilon, 10, 13) == 11);
    assert(uniformDiscreteInvCDF(0.25 - double.epsilon, 10, 13) == 10);
}

/++
Computes the discrete uniform log probability mass function (LPMF).

Returns the natural logarithm of the point probability $(D P(X == x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D lower <= x && x <= upper) with $(D lower < upper)) $(TD $(D -log(n))))
    $(TR $(TD $(D lower >= upper)) $(TD Contract violation))
    $(TR $(TD $(D x < lower) or $(D x > upper)) $(TD Contract violation))
)

Params:
    x = integer outcome at which to evaluate LPMF
    lower = inclusive lower bound
    upper = inclusive upper bound

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution)
+/
@safe pure nothrow @nogc
double uniformDiscreteLPMF(const size_t x, const size_t lower = 0, const size_t upper = 1)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    import mir.math.common: log;

    return -log(cast(double)(upper - lower) + 1);
}

// Translation must not lose adjacent integers, and the full integer domain
// must not overflow its interval size.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, exp;
    const lower = size_t.max - 2;
    const upper = size_t.max;
    foreach (i; 0 .. 3)
    {
        const x = lower + i;
        assert(approxEqual(uniformDiscretePMF(x, lower, upper), 1.0 / 3));
        assert(approxEqual(uniformDiscreteCDF(x, lower, upper), (i + 1.0) / 3));
        assert(approxEqual(uniformDiscreteCCDF(x, lower, upper), (2.0 - i) / 3));
        assert(approxEqual(exp(uniformDiscreteLPMF(x, lower, upper)), 1.0 / 3));
    }
    assert(uniformDiscreteInvCDF(0.5, lower, upper) == lower + 1);
    assert(uniformDiscreteInvCDF(0.0, 0, upper) == 0);
    assert(uniformDiscreteInvCDF(1.0, 0, upper) == upper);
    assert(uniformDiscreteInvCDF(0.5, 0, upper) == upper / 2);
    const probability = 1.0 / (cast(double) upper + 1);
    assert(uniformDiscretePMF(0, 0, upper) == probability);
    assert(uniformDiscreteCDF(0, 0, upper) == probability);
    assert(uniformDiscreteCCDF(upper - 1, 0, upper) == probability);
    assert(uniformDiscreteCDF(upper, 0, upper) == 1);
    assert(uniformDiscreteCCDF(upper, 0, upper) == 0);
    assert(approxEqual(exp(uniformDiscreteLPMF(0, 0, upper)), probability));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    1.uniformDiscreteLPMF.shouldApprox == -log(2.0);
    2.uniformDiscreteLPMF(1, 3).shouldApprox == -log(3.0);
}

// Exact ratios with 0 <= numerator <= denominator <= 2^64. Normalize before
// division so the quotient has exactly the requested significand width. The
// remainder decides rounding without a guard-bit approximation or double rounding.
private @safe pure nothrow @nogc
T uniformDiscreteRatio(T)(UInt!128 numerator, UInt!128 denominator)
    if (is(T == Fp!bits, uint bits))
{
    if (!numerator)
        return T(0);
    if (numerator == denominator)
        return T(1);
    int exponent = cast(int)denominator.ctlz - cast(int)numerator.ctlz;
    if ((numerator << -exponent) < denominator)
        --exponent;
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    const shift = bits - 1 - exponent;
    auto quotient = numerator.toSize!(bits + 128);
    quotient <<= shift;
    auto remainder = denominator;
    quotient.divMod(remainder);
    remainder <<= 1;
    if (remainder > denominator || (remainder == denominator && quotient.bt(0)))
        quotient += 1;
    auto result = T(quotient);
    result.exponent -= shift;
    return result;
}

/// Select Fp precision explicitly; ordinary calls continue to return double.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    assert(uniformDiscretePMF!(Fp!128)(11, 10, 13) == Fp!128(0.25));
    assert(uniformDiscreteCDF!(Fp!128)(11, 10, 13) == Fp!128(0.5));
    assert(uniformDiscreteCCDF!(Fp!128)(11, 10, 13) == Fp!128(0.5));
    static assert(is(typeof(uniformDiscretePMF(0)) == double));
    static assert(is(typeof(uniformDiscreteCDF(0)) == double));
    static assert(is(typeof(uniformDiscreteCCDF(0)) == double));
    static assert(!__traits(compiles, uniformDiscretePMF!int(0)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        static assert(is(typeof(uniformDiscretePMF!T(0)) == T));
        static assert(is(typeof(uniformDiscreteCDF!T(0)) == T));
        static assert(is(typeof(uniformDiscreteCCDF!T(0)) == T));
        assert(uniformDiscretePMF!T(0) == T(0.5));
        assert(uniformDiscreteCDF!T(0) == T(0.5));
        assert(uniformDiscreteCCDF!T(0) == T(0.5));
        foreach (x; [size_t(0), size_t.max]) {
            assert(uniformDiscretePMF!T(x, x, x) == T(1));
            assert(uniformDiscreteCDF!T(x, x, x) == T(1));
            assert(uniformDiscreteCCDF!T(x, x, x) == T(0));
        }
        auto mass = T(1);
        mass.exponent -= size_t.sizeof * 8;
        assert(uniformDiscretePMF!T(0, 0, size_t.max) == mass);
        assert(uniformDiscreteCDF!T(0, 0, size_t.max) == mass);
        assert(uniformDiscreteCCDF!T(size_t.max - 1, 0, size_t.max) == mass);
        foreach (i; 0 .. 3) {
            assert(uniformDiscreteCDF!T(size_t.max - 2 + i, size_t.max - 2, size_t.max)
                == uniformDiscreteCDF!T(i, 0, 2));
            assert(uniformDiscreteCCDF!T(size_t.max - 2 + i, size_t.max - 2, size_t.max)
                == uniformDiscreteCCDF!T(i, 0, 2));
        }
    }}
}

version(mir_stat_test)
@system unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        import std.meta: AliasSeq;
        static foreach (fn; AliasSeq!(uniformDiscretePMF!(Fp!128),
            uniformDiscreteCDF!(Fp!128), uniformDiscreteCCDF!(Fp!128))) {
            assertThrown!AssertError(fn(0, 1, 2));
            assertThrown!AssertError(fn(3, 1, 2));
            assertThrown!AssertError(fn(1, 2, 0));
        }
    }
}

// BEGIN discrete-uniform references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    {
        alias T = Fp!64;
        struct Reference { ulong x, lower, upper; T pmf, cdf, ccdf; }
        enum references = [
            Reference(0UL, 0UL, 0UL, T(false, -63, UInt!64.fromHexString("8000000000000000")), T(false, -63, UInt!64.fromHexString("8000000000000000")), T(0)),
            Reference(0UL, 0UL, 1UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -64, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, 0UL, 2UL, T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -64, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab"))),
            Reference(0UL, 0UL, 2UL, T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -64, UInt!64.fromHexString("aaaaaaaaaaaaaaab"))),
            Reference(2UL, 0UL, 2UL, T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -63, UInt!64.fromHexString("8000000000000000")), T(0)),
            Reference(4UL, 0UL, 6UL, T(false, -66, UInt!64.fromHexString("9249249249249249")), T(false, -64, UInt!64.fromHexString("b6db6db6db6db6db")), T(false, -65, UInt!64.fromHexString("9249249249249249"))),
            Reference(7UL, 0UL, 9UL, T(false, -67, UInt!64.fromHexString("cccccccccccccccd")), T(false, -64, UInt!64.fromHexString("cccccccccccccccd")), T(false, -66, UInt!64.fromHexString("cccccccccccccccd"))),
            Reference(13UL, 10UL, 26UL, T(false, -68, UInt!64.fromHexString("f0f0f0f0f0f0f0f1")), T(false, -66, UInt!64.fromHexString("f0f0f0f0f0f0f0f1")), T(false, -64, UInt!64.fromHexString("c3c3c3c3c3c3c3c4"))),
            Reference(0UL, 0UL, 9007199254740991UL, T(false, -116, UInt!64.fromHexString("8000000000000000")), T(false, -116, UInt!64.fromHexString("8000000000000000")), T(false, -64, UInt!64.fromHexString("fffffffffffff800"))),
            Reference(0UL, 0UL, 9007199254740992UL, T(false, -117, UInt!64.fromHexString("fffffffffffff800")), T(false, -117, UInt!64.fromHexString("fffffffffffff800")), T(false, -64, UInt!64.fromHexString("fffffffffffff800"))),
            Reference(9007199254740991UL, 0UL, 9007199254740992UL, T(false, -117, UInt!64.fromHexString("fffffffffffff800")), T(false, -64, UInt!64.fromHexString("fffffffffffff800")), T(false, -117, UInt!64.fromHexString("fffffffffffff800"))),
            Reference(0UL, 0UL, 18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -64, UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(18446744073709551614UL, 0UL, 18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -127, UInt!64.fromHexString("8000000000000000"))),
            Reference(18446744073709551615UL, 0UL, 18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -63, UInt!64.fromHexString("8000000000000000")), T(0)),
            Reference(18446744073709551614UL, 18446744073709551613UL, 18446744073709551615UL, T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -64, UInt!64.fromHexString("aaaaaaaaaaaaaaab")), T(false, -65, UInt!64.fromHexString("aaaaaaaaaaaaaaab"))),
            Reference(9223372036854775808UL, 0UL, 18446744073709551614UL, T(false, -127, UInt!64.fromHexString("8000000000000001")), T(false, -64, UInt!64.fromHexString("8000000000000002")), T(false, -65, UInt!64.fromHexString("fffffffffffffffd"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const x = cast(size_t)c.x;
            const lower = cast(size_t)c.lower;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscretePMF!T(x, lower, upper) == c.pmf);
            assert(uniformDiscreteCDF!T(x, lower, upper) == c.cdf);
            assert(uniformDiscreteCCDF!T(x, lower, upper) == c.ccdf);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong x, lower, upper; T pmf, cdf, ccdf; }
        enum references = [
            Reference(0UL, 0UL, 0UL, T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(0)),
            Reference(0UL, 0UL, 1UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, 0UL, 2UL, T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -128, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(0UL, 0UL, 2UL, T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -128, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(2UL, 0UL, 2UL, T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(0)),
            Reference(4UL, 0UL, 6UL, T(false, -130, UInt!128.fromHexString("92492492492492492492492492492492")), T(false, -128, UInt!128.fromHexString("b6db6db6db6db6db6db6db6db6db6db7")), T(false, -129, UInt!128.fromHexString("92492492492492492492492492492492"))),
            Reference(7UL, 0UL, 9UL, T(false, -131, UInt!128.fromHexString("cccccccccccccccccccccccccccccccd")), T(false, -128, UInt!128.fromHexString("cccccccccccccccccccccccccccccccd")), T(false, -130, UInt!128.fromHexString("cccccccccccccccccccccccccccccccd"))),
            Reference(13UL, 10UL, 26UL, T(false, -132, UInt!128.fromHexString("f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f1")), T(false, -130, UInt!128.fromHexString("f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f1")), T(false, -128, UInt!128.fromHexString("c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c4"))),
            Reference(0UL, 0UL, 9007199254740991UL, T(false, -180, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -180, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128, UInt!128.fromHexString("fffffffffffff8000000000000000000"))),
            Reference(0UL, 0UL, 9007199254740992UL, T(false, -181, UInt!128.fromHexString("fffffffffffff8000000000000400000")), T(false, -181, UInt!128.fromHexString("fffffffffffff8000000000000400000")), T(false, -128, UInt!128.fromHexString("fffffffffffff8000000000000400000"))),
            Reference(9007199254740991UL, 0UL, 9007199254740992UL, T(false, -181, UInt!128.fromHexString("fffffffffffff8000000000000400000")), T(false, -128, UInt!128.fromHexString("fffffffffffff8000000000000400000")), T(false, -181, UInt!128.fromHexString("fffffffffffff8000000000000400000"))),
            Reference(0UL, 0UL, 18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128, UInt!128.fromHexString("ffffffffffffffff0000000000000000"))),
            Reference(18446744073709551614UL, 0UL, 18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128, UInt!128.fromHexString("ffffffffffffffff0000000000000000")), T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(18446744073709551615UL, 0UL, 18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -127, UInt!128.fromHexString("80000000000000000000000000000000")), T(0)),
            Reference(18446744073709551614UL, 18446744073709551613UL, 18446744073709551615UL, T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -128, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -129, UInt!128.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(9223372036854775808UL, 0UL, 18446744073709551614UL, T(false, -191, UInt!128.fromHexString("80000000000000008000000000000001")), T(false, -128, UInt!128.fromHexString("80000000000000018000000000000002")), T(false, -129, UInt!128.fromHexString("fffffffffffffffcfffffffffffffffd"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const x = cast(size_t)c.x;
            const lower = cast(size_t)c.lower;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscretePMF!T(x, lower, upper) == c.pmf);
            assert(uniformDiscreteCDF!T(x, lower, upper) == c.cdf);
            assert(uniformDiscreteCCDF!T(x, lower, upper) == c.ccdf);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong x, lower, upper; T pmf, cdf, ccdf; }
        enum references = [
            Reference(0UL, 0UL, 0UL, T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(0)),
            Reference(0UL, 0UL, 1UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, 0UL, 2UL, T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -256, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(0UL, 0UL, 2UL, T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -256, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(2UL, 0UL, 2UL, T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(0)),
            Reference(4UL, 0UL, 6UL, T(false, -258, UInt!256.fromHexString("9249249249249249249249249249249249249249249249249249249249249249")), T(false, -256, UInt!256.fromHexString("b6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db6db")), T(false, -257, UInt!256.fromHexString("9249249249249249249249249249249249249249249249249249249249249249"))),
            Reference(7UL, 0UL, 9UL, T(false, -259, UInt!256.fromHexString("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccd")), T(false, -256, UInt!256.fromHexString("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccd")), T(false, -258, UInt!256.fromHexString("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccd"))),
            Reference(13UL, 10UL, 26UL, T(false, -260, UInt!256.fromHexString("f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f1")), T(false, -258, UInt!256.fromHexString("f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f1")), T(false, -256, UInt!256.fromHexString("c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c4"))),
            Reference(0UL, 0UL, 9007199254740991UL, T(false, -308, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -308, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256, UInt!256.fromHexString("fffffffffffff800000000000000000000000000000000000000000000000000"))),
            Reference(0UL, 0UL, 9007199254740992UL, T(false, -309, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000")), T(false, -309, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000")), T(false, -256, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000"))),
            Reference(9007199254740991UL, 0UL, 9007199254740992UL, T(false, -309, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000")), T(false, -256, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000")), T(false, -309, UInt!256.fromHexString("fffffffffffff80000000000003ffffffffffffe000000000000100000000000"))),
            Reference(0UL, 0UL, 18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256, UInt!256.fromHexString("ffffffffffffffff000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551614UL, 0UL, 18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256, UInt!256.fromHexString("ffffffffffffffff000000000000000000000000000000000000000000000000")), T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL, 0UL, 18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -255, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(0)),
            Reference(18446744073709551614UL, 18446744073709551613UL, 18446744073709551615UL, T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -256, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab")), T(false, -257, UInt!256.fromHexString("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"))),
            Reference(9223372036854775808UL, 0UL, 18446744073709551614UL, T(false, -319, UInt!256.fromHexString("8000000000000000800000000000000080000000000000008000000000000001")), T(false, -256, UInt!256.fromHexString("8000000000000001800000000000000180000000000000018000000000000002")), T(false, -257, UInt!256.fromHexString("fffffffffffffffcfffffffffffffffcfffffffffffffffcfffffffffffffffd"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const x = cast(size_t)c.x;
            const lower = cast(size_t)c.lower;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscretePMF!T(x, lower, upper) == c.pmf);
            assert(uniformDiscreteCDF!T(x, lower, upper) == c.cdf);
            assert(uniformDiscreteCCDF!T(x, lower, upper) == c.ccdf);
        }
    }
}
// END discrete-uniform references
