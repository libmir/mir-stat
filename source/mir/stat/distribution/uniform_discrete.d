/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution).

The outcomes are the integers from $(D lower) through $(D upper), inclusive,
with equal probability. Bounds and observations use $(D size_t); negative
outcomes are not represented by this API. Forward functions require an
observation inside the interval rather than extending the tails outside it.

PMF, CDF, and CCDF accept $(D lower == upper). LPMF and inverse CDF instead
require $(D lower < upper). Forward functions return $(D double) by default.
PMF, CDF, CCDF, and LPMF additionally accept an explicit $(D Fp) result type, such as
$(D uniformDiscreteCDF!(Fp!128)(x, lower, upper)). The inverse CDF accepts a
built-in floating-point or $(D Fp) probability and returns $(D size_t).
For $(D Fp), its integer rank is exact for the represented probability.

In the tables, $(D n = upper - lower + 1) denotes the mathematical number
of outcomes. The built-in implementation adds one after conversion to floating point
to avoid integer overflow for the full $(D size_t) domain. Large interval
sizes and interior quantile ranks can still round in the floating-point type.
The $(D Fp) PMF, CDF, and CCDF overloads retain exact integer counts, including the full
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
import mir.stat.internal.fp_probability: fpProbabilityValid;

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

/++
Selects the first outcome whose exact CDF reaches the represented $(D Fp)
probability. Multiply the significand by the exact interval size and determine
the integer rank without rounding through a floating-point type. Endpoint and
bound contracts match the built-in overload.
+/
@safe pure nothrow @nogc
size_t uniformDiscreteInvCDF(T)(const T p, const size_t lower = 0, const size_t upper = 1)
    if (is(T == Fp!bits, uint bits))
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
    in (lower < upper, "lower must be less than upper")
{
    if (!p.coefficient)
        return lower;
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    auto n = UInt!128(upper - lower);
    n += 1;
    auto product = p.coefficient.toSize!(bits + 128);
    product *= n;
    // For positive product, ceil(product / 2^shift)-1 equals
    // floor((product-1) / 2^shift), including exact CDF jumps.
    product -= 1;
    const shift = 0UL - cast(ulong)p.exponent;
    if (shift >= bits + 128)
        return lower;
    product >>= cast(size_t)shift;
    return lower + cast(size_t)product;
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

The explicit $(D Fp) result-type overload computes $(D -log(n)) directly
from the exact integer interval size, including the full $(D size_t) domain.
It uses a range-reduced logarithm series with 192 extra fractional bits and
wide-integer products and division. A certified 512-bit integer constant
provides log(2) when it covers the working precision; wider types compute
it with the same series. No intermediate uses hardware floating
point or the rounded PMF. This provides extended significand precision,
not an additional range benefit: valid results lie between $(D -log(2^64))
and $(D -log(2)). The guarded algorithm does not promise correct rounding
for every interval size. Tested widths are $(D Fp!64), $(D Fp!128),
$(D Fp!256), and $(D Fp!512). Work depends on precision rather than the number of outcomes.

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

/// ditto
@safe pure nothrow @nogc
T uniformDiscreteLPMF(T)(const size_t x, const size_t lower = 0, const size_t upper = 1)
    if (is(T == Fp!bits, uint bits))
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    auto n = UInt!128(upper - lower);
    n += 1;
    auto result = uniformDiscreteLogCount!T(n);
    result.sign = true;
    return result;
}

// log(n) = e*log(2) + 2*atanh((n-2^e)/(n+2^e)), for 2 <= n <= 2^64.
// Both atanh arguments are at most 1/3. Fixed-point truncation at each
// product/division loses only guard bits; no Fp division or subtraction is used.
private @safe pure nothrow @nogc
T uniformDiscreteLogCount(T)(UInt!128 n)
    if (is(T == Fp!bits, uint bits))
{
    enum fractionBits = typeof(T.init.coefficient).sizeof * 8 + 192;
    alias I = UInt!(fractionBits + 64);
    I multiply(I a, I b) {
        auto product = a.toSize!(2 * fractionBits + 128);
        product *= b;
        return (product >> fractionBits).toSize!(fractionBits + 64);
    }
    I logRatio(UInt!128 numerator, UInt!128 denominator) {
        auto z = numerator.toSize!(fractionBits + 64);
        z <<= fractionBits;
        z /= denominator;
        const square = multiply(z,z);
        auto power = z;
        auto sum = z;
        for (uint odd = 3; ; odd += 2) {
            power = multiply(power,square);
            auto term = power;
            term /= UInt!64(odd);
            if (!term)
                break;
            sum += term;
        }
        return sum << 1;
    }
    const exponent = 127 - n.ctlz;
    const base = UInt!128(1) << exponent;
    static if (fractionBits <= 512) {
        // floor(2^512*log(2)); certified by exact rational series bounds in
        // tools/generate_fp_uniform_log_reference.py (--log-two).
        enum encodedLogTwo = UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825");
        enum logTwo = (encodedLogTwo >> (512-fractionBits)).toSize!(fractionBits + 64);
    } else {
        // Preserve all guard bits for result types wider than the constant.
        const logTwo = logRatio(UInt!128(1),UInt!128(3));
    }
    I logarithm = logTwo;
    logarithm *= exponent;
    logarithm += logRatio(n-base,n+base);
    auto result = T(logarithm);
    result.exponent -= fractionBits;
    return result;
}

/// The Fp log mass is independent of the interval's position and the outcome.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    const origin = uniformDiscreteLPMF!(Fp!128)(0,0,2);
    const shifted = uniformDiscreteLPMF!(Fp!128)(size_t.max,size_t.max-2,size_t.max);
    assert(origin == shifted);
    assert(origin.sign);
    static assert(is(typeof(uniformDiscreteLPMF(0)) == double));
    static assert(is(typeof(origin) == const(Fp!128)));
    static assert(!__traits(compiles, uniformDiscreteLPMF!int(0)));
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
        assertThrown!AssertError(uniformDiscreteLPMF!(Fp!128)(0,1,2));
        assertThrown!AssertError(uniformDiscreteLPMF!(Fp!128)(3,1,2));
        assertThrown!AssertError(uniformDiscreteLPMF!(Fp!128)(1,1,1));
        assertThrown!AssertError(uniformDiscreteLPMF!(Fp!128)(1,2,1));
    }
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

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        static assert(is(typeof(uniformDiscreteInvCDF(T(0.5))) == size_t));
        foreach (lower; [size_t(0), size_t.max - 3]) {
            const upper = lower + 3;
            foreach (i; 1 .. 4) {
                auto p = T(i); p.exponent -= 2;
                assert(uniformDiscreteInvCDF(p, lower, upper) == lower + i - 1);
                auto below = p; below.coefficient -= 1;
                auto above = p; above.coefficient += 1;
                assert(uniformDiscreteInvCDF(below, lower, upper) == lower + i - 1);
                assert(uniformDiscreteInvCDF(above, lower, upper) == lower + i);
            }
            assert(uniformDiscreteInvCDF(T(0), lower, upper) == lower);
            assert(uniformDiscreteInvCDF(T(1), lower, upper) == upper);
            assert(uniformDiscreteInvCDF(T(false, long.min, UInt!bits(1)), lower, upper) == lower);
        }
        assert(uniformDiscreteInvCDF(T(0.5), 0, size_t.max) == size_t.max / 2);
        auto aboveHalf = T(0.5); aboveHalf.coefficient += 1;
        assert(uniformDiscreteInvCDF(aboveHalf, 0, size_t.max) == size_t.max / 2 + 1);
        const raw = T(false, 0, UInt!bits(1));
        assert(uniformDiscreteInvCDF(raw, 0, size_t.max) == size_t.max);
    }}
}

/// An exact CDF jump selects its lower outcome; the next Fp value advances it.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    auto p = Fp!128(0.5);
    assert(uniformDiscreteInvCDF(p,10,13) == 11);
    p.coefficient += 1;
    assert(uniformDiscreteInvCDF(p,10,13) == 12);
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
        foreach (p; [-1.0,2.0,double.infinity,double.nan])
            assertThrown!AssertError(uniformDiscreteInvCDF(Fp!128(p)));
        assertThrown!AssertError(uniformDiscreteInvCDF(Fp!128(0.5),1,1));
        assertThrown!AssertError(uniformDiscreteInvCDF(Fp!128(0.5),2,1));
    }
}

// Exact integer inequalities verify ranks for non-dyadic CDF steps. All p in
// this grid are dyadic, so (rank+1)/n >= p iff (rank+1)*256 >= j*n.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        foreach (n; 2 .. 33)
        foreach (j; 1 .. 256) {
            auto p = T(j); p.exponent -= 8;
            const x = uniformDiscreteInvCDF(p,10,10+n-1);
            const rank = x-10;
            assert((rank+1)*256 >= j*n);
            if (rank) assert(rank*256 < j*n);
        }
    }}
}

// BEGIN discrete-uniform log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    {
        alias T = Fp!64;
        struct Reference { ulong upper; T logMass; }
        enum references = [
            Reference(1UL,T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(2UL,T(true,-63,UInt!64.fromHexString("8c9f53d5681854bb"))),
            Reference(3UL,T(true,-63,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(4UL,T(true,-63,UInt!64.fromHexString("ce020fbf6c699b58"))),
            Reference(5UL,T(true,-63,UInt!64.fromHexString("e5585fd151001191"))),
            Reference(6UL,T(true,-63,UInt!64.fromHexString("f913957192d2baa3"))),
            Reference(7UL,T(true,-62,UInt!64.fromHexString("851591f9dd5b9b41"))),
            Reference(8UL,T(true,-62,UInt!64.fromHexString("8c9f53d5681854bb"))),
            Reference(9UL,T(true,-62,UInt!64.fromHexString("935d8dddaaa8ac17"))),
            Reference(10UL,T(true,-62,UInt!64.fromHexString("99771dbbf9ba5dbe"))),
            Reference(11UL,T(true,-62,UInt!64.fromHexString("9f08b5e69cf3e734"))),
            Reference(12UL,T(true,-62,UInt!64.fromHexString("a42821598c6287f4"))),
            Reference(13UL,T(true,-62,UInt!64.fromHexString("a8e650b6bddd3bbd"))),
            Reference(14UL,T(true,-62,UInt!64.fromHexString("ad50b1ca6a40f80a"))),
            Reference(15UL,T(true,-62,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(16UL,T(true,-62,UInt!64.fromHexString("b5535e0fd3fbce78"))),
            Reference(26UL,T(true,-62,UInt!64.fromHexString("d2eefdc01c247f19"))),
            Reference(609UL,T(true,-61,UInt!64.fromHexString("cd3b0e4768b4bd19"))),
            Reference(789UL,T(true,-61,UInt!64.fromHexString("d5814b3c52dfee03"))),
            Reference(1574UL,T(true,-61,UInt!64.fromHexString("eb959726cef5a6b2"))),
            Reference(65534UL,T(true,-60,UInt!64.fromHexString("b17207f7c9cf7456"))),
            Reference(65535UL,T(true,-60,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(65536UL,T(true,-60,UInt!64.fromHexString("b17227f7c9cf7f01"))),
            Reference(239088UL,T(true,-60,UInt!64.fromHexString("c627490bcb08b038"))),
            Reference(751410UL,T(true,-60,UInt!64.fromHexString("d879af264ff1d52f"))),
            Reference(2959200UL,T(true,-60,UInt!64.fromHexString("ee6829236006a312"))),
            Reference(28931257UL,T(true,-59,UInt!64.fromHexString("897186f0d41b906d"))),
            Reference(53308709UL,T(true,-59,UInt!64.fromHexString("8e5537c6a43b089e"))),
            Reference(73219997UL,T(true,-59,UInt!64.fromHexString("90df307458fb3a4a"))),
            Reference(170908811UL,T(true,-59,UInt!64.fromHexString("97a7333e7bf30cfa"))),
            Reference(4294967294UL,T(true,-59,UInt!64.fromHexString("b17217f7c9cf79ac"))),
            Reference(4294967295UL,T(true,-59,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(4294967296UL,T(true,-59,UInt!64.fromHexString("b17217f7d9cf79ac"))),
            Reference(14647204551UL,T(true,-59,UInt!64.fromHexString("bb429766e092f763"))),
            Reference(24429859351UL,T(true,-59,UInt!64.fromHexString("bf5a4267034d91b2"))),
            Reference(29917327740UL,T(true,-59,UInt!64.fromHexString("c0f93fc66cc7c68c"))),
            Reference(2153875302931UL,T(true,-59,UInt!64.fromHexString("e32fb290513eaaa4"))),
            Reference(3927560776543UL,T(true,-59,UInt!64.fromHexString("e7fe08843dfd3b53"))),
            Reference(27547141141602UL,T(true,-59,UInt!64.fromHexString("f7934aba6618e249"))),
            Reference(32423829816464UL,T(true,-59,UInt!64.fromHexString("f8e11afce5db2f7a"))),
            Reference(6545775197693676UL,T(true,-58,UInt!64.fromHexString("91ab9e5880c2b415"))),
            Reference(9007199254740990UL,T(true,-58,UInt!64.fromHexString("92f27bd939bfd0a2"))),
            Reference(9007199254740991UL,T(true,-58,UInt!64.fromHexString("92f27bd939bfd0c2"))),
            Reference(9007199254740992UL,T(true,-58,UInt!64.fromHexString("92f27bd939bfd0e2"))),
            Reference(16025752475463417UL,T(true,-58,UInt!64.fromHexString("95407c162df7d0f9"))),
            Reference(50530498459352047UL,T(true,-58,UInt!64.fromHexString("99d86d0bdd28cb1d"))),
            Reference(129318355547249388UL,T(true,-58,UInt!64.fromHexString("9d9aadcd9d8b4636"))),
            Reference(1318232995562866729UL,T(true,-58,UInt!64.fromHexString("a6e42bf10e2d29c4"))),
            Reference(9223372036854775806UL,T(true,-58,UInt!64.fromHexString("aeac4f97f2883bc5"))),
            Reference(9223372036854775807UL,T(true,-58,UInt!64.fromHexString("aeac4f97f2883bc5"))),
            Reference(9223372036854775808UL,T(true,-58,UInt!64.fromHexString("aeac4f97f2883bc5"))),
            Reference(18446744073709551614UL,T(true,-58,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(18446744073709551615UL,T(true,-58,UInt!64.fromHexString("b17217f7d1cf79ac"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);
            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong upper; T logMass; }
        enum references = [
            Reference(1UL,T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(2UL,T(true,-127,UInt!128.fromHexString("8c9f53d5681854bb520cc6aa829dbe5b"))),
            Reference(3UL,T(true,-127,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(4UL,T(true,-127,UInt!128.fromHexString("ce020fbf6c699b57efbbd28b03ac98fa"))),
            Reference(5UL,T(true,-127,UInt!128.fromHexString("e5585fd15100119136fea076849739b2"))),
            Reference(6UL,T(true,-127,UInt!128.fromHexString("f913957192d2baa37b4a4b67930df023"))),
            Reference(7UL,T(true,-126,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903"))),
            Reference(8UL,T(true,-126,UInt!128.fromHexString("8c9f53d5681854bb520cc6aa829dbe5b"))),
            Reference(9UL,T(true,-126,UInt!128.fromHexString("935d8dddaaa8ac16ea56d62b82d30a29"))),
            Reference(10UL,T(true,-126,UInt!128.fromHexString("99771dbbf9ba5dbde2e5fe653f697d19"))),
            Reference(11UL,T(true,-126,UInt!128.fromHexString("9f08b5e69cf3e7338df83d2143485a85"))),
            Reference(12UL,T(true,-126,UInt!128.fromHexString("a42821598c6287f433ee77c51fec1041"))),
            Reference(13UL,T(true,-126,UInt!128.fromHexString("a8e650b6bddd3bbcb01e1299ca83b5bd"))),
            Reference(14UL,T(true,-126,UInt!128.fromHexString("ad50b1ca6a40f809a0e44c9ac3252bab"))),
            Reference(15UL,T(true,-126,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(16UL,T(true,-126,UInt!128.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d5"))),
            Reference(26UL,T(true,-126,UInt!128.fromHexString("d2eefdc01c247f18fb1329ffc3ec9d88"))),
            Reference(609UL,T(true,-125,UInt!128.fromHexString("cd3b0e4768b4bd18fb1902fbb0182570"))),
            Reference(789UL,T(true,-125,UInt!128.fromHexString("d5814b3c52dfee02802e3815b79d3105"))),
            Reference(1574UL,T(true,-125,UInt!128.fromHexString("eb959726cef5a6b27fb6df74a7e8a7b3"))),
            Reference(65534UL,T(true,-124,UInt!128.fromHexString("b17207f7c9cf7456708e5b0f78bfc133"))),
            Reference(65535UL,T(true,-124,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(65536UL,T(true,-124,UInt!128.fromHexString("b17227f7c9cf7f011b390c2089d0d6d6"))),
            Reference(239088UL,T(true,-124,UInt!128.fromHexString("c627490bcb08b037802b06c4af60346d"))),
            Reference(751410UL,T(true,-124,UInt!128.fromHexString("d879af264ff1d52f69a3dd18d344cf30"))),
            Reference(2959200UL,T(true,-124,UInt!128.fromHexString("ee6829236006a311f498a7a08c8ae716"))),
            Reference(28931257UL,T(true,-123,UInt!128.fromHexString("897186f0d41b906d5a27d729c1ca94eb"))),
            Reference(53308709UL,T(true,-123,UInt!128.fromHexString("8e5537c6a43b089ded23f360b4eca40f"))),
            Reference(73219997UL,T(true,-123,UInt!128.fromHexString("90df307458fb3a4a2d4c5974207986b5"))),
            Reference(170908811UL,T(true,-123,UInt!128.fromHexString("97a7333e7bf30cfa7fd137f3d3114aaa"))),
            Reference(4294967294UL,T(true,-123,UInt!128.fromHexString("b17217f7c9cf79abc5e3b39801484c05"))),
            Reference(4294967295UL,T(true,-123,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(4294967296UL,T(true,-123,UInt!128.fromHexString("b17217f7d9cf79abc5e3b398069da15a"))),
            Reference(14647204551UL,T(true,-123,UInt!128.fromHexString("bb429766e092f762a745ef775caf997b"))),
            Reference(24429859351UL,T(true,-123,UInt!128.fromHexString("bf5a4267034d91b27fdfd1c844667ed6"))),
            Reference(29917327740UL,T(true,-123,UInt!128.fromHexString("c0f93fc66cc7c68c082dc1d0ff135d3a"))),
            Reference(2153875302931UL,T(true,-123,UInt!128.fromHexString("e32fb290513eaaa3ce8bac5b2181ae79"))),
            Reference(3927560776543UL,T(true,-123,UInt!128.fromHexString("e7fe08843dfd3b528038009e7f021c47"))),
            Reference(27547141141602UL,T(true,-123,UInt!128.fromHexString("f7934aba6618e248a78050074ae0e692"))),
            Reference(32423829816464UL,T(true,-123,UInt!128.fromHexString("f8e11afce5db2f7a11b55efb6e93b39b"))),
            Reference(6545775197693676UL,T(true,-122,UInt!128.fromHexString("91ab9e5880c2b41562ce757bce5a8e2a"))),
            Reference(9007199254740990UL,T(true,-122,UInt!128.fromHexString("92f27bd939bfd0a2433090b9e344b449"))),
            Reference(9007199254740991UL,T(true,-122,UInt!128.fromHexString("92f27bd939bfd0c2433090b9e3453449"))),
            Reference(9007199254740992UL,T(true,-122,UInt!128.fromHexString("92f27bd939bfd0e2433090b9e344b449"))),
            Reference(16025752475463417UL,T(true,-122,UInt!128.fromHexString("95407c162df7d0f8b543258c3430bdb6"))),
            Reference(50530498459352047UL,T(true,-122,UInt!128.fromHexString("99d86d0bdd28cb1d0a39daaa38bb0d4f"))),
            Reference(129318355547249388UL,T(true,-122,UInt!128.fromHexString("9d9aadcd9d8b463580043b8d993ba960"))),
            Reference(1318232995562866729UL,T(true,-122,UInt!128.fromHexString("a6e42bf10e2d29c380267f1aa06e3a9f"))),
            Reference(9223372036854775806UL,T(true,-122,UInt!128.fromHexString("aeac4f97f2883bc512bc24c9a3e32ad4"))),
            Reference(9223372036854775807UL,T(true,-122,UInt!128.fromHexString("aeac4f97f2883bc51abc24c9a3e32ad5"))),
            Reference(9223372036854775808UL,T(true,-122,UInt!128.fromHexString("aeac4f97f2883bc522bc24c9a3e32ad4"))),
            Reference(18446744073709551614UL,T(true,-122,UInt!128.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af"))),
            Reference(18446744073709551615UL,T(true,-122,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);
            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong upper; T logMass; }
        enum references = [
            Reference(1UL,T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(2UL,T(true,-255,UInt!256.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6"))),
            Reference(3UL,T(true,-255,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(4UL,T(true,-255,UInt!256.fromHexString("ce020fbf6c699b57efbbd28b03ac98fa24a638607b6f97064285b1563e0addcf"))),
            Reference(5UL,T(true,-255,UInt!256.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc"))),
            Reference(6UL,T(true,-255,UInt!256.fromHexString("f913957192d2baa37b4a4b67930df023261ae732274e1ffd5732a55eb97b611d"))),
            Reference(7UL,T(true,-254,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1"))),
            Reference(8UL,T(true,-254,UInt!256.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6"))),
            Reference(9UL,T(true,-254,UInt!256.fromHexString("935d8dddaaa8ac16ea56d62b82d30a28e28fecf9da5df90e83c61e8201f02d73"))),
            Reference(10UL,T(true,-254,UInt!256.fromHexString("99771dbbf9ba5dbde2e5fe653f697d194d69bb8cd8c396f155057edbd31d3ac7"))),
            Reference(11UL,T(true,-254,UInt!256.fromHexString("9f08b5e69cf3e7338df83d2143485a850ffeb249a7447e92d46c87685a27ca79"))),
            Reference(12UL,T(true,-254,UInt!256.fromHexString("a42821598c6287f433ee77c51fec1040c9dab82d7da45a8ea22109df903f943b"))),
            Reference(13UL,T(true,-254,UInt!256.fromHexString("a8e650b6bddd3bbcb01e1299ca83b5bd634a4462b04d3d8a0e1c98863fa86f1a"))),
            Reference(14UL,T(true,-254,UInt!256.fromHexString("ad50b1ca6a40f809a0e44c9ac3252baa81d82ce6abafeeff30a8d465b357bc4b"))),
            Reference(15UL,T(true,-254,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(16UL,T(true,-254,UInt!256.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e"))),
            Reference(26UL,T(true,-254,UInt!256.fromHexString("d2eefdc01c247f18fb1329ffc3ec9d884e8f322349e86a742e31f32fbcf6e829"))),
            Reference(609UL,T(true,-253,UInt!256.fromHexString("cd3b0e4768b4bd18fb1902fbb018256fc7ce32b00ab76b4c43a972a7c8e06b56"))),
            Reference(789UL,T(true,-253,UInt!256.fromHexString("d5814b3c52dfee02802e3815b79d3104cf35800e7dc31952b8313e2a6a25d077"))),
            Reference(1574UL,T(true,-253,UInt!256.fromHexString("eb959726cef5a6b27fb6df74a7e8a7b34b5ee6b3358376fe86757dbd61b69492"))),
            Reference(65534UL,T(true,-252,UInt!256.fromHexString("b17207f7c9cf7456708e5b0f78bfc132e72dc5078f10fc3da3e7ff71f4a7376a"))),
            Reference(65535UL,T(true,-252,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(65536UL,T(true,-252,UInt!256.fromHexString("b17227f7c9cf7f011b390c2089d0d6d641636beffd97e794e4fefc11ed326528"))),
            Reference(239088UL,T(true,-252,UInt!256.fromHexString("c627490bcb08b037802b06c4af60346d63b99e4b57f97195b0b8e75cb5bf9895"))),
            Reference(751410UL,T(true,-252,UInt!256.fromHexString("d879af264ff1d52f69a3dd18d344cf2fcb0495d25278769cae8636611979d920"))),
            Reference(2959200UL,T(true,-252,UInt!256.fromHexString("ee6829236006a311f498a7a08c8ae715df5807d0517559720eca9126eab7fa4a"))),
            Reference(28931257UL,T(true,-251,UInt!256.fromHexString("897186f0d41b906d5a27d729c1ca94ea8019ea16da284554dfab9650b5ada528"))),
            Reference(53308709UL,T(true,-251,UInt!256.fromHexString("8e5537c6a43b089ded23f360b4eca40e804ef185ca6cbba24ac39ec1b1f0a5b8"))),
            Reference(73219997UL,T(true,-251,UInt!256.fromHexString("90df307458fb3a4a2d4c5974207986b57fc07dfb741430cb893356744a10dbec"))),
            Reference(170908811UL,T(true,-251,UInt!256.fromHexString("97a7333e7bf30cfa7fd137f3d3114aa994396cc2346f5595351aadbe8907443a"))),
            Reference(4294967294UL,T(true,-251,UInt!256.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49"))),
            Reference(4294967295UL,T(true,-251,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(4294967296UL,T(true,-251,UInt!256.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b64"))),
            Reference(14647204551UL,T(true,-251,UInt!256.fromHexString("bb429766e092f762a745ef775caf997b7feb3d0f1061542bcd3f6c6debca14c0"))),
            Reference(24429859351UL,T(true,-251,UInt!256.fromHexString("bf5a4267034d91b27fdfd1c844667ed593a5a8f9b5908ed4dc7e64b62440dd41"))),
            Reference(29917327740UL,T(true,-251,UInt!256.fromHexString("c0f93fc66cc7c68c082dc1d0ff135d3a5435a28662ea250ecbc788a8563c68b7"))),
            Reference(2153875302931UL,T(true,-251,UInt!256.fromHexString("e32fb290513eaaa3ce8bac5b2181ae78871dd3a0fbfc01ef0df12a5b015178d5"))),
            Reference(3927560776543UL,T(true,-251,UInt!256.fromHexString("e7fe08843dfd3b528038009e7f021c46a4f032a8364462dd00b991c00e303053"))),
            Reference(27547141141602UL,T(true,-251,UInt!256.fromHexString("f7934aba6618e248a78050074ae0e691806ff72afefefd8dd0b1d4f0e0c2489a"))),
            Reference(32423829816464UL,T(true,-251,UInt!256.fromHexString("f8e11afce5db2f7a11b55efb6e93b39b7fbda5780be5e471078b39b00327815e"))),
            Reference(6545775197693676UL,T(true,-250,UInt!256.fromHexString("91ab9e5880c2b41562ce757bce5a8e2980033826ae5e8e53bd820c7b1f7a5f5a"))),
            Reference(9007199254740990UL,T(true,-250,UInt!256.fromHexString("92f27bd939bfd0a2433090b9e344b44921c9739bd43bcc330ba82c9d24feec82"))),
            Reference(9007199254740991UL,T(true,-250,UInt!256.fromHexString("92f27bd939bfd0c2433090b9e345344921c9739bd6e676ddb652d757cfa9972c"))),
            Reference(9007199254740992UL,T(true,-250,UInt!256.fromHexString("92f27bd939bfd0e2433090b9e344b44921c9739bd991218860fd81f27a5441d7"))),
            Reference(16025752475463417UL,T(true,-250,UInt!256.fromHexString("95407c162df7d0f8b543258c3430bdb66d12d22ccccb6f0772e7601e43579d83"))),
            Reference(50530498459352047UL,T(true,-250,UInt!256.fromHexString("99d86d0bdd28cb1d0a39daaa38bb0d4f7fc7aec3716f610d2424a74e04d7f53b"))),
            Reference(129318355547249388UL,T(true,-250,UInt!256.fromHexString("9d9aadcd9d8b463580043b8d993ba95f92584822b77d547a537c2ab2580dffd0"))),
            Reference(1318232995562866729UL,T(true,-250,UInt!256.fromHexString("a6e42bf10e2d29c380267f1aa06e3a9ec1c5ca336b41afe1e122fad65d5c5979"))),
            Reference(9223372036854775806UL,T(true,-250,UInt!256.fromHexString("aeac4f97f2883bc512bc24c9a3e32ad47bef7619d8ce5354c93a385372d1a398"))),
            Reference(9223372036854775807UL,T(true,-250,UInt!256.fromHexString("aeac4f97f2883bc51abc24c9a3e32ad483ef7619d8ce5354d3e4e2fe1d7c4e43"))),
            Reference(9223372036854775808UL,T(true,-250,UInt!256.fromHexString("aeac4f97f2883bc522bc24c9a3e32ad47bef7619d8ce5354de8f8da8c826f8ee"))),
            Reference(18446744073709551614UL,T(true,-250,UInt!256.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d7"))),
            Reference(18446744073709551615UL,T(true,-250,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);
            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);
        }
    }
    {
        alias T = Fp!512;
        struct Reference { ulong upper; T logMass; }
        enum references = [
            Reference(1UL,T(true,-512,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(2UL,T(true,-511,UInt!512.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6554bc16906634c282838fb9981cd476ab12e0a838f53de50e79bec7888126332"))),
            Reference(3UL,T(true,-511,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(4UL,T(true,-511,UInt!512.fromHexString("ce020fbf6c699b57efbbd28b03ac98fa24a638607b6f97064285b1563e0addcf3881ca9f2c5a4a6d6e16ee02b3410d47e22cf115a17bb513e37353737d15519a"))),
            Reference(5UL,T(true,-511,UInt!512.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc4927fc793d5922745303a517274a54f327c5619e6fefeef2fb478a0d10c73f45"))),
            Reference(6UL,T(true,-511,UInt!512.fromHexString("f913957192d2baa37b4a4b67930df023261ae732274e1ffd5732a55eb97b611d1ef0beb1439817a048d1b5aaba503b596c0df95c2fbe1d2bcf0a4b81a5dd5ee5"))),
            Reference(7UL,T(true,-510,UInt!512.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba0edca58985270c172402ffe3c783b944cb1e302a850ea18f31d816c5ecd0f4a1c"))),
            Reference(8UL,T(true,-510,UInt!512.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6554bc16906634c282838fb9981cd476ab12e0a838f53de50e79bec7888126332"))),
            Reference(9UL,T(true,-510,UInt!512.fromHexString("935d8dddaaa8ac16ea56d62b82d30a28e28fecf9da5df90e83c61e8201f02d72962f02d7b1a8105ccc70cbc02c5f0d682c622418410be2dafb8f788402e516d6"))),
            Reference(10UL,T(true,-510,UInt!512.fromHexString("99771dbbf9ba5dbde2e5fe653f697d194d69bb8cd8c396f155057edbd31d3ac7198edbf4015ad951ac5493dc14fb72d8b71c258dadb7c8fafde3b37430e79932"))),
            Reference(11UL,T(true,-510,UInt!512.fromHexString("9f08b5e69cf3e7338df83d2143485a850ffeb249a7447e92d46c87685a27ca791e821bc4ba277c603ee7274a6663b13dcf2e5c5ca845ffca877993d0ccbe0dac"))),
            Reference(12UL,T(true,-510,UInt!512.fromHexString("a42821598c6287f433ee77c51fec1040c9dab82d7da45a8ea22109df903f943b1dea39052c17d0f0a764f12b3abe7e68a14cd0f01fea0ef5034b6758a350d3bb"))),
            Reference(13UL,T(true,-510,UInt!512.fromHexString("a8e650b6bddd3bbcb01e1299ca83b5bd634a4462b04d3d8a0e1c98863fa86f1989667ce0bd46f6f639ce2f942fe6a470f152a83b882d16e6f15af48b17491d7c"))),
            Reference(14UL,T(true,-510,UInt!512.fromHexString("ad50b1ca6a40f809a0e44c9ac3252baa81d82ce6abafeeff30a8d465b357bc4ac6e6c604195ecb4acb27f4ce1a872a5949ad7dcc9867c9b265879ff60293da66"))),
            Reference(15UL,T(true,-510,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(16UL,T(true,-510,UInt!512.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e40152cbb658d3099f9b71bb1ac68d24393aae13d3b2272abce48110b6f63672b"))),
            Reference(26UL,T(true,-510,UInt!512.fromHexString("d2eefdc01c247f18fb1329ffc3ec9d884e8f322349e86a742e31f32fbcf6e8297ff1a21d8994f23c3c55796642b3eb2009c50fc556fdcd795b69e2b4cc1b94cb"))),
            Reference(609UL,T(true,-509,UInt!512.fromHexString("cd3b0e4768b4bd18fb1902fbb018256fc7ce32b00ab76b4c43a972a7c8e06b5580627186d8f1fe8ea1fb174eed975b072575d45e4e59806d01f23d2013283505"))),
            Reference(789UL,T(true,-509,UInt!512.fromHexString("d5814b3c52dfee02802e3815b79d3104cf35800e7dc31952b8313e2a6a25d076b187916040fd0ad0bd9277e4be8fb50adeba5a14df02b58c62b67fd4fe257f87"))),
            Reference(1574UL,T(true,-509,UInt!512.fromHexString("eb959726cef5a6b27fb6df74a7e8a7b34b5ee6b3358376fe86757dbd61b694920ea2f5b06a44d132dd5c6238c91b392fa4b0fc23a45750fd594a32d66c0b321f"))),
            Reference(65534UL,T(true,-508,UInt!512.fromHexString("b17207f7c9cf7456708e5b0f78bfc132e72dc5078f10fc3da3e7ff71f4a7376a5d889c607aca007e5915e999640267a104872351e923de7c2beb39464439445a"))),
            Reference(65535UL,T(true,-508,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(65536UL,T(true,-508,UInt!512.fromHexString("b17227f7c9cf7f011b390c2089d0d6d641636beffd97e794e4fefc11ed326527f3908a640780bfe8c3e25936fe1a4b372b893d040543c66b01bb382e786cf7a8"))),
            Reference(239088UL,T(true,-508,UInt!512.fromHexString("c627490bcb08b037802b06c4af60346d63b99e4b57f97195b0b8e75cb5bf9894874b2d43d9d601cc5cfff16124df2667100d65f0aa74a39f517302bf1a4152d0"))),
            Reference(751410UL,T(true,-508,UInt!512.fromHexString("d879af264ff1d52f69a3dd18d344cf2fcb0495d25278769cae8636611979d9207fd6c68f25fb006feb49a4c743a14b6d7fef29105f21aae4e5522080b404cc9e"))),
            Reference(2959200UL,T(true,-508,UInt!512.fromHexString("ee6829236006a311f498a7a08c8ae715df5807d0517559720eca9126eab7fa49800f34997d09583eec8e49f898840d34c779b4eb25667b514fb5a5718bb30e4a"))),
            Reference(28931257UL,T(true,-507,UInt!512.fromHexString("897186f0d41b906d5a27d729c1ca94ea8019ea16da284554dfab9650b5ada52841df8aee3f1a4f32de418ba18fefd79b20f4668760b9cb41a8b215ec9e0a3391"))),
            Reference(53308709UL,T(true,-507,UInt!512.fromHexString("8e5537c6a43b089ded23f360b4eca40e804ef185ca6cbba24ac39ec1b1f0a5b7d1f713b1d55d7d752598e506987cfc7d4f17e6c5646d11bb8c15c339fe30cde9"))),
            Reference(73219997UL,T(true,-507,UInt!512.fromHexString("90df307458fb3a4a2d4c5974207986b57fc07dfb741430cb893356744a10dbebf790e4604314b3b200aeaa99a373d01eec68acce371a4f3dd8081f27f81760fd"))),
            Reference(170908811UL,T(true,-507,UInt!512.fromHexString("97a7333e7bf30cfa7fd137f3d3114aa994396cc2346f5595351aadbe890744398fac58a237f65414f6efa1b84a8f3df8accaa27e38a74c9f4775270221fcd693"))),
            Reference(4294967294UL,T(true,-507,UInt!512.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49288c9362412560338e7c2168310e596c1808302af733d27396d338ba5e531e01"))),
            Reference(4294967295UL,T(true,-507,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(4294967296UL,T(true,-507,UInt!512.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b63fa39ae33f0074e52706a404a20a198717cbb92a6f1a2d67b1d1d11b506546f8b"))),
            Reference(14647204551UL,T(true,-507,UInt!512.fromHexString("bb429766e092f762a745ef775caf997b7feb3d0f1061542bcd3f6c6debca14bf8cad0691401233259132d32d901a2c47e82258c95ee7551714fcabc1ce758bf9"))),
            Reference(24429859351UL,T(true,-507,UInt!512.fromHexString("bf5a4267034d91b27fdfd1c844667ed593a5a8f9b5908ed4dc7e64b62440dd40eb011d6fb5e1e554478bde24cc3d750d01a7cab349ed70b68845a168af4febc2"))),
            Reference(29917327740UL,T(true,-507,UInt!512.fromHexString("c0f93fc66cc7c68c082dc1d0ff135d3a5435a28662ea250ecbc788a8563c68b680812d0f2e1c473d090ad636df3aeffe6f5a8efa0fc5db9b26eccaa521bc97c5"))),
            Reference(2153875302931UL,T(true,-507,UInt!512.fromHexString("e32fb290513eaaa3ce8bac5b2181ae78871dd3a0fbfc01ef0df12a5b015178d4809471b9b48860dfb54b99431af98eca9e2ffb882fe6f5c7ced93909d483966b"))),
            Reference(3927560776543UL,T(true,-507,UInt!512.fromHexString("e7fe08843dfd3b528038009e7f021c46a4f032a8364462dd00b991c00e3030537f878a4cb714cc38fb65ce25d7e85aacd7316a9d588612ce9b9d58e21742fc64"))),
            Reference(27547141141602UL,T(true,-507,UInt!512.fromHexString("f7934aba6618e248a78050074ae0e691806ff72afefefd8dd0b1d4f0e0c248998e8114e74838017b961fe0e3c3474811e6538d3c414be4089be1719e1c3e89d8"))),
            Reference(32423829816464UL,T(true,-507,UInt!512.fromHexString("f8e11afce5db2f7a11b55efb6e93b39b7fbda5780be5e471078b39b00327815e5b0f1e0647dd02299a8e5bbe77ca3d33e8e927dd17ea1e5b80bc6a76f2e18751"))),
            Reference(6545775197693676UL,T(true,-506,UInt!512.fromHexString("91ab9e5880c2b41562ce757bce5a8e2980033826ae5e8e53bd820c7b1f7a5f59bba1361e079b0273a8520d11ebd6b169c02d14f39a3a09d1525d2c1b4f8d2c64"))),
            Reference(9007199254740990UL,T(true,-506,UInt!512.fromHexString("92f27bd939bfd0a2433090b9e344b44921c9739bd43bcc330ba82c9d24feec81b139b0c1c9f619dd1323ecfc5e5b629bff8a5a3fa3b3f6ac228f5caff9fd0596"))),
            Reference(9007199254740991UL,T(true,-506,UInt!512.fromHexString("92f27bd939bfd0c2433090b9e345344921c9739bd6e676ddb652d757cfa9972c5be4c1d2db072aee26dfa8b81a171e6a046aa84484027b8c70943cfe026b8c7f"))),
            Reference(9007199254740992UL,T(true,-506,UInt!512.fromHexString("92f27bd939bfd0e2433090b9e344b44921c9739bd991218860fd81f27a5441d7068fd2e3ec183bff35460f1e807d84e2b3f5a0f40efaab176943c7f6b584be12"))),
            Reference(16025752475463417UL,T(true,-506,UInt!512.fromHexString("95407c162df7d0f8b543258c3430bdb66d12d22ccccb6f0772e7601e43579d837fc693e9a39663b5caff17174a84e036f227f13998b00d42f29863000e254b8a"))),
            Reference(50530498459352047UL,T(true,-506,UInt!512.fromHexString("99d86d0bdd28cb1d0a39daaa38bb0d4f7fc7aec3716f610d2424a74e04d7f53ad2e9bf0bed09c4c8915f711da1dd1464b5ab37ad6c291e645a135d538fd6ce1c"))),
            Reference(129318355547249388UL,T(true,-506,UInt!512.fromHexString("9d9aadcd9d8b463580043b8d993ba95f92584822b77d547a537c2ab2580dffd073b5826e32663f162b96ca1ef4d2f57d995117cc1064da107a7a3a49ee3fef89"))),
            Reference(1318232995562866729UL,T(true,-506,UInt!512.fromHexString("a6e42bf10e2d29c380267f1aa06e3a9ec1c5ca336b41afe1e122fad65d5c597908053f4a74f71d03d1cebc8e3ca0fe9424eae586ffc3691062bf08f651511a82"))),
            Reference(9223372036854775806UL,T(true,-506,UInt!512.fromHexString("aeac4f97f2883bc512bc24c9a3e32ad47bef7619d8ce5354c93a385372d1a3987d6ee99d4189533b2ffab96b1989ee603a8b048dfb4451cfdea65d0459a2d931"))),
            Reference(9223372036854775807UL,T(true,-506,UInt!512.fromHexString("aeac4f97f2883bc51abc24c9a3e32ad483ef7619d8ce5354d3e4e2fe1d7c4e4338199447ec33fde5f43efdaf5dce32a4a979f37cea3340bf16b9de3c6d241145"))),
            Reference(9223372036854775808UL,T(true,-506,UInt!512.fromHexString("aeac4f97f2883bc522bc24c9a3e32ad47bef7619d8ce5354de8f8da8c826f8edd2c43ef296dea890b88341f3a21276e8c3138d1683ccda58f9780a1f2b4ff402"))),
            Reference(18446744073709551614UL,T(true,-506,UInt!512.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d6916320cb18965742ff7330d928d7f8eeca61e168f46b547759f82537b253c6c6"))),
            Reference(18446744073709551615UL,T(true,-506,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
        ];
        foreach (c; references) {
            if (c.upper > size_t.max) continue;
            const upper = cast(size_t)c.upper;
            assert(uniformDiscreteLPMF!T(0,0,upper) == c.logMass);
            assert(uniformDiscreteLPMF!T(upper,0,upper) == c.logMass);
        }
    }
}
// END discrete-uniform log references
