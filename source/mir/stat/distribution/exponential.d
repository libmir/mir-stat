/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution).

The parameter $(D lambda) is a rate (the reciprocal of the mean), not a scale.
The support is $(D x >= 0). The inverse CDF currently requires $(D 0 <= p < 1);
its contract excludes $(D p == 1), although the mathematical quantile tends to
positive infinity as $(D p) approaches one.

The two-argument $(D Fp) overloads require a finite positive rate and use
guarded integer-significand calculations. General transcendental results
have no universal correct-rounding guarantee. In the log density, a
difference unresolved at working precision yields zero.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.exponential;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

/++
Computes the exponential probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

For a large rate, scaling is applied between exponential factors when needed
to preserve a representable tail density that would be lost by computing
$(D exp(-lambda * x)) first.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D lambda)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    lambda = positive rate parameter (reciprocal of the mean)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution)
+/
@safe pure nothrow @nogc
T exponentialPDF(T)(const T x, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (lambda > 0, "lambda must be greater than zero")
{
    import mir.math.common: exp;

    const T exponent = -lambda * x;
    const T decay = exp(exponent);
    if (decay < T.min_normal && lambda > 1 && lambda < T.infinity)
    {
        // Scale between two exponential factors to avoid losing the decay
        // before multiplication by the rate can restore a representable result.
        const T halfDecay = exp(exponent / 2);
        return (lambda * halfDecay) * halfDecay;
    }
    return lambda * decay;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.exponentialPDF(2.0).approxEqual(0.7357589));
    assert(0.75.exponentialPDF(2.0).approxEqual(0.4462603));
    assert(0.25.exponentialPDF(0.5).approxEqual(0.4412485));
}

/// The density starts at the rate and tends to zero in the upper tail.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(exponentialPDF(0.0, 2.0) == 2.0);
    assert(exponentialPDF(double.infinity, 2.0) == 0.0);
}

// Scaling must not lose a representable density when the decay underflows.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp;
    import mir.math.common: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // References computed with 90-digit arithmetic. Power-of-two rates
        // keep the chosen integer exponents exact after standardization.
        static if (T.max_exp == 128)
        {
            enum T exponent = 120;
            enum T expected = 6.52291358810724124141821303035571164e-15L;
            enum T smallExponent = 90;
            enum T smallExpected = 1.63880252479810308607222164267638868e-39L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T exponent = 800;
            enum T expected = 1.64842823987804998528024178106714402e-40L;
            enum T smallExponent = 710;
            enum T smallExpected = 8.95257245135025991216632140458264468e-309L;
        }
        else static if (T.max_exp == 16384)
        {
            enum T exponent = 11420;
            enum T expected = 6.76714733679691181158578826955688255e-29L;
            enum T smallExponent = 11358;
            enum T smallExpected = 3.83976405302577104196109564353492666e-4933L;
        }
        const T rate = ldexp(T(1), T.max_exp - 2);
        // Extended-real exp can accumulate error proportional to its exponent.
        const T tolerance = T.mant_dig > 53 ? 2 * T.epsilon * exponent : 4 * T.epsilon;
        const T actual = exponentialPDF(exponent / rate, rate);
        assert(actual > 0);
        assert(fabs(actual / expected - 1) < tolerance);
        const T smallActual = exponentialPDF(smallExponent / 2, T(2));
        assert(fabs(smallActual - smallExpected) <=
            tolerance * smallExpected + 2 * T.min_normal * T.epsilon);
        assert(exponentialPDF(T(0), rate) == rate);
        assert(exponentialPDF(T.infinity, rate) == 0);
        assert(exponentialPDF(T.max, rate) == 0);
        assert(exponentialPDF(T(0), T.min_normal) == T.min_normal);
        assert(exponentialPDF(T(1), T.infinity) != exponentialPDF(T(1), T.infinity));
    }}

    // A modest rate must not introduce log/exp cancellation near underflow.
    const double nearBoundary = exponentialPDF(354.69820926613204, 2.0);
    assert(fabs(nearBoundary - cast(double) 1.63711785526567331511458767387450060e-308L) <=
        2 * double.min_normal * double.epsilon);

    // Require underflowing exp at CTFE; ordinary inputs may work independently.
    static if (__traits(compiles, {
        import mir.math.common: exp;
        enum double probe = exp(-800.0);
    }))
    {
        enum double ctfeDensity = exponentialPDF(800 / 1e300, 1e300);
        static assert(fabs(ctfeDensity / 3.667874584177790597e-48 - 1) < 2e-13);
    }
    const double density = exponentialPDF(800 / 1e300, 1e300);
    assert(fabs(density / 3.667874584177790597e-48 - 1) < 2e-13);
}

/++
Computes the exponential cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    lambda = positive rate parameter (reciprocal of the mean)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution)
+/
@safe pure nothrow @nogc
T exponentialCDF(T)(const T x, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (lambda > 0, "lambda must be greater than zero")
{
    import std.math: expm1;

    return -expm1(-lambda * x);
}

// Small probabilities must not disappear when exp(-lambda*x) rounds to one.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = T.epsilon / 16;
        assert(exponentialCDF(x, T(1)) > 0);
        assert(approxEqual(exponentialCDF(x, T(1)) / x, T(1), T.epsilon * 4, T(0)));
        assert(approxEqual(exponentialCDF(x / 2, T(2)) / x, T(1), T.epsilon * 4, T(0)));
        assert(approxEqual(exponentialCDF(exponentialInvCDF(x, T(2)), T(2)) / x,
            T(1), T.epsilon * 4, T(0)));
        assert(exponentialCDF(T(0), T(2)) == 0);
        assert(exponentialCDF(T.infinity, T(2)) == 1);
        assert(approxEqual(exponentialCDF(T(0.5), T(2)) + exponentialCCDF(T(0.5), T(2)), T(1)));
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.exponentialCDF(2.0).approxEqual(0.6321206));
    assert(0.75.exponentialCDF(2.0).approxEqual(0.7768698));
    assert(0.25.exponentialCDF(0.5).approxEqual(0.1175031));
}

/++
Computes the exponential complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    lambda = positive rate parameter (reciprocal of the mean)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution)
+/
@safe pure nothrow @nogc
T exponentialCCDF(T)(const T x, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (lambda > 0, "lambda must be greater than zero")
{
    import mir.math.common: exp;

    return exp(-lambda * x);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.exponentialCCDF(2.0).approxEqual(1 - exponentialCDF(0.5, 2.0)));
    assert(0.75.exponentialCCDF(2.0).approxEqual(1 - exponentialCDF(0.75, 2.0)));
    assert(0.25.exponentialCCDF(0.5).approxEqual(1 - exponentialCDF(0.25, 0.5)));
}

/++
Computes the exponential inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD Contract violation))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    lambda = positive rate parameter (reciprocal of the mean)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution)
+/

@safe pure nothrow @nogc
T exponentialInvCDF(T)(const T p, const T lambda)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p < 1, "p must be less than or equal to 1")
    in (lambda > 0, "lambda must be greater than zero")
{
    import mir.math.internal.log1p: log1p;

    return -log1p(-p) / lambda;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.6321206.exponentialInvCDF(2.0).approxEqual(0.5));
    assert(0.7768698.exponentialInvCDF(2.0).approxEqual(0.75));
    assert(0.1175031.exponentialInvCDF(0.5).approxEqual(0.25));
}

// Scaling a tiny probability must preserve a representable quantile.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T p = T.epsilon / 16;
        assert(approxEqual(exponentialInvCDF(p, p), T(1), 4 * T.epsilon, T(0)));
    }}
}

/++
Computes the exponential log probability density function (LPDF).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D log(lambda))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < 0) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    lambda = positive rate parameter (reciprocal of the mean)

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution)
+/
@safe pure nothrow @nogc
T exponentialLPDF(T)(const T x, const T lambda)
    if (isFloatingPoint!T)
    in (x >= 0, "x must be greater than or equal to 0")
    in (lambda > 0, "lambda must be greater than zero")
{
    import mir.math.common: log;

    return log(lambda) - x * lambda;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;

    assert(0.5.exponentialLPDF(2.0).approxEqual(log(exponentialPDF(0.5, 2.0))));
    assert(0.75.exponentialLPDF(2.0).approxEqual(log(exponentialPDF(0.75, 2.0))));
    assert(0.25.exponentialLPDF(0.5).approxEqual(log(exponentialPDF(0.25, 0.5))));
}

private bool validRate(uint bits)(Fp!bits rate) @safe pure nothrow @nogc
{
    return !rate.isSpecial && !rate.sign && !!rate.coefficient;
}

/// Extended-precision quantile for a finite positive rate and 0 <= p < 1.
/// Uses a guarded logarithm and retains the exponent through division.
@safe pure nothrow @nogc
T exponentialInvCDF(T)(const T p, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (validRate(lambda), "lambda must be finite and positive")
    in (validQuantileProbability(p), "p must be in [0, 1)")
{
    import mir.stat.internal.fp_exponential: exponentialQuantile;
    return exponentialQuantile(p, lambda);
}

private bool validQuantileProbability(uint bits)(Fp!bits p) @safe pure nothrow @nogc
{
    import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityLess;
    return fpProbabilityValid(p) && fpProbabilityLess(p, Fp!bits(1));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const tiny = T(false, long.min, U(1));
    assert(exponentialInvCDF(T(0), tiny) == T(0));
    assert(exponentialInvCDF(tiny, T(2)) == tiny);
    assert(exponentialInvCDF(tiny, T(3)) == T(0));
    assert(exponentialInvCDF(tiny, tiny) == T(1));
    assert(exponentialInvCDF(T(0.5), T(2)) == T(false, -65, U(0xb17217f7d1cf79acUL)));
    assert(exponentialInvCDF(T(false,-64,U.max), T(1)) == T(false,-58,U(0xb17217f7d1cf79acUL)));
}

private bool validObservation(uint bits)(Fp!bits x) @safe pure nothrow @nogc
{
    return !x.isNaN && (!x.sign || (!x.coefficient && !x.isInfinity));
}

/// Strict upper tail for a represented observation and finite positive rate. Evaluates the decay directly with guard bits.
@safe pure nothrow @nogc
T exponentialCCDF(T)(const T x, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (validRate(lambda), "lambda must be finite and positive")
    in (validObservation(x), "x must be nonnegative")
{
    import mir.stat.internal.fp_exponential: exponentialDecay;
    return exponentialDecay!false(x, lambda);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    assert(exponentialCCDF(T(0), T(2)) == T(1));
    assert(exponentialCCDF(T(double.infinity), T(2)) == T(0));
    assert(exponentialCCDF(T(0.5), T(2)) == T(false,-65,UInt!64(0xbc5ab1b16779be35UL)));
    assert(exponentialCCDF(T(1), T(false,100,UInt!64(1))) == T(0));
}

/// Lower tail for a finite positive rate. Retains the small-product correction at rounding midpoints.
@safe pure nothrow @nogc
T exponentialCDF(T)(const T x, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (validRate(lambda), "lambda must be finite and positive")
    in (validObservation(x), "x must be nonnegative")
{
    import mir.stat.internal.fp_exponential: exponentialLower;
    return exponentialLower(x, lambda);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const tiny = T(false, long.min, U(1));
    assert(exponentialCDF(tiny,T(0.5)) == T(0));
    assert(exponentialCDF(tiny,T(1.5)) == tiny);
    assert(exponentialCDF(tiny,T(1)) == tiny);
    assert(exponentialCDF(T(0),tiny) == T(0));
    assert(exponentialCDF(T(double.infinity),tiny) == T(1));
    assert(exponentialCDF(T(0.5),T(2)) == T(false,-64,U(0xa1d2a7274c4320e5UL)));
}

/// Density for a finite positive rate. Scales the guarded decay before rounding to the result width.
@safe pure nothrow @nogc
T exponentialPDF(T)(const T x, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (validRate(lambda), "lambda must be finite and positive")
    in (validObservation(x), "x must be nonnegative")
{
    import mir.stat.internal.fp_exponential: exponentialDecay;
    return exponentialDecay!true(x, lambda);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const rate = T(false,long.max - 1,U.max);
    assert(exponentialPDF(T(0),rate) == rate);
    assert(exponentialPDF(T(double.infinity),rate) == T(0));
    assert(exponentialPDF(T(0.5),T(2)) == T(false,-64,U(0xbc5ab1b16779be35UL)));
    // Choose x so the product is 2^63. The decay alone underflows,
    // but the rate rescales it into the finite exponent range.
    const scale = T(false,long.max - 1,U(1));
    const x = T(false,64 - long.max,U(1));
    assert(exponentialCCDF(x,scale) == T(0));
    const density = exponentialPDF(x,scale);
    assert(density == T(false,-4083141060989546749L,U(0x98d3377e65eee9c8UL)));
}

/// Log density for a finite positive rate. Subtracts the exact product from a guarded logarithm. Severe cancellation can reduce accuracy.
@safe pure nothrow @nogc
T exponentialLPDF(T)(const T x, const T lambda)
    if (is(T == Fp!size, size_t size))
    in (validRate(lambda), "lambda must be finite and positive")
    in (validObservation(x), "x must be nonnegative")
{
    import mir.stat.internal.fp_exponential: exponentialLogDensity;
    return exponentialLogDensity(x, lambda);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const tiny = T(false,long.min,U(1));
    assert(exponentialLPDF(tiny,T(1)) == T(true,long.min,U(1)));
    assert(exponentialLPDF(T(0),T(1)) == T(0));
    assert(exponentialLPDF(T(0),T(2)) == T(false,-64,U(0xb17217f7d1cf79acUL)));
    const nearZero = T(false,-65,U(0xb17217f7d1cf79acUL));
    assert(exponentialLPDF(nearZero,T(2)) == T(true,-130,U(0xd871319ff0342543UL)));
    assert(exponentialLPDF(T(double.infinity),T(2)) == T(true,long.max,U(0)));
}

// Both arguments are exactly representable in every type. The references
// use 180-digit decimal arithmetic on x = p = 1/2 and lambda = 5/4.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const x = T(0.5), rate = T(1.25);
        assert(approxEqual(exponentialPDF(x,rate), T(0.66907678564873780244577813502755800L), 8*T.epsilon));
        assert(approxEqual(exponentialCDF(x,rate), T(0.46473857148100975804337749197795359L), 8*T.epsilon));
        assert(approxEqual(exponentialCCDF(x,rate), T(0.53526142851899024195662250802204640L), 8*T.epsilon));
        assert(approxEqual(exponentialLPDF(x,rate), T(-0.4018564486857902442337049096901654L), 8*T.epsilon));
        assert(approxEqual(exponentialInvCDF(x,rate), T(0.55451774444795624753378569716654125L), 8*T.epsilon));
    }}
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const x = T(0.5), rate = T(1.25);
    assert(exponentialPDF(x, rate) == T(false, -64, U(0xab489dc0dfc70cb0UL)));
    assert(exponentialCDF(x, rate) == T(false, -65, U(0xedf236cb66c1854dUL)));
    assert(exponentialCCDF(x, rate) == T(false, -64, U(0x8906e49a4c9f3d59UL)));
    assert(exponentialLPDF(x, rate) == T(true, -65, U(0xcdc020e19597794fUL)));
    assert(exponentialInvCDF(x, rate) == T(false, -64, U(0x8df4dff974a5faf0UL)));
}

// Check the same represented inputs against the 128-bit references.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias Wide = Fp!128;
    alias U = UInt!128;
    const x = Wide(0.5), rate = Wide(1.25);
    enum expectedPDF = Wide(false,-128,U.fromHexString("ab489dc0dfc70cafaecf45d7b385b669"));
    assert(exponentialPDF(x, rate) == expectedPDF);
    enum expectedCDF = Wide(false,-129,U.fromHexString("edf236cb66c1854d4eb45d0d472a0f57"));
    assert(exponentialCDF(x, rate) == expectedCDF);
    enum expectedCCDF = Wide(false,-128,U.fromHexString("8906e49a4c9f3d5958a5d1795c6af854"));
    assert(exponentialCCDF(x, rate) == expectedCCDF);
    enum expectedLPDF = Wide(true,-129,U.fromHexString("cdc020e19597794f689f8434011976d4"));
    assert(exponentialLPDF(x, rate) == expectedLPDF);
    enum expectedInvCDF = Wide(false,-128,U.fromHexString("8df4dff974a5faefd4b62946698f2bbf"));
    assert(exponentialInvCDF(x, rate) == expectedInvCDF);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const huge = T(false,long.max - 1,U.max);
    const tiny = T(false,long.min,U(1));
    assert(exponentialLPDF(huge,huge) == T(true,long.max,U(0)));
    assert(exponentialLPDF(T(2),T(0.5)) == T(true,-63,U(0xd8b90bfbe8e7bcd6UL)));
    assert(exponentialLPDF(tiny,T(2)) == T(false,-64,U(0xb17217f7d1cf79acUL)));
    assert(exponentialCDF(huge,huge) == T(1));
    assert(exponentialCDF(T(64),T(1)) == T(1));
}

// At p = 2^(1-bits), the quadratic term is exactly half an output ulp.
// The positive cubic term decides the midpoint. Across the binade boundary
// the predecessor has half the spacing, and its quantile rounds to p.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    enum bits = 256u;
    alias T = Fp!bits;
    alias U = UInt!bits;
    enum leading = U(1) << (bits - 1);
    enum e = 2 - 2 * long(bits);
    const below = T(false, e - 1, U.max);
    const p = T(false, e, leading);
    const above = T(false, e, leading + 1);
    assert(exponentialInvCDF(below, T(1)) == p);
    assert(exponentialInvCDF(p, T(1)) == above);
    assert(exponentialInvCDF(above, T(1)) == T(false, e, leading + 2));
}

version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(double, Fp!64))
    {{
        const T x = T(0.5), rate = T(2);
        static foreach (f; AliasSeq!(exponentialInvCDF, exponentialPDF, exponentialLPDF,
                                    exponentialCDF, exponentialCCDF))
        {
            static assert(is(typeof(f!T(x, rate)) == T));
            static assert(is(typeof(f(x, rate)) == T));
        }
    }}

    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled)
    {
        alias T = Fp!64;
        assertThrown!AssertError(exponentialInvCDF(T(1), T(1)));
        assertThrown!AssertError(exponentialInvCDF(T(-1), T(1)));
        assertThrown!AssertError(exponentialInvCDF(T(0.5), T(0)));
        static foreach (f; AliasSeq!(exponentialPDF, exponentialLPDF, exponentialCDF, exponentialCCDF))
        {
            assertThrown!AssertError(f(T(-1), T(1)));
            assertThrown!AssertError(f(T(0), T(double.infinity)));
        }
    }
}
