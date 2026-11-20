/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution).

The one-argument overloads use location zero and scale one. The other
overloads take a location and a positive scale; scale is not the standard
deviation (the standard deviation is $(D sqrt(2) * scale)).

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.laplace;

import mir.bignum.fp: Fp;
import mir.stat.internal.standardize: standardize;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Laplace probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate PDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
@safe pure nothrow @nogc
T laplacePDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp, fabs;

    return 0.5 * exp(-fabs(x));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns a density, which may exceed one; it is not a point probability.

For scales below one, tail densities can remain representable even when the
corresponding unit-scale density underflows. This overload preserves such
results and reduces rounding error near the subnormal boundary.
Use $(D laplaceLPDF) when the logarithm of the density is needed, including
when the density itself is outside the representable range.

The table assumes a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == location)) $(TD $(D 0.5 / scale)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate PDF
    location = finite location parameter
    scale = positive finite scale parameter
+/
@safe pure nothrow @nogc
T laplacePDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: exp, fabs;
    const T z = standardize(x, location, scale);
    const T density = laplacePDF(z);
    // Scaling can recover a representable result after the standard density
    // has underflowed or lost precision in the subnormal range.
    if (density < T.min_normal && scale < 1)
    {
        // Quarter-exponentials retain precision even at subnormal scales.
        // Apply the scale before forming the potentially subnormal density.
        const T quarterDecay = exp(-fabs(z) / 4);
        return (((T(0.5) * quarterDecay) / scale) * quarterDecay) * quarterDecay * quarterDecay;
    }
    return density / scale;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    laplacePDF(-2.0).shouldApprox == 0.06766764;
    laplacePDF(-1.0).shouldApprox == 0.1839397;
    laplacePDF(-0.5).shouldApprox == 0.3032653;
    laplacePDF(0.0).shouldApprox == 0.5;
    laplacePDF(0.5).shouldApprox == 0.3032653;
    laplacePDF(1.0).shouldApprox == 0.1839397;
    laplacePDF(2.0).shouldApprox == 0.06766764;

    // Can also provide location/scale parameters
    laplacePDF(-1.0, 2.0, 3.0).shouldApprox == 0.06131324;
    laplacePDF(1.0, 2.0, 3.0).shouldApprox == 0.1194219;
    laplacePDF(4.0, 2.0, 3.0).shouldApprox == 0.08556952;
}

/// The density peaks at the location and tends to zero in either tail.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(laplacePDF(3.0, 3.0, 2.0) == 0.25);
    assert(laplacePDF(-double.infinity) == 0.0);
    assert(laplacePDF(double.infinity) == 0.0);

    // A small scale can restore a representable tail density.
    assert(laplacePDF(800.0) == 0.0);
    assert(laplacePDF(800 * 1e-300, 0.0, 1e-300) > 0.0);
}

// Small scales must restore representable densities in both tails.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: ldexp;
    import mir.math.common: fabs;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        // 90-digit exponential references, adjusted for Laplace normalization.
        static if (T.max_exp == 128)
        {
            enum T exponent = 120;
            enum T tinyExponent = 191;
            enum T tinyExpected = 8.00245628516475785138969687948202631e-39L;
            enum T expected = 6.52291358810724124141821303035571164e-15L;
            enum T smallExponent = 90;
            enum T smallExpected = 1.63880252479810308607222164267638868e-39L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T exponent = 800;
            enum T tinyExponent = 1450;
            enum T tinyExpected = 3.79504197181520648721376487356687376e-307L;
            enum T expected = 1.64842823987804998528024178106714402e-40L;
            enum T smallExponent = 710;
            enum T smallExpected = 8.95257245135025991216632140458264468e-309L;
        }
        else static if (T.max_exp == 16384)
        {
            enum T exponent = 11420;
            enum T tinyExponent = 22750;
            // The reference uses a 64-bit significand; adjust for the
            // smaller minimum subnormal of wider formats.
            const T tinyExpected = ldexp(T(1.73306716928370349897078093711548328e-4930L),
                T.mant_dig - 64);
            enum T expected = 6.76714733679691181158578826955688255e-29L;
            enum T smallExponent = 11358;
            enum T smallExpected = 3.83976405302577104196109564353492666e-4933L;
        }
        const T scale = ldexp(T(1), 2 - T.max_exp);
        const T tolerance = T.mant_dig > 53 ? 2 * T.epsilon * exponent : 4 * T.epsilon;
        foreach (T sign; [T(-1), T(1)])
        {
            const T actual = laplacePDF(sign * exponent * scale, T(0), scale);
            assert(actual > 0);
            assert(fabs(actual / (expected / 2) - 1) < tolerance);
            const T smallActual = laplacePDF(sign * smallExponent / 2, T(0), T(0.5));
            assert(fabs(smallActual - smallExpected / 2) <=
                tolerance * (smallExpected / 2) + 2 * T.min_normal * T.epsilon);
        }
        const T tinyScale = T.min_normal * T.epsilon;
        const T tinyActual = laplacePDF(tinyExponent * tinyScale, T(0), tinyScale);
        assert(tinyActual > 0);
        assert(fabs(tinyActual / (tinyExpected / 2) - 1) < 2 * tolerance);
        assert(laplacePDF(T(0), T(0), scale) == T(0.5) / scale);
        assert(laplacePDF(T.infinity, T(0), scale) == 0);
        assert(laplacePDF(-T.infinity, T(0), scale) == 0);
        assert(laplacePDF(T(1), T(0), T.infinity) == 0);
    }}

    const double nearBoundary = laplacePDF(177.5, 0.0, 0.25);
    assert(fabs(nearBoundary - cast(double) 8.95257245135025991216632140458264468e-309L) <=
        3 * double.min_normal * double.epsilon);

    // Require underflowing exp at CTFE; ordinary inputs may work independently.
    static if (__traits(compiles, {
        import mir.math.common: exp;
        enum double probe = exp(-800.0);
    }))
    {
        enum double ctfeDensity = laplacePDF(800 * 1e-300, 0.0, 1e-300);
        static assert(fabs(ctfeDensity / 1.833937292088697626e-48 - 1) < 2e-13);
    }
    const double density = laplacePDF(800 * 1e-300, 0.0, 1e-300);
    assert(fabs(density / 1.833937292088697626e-48 - 1) < 2e-13);
}

/++
Computes the standard Laplace density at a represented observation.

Params:
    T = Represented $(D Fp) type, inferred from the argument by default.
    x = Observation, which may be infinite or NaN.

Returns:
    $(D exp(-abs(x)) / 2) in type $(D T), with guarded evaluation.
    The density is one half at zero and zero at either infinity. NaN propagates.
+/
@safe pure nothrow @nogc
T laplacePDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return laplacePDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplacePDF(T(0)) == T(0.5));
}

/++
Computes the Laplace density at a represented observation for a given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    $(D exp(-abs(x - location) / scale) / (2 * scale)) in type $(D T).
    The peak density is $(D 0.5 / scale) and can exceed one. Either infinity
    gives zero and NaN propagates. Guarded evaluation applies the scale
    before rounding to the result width and exponent range.
+/
@safe pure nothrow @nogc
T laplacePDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_laplace: laplaceDensity;
    return laplaceDensity(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplacePDF(T(3), T(3), T(0.25)) == T(2));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;

    // exp(-2)/6 at the same exact observations on both sides of the peak.
    static foreach (T; AliasSeq!(float, double, real))
    {
        foreach (x; [T(-5), T(7)])
            assert(approxEqual(laplacePDF(x, T(1), T(3)),
                T(0.0225558805394354486489999158287474L), 8 * T.epsilon));
    }
    alias T = Fp!64;
    alias U = UInt!64;
    const reference = T(false, -69, U(0xb8c71c27fabdd154UL));
    assert(laplacePDF(T(-5), T(1), T(3)) == reference);
    assert(laplacePDF(T(7), T(1), T(3)) == reference);
    assert(laplacePDF(T(double.nan)).isNaN);
    assert(laplacePDF(T(-double.infinity)) == T(0));
    assert(laplacePDF(T(double.infinity)) == T(0));

    const quantum = T(false, long.min, U(1));
    assert(laplacePDF(T(0), T(0), quantum) == T(false, long.max - 63, U(1UL << 63)));
    // At z = 2^63 the standard density rounds to zero. Reducing the scale
    // to the smallest quantum restores a normal density. Reference computed from
    // -z/log(2) - 1 - log2(scale) at 200 and 400 decimal digits.
    assert(laplacePDF(T(false, 63, U(1))) == T(0));
    assert(laplacePDF(T(false, long.min + 63, U(1)), T(0), quantum)
        == T(false, -4083141060989546748L, U(0x98d3377e65eee9c8UL)));
    assert(laplacePDF(T(1), T(0), quantum) == T(0));
    // For scale = 2^long.max the peak is exactly one output quantum.
    const hugeScale = T(false, long.max - 1, U(2));
    assert(laplacePDF(T(0), T(0), hugeScale) == quantum);
    assert(laplacePDF(hugeScale, T(0), hugeScale) == T(0));
    // At twice that scale the peak ties to zero. Its predecessor rounds up.
    assert(laplacePDF(T(0), T(0), T(false, long.max - 1, U(4))) == T(0));
    assert(laplacePDF(T(0), T(0), T(false, long.max - 63, U.max)) == quantum);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias U = UInt!128;
    auto adjacent = T(1);
    adjacent.coefficient += 1;
    const scale = T(false, adjacent.exponent, U(3));
    // z = 1/3 exactly. Scale the exp(-1/3)/6 reference by 2^127.
    enum expected = T(false, -4, U.fromHexString("f49376174c9c33625ccb10d06be5310d"));
    assert(laplacePDF(adjacent, T(1), scale) == expected);
}

/++
Computes the Laplace cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate CDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
@safe pure nothrow @nogc
T laplaceCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp;

    if (x <= 0) {
        return 0.5 * exp(x);
    } else {
        return 1 - 0.5 * exp(-x);
    }
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the lower-tail probability $(D P(X <= x)).

The table assumes a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == location)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate CDF
    location = finite location parameter
    scale = positive finite scale parameter
+/
@safe pure nothrow @nogc
T laplaceCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return laplaceCDF(standardize(x, location, scale));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    laplaceCDF(-2.0).shouldApprox == 0.06766764;
    laplaceCDF(-1.0).shouldApprox == 0.1839397;
    laplaceCDF(-0.5).shouldApprox == 0.3032653;
    laplaceCDF(0.0).shouldApprox == 0.5;
    laplaceCDF(0.5).shouldApprox == 0.6967347;
    laplaceCDF(1.0).shouldApprox == 0.8160603;
    laplaceCDF(2.0).shouldApprox == 0.9323324;

    // Can also provide location/scale parameters
    laplaceCDF(-1.0, 2.0, 3.0).shouldApprox == 0.1839397;
    laplaceCDF(1.0, 2.0, 3.0).shouldApprox == 0.3582657;
    laplaceCDF(4.0, 2.0, 3.0).shouldApprox == 0.7432914;
}

/++
Computes the lower tail of the standard Laplace distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.

Returns:
    $(D P(X <= x)) in type $(D T). The probability is one half at the location.
    Negative and positive infinity give zero and one, respectively. NaN propagates.
    Finite interior results use guarded evaluation.
+/
@safe pure nothrow @nogc
T laplaceCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return laplaceCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceCDF(T(0)) == T(0.5));
}

/++
Computes the lower tail of the Laplace distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    $(D P(X <= x)) in type $(D T). The probability is one half at the location.
    Negative and positive infinity give zero and one, respectively. NaN propagates.
    Finite interior results use guarded evaluation.
+/
@safe pure nothrow @nogc
T laplaceCDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_laplace: laplaceTail;
    return laplaceTail!false(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceCDF(T(3), T(3), T(2)) == T(0.5));
}

/++
Computes the Laplace complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate CCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
@safe pure nothrow @nogc
T laplaceCCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp;

    if (x <= 0) {
        return 1 - 0.5 * exp(x);
    } else {
        return 0.5 * exp(-x);
    }
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

The table assumes a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x == location)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate CCDF
    location = finite location parameter
    scale = positive finite scale parameter
+/
T laplaceCCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return laplaceCCDF(standardize(x, location, scale));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    laplaceCCDF(-2.0).shouldApprox == 0.9323324;
    laplaceCCDF(-1.0).shouldApprox == 0.8160603;
    laplaceCCDF(-0.5).shouldApprox == 0.6967347;
    laplaceCCDF(0.0).shouldApprox == 0.5;
    laplaceCCDF(0.5).shouldApprox == 0.3032653;
    laplaceCCDF(1.0).shouldApprox == 0.1839397;
    laplaceCCDF(2.0).shouldApprox == 0.06766764;

    // Can also provide location/scale parameters
    laplaceCCDF(-1.0, 2.0, 3.0).shouldApprox == 0.8160603;
    laplaceCCDF(1.0, 2.0, 3.0).shouldApprox == 0.6417343;
    laplaceCCDF(4.0, 2.0, 3.0).shouldApprox == 0.2567086;
}

/++
Computes the strict upper tail of the standard Laplace distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.

Returns:
    $(D P(X > x)) in type $(D T). The probability is one half at the location.
    Negative and positive infinity give one and zero, respectively. NaN propagates.
    Finite interior results use guarded evaluation.
+/
@safe pure nothrow @nogc
T laplaceCCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return laplaceCCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceCCDF(T(0)) == T(0.5));
}

/++
Computes the strict upper tail of the Laplace distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    $(D P(X > x)) in type $(D T). The probability is one half at the location.
    Negative and positive infinity give one and zero, respectively. NaN propagates.
    Finite interior results use guarded evaluation.
+/
@safe pure nothrow @nogc
T laplaceCCDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_laplace: laplaceTail;
    return laplaceTail!true(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceCCDF(T(3), T(3), T(2)) == T(0.5));
}

// Exact input pairs with standardized offsets +/-2 and +/-1/3. Compute
// the small tail and its reflection separately at 260 and 400 decimal digits.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const low = T(-5), high = T(7), location = T(1), scale = T(3);
        const small = T(0.0676676416183063459469997474862422L);
        const large = T(0.932332358381693654053000252513758L);
        assert(approxEqual(laplaceCDF(low, location, scale), small, 8 * T.epsilon));
        assert(approxEqual(laplaceCCDF(high, location, scale), small, 8 * T.epsilon));
        assert(approxEqual(laplaceCDF(high, location, scale), large, 8 * T.epsilon));
        assert(approxEqual(laplaceCCDF(low, location, scale), large, 8 * T.epsilon));
    }}
    alias T = Fp!64;
    alias U = UInt!64;
    const small = T(false, -67, U(0x8a95551dfc0e5cffUL));
    const large = T(false, -64, U(0xeead555c407e3460UL));
    assert(laplaceCDF(T(-5), T(1), T(3)) == small);
    assert(laplaceCCDF(T(7), T(1), T(3)) == small);
    assert(laplaceCDF(T(7), T(1), T(3)) == large);
    assert(laplaceCCDF(T(-5), T(1), T(3)) == large);

    // Opposite maximum bounds overflow a finite difference but give z = 2.
    const hi = T(false, long.max - 1, U.max);
    const lo = T(true, hi.exponent, hi.coefficient);
    assert(laplaceCCDF(hi, lo, hi) == small);
    assert(laplaceCDF(hi, lo, hi) == large);
    assert(laplaceCCDF(lo, hi, hi) == large);
    // Adjacent points at the exponent floor differ by precisely one quantum.
    const one = T(false, long.min, U(1)), two = T(false, long.min, U(2));
    const three = T(false, long.min, U(3));
    assert(laplaceCCDF(two, one, three) == T(false, -65, U(0xb76e98917975268aUL)));
    assert(laplaceCCDF(one, two, three) == T(false, -64, U(0xa448b3b743456cbbUL)));
    assert(laplaceCCDF(T(1), one, T(1)) == T(false, -66, U(0xbc5ab1b16779be35UL)));
    assert(laplaceCDF(lo) == T(0));
    assert(laplaceCCDF(lo) == T(1));
    assert(laplaceCDF(T(true, 0, U(0))) == T(0.5));
    assert(laplaceCCDF(T(false, -1, U(2)), T(1), one) == T(0.5));
    assert(laplaceCDF(T(double.infinity)) == T(1));
    assert(laplaceCCDF(T(-double.infinity)) == T(1));
    assert(laplaceCDF(T(double.nan)).isNaN && laplaceCCDF(T(double.nan)).isNaN);

    // At the intervening half step the tail is about 0.742 quanta.
    // Rounding the decay first gives one quantum and halving it ties to zero.
    assert(laplaceCCDF(T(6393154322601327829L)) == one);
    assert(laplaceCCDF(T(false, -1, U(12786308645202655659UL))) == one);
    assert(laplaceCCDF(T(6393154322601327830L)) == T(0));
    assert(laplaceCDF(T(0), T(-2), T(1)) == large);
    assert(laplaceCDF(T(100)) == T(1));
    // The quadratic term puts both probabilities on the center side of a midpoint.
    assert(laplaceCDF(T(false, -64, U(1))) == T(0.5));
    assert(laplaceCCDF(T(false, -65, U(1))) == T(0.5));
    assert(laplaceCCDF(T(100)) == T(false, -209, U(0xd460f8a7157ae57aUL)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias U = UInt!128;
    enum small = T(false, -129, U.fromHexString("b76e989179752689c5984c9c50ebe4ca"));
    enum large = T(false, -128, U.fromHexString("a448b3b743456cbb1d33d9b1d78a0d9b"));
    // Adjacent normalized observations at location one, scaled to z = 1/3.
    auto above = T(1);
    above.coefficient += 1;
    const scale = T(false, above.exponent, U(3));
    assert(laplaceCCDF(above, T(1), scale) == small);
    assert(laplaceCDF(above, T(1), scale) == large);
}

// The cubic term puts each result across the midpoint from one half.
// References use the exact dyadic inputs at 850 and 1150 decimal digits.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    enum bits = 192;
    alias T = Fp!bits;
    alias U = UInt!bits;
    const scale = T(false, -bits, U.max); // 1 - 2^-bits
    const t = T(false, -bits, U(1));
    const location = T(false, -2 * bits - 1, U(1)); // t^2 / 2
    auto successor = T(0.5);
    successor.coefficient += 1;
    assert(laplaceCDF(t, location, scale) == successor);

    const h = T(false, -bits - 1, U(1)); // t / 2
    const predecessor = T(false, -bits - 1, U.max);
    const lowerLocation = T(false, -2 * bits - 3, U(3)); // 3*h^2 / 2
    assert(laplaceCCDF(h, lowerLocation, scale) == predecessor);

    const farLocation = T(true, -3 * bits - 2, (U(1) << (bits - 1)) + U(1));
    assert(laplacePDF(h, farLocation, T(1)) == predecessor);
}

/++
Computes the Laplace inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
@safe pure nothrow @nogc
T laplaceInvCDF(T)(const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;

    if (p <= 0.5) {
        return log(2 * p);
    } else {
        return -log(2 - 2 * p);
    }
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the quantile at cumulative probability $(D p).

The table assumes a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D location)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    location = finite location parameter
    scale = positive finite scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
T laplaceInvCDF(T)(const T p, const T location, const T scale)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: fabs;
    const T z = laplaceInvCDF(p);
    const T offset = scale * z;
    // A finite location can bring an overflowing offset back into range.
    // Halve before multiplying; true endpoint infinities stay unchanged.
    if (fabs(offset) == T.infinity && fabs(z) <= T.max
        && scale <= T.max && fabs(location) <= T.max)
        return ((z * T(0.5)) * scale + location * T(0.5)) * 2;
    return location + offset;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    laplaceInvCDF(0.0).shouldApprox == -double.infinity;
    laplaceInvCDF(0.25).shouldApprox == -0.6931472;
    laplaceInvCDF(0.5).shouldApprox == 0.0;
    laplaceInvCDF(0.75).shouldApprox == 0.6931472;
    laplaceInvCDF(1.0).shouldApprox == double.infinity;

    // Can also provide location/scale parameters
    laplaceInvCDF(0.2, 2, 3).shouldApprox == -0.7488722;
    laplaceInvCDF(0.4, 2, 3).shouldApprox == 1.330569;
    laplaceInvCDF(0.6, 2, 3).shouldApprox == 2.669431;
    laplaceInvCDF(0.8, 2, 3).shouldApprox == 4.748872;
}

/++
Computes the Laplace log probability density function (LPDF).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == 0)) $(TD $(D -log(2))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate LPDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
@safe pure nothrow @nogc
T laplaceLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: fabs;
    import mir.math.constant: LN2;

    return -T(LN2) - fabs(x);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

The table assumes a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == location)) $(TD $(D -log(2) - log(scale))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate LPDF
    location = finite location parameter
    scale = positive finite scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Laplace_distribution, Laplace Distribution)
+/
T laplaceLPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;

    return laplaceLPDF(standardize(x, location, scale)) - log(scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    laplaceLPDF(-2.0).shouldApprox == log(0.06766764);
    laplaceLPDF(-1.0).shouldApprox == log(0.1839397);
    laplaceLPDF(-0.5).shouldApprox == log(0.3032653);
    laplaceLPDF(0.0).shouldApprox == log(0.5);
    laplaceLPDF(0.5).shouldApprox == log(0.3032653);
    laplaceLPDF(1.0).shouldApprox == log(0.1839397);
    laplaceLPDF(2.0).shouldApprox == log(0.06766764);

    // Can also provide location/scale parameters
    laplaceLPDF(-1.0, 2.0, 3.0).shouldApprox == log(0.06131324);
    laplaceLPDF(1.0, 2.0, 3.0).shouldApprox == log(0.1194219);
    laplaceLPDF(4.0, 2.0, 3.0).shouldApprox == log(0.08556952);
}

/++
Computes the natural logarithm of the standard Laplace density.

Params:
    T = Represented $(D Fp) type, inferred from the argument by default.
    x = Observation, which may be infinite or NaN.

Returns:
    $(D -log(2) - abs(x)) in type $(D T). Either infinity gives negative
    infinity and NaN propagates. Finite results use guarded evaluation.
+/
@safe pure nothrow @nogc
T laplaceLPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return laplaceLPDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceLPDF(T(1)) == laplaceLPDF(T(-1)));
}

/++
Computes the natural logarithm of the Laplace density for a given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments by default.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    $(D -log(2 * scale) - abs(x - location) / scale) in type $(D T).
    The peak value is $(D -log(2 * scale)). The logarithm can remain finite
    when the density rounds to zero. Either infinite observation or negative
    overflow gives negative infinity. NaN propagates. Finite results use
    guarded evaluation. Cancellation beyond the working precision can give zero.
+/
@safe pure nothrow @nogc
T laplaceLPDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_laplace: laplaceLogDensity;
    return laplaceLogDensity(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(laplaceLPDF(T(3), T(3), T(0.5)) == T(0));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    // -log(6) - 2 at the same exact observations for all four types.
    static foreach (T; AliasSeq!(float, double, real))
    {
        foreach (x; [T(-5), T(7)])
            assert(approxEqual(laplaceLPDF(x, T(1), T(3)),
                T(-3.7917594692280550008124773583807023L), 8 * T.epsilon));
    }
    alias T = Fp!64;
    alias U = UInt!64;
    const expected = T(true, -62, U(0xf2ac2fe8a88008c9UL));
    assert(laplaceLPDF(T(-5), T(1), T(3)) == expected);
    assert(laplaceLPDF(T(7), T(1), T(3)) == expected);
    const negInf = T(-double.infinity);
    assert(laplaceLPDF(T(double.infinity)) == negInf && laplaceLPDF(negInf) == negInf);
    assert(laplaceLPDF(T(double.nan)).isNaN);

    const quantum = T(false, long.min, U(1));
    const huge = T(false, long.max - 1, U(2));
    assert(laplaceLPDF(quantum) == T(true, -64, U(0xb17217f7d1cf79acUL)));
    assert(laplaceLPDF(T(0), T(0), quantum) == T(false, -1, U(0xb17217f7d1cf79aaUL)));
    assert(laplaceLPDF(T(0), T(0), huge) == T(true, -1, U(0xb17217f7d1cf79acUL)));
    assert(laplaceLPDF(huge, T(0), quantum) == negInf);
    assert(laplaceLPDF(quantum, T(0), T(0.5)) == T(true, long.min, U(2)));

    // With h = 2^-64 these coordinates leave a residual of order h^3.
    const x = T(false, -65, U(1)), location = T(false, -130, U(1));
    assert(laplaceLPDF(x, location, T(false, -65, U.max))
        == T(true, -258, U(0xaaaaaaaaaaaaaaacUL)));

    // z is exactly halfway between two results before the remote log(2)
    // correction. The standard log density rounds toward larger magnitude.
    const high = T(false, 1000, U(1)), low = T(true, 936, U(1));
    assert(laplaceLPDF(high, low, T(1)) == T(true, 937, U(0x8000000000000001UL)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias U = UInt!128;
    // At scale 1/4 the zero lies at log(2)/4. Evaluate its two adjacent
    // represented observations independently at 200 and 400 decimal digits.
    enum below = T(false, -130, U.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"));
    auto above = below;
    above.coefficient += 1;
    enum positive = T(false, -257, U.fromHexString("81e6864ce5316c5b141a2eb71755f458"));
    enum negative = T(true, -256, U.fromHexString("bf0cbcd98d6749d275f2e8a4745505d4"));
    assert(laplaceLPDF(below, T(0), T(0.25)) == positive);
    assert(laplaceLPDF(above, T(0), T(0.25)) == negative);
}

// Finite location and scale can compensate for overflowing intermediates.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp, log;
    import mir.math.constant: LN2;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T m = T.max;
        const T tail = T(0.0676676416183063459469997474862422017L);
        assert(approxEqual(laplaceCCDF(m, -m, m), tail, T(16)*T.epsilon));
        assert(approxEqual(laplaceCDF(-m, m, m), tail, T(16)*T.epsilon));
        assert(approxEqual(laplaceCDF(m, -m, m), 1-tail, T(16)*T.epsilon));
        assert(approxEqual(laplaceCCDF(-m, m, m), 1-tail, T(16)*T.epsilon));
        assert(approxEqual(laplacePDF(m, -m, m), tail/m, T(32)*T.epsilon, T(0)));
        assert(approxEqual(laplaceLPDF(m, -m, m), -T(2)-T(LN2)-log(m), T(16)*T.epsilon));
        assert(laplaceCDF(-T.infinity, T(0), T(1)) == 0);
        assert(laplaceCCDF(T.infinity, T(0), T(1)) == 0);
        assert(laplacePDF(T.infinity, T(0), T(0.5)) == 0);
        assert(laplaceInvCDF(T(0), T(1), m) == -T.infinity);
        assert(laplaceInvCDF(T(1), T(-1), m) == T.infinity);
        // At p=1/8 the standard quantile is -log(4), so adding the
        // location after scaling should recover a finite negative result.
        const T expected = (1-T(2)*T(LN2))*m;
        assert(approxEqual(laplaceInvCDF(T(0.125), m, m), expected, T(32)*T.epsilon));
        assert(approxEqual(laplaceInvCDF(T(0.875), -m, m), -expected, T(32)*T.epsilon));
        assert(laplaceInvCDF(T(0.5), T(3), T(2)) == 3);
        // Split the exponential in the reference so neither factor underflows
        // before scaling. Exercise density recovery for every floating type.
        const T smallScale = T.min_normal * 16;
        const T distance = -log(T.min_normal) + 32;
        const T halfDensity = exp(-distance / 2);
        const T recovered = (halfDensity / smallScale) * halfDensity / 2;
        assert(approxEqual(laplacePDF(distance * smallScale, T(0), smallScale),
            recovered, T(8) * distance * T.epsilon, T(0)));
    }}
    // Decimal references at the exact binary double inputs, evaluated with
    // 100 decimal digits; the standard density underflows before scaling.
    assert(approxEqual(laplacePDF(1e-297, 0.0, 1e-300),
        2.537979448774691293172203145033537e-135, 4e-13, 0.0));
}
