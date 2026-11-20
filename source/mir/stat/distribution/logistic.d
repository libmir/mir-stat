/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution).

The one-argument overloads use location zero and scale one. The other
overloads take a finite location and a positive finite scale. The location
is the median and mean; scale is not the standard deviation. The standard
deviation is $(D scale * PI / sqrt(3)).

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.logistic;

import mir.bignum.fp: Fp;
import mir.stat.internal.standardize: standardize;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Logistic probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == 0)) $(TD $(D 0.25)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate PDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp, fabs;

    const T exp_x = exp(-fabs(x));
    return exp_x / ((1 + exp_x) * (1 + exp_x));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns a density, which may exceed one; it is not a point probability.

The table assumes finite parameters and a positive scale.

For small scales, the tail calculation applies scaling between exponential
factors when needed. This preserves representable densities even when the
standardized density would underflow before division by the scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == location)) $(TD $(D 0.25 / scale)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate PDF
    location = finite location parameter
    scale = positive finite scale parameter
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: exp, fabs;

    const T z = standardize(x, location, scale);
    const T density = logisticPDF(z);
    if (density < T.min_normal && scale < 1)
    {
        // In this tail, 1 + exp(-abs(z)) rounds to one. Divide between
        // four exponential factors. Half-exponentials can themselves become
        // subnormal when a subnormal scale restores a representable density.
        const T quarterDecay = exp(-fabs(z) / 4);
        return ((quarterDecay / scale) * quarterDecay) * quarterDecay * quarterDecay;
    }
    return density / scale;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticPDF(-2.0).shouldApprox == 0.1049936;
    logisticPDF(-1.0).shouldApprox == 0.1966119;
    logisticPDF(-0.5).shouldApprox == 0.2350037;
    logisticPDF(0.0).shouldApprox == 0.25;
    logisticPDF(0.5).shouldApprox == 0.2350037;
    logisticPDF(1.0).shouldApprox == 0.1966119;
    logisticPDF(2.0).shouldApprox == 0.1049936;

    // Can also provide location/scale parameters
    logisticPDF(-1.0, 2.0, 3.0).shouldApprox == 0.06553731;
    logisticPDF(1.0, 2.0, 3.0).shouldApprox == 0.08106072;
    logisticPDF(4.0, 2.0, 3.0).shouldApprox == 0.07471913;
}

/// At the location, the density is one quarter divided by the scale.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(logisticPDF(0.0) == 0.25);
    assert(logisticPDF(3.0, 3.0, 2.0) == 0.125);
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
        // 90-digit references; the logistic denominator rounds to one here.
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
            const T actual = logisticPDF(sign * exponent * scale, T(0), scale);
            assert(actual > 0);
            assert(fabs(actual / expected - 1) < tolerance);
            const T smallActual = logisticPDF(sign * smallExponent / 2, T(0), T(0.5));
            assert(fabs(smallActual - smallExpected) <=
                tolerance * smallExpected + 2 * T.min_normal * T.epsilon);
        }
        const T tinyScale = T.min_normal * T.epsilon;
        const T tinyActual = logisticPDF(tinyExponent * tinyScale, T(0), tinyScale);
        assert(tinyActual > 0);
        assert(fabs(tinyActual / tinyExpected - 1) < 2 * tolerance);
        assert(logisticPDF(T(0), T(0), scale) == T(0.25) / scale);
        assert(logisticPDF(T.infinity, T(0), scale) == 0);
        assert(logisticPDF(-T.infinity, T(0), scale) == 0);
        assert(logisticPDF(T(1), T(0), T.infinity) == 0);
    }}

    // Require underflowing exp at CTFE; ordinary inputs may work independently.
    static if (__traits(compiles, {
        import mir.math.common: exp;
        enum double probe = exp(-800.0);
    }))
    {
        enum double ctfeDensity = logisticPDF(800 * 1e-300, 0.0, 1e-300);
        static assert(fabs(ctfeDensity / 3.667874584177395251e-48 - 1) < 2e-13);
    }
    const double density = logisticPDF(800 * 1e-300, 0.0, 1e-300);
    assert(fabs(density / 3.667874584177395251e-48 - 1) < 2e-13);
}

/++
Computes the density for the standard logistic distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.

Returns:
    The density in type $(D T), using guarded evaluation.
    The peak is one quarter divided by the scale. Either infinity gives zero and NaN propagates.
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return logisticPDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticPDF(T(0)) == T(0.25));
}

/++
Computes the density for the logistic distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    The density in type $(D T), using guarded evaluation.
    The peak is one quarter divided by the scale. Either infinity gives zero and NaN propagates.
+/
@safe pure nothrow @nogc
T logisticPDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_logistic: logisticDensity;
    return logisticDensity(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticPDF(T(3), T(3), T(2)) == T(0.125));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(logisticPDF(R(7), R(1), R(3)),
            R(0.03499786180116883911637472492014181L), 8 * R.epsilon));
    const expected = T(false, -68, U(0x8f59eafddaec198dUL));
    assert(logisticPDF(T(7), T(1), T(3)) == expected);
    assert(logisticPDF(T(-5), T(1), T(3)) == expected);
    assert(logisticPDF(T(double.nan)).isNaN);
    assert(logisticPDF(T(-double.infinity)) == T(0));
    // At z = 2^63, exp(-z) underflows. Division by 2^long.min restores
    // the density; normalize exp(-z) in base two before applying the scale.
    const quantum = T(false, long.min, U(1));
    assert(logisticPDF(T(false, 63, U(1))) == T(0));
    assert(logisticPDF(T(1), T(0), quantum) == T(0));
    assert(logisticPDF(T(false, long.min + 63, U(1)), T(0), quantum)
        == T(false, -4083141060989546747L, U(0x98d3377e65eee9c8UL)));
    const largeScale = T(false, long.max - 1, U(2));
    assert(logisticPDF(T(0), T(0), largeScale) == T(0)); // half a quantum
    assert(logisticPDF(T(0), T(0), T(false, long.max - 64, U.max)) == quantum);
}

/++
Computes the Logistic cumulative distribution function (CDF).

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
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp;

    if (x < 0)
    {
        const T exp_x = exp(x);
        return exp_x / (1 + exp_x);
    }
    return 1 / (1 + exp(-x));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the lower-tail probability $(D P(X <= x)).

The table assumes finite parameters and a positive scale.

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
T logisticCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return logisticCDF(standardize(x, location, scale));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticCDF(-2.0).shouldApprox == 0.1192029;
    logisticCDF(-1.0).shouldApprox == 0.2689414;
    logisticCDF(-0.5).shouldApprox == 0.3775407;
    logisticCDF(0.0).shouldApprox == 0.5;
    logisticCDF(0.5).shouldApprox == 0.6224593;
    logisticCDF(1.0).shouldApprox == 0.7310586;
    logisticCDF(2.0).shouldApprox == 0.8807971;

    // Can also provide location/scale parameters
    logisticCDF(-1.0, 2.0, 3.0).shouldApprox == 0.2689414;
    logisticCDF(1.0, 2.0, 3.0).shouldApprox == 0.4174298;
    logisticCDF(4.0, 2.0, 3.0).shouldApprox == 0.6607564;
}

/++
Computes the lower-tail probability $(D P(X <= x)) for the standard logistic distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.

Returns:
    The lower-tail probability $(D P(X <= x)) in type $(D T), using guarded evaluation.
    At the location the result is one half. Negative and positive infinity give zero and one. NaN propagates.
+/
@safe pure nothrow @nogc
T logisticCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return logisticCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticCDF(T(0)) == T(0.5));
}

/++
Computes the lower-tail probability $(D P(X <= x)) for the logistic distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    The lower-tail probability $(D P(X <= x)) in type $(D T), using guarded evaluation.
    At the location the result is one half. Negative and positive infinity give zero and one. NaN propagates.
+/
@safe pure nothrow @nogc
T logisticCDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_logistic: logisticTail;
    return logisticTail!false(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticCDF(T(3), T(3), T(2)) == T(0.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    assert(logisticCDF(T(double.nan)).isNaN);
    assert(logisticCDF(T(double.infinity)) == T(1));
    assert(logisticCDF(T(-double.infinity)) == T(0));
    assert(logisticCDF(T(true, long.max - 1, U.max)) == T(0));
    assert(logisticCDF(T(false, -1, U(2)), T(1), T(2)) == T(0.5));
}

/++
Computes the Logistic complementary cumulative distribution function (CCDF).

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
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x)
    if (isFloatingPoint!T)
{
    return logisticCDF(-x);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

The table assumes finite parameters and a positive scale.

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
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    return logisticCCDF(standardize(x, location, scale));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticCCDF(-2.0).shouldApprox == 0.8807971;
    logisticCCDF(-1.0).shouldApprox == 0.7310586;
    logisticCCDF(-0.5).shouldApprox == 0.6224593;
    logisticCCDF(0.0).shouldApprox == 0.5;
    logisticCCDF(0.5).shouldApprox == 0.3775407;
    logisticCCDF(1.0).shouldApprox == 0.2689414;
    logisticCCDF(2.0).shouldApprox == 0.1192029;

    // Can also provide location/scale parameters
    logisticCCDF(-1.0, 2.0, 3.0).shouldApprox == 0.7310586;
    logisticCCDF(1.0, 2.0, 3.0).shouldApprox == 0.5825702;
    logisticCCDF(4.0, 2.0, 3.0).shouldApprox == 0.3392436;
}

/++
Computes the strict upper-tail probability $(D P(X > x)) for the standard logistic distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.

Returns:
    The strict upper-tail probability $(D P(X > x)) in type $(D T), using guarded evaluation.
    At the location the result is one half. Negative and positive infinity give one and zero. NaN propagates.
+/
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return logisticCCDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticCCDF(T(0)) == T(0.5));
}

/++
Computes the strict upper-tail probability $(D P(X > x)) for the logistic distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    The strict upper-tail probability $(D P(X > x)) in type $(D T), using guarded evaluation.
    At the location the result is one half. Negative and positive infinity give one and zero. NaN propagates.
+/
@safe pure nothrow @nogc
T logisticCCDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_logistic: logisticTail;
    return logisticTail!true(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticCCDF(T(3), T(3), T(2)) == T(0.5));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const zero = T(0), one = T(1);
    assert(logisticCCDF(T(double.nan)).isNaN);
    assert(logisticCCDF(T(-double.infinity)) == one);
    assert(logisticCCDF(T(double.infinity)) == zero);
    const maximum = T(false, long.max - 1, U.max);
    assert(logisticCCDF(maximum) == zero);
    const minimum = T(true, maximum.exponent, maximum.coefficient);
    assert(logisticCCDF(maximum, minimum, maximum) == T(false, -67, U(0xf420a95d618d754fUL)));
    const quantum = T(false, long.min, U(1));
    assert(logisticCCDF(T(6393154322601327830L)) == quantum);
    assert(logisticCCDF(T(6393154322601327831L)) == zero);
    assert(logisticCCDF(T(100)) == T(false, -208, U(0xd460f8a7157ae57aUL)));
    assert(logisticCCDF(T(true, 0, U(0))) == T(0.5));
}

// Shared dyadic observations have standardized offsets of exactly +/-2.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
    {{
        const small = R(0.119202922022117555940270858697603205L);
        const large = R(0.880797077977882444059729141302396795L);
        assert(approxEqual(logisticCDF(R(-5), R(1), R(3)), small, 8 * R.epsilon));
        assert(approxEqual(logisticCCDF(R(7), R(1), R(3)), small, 8 * R.epsilon));
        assert(approxEqual(logisticCDF(R(7), R(1), R(3)), large, 8 * R.epsilon));
        assert(approxEqual(logisticCCDF(R(-5), R(1), R(3)), large, 8 * R.epsilon));
    }}
    alias T = Fp!64;
    alias U = UInt!64;
    const small = T(false, -67, U(0xf420a95d618d754fUL));
    const large = T(false, -64, U(0xe17bead453ce5156UL));
    assert(logisticCDF(T(-5), T(1), T(3)) == small);
    assert(logisticCCDF(T(7), T(1), T(3)) == small);
    assert(logisticCDF(T(7), T(1), T(3)) == large);
    assert(logisticCCDF(T(-5), T(1), T(3)) == large);
}

// At z = t, the cubic term in 1/2 + z/4 - z^3/48 puts the
// result on the center side of a midpoint. Adding t^3 crosses it.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    enum b = 256;
    alias T = Fp!b;
    alias U = UInt!b;
    const half = T(0.5);
    auto successor = T(0.5);
    successor.coefficient += 1;
    const t = T(false, 1-b, U(1));
    assert(logisticCDF(t) == half);
    assert(logisticCDF(t, T(true, 3*(1-b), U(1)), T(1)) == successor);
    const h = T(false, -b, U(1));
    assert(logisticCCDF(h) == half);
    assert(logisticCCDF(h, T(true, -3*b, U(1)), T(1)) == T(false, -b-1, U.max));
}

/++
Computes the Logistic inverse cumulative distribution function (InvCDF).

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
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticInvCDF(T)(const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;

    return log(p / (1 - p));
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the quantile at cumulative probability $(D p).

The table assumes finite parameters and a positive scale.

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
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticInvCDF(T)(const T p, const T location, const T scale)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (scale > 0, "scale must be greater than zero")
{
    return location + scale * logisticInvCDF(p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    logisticInvCDF(0.0).shouldApprox == -double.infinity;
    logisticInvCDF(0.25).shouldApprox == -1.098612;
    logisticInvCDF(0.5).shouldApprox == 0.0;
    logisticInvCDF(0.75).shouldApprox == 1.098612;
    logisticInvCDF(1.0).shouldApprox == double.infinity;

    // Can also provide location/scale parameters
    logisticInvCDF(0.2, 2, 3).shouldApprox == -2.158883;
    logisticInvCDF(0.4, 2, 3).shouldApprox == 0.7836047;
    logisticInvCDF(0.6, 2, 3).shouldApprox == 3.216395;
    logisticInvCDF(0.8, 2, 3).shouldApprox == 6.158883;
}

/++
Computes the Logistic log probability density function (LPDF).

Returns the natural logarithm of the density.

This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == 0)) $(TD $(D -log(4))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate LPDF

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp, fabs;
    import mir.math.internal.log1p: log1p;

    const T magnitude = fabs(x);
    return -magnitude - 2 * log1p(exp(-magnitude));
}

// Evaluate tails without overflowing exp(-x), including representable small
// probabilities that would be lost by taking the reciprocal of exp(x).
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: exp, approxEqual;
    import std.math: isNaN;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T x = 1000;
        assert(logisticPDF(-x) == logisticPDF(x));
        assert(logisticLPDF(-x) == -x);
        assert(logisticLPDF(x) == -x);
        assert(logisticCCDF(-x) == 1);
        assert(logisticCDF(x) == 1);
        assert(logisticPDF(T.infinity) == 0);
        assert(logisticPDF(-T.infinity) == 0);
        assert(logisticCDF(-T.infinity) == 0);
        assert(logisticCCDF(-T.infinity) == 1);
        assert(logisticLPDF(-T.infinity) == -T.infinity);
        assert(isNaN(logisticPDF(T.nan)));
        assert(isNaN(logisticCDF(T.nan)));
        assert(isNaN(logisticCCDF(T.nan)));
        assert(isNaN(logisticLPDF(T.nan)));
        const T tail = T.max_exp * T(0.695);
        const T expected = exp(-tail);
        assert(expected > 0);
        assert(approxEqual(logisticCDF(-tail) / expected, T(1), T.epsilon * 8, T(0)));
        assert(approxEqual(logisticPDF(-tail) / expected, T(1), T.epsilon * 8, T(0)));
        assert(logisticCCDF(tail) == logisticCDF(-tail));
        assert(logisticPDF(-x, T(0), T(1)) == logisticPDF(-x));
        assert(logisticLPDF(-x, T(0), T(1)) == -x);
    }}
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the natural logarithm of the density.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == location)) $(TD $(D -log(4) - log(scale))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate LPDF
    location = finite location parameter
    scale = positive finite scale parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Logistic_distribution, Logistic Distribution)
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x, const T location, const T scale)
    if (isFloatingPoint!T)
    in (scale > 0, "scale must be greater than zero")
{
    import mir.math.common: log;

    return logisticLPDF(standardize(x, location, scale)) - log(scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    logisticLPDF(-2.0).shouldApprox == log(0.1049936);
    logisticLPDF(-1.0).shouldApprox == log(0.1966119);
    logisticLPDF(-0.5).shouldApprox == log(0.2350037);
    logisticLPDF(0.0).shouldApprox == log(0.25);
    logisticLPDF(0.5).shouldApprox == log(0.2350037);
    logisticLPDF(1.0).shouldApprox == log(0.1966119);
    logisticLPDF(2.0).shouldApprox == log(0.1049936);

    // Can also provide location/scale parameters
    logisticLPDF(-1.0, 2.0, 3.0).shouldApprox == log(0.06553731);
    logisticLPDF(1.0, 2.0, 3.0).shouldApprox == log(0.08106072);
    logisticLPDF(4.0, 2.0, 3.0).shouldApprox == log(0.07471913);
}

/++
Computes the natural logarithm of the density for the standard logistic distribution (location zero, scale one).

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.

Returns:
    The natural logarithm of the density in type $(D T), using guarded evaluation.
    The peak is minus the logarithm of four times the scale. Either infinity gives negative infinity and NaN propagates.
    Cancellation beyond the working precision can give zero.
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
{
    return logisticLPDF(x, T(0), T(1));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticLPDF(T(0)).sign);
}

/++
Computes the natural logarithm of the density for the logistic distribution with the given location and scale.

Params:
    T = Represented $(D Fp) type, inferred from the arguments.
    x = Observation, which may be infinite or NaN.
    location = Finite location parameter.
    scale = Positive finite scale parameter.

Returns:
    The natural logarithm of the density in type $(D T), using guarded evaluation.
    The peak is minus the logarithm of four times the scale. Either infinity gives negative infinity and NaN propagates.
    Cancellation beyond the working precision can give zero.
+/
@safe pure nothrow @nogc
T logisticLPDF(T)(const T x, const T location, const T scale)
    if (is(T == Fp!size, size_t size))
    in (!location.isSpecial, "location must be finite")
    in (!scale.isSpecial && !scale.sign && !!scale.coefficient, "scale must be finite and positive")
{
    import mir.stat.internal.fp_logistic: logisticLogDensity;
    return logisticLogDensity(x, location, scale);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!64;
    assert(logisticLPDF(T(3), T(3), T(2)).sign);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    import mir.math.common: approxEqual;
    import std.meta: AliasSeq;
    static foreach (R; AliasSeq!(float, double, real))
        assert(approxEqual(logisticLPDF(R(7), R(1), R(3)),
            R(-3.35246831075405468428269884963913456752L), 8 * R.epsilon));
    assert(logisticLPDF(T(7), T(1), T(3)) == T(true, -62, U(0xd68ed73ee4297375UL)));
    assert(logisticLPDF(T(0), T(0), T(0.25)) == T(0));
    assert(logisticLPDF(T(double.nan)).isNaN);
    assert(logisticLPDF(T(double.infinity)) == T(-double.infinity));
    const quantum = T(false, long.min, U(1));
    const huge = T(false, long.max - 1, U(2));
    assert(logisticLPDF(huge, T(0), quantum) == T(-double.infinity));
    assert(logisticLPDF(huge, huge, huge).sign);
    // At scale 1/4 the peak vanishes. The leading displacement is -4*x^2.
    const tiny = T(false, -1000, U(1));
    assert(logisticLPDF(tiny, T(0), T(0.25)) == T(true, -1998-63, U(1UL << 63)));
    // Here the second term of the log-cosh series fits the working grid.
    const small = T(false, -160, U(1));
    assert(logisticLPDF(small, T(0), T(0.25)) == T(true, -318-63, U(1UL << 63)));
    const high = T(false, 1000, U(1)), low = T(true, 936, U(1));
    assert(logisticLPDF(high, low, T(1)) == T(true, 937, U(0x8000000000000001UL)));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias U = UInt!128;
    // At scale 1/8 the positive root is log(1+sqrt(2))/4.
    // Check its two enclosing dyadic observations with independent logs.
    enum below = T(false, -130, U.fromHexString("e1a1b30bcea13660d8f99e8dd2518a53"));
    T above = below;
    above.coefficient += 1;
    enum positive = T(false, -255, U.fromHexString("911f25b58e6d78d71cf36598a7153461"));
    enum negative = T(true, -257, U.fromHexString("8f9735f9adc3aeb4f228906b38d628f9"));
    assert(logisticLPDF(below, T(0), T(0.125)) == positive);
    assert(logisticLPDF(above, T(0), T(0.125)) == negative);
}

// A finite standardized value survives an overflowing location subtraction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, fabs, log;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T scale = T.max * T(0.75);
        foreach (sign; [T(-1), T(1)])
        {
            const T x = sign * scale;
            const T location = -x;
            const T z = sign * 2;
            const T expected = logisticPDF(z) / scale;
            const T actual = logisticPDF(x, location, scale);
            assert(actual > 0);
            assert(fabs(actual - expected) <= 8 * T.min_normal * T.epsilon);
            assert(approxEqual(logisticLPDF(x, location, scale),
                logisticLPDF(z) - log(scale), 32 * T.epsilon, T(0)));
            assert(approxEqual(logisticCDF(x, location, scale),
                logisticCDF(z), 32 * T.epsilon, T(0)));
            assert(approxEqual(logisticCCDF(x, location, scale),
                logisticCCDF(z), 32 * T.epsilon, T(0)));
        }
    }}
}
