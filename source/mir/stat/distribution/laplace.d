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
    import mir.math.common: exp, log;
    const T z = standardizedLaplace(x, location, scale);
    const T density = laplacePDF(z);
    // Scaling can recover a representable result after the standard density
    // has underflowed or lost precision in the subnormal range.
    if (density < T.min_normal && scale < 1)
        return exp(laplaceLPDF(z) - log(scale));
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
    return laplaceCDF(standardizedLaplace(x, location, scale));
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
    return laplaceCCDF(standardizedLaplace(x, location, scale));
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

    return laplaceLPDF(standardizedLaplace(x, location, scale)) - log(scale);
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

// Divide first only when subtracting finite opposite-signed values overflows.
// In that case the divisions cannot cancel infinities of the same sign.
private @safe pure nothrow @nogc
T standardizedLaplace(T)(const T x, const T location, const T scale)
{
    import mir.math.common: fabs;
    const T delta = x - location;
    if (fabs(delta) == T.infinity && fabs(x) <= T.max && fabs(location) <= T.max)
        return x / scale - location / scale;
    return delta / scale;
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
