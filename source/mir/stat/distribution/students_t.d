/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution).

The two-argument overloads use location zero and scale one. $(D nu) is a
positive number of degrees of freedom and may be nonintegral. Passing
$(D nu == T.infinity) selects the normal-distribution limit.

In the four-argument overloads, $(D mean) names the location and $(D stdDev)
names the positive scale. The location is the median; the mean exists only
for $(D nu > 1). For finite $(D nu > 2), the standard deviation is
$(D stdDev * sqrt(nu / (nu - 2))). Thus the scale parameter is not generally
the standard deviation.

The special-value tables assume parameters in the stated domain. Finite
expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts
does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.students_t;

import mir.stat.internal.standardize: standardize, logDistance;

import mir.internal.utility: isFloatingPoint;

/++
Computes the Student's t probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.
This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate PDF
    nu = positive degrees of freedom; positive infinity selects the normal limit

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution)
+/
@safe pure nothrow @nogc
T studentsTPDF(T)(const T x, const T nu)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
{
    import mir.math.common: pow, sqrt;
    import mir.stat.constant: M_SQRTPI;
    import mir.stat.distribution.normal: normalPDF;
    import std.mathspecial: gamma;

    if (nu == T.infinity)
        return normalPDF(x);
    const T square = x * x;
    // Keep the direct formula for ordinary shapes and representable ratios.
    // Large shapes overflow gamma; extreme observations can overflow x*x.
    if (nu >= T.min_normal * 2 && nu < 64 && square <= T.max && square / nu <= T.max)
        return (gamma((nu + 1) * 0.5) / gamma(nu * 0.5)) / sqrt(nu) * T(M_SQRTPI) * pow(1 + square / nu, -(nu + 1) * 0.5);
    return studentsTPDFFallback(x, nu);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns a density, which may exceed one; it is not a point probability.
This overload uses a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate PDF
    nu = positive degrees of freedom; positive infinity selects the normal limit
    mean = finite location parameter (the mean exists only for nu > 1)
    stdDev = positive finite scale parameter (not generally the standard deviation)
+/
@safe pure nothrow @nogc
T studentsTPDF(T)(const T x, const T nu, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: exp;
    const T z = standardize(x, mean, stdDev);
    const T density = studentsTPDF(z, nu);
    if (density < T.min_normal && stdDev < 1)
        return exp(studentsTLPDF(x, nu, mean, stdDev));
    return density / stdDev;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTPDF(-3.0, 5).shouldApprox == 0.01729258;
    studentsTPDF(-2.0, 5).shouldApprox == 0.06509031;
    studentsTPDF(-1.0, 5).shouldApprox == 0.2196798;
    studentsTPDF(0.0, 5).shouldApprox == 0.3796067;
    studentsTPDF(1.0, 5).shouldApprox == 0.2196798;
    studentsTPDF(2.0, 5).shouldApprox == 0.06509031;
    studentsTPDF(3.0, 5).shouldApprox == 0.01729258;

    // Can include location/scale
    studentsTPDF(-3.0, 5, 1, 2).shouldApprox == (0.06509031) / 2;
    studentsTPDF(-2.0, 5, 1, 2).shouldApprox == (0.1245173) / 2;
    studentsTPDF(-1.0, 5, 1, 2).shouldApprox == (0.2196798) / 2;
    studentsTPDF(0.0, 5, 1, 2).shouldApprox == (0.3279185) / 2;
    studentsTPDF(1.0, 5, 1, 2).shouldApprox == (0.3796067) / 2;
    studentsTPDF(2.0, 5, 1, 2).shouldApprox == (0.3279185) / 2;
    studentsTPDF(3.0, 5, 1, 2).shouldApprox == (0.2196798) / 2;
}

// Checking other DoF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTPDF(-3.0, 25).shouldApprox == 0.007253748;
    studentsTPDF(-2.0, 25).shouldApprox == 0.0573607;
    studentsTPDF(-1.0, 25).shouldApprox == 0.237211;
    studentsTPDF(0.0, 25).shouldApprox == 0.3949738;
    studentsTPDF(1.0, 25).shouldApprox == 0.237211;
    studentsTPDF(2.0, 25).shouldApprox == 0.0573607;
    studentsTPDF(3.0, 25).shouldApprox == 0.007253748;

    studentsTPDF(-3.0, double.infinity).shouldApprox == 0.004431848;
    studentsTPDF(-2.0, double.infinity).shouldApprox == 0.05399097;
    studentsTPDF(-1.0, double.infinity).shouldApprox == 0.2419707;
    studentsTPDF(0.0, double.infinity).shouldApprox == 0.3989423;
    studentsTPDF(1.0, double.infinity).shouldApprox == 0.2419707;
    studentsTPDF(2.0, double.infinity).shouldApprox == 0.05399097;
    studentsTPDF(3.0, double.infinity).shouldApprox == 0.004431848;
}

// Checking negative location parameter
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTPDF(-3.0, 5, -1, 2).shouldApprox == (0.2196798) / 2;
    studentsTPDF(-2.0, 5, -1, 2).shouldApprox == (0.3279185) / 2;
    studentsTPDF(-1.0, 5, -1, 2).shouldApprox == (0.3796067) / 2;
    studentsTPDF(0.0, 5, -1, 2).shouldApprox == (0.3279185) / 2;
    studentsTPDF(1.0, 5, -1, 2).shouldApprox == (0.2196798) / 2;
    studentsTPDF(2.0, 5, -1, 2).shouldApprox == (0.1245173) / 2;
    studentsTPDF(3.0, 5, -1, 2).shouldApprox == (0.06509031) / 2;
}

/// The location is the median, including when the mean does not exist.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(studentsTCDF(3.0, 1.0, 3.0, 2.0) == 0.5);
    assert(studentsTInvCDF(0.5, 1.0, 3.0, 2.0) == 3.0);
    assert(studentsTInvCDF(0.0, 1.0) == -double.infinity);
    assert(studentsTInvCDF(1.0, 1.0) == double.infinity);
}

/++
Computes the Student's t cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).
This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate CDF
    nu = positive degrees of freedom; positive infinity selects the normal limit

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution)
+/
@safe pure nothrow @nogc
T studentsTCDF(T)(const T x, const T nu)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
{
    import mir.stat.distribution.beta: betaCDF, betaCCDF;
    import mir.stat.distribution.normal: normalCDF;

    import mir.math.common: fabs;
    if (x != x)
        return x;
    if (x == -T.infinity)
        return 0;
    if (x == T.infinity)
        return 1;
    if (nu < T.min_normal * 2)
        return T(0.5); // Finite observations have indistinguishable tails at this precision.
    if (nu != T.infinity && nu + x * x == T.infinity)
    {
        const T tail = studentsTExtremeTail(fabs(x), nu);
        return x > 0 ? 1 - tail : tail;
    }
    if (nu != T.infinity) {
        T output;
        if (nu > x * x) {
            output = betaCCDF(x * x / (nu + x * x), 0.5, 0.5 * nu);
        } else {
            const T z = nu / (nu + x * x);
            // Even a finite denominator can erase the small beta argument.
            // Its power-law tail may still be large when nu is small.
            if (z < T.min_normal)
            {
                const T tail = studentsTExtremeTail(fabs(x), nu);
                return x > 0 ? 1 - tail : tail;
            }
            output = betaCDF(z, 0.5 * nu, 0.5);
        }
        output *= 0.5;
        if (x > 0) {
            output = 1 - output;
        }
        return output;
    } else {
        return normalCDF(x);
    }
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the lower-tail probability $(D P(X <= x)).
This overload uses a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == mean)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate CDF
    nu = positive degrees of freedom; positive infinity selects the normal limit
    mean = finite location parameter (the mean exists only for nu > 1)
    stdDev = positive finite scale parameter (not generally the standard deviation)
+/
@safe pure nothrow @nogc
T studentsTCDF(T)(const T x, const T nu, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    const T z = standardize(x, mean, stdDev);
    return studentsTScaledCDF(z, x, nu, mean, stdDev, false);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCDF(-3.0, 5).shouldApprox == 0.01504962;
    studentsTCDF(-2.0, 5).shouldApprox == 0.05096974;
    studentsTCDF(-1.0, 5).shouldApprox == 0.1816087;
    studentsTCDF(0.0, 5).shouldApprox == 0.5;
    studentsTCDF(1.0, 5).shouldApprox == 0.8183913;
    studentsTCDF(2.0, 5).shouldApprox == 0.9490303;
    studentsTCDF(3.0, 5).shouldApprox == 0.9849504;

    // Can include location/scale
    studentsTCDF(-3.0, 5, 1, 2).shouldApprox == 0.05096974;
    studentsTCDF(-2.0, 5, 1, 2).shouldApprox == 0.09695184;
    studentsTCDF(-1.0, 5, 1, 2).shouldApprox == 0.1816087;
    studentsTCDF(0.0, 5, 1, 2).shouldApprox == 0.3191494;
    studentsTCDF(1.0, 5, 1, 2).shouldApprox == 0.5;
    studentsTCDF(2.0, 5, 1, 2).shouldApprox == 0.6808506;
    studentsTCDF(3.0, 5, 1, 2).shouldApprox == 0.8183913;
}

// Checking other DoF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCDF(-3.0, 25).shouldApprox == 0.00301909;
    studentsTCDF(-2.0, 25).shouldApprox == 0.02823799;
    studentsTCDF(-1.0, 25).shouldApprox == 0.163446;
    studentsTCDF(0.0, 25).shouldApprox == 0.5;
    studentsTCDF(1.0, 25).shouldApprox == 0.836554;
    studentsTCDF(2.0, 25).shouldApprox == 0.971762;
    studentsTCDF(3.0, 25).shouldApprox == 0.9969809;

    studentsTCDF(-3.0, double.infinity).shouldApprox == 0.001349898;
    studentsTCDF(-2.0, double.infinity).shouldApprox == 0.02275013;
    studentsTCDF(-1.0, double.infinity).shouldApprox == 0.1586553;
    studentsTCDF(0.0, double.infinity).shouldApprox == 0.5;
    studentsTCDF(1.0, double.infinity).shouldApprox == 0.8413447;
    studentsTCDF(2.0, double.infinity).shouldApprox == 0.9772499;
    studentsTCDF(3.0, double.infinity).shouldApprox == 0.9986501;
}

// Checking negative location parameter
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCDF(-3.0, 5, -1, 2).shouldApprox == 0.1816087;
    studentsTCDF(-2.0, 5, -1, 2).shouldApprox == 0.3191494;
    studentsTCDF(-1.0, 5, -1, 2).shouldApprox == 0.5;
    studentsTCDF(0.0, 5, -1, 2).shouldApprox == 0.6808506;
    studentsTCDF(1.0, 5, -1, 2).shouldApprox == 0.8183913;
    studentsTCDF(2.0, 5, -1, 2).shouldApprox == 0.9030482;
    studentsTCDF(3.0, 5, -1, 2).shouldApprox == 0.9490303;
}

/++
Computes the Student's t complementary cumulative distribution function (CCDF).

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
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate CCDF
    nu = positive degrees of freedom; positive infinity selects the normal limit

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution)
+/
@safe pure nothrow @nogc
T studentsTCCDF(T)(const T x, const T nu)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
{
    import mir.stat.distribution.normal: normalCCDF;

    return studentsTCDF(-x, nu);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.
This overload uses a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x == mean)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate CCDF
    nu = positive degrees of freedom; positive infinity selects the normal limit
    mean = finite location parameter (the mean exists only for nu > 1)
    stdDev = positive finite scale parameter (not generally the standard deviation)
+/
@safe pure nothrow @nogc
T studentsTCCDF(T)(const T x, const T nu, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    const T z = standardize(x, mean, stdDev);
    return studentsTScaledCDF(z, x, nu, mean, stdDev, true);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCCDF(-3.0, 5).shouldApprox == 0.9849504;
    studentsTCCDF(-2.0, 5).shouldApprox == 0.9490303;
    studentsTCCDF(-1.0, 5).shouldApprox == 0.8183913;
    studentsTCCDF(0.0, 5).shouldApprox == 0.5;
    studentsTCCDF(1.0, 5).shouldApprox == 0.1816087;
    studentsTCCDF(2.0, 5).shouldApprox == 0.05096974;
    studentsTCCDF(3.0, 5).shouldApprox == 0.01504962;

    // Can include location/scale
    studentsTCCDF(-3.0, 5, 1, 2).shouldApprox == 0.9490303;
    studentsTCCDF(-2.0, 5, 1, 2).shouldApprox == 0.9030482;
    studentsTCCDF(-1.0, 5, 1, 2).shouldApprox == 0.8183913;
    studentsTCCDF(0.0, 5, 1, 2).shouldApprox == 0.6808506;
    studentsTCCDF(1.0, 5, 1, 2).shouldApprox == 0.5;
    studentsTCCDF(2.0, 5, 1, 2).shouldApprox == 0.3191494;
    studentsTCCDF(3.0, 5, 1, 2).shouldApprox == 0.1816087;
}

// Checking other DoF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCCDF(-3.0, 25).shouldApprox == 0.9969809;
    studentsTCCDF(-2.0, 25).shouldApprox == 0.971762;
    studentsTCCDF(-1.0, 25).shouldApprox == 0.836554;
    studentsTCCDF(0.0, 25).shouldApprox == 0.5;
    studentsTCCDF(1.0, 25).shouldApprox == 0.163446;
    studentsTCCDF(2.0, 25).shouldApprox == 0.02823799;
    studentsTCCDF(3.0, 25).shouldApprox == 0.00301909;

    studentsTCCDF(-3.0, double.infinity).shouldApprox == 0.9986501;
    studentsTCCDF(-2.0, double.infinity).shouldApprox == 0.9772499;
    studentsTCCDF(-1.0, double.infinity).shouldApprox == 0.8413447;
    studentsTCCDF(0.0, double.infinity).shouldApprox == 0.5;
    studentsTCCDF(1.0, double.infinity).shouldApprox == 0.1586553;
    studentsTCCDF(2.0, double.infinity).shouldApprox == 0.02275013;
    studentsTCCDF(3.0, double.infinity).shouldApprox == 0.001349898;
}

// Checking negative location parameter
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTCCDF(-3.0, 5, -1, 2).shouldApprox == 0.8183913;
    studentsTCCDF(-2.0, 5, -1, 2).shouldApprox == 0.6808506;
    studentsTCCDF(-1.0, 5, -1, 2).shouldApprox == 0.5;
    studentsTCCDF(0.0, 5, -1, 2).shouldApprox == 0.3191494;
    studentsTCCDF(1.0, 5, -1, 2).shouldApprox == 0.1816087;
    studentsTCCDF(2.0, 5, -1, 2).shouldApprox == 0.09695184;
    studentsTCCDF(3.0, 5, -1, 2).shouldApprox == 0.05096974;
}

/++
Computes the Student's t inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).
This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    p = value to evaluate InvCDF
    nu = positive degrees of freedom; positive infinity selects the normal limit

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution)
+/
@safe pure nothrow @nogc
T studentsTInvCDF(T)(const T p, const T nu)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (nu > 0, "nu must be greater than zero")
{
    import mir.math.common: sqrt;
    import mir.stat.distribution.beta: betaInvCDF;
    import mir.stat.distribution.normal: normalInvCDF;

    if (nu < T.min_normal * 2)
        return p == T(0.5) ? T(0) : p < T(0.5) ? -T.infinity : T.infinity;
    if (p == 0) {
        return -T.infinity;
    } else if (p == 1) {
        return T.infinity;
    } else if (nu != T.infinity) {
        byte output_sign = void;
        T p_new = void;
        T output = void;
        if (p > 0.25 && p < 0.75) {
            if (p == 0.5) {
                return 0;
            }
            output_sign = -1;
            p_new = 1 - 2 * p;
            if (p > 0.5) {
                output_sign = 1;
                p_new *= -1;
            }
            output = betaInvCDF(p_new, 0.5, 0.5 * nu);
            if (output > T(0.5))
                return studentsTInverseTail(p < T(0.5) ? p : 1 - p, nu) * output_sign;
            output = sqrt(nu) * sqrt(output / (1 - output));
        } else {
            output_sign = -1;
            p_new = p;
            if (p_new > 0.5) {
                output_sign = 1;
                p_new = 1 - p_new;
            }
            p_new *= 2;
            output = betaInvCDF(p_new, 0.5 * nu, 0.5);
            if (output < T.min_normal)
                return studentsTInverseSmallTail(p < T(0.5) ? p : 1 - p, nu) * output_sign;
            output = (sqrt(nu) / sqrt(output)) * sqrt(1 - output);
        }
        return output_sign * output;
    } else {
        return normalInvCDF(p);
    }
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the quantile at cumulative probability $(D p).
This overload uses a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D mean)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    p = value to evaluate InvCDF
    nu = positive degrees of freedom; positive infinity selects the normal limit
    mean = finite location parameter (the mean exists only for nu > 1)
    stdDev = positive finite scale parameter (not generally the standard deviation)
+/
@safe pure nothrow @nogc
T studentsTInvCDF(T)(const T p, const T nu, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (nu > 0, "nu must be greater than zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: fabs;
    const T q = studentsTInvCDF(p, nu);
    if (fabs(q) == T.infinity && p > 0 && p < 1 && nu >= T.min_normal * 2 && nu <= T.max && stdDev < 1)
    {
        import mir.math.common: exp, log;
        // A standard quantile can overflow although its scaled value fits.
        // Such an extreme finite-nu quantile is in the logarithmic tail.
        const T tail = p < T(0.5) ? p : 1 - p;
        const T logOffset = studentsTInverseSmallTailLog(tail, nu) + log(stdDev);
        const T offset = exp(logOffset);
        if (offset == T.infinity && fabs(mean) <= T.max)
        {
            // The location can cancel an offset larger than T.max. Form
            // half of each term first, then restore the final magnitude.
            const T halfOffset = exp(logOffset - T(0.693147180559945309417232121458176568L));
            return (mean * T(0.5) + (p < T(0.5) ? -halfOffset : halfOffset)) * 2;
        }
        return mean + (p < T(0.5) ? -offset : offset);
    }
    const T offset = stdDev * q;
    if (fabs(offset) == T.infinity && fabs(q) <= T.max &&
        stdDev <= T.max && fabs(mean) <= T.max)
        return ((q * T(0.5)) * stdDev + mean * T(0.5)) * 2;
    return mean + offset;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTInvCDF(0.0, 5).shouldApprox == -double.infinity;
    studentsTInvCDF(0.1, 5).shouldApprox == -1.475884;
    studentsTInvCDF(0.2, 5).shouldApprox == -0.9195438;
    studentsTInvCDF(0.3, 5).shouldApprox == -0.5594296;
    studentsTInvCDF(0.4, 5).shouldApprox == -0.2671809;
    studentsTInvCDF(0.5, 5).shouldApprox == 0.0;
    studentsTInvCDF(0.6, 5).shouldApprox == 0.2671809;
    studentsTInvCDF(0.7, 5).shouldApprox == 0.5594296;
    studentsTInvCDF(0.8, 5).shouldApprox == 0.9195438;
    studentsTInvCDF(0.9, 5).shouldApprox == 1.475884;
    studentsTInvCDF(1.0, 5).shouldApprox == double.infinity;

    // Can include location/scale
    studentsTInvCDF(0.2, 5, 1, 2).shouldApprox == -0.8390876;
    studentsTInvCDF(0.4, 5, 1, 2).shouldApprox == 0.4656382;
    studentsTInvCDF(0.6, 5, 1, 2).shouldApprox == 1.534362;
    studentsTInvCDF(0.8, 5, 1, 2).shouldApprox == 2.839088;
}

// Checking other DoF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTInvCDF(0.1, 25).shouldApprox == -1.316345;
    studentsTInvCDF(0.2, 25).shouldApprox == -0.8562362;
    studentsTInvCDF(0.3, 25).shouldApprox == -0.5311538;
    studentsTInvCDF(0.7, 25).shouldApprox == 0.5311538;
    studentsTInvCDF(0.8, 25).shouldApprox == 0.8562362;
    studentsTInvCDF(0.9, 25).shouldApprox == 1.316345;

    studentsTInvCDF(0.2, double.infinity).shouldApprox == -0.8416212;
    studentsTInvCDF(0.4, double.infinity).shouldApprox == -0.2533471;
    studentsTInvCDF(0.6, double.infinity).shouldApprox == 0.2533471;
    studentsTInvCDF(0.8, double.infinity).shouldApprox == 0.8416212;
}

// Checking negative location parameter
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    studentsTInvCDF(0.2, 5, -1, 2).shouldApprox == -2.839088;
    studentsTInvCDF(0.4, 5, -1, 2).shouldApprox == -1.534362;
    studentsTInvCDF(0.6, 5, -1, 2).shouldApprox == -0.4656383;
    studentsTInvCDF(0.8, 5, -1, 2).shouldApprox == 0.8390876;
}


/++
Computes the Student's t log probability density function (LPDF).

Returns the natural logarithm of the density.
This overload uses location zero and scale one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate LPDF
    nu = positive degrees of freedom; positive infinity selects the normal limit

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Student%27s_t-distribution, Student's t Distribution)
+/
@safe pure nothrow @nogc
T studentsTLPDF(T)(const T x, const T nu)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
{
    import mir.math.common: log;
    import mir.math.internal.log1p: log1p;
    import mir.stat.constant: LOGPI;
    import std.mathspecial: logGamma;
    const T ratio = (x * x) / nu;
    if (nu >= 2 && nu < 64 && ratio >= T.min_normal && ratio <= T.max)
        return logGamma((nu + 1) * 0.5) - logGamma(nu * 0.5)
            - 0.5 * (log(nu) + T(LOGPI) + (nu + 1) * log1p(ratio));
    return studentsTLPDFFallback(x, nu);
}

/++
Ditto, with location and scale parameters (by standardizing `x`).

Returns the natural logarithm of the density.
This overload uses a finite location and positive finite scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
    $(TR $(TD $(D nu == T.infinity)) $(TD Corresponding normal-distribution result))
)

Params:
    x = value to evaluate LPDF
    nu = positive degrees of freedom; positive infinity selects the normal limit
    mean = finite location parameter (the mean exists only for nu > 1)
    stdDev = positive finite scale parameter (not generally the standard deviation)
+/
@safe pure nothrow @nogc
T studentsTLPDF(T)(const T x, const T nu, const T mean, const T stdDev)
    if (isFloatingPoint!T)
    in (nu > 0, "nu must be greater than zero")
    in (stdDev > 0, "stdDev must be greater than zero")
{
    import mir.math.common: fabs, log;
    const T z = standardize(x, mean, stdDev);
    if (fabs(z) == T.infinity && fabs(x) <= T.max && fabs(mean) <= T.max && nu <= T.max)
    {
        // A tiny scale can overflow z even though the scaled heavy-tail
        // density remains representable. Evaluate its exponent in logs.
        const T logZ = logDistance(x, mean) - log(stdDev);
        return studentsTLogNormalization(nu)
            - (nu * T(0.5) + T(0.5)) * (2 * logZ - log(nu)) - log(stdDev);
    }
    return studentsTLPDF(z, nu) - log(stdDev);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: log;

    studentsTLPDF(-3.0, 5).shouldApprox == log(0.01729258);
    studentsTLPDF(-2.0, 5).shouldApprox == log(0.06509031);
    studentsTLPDF(-1.0, 5).shouldApprox == log(0.2196798);
    studentsTLPDF(0.0, 5).shouldApprox == log(0.3796067);
    studentsTLPDF(1.0, 5).shouldApprox == log(0.2196798);
    studentsTLPDF(2.0, 5).shouldApprox == log(0.06509031);
    studentsTLPDF(3.0, 5).shouldApprox == log(0.01729258);

    // Can include location/scale
    studentsTLPDF(-3.0, 5, 1, 2).shouldApprox == log((0.06509031) / 2);
    studentsTLPDF(-2.0, 5, 1, 2).shouldApprox == log((0.1245173) / 2);
    studentsTLPDF(-1.0, 5, 1, 2).shouldApprox == log((0.2196798) / 2);
    studentsTLPDF(0.0, 5, 1, 2).shouldApprox == log((0.3279185) / 2);
    studentsTLPDF(1.0, 5, 1, 2).shouldApprox == log((0.3796067) / 2);
    studentsTLPDF(2.0, 5, 1, 2).shouldApprox == log((0.3279185) / 2);
    studentsTLPDF(3.0, 5, 1, 2).shouldApprox == log((0.2196798) / 2);
}

// Checking other DoF
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: log;

    studentsTLPDF(-3.0, 25).shouldApprox == log(0.007253748);
    studentsTLPDF(-2.0, 25).shouldApprox == log(0.0573607);
    studentsTLPDF(-1.0, 25).shouldApprox == log(0.237211);
    studentsTLPDF(0.0, 25).shouldApprox == log(0.3949738);
    studentsTLPDF(1.0, 25).shouldApprox == log(0.237211);
    studentsTLPDF(2.0, 25).shouldApprox == log(0.0573607);
    studentsTLPDF(3.0, 25).shouldApprox == log(0.007253748);

    studentsTLPDF(-3.0, double.infinity).shouldApprox == log(0.004431848);
    studentsTLPDF(-2.0, double.infinity).shouldApprox == log(0.05399097);
    studentsTLPDF(-1.0, double.infinity).shouldApprox == log(0.2419707);
    studentsTLPDF(0.0, double.infinity).shouldApprox == log(0.3989423);
    studentsTLPDF(1.0, double.infinity).shouldApprox == log(0.2419707);
    studentsTLPDF(2.0, double.infinity).shouldApprox == log(0.05399097);
    studentsTLPDF(3.0, double.infinity).shouldApprox == log(0.004431848);
}

// Checking negative location parameter
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    import mir.math.common: log;

    studentsTLPDF(-3.0, 5, -1, 2).shouldApprox == log((0.2196798) / 2);
    studentsTLPDF(-2.0, 5, -1, 2).shouldApprox == log((0.3279185) / 2);
    studentsTLPDF(-1.0, 5, -1, 2).shouldApprox == log((0.3796067) / 2);
    studentsTLPDF(0.0, 5, -1, 2).shouldApprox == log((0.3279185) / 2);
    studentsTLPDF(1.0, 5, -1, 2).shouldApprox == log((0.2196798) / 2);
    studentsTLPDF(2.0, 5, -1, 2).shouldApprox == log((0.1245173) / 2);
    studentsTLPDF(3.0, 5, -1, 2).shouldApprox == log((0.06509031) / 2);
}

// A change of scale must preserve unit total probability, including the normal limit.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, log;
    static foreach (T; AliasSeq!(float, double, real))
    {
        foreach (nu; [T(1), T(5), T.infinity])
            foreach (scale; [T(0.5), T(2), T(8)])
            {
                const density = studentsTPDF(T(0), nu);
                assert(approxEqual(studentsTPDF(T(3), nu, T(3), scale), density / scale));
                assert(approxEqual(studentsTLPDF(T(3), nu, T(3), scale),
                    studentsTLPDF(T(0), nu) - log(scale)));
            }
        // Independent reference: Gamma(3)/(sqrt(5*pi)*Gamma(5/2))/2.
        assert(approxEqual(studentsTPDF(T(1), T(5), T(1), T(2)),
            T(0.189803344911247215593803380374347484L)));
    }
}
// log(Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2))).
// Use a gamma recurrence near zero, and the log gamma-ratio expansion at
// large nu rather than subtracting nearly equal log-gamma values. At nu=64,
// the first omitted term (-5461/(52*nu^13)) is below 4e-22.
private @safe pure nothrow @nogc
T studentsTLogNormalization(T)(const T nu)
{
    import mir.math.common: log;
    import mir.stat.constant: LOGPI, LOGSQRT2PI;
    import std.mathspecial: logGamma;
    if (nu >= 64)
    {
        const T r = 1 / nu;
        const T r2 = r * r;
        return -T(LOGSQRT2PI) + r * (-T(0.25) + r2 * (T(1)/24
            + r2 * (-T(1)/20 + r2 * (T(17)/112
            + r2 * (-T(31)/36 + r2 * (T(691)/88))))));
    }
    if (nu < 2)
        return logGamma(T(0.5) + nu * T(0.5)) - logGamma(1 + nu * T(0.5))
            + T(0.5) * log(nu) - T(0.693147180559945309417232121458176568L) - T(0.5) * T(LOGPI);
    return logGamma((nu + 1) * T(0.5)) - logGamma(nu * T(0.5))
        - T(0.5) * (log(nu) + T(LOGPI));
}

// Independent 100-digit log-gamma references and the nu=1 Cauchy identity.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {
        assert(approxEqual(studentsTPDF(T(0), T(40000)),
            T(0.398939787019972253893245481608949881L), T(64)*T.epsilon));
        assert(approxEqual(studentsTLPDF(T(0), T(1e20)),
            T(-0.91893853320467274178282973640561764L), T(64)*T.epsilon));
        assert(approxEqual(studentsTLPDF(T(1), T.max),
            T(-1.41893853320467274178032973640561764L), T(64)*T.epsilon));
        assert(studentsTPDF(T.infinity, T(5)) == 0);
        assert(studentsTLPDF(T.infinity, T(5)) == -T.infinity);
    }
    assert(approxEqual(studentsTLPDF(1e155, 1.0), -714.946108714003562234L, 1e-14));
    const density = studentsTPDF(1e155, 1.0);
    assert(density > 0 && approxEqual(density / cast(double)3.18309886183790666969e-311L, 1.0, 1e-12));
}
private @safe pure nothrow @nogc
T studentsTLPDFFallback(T)(const T x, const T nu)
{
    pragma(inline, false);
    import mir.math.common: log;
    import mir.math.internal.log1p: log1p;
    import mir.stat.constant: LOGPI;
    import mir.stat.distribution.normal: normalLPDF;
    import std.mathspecial: logGamma;

    import mir.math.common: fabs, sqrt;
    if (nu == T.infinity)
        return normalLPDF(x);
    const T square = x * x;
    T ratio = square / nu;
    if (ratio == T.infinity || (x != 0 && nu < 1 && square < T.min_normal))
    {
        // A subnormal square can lose significant bits before division by
        // a tiny nu restores the ratio. Rescale before squaring instead.
        const T scaled = fabs(x) / sqrt(nu);
        ratio = scaled * scaled;
    }
    T decay;
    if (ratio == T.infinity)
    {
        // x is finite here for finite log-density; true infinite x still
        // gives infinite decay. Taking logs avoids overflowing its square.
        decay = (nu * T(0.5) + T(0.5)) * (2 * log(fabs(x)) - log(nu));
    }
    else if (ratio < T.min_normal)
    {
        // log1p(ratio) equals ratio to working precision. Multiply before
        // division so large nu does not erase the normal-limit exponent.
        decay = square * T(0.5) + ratio * T(0.5);
    }
    else
        decay = (nu * T(0.5) + T(0.5)) * log1p(ratio);
    return studentsTLogNormalization(nu) - decay;
}

private @safe pure nothrow @nogc
T studentsTPDFFallback(T)(const T x, const T nu)
{
    pragma(inline, false);
    import mir.math.common: exp;
    return exp(studentsTLPDF(x, nu));
}
// Used when nu+x*x overflows or its beta argument becomes subnormal.
// Rescale before forming the beta argument;
// if that argument underflows, integrate its leading small-z term in logs.
private @safe pure nothrow @nogc
T studentsTExtremeTail(T)(const T x, const T nu)
{
    pragma(inline, false);
    import mir.math.common: sqrt, log, exp;
    import mir.math.internal.log1p: log1p;
    import mir.stat.distribution.beta: betaCDF;
    const T r = sqrt(nu) / x;
    const T r2 = r * r;
    const T z = r2 / (1 + r2);
    if (z >= T.min_normal)
        return T(0.5) * betaCDF(z, nu * T(0.5), T(0.5));
    const T logZ = log(nu) - 2 * log(x) - log1p(r2);
    return exp((nu * T(0.5)) * logZ + studentsTLogTailNormalization(nu));
}

// For a subnormal beta quantile z, relative rounding of z contaminates the
// t quantile after the square root. Invert the leading tail in logarithms.
// The omitted relative correction is O(z), below working precision here.
private @safe pure nothrow @nogc
T studentsTInverseSmallTail(T)(const T p, const T nu)
{
    import mir.math.common: exp;
    return exp(studentsTInverseSmallTailLog(p, nu));
}

// The central beta formulation can round to one for small nu. Use the
// complementary beta quantile instead, retaining a small positive argument.
private @safe pure nothrow @nogc
T studentsTInverseTail(T)(const T p, const T nu)
{
    pragma(inline, false);
    import mir.math.common: sqrt;
    import mir.stat.distribution.beta: betaInvCDF;
    const T z = betaInvCDF(2 * p, nu * T(0.5), T(0.5));
    if (z < T.min_normal)
        return studentsTInverseSmallTail(p, nu);
    return (sqrt(nu) / sqrt(z)) * sqrt(1 - z);
}

// The nu=1 distribution is Cauchy: its tail is atan(1/x)/pi and its
// inverse is -cot(pi*p). References use 100-digit evaluation of those identities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(approxEqual(studentsTCDF(-1e155, 1.0), 3.183098861837906692535e-156, 2e-13, 0.0));
    assert(approxEqual(studentsTCCDF(1e155, 1.0), 3.183098861837906692535e-156, 2e-13, 0.0));
    assert(approxEqual(studentsTInvCDF(1e-160, 1.0), -3.183098861837906751548e159, 2e-13, 0.0));
    assert(studentsTCDF(-double.infinity, 5.0) == 0);
    assert(studentsTCDF(double.infinity, 5.0) == 1);
    assert(studentsTCDF(double.nan, 5.0) != studentsTCDF(double.nan, 5.0));
}
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {
        assert(studentsTInvCDF(T(0.5), T.min_normal / 2) == 0);
        assert(studentsTInvCDF(T(0.25), T.min_normal / 2) == -T.infinity);
        assert(studentsTInvCDF(T(0.75), T.min_normal / 2) == T.infinity);
        assert(studentsTCDF(T(1), T.min_normal / 2) == T(0.5));
        foreach (p; [T(0.1), T(0.25), T(0.3), T(0.5), T(0.7), T(0.75), T(0.9)])
            assert(approxEqual(studentsTCDF(studentsTInvCDF(p, T(1)), T(1)), p,
                T(128)*T.epsilon, T(128)*T.epsilon));
    }
    // Both sides of the subnormal beta-quantile transition for nu=1.
    foreach (p; [1e-150, 1e-154, 1e-158, 1e-160])
        assert(approxEqual(-studentsTInvCDF(p, 1.0) * p,
            0.318309886183790671537767526745028724, 2e-13));
    // nu=2 has the exact quantile (2p-1)/sqrt(2p(1-p)).
    assert(approxEqual(studentsTInvCDF(1e-160, 2.0), -7.071067811865475237e79, 2e-13));
}


// Overflow in an intermediate location/scale operation need not imply an
// infinite standardized value or an unrepresentable final density/quantile.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual, log;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {
        {
            const T large = T.max * T(0.75);
            assert(approxEqual(studentsTCDF(large, T(1), -large, large),
                T(0.852416382349566725824598923775259L), T(128)*T.epsilon));
            assert(approxEqual(studentsTCCDF(large, T(1), -large, large),
                T(0.147583617650433274175401076224741L), T(128)*T.epsilon));
            assert(approxEqual(studentsTLPDF(large, T(1), -large, large),
                studentsTLPDF(T(2), T(1)) - log(large), T(128)*T.epsilon));
            // cot(pi/8)=1+sqrt(2); multiplying by large overflows, but
            // adding -large leaves (sqrt(2)*large), which fits for this scale.
            const T scale = T.max * T(0.5);
            assert(approxEqual(studentsTInvCDF(T(0.875), T(1), -scale, scale) / scale,
                T(1.414213562373095048801688724209698L), T(256)*T.epsilon));
            // No representable scale can recover the quantiles at this shape.
            assert(studentsTInvCDF(T(0.25), T.min_normal / 2, T(0), T.min_normal) == -T.infinity);
            assert(studentsTLPDF(large, T(1), -large, T(0.5)) > -T.infinity);
            assert(studentsTLPDF(large, T(1), -T.max, T(0.5)) > -T.infinity);
            assert(studentsTLPDF(T.max, T(1), -large, T(0.5)) > -T.infinity);
        }
    }
    // Cauchy density: scale/(pi*(x*x+scale*scale)); the standardized
    // density underflows before division by scale restores the result.
    assert(approxEqual(studentsTPDF(1.0, 1.0, 0.0, 1e-200) / 1e-200,
        0.31830988618379067154, 2e-13));
    assert(approxEqual(studentsTLPDF(1.0, 1.0, 0.0, cast(double)1e-320L),
        -737.971970776823314, 2e-14));
    assert(studentsTPDF(1.0, 1.0, 0.0, cast(double)1e-320L) > 0);
    assert(studentsTCCDF(1.0, 1.0, 0.0, cast(double)1e-320L) > 0);
    assert(studentsTCDF(-1.0, 1.0, 0.0, cast(double)1e-320L) > 0);
    // Exact binary p and scale are the same: -scale*cot(pi*p) ~ -1/pi.
    assert(approxEqual(studentsTInvCDF(cast(double)1e-320L, 1.0, 0.0, cast(double)1e-320L),
        -0.31830988618379067154, 2e-13));
}

private @safe pure nothrow @nogc
T studentsTInverseSmallTailLog(T)(const T p, const T nu)
{
    import mir.math.common: log;
    const T halfLogNu = T(0.5) * log(nu);
    return halfLogNu + (studentsTLogNormalization(nu) - halfLogNu - log(p)) / nu;
}

private @safe pure nothrow @nogc
T studentsTScaledCDF(T)(const T z, const T x, const T nu,
    const T mean, const T scale, bool complement)
{
    import mir.math.common: fabs, exp, log;
    if (fabs(z) == T.infinity && fabs(x) <= T.max && fabs(mean) <= T.max && nu <= T.max)
    {
        const T logZ = logDistance(x, mean) - log(scale);
        // z is beyond the finite range, so the leading power-law tail
        // differs from the complete beta integral by less than precision.
        const T tail = exp((nu * T(0.5)) * (log(nu) - 2 * logZ)
            + studentsTLogTailNormalization(nu));
        return (z > 0) != complement ? 1 - tail : tail;
    }
    return studentsTCDF(complement ? -z : z, nu);
}

// Central probabilities can still require the complementary beta formula
// at small degrees of freedom. References invert a 100-digit beta integral
// by bisection in log(x), independently of the production inverse routine.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(approxEqual(studentsTInvCDF(0.3, 0.1), -27.170126296735762052108, 2e-13));
    assert(approxEqual(studentsTInvCDF(0.3, 0.001), -1.11660119096013474742e220, 2e-12));
    assert(approxEqual(studentsTInvCDF(0.7, 0.1), 27.170126296735762052108, 2e-13));
    // The rescaled beta argument remains normal even though x*x overflows.
    // The tail here is smaller than the smallest representable double.
    assert(studentsTCDF(-1e155, 1e5) == 0);
}

// The leading beta-tail coefficient tends to 1/2 as nu approaches zero.
// Avoid cancelling two large log(nu)/2 terms in that limit. Omitting its
// O(nu) correction for nu < epsilon changes the result by less than epsilon.
private @safe pure nothrow @nogc
T studentsTLogTailNormalization(T)(const T nu)
{
    import mir.math.common: log;
    if (nu < T.epsilon)
        return -T(0.693147180559945309417232121458176568L);
    return studentsTLogNormalization(nu) - T(0.5) * log(nu);
}

// Both sides of x*x overflow must retain the same small-nu tail. For these
// exact binary inputs, 100-digit beta integrals round to 0.5 as doubles.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(studentsTCDF(-1e154, 1e-20) == 0.5);
    assert(studentsTCDF(-2e154, 1e-20) == 0.5);
    assert(studentsTCCDF(1e154, 1e-20) == 0.5);
    assert(studentsTCCDF(2e154, 1e-20) == 0.5);
    assert(studentsTCDF(1e154, 1e-20) == 0.5);
}

// For nu=1, mean-scale*cot(pi*p) gives an independent Cauchy reference.
// Both the standard quantile and its scaled offset overflow, but cancellation
// with the location leaves a finite result. Inputs are exact binary doubles
// in the 100-digit reference calculation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    const double p = cast(double)1e-320L;
    assert(approxEqual(studentsTInvCDF(p, 1.0, double.max, 1e-11),
        -1.38544116422823804939e308, 5e-13, 0.0));
    // Neither a same-signed location nor an insufficient cancellation may
    // turn a truly out-of-range result into a finite quantile.
    assert(studentsTInvCDF(p, 1.0, -double.max, 1e-11) == -double.infinity);
    assert(studentsTInvCDF(p, 1.0, double.max, 1e-10) == -double.infinity);
}

// Tiny degrees of freedom can make an underflowed square significant again.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown, nextUp;
    import mir.math.common: approxEqual, fabs, log, sqrt;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (nu; [T.min_normal * T.epsilon, T.min_normal / 2, T.min_normal * 2])
        {
            const T root = sqrt(nu);
            // At these nu, the normalization is sqrt(nu)/2 to working
            // precision. Its logarithmic evaluation sets the error scale.
            const T tolerance = 8 * T.epsilon * fabs(log(nu));
            foreach (factor; [T(0), T(.25), T(.5), T(1), T(1.5), T(2)])
            {
                const T x = root * factor;
                const T expected = (root / 2) / sqrt(1 + factor * factor);
                foreach (v; [x, -x])
                {
                    assert(approxEqual(studentsTPDF(v, nu), expected, tolerance, T(0)));
                    assert(fabs(studentsTLPDF(v, nu) - log(expected)) <= tolerance);
                }
            }
        }
        // Check both sides of the normal/subnormal square boundary.
        const T root = sqrt(T.min_normal);
        const T nu = T.min_normal / 2;
        foreach (x; [nextDown(root), root, nextUp(root)])
        {
            const T factor = x / root;
            const T expected = (sqrt(nu) / 2) / sqrt(1 + 2 * factor * factor);
            assert(approxEqual(studentsTPDF(x, nu), expected,
                8 * T.epsilon * fabs(log(nu)), T(0)));
        }
    }}
    // 100-digit reference for exact binary x=2^-538 and nu=2^-1074.
    assert(approxEqual(studentsTPDF(0x1p-538, 0x1p-1074),
        9.940479322862117841371961127712731866e-163, 2e-13, 0.0));
    assert(fabs(studentsTLPDF(0x1p-538, 0x1p-1074)
        - (-373.02475491690768134435402888965391)) < 2e-13);
}
