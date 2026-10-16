/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution).

These functions describe the type-I Pareto distribution. $(D xMin) is the
positive lower support boundary and $(D alpha) is the positive shape parameter.
The forward functions require $(D x >= xMin); they do not extend below the
support. The inverse CDF accepts both probability endpoints.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.pareto;

import mir.internal.utility: isFloatingPoint;

/++
Computes the pareto probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D alpha / xMin)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoPDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp, log;

    const T decay = alpha * paretoLogRatio(x, xMin);
    const T factor = alpha / x;
    const T tail = exp(-decay);
    // Avoid separate powers of xMin and x. If either factor loses range,
    // combine their logarithms before recovering the final density.
    if (factor >= T.min_normal && factor <= T.max && tail >= T.min_normal)
        return factor * tail;
    return exp(log(alpha) - log(x) - decay);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoPDF(1, 3).shouldApprox == 3;
    2.0.paretoPDF(1, 3).shouldApprox == 0.1875;
    3.0.paretoPDF(2, 4).shouldApprox == 0.2633745;
}

/// The density starts at the lower support boundary and tends to zero.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(paretoPDF(2.0, 2.0, 4.0) == 2.0);
    assert(paretoPDF(double.infinity, 2.0, 4.0) == 0.0);
}

/++
Computes the pareto cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D 0)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoCDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;

    return oneMinusExpNeg(alpha * paretoLogRatio(x, xMin));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoCDF(1, 3).shouldApprox == 0;
    2.0.paretoCDF(1, 3).shouldApprox == 0.875;
    3.0.paretoCDF(2, 4).shouldApprox == 0.8024691;
}

/++
Computes the pareto complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D 1)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoCCDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp;

    return exp(-alpha * paretoLogRatio(x, xMin));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    1.0.paretoCCDF(1, 3).shouldApprox == 1;
    2.0.paretoCCDF(1, 3).shouldApprox == 0.125;
    3.0.paretoCCDF(2, 4).shouldApprox == 0.1975309;
}

/++
Computes the pareto inverse cumulative distribution function (InvCDF).

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D xMin)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoInvCDF(T)(const T p, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: exp, log;
    import mir.stat.internal.neg_log1m: negLog1m;

    if (p == 0)
        return xMin;
    const T growth = negLog1m(p) / alpha;
    const T factor = exp(growth);
    if (factor <= T.max)
        return xMin * factor;
    // A small xMin can recover a finite quantile after exp(growth) overflows.
    return exp(log(xMin) + growth);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.0.paretoInvCDF(1, 3).shouldApprox == 1;
    0.875.paretoInvCDF(1, 3).shouldApprox == 2;
    0.8024691.paretoInvCDF(2, 4).shouldApprox == 3;
}

/++
Computes the pareto log probability density function (LPDF).

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == xMin)) $(TD $(D log(alpha) - log(xMin))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x < xMin) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    xMin = positive lower support boundary (scale parameter)
    alpha = positive shape parameter

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Pareto_distribution, Pareto Distribution)
+/
@safe pure nothrow @nogc
T paretoLPDF(T)(const T x, const T xMin, const T alpha)
    if (isFloatingPoint!T)
    in (x >= xMin, "x must be greater than or equal to xMin")
    in (xMin > 0, "xMin must be greater than zero")
    in (alpha > 0, "alpha must be greater than zero")
{
    import mir.math.common: log;

    const T factor = alpha / x;
    const T normalization = factor >= T.min_normal && factor <= T.max
        ? log(factor) : log(alpha) - log(x);
    return normalization - alpha * paretoLogRatio(x, xMin);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    1.0.paretoLPDF(1, 3).shouldApprox == log(paretoPDF(1.0, 1, 3));
    2.0.paretoLPDF(1, 3).shouldApprox == log(paretoPDF(2.0, 1, 3));
    3.0.paretoLPDF(2, 4).shouldApprox == log(paretoPDF(3.0, 2, 4));
}

// Near the lower bound, subtraction is exact. Evaluate log(x/xMin) as
// -log(1 - (x-xMin)/x) to preserve the small relative distance using the
// existing corrected logarithm. Farther away, use a representable ratio;
// separate logarithms avoid overflow across very different scales.
private @safe pure nothrow @nogc
T paretoLogRatio(T)(const T x, const T xMin)
{
    import mir.math.common: log;
    import mir.stat.internal.neg_log1m: negLog1m;
    const T distance = x - xMin;
    if (distance <= xMin)
        return negLog1m(distance / x);
    const T ratio = x / xMin;
    if (ratio <= T.max)
        return log(ratio);
    return log(x) - log(xMin);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, exp, log, sqrt;
    import std.math: nextDown, nextUp;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(paretoPDF(T(2), T(2), T(4)) == 2);
        assert(paretoCDF(T(2), T(2), T(4)) == 0);
        assert(paretoCCDF(T(2), T(2), T(4)) == 1);
        assert(paretoInvCDF(T(0), T(2), T(4)) == 2);
        assert(paretoInvCDF(T(1), T(2), T(4)) == T.infinity);
        assert(paretoPDF(T.infinity, T(2), T(4)) == 0);
        assert(paretoCDF(T.infinity, T(2), T(4)) == 1);
        assert(paretoCCDF(T.infinity, T(2), T(4)) == 0);
        assert(paretoLPDF(T.infinity, T(2), T(4)) == -T.infinity);

        const T large = sqrt(T.max);
        assert(approxEqual(paretoPDF(large, large, T(4)), T(4)/large, T(8)*T.epsilon, T(0)));
        assert(approxEqual(paretoLPDF(T(2), T(2), large), log(large)-log(T(2)), T(8)*T.epsilon));
        // A tiny shape makes the CDF small even well away from the boundary.
        const T tiny = T.epsilon / 16;
        assert(approxEqual(paretoCDF(T(2), T(1), tiny), tiny*log(T(2)), T(8)*T.epsilon, T(0)));
        assert(approxEqual(paretoInvCDF(tiny, T(1), tiny), exp(T(1)), T(8)*T.epsilon));
        // A large shape amplifies errors in ratios rounded near one.
        assert(approxEqual(paretoCCDF(T(1)+T.epsilon, T(1), 1/T.epsilon),
            exp(T(-1)), T(8)*T.epsilon));
        assert(approxEqual(paretoLPDF(T(1), T(1), T(1)+T.epsilon),
            T.epsilon, T(8)*T.epsilon, T(0)));
        // Both logarithm paths agree around x/xMin == 2. For alpha == 1,
        // the probabilities have simple rational references.
        foreach (x; [nextDown(T(2)), T(2), nextUp(T(2))])
        {
            assert(approxEqual(paretoCDF(x, T(1), T(1)), (x-1)/x, T(8)*T.epsilon));
            assert(approxEqual(paretoCCDF(x, T(1), T(1)), 1/x, T(8)*T.epsilon));
        }
    }}
    // Independent Decimal references using exact binary double inputs and
    // 100 decimal digits of working precision.
    assert(approxEqual(paretoPDF(1e200, 1e200, 2.0), 2e-200, 4e-15, 0.0));
    assert(approxEqual(paretoCCDF(1e200, 1e-200, 0.5), 1e-200, 2e-13, 0.0));
    assert(approxEqual(paretoInvCDF(1e-20, 1.0, 1e-20),
        2.71828182845904523537, 4e-15, 0.0));
    assert(approxEqual(paretoLPDF(2.0, 2.0, 1e20),
        45.358554679320968371, 4e-15));
    assert(approxEqual(paretoPDF(cast(double)1e-310L, cast(double)1e-311L, 300.0),
        2999999999955.543257076365, 4e-13, 0.0));
    assert(approxEqual(paretoPDF(1e-200, 1e-201, 400.0),
        3.999999999999942054888553e-198, 4e-13, 0.0));
    assert(approxEqual(paretoInvCDF(0.5, 1e-300, 0.0008),
        1.938628441591554730157366e76, 4e-13, 0.0));
}
