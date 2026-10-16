/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Discrete_uniform_distribution, Discrete Uniform Distribution).

The outcomes are the integers from $(D lower) through $(D upper), inclusive,
with equal probability. Bounds and observations use $(D size_t); negative
outcomes are not represented by this API. Forward functions require an
observation inside the interval rather than extending the tails outside it.

PMF, CDF, and CCDF accept $(D lower == upper). LPMF and inverse CDF instead
require $(D lower < upper). Forward functions return $(D double); the inverse
CDF accepts a built-in floating-point probability and returns $(D size_t).

In the tables, $(D n = upper - lower + 1) denotes the mathematical number
of outcomes. The implementation adds one after conversion to floating point
to avoid integer overflow for the full $(D size_t) domain. Large interval
sizes and interior quantile ranks can still round in the floating-point type.

Finite expressions are subject to floating-point rounding. Contract violations
are not prescribed return values: disabling contracts does not make an invalid
input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.uniform_discrete;

import mir.internal.utility: isFloatingPoint;

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
