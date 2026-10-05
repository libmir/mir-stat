/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Exponential_distribution, Exponential Distribution).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.exponential;

import mir.internal.utility: isFloatingPoint;

/++
Computes the exponential probability density function (PDF).

Params:
    x = value to evaluate PDF
    lambda = number of events in an interval

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

    return lambda * exp(-lambda * x);
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

/++
Computes the exponential cumulative distribution function (CDF).

Params:
    x = value to evaluate CDF
    lambda = number of events in an interval

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

Params:
    x = value to evaluate CCDF
    lambda = number of events in an interval

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

Params:
    p = value to evaluate InvCDF
    lambda = number of events in an interval

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

/++
Computes the exponential log probability density function (LPDF).

Params:
    x = value to evaluate LPDF
    lambda = number of events in an interval

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
