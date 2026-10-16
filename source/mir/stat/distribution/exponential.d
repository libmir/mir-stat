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

    // Require CTFE support from the math primitive, without hiding PDF regressions.
    static if (__traits(compiles, {
        import mir.math.common: exp;
        enum double probe = exp(-1.0);
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
