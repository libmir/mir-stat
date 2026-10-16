/**
 * Normal Distribution
 *
 * Copyright: Based on the CEPHES math library, which is
 *            Copyright (C) 1994 Stephen L. Moshier (moshier@world.std.com).
 * Authors:   Stephen L. Moshier, ported to D by Don Clugston and David Nadlinger. Adopted to Mir by Ilya Yaroshenko.
*/
/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution).

The one-argument overloads use mean zero and standard deviation one.
The other overloads take a finite mean and a positive finite standard
deviation. These parameter requirements are not enforced by contracts;
zero or negative standard deviations do not describe a supported distribution.

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

Compatibility:
These functions are implemented in mir-stat. If importing both this module and
$(D mir.math.func.normal), use a selective import or qualify calls to choose
which module supplies $(D normalPDF), $(D normalCDF), and $(D normalInvCDF).

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-2026 Mir Stat Authors.

+/

module mir.stat.distribution.normal;

import mir.internal.utility: isFloatingPoint;
import mir.math.common: exp, fabs, floor, log, sqrt;

/++
Computes the normal probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

This overload uses mean zero and standard deviation one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D z == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D z == 0)) $(TD $(D 1 / sqrt(2 * PI))))
    $(TR $(TD $(D z == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    z = standardized observation
+/
@safe pure nothrow @nogc
T normalPDF(T)(const T z)
    if (isFloatingPoint!T)
{
    import mir.math.common: exp;
    return T(SQRT2PIINV) * exp(z * z * -0.5);
}

/++
Ditto, with location and scale parameters.

Returns a density, which may exceed one; it is not a point probability.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == mean)) $(TD $(D 1 / sqrt(2 * PI) / stdDev)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = observation
    mean = finite mean
    stdDev = positive finite standard deviation
+/
@safe pure nothrow @nogc
T normalPDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    import mir.stat.constant: LOGSQRT2PI;
    const T z = standardizedNormal(x, mean, stdDev);
    const T density = normalPDF(z);
    // A subnormal intermediate loses precision before division by a small
    // scale. Delay exponentiation until the scale has been included.
    if (density < T.min_normal && stdDev > 0 && stdDev < 1)
        return exp((-T(0.5) * z) * z - T(LOGSQRT2PI) - log(stdDev));
    return density / stdDev;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(normalPDF(0.0).approxEqual(0.3989422804014327));
    assert(normalPDF(2.0, 2.0, 3.0).approxEqual(normalPDF(0.0) / 3));
}

/// A normal density tends to zero in both tails.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(normalPDF(-double.infinity) == 0);
    assert(normalPDF(double.infinity, 2.0, 3.0) == 0);
}

/++
Computes the normal cumulative distribution function (CDF).
For a small upper-tail probability, use $(LREF normalCCDF) rather than
subtracting the CDF from one.

Returns the lower-tail probability $(D P(X <= x)).

This overload uses mean zero and standard deviation one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D a == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D a == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D a == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    a = standardized observation

Accuracy:
Subnormal results have reduced relative precision. Extreme tails also inherit
the rounding error of the platform exponential function.
+/
@safe pure nothrow @nogc
T normalCDF(T)(const T a)
    if (isFloatingPoint!T)
{
    pragma(inline, false);
    import mir.math.constant: SQRT1_2;

    T x = a * T(SQRT1_2);
    T z = fabs(x);

    if (z < 1)
    {
        const T errorFunction = x == 0 ? x :
            x * rationalPoly!(erfNumerator, erfDenominator)(x * x);
        return 0.5f + 0.5f * errorFunction;
    }
    else
    {
        T y = 0.5f * erfce(z);
        /* Multiply by exp(-x^2 / 2)  */
        // Evaluate the half exponent directly. Taking sqrt(exp(-a*a))
        // loses tails as soon as the full exponent underflows.
        z = normalTailExponential(a);
        y = y * z;
        if (y != y)
            y = a != a ? a : 0;
        if (x > 0)
            y = 1 - y;
        return y;
    }
}

/++
Ditto, with location and scale parameters.

Returns the lower-tail probability $(D P(X <= x)).

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D x == mean)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = observation
    mean = finite mean
    stdDev = positive finite standard deviation
+/
@safe pure nothrow @nogc
T normalCDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    return normalCDF(standardizedNormal(x, mean, stdDev));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(normalCDF(0.0) == 0.5);
    assert(normalCDF(2.0, 2.0, 3.0) == 0.5);
    assert(normalCDF(1.0).approxEqual(0.8413447460685429));
}

/++
Computes the normal inverse cumulative distribution function.
Zero and one return negative and positive infinity, respectively.

Returns the quantile at cumulative probability $(D p).

This overload uses mean zero and standard deviation one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = cumulative probability in [0, 1]
+/
@safe pure nothrow @nogc
T normalInvCDF(T)(const T p)
in {
  assert(p >= 0 && p <= 1, "Domain error");
}
do
{
    pragma(inline, false);
    if (p <= 0 || p >= 1)
    {
        if (p == 0)
            return -T.infinity;
        if ( p == 1 )
            return T.infinity;
        return T.nan; // domain error
    }
    int code = 1;
    T y = p;
    if ( y > (1 - T(EXP_2)) )
    {
        y = 1 - y;
        code = 0;
    }

    T x, z, y2, x0, x1;

    if ( y > T(EXP_2) )
    {
        y = y - 0.5L;
        y2 = y * y;
        x = y + y * (y2 * rationalPoly!(P0, Q0)(y2));
        static if (T.mant_dig > double.mant_dig)
            return x * T(SQRT2PI);
        else
            return x * double(SQRT2PI);
    }

    x = sqrt( -2 * log(y) );
    x0 = x - log(x)/x;
    z = 1/x;
    if ( x < 8 )
    {
        x1 = z * rationalPoly!(P1, Q1)(z);
    }
    else if ( x < 32 )
    {
        x1 = z * rationalPoly!(P2, Q2)(z);
    }
    else
    {
        x1 = z * rationalPoly!(P3, Q3)(z);
    }
    x = x0 - x1;
    if ( code != 0 )
    {
        x = -x;
    }
    return x;
}

/++
Ditto, with location and scale parameters.

Returns the quantile at cumulative probability $(D p).

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 0.5)) $(TD $(D mean)))
    $(TR $(TD $(D p == 1)) $(TD $(D T.infinity)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = cumulative probability in [0, 1]
    mean = finite mean
    stdDev = positive finite standard deviation
+/
@safe pure nothrow @nogc
T normalInvCDF(T)(const T p, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    const T z = normalInvCDF(p);
    const T scaled = z * stdDev;
    // A finite opposite-signed mean can bring an overflowing product back
    // into range. Halve both terms first, preserving endpoint infinities.
    if (fabs(scaled) == T.infinity && fabs(z) <= T.max && fabs(stdDev) <= T.max && fabs(mean) <= T.max)
        return ((z * T(0.5)) * stdDev + mean * T(0.5)) * 2;
    return scaled + mean;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(normalInvCDF(0.5) == 0);
    assert(normalInvCDF(0.5, 2.0, 3.0) == 2);
    assert(normalInvCDF(0.975).approxEqual(1.959963984540054));
}


/++
Computes the normal complementary cumulative distribution function (CCDF)

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D x == mean)) $(TD $(D 0.5)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate CCDF
    mean = finite mean
    stdDev = positive finite standard deviation

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution)
+/
@safe pure nothrow @nogc
T normalCCDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    return normalCCDF(standardizedNormal(x, mean, stdDev));
}

/++
Computes the standard normal complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

This overload uses mean zero and standard deviation one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D a == -T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D a == 0)) $(TD $(D 0.5)))
    $(TR $(TD $(D a == T.infinity)) $(TD $(D 0)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    a = standardized observation
+/
@safe pure nothrow @nogc
T normalCCDF(T)(const T a)
    if (isFloatingPoint!T)
{
    return normalCDF(-a);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(0.5.normalCCDF.approxEqual(1 - normalCDF(0.5)));
    assert(0.5.normalCCDF(0, 1.5).approxEqual(1 - normalCDF(0.5, 0, 1.5)));
    assert(1.5.normalCCDF(1, 3).approxEqual(1 - normalCDF(1.5, 1, 3)));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert((-3.0).normalCCDF.approxEqual(1 - normalCDF(-3.0)));
    assert(3.0.normalCCDF.approxEqual(1 - normalCDF(3.0)));
    assert((-7.0).normalCCDF(1, 4).approxEqual(1 - normalCDF(-7.0, 1, 4)));
    assert(9.0.normalCCDF(1, 4).approxEqual(1 - normalCDF(9.0, 1, 4)));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(normalCCDF(double.infinity).approxEqual(1 - normalCDF(double.infinity)));
    assert(normalCCDF(-double.infinity).approxEqual(1 - normalCDF(-double.infinity)));
}

/++
Computes the normal log probability density function (LPDF)

Returns the natural logarithm of the density.

The table assumes finite parameters and a positive scale.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == mean)) $(TD $(D -log(2 * PI) / 2 - log(stdDev))))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = value to evaluate LPDF
    mean = finite mean
    stdDev = positive finite standard deviation

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Normal_distribution, Normal Distribution)
+/
@safe pure nothrow @nogc
T normalLPDF(T)(const T x, const T mean, const T stdDev)
    if (isFloatingPoint!T)
{
    import mir.math.common: log;
    return normalLPDF(standardizedNormal(x, mean, stdDev)) - log(stdDev);
}

/++
Computes the standard normal log probability density function (LPDF).

Returns the natural logarithm of the density.

This overload uses mean zero and standard deviation one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == -T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D x == 0)) $(TD $(D -log(2 * PI) / 2)))
    $(TR $(TD $(D x == T.infinity)) $(TD $(D -T.infinity)))
    $(TR $(TD NaN) $(TD NaN))
)

Params:
    x = standardized observation
+/
@safe pure nothrow @nogc
T normalLPDF(T)(const T x)
    if (isFloatingPoint!T)
{
    import mir.stat.constant: LOGSQRT2PI;

    // Halve before squaring: x*x can overflow while x*x/2 still fits.
    return (-T(0.5) * x) * x - T(LOGSQRT2PI);
}

// Divide first only when subtraction overflows for finite observations.
// In that case x and mean have opposite signs, so the divisions cannot
// introduce cancellation between infinities. Keep the usual rounding otherwise.
private @safe pure nothrow @nogc
T standardizedNormal(T)(T x, T mean, T stdDev)
{
    import mir.math.common: fabs;
    const T delta = x - mean;
    if (fabs(delta) == T.infinity && fabs(x) <= T.max && fabs(mean) <= T.max)
        return x / stdDev - mean / stdDev;
    return delta / stdDev;
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: fabs, sqrt, log;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        // x*x overflows, but the halved square and log density are finite.
        const T x = sqrt(T.max) * T(1.125);
        const T expected = -(x / 2) * x;
        assert(fabs(normalLPDF(x) / expected - 1) <= 4 * T.epsilon);
        const T scale = T.max * T(0.75);
        assert(fabs(normalLPDF(scale, -scale, scale) -
            (normalLPDF(T(2)) - log(scale))) <= 4 * T.epsilon * log(scale));
        assert(fabs(normalCCDF(scale, -scale, scale) / normalCCDF(T(2)) - 1)
            <= 4 * T.epsilon);
        assert(normalLPDF(T.infinity) == -T.infinity);
        assert(normalLPDF(-T.infinity) == -T.infinity);
        assert(normalCCDF(T.infinity, T(0), T(1)) == 0);
        assert(normalCCDF(-T.infinity, T(0), T(1)) == 1);
        assert(normalLPDF(T.nan) != normalLPDF(T.nan));
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(0.5.normalLPDF.approxEqual(log(normalPDF(0.5))));
    assert(0.5.normalLPDF(0, 1.5).approxEqual(log(normalPDF(0.5, 0, 1.5))));
    assert(1.5.normalLPDF(1, 3).approxEqual(log(normalPDF(1.5, 1, 3))));
}


// Cephes coefficients and private kernels retained from mir.math.func.normal.
// Keeping these here lets distribution fixes ship with mir-stat independently.
private:
@safe pure nothrow @nogc:

enum real SQRT2PI = 2.50662827463100050241576528481104525L;
enum real SQRT2PIINV = 1 / SQRT2PI;
enum real EXP_2 = 0.135335283236612691893999494972484403L;

// Compute exp(-a*a/2) using an exact square plus a small residual. Computing
// sqrt(exp(-a*a)) instead would lose representable tails to early underflow.
T normalTailExponential(T)(const T a)
{
    // Cephes Math Library Release 2.9, June 2000.
    // Copyright 2000 by Stephen L. Moshier.
    // Multiples of 1/16 have exact squares throughout float's nonzero tail;
    // double and real can use the finer split to keep the residual small.
    enum T M = T.mant_dig == 24 ? 16 : 32_768;
    enum T MINV = 1 / M;
    const T x = -fabs(a);
    const T m = MINV * floor(M * x + 0.5f);
    const T f = x - m;
    const T u = -(m * m) * T(0.5);
    const T u1 = -(2 * m * f + f * f) * T(0.5);
    return exp(u) * exp(u1);
}

// Exponentially scaled erfc for x >= 1; the CDF applies the exponential.
T erfce(T)(const T x)
{
    T y = 1 / x;

    if (x < 8)
    {
        return rationalPoly!(P, Q)(y);
    }
    else
    {
        return y * rationalPoly!(R, S)(y * y);
    }
}

T rationalPoly(alias numerator, alias denominator, T)(const T x) pure nothrow
{
    return x.poly!numerator / x.poly!denominator;
}

T poly(alias vec, T)(const T x)
{
    import mir.internal.utility: Iota;
    T y = T(vec[$ - 1]);
    foreach_reverse(i; Iota!(vec.length - 1))
    {
        y *= x;
        y += T(vec[i]);
    }
    return y;
}

/* erfc(x) = exp(-x^2) P(1/x)/Q(1/x)
   1/8 <= 1/x <= 1
   Peak relative error 5.8e-21  */
immutable real[10] P = [ -0x1.30dfa809b3cc6676p-17, 0x1.38637cd0913c0288p+18,
   0x1.2f015e047b4476bp+22, 0x1.24726f46aa9ab08p+25, 0x1.64b13c6395dc9c26p+27,
   0x1.294c93046ad55b5p+29, 0x1.5962a82f92576dap+30, 0x1.11a709299faba04ap+31,
   0x1.11028065b087be46p+31, 0x1.0d8ef40735b097ep+30
];

immutable real[11] Q = [ 0x1.14d8e2a72dec49f4p+19, 0x1.0c880ff467626e1p+23,
   0x1.04417ef060b58996p+26, 0x1.404e61ba86df4ebap+28, 0x1.0f81887bc82b873ap+30,
   0x1.4552a5e39fb49322p+31, 0x1.11779a0ceb2a01cep+32, 0x1.3544dd691b5b1d5cp+32,
   0x1.a91781f12251f02ep+31, 0x1.0d8ef3da605a1c86p+30, 1.0
];

/* erfc(x) = exp(-x^2) 1/x R(1/x^2) / S(1/x^2)
    1/128 <= 1/x < 1/8
    Peak relative error 1.9e-21  */
immutable real[5] R = [ 0x1.b9f6d8b78e22459ep-6, 0x1.1b84686b0a4ea43ap-1,
    0x1.b8f6aebe96000c2ap+1, 0x1.cb1dbedac27c8ec2p+2, 0x1.cf885f8f572a4c14p+1
];

immutable real[6] S = [
    0x1.87ae3cae5f65eb5ep-5, 0x1.01616f266f306d08p+0, 0x1.a4abe0411eed6c22p+2,
    0x1.eac9ce3da600abaap+3, 0x1.5752a9ac2faebbccp+3, 1.0
];

/* erf(x)  = x P(x^2)/Q(x^2)
    0 <= x <= 1
    Peak relative error 7.6e-23  */
immutable real[7] erfNumerator = [ 0x1.0da01654d757888cp+20, 0x1.2eb7497bc8b4f4acp+17,
    0x1.79078c19530f72a8p+15, 0x1.4eaf2126c0b2c23p+11, 0x1.1f2ea81c9d272a2ep+8,
    0x1.59ca6e2d866e625p+2, 0x1.c188e0b67435faf4p-4
];

immutable real[7] erfDenominator = [ 0x1.dde6025c395ae34ep+19, 0x1.c4bc8b6235df35aap+18,
    0x1.8465900e88b6903ap+16, 0x1.855877093959ffdp+13, 0x1.e5c44395625ee358p+9,
    0x1.6a0fed103f1c68a6p+5, 1.0
];

static immutable real[8] P0 =
[ -0x1.758f4d969484bfdcp-7, 0x1.53cee17a59259dd2p-3,
-0x1.ea01e4400a9427a2p-1,  0x1.61f7504a0105341ap+1, -0x1.09475a594d0399f6p+2,
0x1.7c59e7a0df99e3e2p+1, -0x1.87a81da52edcdf14p-1,  0x1.1fb149fd3f83600cp-7
];

static immutable real[8] Q0 =
[ -0x1.64b92ae791e64bb2p-7, 0x1.7585c7d597298286p-3,
-0x1.40011be4f7591ce6p+0, 0x1.1fc067d8430a425ep+2, -0x1.21008ffb1e7ccdf2p+3,
0x1.3d1581cf9bc12fccp+3, -0x1.53723a89fd8f083cp+2, 1.0
];

static immutable real[10] P1 =
[ 0x1.20ceea49ea142f12p-13, 0x1.cbe8a7267aea80bp-7,
0x1.79fea765aa787c48p-2, 0x1.d1f59faa1f4c4864p+1, 0x1.1c22e426a013bb96p+4,
0x1.a8675a0c51ef3202p+5, 0x1.75782c4f83614164p+6, 0x1.7a2f3d90948f1666p+6,
0x1.5cd116ee4c088c3ap+5, 0x1.1361e3eb6e3cc20ap+2
];

static immutable real[10] Q1 =
[ 0x1.3a4ce1406cea98fap-13, 0x1.f45332623335cda2p-7,
0x1.98f28bbd4b98db1p-2, 0x1.ec3b24f9c698091cp+1, 0x1.1cc56ecda7cf58e4p+4,
0x1.92c6f7376bf8c058p+5, 0x1.4154c25aa47519b4p+6, 0x1.1b321d3b927849eap+6,
0x1.403a5f5a4ce7b202p+4, 1.0
];

static immutable real[8] P2 =
[ 0x1.8c124a850116a6d8p-21, 0x1.534abda3c2fb90bap-13,
0x1.29a055ec93a4718cp-7, 0x1.6468e98aad6dd474p-3, 0x1.3dab2ef4c67a601cp+0,
0x1.e1fb3a1e70c67464p+1, 0x1.b6cce8035ff57b02p+2, 0x1.9f4c9e749ff35f62p+1
];

static immutable real[8] Q2 =
[ 0x1.af03f4fc0655e006p-21, 0x1.713192048d11fb2p-13,
0x1.4357e5bbf5fef536p-7, 0x1.7fdac8749985d43cp-3, 0x1.4a080c813a2d8e84p+0,
0x1.c3a4b423cdb41bdap+1, 0x1.8160694e24b5557ap+2, 1.0
];

static immutable real[8] P3 =
[ -0x1.55da447ae3806168p-34, -0x1.145635641f8778a6p-24,
-0x1.abf46d6b48040128p-17, -0x1.7da550945da790fcp-11, -0x1.aa0b2a31157775fap-8,
0x1.b11d97522eed26bcp-3, 0x1.1106d22f9ae89238p+1, 0x1.029a358e1e630f64p+1
];

static immutable real[8] Q3 =
[ -0x1.74022dd5523e6f84p-34, -0x1.2cb60d61e29ee836p-24,
-0x1.d19e6ec03a85e556p-17, -0x1.9ea2a7b4422f6502p-11, -0x1.c54b1e852f107162p-8,
0x1.e05268dd3c07989ep-3, 0x1.239c6aff14afbf82p+1, 1.0
];


// Independent mpmath references (100 decimal digits), including tails that
// used to vanish when exp(-x*x) underflowed before taking its square root.
version(mir_stat_test)
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        static if (T.mant_dig == 24)
        {
            enum T x = 12;
            enum T expected = 1.776482112077678997696171001845557e-33L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T x = 30;
            enum T expected = 4.906713927148187059533809256580190e-198L;
        }
        else
        {
            enum T x = 150;
            enum T expected = 4.091457980903002399032776796971440e-4889L;
        }
        // The platform exp implementation can lose O(epsilon*x*x) relative
        // accuracy at the extreme extended-precision exponent.
        static if (T.max_exp > 1024)
            assert(fabs(normalCDF(-x) / expected - 1) <= 4 * T.epsilon*x*x);
        else
            assert(fabs(normalCDF(-x) / expected - 1) <= 16 * T.epsilon);
        assert(normalCDF(T.nan) != normalCDF(T.nan));
        assert(normalCDF(T.max) == 1);
        assert(normalCDF(-T.max) == 0);
    }}
    enum double subnormal = 2.885428360068784308350970481566904e-316L;
    assert(fabs(normalCDF(-38.0) - subnormal) <= 2 * double.min_normal * double.epsilon);
    assert(fabs(normalCDF(-27.0) / 7.389481006885018257517944005216002e-161 - 1)
        <= 8 * double.epsilon);
}

version(mir_stat_test)
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T scale = T.max * T(0.75);
        assert(fabs(normalCDF(scale, -scale, scale) - normalCDF(T(2))) <= 4*T.epsilon);
        assert(fabs(normalPDF(scale, -scale, scale) - normalPDF(T(2)) / scale)
            <= 2*T.min_normal*T.epsilon);
        // At p=.99 the standard quantile exceeds two: the product overflows,
        // but subtracting scale puts the result back inside the finite range.
        const T q = normalInvCDF(T(0.99));
        assert(fabs(normalInvCDF(T(0.99), -scale, scale) / scale - (q - 1))
            <= 8*T.epsilon);
        assert(normalInvCDF(T(0), -scale, scale) == -T.infinity);
        assert(normalInvCDF(T(1), -scale, scale) == T.infinity);
        static if (T.mant_dig == 24)
        {
            enum T z = 16;
            enum T expected = 6.983703985129843791196611283129551e-18L;
        }
        else static if (T.max_exp == 1024)
        {
            enum T z = 40;
            enum T expected = 5.261021768760553166461947867031892e-40L;
        }
        else
        {
            enum T z = 160;
            enum T expected = 1.018639841583995008567464946201700e-627L;
        }
        enum T tinyScale = T.min_normal / 8;
        assert(fabs(normalPDF(z * tinyScale, T(0), tinyScale) / expected - 1)
            <= 16*T.epsilon*z*z);
    }}
}

version(mir_stat_test)
unittest
{
    // The central inverse polynomial must not narrow its final multiplier
    // to double when the caller requested extended precision.
    static if (real.mant_dig > 53)
    {
        enum real q = 0.6744897501960817432022270145413071854L;
        assert(fabs(normalInvCDF(0.25L) + q) <= 16 * real.epsilon);
        assert(fabs(normalInvCDF(0.75L) - q) <= 16 * real.epsilon);
    }
}

version(mir_stat_test)
unittest
{
    // Independent inverse references from mpmath at 400 decimal digits.
    // Exercise both tail polynomials and the smallest double probability.
    assert(fabs(normalInvCDF(1e-100) / -21.2734535609653242951172121887 - 1)
        <= 4 * double.epsilon);
    assert(fabs(normalInvCDF(1e-300) / -37.0470962993611992372229625079 - 1)
        <= 4 * double.epsilon);
    assert(fabs(normalInvCDF(double.min_normal * double.epsilon) /
        -38.4674056171443462507843621685 - 1) <= 4 * double.epsilon);
    assert(normalCDF(double.infinity) == 1);
    assert(normalCDF(-double.infinity) == 0);
    assert(normalPDF(double.infinity) == 0);
    assert(normalPDF(-double.infinity) == 0);
    assert(normalCCDF(double.nan) != normalCCDF(double.nan));
}
