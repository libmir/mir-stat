/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution).

There are multiple alternative formulations of the Negative Binomial Distribution. The
formulation in this module counts failures before the `r`th success. Thus
`k = 0` means that the first `r` trials all succeed, and `r = 1` is geometric.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.negative_binomial;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

/++
Computes the negative binomial probability mass function (PMF).

Params:
    k = number of failures before stopping
    r = number of successes until stopping
    p = `true` probability


See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
pragma(inline, true)
T negativeBinomialPMF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow, exp;
    import mir.combinatorics: binomial;

    // Preserve the inexpensive product only when its coefficient is known to
    // fit. Test by subtraction so k+r-1 cannot wrap before the decision.
    if (k <= 64 && r - 1 <= 64 - k) {
        const ulong coefficient = binomial(k + r - 1, r - 1);
        const T failures = pow(1 - p, k);
        const T successes = pow(p, r);
        if (failures >= T.min_normal && successes >= T.min_normal)
            return coefficient * failures * successes;
    }
    if (p == 0)
        return 0;
    if (p == 1)
        return k == 0 ? 1 : 0;
    if (k == 0)
        return pow(p, r);
    return cast(T)exp(negativeBinomialLogMass(k, r, cast(real)p));
}

private @safe pure nothrow @nogc
real negativeBinomialLogMass(size_t k, size_t r, real p)
{
    import mir.stat.distribution.binomial: binomialLogMassCounts;
    import mir.math.common: log;
    import std.math: log1p;
    if (p == 0)
        return -real.infinity;
    if (p == 1)
        return k == 0 ? 0 : -real.infinity;
    if (k == 0)
        return cast(real)r * log(p);
    // NB(k,r,p) = BinomialMass(r,k+r,p) * r/(k+r). Separate counts
    // avoid overflowing k+r, and log1p preserves a small ratio k/r.
    return binomialLogMassCounts(r, k, p) - log1p(cast(real)k / r);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialPMF(6, 3.0 / 4).shouldApprox == 0.0875988;
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0877915;
    1.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.175583;
    2.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.2048468;
    3.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.1820861;
    4.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.1365645;
    5.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.09104303;
    6.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.05563741;
    7.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0317928;
    8.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.0172211;
    9.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.008929461;
    10.negativeBinomialPMF(6, 2.0 / 3).shouldApprox == 0.00446473;
}

/++
Computes the  negative binomial probability mass function (PMF) directly with extended 
floating point types (e.g. `Fp!128`), which provides additional accuracy for
extreme values of `k`, `r`, or `p`. 

Params:
    k = number of failures before stopping
    r = number of successes until stopping
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T fp_negativeBinomialPMF(T)(const size_t k, const size_t r, const T p)
    if (is(T == Fp!size, size_t size))
    in (r > 0, "number of successes must be larger than zero")
    in (cast(double) p >= 0, "p must be greater than or equal to 0")
    in (cast(double) p <= 1, "p must be less than or equal to 1")
{
    import mir.math.internal.fp_powi: fp_powi;
    import mir.math.numeric: binomialCoefficient;

    return binomialCoefficient(k + r - 1, cast(const uint) (r - 1)) * fp_powi(T(1 - cast(double) p), k) * fp_powi(p, r);
}

/// Extended precision provides accurate values for large counts.
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    1.fp_negativeBinomialPMF(1_000_000, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(1, 1_000_000, 0.75);
}


// test more values
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    0.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(0, val + 5, 0.75);
    1.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(1, val + 5, 0.75);
    2.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(2, val + 5, 0.75);
    5.fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(5, val + 5, 0.75);
    (val / 2).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val / 2, val + 5, 0.75);
    (val - 5).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 5, val + 5, 0.75);
    (val - 2).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 2, val + 5, 0.75);
    (val - 1).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val - 1, val + 5, 0.75);
    (val - 0).fp_negativeBinomialPMF(val + 5, Fp!128(0.75)).fp_log!double.shouldApprox == negativeBinomialLPMF(val, val + 5, 0.75);
}

// using Fp!128
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.conv: to;
    import mir.test: shouldApprox;

    for (size_t i; i <= 5; i++) {
        i.fp_negativeBinomialPMF(5, Fp!128(0.50)).to!double.shouldApprox == negativeBinomialPMF(i, 5, 0.50);
        i.fp_negativeBinomialPMF(5, Fp!128(0.75)).to!double.shouldApprox == negativeBinomialPMF(i, 5, 0.75);
    }
}

/++
Computes the negative binomial cumulative distribution function (CDF).

Params:
    k = number of failures before stopping
    r = number of successes until stopping
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialCDF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow;
    import std.mathspecial: betaIncomplete;

    if (p == 0 || p == 1)
        return p;
    if (r == 1) {
        import mir.stat.distribution.geometric: geometricCDF;
        return geometricCDF(k, p);
    }
    if (k == 0) {
        return pow(p, r);
    }
    // Evaluate the lower tail directly. Form k+1 after conversion so the
    // largest count does not wrap to a zero beta shape.
    return betaIncomplete(cast(T)r, cast(T)k + 1, p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialCDF(6, 3.0 / 4).shouldApprox == 0.9218731;
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;
    
    static double sumOfnegativeBinomialPMFs(T)(size_t k, size_t r, T p) {
        double result = 0.0;
        for (size_t i; i <= k; i++) {
            result += negativeBinomialPMF(i, r, p);
        }
        return result;
    }

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialCDF(5, 0.25).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.25);
        i.negativeBinomialCDF(5, 0.50).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.50);
        i.negativeBinomialCDF(5, 0.75).shouldApprox == sumOfnegativeBinomialPMFs(i, 5, 0.75);

        i.negativeBinomialCDF(6, 0.25).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.25);
        i.negativeBinomialCDF(6, 0.5).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.5);
        i.negativeBinomialCDF(6, 0.75).shouldApprox == sumOfnegativeBinomialPMFs(i, 6, 0.75);
    }
}

/++
Computes the negative binomial complementary cumulative distribution function (CCDF).

Params:
    k = number of failures before stopping
    r = number of successes until stopping
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialCCDF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    import std.mathspecial: betaIncomplete;

    if (p == 0 || p == 1)
        return 1 - p;
    if (r == 1) {
        import mir.stat.distribution.geometric: geometricCCDF;
        return geometricCCDF(k, p);
    }
    if (k == 0) {
        return oneMinusExpNeg(-cast(T)r * log(p));
    }
    return betaIncomplete(cast(T)k + 1, cast(T)r, 1 - p);
}

// Lower tails must not disappear through subtraction, or count addition wrap.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextDown;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        const T p = T.epsilon / 16;
        assert(approxEqual(negativeBinomialCDF(1, 1, p), 2 * p, 16 * T.epsilon, T(0)));
        // P(K<=1) = p^2*(3-2p) for r=2.
        assert(approxEqual(negativeBinomialCDF(1, 2, p), p*p*(3-2*p),
            128 * T.epsilon, T(0)));
        const T nearOne = nextDown(T(1));
        const T failure = 1 - nearOne;
        assert(approxEqual(negativeBinomialCCDF(0, 2, nearOne),
            failure * (2 - failure), 16 * T.epsilon, T(0)));
        foreach (r; [size_t(1), size_t(6)]) {
            foreach (k; [size_t(0), size_t(1), size_t.max]) {
                assert(negativeBinomialCDF(k, r, T(0)) == 0);
                assert(negativeBinomialCDF(k, r, T(1)) == 1);
                assert(negativeBinomialCCDF(k, r, T(0)) == 1);
                assert(negativeBinomialCCDF(k, r, T(1)) == 0);
            }
        }
        assert(negativeBinomialCDF(size_t.max, 1, T(0.5)) == 1);
        assert(negativeBinomialCCDF(size_t.max, 1, T(0.5)) == 0);
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    4.negativeBinomialCCDF(6, 3.0 / 4).shouldApprox == 0.07812691;
}

// test multiple
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialCCDF(5, 0.25).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.25);
        i.negativeBinomialCCDF(5, 0.50).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.50);
        i.negativeBinomialCCDF(5, 0.75).shouldApprox == 1 - negativeBinomialCDF(i, 5, 0.75);

        i.negativeBinomialCCDF(6, 0.25).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.25);
        i.negativeBinomialCCDF(6, 0.5).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.5);
        i.negativeBinomialCCDF(6, 0.75).shouldApprox == 1 - negativeBinomialCDF(i, 6, 0.75);
    }
}

private
@safe pure nothrow @nogc
size_t negativeBinomialInvCDFSearch(T)(const size_t guess, ref T cdfGuess, const T q, const size_t r, const T p, const size_t searchIncrement)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    size_t guessNew = guess;
    if (q <= cdfGuess) {
        T cdfGuessPrevious;
        while (guessNew > 0) {
            cdfGuessPrevious = cdfGuess;
            cdfGuess = negativeBinomialCDF(guessNew - searchIncrement, r, p);
            if (q > cdfGuess) {
                cdfGuess = cdfGuessPrevious;
                break;
            }
            guessNew = guessNew > searchIncrement ? guessNew - searchIncrement : 0;
        }
    } else {
        while (q > cdfGuess) {
            guessNew = guessNew + searchIncrement;
            cdfGuess = negativeBinomialCDF(guessNew, r, p);
        }
    }
    return guessNew;
}

/++
Computes the negative binomial inverse cumulative distribution function (InvCDF).

Params:
    q = value to evaluate InvCDF
    r = number of successes until stopping
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
size_t negativeBinomialInvCDF(T)(const T q, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: floor, sqrt;
    import mir.stat.distribution.normal: normalInvCDF;

    if (q == 0) {
        return 0;
    } else if (q == 1) {
        return size_t.max;
    }

    size_t guess = 0;
    T mu = r * (1 - p) / p;
    T pre_std = sqrt(r * (1 - p));
    T std = pre_std / p;
    T z = normalInvCDF(q);
    if (r > 20 && p > 0.25 && p < 0.75) {
        guess = cast(size_t) floor(mu + std * z - 0.5);
    } else {
        // Cornish-Fisher Approximation
        guess = cast(size_t) floor(mu + std * (z + ((2 - p) / pre_std) * (z * z - 1) / 6));
    }
    T cdfGuess = negativeBinomialCDF(guess, r, p);

    if (guess < 10_000) {
        return negativeBinomialInvCDFSearch(guess, cdfGuess, q, r, p, 1);
    } else {
        // Faster search for large values of guess
        size_t searchIncrement = cast(size_t) floor(guess * 0.001);
        size_t searchIncrementPrevious;
        do {
            searchIncrementPrevious = searchIncrement;
            guess = negativeBinomialInvCDFSearch(guess, cdfGuess, q, r, p, searchIncrement);
            searchIncrement = cast(size_t) floor(searchIncrement * 0.01);
        } while (searchIncrementPrevious > 0 && searchIncrement > guess * (10 * T.epsilon));
        if (searchIncrementPrevious <= 1) {
            return guess;
        } else {
            return negativeBinomialInvCDFSearch(guess, cdfGuess, q, r, p, 1);
        }
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    0.9.negativeBinomialInvCDF(6, 3.0 / 4).should == 4;
}

//
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    0.negativeBinomialInvCDF(5, 0.6).should == 0;
    1.negativeBinomialInvCDF(5, 0.6).should == size_t.max;

    for (double x = 0.05; x < 1; x = x + 0.05) {
        size_t value = x.negativeBinomialInvCDF(5, 0.6);
        value.negativeBinomialCDF(5, 0.6).should!"a >= b"(x);
        if (value > 0) {
            (value - 1).negativeBinomialCDF(5, 0.6).should!"a < b"(x);
        }
    }
}

// alternate guess paths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    static immutable size_t[] ns = [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.05; x < 1; x = x + 0.05) {
            value = x.negativeBinomialInvCDF(ns[i], ps[i]);
            negativeBinomialCDF(value, ns[i], ps[i]).should!"a >= b"(x);
            if (value > 0) {
                negativeBinomialCDF(value - 1, ns[i], ps[i]).should!"a < b"(x);
            }
        }
    }
}

// more detailed guess paths
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    static immutable size_t[] ns = [  25,  37,  34,    25,     25,   105];
    static immutable double[] ps = [0.55, 0.2, 0.15, 0.05, 1.0e-8, 0.025];

    size_t value;
    for (size_t i; i < 6; i++) {
        for (double x = 0.01; x < 1; x = x + 0.01) {
            value = x.negativeBinomialInvCDF(ns[i], ps[i]);
            negativeBinomialCDF(value, ns[i], ps[i]).should!"a >= b"(x);
            if (value > 0) {
                negativeBinomialCDF(value - 1, ns[i], ps[i]).should!"a < b"(x);
            }
        }
    }
}

/++
Computes the negative binomial log probability mass function (LPMF).

Params:
    k = number of failures before stopping
    r = number of successes until stopping
    p = `true` probability

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Negative_binomial_distribution, Negative Binomial Distribution)
+/
@safe pure nothrow @nogc
T negativeBinomialLPMF(T)(const size_t k, const size_t r, const T p)
    if (isFloatingPoint!T)
    in (r > 0, "number of successes must be larger than zero")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.internal.xlogy: xlogy, xlog1py;
    import mir.math.internal.log_binomial: logBinomialCoefficient;

    if (k <= 64 && r - 1 <= 64 - k)
        return logBinomialCoefficient(k + r - 1, cast(uint)(r - 1))
            + xlog1py(k, -p) + xlogy(r, p);
    return cast(T)negativeBinomialLogMass(k, r, cast(real)p);
}

// Independent high-precision references and full-width count boundaries.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        assert(approxEqual(negativeBinomialPMF(100, 100, T(0.5)),
            T(0.0281742395046282111236226320714500749007L), 128 * T.epsilon, T(0)));
        foreach (r; [size_t(1), size_t(100), size_t.max]) {
            assert(negativeBinomialPMF(0, r, T(1)) == 1);
            assert(negativeBinomialPMF(1, r, T(1)) == 0);
            assert(negativeBinomialPMF(0, r, T(0)) == 0);
            assert(negativeBinomialLPMF(1, r, T(1)) == -T.infinity);
        }
        // Their sum exceeds size_t, but the log mass is still finite.
        assert(negativeBinomialLPMF(size_t.max, size_t.max, T(0.5)) < 0);
        assert(negativeBinomialLPMF(size_t.max, size_t.max, T(0.5)) > -100);
    }}
    static if (size_t.sizeof == 8) {
        assert(approxEqual(negativeBinomialLPMF(1, 4_294_967_297UL, 1 - 0x1p-32),
            -1.00000000011641532189921495928864740589, 256 * double.epsilon, 0.0));
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    4.negativeBinomialLPMF(6, 3.0 / 4).exp.shouldApprox == 4.negativeBinomialPMF(6, 3.0 / 4);
}

//
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    for (size_t i; i <= 10; i++) {
        i.negativeBinomialLPMF(5, 0.5).exp.shouldApprox == negativeBinomialPMF(i, 5, 0.5);
        i.negativeBinomialLPMF(5, 0.75).exp.shouldApprox == negativeBinomialPMF(i, 5, 0.75);
    }
}

/// Accurate values for large values of `n`
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    1.negativeBinomialLPMF(1_000_000, 0.75).shouldApprox == fp_negativeBinomialPMF(1, 1_000_000, Fp!128(0.75)).fp_log!double;
}

// testing more values
version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.test: shouldApprox;

    enum size_t val = 1_000_000;

    0.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(0, val + 5, Fp!128(0.75)).fp_log!double;
    1.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(1, val + 5, Fp!128(0.75)).fp_log!double;
    2.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(2, val + 5, Fp!128(0.75)).fp_log!double;
    5.negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(5, val + 5, Fp!128(0.75)).fp_log!double;
    (val / 2).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val / 2, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 5).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 5, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 2).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 2, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 1).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val - 1, val + 5, Fp!128(0.75)).fp_log!double;
    (val - 0).negativeBinomialLPMF(val + 5, 0.75).shouldApprox == fp_negativeBinomialPMF(val, val + 5, Fp!128(0.75)).fp_log!double;
}
