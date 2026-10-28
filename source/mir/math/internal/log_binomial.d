/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.math.internal.log_binomial;

import mir.bignum.fp: Fp;
import mir.internal.utility: isFloatingPoint;

private enum size_t logFactorialAlternative = 1 << 11;

/++
Natural logarithm of the product of count consecutive integers starting at start.
An empty product returns zero. The mathematical endpoint may exceed ulong.max;
large products are evaluated logarithmically without forming that integer sum.
+/
T logFactorial(T = double)(ulong count, ulong start = 1)
    if (isFloatingPoint!T)
    in (start, "start must be larger than zero")
{
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: factorial;
    import std.mathspecial: logGamma;

    if (count == 0)
        return T(0);
    // Compare without wrapping count+start into the small-product path.
    if (count < logFactorialAlternative && start < logFactorialAlternative - count) {
        return fp_log!T(factorial(count, start));
    }
    // A small lower endpoint has no large gamma logarithm to cancel.
    // Phobos already evaluates logGamma in real; convert before adding.
    if (start < 16)
        return T(logGamma(cast(real) count + start)) - fp_log!T(factorial(start - 1));
    return logRisingProduct!T(count, start);
}

private @safe pure nothrow @nogc
T logRisingProduct(T)(ulong count, ulong start)
{
    pragma(inline, false);
    import std.math: log, log1p;
    import mir.math.internal.log_gamma: logGammaStirlingCorrection;
    const T c = count, s = start;
    const T ratio = c / s;
    const T logRatio = log1p(ratio);
    // Combine logGamma(s+c)-logGamma(s) algebraically. Evaluating the two
    // gamma logs first can lose the entire answer when c is small beside s.
    // Both gamma arguments are >=16; reciprocal arithmetic avoids s+c.
    return c * log(s) + (s * logRatio - c) + (c - T(0.5)) * logRatio
        + logGammaStirlingCorrection((1 / s) / (1 + ratio))
        - logGammaStirlingCorrection(1 / s);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(logFactorial(0) == 0);
    assert(logFactorial(1) == 0);
    assert(logFactorial(2).approxEqual(log(1.0 * 2)));
    assert(logFactorial(3).approxEqual(log(1.0 * 2 * 3)));
    assert(logFactorial(4).approxEqual(log(1.0 * 2 * 3 * 4)));
    assert(logFactorial(5).approxEqual(log(1.0 * 2 * 3 * 4 * 5)));
}

// test starting points
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(logFactorial(2, 2).approxEqual(log(2.0 * 3)));
    assert(logFactorial(3, 2).approxEqual(log(2.0 * 3 * 4)));
    assert(logFactorial(4, 2).approxEqual(log(2.0 * 3 * 4 * 5)));
    assert(logFactorial(4, 3).approxEqual(log(3.0 * 4 * 5 * 6)));
    assert(logFactorial(4, 4).approxEqual(log(4.0 * 5 * 6 * 7)));
    assert(logFactorial(5, 2).approxEqual(log(2.0 * 3 * 4 * 5 * 6)));
    assert(logFactorial(5, 3).approxEqual(log(3.0 * 4 * 5 * 6 * 7)));
    assert(logFactorial(5, 4).approxEqual(log(4.0 * 5 * 6 * 7 * 8)));
    assert(logFactorial(5, 5).approxEqual(log(5.0 * 6 * 7 * 8 * 9)));
}

// test larger value
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: fp_log;
    import mir.math.common: approxEqual, log;
    import mir.math.numeric: factorial;
    import std.mathspecial: logGamma;

    size_t x = logFactorialAlternative + 500;
    assert(logFactorial(x).approxEqual(logGamma(cast(double) x + 1)));
    assert(logFactorial(x, 2).approxEqual(logGamma(cast(double) x + 1 + 1) - log(1.0)));
    assert(logFactorial(x, 3).approxEqual(logGamma(cast(double) x + 2 + 1) - log(2.0)));
    assert(logFactorial(x, x / 2).approxEqual(logGamma(cast(double) x + x / 2) - fp_log!double(factorial(x / 2 - 1))));
}

private enum size_t logBinomialCoefficientAlternative = 1 << 11;

// Preserve short products and avoid unsigned wraparound in large endpoints.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (start; [1UL, 16UL, 2047UL, 2048UL, 10_000_000_000_000_000UL, ulong.max])
        {
            assert(logFactorial!T(0, start) == 0);
            assert(approxEqual(logFactorial!T(1, start), log(T(start)),
                128 * T.epsilon, T(0)));
        }
        // Independent 120-digit references, with exact integer inputs.
        assert(approxEqual(logFactorial!T(2048, 16),
            T(13658.47943226613946080615327315399332L), 64 * T.epsilon, T(0)));
        assert(approxEqual(logFactorial!T(2048, 2047),
            T(16405.31054988202626462080640908479586L), 64 * T.epsilon, T(0)));
        assert(approxEqual(logFactorial!T(2, 10_000_000_000_000_000UL),
            T(73.68272297580946198857572654989964964L), 64 * T.epsilon, T(0)));
        assert(approxEqual(logFactorial!T(2, ulong.max),
            T(88.72283911167299960535150143802232549L), 64 * T.epsilon, T(0)));
        // These endpoints used to wrap into the iterative factorial path.
        assert(approxEqual(logFactorial!T(ulong.max),
            T(799877009219260410589.2105935388033377L), 64 * T.epsilon, T(0)));
        assert(approxEqual(logFactorial!T(ulong.max, ulong.max),
            T(825449626509665721883.9553386757654554L), 64 * T.epsilon, T(0)));
    }}
}

///
T logBinomialCoefficient(T = double)(ulong n, uint k)
    if (isFloatingPoint!T)
    in (k <= n, "k must be less than or equal to n")
{
    import mir.math.common: log;
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: binomialCoefficient;

    if (k > n - k) {
        k = cast(uint)(n - k);
    }
    if (k == 0) {
        return T(0.0);
    } else if (k == 1) {
        return log(cast(T) n);
    } else {
        if (n < logBinomialCoefficientAlternative) {
            return fp_log!T(binomialCoefficient(n, k));
        } else {
            import mir.math.internal.log_beta: logBeta;
            // C(n,k) = 1 / (k * B(n-k+1,k)). k>=2 makes n-k+1 safe
            // even at ulong.max; no overflowing n+1 or gamma-log difference.
            return -logBeta(T(n - k + 1), T(k)) - log(T(k));
        }
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: Fp, fp_log;
    import mir.math.numeric: binomialCoefficient;
    import mir.math.common: approxEqual, log;

    assert(logBinomialCoefficient(5, 1).approxEqual(log(5.0)));
    assert(logBinomialCoefficient(5, 2).approxEqual(fp_log!double(binomialCoefficient(5, 2))));
    assert(logBinomialCoefficient(5, 3).approxEqual(fp_log!double(binomialCoefficient(5, 3))));
    assert(logBinomialCoefficient(5, 4).approxEqual(fp_log!double(binomialCoefficient(5, 4))));
}

// test n = 6
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: binomialCoefficient;
    import mir.math.common: approxEqual;

    assert(logBinomialCoefficient(6, 1).approxEqual(fp_log!double(binomialCoefficient(6, 1))));
    assert(logBinomialCoefficient(6, 2).approxEqual(fp_log!double(binomialCoefficient(6, 2))));
    assert(logBinomialCoefficient(6, 3).approxEqual(fp_log!double(binomialCoefficient(6, 3))));
    assert(logBinomialCoefficient(6, 4).approxEqual(fp_log!double(binomialCoefficient(6, 4))));
    assert(logBinomialCoefficient(6, 5).approxEqual(fp_log!double(binomialCoefficient(6, 5))));
}

// test n = 7
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: binomialCoefficient;
    import mir.math.common: approxEqual;

    assert(logBinomialCoefficient(7, 1).approxEqual(fp_log!double(binomialCoefficient(7, 1))));
    assert(logBinomialCoefficient(7, 2).approxEqual(fp_log!double(binomialCoefficient(7, 2))));
    assert(logBinomialCoefficient(7, 3).approxEqual(fp_log!double(binomialCoefficient(7, 3))));
    assert(logBinomialCoefficient(7, 4).approxEqual(fp_log!double(binomialCoefficient(7, 4))));
    assert(logBinomialCoefficient(7, 5).approxEqual(fp_log!double(binomialCoefficient(7, 5))));
    assert(logBinomialCoefficient(7, 6).approxEqual(fp_log!double(binomialCoefficient(7, 6))));
}

// test n = 8
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: binomialCoefficient;
    import mir.math.common: approxEqual;

    assert(logBinomialCoefficient(8, 1).approxEqual(fp_log!double(binomialCoefficient(8, 1))));
    assert(logBinomialCoefficient(8, 2).approxEqual(fp_log!double(binomialCoefficient(8, 2))));
    assert(logBinomialCoefficient(8, 3).approxEqual(fp_log!double(binomialCoefficient(8, 3))));
    assert(logBinomialCoefficient(8, 4).approxEqual(fp_log!double(binomialCoefficient(8, 4))));
    assert(logBinomialCoefficient(8, 5).approxEqual(fp_log!double(binomialCoefficient(8, 5))));
    assert(logBinomialCoefficient(8, 6).approxEqual(fp_log!double(binomialCoefficient(8, 6))));
    assert(logBinomialCoefficient(8, 7).approxEqual(fp_log!double(binomialCoefficient(8, 7))));
}

// test k = 0, n = k
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(logBinomialCoefficient(5, 0) == 0);
    assert(logBinomialCoefficient(5, 5) == 0);
    assert(logBinomialCoefficient(1, 1) == 0);
    assert(logBinomialCoefficient(1, 0) == 0);
}

// Test large values
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.bignum.fp: fp_log;
    import mir.math.numeric: binomialCoefficient;
    import mir.math.common: approxEqual, log;

    size_t x = logBinomialCoefficientAlternative + 500;

    assert(logBinomialCoefficient(x, 1).approxEqual(log(cast(double) x)));
    assert(logBinomialCoefficient(x, 250).approxEqual(logFactorial(x) - logFactorial(250) - logFactorial(x - 250)));
    assert(logBinomialCoefficient(x, cast(uint) x / 2).approxEqual(logFactorial(x) - logFactorial(x / 2) - logFactorial(x / 2)));
    assert(logBinomialCoefficient(x, cast(uint) x - 250).approxEqual(logBinomialCoefficient(x, 250)));
}

// Large coefficients use beta normalization without n+1 or factorial subtraction.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        foreach (n; [2047UL, 2048UL, 4096UL, 10_000_000_000_000_000UL, ulong.max])
        {
            const T expected = log(T(n)) + log(T(n - 1)) - log(T(2));
            assert(approxEqual(logBinomialCoefficient!T(n, 2), expected,
                64 * T.epsilon, T(0)));
            assert(logBinomialCoefficient!T(n, 0) == 0);
            assert(logBinomialCoefficient!T(n, 1) == log(T(n)));
        }
        assert(logBinomialCoefficient!T(2048, 32) == logBinomialCoefficient!T(2048, 2016));
        assert(approxEqual(logBinomialCoefficient!T(2048, 32),
            T(162.1864092813636915658867887771846208L), 64 * T.epsilon, T(0)));
        assert(approxEqual(logBinomialCoefficient!T(ulong.max, uint.max),
            T(99560390359.53630308783576904322921299L), 64 * T.epsilon, T(0)));
    }}
}
