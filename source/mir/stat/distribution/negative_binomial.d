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
    if (useNegativeBinomialCountTail(k, r, p)) {
        import mir.stat.distribution.binomial: binomialCCDF;
        return binomialCCDF(r - 1, k + r, p);
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
    if (useNegativeBinomialCountTail(k, r, p)) {
        import mir.stat.distribution.binomial: binomialCDF;
        return binomialCDF(r - 1, k + r, p);
    }
    return betaIncomplete(cast(T)k + 1, cast(T)r, 1 - p);
}

// For tiny p, rounding 1-p before incomplete-beta evaluation can move a
// large-count quantile by several integers. Below sqrt(epsilon), that input
// error can exceed sqrt(epsilon) relative to p. Sum the equivalent binomial
// tail instead, retaining p itself. Limit r so this fallback stays inexpensive;
// the binomial tail walks away from its mean and stops at negligible terms.
private @safe pure nothrow @nogc
bool useNegativeBinomialCountTail(T)(size_t k, size_t r, T p)
{
    import mir.math.common: sqrt;
    return p < sqrt(T.epsilon) && r <= 64 && k <= size_t.max - r;
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
    static if (size_t.sizeof == 8) {
        // Independent 100-digit finite binomial-tail sum for binary p=1e-8.
        enum double cdf = 0.550000000007872962468353187594624191;
        assert(approxEqual(negativeBinomialCDF(2529614601UL, 25, 1e-8),
            cdf, 256 * double.epsilon, 0.0));
        assert(approxEqual(negativeBinomialCCDF(2529614601UL, 25, 1e-8),
            1 - cdf, 256 * double.epsilon, 0.0));
    }
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

/++
Computes the negative binomial inverse cumulative distribution function (InvCDF).

Returns the smallest representable count whose CDF reaches `q`. Queries at zero
and the point mass `p = 1` return zero. For `q = 1`, `p = 0`, or a quantile
beyond the count range, returns `size_t.max`. Close comparisons use bounded
exact binomial arithmetic when the binary input fractions fit its workspace;
other interior results remain subject to incomplete-beta tail accuracy.

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
    import mir.math.common: sqrt, floor;
    import mir.stat.distribution.normal: normalInvCDF;
    import mir.stat.distribution.geometric: geometricInvCDF;

    if (q == 0 || p == 1)
        return 0;
    if (q == 1 || p == 0)
        return size_t.max;
    if (r == 1) {
        const real value = geometricInvCDF(cast(real)q, cast(real)p);
        return value >= cast(real)size_t.max ? size_t.max : cast(size_t)value;
    }

    static if (is(T == float)) alias F = double;
    else alias F = T;
    const F probability = p;
    const F query = q;
    const bool upper = query > 0.5;
    const F target = upper ? 1 - query : query;
    bool reached(size_t k) {
        import std.math: abs;
        import mir.stat.distribution.binomial: binomialExactQuantileReached;
        const F value = upper ? negativeBinomialCCDF(k, r, probability)
                              : negativeBinomialCDF(k, r, probability);
        if (k < 8192 && r <= 8192 - k && abs(value - target) <=
            128 * (k + r + 1) * F.epsilon * (abs(value) + target + F.min_normal)) {
            // K<=k is equivalent to at least r successes in k+r trials.
            // Reuse the bounded exact comparison at ambiguous CDF steps.
            bool supported;
            const bool exact = binomialExactQuantileReached(r - 1, k + r,
                probability, query, supported, true);
            if (supported)
                return exact;
        }
        return upper ? value <= target : value >= target;
    }

    const F mu = r * (1 - probability) / probability;
    const F root = sqrt(r * (1 - probability));
    const F sigma = root / probability;
    const F z = normalInvCDF(query);
    F estimate;
    if (r > 20 && probability > 0.25 && probability < 0.75)
        estimate = mu + sigma * z - 0.5;
    else
        estimate = mu + sigma * (z + ((2 - probability) / root) * (z*z - 1) / 6);
    // Clamp before conversion, including an indeterminate extreme estimate.
    // The estimate only chooses a starting point; tail comparisons decide.
    size_t guess = !(estimate > 0) ? 0
        : estimate >= cast(F)size_t.max ? size_t.max : cast(size_t)floor(estimate);
    size_t low, high;
    size_t step = 1;
    if (reached(guess)) {
        high = guess;
        while (true) {
            if (high == 0)
                return 0;
            const size_t next = step < high ? high - step : 0;
            if (!reached(next)) {
                low = next;
                break;
            }
            high = next;
            step = step <= size_t.max / 2 ? step * 2 : size_t.max;
        }
    } else {
        low = guess;
        while (true) {
            // Saturate an unrepresentable quantile rather than wrapping a count.
            if (low == size_t.max)
                return size_t.max;
            const size_t next = step < size_t.max - low ? low + step : size_t.max;
            if (reached(next)) {
                high = next;
                break;
            }
            low = next;
            step = step <= size_t.max / 2 ? step * 2 : size_t.max;
        }
    }
    // Invariant: CDF(low)<q<=CDF(high), with upper queries compared as tails.
    // Bisect without adding endpoints, which could overflow.
    while (high - low > 1) {
        const size_t middle = low + (high - low) / 2;
        if (reached(middle)) high = middle;
        else low = middle;
    }
    return high;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;
    0.9.negativeBinomialInvCDF(6, 3.0 / 4).should == 4;
}

// Endpoint precedence, negative initial estimates, and exact CDF steps.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    static foreach (T; AliasSeq!(float, double, real)) {{
        assert(negativeBinomialInvCDF(T(0), 5, T(0)) == 0);
        assert(negativeBinomialInvCDF(T(0.5), 5, T(0)) == size_t.max);
        assert(negativeBinomialInvCDF(T(0.5), 5, T(1)) == 0);
        assert(negativeBinomialInvCDF(T(1), 5, T(1)) == 0);
        assert(negativeBinomialInvCDF(T(1), 5, T(0.5)) == size_t.max);
        assert(negativeBinomialInvCDF(T(1e-20L), 25, T(0.5)) == 0);
        assert(negativeBinomialInvCDF(T(0.5), 1, T(1e-30L)) == size_t.max);
        assert(negativeBinomialInvCDF(T(0.5), 25, T(0.5)) == 24);
        // For r=2, p=1/2, CDF(4)=57/64 exactly.
        const T q = T(57) / 64;
        assert(negativeBinomialInvCDF(nextDown(q), 2, T(0.5)) == 4);
        assert(negativeBinomialInvCDF(q, 2, T(0.5)) == 4);
        assert(negativeBinomialInvCDF(nextUp(q), 2, T(0.5)) == 5);
        assert(negativeBinomialInvCDF(nextUp(T(0.75)), 1, T(0.5)) == 2);
    }}
    // The binary double 0.0001 is slightly larger than the square of the
    // binary double 0.01: an exact rational comparison puts this query at 1.
    assert(negativeBinomialInvCDF(0.0001, 2, 0.01) == 1);
    // Independent 100-digit references for a skewed, wider distribution.
    assert(negativeBinomialInvCDF(0.5, 100, 0.01) == 9867);
    static if (size_t.sizeof == 8) {
        // 100-digit binomial-tail summation for the exact binary p and q.
        // CDF(k-1)=0.5499999992233198939..., CDF(k)=0.5500000000078729624....
        assert(negativeBinomialInvCDF(0.55, 25, 1e-8) == 2529614601UL);
    }
    assert(negativeBinomialInvCDF(0.5, 6, 1e-30) == size_t.max);
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
