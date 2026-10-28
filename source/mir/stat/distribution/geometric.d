/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution).

The count is the number of failures before the first success in independent
trials with success probability $(D p). Thus $(D k == 0) means the first trial
succeeds; it does not mean that no trials were performed. The usual
distribution has $(D 0 < p <= 1). The API also accepts $(D p == 0) as a
limiting case with no success and zero mass at every finite count.

PMF, LPMF, and integer-threshold CDF/CCDF accept built-in floating-point and $(D Fp)
probabilities. The checked inverse $(LREF geometricInvCDFChecked) accepts $(D Fp)
probabilities and returns a proven integer count or an explicit limit status.
The floating-return inverse CDF and floating-threshold overloads use
built-in floating-point probabilities. PMF and LPMF take a $(D size_t) count. CDF and CCDF additionally
accept a floating-point threshold $(D x >= -1), using the integer part of
nonnegative finite thresholds. $(LREF geometricInvCDF) returns a floating-point count
and has explicit endpoint conventions described below.

The $(D Fp) tails use 192 extra working bits and separate intermediate exponents.
CDF sums positive geometric-series terms in binary blocks; CCDF evaluates its
power directly. Neither tail is obtained by subtracting the other from one.
Counts retain their full $(D size_t) width, including when $(D k + 1) would
not fit. Work is logarithmic in the number of trials. These guarded algorithms
improve accuracy without guaranteeing correct rounding for every input.

The tables assume valid parameters except where an invalid input is listed.
Finite expressions are subject to floating-point rounding and range limits.
Contract violations are not prescribed return values: disabling contracts does
not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.geometric;

import mir.internal.utility: isFloatingPoint;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityLess;

// Tiny probabilities and counts whose successor is not an integer size_t.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        const T tiny = T.epsilon / 16;
        assert(geometricCDF(size_t(0), tiny) == tiny);
        assert(approxEqual(geometricCDF(size_t(1), tiny), 2 * tiny, 8 * T.epsilon, T(0)));
        assert(geometricCDF(size_t(0), nextUp(T(0))) == nextUp(T(0)));
        foreach (k; [size_t(0), size_t(64), size_t(65), size_t.max]) {
            assert(geometricPMF(k, T(0)) == 0);
            assert(geometricPMF(k, T(1)) == (k == 0 ? 1 : 0));
            assert(geometricCDF(k, T(0)) == 0);
            assert(geometricCDF(k, T(1)) == 1);
            assert(geometricCCDF(k, T(0)) == 1);
            assert(geometricCCDF(k, T(1)) == 0);
        }
        assert(geometricCDF(size_t.max, T(0.5)) == 1);
        assert(geometricCCDF(size_t.max, T(0.5)) == 0);
        foreach (x; [cast(T)size_t.max, T.infinity]) {
            assert(geometricCDF(x, T(0.5)) == 1);
            assert(geometricCCDF(x, T(0.5)) == 0);
            assert(geometricCDF(x, T(0)) == 0);
            assert(geometricCCDF(x, T(0)) == 1);
            assert(geometricCDF(x, T(1)) == 1);
            assert(geometricCCDF(x, T(1)) == 0);
        }
    }}
    static if (size_t.sizeof == 8) {
        // Independent 100-digit references for exact binary p and k.
        assert(approxEqual(geometricPMF(1UL << 60, 0x1p-60),
            3.190845514646648713140259429307883234771e-19, 8 * double.epsilon, 0.0));
        assert(approxEqual(geometricCCDF(size_t.max, 0x1p-64),
            0.3678794411714423215855522278535, 8 * double.epsilon, 0.0));
    }
}

/++
Computes the geometric probability mass function (PMF) for the number of
failures before the first success.

Returns the point probability $(D P(K == k)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0)) $(TD $(D p)))
    $(TR $(TD $(D p == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D 0)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate PMF
    p = per-trial success probability in $(D [0, 1])

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricPMF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: pow, exp;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (k == 0)
        return p;
    if (p == 0 || p == 1)
        return 0;
    // Keep the short power calculation for ordinary counts. At large counts,
    // rounding 1-p before exponentiation can noticeably change the probability.
    if (k <= 64)
        return pow(1 - p, k) * p;
    return exp(-cast(T)k * negLog1m(p)) * p;
}

/++
Extended-range and extended-precision PMF for an $(D Fp) probability.

Evaluates the represented probability without hardware-floating-point conversions.
Tested widths are $(D Fp!64), $(D Fp!128), and $(D Fp!256). The complement and
power retain 192 extra working bits, with a separate intermediate exponent and
final subnormal rounding. This reduces rounding error but does not promise
universally correct rounding. Counts retain their full $(D size_t) width and
power evaluation costs $(D O(log(k + 1))) multiplications.
+/
@safe pure nothrow @nogc
T geometricPMF(T)(const size_t k, const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpGeometric;
    return fpGeometric(k, p);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    0.geometricPMF(0.5).shouldApprox == 0.5;
    1.geometricPMF(0.5).shouldApprox == 0.25;
    2.geometricPMF(0.25).shouldApprox == 0.140625;
}

/++
Computes the geometric cumulative distribution function (CDF).

Returns the inclusive lower-tail probability $(D P(K <= k)), or $(D P(K <= x))
for a floating-point threshold. The interval $(D -1 <= x && x < 0) is accepted
and lies below the first outcome.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) or $(D 0 <= x && x < 1)) $(TD $(D p)))
    $(TR $(TD $(D -1 <= x && x < 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 0) and $(D x >= 0) (or any $(D k))) $(TD $(D 0), including $(D x == T.infinity)))
    $(TR $(TD $(D p > 0) and $(D x == T.infinity)) $(TD $(D 1)))
    $(TR $(TD $(D p == 1) and $(D x >= 0) (or any $(D k))) $(TD $(D 1)))
    $(TR $(TD $(D x < -1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate CDF
    x = floating-point threshold, at least -1
    p = per-trial success probability in $(D [0, 1])

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricCDF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.stat.internal.neg_log1m: negLog1m;
    import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
    if (k == 0 || p == 0 || p == 1)
        return p;
    // Convert before adding: k+1 need not fit in size_t.
    return oneMinusExpNeg((cast(T)k + 1) * negLog1m(p));
}

/// ditto
@safe pure nothrow @nogc
T geometricCDF(T)(const size_t k, const T p)
    if (is(T == Fp!bits, uint bits))
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpGeometricTail;
    return fpGeometricTail!false(k, p);
}

/// ditto
@safe pure @nogc nothrow
T geometricCDF(T)(const T x, const T p)
    if (isFloatingPoint!T)
    in (x >= -1, "x must be larger than or equal to -1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    if (x < 0)
        return 0;
    // Avoid converting infinity or a count outside size_t to an integer.
    if (x >= cast(T)size_t.max) {
        import mir.math.common: floor;
        import mir.stat.internal.neg_log1m: negLog1m;
        import mir.stat.internal.one_minus_exp: oneMinusExpNeg;
        if (p == 0 || p == 1)
            return p;
        return oneMinusExpNeg((floor(x) + 1) * negLog1m(p));
    }
    return geometricCDF!T(cast(const size_t) x, p);  
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    geometricCDF(-1.0, 0.5).shouldApprox == 0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCDF(0.5).shouldApprox == 0.5;
    1.geometricCDF(0.5).shouldApprox == 0.75;
    2.geometricCDF(0.5).shouldApprox == 0.875;

    geometricCDF(-1.0, 0.25).shouldApprox == 0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCDF(0.25).shouldApprox == 0.25;
    1.geometricCDF(0.25).shouldApprox == 0.4375;
    2.geometricCDF(0.25).shouldApprox == 0.578125;
    2.5.geometricCDF(0.25).shouldApprox == 0.578125;
}

/++
Computes the geometric complementary cumulative distribution function (CCDF).

Returns the strict upper-tail probability $(D P(K > k)), or $(D P(K > x))
for a floating-point threshold. This excludes the mass at the threshold.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) or $(D 0 <= x && x < 1)) $(TD $(D 1 - p)))
    $(TR $(TD $(D -1 <= x && x < 0)) $(TD $(D 1)))
    $(TR $(TD $(D p == 0) and $(D x >= 0) (or any $(D k))) $(TD $(D 1), including $(D x == T.infinity)))
    $(TR $(TD $(D p > 0) and $(D x == T.infinity)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D x >= 0) (or any $(D k))) $(TD $(D 0)))
    $(TR $(TD $(D x < -1) or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate CCDF
    x = floating-point threshold, at least -1
    p = per-trial success probability in $(D [0, 1])

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricCCDF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: exp;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (k == 0 || p == 0 || p == 1)
        return 1 - p;
    return exp(-(cast(T)k + 1) * negLog1m(p));
}

/// ditto
@safe pure nothrow @nogc
T geometricCCDF(T)(const size_t k, const T p)
    if (is(T == Fp!bits, uint bits))
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
{
    import mir.stat.internal.fp_probability: fpGeometricTail;
    return fpGeometricTail!true(k, p);
}

/// ditto
@safe pure @nogc nothrow
T geometricCCDF(T)(const T x, const T p)
    if (isFloatingPoint!T)
    in (x >= -1, "x must be larger than or equal to -1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    if (x < 0)
        return 1;
    if (x >= cast(T)size_t.max) {
        import mir.math.common: floor, exp;
        import mir.stat.internal.neg_log1m: negLog1m;
        if (p == 0 || p == 1)
            return 1 - p;
        return exp(-(floor(x) + 1) * negLog1m(p));
    }
    return geometricCCDF!T(cast(const size_t) x, p);  
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: shouldApprox;

    geometricCCDF(-1.0, 0.5).shouldApprox == 1.0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCCDF(0.5).shouldApprox == 0.5;
    1.geometricCCDF(0.5).shouldApprox == 0.25;
    2.geometricCCDF(0.5).shouldApprox == 0.125;

    geometricCCDF(-1.0, 0.25).shouldApprox == 1.0; // UFCS chaining deduces this as size_t instead of a floating point type
    0.geometricCCDF(0.25).shouldApprox == 0.75;
    1.geometricCCDF(0.25).shouldApprox == 0.5625;
    2.geometricCCDF(0.25).shouldApprox == 0.421875;
    2.5.geometricCCDF(0.25).shouldApprox == 0.421875;
}

/// Count failures, and include the whole CDF step for a fractional threshold.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.math.common: approxEqual;
    assert(geometricPMF(0, 0.5) == 0.5); // Success on the first trial.
    assert(geometricPMF(1, 0.5) == 0.25); // One failure, then success.
    assert(approxEqual(geometricCDF(1.5, 0.5), 0.75));
    assert(approxEqual(geometricCCDF(1.5, 0.5), 0.25));
}

/++
Outcome of $(LREF geometricInvCDFChecked).
+/
enum GeometricInverseStatus
{
    /// The count is proven to be the least nonnegative k with CDF(k) >= q.
    found,
    /// q == 0 and p < 1: the existing inverse convention returns -1.
    belowSupport,
    /// q == 1, or p == 0 and q > 0: the existing convention returns infinity.
    infinite,
    /// The finite mathematical count is proven to exceed maxCount.
    countLimit,
    /// The working precision could not prove a required comparison.
    precisionLimit
}

/++
Checked inverse result. Read count only when status is found, or use
tryGetCount to extract it conditionally. Other statuses do not carry an
approximate count. A default-initialized result is unresolved.
+/
struct GeometricInverseResult
{
    /// Outcome of the search or endpoint convention.
    GeometricInverseStatus status = GeometricInverseStatus.precisionLimit;
    /// Exact integer count for found; otherwise unused and zero.
    size_t count;

    /// Assign the proven count and return true, or leave destination unchanged.
    bool tryGetCount(ref size_t destination) const @safe pure nothrow @nogc
    {
        if (status != GeometricInverseStatus.found)
            return false;
        destination = count;
        return true;
    }
}

/++
Computes a checked geometric inverse CDF for represented $(D Fp) probabilities.

For 0 < p < 1 and 0 < q < 1, found proves that count is the least integer
k >= 0 satisfying 1 - (1-p)^(k+1) >= q in exact arithmetic. Comparisons
use outward-rounded fixed-point intervals, not the rounded public CDF.

Endpoint precedence matches geometricInvCDF: q == 1 returns infinite first;
then p == 1 returns found with count zero; then q == 0 returns belowSupport;
then p == 0 returns infinite. Endpoints are independent of maxCount.

The inclusive maxCount limits the returned integer, not the number of
iterations. A countLimit result proves the finite quantile exceeds that limit;
it is not infinity. The default limit depends on the platform's size_t width.
Increasing maxCount may resolve countLimit, but cannot increase precision.

Comparisons first use bits+128 fractional bits, then bits+512. If neither
proves a comparison, precisionLimit is returned even if the count fits the
limit. In particular, tiny p and q near small-count jumps can require more
fractional bits than this budget. A precisionLimit result supplies neither
an approximate answer nor proof that the count exceeds maxCount. Retrying
identical inputs has the same outcome. Callers must propagate the unresolved
result or use a separately justified method, rather than fall back to a
rounded CDF or hardware logarithms.

The search allocates no memory. At fixed probability precision, it uses
O(log(maxCount+1)^2) wide-integer products in the worst case.

Params:
    q = finite represented cumulative probability in [0,1]
    p = finite represented success probability in [0,1]
    maxCount = inclusive largest count permitted in a found result
Returns:
    A proven count, an endpoint convention, or an explicit resource limit.
Preconditions:
    Both probabilities must be valid even when contracts are disabled.
+/
@safe pure nothrow @nogc
GeometricInverseResult geometricInvCDFChecked(uint bits)(Fp!bits q, Fp!bits p,
    size_t maxCount = size_t.max)
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
    in (fpProbabilityValid(q), "q must be finite and between zero and one")
{
    import mir.stat.internal.fp_geometric_inverse: evaluate = geometricInvCDFChecked;
    return evaluate(q,p,maxCount);
}

/// Extract a proven count or retry with a larger inclusive count limit.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!128;
    alias Status = GeometricInverseStatus;
    auto result = geometricInvCDFChecked(T(0.875),T(0.5),1);
    assert(result.status == Status.countLimit);
    size_t count = 99;
    assert(!result.tryGetCount(count) && count == 99);
    result = geometricInvCDFChecked(T(0.875),T(0.5),2);
    assert(result.status == Status.found);
    assert(result.tryGetCount(count) && count == 2);
}

/// Endpoint conventions and unresolved precision have separate outcomes.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!128;
    alias Status = GeometricInverseStatus;
    assert(geometricInvCDFChecked(T(0),T(0.5)).status == Status.belowSupport);
    assert(geometricInvCDFChecked(T(1),T(1)).status == Status.infinite);
    const zero = geometricInvCDFChecked(T(0),T(1));
    assert(zero.status == Status.found && zero.count == 0);
    auto p = T(1); p.exponent -= 1100;
    auto q = p; q.exponent += 1;
    const unresolved = geometricInvCDFChecked(q,p);
    assert(unresolved.status == Status.precisionLimit);
    size_t count = 99;
    assert(!unresolved.tryGetCount(count) && count == 99);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        const q = T(0.75), p = T(0.5);
        const result = geometricInvCDFChecked(q,p);
        assert(result.status == GeometricInverseStatus.found && result.count == 1);
    }}
    // The established hardware-floating inverse and its return type are unchanged.
    static assert(is(typeof(geometricInvCDF(0.75,0.5)) == double));
    assert(geometricInvCDF(0.75,0.5) == 1);
}

version(mir_stat_test)
@system unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        foreach (value; [-1.0,2.0,double.infinity,double.nan]) {
            assertThrown!AssertError(geometricInvCDFChecked(Fp!128(value),Fp!128(0.5)));
            assertThrown!AssertError(geometricInvCDFChecked(Fp!128(0.5),Fp!128(value)));
        }
    }
}

/++
Computes the geometric inverse cumulative distribution function (InvCDF).

For $(D 0 < q && q < 1) and $(D 0 < p && p < 1), returns the smallest
nonnegative count whose CDF reaches $(D q). The floating return type cannot
represent every integer at large counts. Close comparisons use bounded exact
binary arithmetic where possible; larger cases retain floating-point accuracy.

The $(D q == 1) convention takes precedence over $(D p == 1). In particular,
$(D geometricInvCDF(1.0, 1.0)) returns infinity even though all finite mass
is at zero. The returned $(D -1) at the lower endpoint is a boundary convention,
not a possible number of failures.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 1)) $(TD $(D T.infinity), including when $(D p == 1)))
    $(TR $(TD $(D q < 1) and $(D p == 1)) $(TD $(D 0)))
    $(TR $(TD $(D q == 0) and $(D p < 1)) $(TD $(D -1)))
    $(TR $(TD $(D 0 < q && q < 1) and $(D p == 0)) $(TD $(D T.infinity)))
    $(TR $(TD $(D q < 0), $(D q > 1), or NaN) $(TD Contract violation))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    q = cumulative probability in $(D [0, 1])
    p = per-trial success probability in $(D [0, 1])

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricInvCDF(T)(const T q, const T p)
    if (isFloatingPoint!T)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: ceil;
    import mir.stat.internal.neg_log1m: negLog1m;
    if (q == 1) {
        return T.infinity;
    }
    if (p == 1) {
        return 0;
    }
    if (q == 0)
        return -1;
    if (p == 0)
        return T.infinity;
    // Keep ordinary double logarithms in double; widen float only. The exact
    // fallback below resolves ambiguous steps without widening every call.
    static if (is(T == float)) alias F = double;
    else alias F = T;
    const F lp = negLog1m(cast(F)p);
    const F lq = negLog1m(cast(F)q);
    F guess = ceil(lq / lp) - 1;
    if (guess < 0)
        guess = 0;
    // At counts beyond consecutive-integer precision, stepping by one may
    // leave the value unchanged. Return the representable approximation there.
    if (guess + 1 == guess || guess == F.infinity)
        return cast(T)guess;
    while (guess > 0 && geometricQuantileReached(guess - 1, p, q, lp, lq))
        --guess;
    while (guess + 1 != guess && !geometricQuantileReached(guess, p, q, lp, lq))
        ++guess;
    return cast(T)guess;
}

/// The inverse CDF has explicit conventions at probability zero and one.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    assert(geometricInvCDF(0.0, 0.5) == -1);
    assert(geometricInvCDF(0.75, 0.5) == 1);
    assert(geometricInvCDF(1.0, 0.5) == double.infinity);
    assert(geometricInvCDF(0.0, 1.0) == 0);
    assert(geometricInvCDF(1.0, 1.0) == double.infinity);
}

// Comparing logarithms avoids reconstructing a CDF close to one. Near a
// boundary, do not snap approximately equal values to an integer: adjacent
// queries can lie on opposite sides of the exact CDF step.
private @safe pure nothrow @nogc
bool geometricQuantileReached(T, F)(F k, T p, T q, F lp, F lq)
{
    import std.math: fabs, frexp, ldexp;
    import mir.bignum.integer: BigInt;
    const F value = (k + 1) * lp;
    if (k < 8192 && fabs(value - lq) <= 32 * F.epsilon * (value + lq)) {
        static if (T.mant_dig <= 64) {
            // For p=a/2^b, compare (2^b-a)^(k+1)/2^(b*(k+1))
            // directly with 1-q. Fixed storage keeps the fallback @nogc.
            alias Integer = BigInt!256;
            static void split(T x, out ulong a, out int b) {
                int e;
                a = cast(ulong)ldexp(frexp(x, e), T.mant_dig);
                b = T.mant_dig - e;
                while (a && !(a & 1)) { a >>= 1; --b; }
            }
            static void shift(ref Integer x, int bits) {
                while (bits >= 32) { x *= 1UL << 32; bits -= 32; }
                if (bits) x *= 1UL << bits;
            }
            ulong a, c;
            int b, d;
            split(p, a, b);
            split(q, c, d);
            const size_t m = cast(size_t)k + 1;
            if (b > 0 && b <= 8192 && d > 0 && d <= 8192 && m <= 8192 / b) {
                Integer base = Integer(1);
                shift(base, b);
                Integer probabilityNumerator = Integer(a);
                base -= probabilityNumerator;
                Integer tail = Integer(1);
                size_t exponent = m;
                while (exponent) {
                    if (exponent & 1) tail *= base;
                    exponent >>= 1;
                    if (exponent) base *= base;
                }
                Integer target = Integer(1);
                shift(target, d);
                Integer queryNumerator = Integer(c);
                target -= queryNumerator;
                const int tailExponent = cast(int)m * b;
                if (d > tailExponent) shift(tail, d - tailExponent);
                else shift(target, tailExponent - d);
                return tail <= target;
            }
        }
    }
    return value >= lq;
}

// Exact dyadic boundaries and queries on either side of them.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    import mir.math.common: approxEqual;
    static foreach (T; AliasSeq!(float, double, real)) {{
        foreach (k, q; [T(0.5), T(0.75), T(0.875)]) {
            assert(geometricInvCDF(nextDown(q), T(0.5)) == k);
            assert(geometricInvCDF(q, T(0.5)) == k);
            assert(geometricInvCDF(nextUp(q), T(0.5)) == k + 1);
        }
        assert(geometricInvCDF(T(0.5), T(0)) == T.infinity);
        assert(geometricInvCDF(T(0), T(0.5)) == -1);
        assert(geometricInvCDF(T(0), T(1)) == 0);
        assert(geometricInvCDF(T(1), T(1)) == T.infinity);
        assert(geometricInvCDF(T(1e-20L), T(0.5)) == 0);
        // Dividing the smallest positive query by -log(1-p) may underflow
        // before ceil. The first count still reaches this positive query.
        assert(geometricInvCDF(nextUp(T(0)), T(0.99)) == 0);
        // The next query above CDF(0)=p belongs to count 1. Rounded logarithms
        // can initially suggest count 0, requiring an upward correction.
        const T probability = T(0.0003);
        assert(geometricInvCDF(nextUp(probability), probability) == 1);
        assert(approxEqual(geometricInvCDF(T(0.5), T(1e-20L)),
            T(6.9314718055994530941e19L), 8 * T.epsilon, T(0)));
    }}
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    0.geometricInvCDF(0.5).should == -1;
    0.5.geometricInvCDF(0.5).should == 0;
    0.75.geometricInvCDF(0.5).should == 1;
    0.875.geometricInvCDF(0.5).should == 2;
    0.95.geometricInvCDF(0.5).should == 4;

    0.geometricInvCDF(0.25).should == -1;
    0.25.geometricInvCDF(0.25).should == 0;
    0.4375.geometricInvCDF(0.25).should == 1;
    0.578125.geometricInvCDF(0.25).should == 2;
    0.95.geometricInvCDF(0.25).should == 10;

    0.5.geometricInvCDF(1).should == 0;
    1.geometricInvCDF(0.5).should == double.infinity;
}

/++
Computes the geometric log probability mass function (LPMF).

Returns the natural logarithm of $(D P(K == k)).

The $(D Fp) overload evaluates logarithms directly from the represented probability,
without conversion to a hardware float or first rounding $(D 1 - p).
It uses guarded integer series, retaining 192 extra bits before final rounding;
this is not a universal correctly-rounded guarantee. The two nonpositive log
terms are combined before rounding to the result type.
Counts retain their full $(D size_t) width.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D k == 0) and $(D p > 0)) $(TD $(D log(p))))
    $(TR $(TD $(D p == 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p == 1) and $(D k == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p == 1) and $(D k > 0)) $(TD $(D -T.infinity)))
    $(TR $(TD $(D p < 0), $(D p > 1), or NaN) $(TD Contract violation))
)

Params:
    k = number of failures at which to evaluate LPMF
    p = per-trial success probability in $(D [0, 1])

See_also: $(LINK2 https://en.wikipedia.org/wiki/Geometric_distribution, Geometric Distribution)
+/
@safe pure @nogc nothrow
T geometricLPMF(T)(const size_t k, const T p)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
{
    import mir.math.common: log;
    import mir.math.internal.xlogy: xlog1py;
    return xlog1py(k, -p) + log(p);   
}

/// ditto
@safe pure nothrow @nogc
T geometricLPMF(T)(const size_t k, const T p)
    if (is(T == Fp!bits, uint bits))
    in (fpProbabilityValid(p), "p must be a finite probability in [0, 1]")
{
    if (!p.coefficient) return T(-double.infinity);
    if (!fpProbabilityLess(p,T(1))) return T(k ? -double.infinity : 0);
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    enum bits = typeof(T.init.coefficient).sizeof * 8;
    alias L = ProbabilityLog!bits;
    auto logarithm = L.probability(p);
    if (k)
        logarithm = logarithm.add(L.complement(p).multiply(L.S.from(Fp!(bits + 192)(k))));
    auto result = logarithm.finish!bits;
    result.sign = true;
    return result;
}

/// Preserve Fp precision in the log mass, including for very small probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    import mir.stat.distribution.bernoulli: bernoulliLPMF;
    auto p = Fp!128(1);
    p.exponent -= 2000;
    assert(geometricLPMF(0,p) == bernoulliLPMF(true,p));
    assert(geometricLPMF(size_t.max,p) == geometricLPMF(0,p));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp;
    import mir.test: shouldApprox;

    0.geometricLPMF(0.5).exp.shouldApprox == 0.geometricPMF(0.5);
    1.geometricLPMF(0.5).exp.shouldApprox == 1.geometricPMF(0.5);
    2.geometricLPMF(0.25).exp.shouldApprox == 2.geometricPMF(0.25);
}

// BEGIN geometric references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.distribution.negative_binomial: negativeBinomialPMF;
    {
        alias T = Fp!64;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -65, UInt!64.fromHexString("8000000000000000"))),
            Reference(17UL, T(false, -64, UInt!64.fromHexString("c000000000000000")), T(false, -98, UInt!64.fromHexString("c000000000000000"))),
            Reference(1000UL, T(false, -66, UInt!64.fromHexString("8000000000000000")), T(false, -259, UInt!64.fromHexString("a3b3890f36f74447"))),
            Reference(18446744073709551615UL, T(false, -128, UInt!64.fromHexString("8000000000000000")), T(false, -129, UInt!64.fromHexString("9b4597e37cb04ff4"))),
            Reference(18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -129, UInt!64.fromHexString("bc5ab1b16779be36"))),
            Reference(1152921504606846976UL, T(false, -123, UInt!64.fromHexString("8000000000000000")), T(false, -125, UInt!64.fromHexString("bc5ab1b16779be30"))),
            Reference(7UL, T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -512, UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(0UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(1000UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1163, UInt!64.fromHexString("8000000000000000"))),
            Reference(65UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -128, UInt!64.fromHexString("ffffffffffffffbf"))),
            Reference(100UL, T(false, -73, UInt!64.fromHexString("8000000000000000")), T(false, -74, UInt!64.fromHexString("e82bbc2a781ed0fb"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(17UL, T(false, -128, UInt!128.fromHexString("c0000000000000000000000000000000")), T(false, -162, UInt!128.fromHexString("c0000000000000000000000000000000"))),
            Reference(1000UL, T(false, -130, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -323, UInt!128.fromHexString("a3b3890f36f74446be82c7214803d5bf"))),
            Reference(18446744073709551615UL, T(false, -256, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -257, UInt!128.fromHexString("ffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -193, UInt!128.fromHexString("bc5ab1b16779be35d3eae7ddd466d137"))),
            Reference(1152921504606846976UL, T(false, -187, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -189, UInt!128.fromHexString("bc5ab1b16779be2f92e80179e4dc0049"))),
            Reference(7UL, T(false, -128, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")), T(false, -1024, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(0UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1000UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(65UL, T(false, -255, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -256, UInt!128.fromHexString("ffffffffffffffffffffffffffffffbf"))),
            Reference(100UL, T(false, -137, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -138, UInt!128.fromHexString("e82bbc2a781ed0fadd053012bc81d36d"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; T p, expected; }
        enum references = [
            Reference(1UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(17UL, T(false, -256, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000")), T(false, -290, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1000UL, T(false, -258, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -451, UInt!256.fromHexString("a3b3890f36f74446be82c7214803d5bea9f6b40967456fb173e93441f785bbe5"))),
            Reference(18446744073709551615UL, T(false, -512, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -513, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffff8000000000000001"))),
            Reference(18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -321, UInt!256.fromHexString("bc5ab1b16779be35d3eae7ddd466d136a6fe9c010eb169994b346efec1bf3cab"))),
            Reference(1152921504606846976UL, T(false, -315, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -317, UInt!256.fromHexString("bc5ab1b16779be2f92e80179e4dc0048c9b3e38c2c2c6680ed787ae2afa36e19"))),
            Reference(7UL, T(false, -256, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")), T(false, -2048, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(0UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1000UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(65UL, T(false, -511, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -512, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffbf"))),
            Reference(100UL, T(false, -265, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -266, UInt!256.fromHexString("e82bbc2a781ed0fadd053012bc81d36d283f7cae01c8ed11916fddc996a18599"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            const actual = geometricPMF(cast(size_t)c.k, c.p);
            assert(actual == c.expected);
            assert(negativeBinomialPMF(cast(size_t)c.k, 1, c.p) == actual);
        }
    }
}
// END geometric references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.distribution.negative_binomial: negativeBinomialPMF;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (k; [size_t(0), size_t(1), size_t.max]) {
            assert(geometricPMF(k, T(0)) == T(0));
            assert(geometricPMF(k, T(1)) == T(k == 0 ? 1 : 0));
            auto p = T(false, long.min, UInt!bits(1));
            assert(geometricPMF(k, p) == p);
            assert(negativeBinomialPMF(k, 1, p) == p);
        }
        static if (size_t.sizeof == 8) {
            const p = T(0.5);
            assert(geometricPMF(cast(size_t)long.max, p) == T(false, long.min, UInt!bits(1)));
            assert(geometricPMF(cast(size_t)long.max + 1, p) == T(0));
            assert(geometricPMF(size_t.max, p) == T(0));
        }
        static assert(is(typeof(geometricPMF(1, T(0.5))) == T));
    }}
}

version(mir_stat_test)
@system unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        foreach (p; [-1.0, 2.0, double.infinity, -double.infinity, double.nan])
            assertThrown!AssertError(geometricPMF(1, Fp!128(p)));
    }
}

/// The probability type selects the extended-precision PMF.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    const probability = Fp!128(0.25);
    const mass = geometricPMF(2, probability);
    assert(mass == Fp!128(0.140625));
}

// The tiny-probability shortcut must agree with guarded evaluation on either
// side of its exponent threshold, including nonnormalized coefficients.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.stat.internal.fp_probability: fpProbabilityLess;
    static foreach (bits; [64u, 128u, 256u]) {{
        alias T = Fp!bits;
        foreach (e; [-2L * bits - 67, -2L * bits - 66, -2L * bits - 65]) {
            auto p = T(1);
            p.exponent = e;
            assert(geometricPMF(size_t.max, p) == p);
            const raw = T(false, e, UInt!bits(1));
            const actual = geometricPMF(size_t.max, raw);
            assert(!fpProbabilityLess(actual, raw) && !fpProbabilityLess(raw, actual));
        }
    }}
}

// BEGIN geometric-tail references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { ulong k; T p, cdf, ccdf; }
        enum references = [
            Reference(0UL, T(false, -64, UInt!64.fromHexString("8000000000000000")), T(false, -64L, UInt!64.fromHexString("8000000000000000")), T(false, -64L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, T(false, -65, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("e000000000000000")), T(false, -64L, UInt!64.fromHexString("9000000000000000"))),
            Reference(17UL, T(false, -64, UInt!64.fromHexString("c000000000000000")), T(false, -64L, UInt!64.fromHexString("fffffffff0000000")), T(false, -99L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1000UL, T(false, -66, UInt!64.fromHexString("8000000000000000")), T(false, -63L, UInt!64.fromHexString("8000000000000000")), T(false, -256L, UInt!64.fromHexString("8f3d17ed50185bbe"))),
            Reference(3UL, T(false, -64, UInt!64.fromHexString("ffffffffffffffff")), T(false, -63L, UInt!64.fromHexString("8000000000000000")), T(false, -319L, UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL, T(false, -1163, UInt!64.fromHexString("8000000000000000")), T(false, -1162L, UInt!64.fromHexString("8000000000000000")), T(false, -63L, UInt!64.fromHexString("8000000000000000"))),
            Reference(2UL, T(false, -1100, UInt!64.fromHexString("8000000000000001")), T(false, -1099L, UInt!64.fromHexString("c000000000000001")), T(false, -63L, UInt!64.fromHexString("8000000000000000"))),
            Reference(18446744073709551615UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -64L, UInt!64.fromHexString("a1d2a7274c4320e5")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be35"))),
            Reference(18446744073709551615UL, T(false, -128, UInt!64.fromHexString("8000000000000000")), T(false, -65L, UInt!64.fromHexString("c974d039069f6018")), T(false, -64L, UInt!64.fromHexString("9b4597e37cb04ff4"))),
            Reference(1152921504606846976UL, T(false, -123, UInt!64.fromHexString("8000000000000000")), T(false, -64L, UInt!64.fromHexString("a1d2a7274c4320ee")), T(false, -65L, UInt!64.fromHexString("bc5ab1b16779be24"))),
            Reference(65UL, T(false, -127, UInt!64.fromHexString("8000000000000000")), T(false, -121L, UInt!64.fromHexString("83ffffffffffffef")), T(false, -64L, UInt!64.fromHexString("ffffffffffffffbe"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            assert(geometricCDF(cast(size_t)c.k, c.p) == c.cdf);
            assert(geometricCCDF(cast(size_t)c.k, c.p) == c.ccdf);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; T p, cdf, ccdf; }
        enum references = [
            Reference(0UL, T(false, -128, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, T(false, -129, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -129L, UInt!128.fromHexString("e0000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("90000000000000000000000000000000"))),
            Reference(17UL, T(false, -128, UInt!128.fromHexString("c0000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("fffffffff00000000000000000000000")), T(false, -163L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1000UL, T(false, -130, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -127L, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -320L, UInt!128.fromHexString("8f3d17ed50185bbde6b26e3d1f035b07"))),
            Reference(3UL, T(false, -128, UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")), T(false, -127L, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -639L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL, T(false, -1227, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -1226L, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -127L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(2UL, T(false, -1100, UInt!128.fromHexString("80000000000000000000000000000001")), T(false, -1099L, UInt!128.fromHexString("c0000000000000000000000000000001")), T(false, -127L, UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(18446744073709551615UL, T(false, -191, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("a1d2a7274c4320e57437e4e9c9897680")), T(false, -129L, UInt!128.fromHexString("bc5ab1b16779be351790362c6ced1301"))),
            Reference(18446744073709551615UL, T(false, -256, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -193L, UInt!128.fromHexString("ffffffffffffffffc000000000000000")), T(false, -128L, UInt!128.fromHexString("ffffffffffffffff8000000000000000"))),
            Reference(1152921504606846976UL, T(false, -187, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -128L, UInt!128.fromHexString("a1d2a7274c4320ee19618cce495ff158")), T(false, -129L, UInt!128.fromHexString("bc5ab1b16779be23cd3ce6636d401d50"))),
            Reference(65UL, T(false, -255, UInt!128.fromHexString("80000000000000000000000000000000")), T(false, -249L, UInt!128.fromHexString("83ffffffffffffffffffffffffffffef")), T(false, -128L, UInt!128.fromHexString("ffffffffffffffffffffffffffffffbe"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            assert(geometricCDF(cast(size_t)c.k, c.p) == c.cdf);
            assert(geometricCCDF(cast(size_t)c.k, c.p) == c.ccdf);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; T p, cdf, ccdf; }
        enum references = [
            Reference(0UL, T(false, -256, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, T(false, -257, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -257L, UInt!256.fromHexString("e000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("9000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(17UL, T(false, -256, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("fffffffff0000000000000000000000000000000000000000000000000000000")), T(false, -291L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1000UL, T(false, -258, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffff70c2e812afe7a442")), T(false, -448L, UInt!256.fromHexString("8f3d17ed50185bbde6b26e3d1f035b06d4b7dd883a5cc1bb456c0db9b8950468"))),
            Reference(3UL, T(false, -256, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")), T(false, -255L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1279L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL, T(false, -1355, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -1354L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -255L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(2UL, T(false, -1100, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")), T(false, -1099L, UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000001")), T(false, -255L, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL, T(false, -319, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("a1d2a7274c4320e57437e4e9c989767f967625ee62dab3ceade5168126791677")), T(false, -257L, UInt!256.fromHexString("bc5ab1b16779be351790362c6ced1300d313b4233a4a9862a435d2fdb30dd312"))),
            Reference(18446744073709551615UL, T(false, -512, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -449L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffc000000000000000")), T(false, -256L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffff8000000000000000"))),
            Reference(1152921504606846976UL, T(false, -315, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -256L, UInt!256.fromHexString("a1d2a7274c4320ee19618cce495ff15832661a0910c9cf05d6e2def009917cfb")), T(false, -257L, UInt!256.fromHexString("bc5ab1b16779be23cd3ce6636d401d4f9b33cbedde6c61f4523a421fecdd060a"))),
            Reference(65UL, T(false, -511, UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")), T(false, -505L, UInt!256.fromHexString("83ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffef")), T(false, -256L, UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffbe"))),
        ];
        foreach (c; references) {
            if (c.k > size_t.max) continue;
            assert(geometricCDF(cast(size_t)c.k, c.p) == c.cdf);
            assert(geometricCCDF(cast(size_t)c.k, c.p) == c.ccdf);
        }
    }
}
// END geometric-tail references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        foreach (k; [size_t(0),size_t(1),size_t.max]) {
            assert(geometricCDF(k,T(0)) == T(0));
            assert(geometricCCDF(k,T(0)) == T(1));
            assert(geometricCDF(k,T(1)) == T(1));
            assert(geometricCCDF(k,T(1)) == T(0));
        }
        auto tiny = T(false,long.min,UInt!bits(1));
        assert(geometricCDF(1,tiny) == T(false,long.min,UInt!bits(2)));
        assert(geometricCCDF(size_t.max,tiny) == T(1));
        assert(geometricCDF(size_t.max,T(0.5)) == T(1));
        assert(geometricCCDF(size_t.max,T(0.5)) == T(0));
        auto count = UInt!128(size_t.max); count += 1;
        T expected;
        static if (bits == size_t.sizeof * 8)
            expected = T(false,long.min+1,UInt!bits(1) << (bits-1));
        else
            expected = T(false,long.min,count.toSize!bits);
        assert(geometricCDF(size_t.max,tiny) == expected);
        auto m = UInt!bits(1); m <<= bits-1; m += 1;
        const p = T(false,long.min,m);
        auto product = m.toSize!(bits+64); product *= UInt!64(3); product >>= 1;
        assert(geometricCDF(2,p) == T(false,long.min+1,product.toSize!bits));
        static if (size_t.sizeof == 8) {
            assert(geometricCCDF(cast(size_t)long.max,T(0.5)) == tiny);
            assert(geometricCCDF(cast(size_t)long.max+1,T(0.5)) == T(0));
        }
    }}
}

/// Extended-precision tails accept integer thresholds.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    const p = Fp!128(0.25);
    assert(geometricCDF(1,p) == Fp!128(0.4375));
    assert(geometricCCDF(1,p) == Fp!128(0.5625));
    static assert(is(typeof(geometricCDF(1,p)) == Fp!128));
    static assert(is(typeof(geometricCCDF(1,p)) == Fp!128));
    static assert(!__traits(compiles, geometricCDF(Fp!128(1),p)));
}

version(mir_stat_test)
@system unittest
{
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        import core.exception: AssertError;
        import std.exception: assertThrown;
        foreach (p; [-1.0,2.0,double.infinity,double.nan]) {
            assertThrown!AssertError(geometricCDF(0,Fp!128(p)));
            assertThrown!AssertError(geometricCCDF(0,Fp!128(p)));
        }
    }
}

// BEGIN geometric probability log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        alias T = Fp!64;
        struct Reference { T p; ulong k; T logMass; }
        enum references = [
            Reference(T(false,long.min,UInt!64.fromHexString("1")),17UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,long.min,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,-2000,UInt!64.fromHexString("1")),17UL,T(true,-53,UInt!64.fromHexString("ad496b6802e49cd2"))),
            Reference(T(false,-2000,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,-53,UInt!64.fromHexString("ad496b6802e49cd2"))),
            Reference(T(false,-385,UInt!64.fromHexString("1")),17UL,T(true,-55,UInt!64.fromHexString("856e4b05d94482fe"))),
            Reference(T(false,-385,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,-55,UInt!64.fromHexString("856e4b05d94482fe"))),
            Reference(T(false,-384,UInt!64.fromHexString("1")),17UL,T(true,-55,UInt!64.fromHexString("851591f9dd5b9b41"))),
            Reference(T(false,-384,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,-55,UInt!64.fromHexString("851591f9dd5b9b41"))),
            Reference(T(false,-64,UInt!64.fromHexString("1")),17UL,T(true,-58,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,-64,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,-58,UInt!64.fromHexString("b57217f7d1cf79ac"))),
            Reference(T(false,-1,UInt!64.fromHexString("1")),17UL,T(true,-60,UInt!64.fromHexString("c7a05af6cc0968e1"))),
            Reference(T(false,-1,UInt!64.fromHexString("1")),18446744073709551615UL,T(true,0,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,long.min,UInt!64.fromHexString("3")),17UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(T(false,long.min,UInt!64.fromHexString("3")),18446744073709551615UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(T(false,-2,UInt!64.fromHexString("3")),17UL,T(true,-59,UInt!64.fromHexString("bed665b97587e396"))),
            Reference(T(false,-2,UInt!64.fromHexString("3")),18446744073709551615UL,T(true,1,UInt!64.fromHexString("b17217f7d1cf79ab"))),
            Reference(T(false,-195,UInt!64.fromHexString("155d3680a0b3f28b")),17UL,T(true,-57,UInt!64.fromHexString("ba924d3daa964b98"))),
            Reference(T(false,-195,UInt!64.fromHexString("155d3680a0b3f28b")),18446744073709551615UL,T(true,-57,UInt!64.fromHexString("ba924d3daa964b98"))),
            Reference(T(false,-277,UInt!64.fromHexString("2dfcf807a025849a")),17UL,T(true,-56,UInt!64.fromHexString("955b6db83dc9254e"))),
            Reference(T(false,-277,UInt!64.fromHexString("2dfcf807a025849a")),18446744073709551615UL,T(true,-56,UInt!64.fromHexString("955b6db83dc9254e"))),
            Reference(T(false,-349,UInt!64.fromHexString("47da373f7dfb4d2f")),17UL,T(true,-56,UInt!64.fromHexString("c6d14852b454feba"))),
            Reference(T(false,-349,UInt!64.fromHexString("47da373f7dfb4d2f")),18446744073709551615UL,T(true,-56,UInt!64.fromHexString("c6d14852b454feba"))),
            Reference(T(false,-111,UInt!64.fromHexString("6ce38d733ed6ff1b")),17UL,T(true,-58,UInt!64.fromHexString("85bb2748be58c319"))),
            Reference(T(false,-111,UInt!64.fromHexString("6ce38d733ed6ff1b")),18446744073709551615UL,T(true,-48,UInt!64.fromHexString("d9e889b04fddf0f9"))),
            Reference(T(false,-97,UInt!64.fromHexString("739145488dc18dc8")),17UL,T(true,-59,UInt!64.fromHexString("bd5a7b03e666fdb4"))),
            Reference(T(false,-97,UInt!64.fromHexString("739145488dc18dc8")),18446744073709551615UL,T(true,-34,UInt!64.fromHexString("e7228aefe2d688ca"))),
            Reference(T(false,-64,UInt!64.fromHexString("7fffffffffffffff")),17UL,T(true,-60,UInt!64.fromHexString("c7a05af6cc0968df"))),
            Reference(T(false,-64,UInt!64.fromHexString("7fffffffffffffff")),18446744073709551615UL,T(true,0,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(T(false,long.min,UInt!64.fromHexString("8000000000000000")),17UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf7954"))),
            Reference(T(false,long.min,UInt!64.fromHexString("8000000000000000")),18446744073709551615UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf7954"))),
            Reference(T(false,-64,UInt!64.fromHexString("8000000000000001")),17UL,T(true,-60,UInt!64.fromHexString("c7a05af6cc0968e3"))),
            Reference(T(false,-64,UInt!64.fromHexString("8000000000000001")),18446744073709551615UL,T(true,0,UInt!64.fromHexString("b17217f7d1cf79ae"))),
            Reference(T(false,-245,UInt!64.fromHexString("c3590e78d844c8c3")),17UL,T(true,-57,UInt!64.fromHexString("fb75c72b86861549"))),
            Reference(T(false,-245,UInt!64.fromHexString("c3590e78d844c8c3")),18446744073709551615UL,T(true,-57,UInt!64.fromHexString("fb75c72b86861549"))),
            Reference(T(false,-182,UInt!64.fromHexString("c4283a1bd25a8240")),17UL,T(true,-57,UInt!64.fromHexString("a41d81821f3b6956"))),
            Reference(T(false,-182,UInt!64.fromHexString("c4283a1bd25a8240")),18446744073709551615UL,T(true,-57,UInt!64.fromHexString("a41d81821f3b695c"))),
            Reference(T(false,-203,UInt!64.fromHexString("dbc71f0e08eae614")),17UL,T(true,-57,UInt!64.fromHexString("c100026a481f7aa7"))),
            Reference(T(false,-203,UInt!64.fromHexString("dbc71f0e08eae614")),18446744073709551615UL,T(true,-57,UInt!64.fromHexString("c100026a481f7aa7"))),
            Reference(T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),17UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf7953"))),
            Reference(T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),18446744073709551615UL,T(true,-1,UInt!64.fromHexString("b17217f7d1cf7953"))),
            Reference(T(false,-385,UInt!64.fromHexString("ffffffffffffffff")),17UL,T(true,-56,UInt!64.fromHexString("de80100dbe152790"))),
            Reference(T(false,-385,UInt!64.fromHexString("ffffffffffffffff")),18446744073709551615UL,T(true,-56,UInt!64.fromHexString("de80100dbe152790"))),
            Reference(T(false,-384,UInt!64.fromHexString("ffffffffffffffff")),17UL,T(true,-56,UInt!64.fromHexString("ddce9df5c6435817"))),
            Reference(T(false,-384,UInt!64.fromHexString("ffffffffffffffff")),18446744073709551615UL,T(true,-56,UInt!64.fromHexString("ddce9df5c6435817"))),
            Reference(T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),17UL,T(true,-54,UInt!64.fromHexString("bc8939774eec7147"))),
            Reference(T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),18446744073709551615UL,T(true,6,UInt!64.fromHexString("b17217f7d1cf79ab"))),
        ];
        foreach (r; references) {
            if (r.k <= size_t.max)
                assert(geometricLPMF(cast(size_t)r.k,r.p) == r.logMass);
        }
    }
    {
        alias T = Fp!128;
        struct Reference { T p; ulong k; T logMass; }
        enum references = [
            Reference(T(false,long.min,UInt!128.fromHexString("1")),17UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,long.min,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-2000,UInt!128.fromHexString("1")),17UL,T(true,-117,UInt!128.fromHexString("ad496b6802e49cd1c3285d6273db44e7"))),
            Reference(T(false,-2000,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-117,UInt!128.fromHexString("ad496b6802e49cd1c3285d6273db44e7"))),
            Reference(T(false,-513,UInt!128.fromHexString("1")),17UL,T(true,-119,UInt!128.fromHexString("b1cad103cdb861689fc8a571cff4f02b"))),
            Reference(T(false,-513,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-119,UInt!128.fromHexString("b1cad103cdb861689fc8a571cff4f02b"))),
            Reference(T(false,-512,UInt!128.fromHexString("1")),17UL,T(true,-119,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-512,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-119,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-128,UInt!128.fromHexString("1")),17UL,T(true,-121,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-128,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-121,UInt!128.fromHexString("b17217f7d1cf79abcbe3b39803f2f6af"))),
            Reference(T(false,-1,UInt!128.fromHexString("1")),17UL,T(true,-124,UInt!128.fromHexString("c7a05af6cc0968e143202a0b04715585"))),
            Reference(T(false,-1,UInt!128.fromHexString("1")),18446744073709551615UL,T(true,-64,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,long.min,UInt!128.fromHexString("3")),17UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf79a9976664426391a3c2"))),
            Reference(T(false,long.min,UInt!128.fromHexString("3")),18446744073709551615UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf79a9976664426391a3c2"))),
            Reference(T(false,-2,UInt!128.fromHexString("3")),17UL,T(true,-123,UInt!128.fromHexString("bed665b97587e3958dff5da05c47799f"))),
            Reference(T(false,-2,UInt!128.fromHexString("3")),18446744073709551615UL,T(true,-63,UInt!128.fromHexString("b17217f7d1cf79ab3d445fc29bdaa1f4"))),
            Reference(T(false,-382,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff")),17UL,T(true,-120,UInt!128.fromHexString("b64bbf32e5853dc6a1241a3a00126877"))),
            Reference(T(false,-382,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff")),18446744073709551615UL,T(true,-120,UInt!128.fromHexString("b64bbf32e5853dc6a1241a3a00126877"))),
            Reference(T(false,-191,UInt!128.fromHexString("a80121e07c0fd2678904cf45e2c018")),17UL,T(true,-122,UInt!128.fromHexString("c689de4538e16ebd0e80f43797751a76"))),
            Reference(T(false,-191,UInt!128.fromHexString("a80121e07c0fd2678904cf45e2c018")),18446744073709551615UL,T(true,-122,UInt!128.fromHexString("c68f1e4e47e54f3ba15ebf2d0f4226e9"))),
            Reference(T(false,-360,UInt!128.fromHexString("64f2572158cb0b0e40975b1d17ec17bd")),17UL,T(true,-120,UInt!128.fromHexString("a1bda0903fb3ed35ce48f971d1474472"))),
            Reference(T(false,-360,UInt!128.fromHexString("64f2572158cb0b0e40975b1d17ec17bd")),18446744073709551615UL,T(true,-120,UInt!128.fromHexString("a1bda0903fb3ed35ce48f971d1474472"))),
            Reference(T(false,-128,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),17UL,T(true,-124,UInt!128.fromHexString("c7a05af6cc0968e143202a0b04715583"))),
            Reference(T(false,-128,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-64,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6ad"))),
            Reference(T(false,long.min,UInt!128.fromHexString("80000000000000000000000000000000")),17UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fbbaafebb5d8183e3d"))),
            Reference(T(false,long.min,UInt!128.fromHexString("80000000000000000000000000000000")),18446744073709551615UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fbbaafebb5d8183e3d"))),
            Reference(T(false,-128,UInt!128.fromHexString("80000000000000000000000000000001")),17UL,T(true,-124,UInt!128.fromHexString("c7a05af6cc0968e143202a0b04715587"))),
            Reference(T(false,-128,UInt!128.fromHexString("80000000000000000000000000000001")),18446744073709551615UL,T(true,-64,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6b1"))),
            Reference(T(false,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb")),17UL,T(true,-121,UInt!128.fromHexString("e8c44510834cce7c54fe08dccac9d95f"))),
            Reference(T(false,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb")),18446744073709551615UL,T(true,-121,UInt!128.fromHexString("e8c44510834cce7c54fe08dccacbfbf8"))),
            Reference(T(false,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24")),17UL,T(true,-120,UInt!128.fromHexString("a2c394eebd064edee58f98f5a40c10f3"))),
            Reference(T(false,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24")),18446744073709551615UL,T(true,-120,UInt!128.fromHexString("a2c394eebd064edee58f98f5a40c10f3"))),
            Reference(T(false,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b0e9db0254")),17UL,T(true,-121,UInt!128.fromHexString("849dceb8353d6aebee7b02b26b61b008"))),
            Reference(T(false,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b0e9db0254")),18446744073709551615UL,T(true,-121,UInt!128.fromHexString("849dceb837c458af814fc6f1e07183a2"))),
            Reference(T(false,-201,UInt!128.fromHexString("d4daf2ca7a55cf775dd9cd750c9be8c2")),17UL,T(true,-122,UInt!128.fromHexString("cb2321f2965d0451d359f1b047ec22da"))),
            Reference(T(false,-201,UInt!128.fromHexString("d4daf2ca7a55cf775dd9cd750c9be8c2")),18446744073709551615UL,T(true,-122,UInt!128.fromHexString("cb24cba87bf1f8fd722abef23c0bb8e1"))),
            Reference(T(false,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc04caadc")),17UL,T(true,-121,UInt!128.fromHexString("91ecd8d283c2b13d18884a2d32cb37a4"))),
            Reference(T(false,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc04caadc")),18446744073709551615UL,T(true,-121,UInt!128.fromHexString("91ecd8d283c3869c77b53637e6567142"))),
            Reference(T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),17UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5"))),
            Reference(T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5"))),
            Reference(T(false,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),17UL,T(true,-119,UInt!128.fromHexString("856e4b05d94482fdad4fb88bcef8327f"))),
            Reference(T(false,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-119,UInt!128.fromHexString("856e4b05d94482fdad4fb88bcef8327f"))),
            Reference(T(false,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),17UL,T(true,-119,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903"))),
            Reference(T(false,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-119,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903"))),
            Reference(T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),17UL,T(true,-117,UInt!128.fromHexString("bc8939774eec71468681eed18432261a"))),
            Reference(T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-57,UInt!128.fromHexString("b17217f7d1cf79ab18719ba032237d03"))),
        ];
        foreach (r; references) {
            if (r.k <= size_t.max)
                assert(geometricLPMF(cast(size_t)r.k,r.p) == r.logMass);
        }
    }
    {
        alias T = Fp!256;
        struct Reference { T p; ulong k; T logMass; }
        enum references = [
            Reference(T(false,long.min,UInt!256.fromHexString("1")),17UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,long.min,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-2000,UInt!256.fromHexString("1")),17UL,T(true,-245,UInt!256.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84f"))),
            Reference(T(false,-2000,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-245,UInt!256.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84f"))),
            Reference(T(false,-769,UInt!256.fromHexString("1")),17UL,T(true,-246,UInt!256.fromHexString("8541ee7fdb500f1f425d3f9ee8f735c11c86af2d9f8f2ecfb2ec54ca7fa3265f"))),
            Reference(T(false,-769,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-246,UInt!256.fromHexString("8541ee7fdb500f1f425d3f9ee8f735c11c86af2d9f8f2ecfb2ec54ca7fa3265f"))),
            Reference(T(false,-768,UInt!256.fromHexString("1")),17UL,T(true,-246,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1"))),
            Reference(T(false,-768,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-246,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1"))),
            Reference(T(false,-256,UInt!256.fromHexString("1")),17UL,T(true,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-256,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8b0d175b8baafa2c"))),
            Reference(T(false,-1,UInt!256.fromHexString("1")),17UL,T(true,-252,UInt!256.fromHexString("c7a05af6cc0968e143202a0b047155852911ab8b40ebccf33b4eba46fd205971"))),
            Reference(T(false,-1,UInt!256.fromHexString("1")),18446744073709551615UL,T(true,-192,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,long.min,UInt!256.fromHexString("3")),17UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf79a9976664426391a3c1f8c0287c6821bcc20de491a81be9de4b"))),
            Reference(T(false,long.min,UInt!256.fromHexString("3")),18446744073709551615UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf79a9976664426391a3c1f8c0287c6821bcc20de491a81be9de4b"))),
            Reference(T(false,-2,UInt!256.fromHexString("3")),17UL,T(true,-251,UInt!256.fromHexString("bed665b97587e3958dff5da05c47799f7b210974732cc883b961facfaa960fc5"))),
            Reference(T(false,-2,UInt!256.fromHexString("3")),18446744073709551615UL,T(true,-191,UInt!256.fromHexString("b17217f7d1cf79ab3d445fc29bdaa1f3eee67c7beffaf7d2ab02f5eeafbab334"))),
            Reference(T(false,-503,UInt!256.fromHexString("1e9b726592d30ba658d702c4cd40a6d64c302ae640625db7771e120788e25f79")),17UL,T(true,-248,UInt!256.fromHexString("ad54cfc4ceb558d41b83731273e0d5b69e1baeab78d4c672c565f798f4bc71bf"))),
            Reference(T(false,-503,UInt!256.fromHexString("1e9b726592d30ba658d702c4cd40a6d64c302ae640625db7771e120788e25f79")),18446744073709551615UL,T(true,-248,UInt!256.fromHexString("ad54cfc4ceb558d41b83731273e0d5b69e1baeab78d4c673029cdc641a628908"))),
            Reference(T(false,-500,UInt!256.fromHexString("3ec4002d4882689ddd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a0739")),17UL,T(true,-248,UInt!256.fromHexString("aa889f60a17a6ac22409e5390478487377a9195c206d0f262c1ffd000e226110"))),
            Reference(T(false,-500,UInt!256.fromHexString("3ec4002d4882689ddd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a0739")),18446744073709551615UL,T(true,-248,UInt!256.fromHexString("aa889f60a17a6ac22409e5390478487377a9195c206d0f2a185fffd49648eaa8"))),
            Reference(T(false,-310,UInt!256.fromHexString("4720f4a871f2ec93ca5547d5408492cb24034b668fda7a54a82657222d4b3a8e")),17UL,T(true,-250,UInt!256.fromHexString("9ad7b0641c6f5db36a59a64727f5d81769c4b0ea9af51bb4100fb505ac92319a"))),
            Reference(T(false,-310,UInt!256.fromHexString("4720f4a871f2ec93ca5547d5408492cb24034b668fda7a54a82657222d4b3a8e")),18446744073709551615UL,T(true,-247,UInt!256.fromHexString("a19cdf5d6773c5230e832acfeaeed8a2e704a1001239fed6e7a3b6b6173d9911"))),
            Reference(T(false,-423,UInt!256.fromHexString("4d11226bb2db7011471dca27dde7b62d311503cec0f7abd7f1bd17a3c338aca7")),17UL,T(true,-249,UInt!256.fromHexString("e9e9839efca15d57a7d00a402ee7e8cff36f3691f11c70786134af59c7271fcb"))),
            Reference(T(false,-423,UInt!256.fromHexString("4d11226bb2db7011471dca27dde7b62d311503cec0f7abd7f1bd17a3c338aca7")),18446744073709551615UL,T(true,-249,UInt!256.fromHexString("e9e9839efca15d57a7d00a402ee91d147d1e01ffb14be01ddb89d9411affe3be"))),
            Reference(T(false,-430,UInt!256.fromHexString("7612335515ecf948ebdb260951f5a219d41440275124cbcfcbdd58ef250485fa")),17UL,T(true,-249,UInt!256.fromHexString("f2c353f4ddfc69c00ea9259be92cc6e4619596465661991ff31e51978df6c471"))),
            Reference(T(false,-430,UInt!256.fromHexString("7612335515ecf948ebdb260951f5a219d41440275124cbcfcbdd58ef250485fa")),18446744073709551615UL,T(true,-249,UInt!256.fromHexString("f2c353f4ddfc69c00ea9259be92cca94f3303ef5be2b9e148f6ebfd1eece8e66"))),
            Reference(T(false,-256,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),17UL,T(true,-252,UInt!256.fromHexString("c7a05af6cc0968e143202a0b047155852911ab8b40ebccf33b4eba46fd20596f"))),
            Reference(T(false,-256,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-192,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2a"))),
            Reference(T(false,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),17UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf784a4897f3e4089e92730d537a4e9491450a256d50c33f700b73"))),
            Reference(T(false,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),18446744073709551615UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf784a4897f3e4089e92730d537a4e9491450a256d50c33f700b73"))),
            Reference(T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),17UL,T(true,-252,UInt!256.fromHexString("c7a05af6cc0968e143202a0b047155852911ab8b40ebccf33b4eba46fd205973"))),
            Reference(T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),18446744073709551615UL,T(true,-192,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2e"))),
            Reference(T(false,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de07984ea35c2a2dc0399d61d2294e0d")),17UL,T(true,-249,UInt!256.fromHexString("cfc182533e5dac574b7d8fee956fa9ce8b876a347765cb08d69b84728099c0cd"))),
            Reference(T(false,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de07984ea35c2a2dc0399d61d2294e0d")),18446744073709551615UL,T(true,-249,UInt!256.fromHexString("cfc182533e5dac574b7d8ff75f94099fd73b81c9293aaf490112cc2c0a1207a1"))),
            Reference(T(false,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa08801907f3462d0e84f25bca22b80357ef032e8edb2d5a5")),17UL,T(true,-250,UInt!256.fromHexString("e4e42a2e3d8a4bda61b94e64ec6ca2aa322657cc8842ffdf4ddc0a81630d83c7"))),
            Reference(T(false,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa08801907f3462d0e84f25bca22b80357ef032e8edb2d5a5")),18446744073709551615UL,T(true,-250,UInt!256.fromHexString("e4e42adc7ac6ceb5ab878cc5263bd2a7d300c1b92f3b684c36445f4865feda9b"))),
            Reference(T(false,-294,UInt!256.fromHexString("eee66cfb0ccb08a554bda8b2c9fdfd2bcc24375ff6aefe360e542751a3681ab8")),17UL,T(true,-251,UInt!256.fromHexString("d3451186377f4751cd23d60b1a864e70a7ffb47cd46ee55d6a5474d0a5166618"))),
            Reference(T(false,-294,UInt!256.fromHexString("eee66cfb0ccb08a554bda8b2c9fdfd2bcc24375ff6aefe360e542751a3681ab8")),18446744073709551615UL,T(true,-230,UInt!256.fromHexString("eee6739537151cd4986375a326be27d906133aacce4c5c246d5d836f57c722ba"))),
            Reference(T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),17UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18"))),
            Reference(T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-193,UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18"))),
            Reference(T(false,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),17UL,T(true,-247,UInt!256.fromHexString("b1cad103cdb861689fc8a571cff4f02a9893bcc805d20288a0d21de73970cfa9"))),
            Reference(T(false,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-247,UInt!256.fromHexString("b1cad103cdb861689fc8a571cff4f02a9893bcc805d20288a0d21de73970cfa9"))),
            Reference(T(false,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),17UL,T(true,-247,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-247,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),17UL,T(true,-244,UInt!256.fromHexString("bc8939774eec71468681eed18432261a35027758d9c2419062ade8d14465a9cf"))),
            Reference(T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),18446744073709551615UL,T(true,-184,UInt!256.fromHexString("b17217f7d1cf79ab18719ba032237d03770f8f8e6ea5bf7e4919d435191243fe"))),
        ];
        foreach (r; references) {
            if (r.k <= size_t.max)
                assert(geometricLPMF(cast(size_t)r.k,r.p) == r.logMass);
        }
    }
}
// END geometric probability log references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    foreach (T; AliasSeq!(Fp!64,Fp!128,Fp!256,Fp!512)) {
        const zero = T(0);
        const one = T(1);
        const infinity = T(-double.infinity);
        assert(geometricLPMF(0,zero) == infinity);
        assert(geometricLPMF(0,one) == zero);
        assert(geometricLPMF(1,one) == infinity);
        // Exact-one endpoints must also accept an unnormalized coefficient.
        const rawOne = T(false,0,UInt!(typeof(T.init.coefficient).sizeof * 8)(1));
        assert(geometricLPMF(0,rawOne) == zero);
        assert(geometricLPMF(size_t.max,rawOne) == infinity);
        T negativeZero = zero;
        negativeZero.sign = true;
        assert(geometricLPMF(0,negativeZero) == infinity);
        auto p = T(0.25);
        const original = p;
        p.coefficient >>= 1;
        p.exponent += 1;
        assert(geometricLPMF(17,p) == geometricLPMF(17,original));
        static assert(is(typeof(geometricLPMF(0,one)) == T));
        static assert(!__traits(compiles,geometricLPMF(0,1)));
        debug {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            foreach (bad; [T(-0.5),T(1.5),T(double.infinity),T(double.nan)]) {
                assertThrown!AssertError(geometricLPMF(0,bad));
            }
        }
    }
}
