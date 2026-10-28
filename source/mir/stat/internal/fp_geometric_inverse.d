/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_geometric_inverse;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;
import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityLess;

/++
Outcome of the package-scoped geometric inverse prototype.

These names and the checked-result interface are candidates for a future public
API. They are not exported by mir.stat.distribution.geometric.
+/
package(mir.stat) enum GeometricInverseStatus
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
package(mir.stat) struct GeometricInverseResult
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

private enum Comparison { below, reached, unresolved }

// Fixed-point intervals on [0,1]. Both endpoints are integers scaled by
// 2^-fractionBits. Products use exact wide integers, rounding outward once.
private struct ProbabilityInterval(uint fractionBits)
{
    alias Integer = UInt!(fractionBits + 64);
    Integer lower, upper;

    static Integer one() @safe pure nothrow @nogc
    {
        auto result = Integer(1);
        result <<= fractionBits;
        return result;
    }

    static ProbabilityInterval from(uint bits)(Fp!bits p) @safe pure nothrow @nogc
    {
        auto value = p.coefficient.toSize!(fractionBits + 64);
        const exponent = p.exponent + fractionBits;
        if (exponent >= 0) {
            value <<= cast(size_t)exponent;
            return ProbabilityInterval(value,value);
        }
        const shift = cast(ulong)-exponent;
        if (shift >= bits)
            return ProbabilityInterval(Integer(0),Integer(p.coefficient ? 1 : 0));
        auto lower = value >> cast(size_t)shift;
        auto upper = lower;
        if ((lower << cast(size_t)shift) != value)
            upper += 1;
        return ProbabilityInterval(lower,upper);
    }

    private static Integer product(Integer a, Integer b, bool upward)
        @safe pure nothrow @nogc
    {
        auto exact = a.toSize!(2 * fractionBits + 128);
        exact *= b;
        auto quotient = exact >> fractionBits;
        if (upward && (quotient << fractionBits) != exact)
            quotient += 1;
        return quotient.toSize!(fractionBits + 64);
    }

    ProbabilityInterval multiply(ProbabilityInterval rhs) const
        @safe pure nothrow @nogc
    {
        return ProbabilityInterval(product(lower,rhs.lower,false),
            product(upper,rhs.upper,true));
    }
}

private @safe pure nothrow @nogc
Comparison compareAt(uint fractionBits, uint bits)(size_t k, Fp!bits p, Fp!bits q)
{
    alias I = ProbabilityInterval!fractionBits;
    alias W = Fp!(fractionBits + 64);
    const probability = I.from(p);
    const unit = I.one;
    auto factor = I(unit-probability.upper,unit-probability.lower);
    // Begin with one failure factor, then raise to k more. k+1 never needs
    // to fit size_t, including when k is the largest representable count.
    auto tail = factor;
    size_t count = k;
    while (count) {
        if (count & 1)
            tail = tail.multiply(factor);
        count >>= 1;
        if (count)
            factor = factor.multiply(factor);
    }
    // Raw coefficient/exponent construction preserves very tiny queries;
    // normalizing a widened Fp at long.min would underflow its exponent.
    const query = W(false,q.exponent,q.coefficient.toSize!(fractionBits + 64));
    const lower = W(false,-cast(long)fractionBits,unit-tail.upper);
    const upper = W(false,-cast(long)fractionBits,unit-tail.lower);
    if (!fpProbabilityLess(lower,query))
        return Comparison.reached;
    if (fpProbabilityLess(upper,query))
        return Comparison.below;
    return Comparison.unresolved;
}

private @safe pure nothrow @nogc
Comparison compare(uint bits)(size_t k, Fp!bits p, Fp!bits q)
{
    if (!k)
        return fpProbabilityLess(p,q) ? Comparison.below : Comparison.reached;
    // The union bound CDF(k) <= (k+1)*p proves many out-of-range cases
    // without resolving 1-p on a fixed-point grid.
    auto trials = UInt!128(k);
    trials += 1;
    auto product = p.coefficient.toSize!(bits + 128);
    product *= trials;
    alias W = Fp!(bits + 128);
    const bound = W(false,p.exponent,product);
    const query = W(false,q.exponent,q.coefficient.toSize!(bits + 128));
    if (fpProbabilityLess(bound,query))
        return Comparison.below;
    auto result = compareAt!(bits + 128)(k,p,q);
    if (result == Comparison.unresolved)
        result = compareAt!(bits + 512)(k,p,q);
    return result;
}

/++
Candidate checked inverse CDF for represented Fp probabilities.

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
package(mir.stat) @safe pure nothrow @nogc
GeometricInverseResult geometricInvCDFChecked(uint bits)(Fp!bits q, Fp!bits p,
    size_t maxCount = size_t.max)
    in (fpProbabilityValid(p), "p must be finite and between zero and one")
    in (fpProbabilityValid(q), "q must be finite and between zero and one")
{
    alias Status = GeometricInverseStatus;
    const one = Fp!bits(1);
    if (!fpProbabilityLess(q,one))
        return GeometricInverseResult(Status.infinite);
    if (!fpProbabilityLess(p,one))
        return GeometricInverseResult(Status.found,0);
    if (!q.coefficient)
        return GeometricInverseResult(Status.belowSupport);
    if (!p.coefficient)
        return GeometricInverseResult(Status.infinite);
    if (!fpProbabilityLess(p,q))
        return GeometricInverseResult(Status.found,0);
    if (!maxCount)
        return GeometricInverseResult(Status.countLimit);
    size_t lower;
    size_t upper = 1;
    for (;;) {
        const comparison = compare(upper,p,q);
        if (comparison == Comparison.unresolved)
            return GeometricInverseResult(Status.precisionLimit);
        if (comparison == Comparison.reached)
            break;
        if (upper == maxCount)
            return GeometricInverseResult(Status.countLimit);
        lower = upper;
        upper = upper > maxCount / 2 ? maxCount : upper * 2;
    }
    while (upper-lower > 1) {
        const middle = lower + (upper-lower)/2;
        const comparison = compare(middle,p,q);
        if (comparison == Comparison.unresolved)
            return GeometricInverseResult(Status.precisionLimit);
        if (comparison == Comparison.reached)
            upper = middle;
        else
            lower = middle;
    }
    return GeometricInverseResult(Status.found,upper);
}

// Caller examples exercise the candidate interface within its package scope.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!128;
    alias Status = GeometricInverseStatus;
    size_t destination = 99;
    GeometricInverseResult uninitialized;
    assert(!uninitialized.tryGetCount(destination) && destination == 99);

    // A count limit can be retried with a larger inclusive limit.
    auto result = geometricInvCDFChecked(T(0.875),T(0.5),1);
    assert(result.status == Status.countLimit);
    assert(!result.tryGetCount(destination) && destination == 99);
    result = geometricInvCDFChecked(T(0.875),T(0.5),2);
    assert(result.tryGetCount(destination) && destination == 2);

    auto tiny = T(1); tiny.exponent -= 1100;
    auto query = tiny; query.exponent += 1;
    // The mathematical count is 2, but a larger count budget cannot resolve
    // the comparison immediately below it at this working precision.
    const unresolved = geometricInvCDFChecked(query,tiny,4);
    assert(unresolved.status == Status.precisionLimit);
    assert(geometricInvCDFChecked(query,tiny).status == Status.precisionLimit);

    // Handle every status before interpreting the integer payload.
    foreach (r; [result, geometricInvCDFChecked(T(0),T(0.5)),
        geometricInvCDFChecked(T(1),T(1)),
        geometricInvCDFChecked(T(0.875),T(0.5),1), unresolved]) {
        size_t count = 99;
        final switch (r.status) {
        case Status.found:
            assert(r.tryGetCount(count) && count == 2);
            break;
        case Status.belowSupport: // Map to -1 only in a signed/extended result.
        case Status.infinite: // Preserve the endpoint convention separately.
        case Status.countLimit: // Increase the limit or propagate the limit.
        case Status.precisionLimit: // Propagate; no approximate count is given.
            assert(!r.tryGetCount(count) && count == 99);
            break;
        }
    }

    // A comparison can also exhaust precision after a bracket is established.
    auto p = Fp!64(1); p.exponent -= 300;
    auto q = p; q.exponent += 2;
    assert(compare(2,p,q) == Comparison.below);
    assert(compare(4,p,q) == Comparison.reached);
    assert(compare(3,p,q) == Comparison.unresolved);
    assert(geometricInvCDFChecked(q,p,4).status == Status.precisionLimit);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias I = ProbabilityInterval!64;
    foreach (a; [0u,1u,127u,128u,255u,256u])
    foreach (b; [0u,1u,127u,128u,255u,256u]) {
        const x = I(I.Integer(a),I.Integer(a));
        const y = I(I.Integer(b),I.Integer(b));
        const product = x.multiply(y);
        assert(product.lower == 0);
        assert(product.upper == I.Integer(a && b ? 1 : 0));
    }
    const tiny = Fp!64(false,-66,UInt!64(3));
    assert(I.from(tiny).lower == 0 && I.from(tiny).upper == 1);
    const exact = Fp!64(false,-65,UInt!64(4));
    assert(I.from(exact).lower == 2 && I.from(exact).upper == 2);
    assert(I.from(Fp!64(1)).lower == I.one);
    const unit = I(I.one,I.one);
    const exactProduct = unit.multiply(I(I.Integer(17),I.Integer(17)));
    assert(exactProduct.lower == 17 && exactProduct.upper == 17);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias Status = GeometricInverseStatus;
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        assert(geometricInvCDFChecked(T(1),T(1)).status == Status.infinite);
        assert(geometricInvCDFChecked(T(0),T(1)).status == Status.found);
        assert(geometricInvCDFChecked(T(0),T(0)).status == Status.belowSupport);
        assert(geometricInvCDFChecked(T(0.5),T(0)).status == Status.infinite);
        assert(geometricInvCDFChecked(T(0.25),T(0.25),0).count == 0);
        assert(geometricInvCDFChecked(T(0.5),T(0.25),0).status == Status.countLimit);
        foreach (k,q; [T(0.5),T(0.75),T(0.875)]) {
            auto below = q; below.coefficient -= 1;
            auto above = q; above.coefficient += 1;
            const a = geometricInvCDFChecked(below,T(0.5));
            const b = geometricInvCDFChecked(q,T(0.5));
            const c = geometricInvCDFChecked(above,T(0.5));
            assert(a.status == Status.found && a.count == k);
            assert(b.status == Status.found && b.count == k);
            assert(c.status == Status.found && c.count == k+1);
        }
        const limited = geometricInvCDFChecked(T(0.875),T(0.5),1);
        assert(limited.status == Status.countLimit);
        const exactLimit = geometricInvCDFChecked(T(0.875),T(0.5),2);
        assert(exactLimit.status == Status.found && exactLimit.count == 2);
        auto tiny = T(1); tiny.exponent -= 1100;
        auto query = tiny; query.exponent += 1;
        assert(geometricInvCDFChecked(tiny,tiny).status == Status.found);
        assert(geometricInvCDFChecked(query,tiny).status == Status.precisionLimit);
        const floor = T(false,long.min,UInt!bits(1));
        assert(geometricInvCDFChecked(T(0.5),floor,100).status == Status.countLimit);
        assert(geometricInvCDFChecked(floor,floor).status == Status.found);
        auto p = T(1); p.exponent -= 65;
        assert(geometricInvCDFChecked(T(0.5),p).status == Status.countLimit);
    }}
    // The first grid cannot resolve this tiny CDF jump; the second can.
    auto p = Fp!64(1); p.exponent -= 200;
    auto q = p; q.exponent += 1;
    assert(compareAt!192(1,p,q) == Comparison.unresolved);
    const result = geometricInvCDFChecked(q,p,4);
    assert(result.status == Status.found && result.count == 2);
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

// BEGIN inverse references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias Status = GeometricInverseStatus;
    struct Reference { uint p, q; size_t count; }
    enum references = [
        Reference(1,1,0),
        Reference(1,2,0),
        Reference(1,3,1),
        Reference(1,4,2),
        Reference(1,5,2),
        Reference(1,6,3),
        Reference(1,7,4),
        Reference(1,8,5),
        Reference(1,9,6),
        Reference(1,10,7),
        Reference(1,11,8),
        Reference(1,12,10),
        Reference(1,13,12),
        Reference(1,14,15),
        Reference(1,15,20),
        Reference(2,1,0),
        Reference(2,2,0),
        Reference(2,3,0),
        Reference(2,4,0),
        Reference(2,5,1),
        Reference(2,6,1),
        Reference(2,7,1),
        Reference(2,8,2),
        Reference(2,9,2),
        Reference(2,10,3),
        Reference(2,11,4),
        Reference(2,12,4),
        Reference(2,13,5),
        Reference(2,14,7),
        Reference(2,15,9),
        Reference(3,1,0),
        Reference(3,2,0),
        Reference(3,3,0),
        Reference(3,4,0),
        Reference(3,5,0),
        Reference(3,6,0),
        Reference(3,7,1),
        Reference(3,8,1),
        Reference(3,9,1),
        Reference(3,10,2),
        Reference(3,11,2),
        Reference(3,12,2),
        Reference(3,13,3),
        Reference(3,14,4),
        Reference(3,15,5),
        Reference(4,1,0),
        Reference(4,2,0),
        Reference(4,3,0),
        Reference(4,4,0),
        Reference(4,5,0),
        Reference(4,6,0),
        Reference(4,7,0),
        Reference(4,8,0),
        Reference(4,9,1),
        Reference(4,10,1),
        Reference(4,11,1),
        Reference(4,12,1),
        Reference(4,13,2),
        Reference(4,14,2),
        Reference(4,15,3),
        Reference(5,1,0),
        Reference(5,2,0),
        Reference(5,3,0),
        Reference(5,4,0),
        Reference(5,5,0),
        Reference(5,6,0),
        Reference(5,7,0),
        Reference(5,8,0),
        Reference(5,9,0),
        Reference(5,10,0),
        Reference(5,11,1),
        Reference(5,12,1),
        Reference(5,13,1),
        Reference(5,14,2),
        Reference(5,15,2),
        Reference(6,1,0),
        Reference(6,2,0),
        Reference(6,3,0),
        Reference(6,4,0),
        Reference(6,5,0),
        Reference(6,6,0),
        Reference(6,7,0),
        Reference(6,8,0),
        Reference(6,9,0),
        Reference(6,10,0),
        Reference(6,11,0),
        Reference(6,12,0),
        Reference(6,13,1),
        Reference(6,14,1),
        Reference(6,15,1),
        Reference(7,1,0),
        Reference(7,2,0),
        Reference(7,3,0),
        Reference(7,4,0),
        Reference(7,5,0),
        Reference(7,6,0),
        Reference(7,7,0),
        Reference(7,8,0),
        Reference(7,9,0),
        Reference(7,10,0),
        Reference(7,11,0),
        Reference(7,12,0),
        Reference(7,13,0),
        Reference(7,14,0),
        Reference(7,15,1),
    ];
    static foreach (bits; [64u,128u,256u]) {{
        alias T = Fp!bits;
        foreach (c; references) {
            auto p = T(c.p); p.exponent -= 3;
            auto q = T(c.q); q.exponent -= 4;
            const result = geometricInvCDFChecked(q,p);
            assert(result.status == Status.found && result.count == c.count);
            if (c.count) assert(geometricInvCDFChecked(q,p,c.count-1).status == Status.countLimit);
        }
    }}
    {
        alias T = Fp!64;
        enum q = T(false, -64, UInt!64.fromHexString("fffd39024c9b3da2"));
        const result = geometricInvCDFChecked(q,T(0.25));
        assert(result.status == Status.found && result.count == 35);
    }
    {
        alias T = Fp!128;
        enum q = T(false, -128, UInt!128.fromHexString("ffffffedb75e45ef78ded750bcbe8844"));
        const result = geometricInvCDFChecked(q,T(0.25));
        assert(result.status == Status.found && result.count == 67);
    }
    {
        alias T = Fp!256;
        enum q = T(false, -256, UInt!256.fromHexString("fffffffffffffce7993ebae3e1e77032cbaaa37745e7bc1199e41f59598f4488"));
        const result = geometricInvCDFChecked(q,T(0.25));
        assert(result.status == Status.found && result.count == 131);
    }
    static if (size_t.sizeof == 8) {
        static foreach (bits; [64u,128u,256u]) {{
            alias T = Fp!bits;
            auto p = T(1); p.exponent -= 64;
            const result = geometricInvCDFChecked(T(0.5),p);
            assert(result.status == Status.found && result.count == 12786308645202655659UL);
        }}
    }
}
// END inverse references
