/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_beta_proportion;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_beta: scaledBetaDensity, scaledBetaQuantile, scaledBetaTail;
import mir.stat.internal.fp_location_scale: position;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

@safe pure nothrow @nogc:

private struct ProportionShapes(uint bits)
{
    alias T = Fp!bits;
    alias S = ShapeTransform!bits.S;
    // Leave room in the beta kernel for an exact product with a rounding
    // midpoint (bits+1 significant bits) in the centered residual.
    enum width = 2 * bits + 64;
    alias Narrow = ProbabilityScaled!width;
    S alpha, lower, upper, displacement;
    bool exact;

    this(T mu, T kappa)
    {
        const k = Narrow.from(kappa);
        const a = k.multiply(Narrow.from(mu)); // exact: at most 2*bits
        alpha = S.from(a.fraction);
        alpha.exponent += a.exponent - exponentBias;
        auto gap = k.exponent - a.exponent;
        UInt!128 exponent = k.exponent;
        const coefficient = a.fraction.coefficient;
        const trailing = UInt!128(coefficient.cttz);
        auto minuend = k.fraction.coefficient.toSize!(width + 64);
        // If alpha exceeds the distance from kappa to its binade boundary,
        // beta lies on a finer grid. For inexact alignment alpha is tiny
        // relative to kappa, so only one additional bit can be needed.
        const excess = k.fraction.coefficient ^ (UInt!width(1) << (width - 1));
        if (gap > trailing && (!excess || compare(alpha, onGrid(excess, exponent)) > 0))
        {
            --gap;
            --exponent;
            minuend <<= 1;
        }
        const aligned = gap < width ? coefficient >> cast(uint)gap : UInt!width(0);
        const ceiling = minuend - aligned.toSize!(width + 64);
        exact = gap <= trailing;
        upper = onGrid(ceiling, exponent);
        lower = exact ? upper : onGrid(ceiling - 1, exponent);
        if (!exact)
        {
            const discarded = gap >= width ? coefficient :
                coefficient - (aligned << cast(uint)gap);
            displacement = onGrid(discarded, a.exponent);
        }
    }

    private static S onGrid(uint n)(UInt!n coefficient, UInt!128 exponent)
    {
        auto result = S.from(Fp!n(coefficient));
        result.exponent += exponent - exponentBias - width;
        return result;
    }

    static int compare(S a, S b)
    {
        if (a.exponent != b.exponent) return a.exponent < b.exponent ? -1 : 1;
        return a.fraction.coefficient.opCmp(b.fraction.coefficient);
    }

    // Bound |d(log f)/db| by -log(1-x) + (a/b)*(1+1/b), using
    // psi'(t) <= 1/t + 1/t^2 throughout [b,a+b]. Below 1/2,
    // -log(1-x) <= 2*x; above it, 1-x >= 2^-bits. Double the
    // bound to absorb rounding of the positive working operations.
    S logDensityRadius(T x) const
    {
        const one = S.from(T(1));
        auto logBound = S.from(x);
        ++logBound.exponent;
        if (compare(logBound, one) > 0) logBound = S.from(T(bits));
        const slope = alpha.divide(lower).multiply(one.add(one.divide(lower)));
        auto radius = displacement.multiply(logBound.add(slope));
        ++radius.exponent;
        return radius;
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias P = ProportionShapes!64;
    alias T = Fp!64;
    alias U = UInt!64;
    alias W = UInt!192;
    const last = P.S.from(Fp!192(false, -192, W.max));
    const previous = P.S.from(Fp!192(false, -192, W.max - 1));
    // Adjacent means on either side of 2^-192 give genuinely inexact
    // complements. Check both neighboring bounds and the discarded part.
    const below = T(false, -256, U.max);
    const left = P(below, T(1));
    assert(!left.exact && P.compare(left.lower, last) == 0 && P.compare(left.upper, P.S.from(T(1))) == 0);
    assert(P.compare(left.displacement, P.S.from(below)) == 0);
    const above = T(false, -255, U(0x8000000000000001UL));
    const right = P(above, T(1));
    assert(!right.exact && P.compare(right.lower, previous) == 0 && P.compare(right.upper, last) == 0);
    assert(P.compare(right.displacement, P.S.from(T(false, -255, U(1)))) == 0);
}

package(mir.stat) enum BetaOperation { pdf, lpdf, cdf, ccdf, quantile }

package(mir.stat) Fp!bits betaProportion(BetaOperation op, uint bits)(Fp!bits x, Fp!bits mu, Fp!bits kappa)
{
    alias T = Fp!bits;
    alias P = ProportionShapes!bits;
    const one = T(1);
    const atOne = position(x, one) == 0;
    static if (op == BetaOperation.cdf || op == BetaOperation.ccdf || op == BetaOperation.quantile)
    {
        if (!x.coefficient || atOne)
            return T((op == BetaOperation.ccdf) != atOne ? 1 : 0);
    }
    const shapes = P(mu, kappa);
    T evaluate(P.S b)
    {
        static if (op == BetaOperation.quantile)
            return scaledBetaQuantile!bits(x, shapes.alpha, b);
        else static if (op == BetaOperation.cdf || op == BetaOperation.ccdf)
            return scaledBetaTail!(op == BetaOperation.ccdf, bits)(x, shapes.alpha, b);
        else
            return scaledBetaDensity!(op == BetaOperation.lpdf, bits)(x, shapes.alpha, b);
    }
    if (shapes.exact) return evaluate(shapes.upper);
    const nan = T(false, long.max, UInt!bits(1));
    static if (op == BetaOperation.pdf || op == BetaOperation.lpdf)
    {
        // The exact complementary shape is strictly inside its bounds.
        // At x=1 a bound equal to one must not select a finite endpoint.
        if (atOne)
        {
            const below = P.compare(shapes.upper, P.S.from(one)) <= 0;
            return below || op == BetaOperation.lpdf ? T(!below, long.max, UInt!bits(0)) : T(0);
        }
        if (x.coefficient)
        {
            import mir.stat.internal.fp_beta: betaMeanDensity;
            import mir.stat.internal.fp_log_gamma: PositiveLogGamma;
            enum shift = PositiveLogGamma!bits.shift;
            if (position(x, mu) == 0 && shapes.alpha.exponent > exponentBias + shift &&
                shapes.lower.exponent > exponentBias + shift)
                return betaMeanDensity!(op == BetaOperation.lpdf, bits)(kappa, shapes.alpha, shapes.lower, shapes.upper);
            alias B = ShapeTransform!bits;
            const error = B.Term(shapes.logDensityRadius(x), false, true);
            return scaledBetaDensity!(op == BetaOperation.lpdf, bits)(x, shapes.alpha, shapes.upper, error);
        }
    }
    // Beta is stochastically decreasing in b. Both tail orientations and
    // the inverse are monotone. At x=0 a finite density equals b.
    const lo = evaluate(shapes.lower);
    const hi = evaluate(shapes.upper);
    return lo == hi ? lo : nan;
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias T = Fp!32;
    alias U = UInt!32;
    const mu = T(false, -33, U(0xbfffffff));
    const k = T(false, -29, U(0xe0000001));
    assert(betaProportion!(BetaOperation.cdf)(T(false, -33, U(0xc0000001)), mu, k) ==
        T(false, -32, U(0x86c1ef6d)));
}
