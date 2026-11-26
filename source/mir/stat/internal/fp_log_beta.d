/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_log_beta;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_log_gamma: PositiveLogGamma;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: exponentBias;

@safe pure nothrow @nogc:

// Natural logarithm of B(a,b) for positive finite represented shapes.
// Return the logarithm at target precision after combining the guarded terms.
// Cancellation near a zero can reduce accuracy. Correct rounding is unproven.
package(mir.stat) Fp!bits logBeta(uint bits)(Fp!bits a, Fp!bits b)
    in (!a.sign && !!a.coefficient && a.exponent != long.max)
    in (!b.sign && !!b.coefficient && b.exponent != long.max)
{
    alias G = PositiveLogGamma!bits;
    alias S = G.S;
    alias W = G.W;
    alias T = G.T;
    alias L = ProbabilityLog!(G.work);
    L.Context ctx;

    if (position(a, b) > 0)
    {
        const swap = a;
        a = b;
        b = swap;
    }
    const logA = cast(W)L.logDyadic(a.coefficient, a.exponent, ctx);
    const logB = cast(W)L.logDyadic(b.coefficient, b.exponent, ctx);
    if (position(a, T(1)) == 0 || position(b, T(1)) == 0)
    {
        auto result = cast(T)(position(a, T(1)) == 0 ? logB : logA);
        if (result.coefficient)
            result.sign = !result.sign;
        return result;
    }

    const x = S.from(a);
    const y = S.from(b);
    const sum = x.add(y);
    if (y.exponent <= exponentBias + G.shift)
    {
        // Gamma(1-d)*Gamma(1+d) = u/sin(u), u = pi*d.
        // log B = u^2/6 + u^4/180 + O(u^6).
        if (position(a, T(1)) < 0 && position(b, T(1)) > 0)
        {
            const d = distance!(G.work)(b, T(1));
            if (d.exponent + bits / 2 + 8 <= exponentBias &&
                distance!(G.work)(a, T(1)) == d)
            {
                import mir.stat.internal.fp_pi: guardedPi;
                const angle = S.from(guardedPi!(G.work)()).multiply(d);
                const square = angle.multiply(angle);
                const quadratic = square.divide(S.from(W(6)));
                const quartic = square.multiply(square).divide(S.from(W(180)));
                return quadratic.add(quartic).finish!bits;
            }
        }
        // When both shapes are tiny, B(a,b) = (a+b)/(a*b) * (1+O(a*b)).
        // Form the ratio through logs to accommodate the exponent floor.
        if (sum.exponent + G.work < exponentBias)
        {
            const z = addOffset!(G.work)(W(0), sum, false);
            const logSum = cast(W)L.logDyadic(z.coefficient, z.exponent, ctx);
            return cast(T)addOffset!(G.work)(addOffset!(G.work)(logSum, S.from(logA), !logA.sign),
                S.from(logB), !logB.sign);
        }
        const z = addOffset!(G.work)(W(0), sum, false);
        const gSum = G.guardedRecurrence(z);
        const gA = G.guardedLogarithm(a);
        const gB = G.guardedLogarithm(b);
        W result = gA;
        if (gB.coefficient)
            result = addOffset!(G.work)(result, S.from(gB), gB.sign);
        return gSum.coefficient ? addOffset!bits(result, S.from(gSum), !gSum.sign) : cast(T)result;
    }

    const ratio = x.divide(y);
    // log(1+r) = -log(1-p), p = r/(1+r). Preserve the correction through
    // working precision because the result can nearly cancel near B(a,b)=1.
    // Below the cutoff the relative correction is < r/2.
    S t = ratio;
    if (ratio.exponent + G.work + 16 >= exponentBias)
    {
        const p = ratio.divide(S.from(W(1)).add(ratio));
        t = S.from(L.complement(addOffset!(G.work)(W(0), p, false)).finish!(G.work));
    }

    const corrB = G.correction(y);
    const corrSum = G.correction(sum);
    W result;
    if (x.exponent <= exponentBias + G.shift)
    {
        // log Gamma(a) - a*log(b) + a - (a+b-1/2)*log1p(a/b).
        result = addOffset!(G.work)(G.guardedLogarithm(a), x.multiply(S.from(logB)), true);
        result = addOffset!(G.work)(result, x, false);
    }
    else
    {
        import mir.stat.internal.fp_pi: guardedPi;
        // a*log(a/b) - (a+b)*log1p(a/b) + log(2*pi/a)/2.
        // log(a)-log(b) loses at most the exponent's width in absolute
        // precision here; the log1p term dominates for comparable shapes.
        const logRatio = addOffset!(G.work)(logA, S.from(logB), true);
        result = logRatio.coefficient
            ? addOffset!(G.work)(W(0), x.multiply(S.from(logRatio)), logRatio.sign) : W(0);
        auto pi = guardedPi!(G.work)();
        ++pi.exponent;
        auto constant = cast(W)L.logDyadic(pi.coefficient, pi.exponent, ctx);
        constant = addOffset!(G.work)(constant, S.from(logA), true);
        auto halfConstant = S.from(constant);
        --halfConstant.exponent;
        result = addOffset!(G.work)(result, halfConstant, constant.sign);
        result = addOffset!(G.work)(result, G.correction(x), false);
    }
    result = addOffset!(G.work)(result, sum.multiply(t), true);
    --t.exponent;
    result = addOffset!(G.work)(result, t, false);
    result = addOffset!(G.work)(result, corrB, false);
    return addOffset!bits(result, corrSum, true);
}

// MPFR references at 6144 bits for exact dyadics. At the exponent limits
// the finite references round the analytic leading terms in scaled units.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    alias U = UInt!64;
    const one = T(1);
    assert(logBeta(one, one) == T(0));
    T huge = one;
    huge.exponent += 1800;
    assert(logBeta(one, huge) == T(true, -53, U(0x9bf54710cf6759f0UL)));
    assert(logBeta(huge, T(2)) == T(true, -52, U(0x9bf54710cf6759f0UL)));
    assert(logBeta(T(32), T(32)) == T(true, -58, U(0xb34caa892da42390UL)));
    const half = T(0.5);
    const logPi = T(false, -63, U(0x928682473d0de85fUL));
    assert(logBeta(half, half) == logPi);
    import mir.math.internal.log_beta: builtinLogBeta = logBeta;
    import mir.math.common: approxEqual;
    assert(approxEqual(builtinLogBeta(0.5, 0.5), cast(double)logPi, 32 * double.epsilon, 0.0));
    const above = T(false, -63, U(0x8000000000000001UL));
    const below = T(false, -64, U.max - 1);
    assert(logBeta(below, above) == T(false, -189, U(0xd28d3312983e9918UL)));
    assert(logBeta(above, below) == logBeta(below, above));
    // For d = 6085447 * 2^-63 the quartic term crosses a rounding midpoint.
    enum ulong n = 6085447;
    const left = T(false, -64, U.max - (2*n - 1));
    const right = T(false, -63, U((1UL << 63) + n));
    assert(logBeta(left, right) == T(false, -144, U(0xdd9cb7ea61951024UL)));
    // Their sum is 2 + 2^-64, which rounds to 2 at target precision.
    assert(logBeta(above, T(false, -64, U.max)) == T(true, -128, U(0xfffffffffffffffcUL)));
    const floor = T(false, long.min, U(1));
    const peak = T(false, long.max - 1, U(1) << 63);
    assert(logBeta(floor, floor) == T(false, -1, U(0xb17217f7d1cf79adUL)));
    assert(logBeta(floor, peak) == T(false, -1, U(0xb17217f7d1cf79acUL)));
    assert(logBeta(peak, peak) == T(true, long.max - 1, U(0xb17217f7d1cf79acUL)));
    const largest = T(false, long.max - 1, U.max);
    assert(logBeta(largest, largest) == T(true, long.max, U(0)));
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias U = UInt!128;
    // Near a zero at b=2^208, using a/b for log1p(a/b) loses low bits.
    enum hex(string s) = U.fromHexString(s);
    const a = T(false, -133, hex!"cfef97a87ef1b5398fc697d90f88dd02");
    const b = T(false, 81, U(1) << 127);
    const expected = T(false, -254, hex!"899f6fee74a9c9449a00bbfd442af665");
    assert(logBeta(a, b) == expected);
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!32;
    alias U = UInt!32;
    const half = T(false, -1, U(1));
    assert(logBeta(half, half) == T(false, -31, U(0x92868247)));
    assert(logBeta(T(16), T(16)) == T(true, -27, U(0xb2597550)));
}
