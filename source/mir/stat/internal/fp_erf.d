/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_erf;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset;
import mir.stat.internal.fp_pi: guardedPi;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay, scaledSqrt;

@safe pure nothrow @nogc:

// Nonnegative magnitudes, including positive infinity and either zero.
// NaN propagates. Both functions use guarded evaluation at the input width.
package(mir.stat) Fp!bits erfPositive(uint bits)(Fp!bits x)
    in (x.isNaN || !x.sign || (!x.isSpecial && !x.coefficient))
{
    if (x.isNaN || (!x.isSpecial && !x.coefficient)) return x;
    if (x.isInfinity) return Fp!bits(1);
    return evaluate!false(x);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    const negativeZero = T(true, 0, UInt!64(0));
    assert(erfPositive(negativeZero).sign);
    assert(erfPositive(T(double.nan)).isNaN);
    assert(erfPositive(T(double.infinity)) == T(1));
    const least = T(false, long.min, UInt!64(1));
    assert(erfPositive(least) == least); // 2/sqrt(pi) quanta
    // MPFR at 1, rounded directly to the target width.
    assert(erfPositive(T(1)) == T(false, -64, UInt!64(0xd7bb3d3a08445637UL)));
    // Adjacent inputs straddle the MPFR rounding boundary below one.
    const below = T(false, -61, UInt!64(0xd0d8e9797c8fb412UL));
    const above = T(false, -61, UInt!64(0xd0d8e9797c8fb413UL));
    assert(erfPositive(below) == T(false, -64, UInt!64.max));
    assert(erfPositive(above) == T(1));
    assert(erfPositive(T(256)) == T(1));
}

// exp(x*x)*erfc(x), retaining the reciprocal-scale tail at large x.
package(mir.stat) Fp!bits erfcxPositive(uint bits)(Fp!bits x)
    in (x.isNaN || !x.sign || (!x.isSpecial && !x.coefficient))
{
    if (x.isNaN) return x;
    if (x.isInfinity) return Fp!bits(0);
    if (!x.coefficient) return Fp!bits(1);
    return evaluate!true(x);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!64;
    assert(erfcxPositive(T(0)) == T(1));
    assert(erfcxPositive(T(double.nan)).isNaN);
    assert(erfcxPositive(T(double.infinity)) == T(0));
    assert(erfcxPositive(T(1)) == T(false, -65, UInt!64(0xdaec3c07cab590c1UL)));
    const large = T(false, long.max - 1, UInt!64(1));
    // Leading term 1/(sqrt(pi)*x), in quanta at the exponent floor.
    assert(erfcxPositive(large) == T(false, long.min, UInt!64(2)));
    assert(erfcxPositive(T(false, long.max - 1, UInt!64.max)) == T(0));
}

// MPFR references at the two neighbors of the 128-bit transition x = 4.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!128;
    alias I = UInt!128;
    const below = T(false, -126, I.max);
    const above = T(false, -125, (I(1) << 127) + 1);
    enum erf = T(false, -128, I.fromHexString("ffffffbdc88bb10b2865615db4031938"));
    assert(erfPositive(below) == erf);
    assert(erfPositive(above) == erf);
    enum belowErfcx = T(false, -130, I.fromHexString("8c4995f8470aa213035424e8f9cf13cc"));
    assert(erfcxPositive(below) == belowErfcx);
    enum aboveErfcx = T(false, -130, I.fromHexString("8c4995f8470aa213035424e8f9cf13cb"));
    assert(erfcxPositive(above) == aboveErfcx);
}

version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    alias T = Fp!32;
    assert(erfPositive(T(2)) == T(false, -32, UInt!32(0xfecd70a1)));
    assert(erfcxPositive(T(2)) == T(false, -33, UInt!32(0x82c338db)));
}

private Fp!bits evaluate(bool scaledComplement, uint bits)(Fp!bits input)
{
    alias T = Fp!bits;
    // The surplus covers the central subtraction, decay evaluation, and
    // accumulated arithmetic error. Universal correct rounding is unproven.
    enum work = 2 * bits + 192;
    alias W = Fp!work;
    alias S = ProbabilityScaled!work;
    const x = S.from(input);
    const square = x.multiply(x);

    // For x >= 1, erfc(x) < exp(-x*x). Above this cutoff erf rounds to one.
    static if (!scaledComplement)
        if (square.exponent > S.from(T(bits + 2)).exponent)
            return T(1);

    const sqrtPi = scaledSqrt(S.from(guardedPi!work()), false);
    const limit = S.from(W(bits / 8));
    if (square.exponent < limit.exponent || (square.exponent == limit.exponent &&
        square.fraction.coefficient <= limit.fraction.coefficient))
    {
        const decay = scaledDecay(square);
        const erf = centralErf(x, square, decay, sqrtPi);
        static if (scaledComplement)
        {
            // Here x*x <= bits/8; cancellation in 1-erf costs roughly
            // x*x/log(2) bits plus log2(x). Keep the surplus until division.
            const complement = addOffset!work(W(1), erf, true);
            return S.from(complement).divide(decay).finish!bits;
        }
        else
            return erf.finish!bits;
    }

    const scaled = tailErfcx!bits(x, square, sqrtPi);
    static if (scaledComplement)
        return scaled.finish!bits;
    else
        return addOffset!bits(T(1), scaled.multiply(scaledDecay(square)), true);
}

private auto centralErf(uint work)(ProbabilityScaled!work x, ProbabilityScaled!work square,
    ProbabilityScaled!work decay, ProbabilityScaled!work sqrtPi)
{
    alias W = Fp!work;
    alias S = ProbabilityScaled!work;
    // DLMF 7.6.2. All series terms are positive. Once past the
    // maximum, the ratios keep decreasing. Below 1/2 the remaining
    // tail is bounded by twice the first omitted term.
    S twiceSquare = square;
    twiceSquare.exponent += 1;
    S term = S.from(W(1));
    S sum = term;
    for (uint n = 1; ; ++n)
    {
        const ratio = twiceSquare.divide(S.from(W(2 * n + 1)));
        term = term.multiply(ratio);
        if (ratio.exponent < exponentBias && term.exponent + work < sum.exponent)
            break;
        sum = sum.add(term);
    }
    auto erf = x.multiply(sum).multiply(decay).divide(sqrtPi);
    erf.exponent += 1;
    return erf;
}

private auto tailErfcx(uint bits, uint work)(ProbabilityScaled!work x, ProbabilityScaled!work square,
    ProbabilityScaled!work sqrtPi)
{
    alias W = Fp!work;
    alias S = ProbabilityScaled!work;
    const one = S.from(W(1));
    // Laplace's continued fraction (DLMF 7.9.1), normalized to
    // 1/(1 + a1/(1 + a2/(1 + ...))), with an = n/(2*x*x).
    // In exact arithmetic adjacent convergents enclose the limit. Their
    // separation is product(a1..an)/(Qn*Qprev). Test its size relative
    // to the previous convergent before rounding the final quotient.
    auto step = one.divide(square);
    step.exponent -= 1;
    S p = one, q = one, oldQ = one, determinant = one;
    S oldP; // P[-1] is zero; omit its contribution at n=1.
    for (uint n = 1; ; ++n)
    {
        const a = step.multiply(S.from(W(n)));
        const nextP = n == 1 ? p : p.add(a.multiply(oldP));
        const nextQ = q.add(a.multiply(oldQ));
        determinant = determinant.multiply(a);
        if (determinant.exponent + bits + 64 < nextQ.multiply(p).exponent)
            return nextP.divide(nextQ).divide(sqrtPi).divide(x);
        oldP = p;
        oldQ = q;
        p = nextP;
        q = nextQ;
    }
}
