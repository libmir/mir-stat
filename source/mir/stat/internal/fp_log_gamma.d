/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_log_gamma;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;
import mir.stat.internal.fp_location_scale: addOffset, position;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias;

// Guarded log Gamma on positive finite dyadics. The extra significand
// absorbs recurrence cancellation at the represented neighbors of 1 and 2.
// Rounding of the complete calculation is not certified.
package(mir.stat) struct PositiveLogGamma(uint bits)
{
    static assert(bits <= 512, "recurrence budget supports up to 512 bits");
    alias T = Fp!bits;
    enum work = 2 * bits + 128;
    alias W = Fp!work;
    alias S = ProbabilityScaled!work;
    private alias L = ProbabilityLog!work;

    // |B130/(130*129)| < 2^373. At z >= 2^shift the first
    // omitted term, in units of 1/z, is below 2^(-2*bits-64).
    enum shift = (373 + 2 * bits + 64 + 127) / 128;
    enum crossover = 1u << shift;

    @safe pure nothrow @nogc:

    // R(z) in log Gamma(z) = (z-1/2)log(z)-z+log(2*pi)/2+R(z).
    // Return the correction on a separate exponent even when 1/z is tiny.
    // Reserve three binary exponents below 1/2^^e for the factor 1/(12*f),
    // where z = f*2^^e and 1/2 <= f < 1.
    static S correction(S z)
        in (z.exponent > exponentBias + shift && z.exponent <= UInt!128.max - 2)
    {
        import mir.stat.internal.fp_log_factorial: coefficients;
        const inverse = S.from(W(1)).divide(z);
        // Relative to 1/(12*z), the omitted terms are bounded by
        // 1/(30*z^^2). Test z before squaring its reciprocal.
        if (z.exponent > exponentBias + work / 2)
            return inverse.divide(S.from(W(12)));
        const square = inverse.multiply(inverse);
        auto power = S.from(W(1));
        W sum = W(0);
        foreach (i, ref c; coefficients[0 .. $-1])
        {
            const factor = S.from(W(c.numerator)).divide(S.from(W(c.denominator)));
            const term = power.multiply(factor);
            // For positive z the first omitted term bounds the remainder.
            if (term.exponent + work + 4 < exponentBias)
                break;
            sum = addOffset!work(sum, term, (i & 1) != 0);
            power = power.multiply(square);
        }
        return inverse.multiply(S.from(sum));
    }

    static T logarithm(T x)
        in (!x.sign && !!x.coefficient && x.exponent != long.max)
    {
        return evaluate!bits(x);
    }

    // Extra significand for distribution formulas that combine log Gamma with
    // other terms before final rounding. Keep the original target's crossover.
    static W guardedLogarithm(T x)
        in (!x.sign && !!x.coefficient && x.exponent != long.max)
    {
        return evaluate!work(x);
    }

    // log Gamma(1+a), with a supplied separately. Up to 512 target bits.
    // The linear approximation has absolute error <= zeta(2)*a*a/2 for
    // 0 <= a <= 1. At the exponent floor the returned W can be subnormal.
    static W logGamma1p(T a)
        in ((!a.sign || !a.coefficient) && a.exponent != long.max)
    {
        if (!a.coefficient || position(a, T(1)) == 0)
            return W(0);
        const input = S.from(a);
        if (input.exponent <= exponentBias - 1)
            return addOffset!work(W(0), smallIncrementMagnitude(a), true);
        // For a >= 2^^work, adding log(a) to log Gamma(a) changes the
        // result by at most O(2^^-work) relative.
        if (input.exponent > exponentBias + work)
            return guardedLogarithm(a);
        return recurrence!work(addOffset!work(W(1), input, false));
    }

    // Keep |log Gamma(1+a)| off the subnormal grid until the caller
    // has combined the complete expression, including at the input floor.
    static S smallIncrementMagnitude(T a)
        in (!a.sign && !!a.coefficient && S.from(a).exponent <= exponentBias - 1)
    {
        const input = S.from(a);
        if (input.exponent + bits + 64 <= exponentBias)
        {
            // round(EulerGamma * 2^^1152), generated with MPFR at 4096 bits.
            enum euler = UInt!1152.fromHexString(
                "93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbd" ~
                "8e4b59fa03a9f0eed0649ccb621057d11056ae9132135a08e43b4673d74bafea" ~
                "58deb878cc86d733dbe7bf38154b36cf8a96d1567899aaae0c09d4c8b6b7b86f" ~
                "d2a1ea1de62ff8643ec7c271827977225e6ac2f0bd61c746961542a3ce3bea5d" ~
                "b54fe70e63e6d09f8fc28658e80567a4");
            auto factor = S.from(Fp!1152(euler));
            factor.exponent -= 1152;
            return input.multiply(factor);
        }
        return S.from(recurrence!work(addOffset!work(W(1), input, false)));
    }

    private static Fp!outBits evaluate(uint outBits)(T x)
    {
        alias R = Fp!outBits;
        if (position(x, T(1)) == 0 || position(x, T(2)) == 0)
            return R(0);

        const input = S.from(x);
        L.Context context;
        if (input.exponent + work < exponentBias)
        {
            // -x <= log Gamma(1+x) <= 0 by convexity on [1,2].
            auto result = cast(R)L.logDyadic(x.coefficient, x.exponent, context);
            result.sign = false;
            return result;
        }
        const one = S.from(W(1));
        if (input.exponent > exponentBias + work)
        {
            // The remaining terms are O(log x), below the working
            // precision relative to x*(log x-1). Avoid intermediate overflow.
            const logX = cast(W)L.logDyadic(x.coefficient, x.exponent, context);
            const factor = addOffset!work(logX, one, true);
            return addOffset!outBits(R(0), input.multiply(S.from(factor)), false);
        }

        W z = input.fraction;
        z.exponent += cast(long)cast(ulong)(input.exponent - exponentBias);
        return recurrence!outBits(z);
    }

    // Keep a sum of positive shapes at working precision through the recurrence.
    // The caller handles tiny arguments and selects the Stirling form for large ones.
    static W guardedRecurrence(W z)
        in (!z.sign && !!z.coefficient && z.exponent != long.max)
        in (S.from(z).exponent + work >= exponentBias && position(z, W(2 * crossover)) <= 0)
    {
        if (position(z, W(1)) == 0 || position(z, W(2)) == 0)
            return W(0);
        return recurrence!work(z);
    }

    private static Fp!outBits recurrence(uint outBits)(W z)
    {
        import mir.stat.internal.fp_pi: guardedPi;
        L.Context context;
        const one = S.from(W(1));
        auto product = W(1);
        while (position(z, W(crossover)) < 0)
        {
            product *= z;
            z = addOffset!work(z, one, false);
        }
        const logZ = cast(W)L.logDyadic(z.coefficient, z.exponent, context);
        auto result = addOffset!work(z * logZ, S.from(z), true);
        auto halfLogZ = S.from(logZ);
        halfLogZ.exponent -= 1;
        result = addOffset!work(result, halfLogZ, true);
        auto twicePi = guardedPi!work();
        twicePi.exponent += 1;
        auto constant = S.from(L.logDyadic(twicePi.coefficient, twicePi.exponent, context));
        constant.exponent -= 1;
        result = addOffset!work(result, constant, false);
        result = addOffset!work(result, correction(S.from(z)), false);
        const logProduct = cast(W)L.logDyadic(product.coefficient, product.exponent, context);
        if (logProduct.coefficient)
            return addOffset!outBits(result, S.from(logProduct), !logProduct.sign);
        static if (outBits == work) return result;
        else return cast(Fp!outBits)result;
    }
}

// Rounded from 2048-bit MPFR references at the exact dyadics below. For
// the exponent limits, evaluate the leading logarithm in scaled units.
version(mir_stat_test)
static if (size_t.sizeof == 4)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!32;
    alias T = G.T;
    alias U = UInt!32;
    enum hex(string s) = U.fromHexString(s);
    assert(G.logarithm(T(false, -32, hex!"ffffffff")) == T(false, -64, hex!"93c467e4"));
    assert(G.logarithm(T(false, -31, hex!"80000001")) == T(true, -63, hex!"93c467e2"));
    assert(G.logarithm(T(false, -31, hex!"ffffffff")) == T(true, -64, hex!"d8773038"));
    assert(G.logarithm(T(false, -30, hex!"80000001")) == T(false, -63, hex!"d877303c"));
    assert(G.logarithm(T(false, -33, hex!"c0000000")) == T(false, -32, hex!"dcf26aa0"));
    assert(G.logarithm(T(false, -28, hex!"ffffffff")) == T(false, -27, hex!"df31b531"));
    assert(G.logarithm(T(false, -27, hex!"80000000")) == T(false, -27, hex!"df31b532"));
    assert(G.logarithm(T(false, -27, hex!"80000001")) == T(false, -27, hex!"df31b535"));
    assert(G.logarithm(T(false, 269, hex!"80000000")) == T(false, 276, hex!"cef1b416"));
    assert(G.correction(G.S.from(T(false, -27, hex!"80000000"))).finish!32 == T(false, -39, hex!"aaa4fbef"));
    assert(G.correction(G.S.from(T(false, 269, hex!"80000000"))).finish!32 == T(false, -335, hex!"aaaaaaab"));
    assert(G.logarithm(T(false, long.min, U(1))) == T(false, 31, hex!"b17217f8"));
    assert(G.logarithm(T(false, long.max - 64, U(1))) == T(false, long.max - 33, hex!"b17217f8"));
    assert(G.logarithm(T(false, long.max - 1, U(1) << 31)).isInfinity);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!64;
    alias T = G.T;
    alias U = UInt!64;
    enum hex(string s) = U.fromHexString(s);
    assert(G.logarithm(T(false, -64, hex!"ffffffffffffffff")) == T(false, -128, hex!"93c467e37db0c7a6"));
    assert(G.logarithm(T(false, -63, hex!"8000000000000001")) == T(true, -127, hex!"93c467e37db0c7a3"));
    assert(G.logarithm(T(false, -63, hex!"ffffffffffffffff")) == T(true, -128, hex!"d8773039049e70b5"));
    assert(G.logarithm(T(false, -62, hex!"8000000000000001")) == T(false, -127, hex!"d8773039049e70b9"));
    assert(G.logarithm(T(false, -65, hex!"c000000000000000")) == T(false, -64, hex!"dcf26a9fe1839fba"));
    assert(G.logarithm(T(false, -59, hex!"ffffffffffffffff")) == T(false, -57, hex!"9c2f37ecf2d20200"));
    assert(G.logarithm(T(false, -58, hex!"8000000000000000")) == T(false, -57, hex!"9c2f37ecf2d20201"));
    assert(G.logarithm(T(false, -58, hex!"8000000000000001")) == T(false, -57, hex!"9c2f37ecf2d20203"));
    assert(G.logarithm(T(false, 237, hex!"8000000000000000")) == T(false, 244, hex!"cef1b41669df2295"));
    assert(G.correction(G.S.from(T(false, -58, hex!"8000000000000000"))).finish!64 == T(false, -72, hex!"aaa93eade600183d"));
    assert(G.correction(G.S.from(T(false, 237, hex!"8000000000000000"))).finish!64 == T(false, -367, hex!"aaaaaaaaaaaaaaab"));
    assert(G.logarithm(T(false, long.min, U(1))) == T(false, -1, hex!"b17217f7d1cf79ac"));
    assert(G.logarithm(T(false, long.max - 64, U(1))) == T(false, long.max - 65, hex!"b17217f7d1cf7950"));
    assert(G.logarithm(T(false, long.max - 1, U(1) << 63)).isInfinity);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!128;
    alias T = G.T;
    alias U = UInt!128;
    enum hex(string s) = U.fromHexString(s);
    assert(G.logarithm(T(false, -128, hex!"ffffffffffffffffffffffffffffffff")) == T(false, -256, hex!"93c467e37db0c7a4d1be3f810152cb57"));
    assert(G.logarithm(T(false, -127, hex!"80000000000000000000000000000001")) == T(true, -255, hex!"93c467e37db0c7a4d1be3f810152cb55"));
    assert(G.logarithm(T(false, -127, hex!"ffffffffffffffffffffffffffffffff")) == T(true, -256, hex!"d8773039049e70b65c8380fdfd5a6951"));
    assert(G.logarithm(T(false, -126, hex!"80000000000000000000000000000001")) == T(false, -255, hex!"d8773039049e70b65c8380fdfd5a6955"));
    assert(G.logarithm(T(false, -129, hex!"c0000000000000000000000000000000")) == T(false, -128, hex!"dcf26a9fe1839fb9d75cdea21c8c8ea1"));
    assert(G.logarithm(T(false, -122, hex!"ffffffffffffffffffffffffffffffff")) == T(false, -120, hex!"c902628f3e3b0ae4f39a537926c3ba91"));
    assert(G.logarithm(T(false, -121, hex!"80000000000000000000000000000000")) == T(false, -120, hex!"c902628f3e3b0ae4f39a537926c3ba92"));
    assert(G.logarithm(T(false, -121, hex!"80000000000000000000000000000001")) == T(false, -120, hex!"c902628f3e3b0ae4f39a537926c3ba94"));
    assert(G.logarithm(T(false, 173, hex!"80000000000000000000000000000000")) == T(false, 180, hex!"cef1b41669df22955096d67624a0b915"));
    assert(G.correction(G.S.from(T(false, -121, hex!"80000000000000000000000000000000"))).finish!128 == T(false, -137, hex!"aaaa4fa69a56271b693fcf3e109dd163"));
    assert(G.correction(G.S.from(T(false, 173, hex!"80000000000000000000000000000000"))).finish!128 == T(false, -431, hex!"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"));
    assert(G.logarithm(T(false, long.min, U(1))) == T(false, -65, hex!"b17217f7d1cf79abc9e3b39803f2f6af"));
    assert(G.logarithm(T(false, long.max - 64, U(1))) == T(false, long.max - 129, hex!"b17217f7d1cf794fadf387bf78972d73"));
    assert(G.logarithm(T(false, long.max - 1, U(1) << 127)).isInfinity);
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!256;
    alias T = G.T;
    alias U = UInt!256;
    enum hex(string s) = U.fromHexString(s);
    assert(G.logarithm(T(false, -256, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(false, -512, hex!"93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbe"));
    assert(G.logarithm(T(false, -255, hex!"8000000000000000000000000000000000000000000000000000000000000001")) == T(true, -511, hex!"93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbc"));
    assert(G.logarithm(T(false, -255, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(true, -512, hex!"d8773039049e70b65c8380fdfd5a6952bc62678a13467fcde7f841971eca0084"));
    assert(G.logarithm(T(false, -254, hex!"8000000000000000000000000000000000000000000000000000000000000001")) == T(false, -511, hex!"d8773039049e70b65c8380fdfd5a6952bc62678a13467fcde7f841971eca0087"));
    assert(G.logarithm(T(false, -257, hex!"c000000000000000000000000000000000000000000000000000000000000000")) == T(false, -256, hex!"dcf26a9fe1839fb9d75cdea21c8c8ea0f2aceb4102bc98bbeec1c9d4973206f9"));
    assert(G.logarithm(T(false, -248, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(false, -245, hex!"9136c98848df39e8aeddbe1d69b61c46a7e5698d68dbb0561ad16cadae67c756"));
    assert(G.logarithm(T(false, -247, hex!"8000000000000000000000000000000000000000000000000000000000000000")) == T(false, -245, hex!"9136c98848df39e8aeddbe1d69b61c46a7e5698d68dbb0561ad16cadae67c757"));
    assert(G.logarithm(T(false, -247, hex!"8000000000000000000000000000000000000000000000000000000000000001")) == T(false, -245, hex!"9136c98848df39e8aeddbe1d69b61c46a7e5698d68dbb0561ad16cadae67c758"));
    assert(G.logarithm(T(false, 45, hex!"8000000000000000000000000000000000000000000000000000000000000000")) == T(false, 52, hex!"cef1b41669df22955096d67624a0b915601d12b10e4af57d5dc7575f47ac5d2b"));
    assert(G.correction(G.S.from(T(false, -247, hex!"8000000000000000000000000000000000000000000000000000000000000000"))).finish!256 == T(false, -267, hex!"aaaaa4fa51451319334c6726fbae862a7da5f677748f7360c25b86bb594287c6"));
    assert(G.correction(G.S.from(T(false, 45, hex!"8000000000000000000000000000000000000000000000000000000000000000"))).finish!256 == T(false, -559, hex!"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"));
    assert(G.logarithm(T(false, long.min, U(1))) == T(false, -193, hex!"b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"));
    assert(G.logarithm(T(false, long.max - 64, U(1))) == T(false, long.max - 257, hex!"b17217f7d1cf794fadf387bf78972d72bb520ff47137712e8e84fdd55a1e770c"));
    assert(G.logarithm(T(false, long.max - 1, U(1) << 255)).isInfinity);
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!512;
    alias T = G.T;
    alias U = UInt!512;
    enum hex(string s) = U.fromHexString(s);
    assert(G.logarithm(T(false, -512, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(false, -1024, hex!"93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbd8e4b59fa03a9f0eed0649ccb621057d11056ae9132135a08e43b4673d74bafeb"));
    assert(G.logarithm(T(false, -511, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")) == T(true, -1023, hex!"93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbd8e4b59fa03a9f0eed0649ccb621057d11056ae9132135a08e43b4673d74bafe9"));
    assert(G.logarithm(T(false, -511, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(true, -1024, hex!"d8773039049e70b65c8380fdfd5a6952bc62678a13467fcde7f841971eca0084e3694c0bf8ac1e225f36c6693bdf505ddf52a2dd9bd94bee378973185168a02a"));
    assert(G.logarithm(T(false, -510, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")) == T(false, -1023, hex!"d8773039049e70b65c8380fdfd5a6952bc62678a13467fcde7f841971eca0084e3694c0bf8ac1e225f36c6693bdf505ddf52a2dd9bd94bee378973185168a02e"));
    assert(G.logarithm(T(false, -513, hex!"c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")) == T(false, -512, hex!"dcf26a9fe1839fb9d75cdea21c8c8ea0f2aceb4102bc98bbeec1c9d4973206f8cba3140ee3f631cdf2b06de1cd9f2ddd561206d3803cbb409a7ee4eccd7ffb72"));
    assert(G.logarithm(T(false, -500, hex!"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")) == T(false, -497, hex!"ea24a91c624a57472e6fda9a6164542e2dec18c43f71febca0d06d70145ee1fdd93693e2512f4561b562d4f1ebade511f4710f837123ac3bd04593d5c4143d0d"));
    assert(G.logarithm(T(false, -499, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")) == T(false, -497, hex!"ea24a91c624a57472e6fda9a6164542e2dec18c43f71febca0d06d70145ee1fdd93693e2512f4561b562d4f1ebade511f4710f837123ac3bd04593d5c4143d0e"));
    assert(G.logarithm(T(false, -499, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")) == T(false, -497, hex!"ea24a91c624a57472e6fda9a6164542e2dec18c43f71febca0d06d70145ee1fdd93693e2512f4561b562d4f1ebade511f4710f837123ac3bd04593d5c4143d10"));
    assert(G.logarithm(T(false, -211, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")) == T(false, -204, hex!"cef1b41669df22955096d67624a0b915601d12b10e4af57d5dc7575f47ac5d2b738c2a6e00c9bd68dc7be0a4c185dc4ceb41a362c7c5030e518a9830cd31d37d"));
    assert(G.correction(G.S.from(T(false, -499, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))).finish!512 == T(false, -527, hex!"aaaaaaa4fa4fa69a69a56256271ba804269009857150e15da5409edae33d9f3dfeb4ca5b0fa9ece8d7f761e1c07fb1ea68c994975bef3d184587e2dcd1c9ba28"));
    assert(G.correction(G.S.from(T(false, -211, hex!"80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))).finish!512 == T(false, -815, hex!"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"));
    assert(G.logarithm(T(false, long.min, U(1))) == T(false, -449, hex!"b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"));
    assert(G.logarithm(T(false, long.max - 64, U(1))) == T(false, long.max - 513, hex!"b17217f7d1cf794fadf387bf78972d72bb520ff47137712e8e84fdd55a1e770bcd1299a38118a24ca9e95683794c75b5775a8a99ae3662abb5a2c3dcf2e8d389"));
    assert(G.logarithm(T(false, long.max - 1, U(1) << 511)).isInfinity);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!64;
    alias T = G.T;
    alias U = UInt!64;
    foreach (root; [T(1), T(false, 1, U(1))])
    {
        const result = G.logarithm(root);
        assert(!result.coefficient && !result.sign);
    }
    assert(G.logarithm(T(false, -3, U(3))) == G.logarithm(T(0.375)));

    // Adjacent inputs on the final output grid (units of 2^^(long.max-1)).
    enum last = U.fromHexString("b8aa3b295c17f0be");
    assert(G.logarithm(T(false, long.max - 63, last)) == T(false, long.max - 1, U.max));
    assert(G.logarithm(T(false, long.max - 63, last + 1)).isInfinity);

    // At this shape log Gamma overflows. The separately scaled correction
    // still retains the leading 1/(12*z), even below the Fp exponent floor.
    const z = G.S.from(T(false, long.max - 1, U(1) << 63));
    const scaled = G.correction(z).multiply(z);
    enum twelfth = T(false, -67, U.fromHexString("aaaaaaaaaaaaaaab"));
    assert(scaled.finish!64 == twelfth);

    // Squaring the reciprocal would wrap the separate exponent. The
    // leading correction itself fits, including at the upper input limit.
    auto large = G.S.from(T(1));
    large.exponent += (UInt!128(1) << 126) + 1024;
    assert(G.correction(large).multiply(large).finish!64 == twelfth);
    large = G.S.from(T(0.75));
    large.exponent = UInt!128.max - 2;
    assert(G.correction(large).exponent == 0);
    assert(G.correction(large).multiply(large).finish!64 == twelfth);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!64;
    alias T = G.T;
    alias U = UInt!64;
    assert(!G.logGamma1p(T(0)).coefficient);
    assert(!G.logGamma1p(T(1)).coefficient);
    // MPFR lngamma at the exact sum, including increments that disappear
    // when added to one in the input type.
    enum euler = U.fromHexString("93c467e37db0c7a5");
    assert(cast(T)G.logGamma1p(T(false, -900, U(1))) == T(true, -964, euler));
    assert(cast(T)G.logGamma1p(T(false, -64, U(1))) == T(true, -128, euler - 1));
    assert(cast(T)G.logGamma1p(T(false, -64, U.max)) ==
        T(true, -129, U(0xd8773039049e70b6UL)));
    assert(cast(T)G.logGamma1p(T(0.375)) == T(true, -67, U(0xf129ace61dd11e97UL)));
    // Adjacent Fp!64 increments at 2^256. MPFR lngamma(1+a) at 2048
    // bits covers the unshifted Stirling evaluation and the large-a shortcut.
    assert(cast(T)G.logGamma1p(T(false, 192, U.max)) ==
        T(false, 200, U(0xb07217f7d1cf79abUL)));
    assert(cast(T)G.logGamma1p(T(false, 256, U(1))) ==
        T(false, 200, U(0xb07217f7d1cf79acUL)));
    const floor = G.logGamma1p(T(false, long.min, U(1)));
    assert(floor.sign && floor.exponent == long.min && floor.coefficient == 1);
    const three = T(false, long.min, U(3));
    const scaled = G.smallIncrementMagnitude(three).divide(G.S.from(three));
    assert(scaled.finish!64 == T(false, -64, euler));
    assert(cast(T)G.guardedLogarithm(T(3)) == G.logarithm(T(3)));
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias G = PositiveLogGamma!512;
    alias T = G.T;
    alias U = UInt!512;
    enum expected = U.fromHexString(
        "93c467e37db0c7a4d1be3f810152cb56a1cecc3af65cc0190c03df34709affbd" ~
        "8e4b59fa03a9f0eed0649ccb621057d11056ae9132135a08e43b4673d74bafea");
    // Opposite sides of the tiny-increment shortcut, rounded by MPFR.
    assert(cast(T)G.logGamma1p(T(false, -900, U(1))) == T(true, -1412, expected));
    assert(cast(T)G.logGamma1p(T(false, -512, U(1))) == T(true, -1024, expected));
}
