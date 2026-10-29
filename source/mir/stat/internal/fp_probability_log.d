/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_probability_log;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt, extendedMul;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;

// Guarded logarithms for distribution calculations. The probability methods
// return positive magnitudes and require callers to handle endpoints; the
// positive-value and count methods return signed Fp values, including zero.
// Keep 192 guard bits through the geometric sum and round only at its output.
// Magnitudes are bounded by -log(p) < 2^63 and -log(1-p) <= bits*log(2).
// Even with a 64-bit count, their weighted sum is far below Fp overflow.
// This is guarded evaluation, not a universal correctly-rounded guarantee.
package(mir.stat) struct ProbabilityLog(uint bits)
{
    alias S = ProbabilityScaled!(bits + 192);
    enum fractionBits = 2 * bits + 448;
    alias I = UInt!(fractionBits + 64);


    @safe pure nothrow @nogc:

    private static I multiply(I a, I b)
    {
        auto product = extendedMul(a,b);
        return (product >> fractionBits).toSize!(fractionBits + 64);
    }

    // 2*atanh(a/b), with 0 <= a/b <= 1/3. All truncation is below
    // the fixed-point grid; the series stops only when its term vanishes.
    private static I ratio(I a, I b)
    {
        auto quotient = a.toSize!(2 * fractionBits + 128);
        quotient <<= fractionBits;
        quotient /= b;
        const z = quotient.toSize!(fractionBits + 64);
        const square = multiply(z,z);
        I power = z;
        I sum = z;
        for (uint odd = 3; ; odd += 2) {
            power = multiply(power,square);
            auto term = power;
            term /= UInt!64(odd);
            if (!term) break;
            sum += term;
        }
        return sum << 1;
    }

    private static S scaled(I value)
    {
        auto result = Fp!(bits + 192)(value);
        result.exponent -= fractionBits;
        return S.from(result);
    }

    // For c*2^e < 1, take b=2^(floor(log2(c))+1). Then
    // -log(p) = (-e-log2(b))*log(2) + 2*atanh((b-c)/(b+c)).
    // Both terms are nonnegative, including p immediately below one.
    static S probability(Fp!bits p)
    {
        const top = bits - p.coefficient.ctlz;
        const base = I(1) << top;
        const c = p.coefficient.toSize!(fractionBits + 64);
        const count = (0UL - cast(ulong)p.exponent) - top;
        auto logarithm = ratio(base-c,base+c);
        if (count) {
            auto term = ratio(I(1),I(3));
            term *= count;
            logarithm += term;
        }
        return scaled(logarithm);
    }

    // Signed log of any finite positive Fp value, with the same guard width
    // as the probability helpers. No intermediate reciprocal or widened Fp
    // input is formed, including at either end of the input exponent range.
    static Fp!(bits + 192) logPositive(Fp!bits value)
        in (!value.sign && !!value.coefficient && value.exponent != long.max,
            "value must be finite and positive")
    {
        return logDyadic(value.coefficient, value.exponent);
    }

    // Log of an exact integer count. UInt!128 also admits endpoints such as
    // size_t.max+1 without overflow or rounding through an Fp input.
    static Fp!(bits + 192) logCount(UInt!128 count)
        in (!!count, "count must be positive")
    {
        return logDyadic(count, 0);
    }

    private static Fp!(bits + 192) logDyadic(uint inputBits)(UInt!inputBits coefficient, long exponent)
    {
        const top = inputBits - coefficient.ctlz - 1;
        const c = coefficient.toSize!(fractionBits + 64);
        const negative = exponent < -cast(long)top;
        const base = I(1) << (top + negative);
        // Range reduction keeps the two terms the same sign. Near one,
        // the power-of-two term vanishes, avoiding cancellation of log(2).
        const count = negative
            ? (0UL - cast(ulong)exponent) - top - 1
            : exponent >= 0 ? cast(ulong)exponent + top
                            : top - (0UL - cast(ulong)exponent);
        auto logarithm = negative ? ratio(base-c,base+c) : ratio(c-base,c+base);
        if (count) {
            auto term = ratio(I(1),I(3));
            term *= count;
            logarithm += term;
        }
        alias W = Fp!(bits + 192);
        if (!logarithm)
            return W(0); // Exactly one, including unnormalized input.
        auto result = W(logarithm);
        result.exponent -= fractionBits;
        result.sign = negative;
        return result;
    }

    static S complement(Fp!bits p)
    {
        // -log(1-p)/p = 1 + p/2 + ... . In this range the relative
        // correction is below 2^(-bits-256), including subnormal inputs.
        // Avoid widening Fp at its exponent floor: S keeps a separate exponent.
        const top = bits - p.coefficient.ctlz;
        if (p.exponent <= -cast(long)bits - 256 - top)
            return S.from(p);
        const c = p.coefficient.toSize!(fractionBits + 64);
        const one = I(1) << cast(size_t)-p.exponent;
        if (c <= (one >> 1))
            return scaled(ratio(c,(one << 1)-c));
        // 1-p is an exact dyadic here; cancellation exposes rather than loses
        // significant bits. Its coefficient fits the original input width.
        auto q = Fp!bits(false,p.exponent,(one-c).toSize!bits);
        return probability(q);
    }
}
// BEGIN helper probability log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    {
        enum bits = 64;
        alias T = Fp!64;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!64.fromHexString("1")),T(false,-1,UInt!64.fromHexString("b17217f7d1cf79ac")),T(false,long.min,UInt!64.fromHexString("1"))),
            Reference(T(false,-2000,UInt!64.fromHexString("1")),T(false,-53,UInt!64.fromHexString("ad496b6802e49cd2")),T(false,-2063,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-385,UInt!64.fromHexString("1")),T(false,-55,UInt!64.fromHexString("856e4b05d94482fe")),T(false,-448,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-384,UInt!64.fromHexString("1")),T(false,-55,UInt!64.fromHexString("851591f9dd5b9b41")),T(false,-447,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-64,UInt!64.fromHexString("1")),T(false,-58,UInt!64.fromHexString("b17217f7d1cf79ac")),T(false,-127,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-1,UInt!64.fromHexString("1")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79ac")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,long.min,UInt!64.fromHexString("3")),T(false,-1,UInt!64.fromHexString("b17217f7d1cf79aa")),T(false,long.min,UInt!64.fromHexString("3"))),
            Reference(T(false,-2,UInt!64.fromHexString("3")),T(false,-65,UInt!64.fromHexString("934b1089a6dc93c2")),T(false,-63,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(T(false,-195,UInt!64.fromHexString("155d3680a0b3f28b")),T(false,-57,UInt!64.fromHexString("ba924d3daa964b98")),T(false,-198,UInt!64.fromHexString("aae9b405059f9458"))),
            Reference(T(false,-277,UInt!64.fromHexString("2dfcf807a025849a")),T(false,-56,UInt!64.fromHexString("955b6db83dc9254e")),T(false,-279,UInt!64.fromHexString("b7f3e01e80961268"))),
            Reference(T(false,-349,UInt!64.fromHexString("47da373f7dfb4d2f")),T(false,-56,UInt!64.fromHexString("c6d14852b454feba")),T(false,-350,UInt!64.fromHexString("8fb46e7efbf69a5e"))),
            Reference(T(false,-111,UInt!64.fromHexString("6ce38d733ed6ff1b")),T(false,-58,UInt!64.fromHexString("85bb2748be588940")),T(false,-112,UInt!64.fromHexString("d9c71ae67dae5ad8"))),
            Reference(T(false,-97,UInt!64.fromHexString("739145488dc18dc8")),T(false,-59,UInt!64.fromHexString("bd5a7b03c7b4674d")),T(false,-98,UInt!64.fromHexString("e7228a91359906e7"))),
            Reference(T(false,-64,UInt!64.fromHexString("7fffffffffffffff")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79ae")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(T(false,long.min,UInt!64.fromHexString("8000000000000000")),T(false,-1,UInt!64.fromHexString("b17217f7d1cf7954")),T(false,long.min,UInt!64.fromHexString("8000000000000000"))),
            Reference(T(false,-64,UInt!64.fromHexString("8000000000000001")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79aa")),T(false,-64,UInt!64.fromHexString("b17217f7d1cf79ae"))),
            Reference(T(false,-245,UInt!64.fromHexString("c3590e78d844c8c3")),T(false,-57,UInt!64.fromHexString("fb75c72b86861549")),T(false,-245,UInt!64.fromHexString("c3590e78d844c8c3"))),
            Reference(T(false,-182,UInt!64.fromHexString("c4283a1bd25a8240")),T(false,-57,UInt!64.fromHexString("a41d81821f3b6956")),T(false,-182,UInt!64.fromHexString("c4283a1bd25a8240"))),
            Reference(T(false,-203,UInt!64.fromHexString("dbc71f0e08eae614")),T(false,-57,UInt!64.fromHexString("c100026a481f7aa7")),T(false,-203,UInt!64.fromHexString("dbc71f0e08eae614"))),
            Reference(T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(false,-1,UInt!64.fromHexString("b17217f7d1cf7953")),T(false,long.min,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-385,UInt!64.fromHexString("ffffffffffffffff")),T(false,-56,UInt!64.fromHexString("de80100dbe152790")),T(false,-385,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-384,UInt!64.fromHexString("ffffffffffffffff")),T(false,-56,UInt!64.fromHexString("ddce9df5c6435817")),T(false,-384,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(false,-127,UInt!64.fromHexString("8000000000000000")),T(false,-58,UInt!64.fromHexString("b17217f7d1cf79ac"))),
        ];
        foreach (r; references) {
            assert(ProbabilityLog!bits.probability(r.p).finish!bits == r.logP);
            assert(ProbabilityLog!bits.complement(r.p).finish!bits == r.logQ);
        }
    }
    {
        enum bits = 128;
        alias T = Fp!128;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!128.fromHexString("1")),T(false,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(false,long.min,UInt!128.fromHexString("1"))),
            Reference(T(false,-2000,UInt!128.fromHexString("1")),T(false,-117,UInt!128.fromHexString("ad496b6802e49cd1c3285d6273db44e7")),T(false,-2127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-513,UInt!128.fromHexString("1")),T(false,-119,UInt!128.fromHexString("b1cad103cdb861689fc8a571cff4f02b")),T(false,-640,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-512,UInt!128.fromHexString("1")),T(false,-119,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(false,-639,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-128,UInt!128.fromHexString("1")),T(false,-121,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(false,-255,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-1,UInt!128.fromHexString("1")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,long.min,UInt!128.fromHexString("3")),T(false,-65,UInt!128.fromHexString("b17217f7d1cf79a9976664426391a3c2")),T(false,long.min,UInt!128.fromHexString("3"))),
            Reference(T(false,-2,UInt!128.fromHexString("3")),T(false,-129,UInt!128.fromHexString("934b1089a6dc93c1df5bb3b60554e152")),T(false,-127,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(T(false,-382,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff")),T(false,-120,UInt!128.fromHexString("b64bbf32e5853dc6a1241a3a00126877")),T(false,-390,UInt!128.fromHexString("803bc4493d4d4a0842bef4c83acaff00"))),
            Reference(T(false,-191,UInt!128.fromHexString("a80121e07c0fd2678904cf45e2c018")),T(false,-122,UInt!128.fromHexString("c689de4538e16ebd0e27b39d9833320e")),T(false,-199,UInt!128.fromHexString("a80121e07c0fd267897310c25aab2961"))),
            Reference(T(false,-360,UInt!128.fromHexString("64f2572158cb0b0e40975b1d17ec17bd")),T(false,-120,UInt!128.fromHexString("a1bda0903fb3ed35ce48f971d1474472")),T(false,-361,UInt!128.fromHexString("c9e4ae42b196161c812eb63a2fd82f7a"))),
            Reference(T(false,-128,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6b1")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6ad"))),
            Reference(T(false,long.min,UInt!128.fromHexString("80000000000000000000000000000000")),T(false,-65,UInt!128.fromHexString("b17217f7d1cf78fbbaafebb5d8183e3d")),T(false,long.min,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(T(false,-128,UInt!128.fromHexString("80000000000000000000000000000001")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6ad")),T(false,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6b1"))),
            Reference(T(false,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb")),T(false,-121,UInt!128.fromHexString("e8c44510834cce7c54fe08dccac9d95f")),T(false,-295,UInt!128.fromHexString("88a6330af5f71bfebdcf62dc5cea91fb"))),
            Reference(T(false,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24")),T(false,-120,UInt!128.fromHexString("a2c394eebd064edee58f98f5a40c10f3")),T(false,-362,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24"))),
            Reference(T(false,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b0e9db0254")),T(false,-121,UInt!128.fromHexString("849dceb8353d6aebee7b02b2406be60b")),T(false,-223,UInt!128.fromHexString("a1bb70e4b5310fe8a322d6b150084f92"))),
            Reference(T(false,-201,UInt!128.fromHexString("d4daf2ca7a55cf775dd9cd750c9be8c2")),T(false,-122,UInt!128.fromHexString("cb2321f2965d0451d33dac9c0907e374")),T(false,-201,UInt!128.fromHexString("d4daf2ca7a55cf775e060c4c650c573c"))),
            Reference(T(false,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc04caadc")),T(false,-121,UInt!128.fromHexString("91ecd8d283c2b13d18884a2d32bd0c4f")),T(false,-233,UInt!128.fromHexString("d55f5f2cec0ab39a3a52871fc07920d2"))),
            Reference(T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(false,-65,UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5")),T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(false,-119,UInt!128.fromHexString("856e4b05d94482fdad4fb88bcef8327f")),T(false,-513,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(false,-119,UInt!128.fromHexString("851591f9dd5b9b40d76ac6b202f63903")),T(false,-512,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(false,-255,UInt!128.fromHexString("80000000000000000000000000000000")),T(false,-121,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
        ];
        foreach (r; references) {
            assert(ProbabilityLog!bits.probability(r.p).finish!bits == r.logP);
            assert(ProbabilityLog!bits.complement(r.p).finish!bits == r.logQ);
        }
    }
    {
        enum bits = 256;
        alias T = Fp!256;
        struct Reference { T p; T logP; T logQ; }
        enum references = [
            Reference(T(false,long.min,UInt!256.fromHexString("1")),T(false,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(false,long.min,UInt!256.fromHexString("1"))),
            Reference(T(false,-2000,UInt!256.fromHexString("1")),T(false,-245,UInt!256.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84f")),T(false,-2255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-769,UInt!256.fromHexString("1")),T(false,-246,UInt!256.fromHexString("8541ee7fdb500f1f425d3f9ee8f735c11c86af2d9f8f2ecfb2ec54ca7fa3265f")),T(false,-1024,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-768,UInt!256.fromHexString("1")),T(false,-246,UInt!256.fromHexString("851591f9dd5b9b40d76ac6b202f6390370b6725cd5f288a22789d184a8c03ba1")),T(false,-1023,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-256,UInt!256.fromHexString("1")),T(false,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(false,-511,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-1,UInt!256.fromHexString("1")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,long.min,UInt!256.fromHexString("3")),T(false,-193,UInt!256.fromHexString("b17217f7d1cf79a9976664426391a3c1f8c0287c6821bcc20de491a81be9de4b")),T(false,long.min,UInt!256.fromHexString("3"))),
            Reference(T(false,-2,UInt!256.fromHexString("3")),T(false,-257,UInt!256.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d96")),T(false,-255,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(T(false,-503,UInt!256.fromHexString("1e9b726592d30ba658d702c4cd40a6d64c302ae640625db7771e120788e25f79")),T(false,-248,UInt!256.fromHexString("ad54cfc4ceb558d41b83731273e0d5b69e1baeab78d4c672c565f798f4bc71bb")),T(false,-506,UInt!256.fromHexString("f4db932c96985d32c6b816266a0536b2618157320312edbbb8f0903c4712fbe5"))),
            Reference(T(false,-500,UInt!256.fromHexString("3ec4002d4882689ddd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a0739")),T(false,-248,UInt!256.fromHexString("aa889f60a17a6ac22409e5390478487377a9195c206d0f262c1ffd000e2260ce")),T(false,-502,UInt!256.fromHexString("fb1000b52209a2777504248f67cf06b7f762c7cdff5886c94eb476365a281ed0"))),
            Reference(T(false,-310,UInt!256.fromHexString("4720f4a871f2ec93ca5547d5408492cb24034b668fda7a54a82657222d4b3a8e")),T(false,-250,UInt!256.fromHexString("9ad7b0641c6f5d67d755b34e15da51116dc2e7ac2d75e04ba389c62c9517790d")),T(false,-311,UInt!256.fromHexString("8e41e950e3e5d976a1da5db70c1967f75b26a1c198a4aac4ce591fb21d3f54b4"))),
            Reference(T(false,-423,UInt!256.fromHexString("4d11226bb2db7011471dca27dde7b62d311503cec0f7abd7f1bd17a3c338aca7")),T(false,-249,UInt!256.fromHexString("e9e9839efca15d57a7d00a402ee7e8cff36f3691f107f7eb3c992d1002903be1")),T(false,-424,UInt!256.fromHexString("9a2244d765b6e0228e3b944fbbcf6c5a622a079d821dbe4f983b3c5e4793d45c"))),
            Reference(T(false,-430,UInt!256.fromHexString("7612335515ecf948ebdb260951f5a219d41440275124cbcfcbdd58ef250485fa")),T(false,-249,UInt!256.fromHexString("f2c353f4ddfc69c00ea9259be92cc6e46195964656615a6647d91df1a9880724")),T(false,-431,UInt!256.fromHexString("ec2466aa2bd9f291d7b64c12a3eb4433a828804ea24a7172ba317bc082a21cf6"))),
            Reference(T(false,-256,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2e")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2a"))),
            Reference(T(false,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(false,-193,UInt!256.fromHexString("b17217f7d1cf784a4897f3e4089e92730d537a4e9491450a256d50c33f700b73")),T(false,long.min,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2a")),T(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2e"))),
            Reference(T(false,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de07984ea35c2a2dc0399d61d2294e0d")),T(false,-249,UInt!256.fromHexString("cfc182533e5dac574b7d8fee956fa9ce8b87699f0afb6e22cfa5e91111fc105e")),T(false,-405,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de079983aae9d0063609b2a9f0ed2601"))),
            Reference(T(false,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa08801907f3462d0e84f25bca22b80357ef032e8edb2d5a5")),T(false,-250,UInt!256.fromHexString("e4e42a2e3d8a4bda61b942d2db67f21a4b7363237f2c65e966c5cde7a8f7c141")),T(false,-338,UInt!256.fromHexString("ae3d3c82db49ce4aa088106368d15ba936381f65cba0da3a96f5267f2515c8ae"))),
            Reference(T(false,-294,UInt!256.fromHexString("eee66cfb0ccb08a554bda8b2c9fdfd2bcc24375ff6aefe360e542751a3681ab8")),T(false,-251,UInt!256.fromHexString("d345118635839daa37a4f317628449bcc9c2a201fa08dd5642cdc23397deaa14")),T(false,-294,UInt!256.fromHexString("eee66cfb0e88eb296a5c90e10dae57ee4e21e53fc6eaa4427494b698b0a02a3d"))),
            Reference(T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(false,-193,UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18")),T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(false,-247,UInt!256.fromHexString("b1cad103cdb861689fc8a571cff4f02a9893bcc805d20288a0d21de73970cfa9")),T(false,-769,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(false,-247,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),T(false,-768,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(false,-511,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(false,-248,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
        ];
        foreach (r; references) {
            assert(ProbabilityLog!bits.probability(r.p).finish!bits == r.logP);
            assert(ProbabilityLog!bits.complement(r.p).finish!bits == r.logQ);
        }
    }
}
// END helper probability log references
// BEGIN positive logarithm references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    {
        alias L = ProbabilityLog!64;
        alias W = Fp!256;
        alias T = Fp!64;
        struct ValueReference { T value; W logarithm; }
        enum values = [
            ValueReference(T(false,long.min,UInt!64.fromHexString("1")),W(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            ValueReference(T(false,-1,UInt!64.fromHexString("1")),W(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            ValueReference(T(false,0,UInt!64.fromHexString("1")),W(0)),
            ValueReference(T(false,1,UInt!64.fromHexString("1")),W(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            ValueReference(T(false,long.max-1,UInt!64.fromHexString("1")),W(false,-193,UInt!256.fromHexString("b17217f7d1cf79a9041b53b8bcb51000196474c662ccdb7086400ac1c1482176"))),
            ValueReference(T(false,-2,UInt!64.fromHexString("3")),W(true,-257,UInt!256.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d96"))),
            ValueReference(T(false,-1,UInt!64.fromHexString("3")),W(false,-257,UInt!256.fromHexString("cf991f65fcc25f95b46bb37a02910c0cfa41ff668a8faf856715af1d8b3c76c2"))),
            ValueReference(T(false,1516,UInt!64.fromHexString("8eae61310a69cea")),W(false,-245,UInt!256.fromHexString("887a17b04c1d86741c692bd6f13a2dddc468b66a762ab9c461afb52485e56d13"))),
            ValueReference(T(false,284,UInt!64.fromHexString("47da373f7dfb4d2f")),W(false,-248,UInt!256.fromHexString("eff1d4f1117505201431bf6dae099cdc471dd2503b4fb4cacf68d168e4bd11a6"))),
            ValueReference(T(false,-1617,UInt!64.fromHexString("6ce38d733ed6ff1b")),W(true,-245,UInt!256.fromHexString("86a9ff5a41fb95829553f4c5a1f77504b33848b23cf21b0f4456afc89e08b294"))),
            ValueReference(T(false,-63,UInt!64.fromHexString("8000000000000001")),W(false,-319,UInt!256.fromHexString("ffffffffffffffff000000000000000155555555555555535555555555555559"))),
            ValueReference(T(false,-151,UInt!64.fromHexString("8dc18dc75ad479dc")),W(true,-250,UInt!256.fromHexString("f39459f3e3df6195176cc07601cb576afa0648e86ab0d31acd7d719f738e4ea0"))),
            ValueReference(T(false,-1659,UInt!64.fromHexString("a0b3f28a45b19904")),W(true,-245,UInt!256.fromHexString("8a4121c68d18c7df8639d12abc9c036d2c7cc142caaaff55eb66ef68f91c209b"))),
            ValueReference(T(false,459,UInt!64.fromHexString("b3ff8d25a302d48c")),W(false,-247,UInt!256.fromHexString("b514f5997b25aa866cf7b59b88901e2e81e42a87b1fd785b12fcb7214fc03fe6"))),
            ValueReference(T(false,1646,UInt!64.fromHexString("c3590e78d844c8c3")),W(false,-245,UInt!256.fromHexString("94205c6eb9f190a375b745ca3c17a16cef25600560e065f263ac6e5a66f3759a"))),
            ValueReference(T(false,-1053,UInt!64.fromHexString("c4283a1bd25a8240")),W(true,-246,UInt!256.fromHexString("ab727c134e9da6e8a4c426f4932ba171772eec510e91fa565cf577d61c997273"))),
            ValueReference(T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),W(true,-193,UInt!256.fromHexString("b17217f7d1cf795310d7b7af1c3620cc4f197724791d5e8e106b84223f4fe368"))),
            ValueReference(T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),W(true,-319,UInt!256.fromHexString("800000000000000040000000000000002aaaaaaaaaaaaaaacaaaaaaaaaaaaaab"))),
            ValueReference(T(false,long.max-1,UInt!64.fromHexString("ffffffffffffffff")),W(false,-193,UInt!256.fromHexString("b17217f7d1cf7a01bd274fa1a471e5e30b3e40c85c48330fffe19dfb0da3383a"))),
        ];
        foreach (r; values)
            assert(L.logPositive(r.value) == r.logarithm);
        struct CountReference { UInt!128 count; W logarithm; }
        enum counts = [
            CountReference(UInt!128.fromHexString("1"),W(0)),
            CountReference(UInt!128.fromHexString("2"),W(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            CountReference(UInt!128.fromHexString("3"),W(false,-255,UInt!256.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6"))),
            CountReference(UInt!128.fromHexString("11"),W(false,-254,UInt!256.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e"))),
            CountReference(UInt!128.fromHexString("ffffffff"),W(false,-251,UInt!256.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49"))),
            CountReference(UInt!128.fromHexString("100000001"),W(false,-251,UInt!256.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b64"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffff"),W(false,-250,UInt!256.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d7"))),
            CountReference(UInt!128.fromHexString("10000000000000000"),W(false,-250,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            CountReference(UInt!128.fromHexString("10000000000000001"),W(false,-250,UInt!256.fromHexString("b17217f7d1cf79abcde3b39803f2f6af3ef343267298b62d8b626cb0e1004f81"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"),W(false,-249,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af3ef343267298b62d8a0d175b8baafa2c"))),
        ];
        foreach (r; counts)
            assert(L.logCount(r.count) == r.logarithm);
    }
    {
        alias L = ProbabilityLog!128;
        alias W = Fp!320;
        alias T = Fp!128;
        struct ValueReference { T value; W logarithm; }
        enum values = [
            ValueReference(T(false,long.min,UInt!128.fromHexString("1")),W(true,-257,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98"))),
            ValueReference(T(false,-1,UInt!128.fromHexString("1")),W(true,-320,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98"))),
            ValueReference(T(false,0,UInt!128.fromHexString("1")),W(0)),
            ValueReference(T(false,1,UInt!128.fromHexString("1")),W(false,-320,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98"))),
            ValueReference(T(false,long.max-1,UInt!128.fromHexString("1")),W(false,-257,UInt!320.fromHexString("b17217f7d1cf79a9041b53b8bcb51000196474c662ccdb7086400ac1c1482175bf8418b23f3fc3e9"))),
            ValueReference(T(false,-2,UInt!128.fromHexString("3")),W(true,-321,UInt!320.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d9649b2d2dd9e2181c1"))),
            ValueReference(T(false,-1,UInt!128.fromHexString("3")),W(false,-321,UInt!320.fromHexString("cf991f65fcc25f95b46bb37a02910c0cfa41ff668a8faf856715af1d8b3c76c185be19633db5d770"))),
            ValueReference(T(false,1776,UInt!128.fromHexString("3a52871fc04caadb53a4493588a6330b")),W(false,-309,UInt!320.fromHexString("a4c8bcbead4e457b9b8a8eb4013b99464e5ca8087e8b30f9ac3dedb33b6947ebcb4d6985bb131181"))),
            ValueReference(T(false,-385,UInt!128.fromHexString("58cb0b0e40975b1d17ec17bc75633c70")),W(true,-312,UInt!320.fromHexString("b3329c0a417a038b993e7137481bb4f910ecdf14ab1edfaea8fb43d511e120c507d676c714662468"))),
            ValueReference(T(false,-43,UInt!128.fromHexString("5dd9cd750c9be8c1745b70e7dc4993b2")),W(false,-314,UInt!320.fromHexString("e7a7f9dd7ac8879dffc44287b2e20e2d6b4235981ce5b49950933a9aac01ddce38663564b987b577"))),
            ValueReference(T(false,1591,UInt!128.fromHexString("678904cf45e2c0172fa54868a1bb70e5")),W(false,-309,UInt!320.fromHexString("94d3abeec40bdc61703c1e2da8f49cd713a471addbc04fec8e0c61c951991feb77fecbd33742eac6"))),
            ValueReference(T(false,-127,UInt!128.fromHexString("80000000000000000000000000000001")),W(false,-447,UInt!320.fromHexString("ffffffffffffffffffffffffffffffff000000000000000000000000000000015555555555555555"))),
            ValueReference(T(false,400,UInt!128.fromHexString("9121269a633afb42483c5c0f94afce24")),W(false,-311,UInt!320.fromHexString("b6b5033474bea0a538b015ce6bda90a9827b9dfa455099a2b538d0825da04f61c628cd7b4f56a008"))),
            ValueReference(T(false,899,UInt!128.fromHexString("a322d6b0e9db025324afe5a4d4daf2cb")),W(false,-310,UInt!320.fromHexString("b1da57070edf49ad1ee5c13f0a350d522383499c93b61585f01c1764119af0e53239ea432f60b555"))),
            ValueReference(T(false,1935,UInt!128.fromHexString("bdcf62dc5cea91fa1f86d37600a80122")),W(false,-309,UInt!320.fromHexString("b2b53b3abf3b5c2a9933b35014ede6a7103556d5acc53cea87f5539aa30c5a1e44577b2d4af0577a"))),
            ValueReference(T(false,-1868,UInt!128.fromHexString("c83acafe34c61c51c078fe2ed55f5f2d")),W(true,-309,UInt!320.fromHexString("96ca4c3d60f2101959b82a8ea90162f5f67f76c09da96dd2c5a8864832d9dbad01fcc81e5db313eb"))),
            ValueReference(T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),W(true,-257,UInt!320.fromHexString("b17217f7d1cf78fa57cbbbc634794ae55d3fab227fa206ee96c9f0e8f2f4cca1daa11a94c2f180b2"))),
            ValueReference(T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),W(true,-447,UInt!320.fromHexString("80000000000000000000000000000000400000000000000000000000000000002aaaaaaaaaaaaaab"))),
            ValueReference(T(false,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),W(false,-257,UInt!320.fromHexString("b17217f7d1cf7a5a76334b8a8c2ebbc9fd180cca55c38aaf7983313459fe4effcc9b743dea39efcf"))),
        ];
        foreach (r; values)
            assert(L.logPositive(r.value) == r.logarithm);
        struct CountReference { UInt!128 count; W logarithm; }
        enum counts = [
            CountReference(UInt!128.fromHexString("1"),W(0)),
            CountReference(UInt!128.fromHexString("2"),W(false,-320,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98"))),
            CountReference(UInt!128.fromHexString("3"),W(false,-319,UInt!320.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6554bc16906634c28"))),
            CountReference(UInt!128.fromHexString("11"),W(false,-318,UInt!320.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e40152cbb658d309a"))),
            CountReference(UInt!128.fromHexString("ffffffff"),W(false,-315,UInt!320.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49288c936241256034"))),
            CountReference(UInt!128.fromHexString("100000001"),W(false,-315,UInt!320.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b63fa39ae33f0074e52"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffff"),W(false,-314,UInt!320.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d6916320cb18965743"))),
            CountReference(UInt!128.fromHexString("10000000000000000"),W(false,-314,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98"))),
            CountReference(UInt!128.fromHexString("10000000000000001"),W(false,-314,UInt!320.fromHexString("b17217f7d1cf79abcde3b39803f2f6af3ef343267298b62d8b626cb0e1004f813c0dcb75c34101ee"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"),W(false,-313,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af3ef343267298b62d8a0d175b8baafa2be6b876206debac98"))),
        ];
        foreach (r; counts)
            assert(L.logCount(r.count) == r.logarithm);
    }
    {
        alias L = ProbabilityLog!256;
        alias W = Fp!448;
        alias T = Fp!256;
        struct ValueReference { T value; W logarithm; }
        enum values = [
            ValueReference(T(false,long.min,UInt!256.fromHexString("1")),W(true,-385,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144"))),
            ValueReference(T(false,-1,UInt!256.fromHexString("1")),W(true,-448,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144"))),
            ValueReference(T(false,0,UInt!256.fromHexString("1")),W(0)),
            ValueReference(T(false,1,UInt!256.fromHexString("1")),W(false,-448,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144"))),
            ValueReference(T(false,long.max-1,UInt!256.fromHexString("1")),W(false,-385,UInt!448.fromHexString("b17217f7d1cf79a9041b53b8bcb51000196474c662ccdb7086400ac1c1482175bf8418b23f3fc3e8b6b37a79934b68af96d96248954fb500"))),
            ValueReference(T(false,-2,UInt!256.fromHexString("3")),W(true,-449,UInt!448.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d9649b2d2dd9e2181c0b5715d8724b34e98f0028ec8c7910bcd"))),
            ValueReference(T(false,-1,UInt!256.fromHexString("3")),W(false,-449,UInt!448.fromHexString("cf991f65fcc25f95b46bb37a02910c0cfa41ff668a8faf856715af1d8b3c76c185be19633db5d76ff5b9486f7140e788ea5acda2badf36bb"))),
            ValueReference(T(false,861,UInt!256.fromHexString("471dca27dde7b62d311503cec0f7abd7f1bd17a3c338aca67bc9f7121e9b7266")),W(false,-438,UInt!448.fromHexString("c13dbb090db6e313faed8f4e0f0cf442115703e033bc2605f7e97f4975d5e3f0fc53d0344233df1f45dc543b62cb273c97e4a497f448393d"))),
            ValueReference(T(false,349,UInt!256.fromHexString("58d702c4cd40a6d64c302ae640625db7771e120788e25f781b082b744720f4a9")),W(false,-439,UInt!448.fromHexString("d125d9947113757cc925531b0ed7cc0d4abd82e9692928f4ae4c1f03b2a68dec52694111ee7284fe9a4885782757bc7ea0de0331b68dd388"))),
            ValueReference(T(false,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),W(false,-703,UInt!448.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff000000000000000000000000000000000000000000000000"))),
            ValueReference(T(false,895,UInt!256.fromHexString("8ca245fd14bb41832e863a0f77d20fb4de07984ea35c2a2dc0399d61d2294e0d")),W(false,-438,UInt!448.fromHexString("c74da8233590afec2dde61022a557df2ac03e70bee1c7183764de18516108d8e06a1faafe2258ca5259d4ce52f8a60dd24de069321c141ee"))),
            ValueReference(T(false,1716,UInt!256.fromHexString("a22b80357ef032e8edb2d5a4135af5d2eee66cfb0ccb08a554bda8b2c9fdfd2c")),W(false,-437,UInt!448.fromHexString("aacdc02993ab3f1f58e3eb76cd11418cf5e5323ca6132926f35a0bc59d6d88219866e4b246d3c4a4d47cc60fdae0614cad4071c8055c42ed"))),
            ValueReference(T(false,-177,UInt!256.fromHexString("ca5547d5408492cb24034b668fda7a54a82657222d4b3a8d4ae75504bd912cd6")),W(false,-442,UInt!448.fromHexString("da17ed1302b90053bdb6d3626c3d4151016ecaf28bf26a9b57ce929fd8c9f0f247ac8b32e8d1437ca0c3c318d6f166c92877ec8adde0939c"))),
            ValueReference(T(false,-840,UInt!256.fromHexString("dd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a07385749b34376123356")),W(true,-439,UInt!448.fromHexString("ca78cf1250c1e5a6a67cef3e40aaa6fb28c8155fc47320653ee695d47f183d174640f7086463f7db963f703dcf96c21bda82950dc4cd1ee8"))),
            ValueReference(T(false,-1650,UInt!256.fromHexString("ebdb260951f5a219d41440275124cbcfcbdd58ef250485f953c67bc04d11226c")),W(true,-438,UInt!448.fromHexString("f1951064592885c1e51290afde9f63a890d48aa9640f072e347d56beee8624df0284394bc03237f33fcbfe9643d249b8d5d713ee3dc10a78"))),
            ValueReference(T(false,1266,UInt!256.fromHexString("f6aefe360e542751a3681ab77a408745e173d61ec719d860bd429a593ec4002e")),W(false,-437,UInt!448.fromHexString("83dddaa0526c0e6fcb5b963376c746b57a0b78690ee0a49d3c8d4750995eb144884fbfa16399d0ff0a5748752d00924b93a46e3a4b77eb8e"))),
            ValueReference(T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(true,-385,UInt!448.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f17cd89bf0917f754cae4a9121f73a0ea65c288b79fcd01ff6a"))),
            ValueReference(T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(true,-703,UInt!448.fromHexString("8000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000"))),
            ValueReference(T(false,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(false,-385,UInt!448.fromHexString("b17217f7d1cf7b0be84b435c5ba86793e0cba4ce48ba39f26cc657a6f2b47c89d9b2cfc995341bb6279fbb556aa4995ac17f58de8985d6db"))),
        ];
        foreach (r; values)
            assert(L.logPositive(r.value) == r.logarithm);
        struct CountReference { UInt!128 count; W logarithm; }
        enum counts = [
            CountReference(UInt!128.fromHexString("1"),W(0)),
            CountReference(UInt!128.fromHexString("2"),W(false,-448,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144"))),
            CountReference(UInt!128.fromHexString("3"),W(false,-447,UInt!448.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6554bc16906634c282838fb9981cd476ab12e0a838f53de51"))),
            CountReference(UInt!128.fromHexString("11"),W(false,-446,UInt!448.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e40152cbb658d3099f9b71bb1ac68d24393aae13d3b2272ac"))),
            CountReference(UInt!128.fromHexString("ffffffff"),W(false,-443,UInt!448.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49288c9362412560338e7c2168310e596c1808302af733d274"))),
            CountReference(UInt!128.fromHexString("100000001"),W(false,-443,UInt!448.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b63fa39ae33f0074e52706a404a20a198717cbb92a6f1a2d67b"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffff"),W(false,-442,UInt!448.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d6916320cb18965742ff7330d928d7f8eeca61e168f46b5477"))),
            CountReference(UInt!128.fromHexString("10000000000000000"),W(false,-442,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144"))),
            CountReference(UInt!128.fromHexString("10000000000000001"),W(false,-442,UInt!448.fromHexString("b17217f7d1cf79abcde3b39803f2f6af3ef343267298b62d8b626cb0e1004f813c0dcb75c34101edabb7751d6d1c3d330ea625ad38af98bc"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"),W(false,-441,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af3ef343267298b62d8a0d175b8baafa2be6b876206debac98559552fb4afa1b10ec84038b168d7699"))),
        ];
        foreach (r; counts)
            assert(L.logCount(r.count) == r.logarithm);
    }
    {
        alias L = ProbabilityLog!512;
        alias W = Fp!704;
        alias T = Fp!512;
        struct ValueReference { T value; W logarithm; }
        enum values = [
            ValueReference(T(false,long.min,UInt!512.fromHexString("1")),W(true,-641,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b"))),
            ValueReference(T(false,-1,UInt!512.fromHexString("1")),W(true,-704,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b"))),
            ValueReference(T(false,0,UInt!512.fromHexString("1")),W(0)),
            ValueReference(T(false,1,UInt!512.fromHexString("1")),W(false,-704,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b"))),
            ValueReference(T(false,long.max-1,UInt!512.fromHexString("1")),W(false,-641,UInt!704.fromHexString("b17217f7d1cf79a9041b53b8bcb51000196474c662ccdb7086400ac1c1482175bf8418b23f3fc3e8b6b37a79934b68af96d96248954fb500729c82520c893314a139dd71dca408302070b1b8a850956a4ebb337eeb457e40"))),
            ValueReference(T(false,-2,UInt!512.fromHexString("3")),W(true,-705,UInt!704.fromHexString("934b1089a6dc93c1df5bb3b60554e15187a486e65aa1bcd5ad047f998c197d9649b2d2dd9e2181c0b5715d8724b34e98f0028ec8c7910bccfeed3ac2255d53cbecb9fe944c4201c9db708b20cccdf764a99410c753d57f1e"))),
            ValueReference(T(false,-1,UInt!512.fromHexString("3")),W(false,-705,UInt!704.fromHexString("cf991f65fcc25f95b46bb37a02910c0cfa41ff668a8faf856715af1d8b3c76c185be19633db5d76ff5b9486f7140e788ea5acda2badf36bb4fc13b8ffd761c7e90739597f853cfc05a272901962a7998ca4126c00e974158"))),
            ValueReference(T(false,1993,UInt!512.fromHexString("6019d4c9bca9baa7c744a2933a1555c8e7bd79c88f52880d3ed13931b2a6d75b7db9e58d172de8bb4bc511f936209ad644f72dec03aa3b347eb9352f742ae82f")),W(false,-693,UInt!704.fromHexString("d8eb533811d1f128b3f2b13f84b7ca83aa94219927926ac9a5815b542e50fd90af1e9f58eccdcb38dde971ee1d7b6e21dd8b9b20d454e2ca879294747125cb48b1288a8d1e6c1cdaf75e45385790b8d68617e3eab2d5df9d"))),
            ValueReference(T(false,729,UInt!512.fromHexString("6648085078841fcf8fa585628a4034db1cbce41552ed21856e953d68c1745652c0b154d7208854e5511df6adb29e329a89a5803a92d6f8536ccf8fa745a6d89c")),W(false,-694,UInt!704.fromHexString("d6d1ce0c958b6b477b0025e4d73b55839ae2c22699af42445d3349682bf52e616f07506e54feea52a480e2cebebc487a7f41fba6d8c7fb02108b68e45875e2e1da014df839beb1f825acb684fadf0319385d7ae050a9bd2f"))),
            ValueReference(T(false,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),W(false,-1215,UInt!704.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff000000000000000000000000000000000000000000000000"))),
            ValueReference(T(false,-321,UInt!512.fromHexString("93119cc737ae28717bf22938f3b95a8e8627eab8a5f6548d2705af26dce0d4ff43414524547b867fccd2a83783133ebd11a0703c8437994dc1ce825b007434e5")),W(false,-696,UInt!704.fromHexString("83d63ac9c77456756201b0c05424bc86bfb3955eec836de46319994db042f2fe09e32a23ae49bf6afadf4f55e3778a286c3794b56578740e3276b57750686fe53f8340efc25bbf3f9c8fec3f7787a7ce5db86c409a4df08f"))),
            ValueReference(T(false,138,UInt!512.fromHexString("a888b22c82a9fadd99affcd8dabadbc9ebb7d2166152fb62c82a1285bd2cca62f63049385c395caaee199ad932af547a378193a53a3adb6e6e9ef6459db23d1d")),W(false,-695,UInt!704.fromHexString("e1105616eaeed6707140bf5f29ffb502dcfb8fe341de00268497e439ee7818d458bdbe8d4532f20a11de0db5dcd000355c605f94b4377c055741b63d201ef31bbe0b0eddb753075b9e61972f241a2d9e19f4054091d0a652"))),
            ValueReference(T(false,1896,UInt!512.fromHexString("b1d34f1da38704df6a81ab32ec14d47dcea9225e82db0e15db2053a9600032d886f83afa2f18c4915392a9225ec0583c5975e91edfbe098b6eb1484692eb54ee")),W(false,-693,UInt!704.fromHexString("d0977d385a28f7546ce136027b6e3e8e0af05f2b690508d0db4e7c44fa4ec871e1aad5a3f094d5b986734cd5056cbeb1b5da99ca0972340fea0e66d4894afe18de126292464571c3f83e004637b199193436f6f29de51fce"))),
            ValueReference(T(false,614,UInt!512.fromHexString("b451e2e4893dc834281b2d9cca7c45bcb58572d2eacbdb9dee130f914c70d4244e341dc44d678d8d31a99d29a4deb9f77eeaf0108532b7c299caf5024d473a36")),W(false,-694,UInt!704.fromHexString("c30887a99e4d6cb96bc605d6da573b8c1841f65f0506b7b03a1b3bcad990ede56fc2afa6af11375783c6fa4fc6d429cc6b8ea1f4ee764db1026e1736e42071cde83b62d11d90b033fa4e2c20b7ba14c8d6b0087b655af865"))),
            ValueReference(T(false,63,UInt!512.fromHexString("c70c20aa26faba03a8fbc95c69277eab4b4887c55b5f7c2c8e9ef507ee9e9ef93e81c6e63b9b7f76456b48e086c1ef52cf31059e2b9d20f27115d4ca13ee9a63")),W(false,-695,UInt!704.fromHexString("c7276c5adf5cd774ebe677fc369604d40c57dfb98d409a1c7a63d294405f21b4336a9246365cdbfe9731eff178d7b510235767ed0d34c79b52e88d6af286a6be32d598b05aedff4415c1292f934623ff5e2436b8b9bffdac"))),
            ValueReference(T(false,-1680,UInt!512.fromHexString("d629eaa082f2af6a7d1fac564ea8828abd09c99877dda8e84a198cac54e583e60c46d9672c2470ab766490e3445121cbf8232d586cc02d7d2f1ade425dc2c990")),W(true,-694,UInt!704.fromHexString("ca718ed802598c6bc16d679821901bfbe8a954735067360802a05a2fbe17bbd15241a03b6941f244b20289f146bc6480e9bdb548c941e2f3eb460f04497498a612d01a51cc37cc690574f553ac646a3007bcd2e9698760c7"))),
            ValueReference(T(false,long.min,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(true,-641,UInt!704.fromHexString("b17217f7d1cf76e60183d450c60c4787b224e316a6bdf929bd007d9128d24403b35b07f1c202fcf973bcd1439c47b9ba97e2c109d8cbdd8f6c9e642430e4a787e1aa25d07b6a53ccbfa3818805d924138a8256fdc0546553"))),
            ValueReference(T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(true,-1215,UInt!704.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000"))),
            ValueReference(T(false,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),W(false,-641,UInt!704.fromHexString("b17217f7d1cf7c6ecc7b32fffa9bbf27a832d4d62ea79874534ca48c2420d79df3e186e0eb287387988bfc3141fdca05ec254f747dbbf8b52d555956ed0e43b1fe2681b783849d287b990a41d3f3a9d57e237844dc277928"))),
        ];
        foreach (r; values)
            assert(L.logPositive(r.value) == r.logarithm);
        struct CountReference { UInt!128 count; W logarithm; }
        enum counts = [
            CountReference(UInt!128.fromHexString("1"),W(0)),
            CountReference(UInt!128.fromHexString("2"),W(false,-704,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b"))),
            CountReference(UInt!128.fromHexString("3"),W(false,-703,UInt!704.fromHexString("8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6554bc16906634c282838fb9981cd476ab12e0a838f53de50e79bec788812633243684a710f3a6852a3efb748fe48baa58f859791dc410074"))),
            CountReference(UInt!128.fromHexString("11"),W(false,-702,UInt!704.fromHexString("b5535e0fd3fbce77f97ce1c5e15954d575c516735005ff93266e5f1e3168a03e40152cbb658d3099f9b71bb1ac68d24393aae13d3b2272abce48110b6f63672ad8d6486d7a189550530930223a6d7014ed23fd3b0e2ae798"))),
            CountReference(UInt!128.fromHexString("ffffffff"),W(false,-699,UInt!704.fromHexString("b17217f7c9cf79abc5e3b39801484c049448987bc65471e944737dc1f0ecce49288c9362412560338e7c2168310e596c1808302af733d27396d338ba5e531e016d4e1aa27068ce0f9556a3413036b5a8beb91aca9cb98b51"))),
            CountReference(UInt!128.fromHexString("100000001"),W(false,-699,UInt!704.fromHexString("b17217f7d9cf79abc5e3b398069da159e99dedd11edcfa71ccfc064a7bbe7b63fa39ae33f0074e52706a404a20a198717cbb92a6f1a2d67b1d1d11b506546f8ae2fcbb5df16ed797e1319fea2350c4454358407f01d6f762"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffff"),W(false,-698,UInt!704.fromHexString("b17217f7d1cf79abc5e3b39803f2f6af3ef343267298b62d88b7c2063655a4d6916320cb18965742ff7330d928d7f8eeca61e168f46b547759f82537b253c6c628256b0030ebd2d3bb442195a9c3bcf70108ada4cf484159"))),
            CountReference(UInt!128.fromHexString("10000000000000000"),W(false,-698,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b"))),
            CountReference(UInt!128.fromHexString("10000000000000001"),W(false,-698,UInt!704.fromHexString("b17217f7d1cf79abcde3b39803f2f6af3ef343267298b62d8b626cb0e1004f813c0dcb75c34101edabb7751d6d1c3d330ea625ad38af98bb9f60fbc51b2a542efeb2d3d6be54a96124fe3d3763df5eb11caa67c071025cfb"))),
            CountReference(UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"),W(false,-697,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af3ef343267298b62d8a0d175b8baafa2be6b876206debac98559552fb4afa1b10ec84038b168d76997cac907e66bf0d7a936c1f6b77a03e1a70212f6686d18dd40ed98ab2a0254f2a"))),
        ];
        foreach (r; counts)
            assert(L.logCount(r.count) == r.logarithm);
    }
}
// END positive logarithm references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    foreach (bits; AliasSeq!(64u,128u,256u,512u)) {
        alias L = ProbabilityLog!bits;
        alias T = Fp!bits;
        alias W = Fp!(bits + 192);
        assert(L.logPositive(T(1)) == W(0));
        assert(!L.logPositive(T(1)).sign);
        foreach (value; [T(0.25),T(0.75),T(1.5),T(2)]) {
            T alternate = value;
            alternate.coefficient >>= 1;
            alternate.exponent += 1;
            assert(L.logPositive(value) == L.logPositive(alternate));
        }
        foreach (p; [T(0.25),T(0.75)]) {
            auto logarithm = L.logPositive(p);
            logarithm.sign = false;
            assert(cast(T)logarithm == L.probability(p).finish!bits);
        }
        static assert(is(typeof(L.logPositive(T(2))) == W));
        static assert(is(typeof(L.logCount(UInt!128(2))) == W));
        debug {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            foreach (bad; [T(0),T(-1),T(double.infinity),T(double.nan)])
                assertThrown!AssertError(L.logPositive(bad));
            assertThrown!AssertError(L.logCount(UInt!128(0)));
        }
    }
}
