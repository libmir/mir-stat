/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_probability_log;

import mir.bignum.fp: Fp;
import mir.bignum.fixed: UInt;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;

// Positive magnitudes of probability logarithms. The caller handles endpoints.
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
        auto product = a.toSize!(2 * fractionBits + 128);
        product *= b;
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
