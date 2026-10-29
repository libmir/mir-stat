/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_log_factorial;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability_log: ProbabilityLog;

// Internal count logarithms for future centered log-mass calculations.
// Return 192 guard bits; another 128 fractional bits absorb summation,
// cancellation in the small-count correction, and fixed-point truncation.
// This is guarded evaluation, not a universal correct-rounding guarantee.
package(mir.stat) struct CountLog(uint bits)
{
    alias W = Fp!(bits + 192);
    enum fractionBits = bits + 320;
    alias I = UInt!(fractionBits + 192);
    // n^129 for a 64-bit n, coefficient denominators, and shifted numerators.
    alias Wide = UInt!(fractionBits + 8448);
    alias Logs = ProbabilityLog!(bits + 128);

    @safe pure nothrow @nogc:

    private static I fixed(uint size)(Fp!size value)
    {
        if (!value.coefficient) return I(0);
        auto result = value.coefficient.toSize!(fractionBits + 192);
        const shift = value.exponent + fractionBits;
        if (shift >= 0) result <<= cast(size_t)shift;
        else result >>= cast(size_t)-shift;
        return result;
    }

    private static W finish(I value)
    {
        if (!value) return W(0);
        auto result = W(value);
        result.exponent -= fractionBits;
        return result;
    }

    // Machin's identity pi=16*atan(1/5)-4*atan(1/239), with 64 extra
    // fractional bits. Integer series divisions never use hardware floats.
    private static I halfLogTwoPi()
    {
        enum piBits = fractionBits + 64;
        alias P = UInt!(piBits + 64);
        P atanInverse(uint denominator) {
            P power = P(1) << piBits;
            power /= UInt!64(denominator);
            P sum = power;
            bool negative = true;
            for (uint odd = 3; ; odd += 2) {
                power /= UInt!64(denominator * denominator);
                auto term = power;
                term /= UInt!64(odd);
                if (!term) break;
                if (negative) sum -= term;
                else sum += term;
                negative = !negative;
            }
            return sum;
        }
        auto pi = (atanInverse(5) << 4) - (atanInverse(239) << 2);
        auto twicePi = Fp!fractionBits(pi);
        twicePi.exponent += 1 - cast(long)piBits;
        return fixed(ProbabilityLog!fractionBits.logPositive(twicePi)) >> 1;
    }

    private static I coefficientTerm(uint index, Wide power)
    {
        auto numerator = coefficients[index].numerator.toSize!(fractionBits + 8448);
        numerator <<= fractionBits;
        power *= coefficients[index].denominator;
        return (numerator / power).toSize!(fractionBits + 192);
    }

    // For positive n, the remainder has the sign of the first omitted
    // Stirling term and magnitude bounded by that term (DLMF 5.11(ii)).
    // Use 64 terms only when the exact 65th term is below one grid unit.
    // The crossover therefore grows with the requested precision.
    static bool usesStirling(ulong n)
        in (n != 0, "count must be positive")
    {
        Wide power = Wide(n);
        foreach (i; 0 .. 128) power *= UInt!64(n);
        auto numerator = coefficients[64].numerator.toSize!(fractionBits + 8448);
        numerator <<= fractionBits;
        power *= coefficients[64].denominator;
        return numerator < power;
    }

    private static I correctionLarge(ulong n)
    {
        Wide power = Wide(n);
        // n is a 64-bit count, so its square needs only 128 bits. Keeping
        // the multiplier narrow avoids a huge unrolled Wide-by-Wide product.
        UInt!128 square = UInt!128(n);
        square *= UInt!64(n);
        I sum;
        foreach (i; 0 .. 64) {
            const term = coefficientTerm(i,power);
            if (!term) break;
            if (i & 1) sum -= term;
            else sum += term;
            power *= square;
        }
        return sum;
    }

    private static I factorialSmall(ulong n)
    {
        I sum;
        if (n < 2) return sum;
        for (ulong i = 2; ; ++i) {
            sum += fixed(Logs.logCount(UInt!128(i)));
            if (i == n) break;
        }
        return sum;
    }

    private static I leading(ulong n)
    {
        auto result = fixed(Logs.logCount(UInt!128(n)));
        auto twiceCountPlusOne = UInt!128(n);
        twiceCountPlusOne <<= 1;
        twiceCountPlusOne += 1;
        result *= twiceCountPlusOne;
        return result >> 1;
    }

    // log(n!), including zero and one. Small counts cost O(n); the large
    // path uses a bounded number of integer powers and logarithm series.
    static W factorial(ulong n)
    {
        if (n < 2) return W(0);
        if (!usesStirling(n)) return finish(factorialSmall(n));
        auto result = leading(n);
        result -= I(n) << fractionBits;
        result += halfLogTwoPi();
        result += correctionLarge(n);
        return finish(result);
    }

    // log(n!) - (n+1/2)*log(n) + n - log(2*pi)/2, n>=1.
    // Expose the correction separately so callers can combine the leading
    // terms analytically instead of subtracting large log-factorials.
    static W correction(ulong n)
        in (n != 0, "count must be positive")
    {
        if (usesStirling(n)) return finish(correctionLarge(n));
        auto result = factorialSmall(n);
        result += I(n) << fractionBits;
        result -= leading(n);
        result -= halfLogTwoPi();
        return finish(result);
    }
}

// BEGIN Stirling coefficients
private struct StirlingCoefficient { UInt!512 numerator; UInt!512 denominator; }
private immutable StirlingCoefficient[65] coefficients = [
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("c")),
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("168")),
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("4ec")),
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("690")),
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("4a4")),
    StirlingCoefficient(UInt!512.fromHexString("2b3"),UInt!512.fromHexString("57fa8")),
    StirlingCoefficient(UInt!512.fromHexString("1"),UInt!512.fromHexString("9c")),
    StirlingCoefficient(UInt!512.fromHexString("e21"),UInt!512.fromHexString("1de20")),
    StirlingCoefficient(UInt!512.fromHexString("ab5b"),UInt!512.fromHexString("3b9dc")),
    StirlingCoefficient(UInt!512.fromHexString("2aa13"),UInt!512.fromHexString("1e9d8")),
    StirlingCoefficient(UInt!512.fromHexString("12f73"),UInt!512.fromHexString("16a4")),
    StirlingCoefficient(UInt!512.fromHexString("e16a13b"),UInt!512.fromHexString("16fe90")),
    StirlingCoefficient(UInt!512.fromHexString("a0a0b"),UInt!512.fromHexString("12c")),
    StirlingCoefficient(UInt!512.fromHexString("ca39b773"),UInt!512.fromHexString("16f08")),
    StirlingCoefficient(UInt!512.fromHexString("19134d4ece1"),UInt!512.fromHexString("26067c")),
    StirlingCoefficient(UInt!512.fromHexString("702f7579541"),UInt!512.fromHexString("7b840")),
    StirlingCoefficient(UInt!512.fromHexString("234dc64fcf"),UInt!512.fromHexString("18c")),
    StirlingCoefficient(UInt!512.fromHexString("16d32a00d9fb1c9fd"),UInt!512.fromHexString("90227d48")),
    StirlingCoefficient(UInt!512.fromHexString("8c40dc8c82ed"),UInt!512.fromHexString("1bc")),
    StirlingCoefficient(UInt!512.fromHexString("e274025326e284f03"),UInt!512.fromHexString("1421070")),
    StirlingCoefficient(UInt!512.fromHexString("526797b1522cd66903"),UInt!512.fromHexString("2f742c")),
    StirlingCoefficient(UInt!512.fromHexString("892aed98ac0799e88d"),UInt!512.fromHexString("1cf98")),
    StirlingCoefficient(UInt!512.fromHexString("57dcfe957e13e3a8c5f"),UInt!512.fromHexString("6324")),
    StirlingCoefficient(UInt!512.fromHexString("121ffd5df3642920c21b3e7b"),UInt!512.fromHexString("63d9c20")),
    StirlingCoefficient(UInt!512.fromHexString("10614c31e1f3009f996f9d"),UInt!512.fromHexString("1944")),
    StirlingCoefficient(UInt!512.fromHexString("c721989aefa9abed27d1a4e1"),UInt!512.fromHexString("4f308")),
    StirlingCoefficient(UInt!512.fromHexString("16fec9cc680c7eda8fe2a855f53"),UInt!512.fromHexString("22d964")),
    StirlingCoefficient(UInt!512.fromHexString("11769ea96ec5fe7e0a7c3d7e975b"),UInt!512.fromHexString("5d750")),
    StirlingCoefficient(UInt!512.fromHexString("8fa21b8005788631fc84c1a6eff9"),UInt!512.fromHexString("9da4")),
    StirlingCoefficient(UInt!512.fromHexString("df33fcd77896a343f435047daa8a7e64393"),UInt!512.fromHexString("2ece0670c8")),
    StirlingCoefficient(UInt!512.fromHexString("4c6b68c0f2174451554e0b6b0765f9"),UInt!512.fromHexString("2dc")),
    StirlingCoefficient(UInt!512.fromHexString("139cf399b169aabfc7c7f11711cae210b81"),UInt!512.fromHexString("1f6080")),
    StirlingCoefficient(UInt!512.fromHexString("600c897ae87f32388b90c08b850bcd5501037"),UInt!512.fromHexString("18127ec")),
    StirlingCoefficient(UInt!512.fromHexString("353142a5baa3d5cf24827433423b27305de9"),UInt!512.fromHexString("1f68")),
    StirlingCoefficient(UInt!512.fromHexString("788acde1db061c9795c727834afda4c18446a19"),UInt!512.fromHexString("9de0c")),
    StirlingCoefficient(UInt!512.fromHexString("3cd8c09a3e273f12515c307e590205d17be20c9315664d"),UInt!512.fromHexString("a6c0964fd0")),
    StirlingCoefficient(UInt!512.fromHexString("a1ae781722adefa4d8a48c2b315fc58b4e062c43"),UInt!512.fromHexString("36c")),
    StirlingCoefficient(UInt!512.fromHexString("377e0f6144bfdd4f2d1a0e3860824e4d786ac5793f3"),UInt!512.fromHexString("2328")),
    StirlingCoefficient(UInt!512.fromHexString("14d2b495fb9f259891decb2150b3966308fc0493397b85b"),UInt!512.fromHexString("1763f4")),
    StirlingCoefficient(UInt!512.fromHexString("2dd6cd6d086747768a80e0046e172e59e07b6fe7c5d41f609a3"),UInt!512.fromHexString("56a523e0")),
    StirlingCoefficient(UInt!512.fromHexString("68424fdf52e757df209701152d7730623b2dab764ca57a485"),UInt!512.fromHexString("13b24")),
    StirlingCoefficient(UInt!512.fromHexString("133978dc1f2002a1046ff4c69f3dac68e63aa1cdfe6d75ca93e13f3d"),UInt!512.fromHexString("586b50f28")),
    StirlingCoefficient(UInt!512.fromHexString("98fdbc0acf778618effe5752cf6d3300859155a0eb11cd3bba5"),UInt!512.fromHexString("3fc")),
    StirlingCoefficient(UInt!512.fromHexString("46c113339bd77b0868e16e77efc1efbfbd740e6961b8fa6c978ac10fd"),UInt!512.fromHexString("28c2e70")),
    StirlingCoefficient(UInt!512.fromHexString("222abba7606e843daedd9e8f60ea2b43be619d42b4f28b521ac21384af7b"),UInt!512.fromHexString("19fbcf6c")),
    StirlingCoefficient(UInt!512.fromHexString("82963a88883070167d7a2a7352d614fd63ea0089494abb06aefe241ac11"),UInt!512.fromHexString("7d4d8")),
    StirlingCoefficient(UInt!512.fromHexString("3c3751c5a3badc91ca8b1596c2c40d60451576f21131fce82d8a46603c9"),UInt!512.fromHexString("45c")),
    StirlingCoefficient(UInt!512.fromHexString("7236aca47d00be6048d20d18760f53b1c625dd07a328ba43b65f4ff3683fcbe99bb"),UInt!512.fromHexString("98f230440")),
    StirlingCoefficient(UInt!512.fromHexString("31061c4b01669c905e9dc0b1ac099c9febdfa74d9763678823ae19ce080816f"),UInt!512.fromHexString("48c")),
    StirlingCoefficient(UInt!512.fromHexString("c7744997e8aeae3d14e99e99995eda6ebf5a3b5944a5e7e797acd2f4d0e7b433468df"),UInt!512.fromHexString("13aae598")),
    StirlingCoefficient(UInt!512.fromHexString("18d619a0eb1def0deed030fd0a23e473d34d7d6ccb235b95b745762c741934a968e457"),UInt!512.fromHexString("280074")),
    StirlingCoefficient(UInt!512.fromHexString("ca706984484b43d23960aa100d6a95b2967528cc493382eb87c9810bf1093b71ce44f09"),UInt!512.fromHexString("13fdd0")),
    StirlingCoefficient(UInt!512.fromHexString("16146e374541630d56f7aa75d31478a1773afb059830cc31cee9375a4c6215dbdbec40171"),UInt!512.fromHexString("20ea4")),
    StirlingCoefficient(UInt!512.fromHexString("19fcdc8928d1dc29249daa85deec289d3503db7da1d455166ccada531f98355e8d60ca7cc5e0b9bc9"),UInt!512.fromHexString("232d96cb638")),
    StirlingCoefficient(UInt!512.fromHexString("2eeb2a83cbd559314495bc18cbe86d751c551bb3d9436fe6783438db0a8abd06f17fa7c0fa9e6f"),UInt!512.fromHexString("378b64")),
    StirlingCoefficient(UInt!512.fromHexString("b14b313082862e1183de1e4ea6f9d14cb4ac2afe9ea4f694493febd7cac54d0aa256e881a27703492b"),UInt!512.fromHexString("b0eac3a0")),
    StirlingCoefficient(UInt!512.fromHexString("891efa27a3bf84b4f155963866aadd769d8ebeb49ffbf188ce23479950e4a85de771c153708af763"),UInt!512.fromHexString("6f3c")),
    StirlingCoefficient(UInt!512.fromHexString("138559617dcd2e5bfc061e5e293e6b99f7aa10a33ed83ea1c1c8cdcd90eff7ed73d9b3f205ba580fa39f"),UInt!512.fromHexString("c6c78")),
    StirlingCoefficient(UInt!512.fromHexString("b5fde6652ac68b1ab30264aed1399d34ef256a31adfa1cafb3ecc24e1bc7a20b7ef588546a4754f47f5"),UInt!512.fromHexString("57c")),
    StirlingCoefficient(UInt!512.fromHexString("55abae5049fb293e33fa9dca3eb62cc93a14239e876528f6f1d67e2111556aa0e798f1f57ecf671097f8923a1ffa73"),UInt!512.fromHexString("1e3d091c2f50")),
    StirlingCoefficient(UInt!512.fromHexString("16b4912ae29c73c196536cd653c77018c5d34e10ddd232871c738adecc7de64b046c71c7c8ea7e79207059fb"),UInt!512.fromHexString("5ac")),
    StirlingCoefficient(UInt!512.fromHexString("151207496e884eea37dcc31ec5b024c0d6bfc78c1efd32996643a13cf28d84d8ccece8d4f446f5e6bd552f80677"),UInt!512.fromHexString("39a8")),
    StirlingCoefficient(UInt!512.fromHexString("8d0434cfd60bcfcbcaccf0b6455b2718a3dfe0d97bae982eaed795175175dfca6d895f3aa893b87fd4cf99f4c9e3ab07bf"),UInt!512.fromHexString("ffb0f12c4")),
    StirlingCoefficient(UInt!512.fromHexString("6cba32f410bf8ad4ebcf8504b386383f8f192e34d8d53fb98984c9c89cfdc0712c0b5ea74bb8e5eab5971f67262899801"),UInt!512.fromHexString("7e8100")),
    StirlingCoefficient(UInt!512.fromHexString("2f0e0698193452dabc1b948981efa60c0fd66fbbfb791f5031296ce6cee931cc9122a70652fa0eea584e22a81d4ca18659f"),UInt!512.fromHexString("22098c")),
];
// END Stirling coefficients
// BEGIN log-factorial references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias L = CountLog!64;
    alias W = Fp!256;
    assert(!L.usesStirling(58));
    assert(L.usesStirling(59));
    assert(L.factorial(0) == W(0));
    struct Reference { ulong n; W factorial; W correction; }
    enum references = [
        Reference(1UL,W(0),W(false,-259,UInt!256.fromHexString("a6038de6d052d65b5a1b7ff5df36fee1827b839765d3a59109cae76612af5517"))),
        Reference(2UL,W(false,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c")),W(false,-260,UInt!256.fromHexString("a954dc91f532449bc6de29ab5fa6dd54ee28bb940d5636dd225b9e370d573610"))),
        Reference(3UL,W(false,-255,UInt!256.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc")),W(false,-261,UInt!256.fromHexString("e2bcd1346c0599d35ce8484658a3d8a2c2018f22743900e22953dfd0593bdde4"))),
        Reference(16UL,W(false,-251,UInt!256.fromHexString("f55ff830f8d421f9f49e965b81d1d762c8a0e9f27f0d5e833a86d6c247f2eeea")),W(false,-263,UInt!256.fromHexString("aaa4fbee889ddff930f71ef8bf198342daaf6ca183a6aded5a17cddf4fa44431"))),
        Reference(32UL,W(false,-249,UInt!256.fromHexString("a31dacdca1041cc1b0f2456e455e9f55134855cc62ee255b578f2661276220ee")),W(false,-264,UInt!256.fromHexString("aaa93eade600183d5ca722e97cf4271a1036bca5dd7e75c77d39e44b5e9487ce"))),
        Reference(58UL,W(false,-248,UInt!256.fromHexString("b474cf83abcbc52b5d5188f08e51483f51b464d1be104fb07a3c857bac2664c1")),W(false,-265,UInt!256.fromHexString("bc51e9c34aded147733179080ab6e3234b4afc1a59b7e5e5656186367f05c35d"))),
        Reference(59UL,W(false,-248,UInt!256.fromHexString("b888a9021d96235906650e743d5dcaa020103ceb654d8c71be30ae98bb022c87")),W(false,-265,UInt!256.fromHexString("b920cfce942e0b8d93921962c8000bcfa4caf2fd4ff64a24b369675f2806ad2b"))),
        Reference(60UL,W(false,-248,UInt!256.fromHexString("bca0cff936e2c62c847c670dd872453c289984254ce17d9a16076c1908e6e16c")),W(false,-265,UInt!256.fromHexString("b60af240ab1806c6746161ae647800b0fccd0aaf740c0bd1b9471a38ee3362e9"))),
        Reference(4294967295UL,W(false,-219,UInt!256.fromHexString("a97217f780706a21fad64e966e51ac0df3bf8b1f9596e8bbb3ac2db386dfb7e8")),W(false,-291,UInt!256.fromHexString("aaaaaaab55555555fa4fa4fae93e93e9c8bc8bc936736736dfe5fe60498c98ca"))),
        Reference(4294967296UL,W(false,-219,UInt!256.fromHexString("a97217f831e28219cca5c84238355fa5f7b281ced68a2be22644e3e110eccf44")),W(false,-291,UInt!256.fromHexString("aaaaaaaaaaaaaaaaa4fa4fa4fa4fa4fa51451451451451451319319319319319"))),
        Reference(1000000000000UL,W(false,-211,UInt!256.fromHexString("c1c4246dd13a42e1d58e9b9ffd5051f4604334285ac4935270d875e7c54ade08")),W(false,-299,UInt!256.fromHexString("bba66561e9c0b6654c54ad6486caca31eb9ac6a00c3dd93b10a93e1a95b25079"))),
        Reference(18446744073709551615UL,W(false,-186,UInt!256.fromHexString("ad7217f7d1cf79ab74d7a5d527a3106e2ea9b0efcb04f4abce278d5bdc010a98")),W(false,-323,UInt!256.fromHexString("aaaaaaaaaaaaaaab5555555555555555fa4fa4fa4fa4fa503e93e93e93e93e94"))),
    ];
    foreach (r; references) {
        assert(L.factorial(r.n) == r.factorial);
        assert(L.correction(r.n) == r.correction);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias L = CountLog!128;
    alias W = Fp!320;
    assert(!L.usesStirling(82));
    assert(L.usesStirling(83));
    assert(L.factorial(0) == W(0));
    struct Reference { ulong n; W factorial; W correction; }
    enum references = [
        Reference(1UL,W(0),W(false,-323,UInt!320.fromHexString("a6038de6d052d65b5a1b7ff5df36fee1827b839765d3a59109cae76612af5516f6831400ff8bca03"))),
        Reference(2UL,W(false,-320,UInt!320.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98")),W(false,-324,UInt!320.fromHexString("a954dc91f532449bc6de29ab5fa6dd54ee28bb940d5636dd225b9e370d57361033bb14f7b0ff65bf"))),
        Reference(3UL,W(false,-319,UInt!320.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc4927fc793d592274")),W(false,-325,UInt!320.fromHexString("e2bcd1346c0599d35ce8484658a3d8a2c2018f22743900e22953dfd0593bdde381c2326dbd9521ff"))),
        Reference(16UL,W(false,-315,UInt!320.fromHexString("f55ff830f8d421f9f49e965b81d1d762c8a0e9f27f0d5e833a86d6c247f2eee9ba007059bc5dfba7")),W(false,-327,UInt!320.fromHexString("aaa4fbee889ddff930f71ef8bf198342daaf6ca183a6aded5a17cddf4fa44430a7fcadaa77377224"))),
        Reference(32UL,W(false,-313,UInt!320.fromHexString("a31dacdca1041cc1b0f2456e455e9f55134855cc62ee255b578f2661276220ed870b971579be2ce6")),W(false,-328,UInt!320.fromHexString("aaa93eade600183d5ca722e97cf4271a1036bca5dd7e75c77d39e44b5e9487ce04f1a738eeddc4ed"))),
        Reference(82UL,W(false,-311,UInt!320.fromHexString("8d3cb59f6f0d6014da3e4022710745f3c6a727a808baeebec3f8e523310531b6dda8215c8398da5a")),W(false,-329,UInt!320.fromHexString("8533dd0d19a879242f643c77972552579bc71dc3efec02dbe0a50abe825bc994bd65b91c08254827"))),
        Reference(83UL,W(false,-311,UInt!320.fromHexString("8f7252311b69f0489c83da7d69f337491b5a1c495b1772506805fe3a31e54bf35ee9f112698350d7")),W(false,-329,UInt!320.fromHexString("83990697b25092168e24e16fcd8fa44de58cdd839626fb43af9f7cb7a0b21d8865bcae668e9e3a67"))),
        Reference(84UL,W(false,-311,UInt!320.fromHexString("91a977325a36aad1a71b1543140cd5ee48a03495208d496d8b0609ee615316e96e45e608a43b37e8")),W(false,-329,UInt!320.fromHexString("8207f840770c379742f1b7d7af6314555d2ab27896275a4b268cbfdb0e9ab76bcbce0a5daef1c2f1"))),
        Reference(4294967295UL,W(false,-283,UInt!320.fromHexString("a97217f780706a21fad64e966e51ac0df3bf8b1f9596e8bbb3ac2db386dfb7e800e362e6f6322007")),W(false,-355,UInt!320.fromHexString("aaaaaaab55555555fa4fa4fae93e93e9c8bc8bc936736736dfe5fe60498c98c9e957cbcaa3d1ff11"))),
        Reference(4294967296UL,W(false,-283,UInt!320.fromHexString("a97217f831e28219cca5c84238355fa5f7b281ced68a2be22644e3e110eccf438c8e5d12ddea9628")),W(false,-355,UInt!320.fromHexString("aaaaaaaaaaaaaaaaa4fa4fa4fa4fa4fa51451451451451451319319319319319334c6b1444a923c9"))),
        Reference(1000000000000UL,W(false,-275,UInt!320.fromHexString("c1c4246dd13a42e1d58e9b9ffd5051f4604334285ac4935270d875e7c54ade0852e79db6af2eb8d2")),W(false,-363,UInt!320.fromHexString("bba66561e9c0b6654c54ad6486caca31eb9ac6a00c3dd93b10a93e1a95b2507885029943f5fec368"))),
        Reference(18446744073709551615UL,W(false,-250,UInt!320.fromHexString("ad7217f7d1cf79ab74d7a5d527a3106e2ea9b0efcb04f4abce278d5bdc010a97af7f8226c789ffaa")),W(false,-387,UInt!320.fromHexString("aaaaaaaaaaaaaaab5555555555555555fa4fa4fa4fa4fa503e93e93e93e93e947367367367367368"))),
    ];
    foreach (r; references) {
        assert(L.factorial(r.n) == r.factorial);
        assert(L.correction(r.n) == r.correction);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias L = CountLog!256;
    alias W = Fp!448;
    assert(!L.usesStirling(163));
    assert(L.usesStirling(164));
    assert(L.factorial(0) == W(0));
    struct Reference { ulong n; W factorial; W correction; }
    enum references = [
        Reference(1UL,W(0),W(false,-451,UInt!448.fromHexString("a6038de6d052d65b5a1b7ff5df36fee1827b839765d3a59109cae76612af5516f6831400ff8bca0360aa037d1d54b2227d1aa54725992d22"))),
        Reference(2UL,W(false,-448,UInt!448.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c1382144")),W(false,-452,UInt!448.fromHexString("a954dc91f532449bc6de29ab5fa6dd54ee28bb940d5636dd225b9e370d57361033bb14f7b0ff65bebb543f6b3336daaebdd4f5842def3bdf"))),
        Reference(3UL,W(false,-447,UInt!448.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc4927fc793d5922745303a517274a54f327c5619e6fefeef3")),W(false,-453,UInt!448.fromHexString("e2bcd1346c0599d35ce8484658a3d8a2c2018f22743900e22953dfd0593bdde381c2326dbd9521ff11b52d6cb44987f8dd79c99b28fdea7a"))),
        Reference(16UL,W(false,-443,UInt!448.fromHexString("f55ff830f8d421f9f49e965b81d1d762c8a0e9f27f0d5e833a86d6c247f2eee9ba007059bc5dfba74f6911361eafaf88c49670c9d644f38d")),W(false,-455,UInt!448.fromHexString("aaa4fbee889ddff930f71ef8bf198342daaf6ca183a6aded5a17cddf4fa44430a7fcadaa773772245c01351292c67fde344203ee6c81e0e6"))),
        Reference(32UL,W(false,-441,UInt!448.fromHexString("a31dacdca1041cc1b0f2456e455e9f55134855cc62ee255b578f2661276220ed870b971579be2ce66da89562bcb12e3499d536344190ce2f")),W(false,-456,UInt!448.fromHexString("aaa93eade600183d5ca722e97cf4271a1036bca5dd7e75c77d39e44b5e9487ce04f1a738eeddc4ed16c54ae47f8252ac1cd9aa86f91f5b24"))),
        Reference(163UL,W(false,-438,UInt!448.fromHexString("a7afd8cd98d9f560743b94d2940e282dbba32fb52ed8dde605d27a79c5a9c03227ca810af2c8628dd725336dbf34b650c31aef2977dd0916")),W(false,-458,UInt!448.fromHexString("8605312f7014e9fc72fb97535bbf97d7ac9addbc8191d8190a0e47ab70f2af18f89ccf5c2829fdbedbec163b4b951b5ee8a4b3a25e764aab"))),
        Reference(164UL,W(false,-438,UInt!448.fromHexString("a8f63d03c1383f8aa7878682f9df7ac272e4ef7bc78df4453449a93fdb38711f5f5a75e92f6221746a2cbe5b8c3a3102862b61db14c001e2")),W(false,-458,UInt!448.fromHexString("8533fd81a03641589858c8ce3eaa831483ac3ec42b9053c694e3c38562bd4115133de9ef2a93feefddd6a1dc46e3f2d978eef204ff65d2c8"))),
        Reference(165UL,W(false,-438,UInt!448.fromHexString("aa3d04d3479c3ae06f0b50cdf9e2096b36b431643b1267cb24cf57931cbee616713aeb8b277cdb1906a43ae43669b39fb82c2b7e6f062175")),W(false,-458,UInt!448.fromHexString("846552fc182a632da2d77e9c90897432979a9ace75152eb73177b82ddd70b56936de23dd38c2f803dda4c9bada575bdb938bce9e4ead5494"))),
        Reference(4294967295UL,W(false,-411,UInt!448.fromHexString("a97217f780706a21fad64e966e51ac0df3bf8b1f9596e8bbb3ac2db386dfb7e800e362e6f632200736d33d50311370b3931a7aaba6906f86")),W(false,-483,UInt!448.fromHexString("aaaaaaab55555555fa4fa4fae93e93e9c8bc8bc936736736dfe5fe60498c98c9e957cbcaa3d1ff10b6cbe8ddddcfcd79473c35b416c073c9"))),
        Reference(4294967296UL,W(false,-411,UInt!448.fromHexString("a97217f831e28219cca5c84238355fa5f7b281ced68a2be22644e3e110eccf438c8e5d12ddea9627a4bee9e886a8c3aede1495bc93bf1dbc")),W(false,-483,UInt!448.fromHexString("aaaaaaaaaaaaaaaaa4fa4fa4fa4fa4fa51451451451451451319319319319319334c6b1444a923c8862eb07e6cc0024dc5612f4942278751"))),
        Reference(1000000000000UL,W(false,-403,UInt!448.fromHexString("c1c4246dd13a42e1d58e9b9ffd5051f4604334285ac4935270d875e7c54ade0852e79db6af2eb8d1dcd740760ac6f11b1d839607f103bc5f")),W(false,-491,UInt!448.fromHexString("bba66561e9c0b6654c54ad6486caca31eb9ac6a00c3dd93b10a93e1a95b2507885029943f5fec3686bbb5f258fa6f83322a3ffad1d24160f"))),
        Reference(18446744073709551615UL,W(false,-378,UInt!448.fromHexString("ad7217f7d1cf79ab74d7a5d527a3106e2ea9b0efcb04f4abce278d5bdc010a97af7f8226c789ffaa5365a8de3b56fc079331a795db2448e1")),W(false,-515,UInt!448.fromHexString("aaaaaaaaaaaaaaab5555555555555555fa4fa4fa4fa4fa503e93e93e93e93e947367367367367367b05b05b05b05b05b722f22f22f22f230"))),
    ];
    foreach (r; references) {
        assert(L.factorial(r.n) == r.factorial);
        assert(L.correction(r.n) == r.correction);
    }
}

version(mir_stat_test_slow_numerics)
@safe pure nothrow @nogc unittest
{
    alias L = CountLog!512;
    alias W = Fp!704;
    assert(!L.usesStirling(646));
    assert(L.usesStirling(647));
    assert(L.factorial(0) == W(0));
    struct Reference { ulong n; W factorial; W correction; }
    enum references = [
        Reference(1UL,W(0),W(false,-707,UInt!704.fromHexString("a6038de6d052d65b5a1b7ff5df36fee1827b839765d3a59109cae76612af5516f6831400ff8bca0360aa037d1d54b2227d1aa54725992d2184c898f446a2906b6cc4789ad5dbc2c67c13cc4e51f74b4b6d59767dad30240f"))),
        Reference(2UL,W(false,-704,UInt!704.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b8253e96ca16224ae8c51acbda11317c387eb9ea9bc3b136603b")),W(false,-708,UInt!704.fromHexString("a954dc91f532449bc6de29ab5fa6dd54ee28bb940d5636dd225b9e370d57361033bb14f7b0ff65bebb543f6b3336daaebdd4f5842def3bdf5963a60eeb5bdd58fb65ff2274b1b312750b270000494ab56cb452a2bd474292"))),
        Reference(3UL,W(false,-703,UInt!704.fromHexString("e5585fd15100119136fea076849739b27f83c300153ca20ee3d28322ee7a17dc4927fc793d5922745303a517274a54f327c5619e6fefeef2fb478a0d10c73f44e2b3af7c205fdcb53155a4519706d6e4ec7ae573b4dc3091")),W(false,-709,UInt!704.fromHexString("e2bcd1346c0599d35ce8484658a3d8a2c2018f22743900e22953dfd0593bdde381c2326dbd9521ff11b52d6cb44987f8dd79c99b28fdea7a3c95fd9e3c4346eb64bc9e861c4af016d3f7e5c089ef958b3f3a974332ec4f63"))),
        Reference(16UL,W(false,-699,UInt!704.fromHexString("f55ff830f8d421f9f49e965b81d1d762c8a0e9f27f0d5e833a86d6c247f2eee9ba007059bc5dfba74f6911361eafaf88c49670c9d644f38d7df828d7e9e72c4b635f5b0210c1a6df906336b9a88c33c340ae7a83ce41211d")),W(false,-711,UInt!704.fromHexString("aaa4fbee889ddff930f71ef8bf198342daaf6ca183a6aded5a17cddf4fa44430a7fcadaa773772245c01351292c67fde344203ee6c81e0e59076c2a43c30efdb51ead04dd028bcf9ad8c280cdf6892ca7f2b6ae6e2ac6c53"))),
        Reference(32UL,W(false,-697,UInt!704.fromHexString("a31dacdca1041cc1b0f2456e455e9f55134855cc62ee255b578f2661276220ed870b971579be2ce66da89562bcb12e3499d536344190ce2ef3d794ac57f691d9749fdb02544bcf3325f4555894307829bc32751cca54330d")),W(false,-712,UInt!704.fromHexString("aaa93eade600183d5ca722e97cf4271a1036bca5dd7e75c77d39e44b5e9487ce04f1a738eeddc4ed16c54ae47f8252ac1cd9aa86f91f5b23845bb8f681db2382cec930bc1650ce474739e9f92121d68379dd7ce2bb75a9f6"))),
        Reference(646UL,W(false,-692,UInt!704.fromHexString("dd24a7b7c17fb7d59f58604bb6e7ae4a99b7fcbd2433cf10f6c977765b617d33e31dd7a3b3157f43903f0aac408490d84e06161ebb508520c4868c9e26e91cea9af8fe10c3c49866520e8b883799459a3d6e555082fd2652")),W(false,-716,UInt!704.fromHexString("8743e4df78bb6ad6da386978d3e439593ce3bf75aa101c294ea9f42eeb6891cfac0c39f570089436c1455d9c1786780371adfc42634a9802c26213d65eb34bde4c4901286b0dcb939c774957c61d83eb07e6a574b3fd9ecf"))),
        Reference(647UL,W(false,-692,UInt!704.fromHexString("dd8c3672be74022ff824fbd9bffb5a59ecfe7de0be600f10aee3d34520714fe84f6983314a12e36dee345df3e154e2eb8d54db20984c0f4db83513e276cd3353ca64d0e71730b70d8b05eb0e037ae193a9c03ee4b3b35d1d")),W(false,-716,UInt!704.fromHexString("870e5f93520b583e50caa56804c9ede14b52a2fb6dd35ab15f81b805ef103307980c2e095cdc3afb9ca96785e651de8b1f74a25dcdba88fc2b974f118d7a03e558086edad07ae4c84c6ef58424c728cf466bcf1ed27b0fd5"))),
        Reference(648UL,W(false,-692,UInt!704.fromHexString("ddf3cb81279f65412603dceec1bd66c75b4a308e0c0383d6537d1bb53c2fd2449acf9ba824aab1445ed886713d33d87455d9eae68427f3c31d704233cac48057ea82415233dec24d594ce9228b334d3b944addadb152b7d0")),W(false,-716,UInt!704.fromHexString("86d90490e21fa1636ecd98a9ff724b1581e6a100d228532f4cc2e2bcd644dc8f998f1fc2a39b06b63609e1161dc50c703bd42e0392874876bfc8b674a9b0c5d384fe30eee2d4033e4855233a4c86ba7c3a2420d1988ce054"))),
        Reference(4294967295UL,W(false,-667,UInt!704.fromHexString("a97217f780706a21fad64e966e51ac0df3bf8b1f9596e8bbb3ac2db386dfb7e800e362e6f632200736d33d50311370b3931a7aaba6906f864d5af289257b797f130a2eaf6aebe5dc280b83f171e9e631c2cc2b9fe41741b7")),W(false,-739,UInt!704.fromHexString("aaaaaaab55555555fa4fa4fae93e93e9c8bc8bc936736736dfe5fe60498c98c9e957cbcaa3d1ff10b6cbe8ddddcfcd79473c35b416c073c92c459e1c96e99665bf237ad2da95f5ea8062f4f2279d41cfed801a978b7e501d"))),
        Reference(4294967296UL,W(false,-667,UInt!704.fromHexString("a97217f831e28219cca5c84238355fa5f7b281ced68a2be22644e3e110eccf438c8e5d12ddea9627a4bee9e886a8c3aede1495bc93bf1dbc0e9313cd4cd2b4a82473e6d4a982aff24a566cb68cb5c042f448641e9e01dd7b")),W(false,-739,UInt!704.fromHexString("aaaaaaaaaaaaaaaaa4fa4fa4fa4fa4fa51451451451451451319319319319319334c6b1444a923c8862eb07e6cc0024dc5612f49422787516ee33c5359269ab93afa05944fae87eff1a366db4388d401b8fdc48a10e11c83"))),
        Reference(1000000000000UL,W(false,-659,UInt!704.fromHexString("c1c4246dd13a42e1d58e9b9ffd5051f4604334285ac4935270d875e7c54ade0852e79db6af2eb8d1dcd740760ac6f11b1d839607f103bc5ea4cd553b8d7cc4b998f8db3dcb7e15154ed887d3eb8387563d7ce2ba4edbaa56")),W(false,-747,UInt!704.fromHexString("bba66561e9c0b6654c54ad6486caca31eb9ac6a00c3dd93b10a93e1a95b2507885029943f5fec3686bbb5f258fa6f83322a3ffad1d24160eeaae37bc0dbd56a6adef3bdb1ace0bae4d62338c076c73bd1c19cc90252b8d8b"))),
        Reference(18446744073709551615UL,W(false,-634,UInt!704.fromHexString("ad7217f7d1cf79ab74d7a5d527a3106e2ea9b0efcb04f4abce278d5bdc010a97af7f8226c789ffaa5365a8de3b56fc079331a795db2448e09760d4b36cb8d2cc18200ba956b62fa1893e4a0d472319af008af5b5261c4793")),W(false,-771,UInt!704.fromHexString("aaaaaaaaaaaaaaab5555555555555555fa4fa4fa4fa4fa503e93e93e93e93e947367367367367367b05b05b05b05b05b722f22f22f22f22f8680680680680680c4c582a5bdc0b541f575a4edb3b47194f655b2ee93c27275"))),
    ];
    foreach (r; references) {
        assert(L.factorial(r.n) == r.factorial);
        assert(L.correction(r.n) == r.correction);
    }
}

// END log-factorial references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    foreach (bits; AliasSeq!(64u,128u,256u,512u)) {
        alias L = CountLog!bits;
        alias W = Fp!(bits + 192);
        assert(L.factorial(0) == W(0));
        assert(!L.factorial(0).sign);
        assert(L.factorial(1) == W(0));
        static assert(is(typeof(L.factorial(2)) == W));
        static assert(is(typeof(L.correction(2)) == W));
        debug {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            assertThrown!AssertError(L.correction(0));
            assertThrown!AssertError(L.usesStirling(0));
        }
    }
}
