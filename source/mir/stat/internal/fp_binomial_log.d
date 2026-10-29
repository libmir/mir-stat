/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_binomial_log;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_log_factorial: CountLog;
import mir.stat.internal.fp_probability: fpProbabilityLess;
import mir.stat.internal.fp_probability_log: ProbabilityLog;

// Interior binomial log masses. Endpoint handling belongs to the public API.
// The centered identity follows Loader's saddle-point formulation:
// https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
// This is guarded evaluation, not a universal correct-rounding guarantee.
package(mir.stat) struct BinomialLog(uint bits)
{
    alias T = Fp!bits;
    alias Logs = ProbabilityLog!bits;
    alias Counts = CountLog!bits;
    enum fractionBits = bits + 320;
    alias I = UInt!(fractionBits + 192);

    @safe pure nothrow @nogc:

    // All values are positive magnitudes. Very small probabilities and
    // complements may lie below this absolute grid; their logarithms are
    // evaluated from the original input, never from the truncated value.
    private static I fixed(uint width)(Fp!width value)
    {
        if (!value.coefficient) return I(0);
        const shift = value.exponent + fractionBits;
        if (shift <= -cast(long)width) return I(0);
        auto result = value.coefficient.toSize!(fractionBits + 192);
        if (shift >= 0) result <<= cast(size_t)shift;
        else result >>= cast(size_t)-shift;
        return result;
    }

    // x*log(x/mean)+mean-x. Near the mean, v=(x-mean)/(x+mean)
    // gives delta*v + 2*x*(v^3/3+v^5/5+...). Here |v|<1/3.
    // The grid controls absolute error in the final interior log mass;
    // this is not a relative-accuracy API for arbitrarily tiny deviances.
    private static I deviance(ulong x, I mean, I logX, I logN, I negativeLogP)
    {
        const count = I(x) << fractionBits;
        const negative = count < mean;
        const delta = negative ? mean-count : count-mean;
        if ((delta << 1) < count)
        {
            import mir.stat.internal.fp_count_deviance: fpCenteredDeviance;
            return fpCenteredDeviance!fractionBits(x,mean);
        }
        // Outside the centered region cancellation is bounded. Separate
        // logarithms also avoid forming an overflowing ratio x/mean.
        auto positive = logX + negativeLogP;
        positive *= UInt!64(x);
        positive += mean;
        auto subtract = logN;
        subtract *= UInt!64(x);
        subtract += count;
        assert(positive >= subtract);
        return positive - subtract;
    }

    static T mass(ulong k, ulong n, T p)
        in (k > 0 && k < n, "counts must be interior")
        in (!p.sign && !!p.coefficient && fpProbabilityLess(p,T(1)),
            "probability must be strictly between zero and one")
    {
        const m = n-k;
        const logK = fixed(Logs.logCount(UInt!128(k)));
        const logM = fixed(Logs.logCount(UInt!128(m)));
        const logN = fixed(Logs.logCount(UInt!128(n)));
        // Retain the complement on the integer grid instead of rounding
        // 1-p to the input Fp width. Mean error is below n*2^-fractionBits.
        auto mean = fixed(p);
        mean *= UInt!64(n);
        const complementMean = (I(n) << fractionBits) - mean;
        const logP = fixed(Logs.probability(p).finish!(bits + 128));
        const logQ = fixed(Logs.complement(p).finish!(bits + 128));
        auto magnitude = deviance(k,mean,logK,logN,logP);
        magnitude += deviance(m,complementMean,logM,logN,logQ);
        // 1-correction(1) = log(2*pi)/2. Reuse the guarded constant
        // calculation without introducing a hardware-precision constant.
        auto normalization = (I(1) << fractionBits) - fixed(Counts.correction(1));
        normalization <<= 1;
        normalization += logK;
        normalization += logM;
        normalization -= logN;
        magnitude += normalization >> 1;
        magnitude += fixed(Counts.correction(k));
        magnitude += fixed(Counts.correction(m));
        magnitude -= fixed(Counts.correction(n));
        // Every interior binomial mass is <=1/2, so its log magnitude is
        // bounded away from zero. Round only after combining the terms.
        auto result = T(magnitude);
        result.exponent -= fractionBits;
        result.sign = true;
        return result;
    }
}
// BEGIN centered binomial references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    {
        alias T = Fp!64;
        struct Reference { ulong k; ulong n; T p; T value; }
        enum references = [
            Reference(1UL,2UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-64,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(1UL,3UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-64,UInt!64.fromHexString("fb17a03ca53dc38d"))),
            Reference(1UL,32UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("95b8443919070ea9"))),
            Reference(1UL,128UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-57,UInt!64.fromHexString("a7bddaa844562104"))),
            Reference(1UL,4294967297UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-32,UInt!64.fromHexString("b17217e254fe92a8"))),
            Reference(1UL,1000000000000UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-24,UInt!64.fromHexString("a162ca9df4507533"))),
            Reference(1UL,18446744073709551614UL,T(false,-1,UInt!64.fromHexString("1")),T(true,0,UInt!64.fromHexString("b17217f7d1cf797e"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf7953"))),
            Reference(1UL,18446744073709551615UL,T(false,-2000,UInt!64.fromHexString("1")),T(true,-53,UInt!64.fromHexString("a7bddaa844562104"))),
            Reference(1UL,18446744073709551615UL,T(false,-384,UInt!64.fromHexString("1")),T(true,-56,UInt!64.fromHexString("ddce9df5c6435817"))),
            Reference(1UL,18446744073709551615UL,T(false,-383,UInt!64.fromHexString("1")),T(true,-56,UInt!64.fromHexString("dd1d2bddce71889d"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("1")),T(true,-64,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(1UL,18446744073709551615UL,T(false,-1,UInt!64.fromHexString("1")),T(true,0,UInt!64.fromHexString("b17217f7d1cf797f"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf78fa"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,6,UInt!64.fromHexString("b17217f7d1cf79aa"))),
            Reference(2UL,3UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-64,UInt!64.fromHexString("fb17a03ca53dc38d"))),
            Reference(16UL,32UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-63,UInt!64.fromHexString("fbb54e73f674a98f"))),
            Reference(31UL,32UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("95b8443919070ea9"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("1fffffffffffffff")),T(true,-60,UInt!64.fromHexString("9e49daeef952a469"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("2000000000000000")),T(true,-60,UInt!64.fromHexString("9e49daeef952a460"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("2000000000000001")),T(true,-60,UInt!64.fromHexString("9e49daeef952a457"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("5fffffffffffffff")),T(true,-61,UInt!64.fromHexString("e13d9fe64bbd91f2"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("6000000000000000")),T(true,-61,UInt!64.fromHexString("e13d9fe64bbd91fb"))),
            Reference(32UL,128UL,T(false,-64,UInt!64.fromHexString("6000000000000001")),T(true,-61,UInt!64.fromHexString("e13d9fe64bbd9203"))),
            Reference(64UL,128UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-62,UInt!64.fromHexString("a9d7327657b0218f"))),
            Reference(127UL,128UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-57,UInt!64.fromHexString("a7bddaa844562104"))),
            Reference(6134UL,55753UL,T(false,-64,UInt!64.fromHexString("47da373f7dfb4d2f")),T(true,-51,UInt!64.fromHexString("96bcf1dc5de5e168"))),
            Reference(90440365UL,132882093UL,T(false,-64,UInt!64.fromHexString("3f8706accf6c46d5")),T(true,-38,UInt!64.fromHexString("d17aafc5d211a579"))),
            Reference(2147483648UL,4294967297UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-60,UInt!64.fromHexString("b50eef5c9b11dc64"))),
            Reference(4294967296UL,4294967297UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-32,UInt!64.fromHexString("b17217e254fe92a8"))),
            Reference(74538318300UL,106608362049UL,T(false,-64,UInt!64.fromHexString("a302d48b73914549")),T(true,-34,UInt!64.fromHexString("dc31df71231bdd2e"))),
            Reference(500000000000UL,1000000000000UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-60,UInt!64.fromHexString("e0a92c313543cac1"))),
            Reference(999999999999UL,1000000000000UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-24,UInt!64.fromHexString("a162ca9df4507533"))),
            Reference(2259511228033UL,4576719449132UL,T(false,-64,UInt!64.fromHexString("2dfcf807a025849a")),T(true,-23,UInt!64.fromHexString("87bcccc0811d6302"))),
            Reference(9223372036854775807UL,18446744073709551614UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("b34083aa2c70ab08"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,62,UInt!64.fromHexString("b17217f7d1cf79a8"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-2000,UInt!64.fromHexString("1")),T(true,10,UInt!64.fromHexString("ad1d0ee204f028f2"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-384,UInt!64.fromHexString("1")),T(true,8,UInt!64.fromHexString("84641fe1e589cbc6"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-383,UInt!64.fromHexString("1")),T(true,8,UInt!64.fromHexString("840b66d5e9a0e409"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-1,UInt!64.fromHexString("1")),T(true,-59,UInt!64.fromHexString("b34083aa2c70ab08"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("7fffffffffffffff")),T(true,-59,UInt!64.fromHexString("b34083aa2c70ab08"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("8000000000000000")),T(true,-59,UInt!64.fromHexString("b34083aa2c70ab08"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("8000000000000001")),T(true,-59,UInt!64.fromHexString("b34083aa2c70ab08"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(true,62,UInt!64.fromHexString("b17217f7d1cf794f"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,5,UInt!64.fromHexString("abe687381340fddf"))),
            Reference(18446744073709551613UL,18446744073709551614UL,T(false,-1,UInt!64.fromHexString("1")),T(true,0,UInt!64.fromHexString("b17217f7d1cf797e"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!64.fromHexString("1")),T(true,0,UInt!64.fromHexString("b17217f7d1cf797f"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-64,UInt!64.fromHexString("ffffffffffffffff")),T(true,-64,UInt!64.fromHexString("ffffffffffffffff"))),
        ];
        foreach (r; references)
            assert(BinomialLog!64.mass(r.k,r.n,r.p) == r.value);
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; ulong n; T p; T value; }
        enum references = [
            Reference(1UL,2UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-128,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(1UL,3UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-128,UInt!128.fromHexString("fb17a03ca53dc38cb9918d73069d6758"))),
            Reference(1UL,32UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("95b8443919070ea8f2581f8843550024"))),
            Reference(1UL,128UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-121,UInt!128.fromHexString("a7bddaa84456210464d93fc5b3bbad32"))),
            Reference(1UL,4294967297UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-96,UInt!128.fromHexString("b17217e254fe92a861c3f7cb11603746"))),
            Reference(1UL,1000000000000UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-88,UInt!128.fromHexString("a162ca9df4507532b79d3eea928360a7"))),
            Reference(1UL,18446744073709551614UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-64,UInt!128.fromHexString("b17217f7d1cf797e0a7985b3ec759867"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf795310d7b7af1c3620cc"))),
            Reference(1UL,18446744073709551615UL,T(false,-2000,UInt!128.fromHexString("1")),T(true,-117,UInt!128.fromHexString("a7bddaa84456210464f93fc5b3bbad32"))),
            Reference(1UL,18446744073709551615UL,T(false,-448,UInt!128.fromHexString("1")),T(true,-119,UInt!128.fromHexString("851591f9dd5b9b40d7eac6b202f63903"))),
            Reference(1UL,18446744073709551615UL,T(false,-447,UInt!128.fromHexString("1")),T(true,-119,UInt!128.fromHexString("84bcd8ede172b3840205d4d836f43f88"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!128.fromHexString("1")),T(true,-128,UInt!128.fromHexString("ffffffffffffffff8000000000000000"))),
            Reference(1UL,18446744073709551615UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-64,UInt!128.fromHexString("b17217f7d1cf797ebbeb9dabbe451212"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf78a19ebfbfdd4cbc7502"))),
            Reference(1UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-57,UInt!128.fromHexString("b17217f7d1cf79aa0e4677ac776c4682"))),
            Reference(2UL,3UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-128,UInt!128.fromHexString("fb17a03ca53dc38cb9918d73069d6758"))),
            Reference(16UL,32UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-127,UInt!128.fromHexString("fbb54e73f674a98ef57ca95f21c28208"))),
            Reference(31UL,32UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("95b8443919070ea8f2581f8843550024"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("1fffffffffffffffffffffffffffffff")),T(true,-124,UInt!128.fromHexString("9e49daeef952a46009b7b73b8f663f89"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("20000000000000000000000000000000")),T(true,-124,UInt!128.fromHexString("9e49daeef952a46009b7b73b8f663f80"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("20000000000000000000000000000001")),T(true,-124,UInt!128.fromHexString("9e49daeef952a46009b7b73b8f663f77"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("5fffffffffffffffffffffffffffffff")),T(true,-125,UInt!128.fromHexString("e13d9fe64bbd91fa98648dd07afeb7f9"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("60000000000000000000000000000000")),T(true,-125,UInt!128.fromHexString("e13d9fe64bbd91fa98648dd07afeb801"))),
            Reference(32UL,128UL,T(false,-128,UInt!128.fromHexString("60000000000000000000000000000001")),T(true,-125,UInt!128.fromHexString("e13d9fe64bbd91fa98648dd07afeb80a"))),
            Reference(64UL,128UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-126,UInt!128.fromHexString("a9d7327657b0218e94e0dc5b04fc8206"))),
            Reference(127UL,128UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-121,UInt!128.fromHexString("a7bddaa84456210464d93fc5b3bbad32"))),
            Reference(1306UL,1872UL,T(false,-128,UInt!128.fromHexString("45e2c0172fa54868a1bb70e4b5310fe9")),T(true,-118,UInt!128.fromHexString("b72debe0ba6f5204d4d7638a3a3ff314"))),
            Reference(344074UL,117694592UL,T(false,-128,UInt!128.fromHexString("f5f71bfebdcf62dc5cea91fa1f86d377")),T(true,-99,UInt!128.fromHexString("b4232ec7be9b5267b21f79d10f4f1896"))),
            Reference(61004680UL,115494047UL,T(false,-128,UInt!128.fromHexString("d4daf2ca7a55cf775dd9cd750c9be8c2")),T(true,-103,UInt!128.fromHexString("d8c72f510bee7ce891972fd82de7cacc"))),
            Reference(2147483648UL,4294967297UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-124,UInt!128.fromHexString("b50eef5c9b11dc643e91415ebca003f1"))),
            Reference(4294967296UL,4294967297UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-96,UInt!128.fromHexString("b17217e254fe92a861c3f7cb11603746"))),
            Reference(161348658843UL,211665312785UL,T(false,-128,UInt!128.fromHexString("58cb0b0e40975b1d17ec17bc75633c70")),T(true,-91,UInt!128.fromHexString("8dec51fd320dd8f927f6372a45c207a6"))),
            Reference(500000000000UL,1000000000000UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-124,UInt!128.fromHexString("e0a92c313543cac14c7b498c1e9ebbd0"))),
            Reference(999999999999UL,1000000000000UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-88,UInt!128.fromHexString("a162ca9df4507532b79d3eea928360a7"))),
            Reference(9223372036854775807UL,18446744073709551614UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("b34083aa2c70ab080a3a7a7b4df427fb"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-2,UInt!128.fromHexString("b17217f7d1cf79a7a13723c919161d05"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-2000,UInt!128.fromHexString("1")),T(true,-54,UInt!128.fromHexString("ad1d0ee204f028f203696e48d76907bd"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-448,UInt!128.fromHexString("1")),T(true,-56,UInt!128.fromHexString("9a9262e0dfc3bafb856f320cf9a574d1"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-447,UInt!128.fromHexString("1")),T(true,-56,UInt!128.fromHexString("9a39a9d4e3dad33eb03bb24b25754acf"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-123,UInt!128.fromHexString("b34083aa2c70ab08123a7a7b4df427fb"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-123,UInt!128.fromHexString("b34083aa2c70ab08123a7a7b4df427fb"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-123,UInt!128.fromHexString("b34083aa2c70ab08123a7a7b4df427fb"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-123,UInt!128.fromHexString("b34083aa2c70ab08123a7a7b4df427fb"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-2,UInt!128.fromHexString("b17217f7d1cf78f62f1f2bf7499c729e"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-58,UInt!128.fromHexString("aeac4f97f2883bc577222efe9962be3f"))),
            Reference(18446744073709551613UL,18446744073709551614UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-64,UInt!128.fromHexString("b17217f7d1cf797e0a7985b3ec759867"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!128.fromHexString("1")),T(true,-64,UInt!128.fromHexString("b17217f7d1cf797ebbeb9dabbe451212"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-128,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-122,UInt!128.fromHexString("b17217f7d1cf79abd1e3b39803f2f6af"))),
        ];
        foreach (r; references)
            assert(BinomialLog!128.mass(r.k,r.n,r.p) == r.value);
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; ulong n; T p; T value; }
        enum references = [
            Reference(1UL,2UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-256,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(1UL,3UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-256,UInt!256.fromHexString("fb17a03ca53dc38cb9918d73069d675804c586999fe99498608f572851b7b8f7"))),
            Reference(1UL,32UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("95b8443919070ea8f2581f8843550023decd40a870b0d9b66c7b0bb53dd84315"))),
            Reference(1UL,128UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-249,UInt!256.fromHexString("a7bddaa84456210464d93fc5b3bbad31ab65f57a58545c370c8060148a07a07e"))),
            Reference(1UL,4294967297UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-224,UInt!256.fromHexString("b17217e254fe92a861c3f7cb1160374671320e987fce3f3021067f4f76374838"))),
            Reference(1UL,1000000000000UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-216,UInt!256.fromHexString("a162ca9df4507532b79d3eea928360a6cba50c37a2b24964e0d74e22a68ced55"))),
            Reference(1UL,18446744073709551614UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-192,UInt!256.fromHexString("b17217f7d1cf797e0a7985b3ec759867343ef5f56df51d00cb55c772004c0271"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf795310d7b7af1c3620cc4f197724791d5e8e106b84223f4fe368"))),
            Reference(1UL,18446744073709551615UL,T(false,-2000,UInt!256.fromHexString("1")),T(true,-245,UInt!256.fromHexString("a7bddaa84456210464f93fc5b3bbad31ab75f57a58545c370c8b0abf34b24b28"))),
            Reference(1UL,18446744073709551615UL,T(false,-576,UInt!256.fromHexString("1")),T(true,-247,UInt!256.fromHexString("b17217f7d1cf79abca63b39803f2f6af413343267298b62d8a37c2063655a4d7"))),
            Reference(1UL,18446744073709551615UL,T(false,-575,UInt!256.fromHexString("1")),T(true,-247,UInt!256.fromHexString("b1195eebd5e691eef47ec1be37f0fd33e992c984df5f69d27372bb7a888fcf59"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!256.fromHexString("1")),T(true,-256,UInt!256.fromHexString("ffffffffffffffff7fffffffffffffffd5555555555555554000000000000000"))),
            Reference(1UL,18446744073709551615UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-192,UInt!256.fromHexString("b17217f7d1cf797ebbeb9dabbe451211fe22a98d71e813ae8c490a9872e4b89c"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf77f02ca7c80b7d42c93887b2471c9330000c29e5373d0de38853"))),
            Reference(1UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-184,UInt!256.fromHexString("b17217f7d1cf79aa3aa2fdaa6be024ecbbb2ef1069b60b233869c04509d36045"))),
            Reference(2UL,3UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-256,UInt!256.fromHexString("fb17a03ca53dc38cb9918d73069d675804c586999fe99498608f572851b7b8f7"))),
            Reference(16UL,32UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-255,UInt!256.fromHexString("fbb54e73f674a98ef57ca95f21c28208513bfd9e4faddc6a0de2b5b7e0854494"))),
            Reference(31UL,32UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("95b8443919070ea8f2581f8843550023decd40a870b0d9b66c7b0bb53dd84315"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-252,UInt!256.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e100f"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("2000000000000000000000000000000000000000000000000000000000000000")),T(true,-252,UInt!256.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e1006"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("2000000000000000000000000000000000000000000000000000000000000001")),T(true,-252,UInt!256.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e0ffd"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("5fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-253,UInt!256.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee54683991e"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("6000000000000000000000000000000000000000000000000000000000000000")),T(true,-253,UInt!256.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee546839926"))),
            Reference(32UL,128UL,T(false,-256,UInt!256.fromHexString("6000000000000000000000000000000000000000000000000000000000000001")),T(true,-253,UInt!256.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee54683992f"))),
            Reference(64UL,128UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-254,UInt!256.fromHexString("a9d7327657b0218e94e0dc5b04fc82062832af4eb28d54e2341553c4b9027b3c"))),
            Reference(127UL,128UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-249,UInt!256.fromHexString("a7bddaa84456210464d93fc5b3bbad31ab65f57a58545c370c8060148a07a07e"))),
            Reference(2147483648UL,4294967297UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-252,UInt!256.fromHexString("b50eef5c9b11dc643e91415ebca003f0f889c297eb4e2e2992035ac2d151043f"))),
            Reference(4294967296UL,4294967297UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-224,UInt!256.fromHexString("b17217e254fe92a861c3f7cb1160374671320e987fce3f3021067f4f76374838"))),
            Reference(29007260632UL,129362522727UL,T(false,-256,UInt!256.fromHexString("5124cbcfcbdd58ef250485f953c67bc04d11226bb2db7011471dca27dde7b62e")),T(true,-224,UInt!256.fromHexString("a31a53d92602b9fa75db07e2e7c2c7461fdeed8f501d787aa47a91f09e96860c"))),
            Reference(33459357654UL,34963934056UL,T(false,-256,UInt!256.fromHexString("58d702c4cd40a6d64c302ae640625db7771e120788e25f781b082b744720f4a9")),T(true,-221,UInt!256.fromHexString("de62547c86fcc56dc55742ce54b9ec69dabbe0bf0c5c75f93c5456c0e1e60d8c"))),
            Reference(500000000000UL,1000000000000UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-252,UInt!256.fromHexString("e0a92c313543cac14c7b498c1e9ebbcff7a2fe3828f2abebca4a81083ab1702a"))),
            Reference(999999999999UL,1000000000000UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-216,UInt!256.fromHexString("a162ca9df4507532b79d3eea928360a6cba50c37a2b24964e0d74e22a68ced55"))),
            Reference(8431621527541839UL,11495440243072355UL,T(false,-256,UInt!256.fromHexString("a82657222d4b3a8d4ae75504bd912cd5b4ff9e848ca245fd14bb41832e863a10")),T(true,-208,UInt!256.fromHexString("8e06b90e28062fe9f43de4ed9404c544ab864177c7a7ac0b1ed862c9e86a816f"))),
            Reference(16617026972612937UL,33193789266174491UL,T(false,-256,UInt!256.fromHexString("4882689ddd410923d9f3c1adfdd8b1f37fd621b253ad1d8d968a07385749b344")),T(true,-204,UInt!256.fromHexString("c5777a689ff415ec712d2d1dfb044cc6ad5e3f3635ccc7d652553d7ace0069cb"))),
            Reference(9223372036854775807UL,18446744073709551614UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("b34083aa2c70ab080a3a7a7b4df427fae1692d89a79e1cd689b2e3b9493af305"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-130,UInt!256.fromHexString("b17217f7d1cf79a7a13723c919161d04eba7428bda7a58ca4925907be5f68d4a"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-2000,UInt!256.fromHexString("1")),T(true,-182,UInt!256.fromHexString("ad1d0ee204f028f203696e48d76907bc9ed0e4be9b068fea258eac9f8e8960d4"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-576,UInt!256.fromHexString("1")),T(true,-184,UInt!256.fromHexString("c6eee8ded43799661f2f12f711ba75a6a1a1973513590128c78f6f0d7b2370e2"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-575,UInt!256.fromHexString("1")),T(true,-184,UInt!256.fromHexString("c6962fd2d84eb1a949fb93353d8a4ba4f5cb01471823a7c4600b5bc4f3d0341b"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-251,UInt!256.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db0"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-251,UInt!256.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db0"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-251,UInt!256.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db0"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-251,UInt!256.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db0"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-130,UInt!256.fromHexString("b17217f7d1cf7844bd0734257a22c836ec9ff1cb3273a96ff16da3a68064ef3a"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-185,UInt!256.fromHexString("b00f33c7e22bdab8a082f14b4eaada773e93c2a6293874ad46d52f158b605cda"))),
            Reference(18446744073709551613UL,18446744073709551614UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-192,UInt!256.fromHexString("b17217f7d1cf797e0a7985b3ec759867343ef5f56df51d00cb55c772004c0271"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!256.fromHexString("1")),T(true,-192,UInt!256.fromHexString("b17217f7d1cf797ebbeb9dabbe451211fe22a98d71e813ae8c490a9872e4b89c"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-256,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-248,UInt!256.fromHexString("851591f9dd5b9b40d86ac6b202f639037136725cd5f288a228df26d9fe1590f6"))),
        ];
        foreach (r; references)
            assert(BinomialLog!256.mass(r.k,r.n,r.p) == r.value);
    }
    {
        alias T = Fp!512;
        struct Reference { ulong k; ulong n; T p; T value; }
        enum references = [
            Reference(1UL,2UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-512,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(1UL,3UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-512,UInt!512.fromHexString("fb17a03ca53dc38cb9918d73069d675804c586999fe99498608f572851b7b8f70c91df8f3cfc6d78b04e01bedd53c25d652ff59a2500a72aa6cdd88a2418620b"))),
            Reference(1UL,32UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("95b8443919070ea8f2581f8843550023decd40a870b0d9b66c7b0bb53dd843150b83a3ab5cbed9a08835fe04074306d6481f62fd5b075c11813199eaa6b1335f"))),
            Reference(1UL,128UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-505,UInt!512.fromHexString("a7bddaa84456210464d93fc5b3bbad31ab65f57a58545c370c8060148a07a07d810c5faaa7e8c92800e728718ce06d96003620aed0a70f726d3075ecd275f013"))),
            Reference(1UL,4294967297UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-480,UInt!512.fromHexString("b17217e254fe92a861c3f7cb1160374671320e987fce3f3021067f4f76374837fb9403cd0e6e5c3ac2973545934764c823f5bb1716f47a9bb4348d09951ebcae"))),
            Reference(1UL,1000000000000UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-472,UInt!512.fromHexString("a162ca9df4507532b79d3eea928360a6cba50c37a2b24964e0d74e22a68ced5483ebf66219fa13ec4f5fa879a588c0799bb87ff35c8002c204d868eed5347ea0"))),
            Reference(1UL,18446744073709551614UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-448,UInt!512.fromHexString("b17217f7d1cf797e0a7985b3ec759867343ef5f56df51d00cb55c772004c0270fb031b311681d7f542b189499ee2467bedc05a7d7dce3802bd0a0d08fcace061"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-449,UInt!512.fromHexString("b17217f7d1cf795310d7b7af1c3620cc4f197724791d5e8e106b84223f4fe3678bd773054319414fa404ed6effce7991339641a1553ba9def66686aedbbf7c78"))),
            Reference(1UL,18446744073709551615UL,T(false,-2000,UInt!512.fromHexString("1")),T(true,-501,UInt!512.fromHexString("a7bddaa84456210464f93fc5b3bbad31ab75f57a58545c370c8b0abf34b24b282bbf0a55529373d2ab9839829df17ea7114c8715370d75d8d39b6e9c5d6e9f9e"))),
            Reference(1UL,18446744073709551615UL,T(false,-832,UInt!512.fromHexString("1")),T(true,-502,UInt!512.fromHexString("851591f9dd5b9b40d7aac6b202f6390370d6725cd5f288a2279f26d9fe1590f6432fadeda7c616c79592205e9a5db66ed40fcf751db6e5bfea575dbde300a932"))),
            Reference(1UL,18446744073709551615UL,T(false,-831,UInt!512.fromHexString("1")),T(true,-502,UInt!512.fromHexString("84e93573df6727626cb84dc51cf53c45c506358c0c55e2749c3ca3942732a637b835bfd01faa9bdc6f7cbb09db8af7e80fd483c9904697b7994d87ef18bc4ec4"))),
            Reference(1UL,18446744073709551615UL,T(false,-64,UInt!512.fromHexString("1")),T(true,-512,UInt!512.fromHexString("ffffffffffffffff7fffffffffffffffd5555555555555553ffffffffffffffff3333333333333332aaaaaaaaaaaaaaaa4924924924924924492492492492492"))),
            Reference(1UL,18446744073709551615UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-448,UInt!512.fromHexString("b17217f7d1cf797ebbeb9dabbe451211fe22a98d71e813ae8c490a9872e4b89c2fbadd374cd77cc81514aa14b7789db8bacd24f0403fca80a1b032b6355c790b"))),
            Reference(1UL,18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-449,UInt!512.fromHexString("b17217f7d1cf768d4877d867de4f71a4c04b1714ad42a18a435eea57dc772d3f577a04d6973091b0c22c6bb7511c183ade4a54756ccf662a3badafa9fb3a6bdb"))),
            Reference(1UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-439,UInt!512.fromHexString("b17217f7d1cf79aa50d140a9661a1422346f65836a3469f9204828a9d826770b22874f288fcb03a5b3f802a10c0ff6f7e215a2240628ec0473ada290706c0b0e"))),
            Reference(2UL,3UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-512,UInt!512.fromHexString("fb17a03ca53dc38cb9918d73069d675804c586999fe99498608f572851b7b8f70c91df8f3cfc6d78b04e01bedd53c25d652ff59a2500a72aa6cdd88a2418620b"))),
            Reference(16UL,32UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-511,UInt!512.fromHexString("fbb54e73f674a98ef57ca95f21c28208513bfd9e4faddc6a0de2b5b7e0854493f8afa7dffaef04d3dc51fdc9594c1500f06b6f8677ecfa353e93a27855dc9563"))),
            Reference(31UL,32UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("95b8443919070ea8f2581f8843550023decd40a870b0d9b66c7b0bb53dd843150b83a3ab5cbed9a08835fe04074306d6481f62fd5b075c11813199eaa6b1335f"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("1fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-508,UInt!512.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e1005b82394ab6a8b7140d72ac91da983991294b0422ab8f01b8cc8ea409b5a176a69"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-508,UInt!512.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e1005b82394ab6a8b7140d72ac91da983991294b0422ab8f01b8cc8ea409b5a176a60"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("20000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-508,UInt!512.fromHexString("9e49daeef952a46009b7b73b8f663f800b6af9e860bfc4c4af253ce17c8e1005b82394ab6a8b7140d72ac91da983991294b0422ab8f01b8cc8ea409b5a176a57"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("5fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-509,UInt!512.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee546839926604fffc0cfc7be04ee106d2fee0948748d08f6d64d7b06d06b1c5ec6465cfa1a"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-509,UInt!512.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee546839926604fffc0cfc7be04ee106d2fee0948748d08f6d64d7b06d06b1c5ec6465cfa23"))),
            Reference(32UL,128UL,T(false,-512,UInt!512.fromHexString("60000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-509,UInt!512.fromHexString("e13d9fe64bbd91fa98648dd07afeb80141754c11feda28f258219ee546839926604fffc0cfc7be04ee106d2fee0948748d08f6d64d7b06d06b1c5ec6465cfa2b"))),
            Reference(64UL,128UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-510,UInt!512.fromHexString("a9d7327657b0218e94e0dc5b04fc82062832af4eb28d54e2341553c4b9027b3b956bbf92b417b9a29c4660e0b78e8dc7fef6f1572d0b70675d8327e7af6b34c3"))),
            Reference(127UL,128UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-505,UInt!512.fromHexString("a7bddaa84456210464d93fc5b3bbad31ab65f57a58545c370c8060148a07a07d810c5faaa7e8c92800e728718ce06d96003620aed0a70f726d3075ecd275f013"))),
            Reference(15272UL,16003UL,T(false,-512,UInt!512.fromHexString("5c395caaee199ad932af547a378193a53a3adb6e6e9ef6459db23d1c80f6bf5cfd1997a6c70c20aa26faba03a8fbc95c69277eab4b4887c55b5f7c2c8e9ef508")),T(true,-498,UInt!512.fromHexString("ca630ce0274cde4fb01b2ba09e1bdf7217ebe25508c12fa163e3bd4069a3708d1fd7304fd1cf6e7c8aee6295fd92f6dd91b90b479ba6552e6efa9827e0b58f6d"))),
            Reference(1091159UL,1259872UL,T(false,-512,UInt!512.fromHexString("6eb1484692eb54eda362ab2bb451e2e4893dc834281b2d9cca7c45bcb58572d2eacbdb9dee130f914c70d4244e341dc44d678d8d31a99d29a4deb9f77eeaf011")),T(true,-493,UInt!512.fromHexString("fb21fe6f05b6a73b68be742da398cec0100cb330f3a0abe2960bea24c62b97b58039f15ddb5a847972772b671629dc1d96017113ba7e90b5ae851c9812ed6ac4"))),
            Reference(2147483648UL,4294967297UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-508,UInt!512.fromHexString("b50eef5c9b11dc643e91415ebca003f0f889c297eb4e2e2992035ac2d151043e83684a2017a843daff17839d5cbb11e03674fc83f31764c9197a01e52bc3d433"))),
            Reference(4294967296UL,4294967297UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-480,UInt!512.fromHexString("b17217e254fe92a861c3f7cb1160374671320e987fce3f3021067f4f76374837fb9403cd0e6e5c3ac2973545934764c823f5bb1716f47a9bb4348d09951ebcae"))),
            Reference(500000000000UL,1000000000000UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-508,UInt!512.fromHexString("e0a92c313543cac14c7b498c1e9ebbcff7a2fe3828f2abebca4a81083ab1702a4cde5f730aac2eb1032fa79a1317ba641d5b52b8bbad1c2b863aed93352026f2"))),
            Reference(999999999999UL,1000000000000UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-472,UInt!512.fromHexString("a162ca9df4507532b79d3eea928360a6cba50c37a2b24964e0d74e22a68ced5483ebf66219fa13ec4f5fa879a588c0799bb87ff35c8002c204d868eed5347ea0"))),
            Reference(13256555400177955UL,53340265501813024UL,T(false,-512,UInt!512.fromHexString("456b48e086c1ef52cf31059e2b9d20f27115d4ca13ee9a62f387483eb1d34f1da38704df6a81ab32ec14d47dcea9225e82db0e15db2053a9600032d886f83afb")),T(true,-465,UInt!512.fromHexString("8022d791ee099054ce5b2f53f14716df31e0549ec2c7c1e78bb351c9ae9bf40995da663d78d7f97f471f8deb6571539b72f023bf45d97adcdb531f251f2e899c"))),
            Reference(5537174010354916298UL,7211680560550038116UL,T(false,-512,UInt!512.fromHexString("3a1555c8e7bd79c88f52880d3ed13931b2a6d75b7db9e58d172de8bb4bc511f936209ad644f72dec03aa3b347eb9352f742ae82e85aed08ba888b22c82a9fade")),T(true,-449,UInt!512.fromHexString("8373dd6910256cc35e6a0063fd206b819bd676e8f5862d63571e2b7ba42d9405cfba6edcf6022b98cf4c943dee0ba49aa5d978e066ac25d9067da83926c45138"))),
            Reference(9223372036854775807UL,18446744073709551614UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("b34083aa2c70ab080a3a7a7b4df427fae1692d89a79e1cd689b2e3b9493af3056f842f5b697cc3698e54cb2cc051c2530ebf959643803e8459b3c27770a18870"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-386,UInt!512.fromHexString("b17217f7d1cf79a7a13723c919161d04eba7428bda7a58ca4925907be5f68d4a63cdbb68c15b138d3e3e1da0d8eed03ca1fd97ddcac13b5d3b1c2a560104a87a"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-2000,UInt!512.fromHexString("1")),T(true,-438,UInt!512.fromHexString("ad1d0ee204f028f203696e48d76907bc9ed0e4be9b068fea258eac9f8e8960d3a37413584cf4d2afe8c58beb0a71e92a0426d23269c77452ad0042db5fb45f36"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-832,UInt!512.fromHexString("1")),T(true,-439,UInt!512.fromHexString("8fd3fa6d5e8fab1da9576a65a0f23ba93c1bc298245932c825d15bca67301be515879d3c7326df36388992a96e202d303d010afcc63570723a3383a688543446"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-831,UInt!512.fromHexString("1")),T(true,-439,UInt!512.fromHexString("8fa79de7609b373f3ebdaa84b6da26a8663077a126be8615f20f522623867d81a152b5aa98d139c82868098fbf84647fc4f089fab66a9f7771a0452ed8f075e9"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-507,UInt!512.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db01c2eda0614276e143a990f710496069754592f2fdd19d81df471ee5a2ecd6b2e"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-507,UInt!512.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db01c2eda0614276e143a990f710496069754592f2fdd19d81df471ee5a2ecd6b2e"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-507,UInt!512.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db01c2eda0614276e143a990f710496069754592f2fdd19d81df471ee5a2ecd6b2e"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-507,UInt!512.fromHexString("b34083aa2c70ab08123a7a7b4df427fae5692d89a79e1cd68c5d8e63f3e59db01c2eda0614276e143a990f710496069754592f2fdd19d81df471ee5a2ecd6b2e"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-386,UInt!512.fromHexString("b17217f7d1cf76e1d8d74481db2f7368ed98a10a8a6cfa1599b5b6d11ad35129c98980cedb23d03ec5207846820dce241062ae213fb9ba552afb2da8f1581f46"))),
            Reference(9223372036854775807UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-440,UInt!512.fromHexString("b0c0a5dfd9fdaa3235335271a94ee8933fc382e64de8956d687123388b85ab82bae67c8e27861f9becad515f32a1fa64fbea36c8a754a410dea8388219bcc192"))),
            Reference(18446744073709551613UL,18446744073709551614UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-448,UInt!512.fromHexString("b17217f7d1cf797e0a7985b3ec759867343ef5f56df51d00cb55c772004c0270fb031b311681d7f542b189499ee2467bedc05a7d7dce3802bd0a0d08fcace061"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-1,UInt!512.fromHexString("1")),T(true,-448,UInt!512.fromHexString("b17217f7d1cf797ebbeb9dabbe451211fe22a98d71e813ae8c490a9872e4b89c2fbadd374cd77cc81514aa14b7789db8bacd24f0403fca80a1b032b6355c790b"))),
            Reference(18446744073709551614UL,18446744073709551615UL,T(false,-512,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-503,UInt!512.fromHexString("9b43d4f8d7958a7651273d25037497d95914dac1a4459f67d8f61f1ac4e04591158c12070ad8e1aff5a6ece025df1bf313e27208a2aab6b53c9836821b1f3f4c"))),
        ];
        foreach (r; references)
            assert(BinomialLog!512.mass(r.k,r.n,r.p) == r.value);
    }
}
// END centered binomial references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    foreach (bits; AliasSeq!(64u,128u,256u,512u))
    {
        alias T = Fp!bits;
        alias B = BinomialLog!bits;
        static assert(is(typeof(B.mass(1,2,T(0.5))) == T));
        // Equivalent normalized and raw inputs, and exact dyadic symmetry.
        assert(B.mass(1,2,T(0.5)) == B.mass(1,2,T(false,-1,UInt!bits(1))));
        assert(B.mass(3,17,T(0.25)) == B.mass(14,17,T(0.75)));
        debug
        {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            assertThrown!AssertError(B.mass(0,2,T(0.5)));
            assertThrown!AssertError(B.mass(2,2,T(0.5)));
            assertThrown!AssertError(B.mass(3,2,T(0.5)));
            assertThrown!AssertError(B.mass(1,2,T(0)));
            assertThrown!AssertError(B.mass(1,2,T(1)));
            assertThrown!AssertError(B.mass(1,2,T(false,0,UInt!bits(1))));
            assertThrown!AssertError(B.mass(1,2,T(-0.5)));
            assertThrown!AssertError(B.mass(1,2,T(double.infinity)));
            assertThrown!AssertError(B.mass(1,2,T(double.nan)));
        }
    }
}
