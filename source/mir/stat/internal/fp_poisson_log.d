/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_poisson_log;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_count_deviance: fpCenteredDeviance;
import mir.stat.internal.fp_log_factorial: CountLog;
import mir.stat.internal.fp_probability_log: ProbabilityLog;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled;

// Positive finite rates and positive counts; the public API handles k=0
// and infinity. Guarded evaluation, not a universal rounding guarantee.
package(mir.stat) struct PoissonLog(uint bits)
{
    alias T = Fp!bits;
    alias Logs = ProbabilityLog!bits;
    alias Counts = CountLog!bits;
    enum fractionBits = bits + 320;
    // The direct path admits rates below 2^(bits+192), as well as count
    // times logarithms below 2^128. Leave ample integer headroom for both.
    enum size = fractionBits + bits + 256;
    alias I = UInt!size;

    @safe pure nothrow @nogc:

    private static I fixed(uint width)(Fp!width value)
    {
        if (!value.coefficient) return I(0);
        const shift = value.exponent + fractionBits;
        if (shift <= -cast(long)width) return I(0);
        auto result = value.coefficient.toSize!size;
        if (shift >= 0) result <<= cast(size_t)shift;
        else result >>= cast(size_t)-shift;
        return result;
    }

    static T mass(ulong k, T lambda)
        in (k > 0, "count must be positive")
        in (!lambda.sign && !!lambda.coefficient && lambda.exponent != long.max,
            "rate must be finite and positive")
    {
        const top = bits - lambda.coefficient.ctlz;
        if (lambda.exponent >= cast(long)bits + 193 - cast(long)top)
        {
            // For lambda>=2^(bits+192), k*log(lambda)-log(k!) is far
            // below half an output ulp, even with k=ulong.max. At the
            // cutoff it is <2^64*(bits+193), whereas a half ulp below
            // the power of two is 2^191. The relative bound decreases
            // thereafter as log(lambda)/lambda decreases. Normalize via
            // a separate exponent, including at the upper Fp boundary.
            auto result = ProbabilityScaled!(bits + 192).from(lambda).finish!bits;
            result.sign = true;
            return result;
        }
        const rate = fixed(lambda);
        const count = I(k) << fractionBits;
        const delta = count < rate ? rate-count : count-rate;
        I magnitude;
        if ((delta << 1) < count)
        {
            magnitude = fpCenteredDeviance!fractionBits(k,rate);
            magnitude += fixed(Logs.logCount(UInt!128(k))) >> 1;
            // log(2*pi)/2 = 1-correction(1).
            magnitude += (I(1) << fractionBits) - fixed(Counts.correction(1));
            magnitude += fixed(Counts.correction(k));
        }
        else
        {
            // Away from the mode cancellation is bounded. Taking the log
            // of the original rate avoids truncating a tiny rate to zero.
            const logRate = Logs.logPositive(lambda);
            auto power = fixed(logRate); // Magnitude; sign is handled below.
            power *= UInt!64(k);
            magnitude = rate + fixed(Counts.factorial(k));
            if (logRate.sign) magnitude += power;
            else
            {
                assert(magnitude >= power);
                magnitude -= power;
            }
        }
        auto result = T(magnitude);
        result.exponent -= fractionBits;
        result.sign = true;
        return result;
    }
}
// BEGIN Poisson log references
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    {
        alias T = Fp!64;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf79ac"))),
            Reference(1UL,T(false,-2000,UInt!64.fromHexString("1")),T(true,-53,UInt!64.fromHexString("ad496b6802e49cd2"))),
            Reference(1UL,T(false,0,UInt!64.fromHexString("1")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,66,UInt!64.fromHexString("1")),T(true,2,UInt!64.fromHexString("fffffffffffffff5"))),
            Reference(1UL,T(false,long.max-1,UInt!64.fromHexString("1")),T(true,9223372036854775743,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,-63,UInt!64.fromHexString("7fffffffffffffff")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,193,UInt!64.fromHexString("7fffffffffffffff")),T(true,192,UInt!64.fromHexString("fffffffffffffffe"))),
            Reference(1UL,T(false,-63,UInt!64.fromHexString("8000000000000000")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,193,UInt!64.fromHexString("8000000000000000")),T(true,193,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,-63,UInt!64.fromHexString("8000000000000001")),T(true,-63,UInt!64.fromHexString("8000000000000000"))),
            Reference(1UL,T(false,193,UInt!64.fromHexString("8000000000000001")),T(true,193,UInt!64.fromHexString("8000000000000001"))),
            Reference(1UL,T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(true,-1,UInt!64.fromHexString("b17217f7d1cf7953"))),
            Reference(1UL,T(false,long.max-1,UInt!64.fromHexString("ffffffffffffffff")),T(true,long.max-1,UInt!64.fromHexString("ffffffffffffffff"))),
            Reference(2UL,T(false,0,UInt!64.fromHexString("2")),T(true,-63,UInt!64.fromHexString("a746f4041718432a"))),
            Reference(2UL,T(false,-62,UInt!64.fromHexString("7fffffffffffffff")),T(true,-63,UInt!64.fromHexString("a746f4041718432a"))),
            Reference(2UL,T(false,-62,UInt!64.fromHexString("8000000000000000")),T(true,-63,UInt!64.fromHexString("a746f4041718432a"))),
            Reference(2UL,T(false,-62,UInt!64.fromHexString("8000000000000001")),T(true,-63,UInt!64.fromHexString("a746f4041718432a"))),
            Reference(3UL,T(false,0,UInt!64.fromHexString("3")),T(true,-63,UInt!64.fromHexString("bf7a645118b7135f"))),
            Reference(3UL,T(false,-62,UInt!64.fromHexString("bfffffffffffffff")),T(true,-63,UInt!64.fromHexString("bf7a645118b7135f"))),
            Reference(3UL,T(false,-62,UInt!64.fromHexString("c000000000000000")),T(true,-63,UInt!64.fromHexString("bf7a645118b7135f"))),
            Reference(3UL,T(false,-62,UInt!64.fromHexString("c000000000000001")),T(true,-63,UInt!64.fromHexString("bf7a645118b7135f"))),
            Reference(32UL,T(false,0,UInt!64.fromHexString("20")),T(true,-62,UInt!64.fromHexString("a9e1dcdb5818955f"))),
            Reference(32UL,T(false,-59,UInt!64.fromHexString("7fffffffffffffff")),T(true,-60,UInt!64.fromHexString("8d5ca72679a518b1"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("7fffffffffffffff")),T(true,-62,UInt!64.fromHexString("a9e1dcdb5818955f"))),
            Reference(32UL,T(false,-59,UInt!64.fromHexString("8000000000000000")),T(true,-60,UInt!64.fromHexString("8d5ca72679a518af"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("8000000000000000")),T(true,-62,UInt!64.fromHexString("a9e1dcdb5818955f"))),
            Reference(32UL,T(false,-59,UInt!64.fromHexString("8000000000000001")),T(true,-60,UInt!64.fromHexString("8d5ca72679a518ad"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("8000000000000001")),T(true,-62,UInt!64.fromHexString("a9e1dcdb5818955f"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("bfffffffffffffff")),T(true,-61,UInt!64.fromHexString("b5beafa1b2878b81"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("c000000000000000")),T(true,-61,UInt!64.fromHexString("b5beafa1b2878b84"))),
            Reference(32UL,T(false,-58,UInt!64.fromHexString("c000000000000001")),T(true,-61,UInt!64.fromHexString("b5beafa1b2878b87"))),
            Reference(1000UL,T(false,0,UInt!64.fromHexString("3e8")),T(true,-61,UInt!64.fromHexString("8beecaf1e27ae0cf"))),
            Reference(1000UL,T(false,-54,UInt!64.fromHexString("f9ffffffffffffff")),T(true,-61,UInt!64.fromHexString("8beecaf1e27ae0cf"))),
            Reference(1000UL,T(false,-54,UInt!64.fromHexString("fa00000000000000")),T(true,-61,UInt!64.fromHexString("8beecaf1e27ae0cf"))),
            Reference(1000UL,T(false,-54,UInt!64.fromHexString("fa00000000000001")),T(true,-61,UInt!64.fromHexString("8beecaf1e27ae0cf"))),
            Reference(1000000000000UL,T(false,0,UInt!64.fromHexString("e8d4a51000")),T(true,-60,UInt!64.fromHexString("ebc04db0b25dd3c2"))),
            Reference(1000000000000UL,T(false,-24,UInt!64.fromHexString("e8d4a50fffffffff")),T(true,-60,UInt!64.fromHexString("ebc04db0b25dd3c2"))),
            Reference(1000000000000UL,T(false,-24,UInt!64.fromHexString("e8d4a51000000000")),T(true,-60,UInt!64.fromHexString("ebc04db0b25dd3c2"))),
            Reference(1000000000000UL,T(false,-24,UInt!64.fromHexString("e8d4a51000000001")),T(true,-60,UInt!64.fromHexString("ebc04db0b25dd3c2"))),
            Reference(1099664128727697408UL,T(false,52,UInt!64.fromHexString("f03d2694164f890d")),T(true,52,UInt!64.fromHexString("f03d2694164f6317"))),
            Reference(4142473531082317703UL,T(false,-129,UInt!64.fromHexString("1f8a3c4272fc6cf3")),T(true,5,UInt!64.fromHexString("9feb388df7732149"))),
            Reference(6342190202407639835UL,T(false,7,UInt!64.fromHexString("4957c524874a0bf")),T(true,1,UInt!64.fromHexString("a5e0ad8e4a79897f"))),
            Reference(8478113422545191011UL,T(false,-143,UInt!64.fromHexString("a8a996d261fda701")),T(true,6,UInt!64.fromHexString("b3b8bcca4aac553e"))),
            Reference(9052491066486425507UL,T(false,-141,UInt!64.fromHexString("828c68f03b5293cd")),T(true,6,UInt!64.fromHexString("bdcec7b8c9d5fbe7"))),
            Reference(10379856868825654344UL,T(false,-91,UInt!64.fromHexString("3f8e3216c0897859")),T(true,6,UInt!64.fromHexString("8d8fc07deb4ccd32"))),
            Reference(14789701164967083872UL,T(false,-79,UInt!64.fromHexString("e37c07dd80c735d4")),T(true,6,UInt!64.fromHexString("ac1306ba68730bd8"))),
            Reference(16557985816658722250UL,T(false,-112,UInt!64.fromHexString("b1760b472978f7ae")),T(true,7,UInt!64.fromHexString("8a093dab7af86a7d"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("1")),T(true,63,UInt!64.fromHexString("b17217f7d1cf7a02"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!64.fromHexString("1")),T(true,11,UInt!64.fromHexString("b2b4fc27c173189e"))),
            Reference(18446744073709551615UL,T(false,66,UInt!64.fromHexString("1")),T(true,1,UInt!64.fromHexString("ce8de8082e308660"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!64.fromHexString("1")),T(true,9223372036854775743,UInt!64.fromHexString("8000000000000000"))),
            Reference(18446744073709551615UL,T(false,193,UInt!64.fromHexString("7fffffffffffffff")),T(true,192,UInt!64.fromHexString("fffffffffffffffe"))),
            Reference(18446744073709551615UL,T(false,193,UInt!64.fromHexString("8000000000000000")),T(true,193,UInt!64.fromHexString("8000000000000000"))),
            Reference(18446744073709551615UL,T(false,193,UInt!64.fromHexString("8000000000000001")),T(true,193,UInt!64.fromHexString("8000000000000001"))),
            Reference(18446744073709551615UL,T(false,0,UInt!64.fromHexString("fffffffffffffffe")),T(true,-59,UInt!64.fromHexString("b8cc1469eaff26d5"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!64.fromHexString("ffffffffffffffff")),T(true,63,UInt!64.fromHexString("b17217f7d1cf79a9"))),
            Reference(18446744073709551615UL,T(false,0,UInt!64.fromHexString("ffffffffffffffff")),T(true,-59,UInt!64.fromHexString("b8cc1469eaff26d5"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!64.fromHexString("ffffffffffffffff")),T(true,long.max-1,UInt!64.fromHexString("ffffffffffffffff"))),
        ];
        foreach (r; references)
            assert(PoissonLog!64.mass(r.k,r.rate) == r.value);
    }
    {
        alias T = Fp!128;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af"))),
            Reference(1UL,T(false,-2000,UInt!128.fromHexString("1")),T(true,-117,UInt!128.fromHexString("ad496b6802e49cd1c3285d6273db44e7"))),
            Reference(1UL,T(false,0,UInt!128.fromHexString("1")),T(true,-127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,66,UInt!128.fromHexString("1")),T(true,-62,UInt!128.fromHexString("fffffffffffffff490257486fa20a86d"))),
            Reference(1UL,T(false,long.max-1,UInt!128.fromHexString("1")),T(true,9223372036854775679,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,-127,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,192,UInt!128.fromHexString("fffffffffffffffffffffffffffffffe"))),
            Reference(1UL,T(false,-127,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,193,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,-127,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-127,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,193,UInt!128.fromHexString("80000000000000000000000000000001"))),
            Reference(1UL,T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-65,UInt!128.fromHexString("b17217f7d1cf78fa57cbbbc634794ae5"))),
            Reference(1UL,T(false,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
            Reference(2UL,T(false,0,UInt!128.fromHexString("2")),T(true,-127,UInt!128.fromHexString("a746f4041718432a1b0e2633fe0684a8"))),
            Reference(2UL,T(false,-126,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-127,UInt!128.fromHexString("a746f4041718432a1b0e2633fe0684a8"))),
            Reference(2UL,T(false,-126,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-127,UInt!128.fromHexString("a746f4041718432a1b0e2633fe0684a8"))),
            Reference(2UL,T(false,-126,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-127,UInt!128.fromHexString("a746f4041718432a1b0e2633fe0684a8"))),
            Reference(3UL,T(false,0,UInt!128.fromHexString("3")),T(true,-127,UInt!128.fromHexString("bf7a645118b7135f40d84c76fcbdfea2"))),
            Reference(3UL,T(false,-126,UInt!128.fromHexString("bfffffffffffffffffffffffffffffff")),T(true,-127,UInt!128.fromHexString("bf7a645118b7135f40d84c76fcbdfea2"))),
            Reference(3UL,T(false,-126,UInt!128.fromHexString("c0000000000000000000000000000000")),T(true,-127,UInt!128.fromHexString("bf7a645118b7135f40d84c76fcbdfea2"))),
            Reference(3UL,T(false,-126,UInt!128.fromHexString("c0000000000000000000000000000001")),T(true,-127,UInt!128.fromHexString("bf7a645118b7135f40d84c76fcbdfea2"))),
            Reference(32UL,T(false,0,UInt!128.fromHexString("20")),T(true,-126,UInt!128.fromHexString("a9e1dcdb5818955e92b49e080ddd5f40"))),
            Reference(32UL,T(false,-123,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-124,UInt!128.fromHexString("8d5ca72679a518af38748eb20b5d4531"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,-126,UInt!128.fromHexString("a9e1dcdb5818955e92b49e080ddd5f40"))),
            Reference(32UL,T(false,-123,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-124,UInt!128.fromHexString("8d5ca72679a518af38748eb20b5d452f"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,-126,UInt!128.fromHexString("a9e1dcdb5818955e92b49e080ddd5f40"))),
            Reference(32UL,T(false,-123,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-124,UInt!128.fromHexString("8d5ca72679a518af38748eb20b5d452d"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,-126,UInt!128.fromHexString("a9e1dcdb5818955e92b49e080ddd5f40"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("bfffffffffffffffffffffffffffffff")),T(true,-125,UInt!128.fromHexString("b5beafa1b2878b83e082e81001cc9784"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("c0000000000000000000000000000000")),T(true,-125,UInt!128.fromHexString("b5beafa1b2878b83e082e81001cc9786"))),
            Reference(32UL,T(false,-122,UInt!128.fromHexString("c0000000000000000000000000000001")),T(true,-125,UInt!128.fromHexString("b5beafa1b2878b83e082e81001cc9789"))),
            Reference(1000UL,T(false,0,UInt!128.fromHexString("3e8")),T(true,-125,UInt!128.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176"))),
            Reference(1000UL,T(false,-118,UInt!128.fromHexString("f9ffffffffffffffffffffffffffffff")),T(true,-125,UInt!128.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176"))),
            Reference(1000UL,T(false,-118,UInt!128.fromHexString("fa000000000000000000000000000000")),T(true,-125,UInt!128.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176"))),
            Reference(1000UL,T(false,-118,UInt!128.fromHexString("fa000000000000000000000000000001")),T(true,-125,UInt!128.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176"))),
            Reference(1000000000000UL,T(false,0,UInt!128.fromHexString("e8d4a51000")),T(true,-124,UInt!128.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e9"))),
            Reference(1000000000000UL,T(false,-88,UInt!128.fromHexString("e8d4a50fffffffffffffffffffffffff")),T(true,-124,UInt!128.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e9"))),
            Reference(1000000000000UL,T(false,-88,UInt!128.fromHexString("e8d4a510000000000000000000000000")),T(true,-124,UInt!128.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e9"))),
            Reference(1000000000000UL,T(false,-88,UInt!128.fromHexString("e8d4a510000000000000000000000001")),T(true,-124,UInt!128.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e9"))),
            Reference(345744063286257038UL,T(false,24,UInt!128.fromHexString("e4e929a8091e4558c87da6c28196680c")),T(true,24,UInt!128.fromHexString("e4e929a8091e4558c87da5867d46c1ee"))),
            Reference(5994408007683596875UL,T(false,-130,UInt!128.fromHexString("49b6f2eda70c031494291891f24f86dc")),T(true,-60,UInt!128.fromHexString("e9494f58092095ca2ca80f45eb692501"))),
            Reference(6804041076389538662UL,T(false,21,UInt!128.fromHexString("7683f1d121a4b9db29e78cfb7b0967f9")),T(true,20,UInt!128.fromHexString("ed07e3a2434973b653cdb7043d92d80c"))),
            Reference(7031854698876746436UL,T(false,-198,UInt!128.fromHexString("3cfae9dec7019b7b772426fd4c712a55")),T(true,-58,UInt!128.fromHexString("8cd13b8a05768daeea1c9556834f1e79"))),
            Reference(15629880735743270439UL,T(false,-96,UInt!128.fromHexString("8dac33df4ef0d65db10e2ecdffdd7a17")),T(true,-59,UInt!128.fromHexString("92754bdf2f6ba51d434a3a81c9b1c1db"))),
            Reference(16164390881817304039UL,T(false,4,UInt!128.fromHexString("a297ed74f73b3736513f2805b00e3854")),T(true,4,UInt!128.fromHexString("a297ed74f73b3733b2e6fe44dec53ea3"))),
            Reference(17060941793812955829UL,T(false,72,UInt!128.fromHexString("cd150e049003b2684b2c9d694fb9b7c7")),T(true,72,UInt!128.fromHexString("cd150e049003b2684b2c9d694fb9b7c7"))),
            Reference(17533496639847685772UL,T(false,-13,UInt!128.fromHexString("420f63a6167b355977a53c4cb74fc387")),T(true,-14,UInt!128.fromHexString("841ec74c2cee16c19a59f8a94a24f965"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("1")),T(true,-1,UInt!128.fromHexString("b17217f7d1cf7a01d17d978919e052be"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!128.fromHexString("1")),T(true,-53,UInt!128.fromHexString("b2b4fc27c173189e7185af291a33c099"))),
            Reference(18446744073709551615UL,T(false,66,UInt!128.fromHexString("1")),T(true,-63,UInt!128.fromHexString("ce8de8082e308660744faafe7dcef053"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!128.fromHexString("1")),T(true,9223372036854775679,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,0,UInt!128.fromHexString("ffffffffffffffff")),T(true,-123,UInt!128.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b"))),
            Reference(18446744073709551615UL,T(false,193,UInt!128.fromHexString("7fffffffffffffffffffffffffffffff")),T(true,192,UInt!128.fromHexString("fffffffffffffffffffffffffffffffe"))),
            Reference(18446744073709551615UL,T(false,193,UInt!128.fromHexString("80000000000000000000000000000000")),T(true,193,UInt!128.fromHexString("80000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,193,UInt!128.fromHexString("80000000000000000000000000000001")),T(true,193,UInt!128.fromHexString("80000000000000000000000000000001"))),
            Reference(18446744073709551615UL,T(false,-64,UInt!128.fromHexString("fffffffffffffffeffffffffffffffff")),T(true,-123,UInt!128.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b"))),
            Reference(18446744073709551615UL,T(false,-64,UInt!128.fromHexString("ffffffffffffffff0000000000000000")),T(true,-123,UInt!128.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b"))),
            Reference(18446744073709551615UL,T(false,-64,UInt!128.fromHexString("ffffffffffffffff0000000000000001")),T(true,-123,UInt!128.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,-1,UInt!128.fromHexString("b17217f7d1cf79505f659fb74a66a7a5"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!128.fromHexString("ffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references)
            assert(PoissonLog!128.mass(r.k,r.rate) == r.value);
    }
    {
        alias T = Fp!256;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2c"))),
            Reference(1UL,T(false,-2000,UInt!256.fromHexString("1")),T(true,-245,UInt!256.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84f"))),
            Reference(1UL,T(false,0,UInt!256.fromHexString("1")),T(true,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,66,UInt!256.fromHexString("1")),T(true,-190,UInt!256.fromHexString("fffffffffffffff490257486fa20a86d7cd2ecb3bed719b450522c059d284211"))),
            Reference(1UL,T(false,long.max-1,UInt!256.fromHexString("1")),T(true,9223372036854775551,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,-255,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,192,UInt!256.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"))),
            Reference(1UL,T(false,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-255,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001"))),
            Reference(1UL,T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-193,UInt!256.fromHexString("b17217f7d1cf7848e5b3c3f464ff9f1b798c131e8cab57aba386ca765a3e9f18"))),
            Reference(1UL,T(false,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(2UL,T(false,0,UInt!256.fromHexString("2")),T(true,-255,UInt!256.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea"))),
            Reference(2UL,T(false,-254,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-255,UInt!256.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea"))),
            Reference(2UL,T(false,-254,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-255,UInt!256.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea"))),
            Reference(2UL,T(false,-254,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-255,UInt!256.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea"))),
            Reference(3UL,T(false,0,UInt!256.fromHexString("3")),T(true,-255,UInt!256.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c4789"))),
            Reference(3UL,T(false,-254,UInt!256.fromHexString("bfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-255,UInt!256.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c4789"))),
            Reference(3UL,T(false,-254,UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000")),T(true,-255,UInt!256.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c4789"))),
            Reference(3UL,T(false,-254,UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000001")),T(true,-255,UInt!256.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c4789"))),
            Reference(32UL,T(false,0,UInt!256.fromHexString("20")),T(true,-254,UInt!256.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d5"))),
            Reference(32UL,T(false,-251,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-252,UInt!256.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360f"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-254,UInt!256.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d5"))),
            Reference(32UL,T(false,-251,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-252,UInt!256.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360d"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,-254,UInt!256.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d5"))),
            Reference(32UL,T(false,-251,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-252,UInt!256.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360b"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,-254,UInt!256.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d5"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("bfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-253,UInt!256.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95e5"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000000")),T(true,-253,UInt!256.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95e7"))),
            Reference(32UL,T(false,-250,UInt!256.fromHexString("c000000000000000000000000000000000000000000000000000000000000001")),T(true,-253,UInt!256.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95ea"))),
            Reference(1000UL,T(false,0,UInt!256.fromHexString("3e8")),T(true,-253,UInt!256.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce6"))),
            Reference(1000UL,T(false,-246,UInt!256.fromHexString("f9ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-253,UInt!256.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce6"))),
            Reference(1000UL,T(false,-246,UInt!256.fromHexString("fa00000000000000000000000000000000000000000000000000000000000000")),T(true,-253,UInt!256.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce6"))),
            Reference(1000UL,T(false,-246,UInt!256.fromHexString("fa00000000000000000000000000000000000000000000000000000000000001")),T(true,-253,UInt!256.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce6"))),
            Reference(1000000000000UL,T(false,0,UInt!256.fromHexString("e8d4a51000")),T(true,-252,UInt!256.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d4"))),
            Reference(1000000000000UL,T(false,-216,UInt!256.fromHexString("e8d4a50fffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-252,UInt!256.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d4"))),
            Reference(1000000000000UL,T(false,-216,UInt!256.fromHexString("e8d4a51000000000000000000000000000000000000000000000000000000000")),T(true,-252,UInt!256.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d4"))),
            Reference(1000000000000UL,T(false,-216,UInt!256.fromHexString("e8d4a51000000000000000000000000000000000000000000000000000000001")),T(true,-252,UInt!256.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d4"))),
            Reference(3021064571077540289UL,T(false,-125,UInt!256.fromHexString("71c38309945187a3ecef117aff10e5a526a8c9b08b0db2aa3c5b3924b94be8c2")),T(true,-126,UInt!256.fromHexString("e387061328a30f45de283b8fdf81eec08ae9e416803e3de71bc37665c059ac9e"))),
            Reference(9022863216720369912UL,T(false,-237,UInt!256.fromHexString("2616afbbcf6709d3fa9d3addbac89b6ae6dce0ad496514bbf22ab56ff0bafe28")),T(true,-188,UInt!256.fromHexString("f598aaa61675466758cee1831e549d7fe5898f0f3d274ed8b0c9cb16b8943e21"))),
            Reference(11261251315053004022UL,T(false,-268,UInt!256.fromHexString("9d7749b4be0b6bc39badb167913e9129533cff027e32630958cb661c7df0214e")),T(true,-187,UInt!256.fromHexString("fc5a766a3c20ff7084eb53d8a12f5dd35165faaac7c4e9e7a3f9e881b31bd011"))),
            Reference(12838522469342613016UL,T(false,-314,UInt!256.fromHexString("fb6ddd67b515d4105d0490fcd17fd1d34208eb0a2818bf345a225a033fc0a115")),T(true,-186,UInt!256.fromHexString("e7acfe9e4ad940e5e106e1f6de9afc50284501c205ebd7eeb68fed6b31b31d71"))),
            Reference(14666964036035351210UL,T(false,-127,UInt!256.fromHexString("ec58c59d831368c8532473fe1493c628b1b7277712107d60caeaeb1718aa4fc9")),T(true,-127,UInt!256.fromHexString("ec58c59d831368b5f4d8f8587340ab1b96e1b6b3999920cf5738f253b3f1aa3c"))),
            Reference(15386617441947756486UL,T(false,-242,UInt!256.fromHexString("29ed604cc8d02c141b9df676744222e6889f5d6a9f2c0728bd163353be475822")),T(true,-187,UInt!256.fromHexString("eb743b973e466add266a611560b75efde008b4e579904a7b1ab3d99ec1e825af"))),
            Reference(16406198991776268394UL,T(false,-65,UInt!256.fromHexString("9c59f36ca3f858bcc333bb52f20c81cbf3676a79a834f0fc3955268d4c62c02e")),T(true,-65,UInt!256.fromHexString("9c59f36ca3f858bcc333bb52f20c812e41ad641035939431a47726996b51c04c"))),
            Reference(18327562747559799466UL,T(false,-166,UInt!256.fromHexString("8d34affb59ed3496eff3b9cb5ea9dbfa43235cc84ab2f27090f2ba3f5570f976")),T(true,-166,UInt!256.fromHexString("8d34ab673d2d53fdeadea33082d730f4d3ee64299043dd694c83fbd6c9bc11b1"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("1")),T(true,-129,UInt!256.fromHexString("b17217f7d1cf7a01d17d978919e052bde2e27a22402df6959df24c1a9b8c99e5"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!256.fromHexString("1")),T(true,-181,UInt!256.fromHexString("b2b4fc27c173189e7185af291a33c098d3ba7fb8966604a6b1d475a6b95bdebb"))),
            Reference(18446744073709551615UL,T(false,66,UInt!256.fromHexString("1")),T(true,-191,UInt!256.fromHexString("ce8de8082e308660744faafe7dcef0537c349c9d1d40e632589d50a344c68f2c"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!256.fromHexString("1")),T(true,9223372036854775551,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,0,UInt!256.fromHexString("ffffffffffffffff")),T(true,-251,UInt!256.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7c"))),
            Reference(18446744073709551615UL,T(false,193,UInt!256.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,192,UInt!256.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"))),
            Reference(18446744073709551615UL,T(false,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000")),T(true,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001")),T(true,193,UInt!256.fromHexString("8000000000000000000000000000000000000000000000000000000000000001"))),
            Reference(18446744073709551615UL,T(false,-192,UInt!256.fromHexString("fffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-251,UInt!256.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7c"))),
            Reference(18446744073709551615UL,T(false,-192,UInt!256.fromHexString("ffffffffffffffff000000000000000000000000000000000000000000000000")),T(true,-251,UInt!256.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7c"))),
            Reference(18446744073709551615UL,T(false,-192,UInt!256.fromHexString("ffffffffffffffff000000000000000000000000000000000000000000000001")),T(true,-251,UInt!256.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7c"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-129,UInt!256.fromHexString("b17217f7d1cf789eed4da7e57aecfc8cffab39bdf933efa77ed32f3d500d9d53"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!256.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references)
            assert(PoissonLog!256.mass(r.k,r.rate) == r.value);
    }
    {
        alias T = Fp!512;
        struct Reference { ulong k; T rate; T value; }
        enum references = [
            Reference(1UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-449,UInt!512.fromHexString("b17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206debac98559552fb4afa1b10ed2eae35c138214427573b291169b825"))),
            Reference(1UL,T(false,-2000,UInt!512.fromHexString("1")),T(true,-501,UInt!512.fromHexString("ad496b6802e49cd1c3285d6273db44e7256d8f938be921e878d0c8cf6664f84ee04a235bab58268cc393d30967383e6e879f96207eb0d07c8e6b2fc61b013dd4"))),
            Reference(1UL,T(false,0,UInt!512.fromHexString("1")),T(true,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,66,UInt!512.fromHexString("1")),T(true,-446,UInt!512.fromHexString("fffffffffffffff490257486fa20a86d7cd2ecb3bed719b450522c059d2842109a27fe997f7ae02b909c62e8ea4f602e7be026cdaae14168b67dc5890be1db1b"))),
            Reference(1UL,T(false,long.max-1,UInt!512.fromHexString("1")),T(true,9223372036854775295,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,-511,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,192,UInt!512.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"))),
            Reference(1UL,T(false,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-511,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(1UL,T(false,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"))),
            Reference(1UL,T(false,long.min,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-449,UInt!512.fromHexString("b17217f7d1cf76e60183d450c60c4787b224e316a6bdf929bd007d9128d24403b35b07f1c202fcf973bcd1439c47b9ba97e2c109d8cbdd8f6c9e642430e4a788"))),
            Reference(1UL,T(false,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
            Reference(2UL,T(false,0,UInt!512.fromHexString("2")),T(true,-511,UInt!512.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea0c23c4efc90a29b3d53556825a82f2778968a8e51f63ef5dec54626b774b23ed"))),
            Reference(2UL,T(false,-510,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-511,UInt!512.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea0c23c4efc90a29b3d53556825a82f2778968a8e51f63ef5dec54626b774b23ed"))),
            Reference(2UL,T(false,-510,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-511,UInt!512.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea0c23c4efc90a29b3d53556825a82f2778968a8e51f63ef5dec54626b774b23ed"))),
            Reference(2UL,T(false,-510,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-511,UInt!512.fromHexString("a746f4041718432a1b0e2633fe0684a85f865e6cc6b3a4e93af974523a2a82ea0c23c4efc90a29b3d53556825a82f2778968a8e51f63ef5dec54626b774b23ed"))),
            Reference(3UL,T(false,0,UInt!512.fromHexString("3")),T(true,-511,UInt!512.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c47894944b83e2a2f3dfbda58b24aa1e27eb3143b4213c1f454004473c4a3789015ae"))),
            Reference(3UL,T(false,-510,UInt!512.fromHexString("bfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-511,UInt!512.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c47894944b83e2a2f3dfbda58b24aa1e27eb3143b4213c1f454004473c4a3789015ae"))),
            Reference(3UL,T(false,-510,UInt!512.fromHexString("c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-511,UInt!512.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c47894944b83e2a2f3dfbda58b24aa1e27eb3143b4213c1f454004473c4a3789015ae"))),
            Reference(3UL,T(false,-510,UInt!512.fromHexString("c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-511,UInt!512.fromHexString("bf7a645118b7135f40d84c76fcbdfea1e2655eb9816bcd26876e9cc3748c47894944b83e2a2f3dfbda58b24aa1e27eb3143b4213c1f454004473c4a3789015ae"))),
            Reference(32UL,T(false,0,UInt!512.fromHexString("20")),T(true,-510,UInt!512.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d4aca06d9e0af2a50055bdb513df118bee2b5b8e220154933855515720464d755d"))),
            Reference(32UL,T(false,-507,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-508,UInt!512.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360cfa9907a85e940270c09a133b8db8991d65343ff402c567566402cc1a3466cda4"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-510,UInt!512.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d4aca06d9e0af2a50055bdb513df118bee2b5b8e220154933855515720464d755d"))),
            Reference(32UL,T(false,-507,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-508,UInt!512.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360cfa9907a85e940270c09a133b8db8991d65343ff402c567566402cc1a3466cda2"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-510,UInt!512.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d4aca06d9e0af2a50055bdb513df118bee2b5b8e220154933855515720464d755d"))),
            Reference(32UL,T(false,-507,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-508,UInt!512.fromHexString("8d5ca72679a518af38748eb20b5d452e92a8952f82ab796e6c10782cddb9360cfa9907a85e940270c09a133b8db8991d65343ff402c567566402cc1a3466cda0"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-510,UInt!512.fromHexString("a9e1dcdb5818955e92b49e080ddd5f4043083b8a75e8344d5fd925d7198d06d4aca06d9e0af2a50055bdb513df118bee2b5b8e220154933855515720464d755d"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("bfffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-509,UInt!512.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95e74ad404088a0da3a03f6c49ab0d06f6e540f82bcb8aebdc258b263470283a81af"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-509,UInt!512.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95e74ad404088a0da3a03f6c49ab0d06f6e540f82bcb8aebdc258b263470283a81b1"))),
            Reference(32UL,T(false,-506,UInt!512.fromHexString("c0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-509,UInt!512.fromHexString("b5beafa1b2878b83e082e81001cc97862d001ef825d4bb1be1c134b0764d95e74ad404088a0da3a03f6c49ab0d06f6e540f82bcb8aebdc258b263470283a81b4"))),
            Reference(1000UL,T(false,0,UInt!512.fromHexString("3e8")),T(true,-509,UInt!512.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce5f9c6385a5de682728bb3b9a635eca8a95af22fcb4107ef7369eed1d0a7759192"))),
            Reference(1000UL,T(false,-502,UInt!512.fromHexString("f9ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-509,UInt!512.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce5f9c6385a5de682728bb3b9a635eca8a95af22fcb4107ef7369eed1d0a7759192"))),
            Reference(1000UL,T(false,-502,UInt!512.fromHexString("fa000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-509,UInt!512.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce5f9c6385a5de682728bb3b9a635eca8a95af22fcb4107ef7369eed1d0a7759192"))),
            Reference(1000UL,T(false,-502,UInt!512.fromHexString("fa000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-509,UInt!512.fromHexString("8beecaf1e27ae0cf4825d5a85dc9f176037cb5d00ba118495720afc2028c7ce5f9c6385a5de682728bb3b9a635eca8a95af22fcb4107ef7369eed1d0a7759192"))),
            Reference(1000000000000UL,T(false,0,UInt!512.fromHexString("e8d4a51000")),T(true,-508,UInt!512.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d42f8ea980296e0afc5e6fc996cf24c22ebbe42027c3ace73635a7dddc47e8327d"))),
            Reference(1000000000000UL,T(false,-472,UInt!512.fromHexString("e8d4a50fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-508,UInt!512.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d42f8ea980296e0afc5e6fc996cf24c22ebbe42027c3ace73635a7dddc47e8327d"))),
            Reference(1000000000000UL,T(false,-472,UInt!512.fromHexString("e8d4a510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-508,UInt!512.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d42f8ea980296e0afc5e6fc996cf24c22ebbe42027c3ace73635a7dddc47e8327d"))),
            Reference(1000000000000UL,T(false,-472,UInt!512.fromHexString("e8d4a510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-508,UInt!512.fromHexString("ebc04db0b25dd3c27391ddc2c548b9e8eb9bfa052d25258030d5b06fe93906d42f8ea980296e0afc5e6fc996cf24c22ebbe42027c3ace73635a7dddc47e8327d"))),
            Reference(543865698368542469UL,T(false,-599,UInt!512.fromHexString("20fc54bd9f871d6b2047f3facc09965db7ded338ae3830c5b1ad2df3a09a57d875b2ce0e24a7686540520e20c7b6951274549fa4dbb70ef2f5dd6b64a8ced924")),T(true,-446,UInt!512.fromHexString("c0d30b404acc22170cdea9a7195e38cd6c423cedc15176a6a5da975e41cf4476e0d3768030857c480f259e8ab2108a6bb1b43f0d54f59c6a81e5daa29b549f8a"))),
            Reference(5210337937270514987UL,T(false,-329,UInt!512.fromHexString("6483acc6929cd44df7b6db6d5e492b0fe97713e856e667547c9ac2f2c0088f39f531f0cec821cfaafa342e9acf79fd1a6ee5d868b860aea770161718327c1629")),T(true,-330,UInt!512.fromHexString("c907598d2539a89bef6db6dabc91f76e2df749ee7dc7a2aaff0f0dcde5a6425cf675fecd3f29b6f604001b74364de7dd282f38e3c915e13cc308d5a323a11001"))),
            Reference(6841336277536246018UL,T(false,-391,UInt!512.fromHexString("958fa3317300fb12cef39fb0796422f16e8438fc9d7245d38fa580a6b8f3c81f5a44ef63a02a90896fedd7618491321e3d0aa2a58e5360dd5d644f629033fa7e")),T(true,-391,UInt!512.fromHexString("958fa3317300f37a33c827ed1f25faed7ab763aff01a627a97c79c5e4c7ed951e7aea96bc9a39c000bd3d21e214d701b5126150722089d551cba7e68594156a8"))),
            Reference(6977135136597304087UL,T(false,-456,UInt!512.fromHexString("ec26273bfd7427c369e25b0f955f1f5918c91a70b219cf31999062c5e038a87ecd82be35b7a5b49a1cf7f3cb6884f39e4340a0155db4c4501d9dbe9a5b36c85b")),T(true,-447,UInt!512.fromHexString("b158e285b0b458c7db7e5ff3f3905083656f9b6743d426b79acc053bb49124b33d92ae926a928cbe439ebea0e6563a4473b7760cd5bfc0440a63c30d7209765b"))),
            Reference(12495840442494484844UL,T(false,-591,UInt!512.fromHexString("d5d892bc4291a8b8e5bd87b74d5e9bbc3975f43ab56c9c50ca55a6eb34fe1cf742cf7c832c607e6ba3953d7ae118a7a26458f363180cc099f2a89f26caf5cd42")),T(true,-441,UInt!512.fromHexString("84a63768cf118fc8ce22bd3f461d1aaddb774303d44b4fe6c3c2b9b7e76deca00c4c00099edf705ad46771f3daec05d0afdfff8dfd650b1dd94f05b65c8ab271"))),
            Reference(14298266316695311713UL,T(false,-374,UInt!512.fromHexString("db9d798fa95a1ac12d1a21b78adec6171aaf78de6a57107aceda705028bc5e6893f5f48972e556c8ee728151a67249c1fdd2e5e262c17af7606bc3b659bb767c")),T(true,-374,UInt!512.fromHexString("db9d798fa95a1ac122f3006d4f78c813f4ef8c31dfa87e85aadc76ea7d76e3c23b1b09283fff78bc9305c94a4f74d235e50a03ff01a482e7e7cd43c20e8ad0c0"))),
            Reference(14578354418426048924UL,T(false,-451,UInt!512.fromHexString("27f6d3539a0691868ebd1c885617abb561b4336ee55e77f4f898723db819fb4896aace8a77ea3dfc0df65b3dc204801ee6c47759796b5c7bed6382c8303fc304")),T(true,-446,UInt!512.fromHexString("89e072bd08288ced9224b9e2024cc412372bf7335c1802287bb3ad2cdbad6dc00e0be738d21c3a692e90a67ec5a05df00f0ca89dddc5be953a753bcffd041c50"))),
            Reference(17722594396922235376UL,T(false,-321,UInt!512.fromHexString("4fc5fe35fd03c6f03840dc4a4b8ce2faccb5eee56d14e2411374a1c3a29787d12b090d5cf7f53c33d6e4ea267263a92f3e8d0f6d21d1e09fb3c2cff9bf8318a7")),T(true,-322,UInt!512.fromHexString("9f8bfc6bfa078de07081b8949719c4a3c991526fe9e7ba1400837e925d9b406b2b955040408d947fff4aae3c23817510715c93453f58107a60285ee8e5081a26"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("1")),T(true,-385,UInt!512.fromHexString("b17217f7d1cf7a01d17d978919e052bde2e27a22402df6959df24c1a9b8c99e571720cb2e2c5fe442d9df03ea20e43a24a6dca5821bc09fcd2fc57e0e256072d"))),
            Reference(18446744073709551615UL,T(false,-2000,UInt!512.fromHexString("1")),T(true,-437,UInt!512.fromHexString("b2b4fc27c173189e7185af291a33c098d3ba7fb8966604a6b1d475a6b95bdebb24f5569d7b2f7e3b35e4dcf4adbacfc200a55053c651b4550b86a04b37b633ee"))),
            Reference(18446744073709551615UL,T(false,66,UInt!512.fromHexString("1")),T(true,-447,UInt!512.fromHexString("ce8de8082e308660744faafe7dcef0537c349c9d1d40e632589d50a344c68f2bdcd90d77c6e8f550433ca390ec034366db703032268a4f786ce06b1040b5e67f"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!512.fromHexString("1")),T(true,9223372036854775295,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,0,UInt!512.fromHexString("ffffffffffffffff")),T(true,-507,UInt!512.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7bcaab31a0562ab4b79085ee0c1f21dab00f9521c95da0c0fa93e58bc1b03c4728"))),
            Reference(18446744073709551615UL,T(false,193,UInt!512.fromHexString("7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,192,UInt!512.fromHexString("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"))),
            Reference(18446744073709551615UL,T(false,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"))),
            Reference(18446744073709551615UL,T(false,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,193,UInt!512.fromHexString("80000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"))),
            Reference(18446744073709551615UL,T(false,-448,UInt!512.fromHexString("fffffffffffffffeffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-507,UInt!512.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7bcaab31a0562ab4b79085ee0c1f21dab00f9521c95da0c0fa93e58bc1b03c4728"))),
            Reference(18446744073709551615UL,T(false,-448,UInt!512.fromHexString("ffffffffffffffff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")),T(true,-507,UInt!512.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7bcaab31a0562ab4b79085ee0c1f21dab00f9521c95da0c0fa93e58bc1b03c4728"))),
            Reference(18446744073709551615UL,T(false,-448,UInt!512.fromHexString("ffffffffffffffff0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")),T(true,-507,UInt!512.fromHexString("b8cc1469eaff26d56b3442c2b8be6a5b08c61cf8308837dd4da846c3ca929a7bcaab31a0562ab4b79085ee0c1f21dab00f9521c95da0c0fa93e58bc1b03c4728"))),
            Reference(18446744073709551615UL,T(false,long.min,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,-385,UInt!512.fromHexString("b17217f7d1cf773c091db841dbf9a65c1c73f959b239e8b95fb41260048ea0c10a21384e99b604cd8022dcb59f4491ead6fa5ee3e802279e6d8f6e07ea3d3a44"))),
            Reference(18446744073709551615UL,T(false,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")),T(true,long.max-1,UInt!512.fromHexString("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"))),
        ];
        foreach (r; references)
            assert(PoissonLog!512.mass(r.k,r.rate) == r.value);
    }
}
// END Poisson log references

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    foreach (bits; AliasSeq!(64u,128u,256u,512u))
    {
        alias T = Fp!bits;
        alias P = PoissonLog!bits;
        static assert(is(typeof(P.mass(1,T(1))) == T));
        assert(P.mass(1,T(1)) == T(-1));
        debug
        {
            import core.exception: AssertError;
            import std.exception: assertThrown;
            assertThrown!AssertError(P.mass(0,T(1)));
            foreach (bad; [T(0),T(-1),T(double.infinity),T(double.nan)])
                assertThrown!AssertError(P.mass(1,bad));
        }
    }
}
