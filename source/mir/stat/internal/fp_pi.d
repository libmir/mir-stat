/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_pi;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;

// Machin's identity pi = 16 atan(1/5) - 4 atan(1/239). Keep 64 extra
// fractional bits through the integer divisions before narrowing.
package(mir.stat) @safe pure nothrow @nogc
Fp!bits guardedPi(uint bits)()
{
    enum piBits = bits + 64;
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
    auto result = Fp!bits(pi);
    result.exponent -= piBits;
    return result;
}
