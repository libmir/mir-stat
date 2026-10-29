/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.fp_count_deviance;

import mir.bignum.fixed: UInt;

// Shared fixed-point series for x*log(x/mean)+mean-x near the mean.
// Inputs and output have fractionBits fractional bits. Callers provide
// integer headroom for the count and bound absolute truncation error in
// their final log mass; tiny deviances have no relative-error guarantee.
package(mir.stat) @safe pure nothrow @nogc
UInt!size fpCenteredDeviance(uint fractionBits, ulong size)(ulong x, UInt!size mean)
    in (x != 0, "count must be positive")
{
    alias I = UInt!size;
    static assert(size >= fractionBits + 128);
    const count = I(x) << fractionBits;
    const negative = count < mean;
    const delta = negative ? mean-count : count-mean;
    assert((delta << 1) < count, "mean must be within half the count");
    I multiply(I a, I b)
    {
        auto product = a.toSize!(2 * size);
        product *= b;
        return (product >> fractionBits).toSize!size;
    }
    // v=(x-mean)/(x+mean), |v|<1/3:
    // deviance=delta*v+2*x*(v^3/3+v^5/5+...).
    auto quotient = delta.toSize!(2 * size);
    quotient <<= fractionBits;
    quotient /= count + mean;
    const v = quotient.toSize!size;
    const square = multiply(v,v);
    I sum = multiply(delta,v);
    I term = v;
    term *= UInt!128(x) << 1;
    for (uint odd = 3; ; odd += 2)
    {
        term = multiply(term,square);
        auto next = term;
        next /= UInt!64(odd);
        if (!next) break;
        if (negative) sum -= next;
        else sum += next;
    }
    return sum;
}
