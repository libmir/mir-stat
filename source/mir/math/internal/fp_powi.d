/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.math.internal.fp_powi;

import mir.bignum.fp: Fp;

///
package(mir)
T fp_powi(T)(const T x, size_t i)
    if (is(T == Fp!size, size_t size))
{
    // Exponentiation by squaring keeps full-width counts practical. Avoid an
    // unused final square, which could overflow even when the result fits.
    T output = T(1);
    T factor = x;
    while (i) {
        if (i & 1)
            output *= factor;
        i >>= 1;
        if (i)
            factor *= factor;
    }
    return output;
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.conv: to;
    auto x = Fp!128(3.0);
    assert(x.fp_powi(0).to!double == 1);
    assert(x.fp_powi(1).to!double == 3);
    assert(x.fp_powi(2).to!double == 9);
    assert(x.fp_powi(5).to!double == 243);
    assert(Fp!128(1).fp_powi(size_t.max).to!double == 1);
    assert(Fp!128(0).fp_powi(0).to!double == 1);
    assert(Fp!128(0).fp_powi(17).to!double == 0);
}
