/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.uint_sqrt;

import mir.bignum.fixed: UInt;

/++
Computes the integer square root and remainder.

Params:
    bits = Width of the input and outputs in bits. Any width supported by
        $(D UInt) on the target is accepted.
    value = Unsigned radicand.
    remainder = Receives $(D value - root * root), between zero and
        $(D 2 * root) inclusive. Zero identifies a perfect square.

Returns:
    The greatest $(D UInt!bits) integer $(D root) whose square is at most
    $(D value). For a zero input both outputs are zero.
+/
package(mir.stat) @safe pure nothrow @nogc
UInt!bits isqrtRem(size_t bits)(UInt!bits value, out UInt!bits remainder)
{
    UInt!bits root;
    const significantBits = bits - value.ctlz;
    for (size_t pairs = (significantBits + 1) / 2; pairs; --pairs)
    {
        // Append two bits to the processed prefix, so its root grows
        // from r to either 2r or 2r+1. The squares differ by 4r+1.
        const bit = 2 * (pairs - 1);
        remainder <<= 2;
        remainder |= (cast(size_t)value.bt(bit + 1) << 1) | value.bt(bit);
        root <<= 1;
        auto trial = root << 1;
        trial |= 1u;
        if (remainder >= trial)
        {
            remainder -= trial;
            root += 1u;
        }
    }
    return root;
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    // Sweep small inputs with an independent oracle based on successive
    // squares. Reuse the output to exercise zero and perfect squares.
    alias U = UInt!(size_t.sizeof * 8);
    U remainder = U.max;
    size_t expected;
    foreach (size_t n; 0 .. 256)
    {
        if (n == (expected + 1) * (expected + 1))
            ++expected;
        assert(isqrtRem(U(n), remainder) == U(expected));
        assert(remainder == U(n - expected * expected));
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    alias U = UInt!192;
    enum root = U.fromHexString("800000000123456789abcdef");
    const square = root * root;
    U remainder;
    assert(isqrtRem(square - 1, remainder) == root - 1);
    assert(remainder == (root - 1) * 2);
    assert(isqrtRem(square, remainder) == root);
    assert(remainder == U(0));
    assert(isqrtRem(square + 1, remainder) == root);
    assert(remainder == U(1));

    const largestRoot = (U(1) << 96) - 1;
    assert(isqrtRem(U.max, remainder) == largestRoot);
    assert(remainder == largestRoot * 2);
}
