/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.internal.fp_probability;

import mir.bignum.fp: Fp;

// Validate the represented value, including probabilities that round to 0 or
// 1 in a hardware type. Fp's special exponent denotes infinity or NaN.
package(mir.stat) @safe pure nothrow @nogc
bool fpProbabilityValid(uint bits)(Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    if (p.exponent == long.max)
        return false;
    if (!p.coefficient)
        return true;
    if (p.sign || p.exponent > 0)
        return false;
    if (p.exponent <= -cast(long)bits)
        return true;
    UInt!bits one = UInt!bits(1);
    one <<= cast(size_t)-p.exponent;
    return p.coefficient <= one;
}

// Fp does not supply subtraction. Form 1-p exactly as an integer at p's
// binary exponent, then let Fp's integer constructor round to its precision.
// The temporary accommodates cancellation near one and guard bits near zero.
package(mir.stat) @safe pure nothrow @nogc
Fp!bits fpProbabilityComplement(uint bits)(Fp!bits p)
{
    import mir.bignum.fixed: UInt;
    if (!p.coefficient || p.exponent < -2 * cast(long)bits)
        return Fp!bits(1); // p is less than half the spacing immediately below 1.
    UInt!(2 * bits + 64) difference = UInt!(2 * bits + 64)(1);
    difference <<= cast(size_t)-p.exponent;
    difference -= p.coefficient;
    auto result = Fp!bits(difference);
    result.exponent += p.exponent;
    return result;
}
