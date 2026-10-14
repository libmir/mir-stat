/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022 Mir Stat Authors.

+/

module mir.math.internal.powi;

import mir.internal.utility: isFloatingPoint;
import std.traits: isIntegral, isSigned, Unqual;

// Keep the core implementation for int exponents. For wider exponents,
// preserve every integer bit rather than converting through floating point.
private T floatingPowi(T, U)(T x, U i)
{
    import mir.math.common: corePowi = powi;

    static if (isSigned!U)
    {
        if (i >= int.min && i <= int.max)
            return corePowi(x, cast(int) i);
    }
    else
    {
        if (i <= int.max)
            return corePowi(x, cast(int) i);
    }

    ulong magnitude = cast(ulong) i;
    static if (isSigned!U)
    {
        if (i < 0)
        {
            // Unsigned subtraction also handles long.min without overflow.
            magnitude = 0UL - magnitude;
            // Invert first so a negative power does not overflow its positive
            // intermediate before taking the reciprocal.
            x = 1 / x;
        }
    }

    T result = 1;
    while (magnitude)
    {
        if (magnitude & 1)
            result *= x;
        magnitude >>= 1;
        if (magnitude)
            x *= x;
    }
    return result;
}

///
package(mir)
Unqual!T powi(T, U)(T x, U i)
    if (isFloatingPoint!(Unqual!T) && isIntegral!(Unqual!U))
{
    return floatingPowi(cast(Unqual!T) x, cast(Unqual!U) i);
}

///
package(mir)
double powi(T, U)(T x, U i)
    if (!isFloatingPoint!(Unqual!T) && is(Unqual!T : double)
        && isIntegral!(Unqual!U))
{
    return floatingPowi(cast(double) x, cast(Unqual!U) i);
}

///
package(mir)
Unqual!T powi(T)(T x, size_t i)
    if (!isFloatingPoint!(Unqual!(T)) && !is(Unqual!(T) : double))
{
    if (i == 0) {
        return cast(Unqual!T) 1;
    } else if (i == 1) {
        return cast(Unqual!T) x;
    } else {
        auto output = cast(Unqual!T) x;
        for (size_t j = 1; j < i; j++) {
            output *= x;
        }
        return output;
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    alias C = Complex!double;
    auto x = C(1, 2);
    assert(x.powi(0) == 1);
    assert(x.powi(1) == x);
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.complex: Complex;
    Complex!double x = Complex!double(1.0, 2.0);
    assert(x.powi(0) == 1);
    assert(x.powi(1) == x);
}

// Full-width exponents must behave identically with assertions disabled.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: signbit, isNaN;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        assert(powi(T(2), 3L) == 8);
        assert(powi(T(2), -3L) == T(0.125));
        assert(powi(T(1), cast(uint) int.max) == 1);
        assert(powi(T(0.5), uint.max) == 0);
        assert(powi(T(2), long.min) == 0);
        assert(powi(T(-1), long.min) == 1);
        assert(powi(T(-1), long.min + 1) == -1);
        assert(powi(T(-1), long.max) == -1);
        assert(powi(T(-1), ulong.max) == -1);
        assert(powi(T(-1), ulong.max - 1) == 1);
        assert(powi(T(2), ulong.max) == T.infinity);
        assert(powi(-T(0), ulong.max) == 0);
        assert(signbit(powi(-T(0), ulong.max)));
        assert(powi(-T(0), long.min + 1) == -T.infinity);
        assert(signbit(powi(-T.infinity, long.min + 1)));
        assert(isNaN(powi(T.nan, ulong.max)));
        static assert(powi(T(-1), ulong.max) == -1);
        static assert(powi(T(2), long.min) == 0);
        static assert(is(typeof(powi(T(2), ulong.max)) == T));
    }}
    // isSigned also includes floating-point types; do not truncate them.
    static assert(!__traits(compiles, powi(2.0, 0.5)));
    static assert(!__traits(compiles, powi(2, 0.5)));
    static foreach (U; AliasSeq!(byte, ubyte, short, ushort, int, uint, long, ulong))
        assert(powi(2.0, U(3)) == 8);
    const long negative = -3;
    immutable ulong positive = ulong.max;
    assert(powi(2, negative) == 0.125);
    assert(powi(-1, positive) == -1);
    static assert(is(typeof(powi(2, positive)) == double));
}
