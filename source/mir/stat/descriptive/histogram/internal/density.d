/++
Internal numeric bin geometry for histogram densities.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.density;

import std.traits: isIntegral, isFloatingPoint;

package(mir.stat.descriptive.histogram) template supportsDensityAxis(Axis)
{
    alias ConstAxis = const(Axis);
    static if (__traits(compiles, ConstAxis.init.bin(size_t.init)))
    {
        alias B = typeof(ConstAxis.init.bin(size_t.init));
        static if (__traits(hasMember, B, "low") && __traits(hasMember, B, "high") &&
            __traits(compiles, B.init.low + 0, B.init.high + 0))
            enum supportsDensityAxis =
                (isIntegral!(typeof(B.init.low + 0)) || isFloatingPoint!(typeof(B.init.low + 0))) &&
                (isIntegral!(typeof(B.init.high + 0)) || isFloatingPoint!(typeof(B.init.high + 0)));
        else
            enum supportsDensityAxis = false;
    }
    else
        enum supportsDensityAxis = false;
}

// Keep volume as a normalized mantissa and exponent: multiplying widths can
// overflow or underflow even when the final density is representable.
package(mir.stat.descriptive.histogram) struct ScaledBinVolume
{
    private real mantissa = 0.5L;
    private int exponent = 1;

    void include(Bin)(Bin bin)
    {
        import std.math: isFinite, frexp;
        const low = bin.low;
        const high = bin.high;
        assert(high > low, "density: bin boundaries must increase");
        real width;
        static if (isIntegral!(typeof(low)) && isIntegral!(typeof(high)))
        {
            // Unsigned subtraction preserves small differences near integer
            // limits and also represents widths spanning the signed range.
            width = cast(real)(cast(ulong) high - cast(ulong) low);
        }
        else
        {
            assert(isFinite(cast(real) low) && isFinite(cast(real) high),
                "density: bin boundaries must be finite");
            width = cast(real) high - cast(real) low;
        }
        assert(width > 0 && isFinite(width), "density: bin width must be positive and finite");
        int power;
        mantissa *= frexp(width, power);
        exponent += power;
        mantissa = frexp(mantissa, power);
        exponent += power;
    }

    T normalize(T)(real relativeFrequency) const
        if (isFloatingPoint!T)
    {
        import std.math: ldexp;
        return cast(T) ldexp(relativeFrequency / mantissa, -exponent);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: Bin;
    import mir.math.common: approxEqual;
    ScaledBinVolume large;
    large.include(Bin!double(0, 1e300));
    large.include(Bin!double(0, 1e300));
    large.include(Bin!double(0, 1e-300));
    assert(approxEqual(large.normalize!double(1), 1e-300, 1e-14, 0));
    ScaledBinVolume tiny;
    tiny.include(Bin!double(0, 1e-300));
    tiny.include(Bin!double(0, 1e-300));
    tiny.include(Bin!double(0, 1e300));
    assert(approxEqual(tiny.normalize!double(1), 1e300, 1e-14, 0));
    ScaledBinVolume adjacent;
    adjacent.include(Bin!ulong(ulong.max - 1, ulong.max));
    assert(adjacent.normalize!float(0.5) == 0.5f);
    ScaledBinVolume full;
    full.include(Bin!long(long.min, long.max));
    assert(approxEqual(full.normalize!real(1), 1 / cast(real) ulong.max, 1e-18L, 0));
}

version(mir_stat_test)
@system pure
unittest
{
    import mir.stat.descriptive.histogram.axis: Bin;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    ScaledBinVolume volume;
    assertThrown!AssertError(volume.include(Bin!double(0, double.infinity)));
    assertThrown!AssertError(volume.include(Bin!double(1, 1)));
    assertThrown!AssertError(volume.include(Bin!double(2, 1)));
    assertThrown!AssertError(volume.include(Bin!double(double.nan, 1)));
}
