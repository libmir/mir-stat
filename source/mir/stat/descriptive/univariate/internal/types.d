/++
Univariate types and associated algorithms.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall, Ilya Yaroshenko
Copyright: 2022-3 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
MATHREF_ALT = $(GREF_ALTTEXT mir-algorithm, $(B $(TT $2)), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T3=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.univariate.internal.types;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;


///
package(mir)
template statType(T, bool checkComplex = true)
{
    import mir.internal.utility: isFloatingPoint;

    static if (isFloatingPoint!T) {
        import std.traits: Unqual;
        alias statType = Unqual!T;
    } else static if (is(T : double)) {
        alias statType = double;
    } else static if (checkComplex) {
        import mir.internal.utility: isComplex;
        static if (isComplex!T) {
            static if (__traits(getAliasThis, T).length == 1)
            {
                alias statType = .statType!(typeof(__traits(getMember, T, __traits(getAliasThis, T)[0])));
            }
            else
            {
                import std.traits: Unqual;
                alias statType = Unqual!T;
            }
        } else {
            static assert(0, "statType: type " ~ T.stringof ~ " must be convertible to a complex floating point type");
        }
    } else {
        static assert(0, "statType: type " ~ T.stringof ~ " must be convertible to a floating point type");
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static assert(is(statType!int == double));
    static assert(is(statType!uint == double));
    static assert(is(statType!double == double));
    static assert(is(statType!float == float));
    static assert(is(statType!real == real));

    static assert(is(statType!(const(int)) == double));
    static assert(is(statType!(immutable(int)) == double));
    static assert(is(statType!(const(double)) == double));
    static assert(is(statType!(immutable(double)) == double));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex: Complex;

    static assert(is(statType!(Complex!float) == Complex!float));
    static assert(is(statType!(Complex!double) == Complex!double));
    static assert(is(statType!(Complex!real) == Complex!real));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    static assert(is(statType!Foo == double)); // note: this is not float
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static struct Foo {
        Complex!float x;
        alias x this;
    }

    static assert(is(statType!Foo == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        double x;
        alias x this;
    }

    static assert(is(statType!Foo == double));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static struct Foo {
        Complex!double x;
        alias x this;
    }

    static assert(is(statType!Foo == Complex!double));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        real x;
        alias x this;
    }

    static assert(is(statType!Foo == double)); // note: this is not real
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static struct Foo {
        Complex!real x;
        alias x this;
    }

    static assert(is(statType!Foo == Complex!real));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        int x;
        alias x this;
    }

    static assert(is(statType!Foo == double)); // note: this is not ints
}

///
package(mir)
template meanType(T)
{
    import mir.math.sum: sumType;

    alias U = sumType!T;

    static if (__traits(compiles, {
        auto temp = U.init + U.init;
        auto a = temp / 2;
        temp += U.init;
    })) {
        alias V = typeof((U.init + U.init) / 2);
        alias meanType = statType!V;
    } else {
        static assert(0, "meanType: Can't calculate mean of elements of type " ~ U.stringof);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static assert(is(meanType!(int[]) == double));
    static assert(is(meanType!(double[]) == double));
    static assert(is(meanType!(float[]) == float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static assert(is(meanType!(Complex!float[]) == Complex!float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    static assert(is(meanType!(Foo[]) == float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.complex;
    static struct Foo {
        Complex!float x;
        alias x this;
    }

    static assert(is(meanType!(Foo[]) == Complex!float));
}

///
package(mir)
template stdevType(T)
{
    import mir.internal.utility: isFloatingPoint;

    alias U = meanType!T;

    static if (isFloatingPoint!U) {
        alias stdevType = U;
    } else {
        static assert(0, "stdevType: Can't calculate standard deviation of elements of type " ~ U.stringof);
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static assert(is(stdevType!(int[]) == double));
    static assert(is(stdevType!(double[]) == double));
    static assert(is(stdevType!(float[]) == float));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    static struct Foo {
        float x;
        alias x this;
    }

    static assert(is(stdevType!(Foo[]) == float));
}
