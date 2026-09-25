/++
This module contains algorithms for traits used when dealing with histograms.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
MATHREF = $(REF_ALTTEXT $(TT $2), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(REF_ALTTEXT $(TT $2), $2, mir, ndslice, $1)$(NBSP)
T2=$(TR $(TDNW $(LREF $1)) $(TD $+))
T4=$(TR $(TDNW $(LREF $1)) $(TD $2) $(TD $3) $(TD $4))
+/

module mir.stat.descriptive.histogram.traits;

import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        RegularAxis, TransformAxis, EnumAxis, CategoryAxis;

alias DefaultCountType = size_t;

package
void checkUnderflow(BinType, AxisOptions axisOptions)(BinType x, BinType low)
{
    static if (!axisOptions.isRightClosed || axisOptions.isCircular) {
        assert(x >= low, "checkUnderflow: x must be greater than or equal to low");
    } else {
        assert(x > low, "checkUnderflow: x must greater than low");
    }
}

package
void checkOverflow(BinType, AxisOptions axisOptions)(BinType x, BinType high)
{
    import mir.internal.utility: isFloatingPoint;

    static if (axisOptions.isRightClosed || axisOptions.isCircular) {
        assert(x <= high, "checkUnderflow: x must be less than or equal to high");
    } else {
        assert(x < high, "checkUnderflow: x must be less than high");
    }
}

package
void checkOverUnderFlow(BinType, AxisOptions axisOptions)(BinType x, BinType low, BinType high) {
    checkUnderflow!(BinType, axisOptions)(x, low);
    checkOverflow!(BinType, axisOptions)(x, high);
}

package
template includeOverflow(AxisType)
    if (isAxis!(AxisType))
{
    import std.traits: hasMember;

    static if (hasMember!(AxisType, "isOverflow")) {
        static if (hasAxisOptions!AxisType)
            enum bool includeOverflow = AxisType.options.enableOverflow;
        else
            enum bool includeOverflow = true;
    } else {
        enum bool includeOverflow = false;
    }
}

package
template includeUnderflow(AxisType)
    if (isAxis!(AxisType))
{
    import std.traits: hasMember;

    static if (hasMember!(AxisType, "isUnderflow")) {
        static if (hasAxisOptions!AxisType)
            enum bool includeUnderflow = AxisType.options.enableUnderflow;
        else
            enum bool includeUnderflow = true;
    } else {
        enum bool includeUnderflow = false;
    }
}

/++
Number of ordinary bins as a storage index. The axis contract requires an integral
N_bin; validates positivity and representability as size_t before conversion.
Counter precision does not determine the type used for storage shapes or traversal.
Params:
    axis = axis defining the ordinary bins
+/
size_t ordinaryBinCount(A)(auto ref const A axis)
    if (isAxis!A)
{
    return checkedBinCount(axis.N_bin);
}

// Preserve the signed input until it has been checked, before storing an index.
package size_t checkedBinCount(T)(T count)
    if (isIntegralBinCount!T)
{
    assert(count > 0 && count <= size_t.max,
        "Histogram: ordinary bin count is out of range");
    return cast(size_t) count;
}

package template isIntegralBinCount(T)
{
    import std.traits: isIntegral, Unqual;
    enum isIntegralBinCount = isIntegral!T && !is(Unqual!T == bool);
}

/++
Number of storage positions required by an axis, including enabled
underflow and overflow bins. N_bin continues to count only ordinary bins.
Params:
    axis = axis defining the ordinary bins and enabled underflow/overflow bins
+/
size_t storageExtent(A)(auto ref const A axis)
    if (isAxis!A)
{
    enum size_t extra = includeUnderflow!A + includeOverflow!A;
    const count = ordinaryBinCount(axis);
    assert(count <= size_t.max - extra,
        "Histogram: storage extent is out of range");
    return count + extra;
}

// Checks whether type `T` can be used in a switch statement. This is useful for
// compile-time generation of switch case statements.
package
template isSwitchable(E)
{
    enum bool isSwitchable = is(typeof({
        switch (E.init) { default: }
    }));
}

/++
Detect whether a type is an `Axis`. An `Axis` type must have `BinType`,
`N_bin`, and callable `index` members. Bin counts and indices must be integral
(non-bool) types; no counter-storage type is required.

Params:
    T = type
Returns:
    `true` if `T` is an `Axis` type, `false` otherwise
+/
template isAxis(T)
{
    import std.traits: hasMember;

    static if (hasMember!(T, "index") &&
               hasMember!(T, "BinType") &&
               hasMember!(T, "N_bin")) {
        import std.traits: isIntegral, Unqual;
        static if (is(typeof(() { auto count = T.init.N_bin; return count; }()) Size) &&
            is(typeof(T.init.index(T.BinType.init)) Index))
            enum bool isAxis = isIntegral!Size && !is(Unqual!Size == bool) &&
                isIntegral!Index && !is(Unqual!Index == bool);
        else
            enum bool isAxis = false;
    } else {
        enum bool isAxis = false;
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct FooAxis
    {
        size_t index(size_t value) const @safe pure nothrow @nogc;
        alias BinType = size_t;
        size_t N_bin = 2;
    }
    static assert(isAxis!FooAxis);
}

/++
Detect whether an axis type has options (and such options must be of type
`AxisOptions`).

Params:
    T = type
Returns:
    `true` if `T` has an `options` member of type `AxisOptions`, `false` otherwise
+/
template hasAxisOptions(T)
    if (isAxis!T)
{
    import std.traits: hasMember;
    import mir.stat.descriptive.histogram.axis: AxisOptions;

    static if (hasMember!(T, "options") && is(typeof(T.options) == AxisOptions)) {
        enum bool hasAxisOptions = true;
    }  else {
        enum bool hasAxisOptions = false;
    }
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions;

    struct FooAxis(AxisOptions axisOptions)
    {
        size_t index(size_t value) const @safe pure nothrow @nogc;
        alias BinType = size_t;
        size_t N_bin = 2;
        alias options = axisOptions;
    }

    static assert(hasAxisOptions!(FooAxis!(AxisOptions())));
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct FooAxis
    {
        size_t index(size_t value) const @safe pure nothrow @nogc;
        alias BinType = size_t;
        size_t N_bin = 2;
    }
    static assert(!hasAxisOptions!FooAxis);
}


/// Test if type is an integral axis
enum bool isIntegralAxis(T) = is(T : IntegralAxis!(BinType, axisOptions), BinType, AxisOptions axisOptions);

/// Test if type is an regular axis
enum bool isRegularAxis(T) = is(T : RegularAxis!(BinType, axisOptions), BinType, AxisOptions axisOptions);
enum bool isTransformAxis(T) = is(T : TransformAxis!(BinType, transform, inverseTransform, axisOptions), BinType, alias transform, alias inverseTransform, AxisOptions axisOptions);

/// Test if type is an enum axis
enum bool isEnumAxis(T) = is(T : EnumAxis!(BinType), BinType);

/// Test if type is a category axis
enum bool isCategoryAxis(T) = is(T : CategoryAxis!(BinType, axisOptions), BinType, AxisOptions axisOptions);

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.axis: AxisOptions, IntegralAxis,
        RegularAxis, TransformAxis, EnumAxis, CategoryAxis, inverseTransformMapping;
    import mir.math.common: log10;

    enum Foo
    {
        A,
        B
    }

    IntegralAxis!(double, AxisOptions()) integralAxis;
    RegularAxis!(double, AxisOptions()) regularAxis;
    TransformAxis!(double, log10, inverseTransformMapping!log10, AxisOptions()) transformAxis;
    EnumAxis!(Foo) enumAxis;
    CategoryAxis!(Foo, AxisOptions()) categoryAxis;

    static assert(isIntegralAxis!(typeof(integralAxis)));
    static assert(!isIntegralAxis!(typeof(regularAxis)));
    static assert(!isIntegralAxis!(typeof(transformAxis)));
    static assert(!isIntegralAxis!(typeof(enumAxis)));
    static assert(!isIntegralAxis!(typeof(categoryAxis)));

    static assert(!isRegularAxis!(typeof(integralAxis)));
    static assert(isRegularAxis!(typeof(regularAxis)));
    static assert(!isRegularAxis!(typeof(transformAxis)));
    static assert(!isRegularAxis!(typeof(enumAxis)));
    static assert(!isRegularAxis!(typeof(categoryAxis)));

    static assert(!isTransformAxis!(typeof(integralAxis)));
    static assert(!isTransformAxis!(typeof(regularAxis)));
    static assert(!isTransformAxis!(typeof(enumAxis)));
    static assert(isTransformAxis!(typeof(transformAxis)));
    static assert(!isTransformAxis!(typeof(categoryAxis)));

    static assert(!isEnumAxis!(typeof(integralAxis)));
    static assert(!isEnumAxis!(typeof(regularAxis)));
    static assert(!isEnumAxis!(typeof(transformAxis)));
    static assert(isEnumAxis!(typeof(enumAxis)));
    static assert(!isEnumAxis!(typeof(categoryAxis)));

    static assert(!isCategoryAxis!(typeof(integralAxis)));
    static assert(!isCategoryAxis!(typeof(regularAxis)));
    static assert(!isCategoryAxis!(typeof(transformAxis)));
    static assert(!isCategoryAxis!(typeof(enumAxis)));
    static assert(isCategoryAxis!(typeof(categoryAxis)));
}

/++
Get the `BinType` of an `Axis` type.

Params:
    T = type
+/
template BinTypeOf(T)
    if (isAxis!T)
{
    alias BinTypeOf = T.BinType;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct FooAxis
    {
        size_t index(size_t value) const @safe pure nothrow @nogc;
        alias BinType = size_t;
        size_t N_bin = 2;
    }
    static assert(is(BinTypeOf!FooAxis == size_t));
}


/++
CHeck if `breakFunction` is `sturges`, `scott`, or `freedmanDiaconis`.

Params:
    breakFunction = function
+/
template isBreakFunction(alias breakFunction)
{
    import mir.stat.descriptive.histogram.breaks: sturges, scott, freedmanDiaconis;
    import std.traits: isInstanceOf, TemplateOf;

    static if (!isInstanceOf!(sturges, breakFunction) &&
               !isInstanceOf!(scott, breakFunction) &&
               !isInstanceOf!(freedmanDiaconis, breakFunction)) {
        enum bool isBreakFunction = __traits(isSame, breakFunction, sturges) ||
                                    __traits(isSame, breakFunction, scott) ||
                                    __traits(isSame, breakFunction, freedmanDiaconis);
    } else {
        enum bool isBreakFunction = __traits(isSame, TemplateOf!breakFunction, sturges) ||
                                    __traits(isSame, TemplateOf!breakFunction, scott) ||
                                    __traits(isSame, TemplateOf!breakFunction, freedmanDiaconis);
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.breaks: sturges, scott, freedmanDiaconis;

    static assert(isBreakFunction!sturges);
    static assert(isBreakFunction!scott);
    static assert(isBreakFunction!freedmanDiaconis);
    static assert(isBreakFunction!(sturges!uint));
    static assert(isBreakFunction!(scott!uint));
    static assert(isBreakFunction!(freedmanDiaconis!uint));
}

// Match observation types without discarding qualifiers on referenced data.
package template acceptsAxisValue(Axis, T)
{
    import std.traits: Unqual, isSomeString;
    enum acceptsAxisValue = is(Unqual!T == Unqual!(BinTypeOf!Axis)) ||
        (isCategoryAxis!Axis && isSomeString!T);
}

// Test the rule against the actual observation view, rather than its name.
package template acceptsBreakFunction(alias rule, S)
{
    import std.traits: isIntegral, Unqual;
    static if (is(typeof(rule(S.init.lightScope)) Result))
        enum acceptsBreakFunction = isIntegral!Result && !is(Unqual!Result == bool);
    else
        enum acceptsBreakFunction = false;
}

// Validate before narrowing so a large rule result cannot wrap into a valid count.
package size_t checkedBreakCount(alias rule, S)(S observations)
    if (acceptsBreakFunction!(rule, S))
{
    return checkedBinCount(rule(observations.lightScope));
}

// Rule results are checked against index capacity, independently of counters.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    static uint large(S)(S data) { return (1u << 24) + 1; }
    static int negative(S)(S data) { return -1; }
    static uint zero(S)(S data) { return 0; }
    double[1] values = [1];
    auto data = values[].sliced;
    assert(checkedBreakCount!large(data) == (1u << 24) + 1);
    assertThrown!AssertError(checkedBreakCount!negative(data));
    assertThrown!AssertError(checkedBreakCount!zero(data));
}

version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct Axis(Size, Index = size_t)
    {
        alias BinType = double;
        Size N_bin;
        Index index(double value) const @safe pure nothrow @nogc;
    }
    static assert(isAxis!(Axis!uint));
    static assert(isAxis!(Axis!size_t));
    static assert(!isAxis!(Axis!double));
    static assert(!isAxis!(Axis!bool));
    static assert(!isAxis!(Axis!(size_t, double)));
    static assert(!isAxis!(Axis!(size_t, bool)));
    assert(ordinaryBinCount(Axis!uint(3)) == 3);
    assert(storageExtent(Axis!uint(3)) == 3);
    assert(ordinaryBinCount(Axis!size_t(size_t.max)) == size_t.max);
}

version(mir_stat_test)
@system pure nothrow @nogc
unittest
{
    import core.exception: AssertError;
    static void rejects(scope void delegate() pure nothrow @nogc operation) pure nothrow @nogc
    {
        bool rejected;
        try { operation(); } catch (AssertError) { rejected = true; }
        assert(rejected);
    }
    struct Axis(Size)
    {
        alias BinType = double;
        Size N_bin;
        size_t index(double value) const @safe pure nothrow @nogc;
        enum bool isOverflow = false;
    }
    rejects(() { cast(void) ordinaryBinCount(Axis!int(0)); });
    rejects(() { cast(void) ordinaryBinCount(Axis!int(-1)); });
    rejects(() { cast(void) storageExtent(Axis!size_t(size_t.max)); });
    assert(storageExtent(Axis!size_t(size_t.max - 1)) == size_t.max);
    static if (size_t.sizeof < ulong.sizeof)
        rejects(() { cast(void) ordinaryBinCount(Axis!ulong(ulong.max)); });
}
