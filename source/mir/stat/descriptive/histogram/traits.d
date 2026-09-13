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
Detect whether a type is an `Axis`. An `Axis` type must have `index`, `BinType`,
`CountType`, and `N_bin` members.

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
               hasMember!(T, "CountType") &&
               hasMember!(T, "N_bin")) {
        enum bool isAxis = true;
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
        size_t index = 1;
        alias BinType = size_t;
        alias CountType = double;
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
        size_t index = 1;
        alias BinType = size_t;
        alias CountType = double;
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
        size_t index = 1;
        alias BinType = size_t;
        alias CountType = double;
        size_t N_bin = 2;
    }
    static assert(!hasAxisOptions!FooAxis);
}


/// Test if type is an integral axis
enum bool isIntegralAxis(T) = is(T : IntegralAxis!(CountType, BinType, axisOptions), CountType, BinType, AxisOptions axisOptions);

/// Test if type is an regular axis
enum bool isRegularAxis(T) = is(T : RegularAxis!(CountType, BinType, axisOptions), CountType, BinType, AxisOptions axisOptions);
enum bool isTransformAxis(T) = is(T : TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions), CountType, BinType, alias transform, alias inverseTransform, AxisOptions axisOptions);

/// Test if type is an enum axis
enum bool isEnumAxis(T) = is(T : EnumAxis!(CountType, BinType), CountType, BinType);

/// Test if type is a category axis
enum bool isCategoryAxis(T) = is(T : CategoryAxis!(CountType, BinType, axisOptions), CountType, BinType, AxisOptions axisOptions);

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

    IntegralAxis!(size_t, double, AxisOptions()) integralAxis;
    RegularAxis!(size_t, double, AxisOptions()) regularAxis;
    TransformAxis!(size_t, double, log10, inverseTransformMapping!log10, AxisOptions()) transformAxis;
    EnumAxis!(size_t, Foo) enumAxis;
    CategoryAxis!(size_t, Foo, AxisOptions()) categoryAxis;

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
        size_t index = 1;
        alias BinType = size_t;
        alias CountType = double;
        size_t N_bin = 2;
    }
    static assert(is(BinTypeOf!FooAxis == size_t));
}

/++
Get the `CountType` of an `Axis` type.

Params:
    T = type
+/
template CountTypeOf(T)
    if (isAxis!T)
{
    alias CountTypeOf = T.CountType;
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    struct FooAxis
    {
        size_t index = 1;
        alias BinType = size_t;
        alias CountType = double;
        size_t N_bin = 2;
    }
    static assert(is(CountTypeOf!FooAxis == double));
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
