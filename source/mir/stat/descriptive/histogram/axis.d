/++
This module contains algorithms for histogram axes.

Bin-count rules are supplied as template aliases, such as functions or function
templates. Runtime callbacks that capture local variables are not supported.

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

module mir.stat.descriptive.histogram.axis;

import mir.functional: naryFun;
import mir.stat.descriptive.histogram.traits: DefaultCountType, isBreakFunction, acceptsBreakFunction, checkedBreakCount;
import mir.ndslice.slice: isSlice;
import mir.ndslice.traits: isContiguousVector;
import std.meta: NoDuplicates;
import std.traits: EnumMembers, isFunction;

///
struct IsRightClosed
{
    bool isRightClosed = false;
}

///
struct EnableOverflow
{
    bool enableOverflow = false;
}

///
struct EnableUnderflow
{
    bool enableUnderflow = false;
}

///
struct IsCircular
{
    bool isCircular = false;
}

/++
Options to provide to an `AxisType`.

+/
struct AxisOptions
{

private:

    /++
    Axis breaks are assumed to be non-overlapping. If `isRightClosed` equals
    `false` (default), then calculations assume the axis is left-closed and
    right-open, as in `[a, b)` or `a <= x < b`; otherwise, if `isRightClosed`
    equals `true`, then the calculations assume the axis is left-open and
    right-closed, as in `(a, b]` or `a < x <= b`.
    +/
    IsRightClosed value_isRightClosed = IsRightClosed();

    ///
    EnableOverflow value_enableOverflow = EnableOverflow();

    ///
    EnableUnderflow value_enableUnderflow = EnableUnderflow();

    /++
    Breaks are assumed to not wrap-around by default. If `isCircular` equals
    `true`, then the axis is circular and will wrap around. For instance, if the
    breaks are `[a, b)` and `[b, c)` then a value of `x = c` will be placed
    into the first break instead of overflow (assuming it is enabled). One
    use-case of circular breaks is data in polar coordinates.
    +/
    IsCircular value_isCircular = IsCircular();

public:

    ///
    @safe pure nothrow @nogc
    bool isRightClosed() const
    {
        return value_isRightClosed.isRightClosed;
    }

    ///
    @safe pure nothrow @nogc
    bool enableOverflow() const
    {
        return value_enableOverflow.enableOverflow;
    }

    ///
    @safe pure nothrow @nogc
    bool enableUnderflow() const
    {
        return value_enableUnderflow.enableUnderflow;
    }

    ///
    @safe pure nothrow @nogc
    bool isCircular() const
    {
        return value_isCircular.isCircular;
    }

    ///
    @safe pure nothrow @nogc
    this(bool x) {
        value_isRightClosed = IsRightClosed(x);
    }

    ///
    @safe pure nothrow @nogc
    this(bool x, bool y) {
        value_isRightClosed = IsRightClosed(x);
        value_enableOverflow = EnableOverflow(y);
    }

    ///
    @safe pure nothrow @nogc
    this(bool x, bool y, bool z) {
        value_isRightClosed = IsRightClosed(x);
        value_enableOverflow = EnableOverflow(y);
        value_enableUnderflow = EnableUnderflow(z);
    }

    ///
    @safe pure nothrow @nogc
    this(bool w, bool x, bool y, bool z) {
        value_isRightClosed = IsRightClosed(w);
        value_enableOverflow = EnableOverflow(x);
        value_enableUnderflow = EnableUnderflow(y);
        value_isCircular = IsCircular(z);
    }

    @safe pure nothrow @nogc
    void set(Arg)(Arg arg) {
        static if (is(Arg == IsRightClosed)) {
            this.value_isRightClosed = arg;
        } else static if (is(Arg == EnableOverflow)) {
            this.value_enableOverflow = arg;
        } else static if (is(Arg == EnableUnderflow)) {
            this.value_enableUnderflow = arg;
        } else static if (is(Arg == IsCircular)) {
            this.value_isCircular = arg;
        } else {
            static assert(0, "AxisOptions.set: option not supported");
        }
    }

    ///
    @safe pure nothrow @nogc
    void set(Arg)(bool value) {
        set(Arg(value));
    }

    @safe pure nothrow @nogc
    Arg get(Arg)() const
    {
        static if (is(Arg == IsRightClosed)) {
            return value_isRightClosed;
        } else static if (is(Arg == EnableOverflow)) {
            return value_enableOverflow;
        } else static if (is(Arg == EnableUnderflow)) {
            return value_enableUnderflow;
        } else static if (is(Arg == IsCircular)) {
            return value_isCircular;
        } else {
            static assert(0, "AxisOptions.get: option not supported");
        }
    }

    ///
    @safe pure nothrow @nogc
    this(Arg)(Arg arg) {
        set(arg);
    }

    ///
    @safe pure nothrow @nogc
    this(Args...)(Args args) {
        foreach (arg; args)
        {
            set(arg);
        }
    }
}


// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    AxisOptions x1 = AxisOptions(IsRightClosed(true));
    assert(x1.isRightClosed == true);
    AxisOptions x2 = AxisOptions(EnableOverflow(true));
    assert(x2.enableOverflow == true);
    AxisOptions x3 = AxisOptions(EnableUnderflow(true));
    assert(x3.enableUnderflow == true);
    AxisOptions x4 = AxisOptions(IsCircular(true));
    assert(x4.isCircular == true);

    AxisOptions x5 = AxisOptions(IsRightClosed(true), IsCircular(true));
    assert(x5.isRightClosed == true);
    assert(x5.isCircular == true);

    x5.set!EnableOverflow(true);
    x5.set!EnableUnderflow(true);
    assert(x5.get!EnableOverflow == EnableOverflow(true));
    assert(x5.get!EnableUnderflow == EnableUnderflow(true));

    AxisOptions x6 = AxisOptions(true, true, true, true);
    assert(x6.isRightClosed == true);
    assert(x6.enableOverflow == true);
    assert(x6.enableUnderflow == true);
    assert(x6.isCircular == true);
}

// Complete checks
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    AxisOptions x1 = AxisOptions(IsRightClosed(true));
    assert(x1.isRightClosed == true);
    assert(x1.enableOverflow == false);
    assert(x1.enableUnderflow == false);
    assert(x1.isCircular == false);
    AxisOptions x2 = AxisOptions(EnableOverflow(true));
    assert(x2.isRightClosed == false);
    assert(x2.enableOverflow == true);
    assert(x2.enableUnderflow == false);
    assert(x2.isCircular == false);
    AxisOptions x3 = AxisOptions(EnableUnderflow(true));
    assert(x3.isRightClosed == false);
    assert(x3.enableOverflow == false);
    assert(x3.enableUnderflow == true);
    assert(x3.isCircular == false);
    AxisOptions x4 = AxisOptions(IsCircular(true));
    assert(x4.isRightClosed == false);
    assert(x4.enableOverflow == false);
    assert(x4.enableUnderflow == false);
    assert(x4.isCircular == true);

    AxisOptions x5 = AxisOptions(IsRightClosed(true), IsCircular(true));
    assert(x5.isRightClosed == true);
    assert(x5.enableOverflow == false);
    assert(x5.enableUnderflow == false);
    assert(x5.isCircular == true);

    x5.set!EnableOverflow(true);
    x5.set!EnableUnderflow(true);
    assert(x5.get!EnableOverflow == EnableOverflow(true));
    assert(x5.get!EnableUnderflow == EnableUnderflow(true));

    AxisOptions x6 = AxisOptions(true);
    assert(x6.isRightClosed == true);
    assert(x6.enableOverflow == false);
    assert(x6.enableUnderflow == false);
    assert(x6.isCircular == false);
    AxisOptions x7 = AxisOptions(true, true);
    assert(x7.isRightClosed == true);
    assert(x7.enableOverflow == true);
    assert(x7.enableUnderflow == false);
    assert(x7.isCircular == false);
    AxisOptions x8 = AxisOptions(true, true, true);
    assert(x8.isRightClosed == true);
    assert(x8.enableOverflow == true);
    assert(x8.enableUnderflow == true);
    assert(x8.isCircular == false);
    AxisOptions x9 = AxisOptions(true, true, true, true);
    assert(x9.isRightClosed == true);
    assert(x9.enableOverflow == true);
    assert(x9.enableUnderflow == true);
    assert(x9.isCircular == true);
}

///
struct Bin(T)
    if (!is(T == enum) && !isSlice!T)
{
    ///
    T low;
    ///
    T high;
}

///
struct Bin(T)
    if (is(T == enum))
{
    ///
    T slot;
}

///
struct Bin(T)
    if (isContiguousVector!T)
{
    import mir.primitives: DeepElementType;
    import mir.ndslice.slice: Slice;

    private T _payload;

    ///
    DeepElementType!T low()() const
    {
        return _payload[0];
    }

    ///
    DeepElementType!T high()() const
    {
        return _payload[1];
    }

    this(Iterator)(Slice!Iterator x)
    {
        assert(x.length == 2);
        _payload = x;
    }
}

// Keep floating-to-integer conversion out of a caller's zero comparison.
// DMD 2.113 can crash compiling that expression after inlining. Limiting this
// barrier to the conversion lets the rest of axis lookup remain inlineable.
// Revisit the workaround when https://github.com/dlang/dmd/issues/23833 is fixed.
private CountType floatingBinIndex(CountType, T)(T value)
{
    version (DigitalMars)
        pragma(inline, false);
    return cast(CountType) value;
}

/++
Axis for an interval of integral values with unit steps.

Params:
    CountT = the type that is used to count in histogram bins
    BinT = the type of the values that are compared in histogram bins
    axisOptions = options

See_also:
    $(LREF AxisOptions),
    $(LREF RegularAxis),
    $(LREF TransformAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
struct IntegralAxis(CountT, BinT, AxisOptions axisOptions)
{
private:
    CountType _N_bin;
    BinType _low;

public:
    ///
    alias CountType = CountT;

    ///
    alias BinType = BinT;

    ///
    alias options = axisOptions;

    /++
    Construct a positive number of unit-width bins. The upper bound must be
    finite, representable, and greater than low. Floating-point boundaries must
    remain strictly increasing; checking them takes O(N_bin) construction time.
    +/
    this(CountType N_bin, BinType low)
    {
        assert(N_bin > 0, "IntegralAxis.this: N_bin must be positive");
        import std.traits: isIntegral;
        static if (isIntegral!CountType && isIntegral!BinType)
        {
            // Check before high() narrows the count or adds it to low.
            // Positive integral counts can be compared without signed promotion.
            assert(cast(ulong) N_bin <= cast(ulong) BinType.max,
                "IntegralAxis.this: N_bin must fit BinType");
            assert(low <= BinType.max - cast(BinType) N_bin,
                "IntegralAxis.this: upper bound must fit BinType");
        }
        _N_bin = N_bin;
        _low = low;
        assert(high > low, "IntegralAxis.this: upper bound must exceed low");
        import mir.internal.utility: isFloatingPoint;
        static if (isFloatingPoint!BinType)
        {
            import std.math: isFinite;
            assert(isFinite(low) && isFinite(high),
                "IntegralAxis.this: bounds must be finite");
            assert(hasStrictBoundaries(this),
                "IntegralAxis.this: boundaries must be strictly increasing");
        }
    }

    ///
    CountType N_bin()() const
    {
        return _N_bin;
    }

    ///
    BinType low()() const
    {
        return _low;
    }

    ///
    BinType high()() const
    {
        return _low + cast(BinType) _N_bin;
    }

    ///
    bool isUnderflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed &&
                   !axisOptions.isCircular) {
            return x <= _low;
        } else {
            return x < _low;
        }
    }

    ///
    bool isOverflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed ||
                   axisOptions.isCircular) {
            return x > high();
        } else {
            return x >= high();
        }
    }

    ///
    CountType index()(BinType x) const
    {
        import mir.stat.descriptive.histogram.traits: checkOverUnderFlow;

        checkOverUnderFlow!(BinType, axisOptions)(x, _low, high());

        import std.traits: isIntegral;

        static if (axisOptions.isCircular)
        {
            static if (axisOptions.isRightClosed)
            {
                if (x == _low)
                    return cast(CountType) (_N_bin - 1);
            }
            else if (x == high())
                return cast(CountType) 0;
        }
        static if (isIntegral!BinType)
        {
            static if (axisOptions.isRightClosed)
                return cast(CountType) (x - _low - 1);
            else
                return cast(CountType) (x - _low);
        }
        else
        {
            // Subtraction supplies an estimate only: compare the original value
            // with the same rounded edges exposed by bin().
            return cast(CountType) locateBoundaryBin!(axisOptions.isRightClosed())(
                this, x, x - _low);
        }
    }

    private BinType boundary(size_t i) const
    {
        return _low + cast(BinType) i;
    }

    ///
    Bin!BinType bin()(size_t x) const
    {
        assert(x < N_bin, "IntegralAxis.bin: input must be less than N_bin");
        return Bin!(BinType)(boundary(x), boundary(x + 1));
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions())(10, 2.0);
    assert(integralAxis.high == 12);

    assert(!integralAxis.isOverflow(5.0));
    assert(!integralAxis.isUnderflow(5.0));
    assert(integralAxis.isOverflow(13.0));
    assert(integralAxis.isUnderflow(1.0));

    assert(integralAxis.index(2.0) == 0);
    assert(integralAxis.index(2.5) == 0);
    assert(integralAxis.index(3.0) == 1);
    assert(integralAxis.index(11.5) == 9);

    assert(integralAxis.bin(0) == Bin!double(2.0, 3.0));
    assert(integralAxis.bin(1) == Bin!double(3.0, 4.0));
    assert(integralAxis.bin(9) == Bin!double(11.0, 12.0));
}

// Regression: an inlined floating-to-ulong bin index can be compared with zero.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;

    // Force the caller pattern even in the default test build. Only the
    // conversion helper needs the DMD workaround; lookup can still be inlined.
    pragma(inline, true)
    ulong lookup(T, bool rightClosed)(T value) @safe pure nothrow @nogc
    {
        auto axis = IntegralAxis!(ulong, T, AxisOptions(rightClosed))(4, T(-1));
        return axis.index(value);
    }

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (rightClosed; [false, true])
    {{
        assert((lookup!(T, rightClosed)(T(-0.5))) == 0);
        assert((lookup!(T, rightClosed)(T(0.5))) == 1);
        assert((lookup!(T, rightClosed)(T(2.5))) == 3);
        assert((lookup!(T, rightClosed)(T(0))) == (rightClosed ? 0 : 1));
    }}
}

// Fractional lower bounds
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto axis = IntegralAxis!(size_t, double, AxisOptions())(3, 0.5);

    assert(axis.index(0.5) == 0);
    assert(axis.index(1.25) == 0);
    assert(axis.index(1.5) == 1);
    assert(axis.index(2.25) == 1);
    assert(axis.index(2.5) == 2);
    assert(axis.index(3.25) == 2);
}

// With isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions(true))(10, 2.0);

    assert(integralAxis.index(2.5) == 0);
    assert(integralAxis.index(3.0) == 0);
    assert(integralAxis.index(3.5) == 1);
    assert(integralAxis.index(4.0) == 1);
    assert(integralAxis.index(4.5) == 2);
    assert(integralAxis.index(5.0) == 2);
    assert(integralAxis.index(5.5) == 3);
    assert(integralAxis.index(6.0) == 3);
    assert(integralAxis.index(12.0) == 9);
    assert(integralAxis.index(11.5) == 9);
}

// Some more tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, double, AxisOptions())(10, 2.0);

    assert(integralAxis.index(3.5) == 1);
    assert(integralAxis.index(4.0) == 2);
    assert(integralAxis.index(4.5) == 2);
    assert(integralAxis.index(5.0) == 3);
    assert(integralAxis.index(5.5) == 3);
    assert(integralAxis.index(6.0) == 4);
}

// integral test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, int, AxisOptions())(10, 2);

    assert(integralAxis.index(2) == 0);
    assert(integralAxis.index(4) == 2);
    assert(integralAxis.index(5) == 3);
    assert(integralAxis.index(6) == 4);
}

// integral test, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, int, AxisOptions(true))(10, 2);

    assert(integralAxis.index(4) == 1);
    assert(integralAxis.index(5) == 2);
    assert(integralAxis.index(6) == 3);
    assert(integralAxis.index(12) == 9);
}

// integral test, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, int, AxisOptions(IsCircular(true)))(10, 2);

    assert(integralAxis.index(2) == 0);
    assert(integralAxis.index(5) == 3);
    assert(integralAxis.index(12) == 0);
}

// integral test, isRightClosed = true, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto integralAxis = IntegralAxis!(size_t, int, AxisOptions(IsRightClosed(true), IsCircular(true)))(10, 2);

    assert(integralAxis.index(2) == 9);
    assert(integralAxis.index(5) == 2);
    assert(integralAxis.index(12) == 9);
}

/++
Factory function to produce $(LREF IntegralAxis) object

Params:
    N_bin = number of bins
    low = value of smallest bin

See_also:
    $(LREF IntegralAxis)
+/
IntegralAxis!(CountType, BinType, axisOptions)
    integralAxis(CountType, BinType, AxisOptions axisOptions = AxisOptions())(CountType N_bin, BinType low)
{
    return IntegralAxis!(CountType, BinType, axisOptions)(N_bin, low);
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    axisOptions = options
    N_bin = number of bins
    low = value of smallest bin
+/
IntegralAxis!(DefaultCountType, BinType, axisOptions)
    integralAxis(BinType, AxisOptions axisOptions = AxisOptions())(DefaultCountType N_bin, BinType low)
{
    return .integralAxis!(DefaultCountType, BinType, axisOptions)(N_bin, low);
}

/++
Choose the number of bins with a callable on a light-scope observation view.
The rule must not mutate or retain the view. Its result must be a positive integer
representable by CountType; assertions check the value before conversion.

Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template integralAxis(CountType, BinType, alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
    +/
    IntegralAxis!(CountType, BinType, axisOptions)
        integralAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low)
        if (acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        return .integralAxis!(CountType, BinType, axisOptions)(checkedBreakCount!(CountType, breakFunction)(slice), low);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template integralAxis(BinType, alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
    +/
    IntegralAxis!(DefaultCountType, BinType, axisOptions)
        integralAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low)
        if (acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .integralAxis!(DefaultCountType, BinType, breakFunction, axisOptions)(slice.move, low);
    }
}

/++
Params:
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template integralAxis(alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.primitives: DeepElementType;

    /++
    Params:
        slice = slice
        low = value of smallest bin
    +/
    IntegralAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions)
        integralAxis(Iterator, size_t N, SliceKind kind, BinType)(Slice!(Iterator, N, kind) slice, BinType low)
            if (is(BinType : DeepElementType!(Slice!(Iterator, N, kind))) &&
                acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .integralAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), breakFunction, axisOptions)(slice.move, low);
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    auto x0 = integralAxis!(size_t, double, AxisOptions())(10, 2.0);
    auto x1 = integralAxis!(size_t, double)(10, 2.0);
    auto x2 = integralAxis!double(10, 2.0);
    auto x3 = integralAxis(10, 2.0);

    static assert(is(typeof(x0) == IntegralAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(x1) == IntegralAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(x2) == IntegralAxis!(DefaultCountType, double, AxisOptions())));
    static assert(is(typeof(x2) == IntegralAxis!(DefaultCountType, double, AxisOptions())));
}

/// Example with break function
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y0 = integralAxis!(size_t, double, sturges, AxisOptions())(x, 2.0);
    auto y1 = integralAxis!(size_t, double, sturges)(x, 2.0);
    auto y2 = integralAxis!(double, sturges)(x, 2.0);
    auto y3 = integralAxis!sturges(x, 2.0);

    static assert(is(typeof(y0) == IntegralAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(y1) == IntegralAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(y2) == IntegralAxis!(DefaultCountType, double, AxisOptions())));
    static assert(is(typeof(y3) == IntegralAxis!(DefaultCountType, double, AxisOptions())));
}

// Check number of bins
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y = integralAxis!(size_t, double, sturges, AxisOptions())(x, 2.0);

    assert(y.N_bin == 4);
}

// Boundary descriptions and lookup share a single rounded grid. Keep the
// original observation for comparison: normalization may lose its distance
// from an edge. These helpers do not allocate or retain the axis.
private bool hasStrictBoundaries(Axis)(scope ref const Axis axis)
{
    auto previous = axis.boundary(0);
    const n = cast(size_t) axis.N_bin;
    for (size_t i = 0; i < n; ++i)
    {
        auto next = axis.boundary(i + 1);
        if (!(previous < next))
            return false;
        previous = next;
    }
    return true;
}

private size_t locateBoundaryBin(bool rightClosed, Axis, Value, Scaled)(
    scope ref const Axis axis, Value x, Scaled scaled)
{
    import mir.math.common: floor;

    const n = cast(size_t) axis.N_bin;
    size_t candidate;
    if (scaled >= n)
        candidate = n - 1;
    else if (scaled > 0) // Zero, negative, or NaN estimates fall back from bin zero.
    {
        candidate = floatingBinIndex!size_t(floor(scaled));
        static if (rightClosed)
            if (scaled == candidate)
                --candidate;
    }
    auto lower = axis.boundary(candidate);
    auto upper = axis.boundary(candidate + 1);
    static if (rightClosed)
    {
        if (lower < x && x <= upper)
            return candidate;
    }
    else
    {
        if (lower <= x && x < upper)
            return candidate;
    }

    // A fixed one-bin adjustment is insufficient when arithmetic loses several
    // boundary distinctions. Search for the first upper edge containing x.
    size_t first = 0, last = n;
    while (first < last)
    {
        const middle = first + (last - first) / 2;
        auto edge = axis.boundary(middle + 1);
        static if (rightClosed)
            const contains = x <= edge;
        else
            const contains = x < edge;
        if (contains)
            last = middle;
        else
            first = middle + 1;
    }
    assert(first < n, "Axis.index: no bin contains the observation");
    return first;
}

/++
Axis for an interval of values with equal width steps.

See $(LREF TransformAxis) for an alternative axis that allows for monotonic
transformations.

Params:
    CountT = the type that is used to count in histogram bins
    BinT = the type of the values that are compared in histogram bins
    axisOptions = options

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF TransformAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
struct RegularAxis(CountT, BinT, AxisOptions axisOptions)
{
    import mir.math.common: fmamath;

private:
    CountType _N_bin;
    BinType _low;
    BinType _high;

public:
    ///
    alias CountType = CountT;

    ///
    alias BinType = BinT;

    ///
    alias options = axisOptions;

    /++
    Construct a positive number of equal-width bins. The upper bound must
    exceed the lower bound; floating-point bounds and their width must be finite.
    Adjacent rounded boundaries must be strictly increasing. Assertion-enabled
    construction checks all bins in O(N_bin) time without allocating storage.
    +/
    this(CountType N_bin, BinType low, BinType high)
    {
        assert(N_bin > 0, "RegularAxis.this: N_bin must be positive");
        assert(high > low, "RegularAxis.this: high must be greater than low");
        import mir.internal.utility: isFloatingPoint;
        static if (isFloatingPoint!BinType)
        {
            import std.math: isFinite;
            assert(isFinite(low) && isFinite(high) && isFinite(high - low),
                "RegularAxis.this: bounds and width must be finite");
        }
        _N_bin = N_bin;
        _low = low;
        _high = high;
        assert(hasStrictBoundaries(this),
            "RegularAxis.this: adjacent boundaries must be strictly increasing");
    }

    // Preserve exact endpoints. Form the fraction before multiplying so a tiny
    // step need not be rounded to zero first. Use ordinary floating-point
    // evaluation here, without the fmamath contraction annotation.
    private BinType boundary()(size_t i) const
    {
        if (i == 0)
            return _low;
        if (i == _N_bin)
            return _high;
        import mir.internal.utility: isFloatingPoint;
        static if (isFloatingPoint!BinType)
        {
            const BinType fraction = cast(BinType) i / cast(BinType) _N_bin;
            const BinType offset = (_high - _low) * fraction;
            return _low + offset;
        }
        else
            return _low + cast(BinType) i * stepSize();
    }

    ///
    CountType N_bin()() const
    {
        return _N_bin;
    }

    ///
    BinType low()() const
    {
        return _low;
    }

    ///
    BinType high()() const
    {
        return _high;
    }

    ///
    bool isUnderflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed &&
                   !axisOptions.isCircular) {
            return x <= _low;
        } else {
            return x < _low;
        }
    }

    ///
    bool isOverflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed ||
                   axisOptions.isCircular) {
            return x > _high;
        } else {
            return x >= _high;
        }
    }

    /// Nominal width; use bin for actual rounded boundaries.
    @fmamath BinType stepSize()() const
    {
        return (_high - _low) / (cast(BinType) _N_bin);
    }

    ///
    @fmamath BinType value()(BinType x) const
    {
        return (x - _low) / (_high - _low);
    }

    /++
    Index using the same rounded boundaries returned by bin.
    Normalization supplies a candidate; the original observation is checked
    against its edges. A mismatch uses an O(log N_bin) boundary search.
    +/
    CountType index()(BinType x) const
    {
        import mir.stat.descriptive.histogram.traits: checkOverUnderFlow;
        checkOverUnderFlow!(BinType, axisOptions)(x, _low, _high);

        static if (axisOptions.isCircular)
        {
            static if (axisOptions.isRightClosed)
            {
                if (x == _low)
                    return _N_bin - 1;
            }
            else if (x == _high)
                return 0;
        }
        return cast(CountType) locateBoundaryBin!(axisOptions.isRightClosed())(
            this, x, _N_bin * this.value(x));
    }

    ///
    Bin!BinType bin()(size_t x) const
    {
        assert(x < N_bin, "RegularAxis.bin: input must be less than N_bin");
        return Bin!(BinType)(boundary(x), boundary(x + 1));
    }
}

/// Shared boundaries determine membership even when normalization loses precision.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.math: nextDown, nextUp;
    auto axis = RegularAxis!(uint, double, AxisOptions())(2, -1.0, 1.0);

    // Both descriptions use exactly the same shared edge, at zero.
    assert(axis.bin(0).high == axis.bin(1).low);
    assert(axis.bin(1).low == 0.0);

    // Adding one to the smallest negative double rounds to one. Lookup still
    // uses the original observation to place it below the boundary at zero.
    assert(axis.index(nextDown(0.0)) == 0);
    assert(axis.index(0.0) == 1);
    assert(axis.index(nextUp(0.0)) == 1);
}

/// Basic tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions())(10, 2.0, 12.0);
    assert(regularAxis.low == 2);
    assert(regularAxis.high == 12);

    assert(!regularAxis.isOverflow(5.0));
    assert(!regularAxis.isUnderflow(5.0));
    assert(regularAxis.isOverflow(13.0));
    assert(regularAxis.isUnderflow(1.0));

    assert(regularAxis.index(2.0) == 0);
    assert(regularAxis.index(2.5) == 0);
    assert(regularAxis.index(3.0) == 1);
    assert(regularAxis.index(11.5) == 9);

    assert(regularAxis.bin(0) == Bin!double(2.0, 3.0));
    assert(regularAxis.bin(1) == Bin!double(3.0, 4.0));
    assert(regularAxis.bin(9) == Bin!double(11.0, 12.0));
}

// isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(true))(10, 2.0, 12.0);

    assert(regularAxis.index(2.5) == 0);
    assert(regularAxis.index(3.0) == 0);
    assert(regularAxis.index(3.5) == 1);
    assert(regularAxis.index(12.0) == 9);
}

// isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(IsCircular(true)))(10, 2.0, 12.0);

    assert(regularAxis.index(2.5) == 0);
    assert(regularAxis.index(3.0) == 1);
    assert(regularAxis.index(3.5) == 1);
    assert(regularAxis.index(12.0) == 0);
}

// isRightClosed = true, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(IsRightClosed(true), IsCircular(true)))(10, 2.0, 12.0);

    assert(regularAxis.index(2.0) == 9);
    assert(regularAxis.index(2.5) == 0);
    assert(regularAxis.index(3.0) == 0);
    assert(regularAxis.index(3.5) == 1);
    assert(regularAxis.index(12.0) == 9);
}

// Some more tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions())(10, 2.0, 12.0);
    assert(regularAxis.stepSize == 1);
    assert(regularAxis.value(7.0) == 0.5);

    assert(regularAxis.index(3.5) == 1);
    assert(regularAxis.index(4.0) == 2);
    assert(regularAxis.index(4.5) == 2);
    assert(regularAxis.index(5.0) == 3);
    assert(regularAxis.index(5.5) == 3);
    assert(regularAxis.index(6.0) == 4);
}

// Double N_bin
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions())(20, 2.0, 12.0);

    assert(regularAxis.index(2.0) == 0);
    assert(regularAxis.index(2.25) == 0);
    assert(regularAxis.index(2.5) == 1);
    assert(regularAxis.index(2.75) == 1);
    assert(regularAxis.index(3.0) == 2);
    assert(regularAxis.index(3.5) == 3);
    assert(regularAxis.index(4.0) == 4);
    assert(regularAxis.index(4.5) == 5);
    assert(regularAxis.index(5.0) == 6);
    assert(regularAxis.index(5.5) == 7);
    assert(regularAxis.index(6.0) == 8);
    assert(regularAxis.index(11.5) == 19);
    assert(regularAxis.index(11.75) == 19);
}

// Double N_bin, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    auto regularAxis = RegularAxis!(size_t, double, AxisOptions(true))(20, 2.0, 12.0);

    assert(regularAxis.index(2.25) == 0);
    assert(regularAxis.index(2.5) == 0);
    assert(regularAxis.index(2.75) == 1);
    assert(regularAxis.index(3.0) == 1);
    assert(regularAxis.index(3.5) == 2);
    assert(regularAxis.index(4.0) == 3);
    assert(regularAxis.index(4.5) == 4);
    assert(regularAxis.index(5.0) == 5);
    assert(regularAxis.index(5.5) == 6);
    assert(regularAxis.index(6.0) == 7);
    assert(regularAxis.index(11.5) == 18);
    assert(regularAxis.index(11.75) == 19);
    assert(regularAxis.index(12.0) == 19);
}

/++
Factory function to produce $(LREF RegularAxis) object

Params:
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin

See_also:
    $(LREF RegularAxis)
+/
RegularAxis!(CountType, BinType, axisOptions)
    regularAxis(CountType, BinType, AxisOptions axisOptions = AxisOptions())(CountType N_bin, BinType low, BinType high)
{
    return RegularAxis!(CountType, BinType, axisOptions)(N_bin, low, high);
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    axisOptions = options
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin
+/
RegularAxis!(DefaultCountType, BinType, axisOptions)
    regularAxis(BinType, AxisOptions axisOptions = AxisOptions())(DefaultCountType N_bin, BinType low, BinType high)
{
    return .regularAxis!(DefaultCountType, BinType, axisOptions)(N_bin, low, high);
}

/++
Choose the number of bins with a callable on a light-scope observation view.
The rule must not mutate or retain the view. Its result must be a positive integer
representable by CountType; assertions check the value before conversion.

Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template regularAxis(CountType, BinType, alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    RegularAxis!(CountType, BinType, axisOptions)
        regularAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        return .regularAxis!(CountType, BinType, axisOptions)(checkedBreakCount!(CountType, breakFunction)(slice), low, high);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template regularAxis(BinType, alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    RegularAxis!(DefaultCountType, BinType, axisOptions)
        regularAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .regularAxis!(DefaultCountType, BinType, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/++
Params:
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template regularAxis(alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.primitives: DeepElementType;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    RegularAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), axisOptions)
        regularAxis(Iterator, size_t N, SliceKind kind, BinType)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (is(BinType : DeepElementType!(Slice!(Iterator, N, kind))) &&
                acceptsBreakFunction!(breakFunction, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .regularAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), breakFunction, axisOptions)(slice.move, low, high);
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    auto x0 = regularAxis!(size_t, double, AxisOptions())(10, 2.0, 12.0);
    auto x1 = regularAxis!(size_t, double)(10, 2.0, 12.0);
    auto x2 = regularAxis!double(10, 2.0, 12.0);
    auto x3 = regularAxis(10, 2.0, 12.0);

    static assert(is(typeof(x0) == RegularAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(x1) == RegularAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(x2) == RegularAxis!(DefaultCountType, double, AxisOptions())));
    static assert(is(typeof(x2) == RegularAxis!(DefaultCountType, double, AxisOptions())));
}

/// Example with break function
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y0 = regularAxis!(size_t, double, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y1 = regularAxis!(size_t, double, sturges)(x, 2.0, 12.0);
    auto y2 = regularAxis!(double, sturges)(x, 2.0, 12.0);
    auto y3 = regularAxis!sturges(x, 2.0, 12.0);

    static assert(is(typeof(y0) == RegularAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(y1) == RegularAxis!(size_t, double, AxisOptions())));
    static assert(is(typeof(y2) == RegularAxis!(DefaultCountType, double, AxisOptions())));
    static assert(is(typeof(y3) == RegularAxis!(DefaultCountType, double, AxisOptions())));
}

// Check number of bins
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y = regularAxis!(size_t, double, sturges, AxisOptions())(x, 2.0, 12.0);

    assert(y.N_bin == 4);
}

/++
Axis for an interval of values with equal underlying steps that may be
transformed to provide fast, unequal steps.

A $(LREF RegularAxis) is equivalent to a $(LREF TransformAxis) with an identity
`transform` function.

Params:
    CountT = the type that is used to count in histogram bins
    BinT = the type of the values that are compared in histogram bins
    transform = function to transform axis
    axisOptions = options

See_also:
    $(MATHREF common, log10),
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
struct TransformAxis(CountT, BinT, alias transform, alias inverseTransform, AxisOptions axisOptions)
{
    import mir.math.common: fmamath;

private:
    RegularAxis!(CountType, BinType, axisOptions) regularAxis = void;
    BinType _low;
    BinType _high;

    alias transformFunction = naryFun!transform;
    alias inverseTransformFunction = naryFun!inverseTransform;
    alias transformType = typeof(transformFunction(BinType.init));
    alias inverseTransformType = typeof(inverseTransformFunction(BinType.init));
    static assert (is(BinType == inverseTransformType), "the return type of inverseTransform must match BinType");

public:
    ///
    alias CountType = CountT;

    ///
    alias BinType = BinT;

    ///
    alias options = regularAxis.options;

    /++
    Construct bins in transformed space with shared original-space boundaries.
    Transform functions must be deterministic, and inverse-transformed edges
    between low and high must be strictly increasing. Assertion-enabled
    construction checks all edges in O(N_bin) time without allocating storage.
    +/
    this(CountType N_bin, BinType low, BinType high)
    {
        assert(high > low, "TransformAxis.this: high must be greater than low");
        regularAxis = RegularAxis!(CountType, BinType, axisOptions)(N_bin, transformFunction(low), transformFunction(high));
        _low = low;
        _high = high;
        assert(hasStrictBoundaries(this),
            "TransformAxis.this: inverse-transformed boundaries must be strictly increasing");
    }

    private BinType boundary()(size_t i) const
    {
        if (i == 0)
            return _low;
        if (i == N_bin)
            return _high;
        return inverseTransformFunction(regularAxis.boundary(i));
    }

    ///
    CountType N_bin()() const
    {
        return regularAxis._N_bin;
    }

    ///
    BinType low()() const
    {
        return _low;
    }

    ///
    BinType high()() const
    {
        return _high;
    }

    ///
    transformType lowTransform()() const
    {
        return regularAxis._low;
    }

    ///
    transformType highTransform()() const
    {
        return regularAxis._high;
    }

    ///
    bool isUnderflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed &&
                   !axisOptions.isCircular) {
            return x <= _low;
        } else {
            return x < _low;
        }
    }

    ///
    bool isOverflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed ||
                   axisOptions.isCircular) {
            return x > _high;
        } else {
            return x >= _high;
        }
    }

    ///
    @fmamath BinType stepSize()() const
    {
        return regularAxis.stepSize();
    }

    ///
    @fmamath BinType value()(BinType x) const
    {
        return regularAxis.value(transformFunction(x));
    }

    /++
    Classify the original observation against the boundaries returned by bin.
    The forward transform provides only a candidate: rounded transform values
    may coincide even when observations lie on opposite sides of an edge.
    +/
    CountType index()(BinType x) const
    {
        import mir.stat.descriptive.histogram.traits: checkOverUnderFlow;
        checkOverUnderFlow!(BinType, axisOptions)(x, _low, _high);
        static if (axisOptions.isCircular)
        {
            static if (axisOptions.isRightClosed)
            {
                if (x == _low)
                    return N_bin - 1;
            }
            else if (x == _high)
                return 0;
        }
        return cast(CountType) locateBoundaryBin!(axisOptions.isRightClosed())(
            this, x, N_bin * regularAxis.value(transformFunction(x)));
    }

    ///
    Bin!BinType bin(size_t x) const
    {
        assert(x < N_bin, "TransformAxis.bin: input must be less than N_bin");
        return Bin!BinType(boundary(x), boundary(x + 1));
    }
}

/// Transformed lookup follows displayed boundaries in the original input space.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;
    import std.math: nextDown, nextUp;
    auto axis = TransformAxis!(uint, double, log10, "10.0 ^^ a", AxisOptions())(
        20, 1.0, 1.0e12);
    auto edge = axis.bin(1).low;

    // Adjacent descriptions share one inverse-transformed edge.
    assert(axis.bin(0).high == edge);
    // Even if logarithms of these nearby values round identically, the
    // original observations distinguish which side of the edge they occupy.
    assert(axis.index(nextDown(edge)) == 0);
    assert(axis.index(edge) == 1);
    assert(axis.index(nextUp(edge)) == 1);
    // The outside endpoints retain the exact values supplied at construction.
    assert(axis.bin(0).low == 1.0);
    assert(axis.bin(19).high == 1.0e12);
}

/// Basic tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.low == 10.0 ^^ 2.0);
    assert(transformAxis.high == 10.0 ^^ 12.0);

    assert(!transformAxis.isOverflow(10.0 ^^ 5.0));
    assert(!transformAxis.isUnderflow(10.0 ^^ 5.0));
    assert(transformAxis.isOverflow(10.0 ^^ 13.0));
    assert(transformAxis.isUnderflow(10.0 ^^ 1.0));

    assert(transformAxis.index(10.0 ^^ 2.0) == 0);
    assert(transformAxis.index(10.0 ^^ 2.5) == 0);
    assert(transformAxis.index(10.0 ^^ 3.0) == 1);
    assert(transformAxis.index(10.0 ^^ 11.5) == 9);

    assert(transformAxis.bin(0) == Bin!double(10.0 ^^ 2.0, 10.0 ^^ 3.0));
    assert(transformAxis.bin(1) == Bin!double(10.0 ^^ 3.0, 10.0 ^^ 4.0));
    assert(transformAxis.bin(9) == Bin!double(10.0 ^^ 11.0, 10.0 ^^ 12.0));

    // Can also supply lambda
    auto transformAxis2 = TransformAxis!(size_t, double, a => log10(a), a => 10.0 ^^ a, AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis2.index(10.0 ^^ 3.0) == 1);

    // Or string lambda
    auto transformAxis3 = TransformAxis!(size_t, double, "log10(a)", "10.0 ^^ a", AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis3.index(10.0 ^^ 3.0) == 1);
}

// isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions(true))(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.index(10.0 ^^ 2.5) == 0);
    assert(transformAxis.index(10.0 ^^ 3.0) == 0);
    assert(transformAxis.index(10.0 ^^ 3.5) == 1);
    assert(transformAxis.index(10.0 ^^ 12.0) == 9);
}

// isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions(IsCircular(true)))(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.index(10.0 ^^ 2.5) == 0);
    assert(transformAxis.index(10.0 ^^ 3.0) == 1);
    assert(transformAxis.index(10.0 ^^ 3.5) == 1);
    assert(transformAxis.index(10.0 ^^ 12.0) == 0);
}

// isRightClosed = true, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions(IsRightClosed(true), IsCircular(true)))(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.index(10.0 ^^ 2.0) == 9);
    assert(transformAxis.index(10.0 ^^ 2.5) == 0);
    assert(transformAxis.index(10.0 ^^ 3.0) == 0);
    assert(transformAxis.index(10.0 ^^ 3.5) == 1);
    assert(transformAxis.index(10.0 ^^ 12.0) == 9);
}

// Some more tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis.stepSize == 1);
    assert(transformAxis.value(10.0 ^^ 7.0) == 0.5);

    assert(transformAxis.index(10.0 ^^ 3.5) == 1);
    assert(transformAxis.index(10.0 ^^ 4.0) == 2);
    assert(transformAxis.index(10.0 ^^ 4.5) == 2);
    assert(transformAxis.index(10.0 ^^ 5.0) == 3);
    assert(transformAxis.index(10.0 ^^ 5.5) == 3);
    assert(transformAxis.index(10.0 ^^ 6.0) == 4);
}

// Double N_bin
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions())(20, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.index(10.0 ^^ 2.0) == 0);
    assert(transformAxis.index(10.0 ^^ 2.25) == 0);
    assert(transformAxis.index(10.0 ^^ 2.5) == 1);
    assert(transformAxis.index(10.0 ^^ 2.75) == 1);
    assert(transformAxis.index(10.0 ^^ 3.0) == 2);
    assert(transformAxis.index(10.0 ^^ 3.5) == 3);
    assert(transformAxis.index(10.0 ^^ 4.0) == 4);
    assert(transformAxis.index(10.0 ^^ 4.5) == 5);
    assert(transformAxis.index(10.0 ^^ 5.0) == 6);
    assert(transformAxis.index(10.0 ^^ 5.5) == 7);
    assert(transformAxis.index(10.0 ^^ 6.0) == 8);
    assert(transformAxis.index(10.0 ^^ 11.4) == 18);
    assert(transformAxis.index(10.0 ^^ 11.6) == 19);
    assert(transformAxis.index(10.0 ^^ 11.75) == 19);
}

// Double N_bin
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions(true))(20, 10.0 ^^ 2.0, 10.0 ^^ 12.0);

    assert(transformAxis.index(10.0 ^^ 2.25) == 0);
    assert(transformAxis.index(10.0 ^^ 2.5) == 0);
    assert(transformAxis.index(10.0 ^^ 2.75) == 1);
    assert(transformAxis.index(10.0 ^^ 3.0) == 1);
    assert(transformAxis.index(10.0 ^^ 3.5) == 2);
    assert(transformAxis.index(10.0 ^^ 4.0) == 3);
    assert(transformAxis.index(10.0 ^^ 4.5) == 4);
    assert(transformAxis.index(10.0 ^^ 5.0) == 5);
    assert(transformAxis.index(10.0 ^^ 5.5) == 6);
    assert(transformAxis.index(10.0 ^^ 6.0) == 7);
    assert(transformAxis.index(10.0 ^^ 11.5) == 18);
    assert(transformAxis.index(10.0 ^^ 11.75) == 19);
    assert(transformAxis.index(10.0 ^^ 12.0) == 19);
}

// Check bin
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;

    double inverseLog10(double x) {
        return 10.0 ^^ x;
    }

    auto transformAxis1 = TransformAxis!(size_t, double, log10, inverseLog10, AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis1.bin(0) == Bin!double(10.0 ^^ 2.0, 10.0 ^^ 3.0));

    // lambda function
    auto transformAxis2 = TransformAxis!(size_t, double, a => log10(a), a => (10.0 ^^ a), AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis2.bin(0) == Bin!double(10.0 ^^ 2.0, 10.0 ^^ 3.0));

    // string lambda
    auto transformAxis3 = TransformAxis!(size_t, double, "log10(a)", "(10.0 ^^ a)", AxisOptions())(10, 10.0 ^^ 2.0, 10.0 ^^ 12.0);
    assert(transformAxis3.bin(0) == Bin!double(10.0 ^^ 2.0, 10.0 ^^ 3.0));
}

// Additional bin tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp, log, log2, sqrt;

    auto transformAxis1 = TransformAxis!(size_t, double, log, exp, AxisOptions())(10, exp(2.0), exp(12.0));
    assert(transformAxis1.bin(0) == Bin!double(exp(2.0), exp(3.0)));

    auto transformAxis2 = TransformAxis!(size_t, double, log2, "2.0 ^^ a", AxisOptions())(10, 2.0 ^^ 2.0, 2.0 ^^ 12.0);
    assert(transformAxis2.bin(0) == Bin!double(4.0, 8.0));

    auto transformAxis3 = TransformAxis!(size_t, double, sqrt, (a => a ^^ 2.0), AxisOptions())(10, 4.0, 144.0);
    assert(transformAxis3.bin(0) == Bin!double(4.0, 9.0));
}

private T exp10(T)(T x) {
    return 10 ^^ x;
}

private T square(T)(T x) {
    return x ^^ 2;
}

/++
Provides a built-in inverse to a $(LREF transform) function.

The following functions are supported: $(MATHREF common, exp), $(MATHREF common, exp2),
$(MATHREF common, log), $(MATHREF common, log2), $(MATHREF common, log10),
$(MATHREF common, sqrt).

Params:
    transform = function to transform axis

See_also:
    $(LREF TransformAxis),
    $(LREF transformAxis),
    $(LREF inverseTransformMapping),
    $(MATHREF common, exp),
    $(MATHREF common, exp2),
    $(MATHREF common, log),
    $(MATHREF common, log2),
    $(MATHREF common, log10),
    $(MATHREF common, sqrt)
+/
template inverseTransformMapping(alias transform)
{
    import mir.math.common: exp, exp2, log, log2, log10, sqrt;

    static if (__traits(isSame, transform, exp)) {
        alias inverseTransformMapping = log;
    } else static if (__traits(isSame, transform, exp2)) {
        alias inverseTransformMapping = log2;
    } else static if (__traits(isSame, transform, log)) {
        alias inverseTransformMapping = exp;
    } else static if (__traits(isSame, transform, log2)) {
        alias inverseTransformMapping = exp2;
    } else static if (__traits(isSame, transform, log10)) {
        alias inverseTransformMapping = exp10;
    } else static if (__traits(isSame, transform, sqrt)) {
        alias inverseTransformMapping = square;
    } else {
        static assert (0, "inverseTransformMapping: transform does not match built-in support");
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp, exp2, log, log2, log10, sqrt, approxEqual;

    assert(inverseTransformMapping!exp(5f).approxEqual(log(5f)));
    assert(inverseTransformMapping!exp2(5f).approxEqual(log2(5f)));
    assert(inverseTransformMapping!log(5f).approxEqual(exp(5f)));
    assert(inverseTransformMapping!log2(5f).approxEqual(exp2(5f)));
    assert(inverseTransformMapping!log10(5f) == 100_000);
    assert(inverseTransformMapping!sqrt(5f) == 25);
}

/++
Check that an inverse function is provided by default for $(LREF transform).

The following functions are supported: $(MATHREF common, exp),
$(MATHREF common, exp2), $(MATHREF common, log), $(MATHREF common, log2),
$(MATHREF common, log10), $(MATHREF common, sqrt).

Params:
    transform = function to transform axis

See_also:
    $(LREF TransformAxis),
    $(LREF transformAxis),
    $(LREF inverseTransformMapping),
    $(MATHREF common, exp),
    $(MATHREF common, exp2),
    $(MATHREF common, log),
    $(MATHREF common, log2),
    $(MATHREF common, log10),
    $(MATHREF common, sqrt)
+/
template hasInverseTransformMapping(alias transform)
{
    import mir.math.common: exp, exp2, log, log2, log10, sqrt;

    static if (__traits(isSame, transform, exp)) {
        enum bool hasInverseTransformMapping = true;
    } else static if (__traits(isSame, transform, exp2)) {
        enum bool hasInverseTransformMapping = true;
    } else static if (__traits(isSame, transform, log)) {
        enum bool hasInverseTransformMapping = true;
    } else static if (__traits(isSame, transform, log2)) {
        enum bool hasInverseTransformMapping = true;
    } else static if (__traits(isSame, transform, log10)) {
        enum bool hasInverseTransformMapping = true;
    } else static if (__traits(isSame, transform, sqrt)) {
        enum bool hasInverseTransformMapping = true;
    } else {
        enum bool hasInverseTransformMapping = false;
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp, exp2, log, log2, log10, sqrt;

    static assert(hasInverseTransformMapping!exp);
    static assert(hasInverseTransformMapping!exp2);
    static assert(hasInverseTransformMapping!log);
    static assert(hasInverseTransformMapping!log2);
    static assert(hasInverseTransformMapping!log10);
    static assert(hasInverseTransformMapping!sqrt);
}

///
package
template isTransformFunction(alias T, BinT)
{
    static if (!isBreakFunction!T) {
        import std.traits: isSomeFunction;
        static if (isSomeFunction!T) {
            enum bool isTransformFunction = is(typeof(naryFun!T(BinT.init)));
        } else static if (__traits(isTemplate, T)) {
            // Probe the invocation used by the axis, so incompatible template
            // constraints reject the candidate without a hard error.
            enum bool isTransformFunction = is(typeof(naryFun!T(BinT.init)));
        } else static if (is(typeof(T) : string)) {
            enum bool isTransformFunction = is(typeof(naryFun!T(cast(BinT) 0.5f)));
        } else {
            enum bool isTransformFunction = false;
        }
    } else {
        enum bool isTransformFunction = false;
    }
}

// Match the coordinate type used by the transformed axis's regular bins.
package auto transformedBreakData(BinType, alias transform, S)(S observations)
{
    import mir.ndslice.topology: map;
    return observations.lightScope.map!((value) =>
        cast(BinType) naryFun!transform(cast(BinType) value));
}

package template acceptsTransformedBreakFunction(alias rule, alias transform, BinType, S)
{
    static if (isTransformFunction!(transform, BinType) &&
        is(typeof(transformedBreakData!(BinType, transform)(S.init)) Mapped))
        enum acceptsTransformedBreakFunction = acceptsBreakFunction!(rule, Mapped);
    else
        enum acceptsTransformedBreakFunction = false;
}

/++
Factory function to produce $(LREF TransformAxis) object.

Params:
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin

See_also:
    $(LREF TransformAxis)
+/
TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)
    transformAxis(CountType, BinType, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())(CountType N_bin, BinType low, BinType high)
        if (isTransformFunction!(transform, BinType) &&
            isTransformFunction!(inverseTransform, BinType))
{
    return TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    axisOptions = options
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin
+/
TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions)
    transformAxis(CountType, BinType, alias transform, AxisOptions axisOptions = AxisOptions())(CountType N_bin, BinType low, BinType high)
        if (hasInverseTransformMapping!transform)
{
    alias inverseTransform = inverseTransformMapping!transform;
    return TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin
+/
TransformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)
    transformAxis(BinType, alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())(DefaultCountType N_bin, BinType low, BinType high)
        if (isTransformFunction!(transform, BinType) &&
            isTransformFunction!(inverseTransform, BinType))
{
    return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    axisOptions = options
    N_bin = number of bins
    low = value of smallest bin
    high = value of the largest bin
+/
TransformAxis!(DefaultCountType, BinType, transform, inverseTransformMapping!transform, axisOptions)
    transformAxis(BinType, alias transform, AxisOptions axisOptions = AxisOptions())(DefaultCountType N_bin, BinType low, BinType high)
        if (hasInverseTransformMapping!transform)
{
    alias inverseTransform = inverseTransformMapping!transform;
    return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
}

/++
Params:
    transform = function to transform axis
    inverseTransform = function to undo transform
    axisOptions = options
+/
template transformAxis(alias transform, alias inverseTransform, AxisOptions axisOptions = AxisOptions())
{
    /++
    Params:
        N_bin = number of bins
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)
        transformAxis(BinType)(DefaultCountType N_bin, BinType low, BinType high)
            if (isTransformFunction!(transform, BinType) &&
                isTransformFunction!(inverseTransform, BinType))
    {
        return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
    }
}

/++
Params:
    transform = function to transform axis
    axisOptions = options
+/
template transformAxis(alias transform, AxisOptions axisOptions = AxisOptions())
    if (hasInverseTransformMapping!transform)
{
    /++
    Params:
        N_bin = number of bins
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, BinType, transform, inverseTransformMapping!transform, axisOptions)
        transformAxis(BinType)(DefaultCountType N_bin, BinType low, BinType high)
    {
        alias inverseTransform = inverseTransformMapping!transform;
        return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)(N_bin, low, high);
    }
}

/++
Choose the bin count by applying the rule to transformed observations.
The rule receives a lazy, light-scope view in the same coordinate type used by
this axis's regular bins. It must not mutate or retain that view. The result
must be a positive integer representable by CountType.
Bounds and the observations later inserted into the histogram remain in original
units. To choose a count from original data instead, calculate it separately and
use the overload taking an explicit N_bin.

Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    inverseTransform = function to undo transform
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(CountType, BinType, alias transform, alias inverseTransform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (isTransformFunction!(transform, BinType) &&
        isTransformFunction!(inverseTransform, BinType))
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, Slice!(Iterator, N, kind)))
    {
        auto transformed = transformedBreakData!(BinType, transform)(slice);
        const count = checkedBreakCount!(CountType, breakFunction)(transformed);
        return .transformAxis!(CountType, BinType, transform, inverseTransform, axisOptions)(count, low, high);
    }
}

/++
Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(CountType, BinType, alias transform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (hasInverseTransformMapping!transform)
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(CountType, BinType, transform, inverseTransformMapping!transform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        alias inverseTransform = inverseTransformMapping!transform;
        return .transformAxis!(CountType, BinType, transform, inverseTransform, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    inverseTransform = function to undo transform
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(BinType, alias transform, alias inverseTransform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (isTransformFunction!(transform, BinType) &&
        isTransformFunction!(inverseTransform, BinType))
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, BinType, transform, inverseTransform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/++
Params:
    BinType = the type of the values that are compared in histogram bins
    transform = function to transform axis
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(BinType, alias transform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (hasInverseTransformMapping!transform)
{
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, BinType, transform, inverseTransformMapping!transform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
        if (acceptsTransformedBreakFunction!(breakFunction, transform, BinType, Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        alias inverseTransform = inverseTransformMapping!transform;
        return .transformAxis!(DefaultCountType, BinType, transform, inverseTransform, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/++
Params:
    transform = function to transform axis
    inverseTransform = function to undo transform
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(alias transform, alias inverseTransform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.primitives: DeepElementType;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), transform, inverseTransform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind, BinType)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (isTransformFunction!(transform, DeepElementType!(Slice!(Iterator, N, kind))) &&
                isTransformFunction!(inverseTransform, DeepElementType!(Slice!(Iterator, N, kind))) &&
                is(BinType : DeepElementType!(Slice!(Iterator, N, kind))) &&
                acceptsTransformedBreakFunction!(breakFunction, transform, DeepElementType!(Slice!(Iterator, N, kind)), Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;
        return .transformAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), transform, inverseTransform, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/++
Params:
    transform = function to transform axis
    breakFunction = function used to determine breaks
    axisOptions = options
+/
template transformAxis(alias transform, alias breakFunction, AxisOptions axisOptions = AxisOptions())
    if (hasInverseTransformMapping!transform)
{
    import mir.ndslice.slice: Slice, SliceKind;
    import mir.primitives: DeepElementType;

    /++
    Params:
        slice = slice
        low = value of smallest bin
        high = value of the largest bin
    +/
    TransformAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), transform, inverseTransformMapping!transform, axisOptions)
        transformAxis(Iterator, size_t N, SliceKind kind, BinType)(Slice!(Iterator, N, kind) slice, BinType low, BinType high)
            if (is(BinType : DeepElementType!(Slice!(Iterator, N, kind))) &&
                acceptsTransformedBreakFunction!(breakFunction, transform, DeepElementType!(Slice!(Iterator, N, kind)), Slice!(Iterator, N, kind)))
    {
        import core.lifetime: move;

        alias inverseTransform = inverseTransformMapping!transform;
        return .transformAxis!(DefaultCountType, DeepElementType!(Slice!(Iterator, N, kind)), transform, inverseTransform, breakFunction, axisOptions)(slice.move, low, high);
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    import mir.math.common: exp, log;

    auto x0 = transformAxis!(size_t, double, exp, log, AxisOptions())(10, 2.0, 12.0);
    auto x1 = transformAxis!(size_t, double, exp, log)(10, 2.0, 12.0);
    auto x2 = transformAxis!(size_t, double, exp)(10, 2.0, 12.0);
    auto x3 = transformAxis!(double, exp, log)(10, 2.0, 12.0);
    auto x4 = transformAxis!(double, exp)(10, 2.0, 12.0);
    auto x5 = transformAxis!(exp, log)(10, 2.0, 12.0);
    auto x6 = transformAxis!exp(10, 2.0, 12.0);

    static assert(is(typeof(x0) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(x1) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(x2) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(x3) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
    static assert(is(typeof(x4) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
    static assert(is(typeof(x5) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
    static assert(is(typeof(x6) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
}

/// Example with break function
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    import mir.math.common: exp, log;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y0 = transformAxis!(size_t, double, exp, log, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y1 = transformAxis!(size_t, double, exp, log, sturges)(x, 2.0, 12.0);
    auto y2 = transformAxis!(double, exp, log, sturges)(x, 2.0, 12.0);
    auto y3 = transformAxis!(exp, log, sturges)(x, 2.0, 12.0);

    static assert(is(typeof(y0) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(y1) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(y2) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
    static assert(is(typeof(y3) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));

    auto y4 = transformAxis!(size_t, double, exp, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y5 = transformAxis!(size_t, double, exp, sturges)(x, 2.0, 12.0);
    auto y6 = transformAxis!(double, exp, sturges)(x, 2.0, 12.0);
    auto y7 = transformAxis!(exp, sturges)(x, 2.0, 12.0);

    static assert(is(typeof(y4) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(y5) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(y6) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
    static assert(is(typeof(y7) == TransformAxis!(DefaultCountType, double, exp, log, AxisOptions())));
}

// Check number of bins
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: exp, log;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y = transformAxis!(size_t, double, exp, log, sturges, AxisOptions())(x, 2.0, 12.0);

    assert(y.N_bin == 4);
}

// Check all inverseTransform mappings
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp, exp2, log, log2, log10, sqrt;

    auto x0 = transformAxis!(size_t, double, exp, AxisOptions())(10, 2.0, 12.0);
    auto x1 = transformAxis!(size_t, double, exp2, AxisOptions())(10, 2.0, 12.0);
    auto x2 = transformAxis!(size_t, double, log, AxisOptions())(10, 2.0, 12.0);
    auto x3 = transformAxis!(size_t, double, log2, AxisOptions())(10, 2.0, 12.0);
    auto x4 = transformAxis!(size_t, double, log10, AxisOptions())(10, 2.0, 12.0);
    auto x5 = transformAxis!(size_t, double, sqrt, AxisOptions())(10, 2.0, 12.0);

    static assert(is(typeof(x0) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(x1) == TransformAxis!(size_t, double, exp2, log2, AxisOptions())));
    static assert(is(typeof(x2) == TransformAxis!(size_t, double, log, exp, AxisOptions())));
    static assert(is(typeof(x3) == TransformAxis!(size_t, double, log2, exp2, AxisOptions())));
    static assert(is(typeof(x4) == TransformAxis!(size_t, double, log10, exp10, AxisOptions())));
    static assert(is(typeof(x5) == TransformAxis!(size_t, double, sqrt, square, AxisOptions())));
}

// test string and lambda functions
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: exp, log;

    alias f = a => exp(a);
    alias g = a => log(a);

    auto x00 = transformAxis!(size_t, double, exp, log, AxisOptions())(10, 2.0, 12.0);
    auto x01 = transformAxis!(size_t, double, "exp(a)", "log(a)", AxisOptions())(10, 2.0, 12.0);
    auto x02 = transformAxis!(size_t, double, exp, "log(a)", AxisOptions())(10, 2.0, 12.0);
    auto x03 = transformAxis!(size_t, double, "exp(a)", log, AxisOptions())(10, 2.0, 12.0);
    auto x04 = transformAxis!(size_t, double, f, g, AxisOptions())(10, 2.0, 12.0);
    auto x05 = transformAxis!(size_t, double, exp, g, AxisOptions())(10, 2.0, 12.0);
    auto x06 = transformAxis!(size_t, double, f, log, AxisOptions())(10, 2.0, 12.0);
    auto x07 = transformAxis!(double, "exp(a)", "log(a)")(10, 2.0, 12.0);
    auto x08 = transformAxis!(double, exp, "log(a)")(10, 2.0, 12.0);
    auto x09 = transformAxis!(double, "exp(a)", log)(10, 2.0, 12.0);
    auto x10 = transformAxis!(double, f, g)(10, 2.0, 12.0);
    auto x11 = transformAxis!(double, exp, g)(10, 2.0, 12.0);
    auto x12 = transformAxis!(double, f, log)(10, 2.0, 12.0);
    auto x13 = transformAxis!("exp(a)", "log(a)")(10, 2.0, 12.0);
    auto x14 = transformAxis!(exp, "log(a)")(10, 2.0, 12.0);
    auto x15 = transformAxis!("exp(a)", log)(10, 2.0, 12.0);
    auto x16 = transformAxis!(f, g)(10, 2.0, 12.0);
    auto x17 = transformAxis!(exp, g)(10, 2.0, 12.0);
    auto x18 = transformAxis!(f, log)(10, 2.0, 12.0);

    static assert(is(typeof(x00) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(x01) == TransformAxis!(size_t, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(x02) == TransformAxis!(size_t, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(x03) == TransformAxis!(size_t, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(x04) == TransformAxis!(size_t, double, f, g, AxisOptions())));
    static assert(is(typeof(x05) == TransformAxis!(size_t, double, exp, g, AxisOptions())));
    static assert(is(typeof(x06) == TransformAxis!(size_t, double, f, log, AxisOptions())));
    static assert(is(typeof(x07) == TransformAxis!(DefaultCountType, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(x08) == TransformAxis!(DefaultCountType, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(x09) == TransformAxis!(DefaultCountType, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(x10) == TransformAxis!(DefaultCountType, double, f, g, AxisOptions())));
    static assert(is(typeof(x11) == TransformAxis!(DefaultCountType, double, exp, g, AxisOptions())));
    static assert(is(typeof(x12) == TransformAxis!(DefaultCountType, double, f, log, AxisOptions())));
    static assert(is(typeof(x13) == TransformAxis!(DefaultCountType, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(x14) == TransformAxis!(DefaultCountType, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(x15) == TransformAxis!(DefaultCountType, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(x16) == TransformAxis!(DefaultCountType, double, f, g, AxisOptions())));
    static assert(is(typeof(x17) == TransformAxis!(DefaultCountType, double, exp, g, AxisOptions())));
    static assert(is(typeof(x18) == TransformAxis!(DefaultCountType, double, f, log, AxisOptions())));
}

// test string and lambda functions with breaks
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: exp, log;
    import mir.ndslice.slice: sliced;
    import mir.stat.descriptive.histogram.breaks: sturges;

    alias f = a => exp(a);
    alias g = a => log(a);

    auto x = [0.0, 1, 2, 3, 4, 5, 6, 7].sliced;

    auto y00 = transformAxis!(size_t, double, exp, log, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y01 = transformAxis!(size_t, double, "exp(a)", "log(a)", sturges, AxisOptions())(x, 2.0, 12.0);
    auto y02 = transformAxis!(size_t, double, exp, "log(a)", sturges, AxisOptions())(x, 2.0, 12.0);
    auto y03 = transformAxis!(size_t, double, "exp(a)", log, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y04 = transformAxis!(size_t, double, f, g, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y05 = transformAxis!(size_t, double, exp, g, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y06 = transformAxis!(size_t, double, f, log, sturges, AxisOptions())(x, 2.0, 12.0);
    auto y07 = transformAxis!(double, "exp(a)", "log(a)", sturges)(x, 2.0, 12.0);
    auto y08 = transformAxis!(double, exp, "log(a)", sturges)(x, 2.0, 12.0);
    auto y09 = transformAxis!(double, "exp(a)", log, sturges)(x, 2.0, 12.0);
    auto y10 = transformAxis!(double, f, g, sturges)(x, 2.0, 12.0);
    auto y11 = transformAxis!(double, exp, g, sturges)(x, 2.0, 12.0);
    auto y12 = transformAxis!(double, f, log, sturges)(x, 2.0, 12.0);
    auto y13 = transformAxis!("exp(a)", "log(a)", sturges)(x, 2.0, 12.0);
    auto y14 = transformAxis!(exp, "log(a)", sturges)(x, 2.0, 12.0);
    auto y15 = transformAxis!("exp(a)", log, sturges)(x, 2.0, 12.0);
    auto y16 = transformAxis!(f, g, sturges)(x, 2.0, 12.0);
    auto y17 = transformAxis!(exp, g, sturges)(x, 2.0, 12.0);
    auto y18 = transformAxis!(f, log, sturges)(x, 2.0, 12.0);

    static assert(is(typeof(y00) == TransformAxis!(size_t, double, exp, log, AxisOptions())));
    static assert(is(typeof(y01) == TransformAxis!(size_t, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(y02) == TransformAxis!(size_t, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(y03) == TransformAxis!(size_t, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(y04) == TransformAxis!(size_t, double, f, g, AxisOptions())));
    static assert(is(typeof(y05) == TransformAxis!(size_t, double, exp, g, AxisOptions())));
    static assert(is(typeof(y06) == TransformAxis!(size_t, double, f, log, AxisOptions())));
    static assert(is(typeof(y07) == TransformAxis!(DefaultCountType, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(y08) == TransformAxis!(DefaultCountType, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(y09) == TransformAxis!(DefaultCountType, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(y10) == TransformAxis!(DefaultCountType, double, f, g, AxisOptions())));
    static assert(is(typeof(y11) == TransformAxis!(DefaultCountType, double, exp, g, AxisOptions())));
    static assert(is(typeof(y12) == TransformAxis!(DefaultCountType, double, f, log, AxisOptions())));
    static assert(is(typeof(y13) == TransformAxis!(DefaultCountType, double, "exp(a)", "log(a)", AxisOptions())));
    static assert(is(typeof(y14) == TransformAxis!(DefaultCountType, double, exp, "log(a)", AxisOptions())));
    static assert(is(typeof(y15) == TransformAxis!(DefaultCountType, double, "exp(a)", log, AxisOptions())));
    static assert(is(typeof(y16) == TransformAxis!(DefaultCountType, double, f, g, AxisOptions())));
    static assert(is(typeof(y17) == TransformAxis!(DefaultCountType, double, exp, g, AxisOptions())));
    static assert(is(typeof(y18) == TransformAxis!(DefaultCountType, double, f, log, AxisOptions())));
}

/++
Axis where the bins are made up of values from an enum.

Does not allow for overflow or underflow for improved performance.

Params:
    CountT = the type that is used to count in histogram bins
    BinT = the type of the values that are compared in histogram bins

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF TransformAxis),
    $(LREF CategoryAxis),
    $(LREF VariableAxis)
+/
struct EnumAxis(CountT, BinT)
    if (is(BinT == enum) &&
        EnumMembers!BinT.length == NoDuplicates!(EnumMembers!BinT).length)
{
    ///
    alias CountType = CountT;

    ///
    alias BinType = BinT;

    ///
    CountType N_bin()() const
    {
        import std.traits: EnumMembers;

        return EnumMembers!(BinType).length;
    }

    ///
    CountType index()(BinType value) const
    {
        import std.traits: OriginalType, EnumMembers;
        import mir.stat.descriptive.histogram.traits: isSwitchable;

        static if (isSwitchable!(OriginalType!BinType) && EnumMembers!BinType.length <= 50)
        {
            final switch (value)
            {
                foreach (size_t i, member; EnumMembers!BinType)
                {
                    case member:
                        return cast(CountType) i;
                }
            }
        }
        else
        {
            foreach (size_t i, member; EnumMembers!BinType)
            {
                if (value == member) {
                    return cast(CountType) i;
                }
            }
            assert(0, "EnumAxis.index: value is not an enum member");
        }
    }

    ///
    Bin!BinType bin(size_t x)() const
    {
        import std.traits: EnumMembers;

        assert(x < N_bin(), "EnumAxis.bin: input must be less than N_bin()");
        return Bin!(BinType)(EnumMembers!BinType[x]);
    }

    ///
    Bin!BinType bin()(size_t x) const
    {
        assert(x < N_bin(), "EnumAxis.bin: input must be less than N_bin()");

        import std.traits: OriginalType, EnumMembers;
        import mir.stat.descriptive.histogram.traits: isSwitchable;

        static if (isSwitchable!(OriginalType!BinType) && EnumMembers!BinType.length <= 50)
        {
            final switch (x)
            {
                foreach (size_t i, member; EnumMembers!BinType)
                {
                    case i:
                        return Bin!BinType(member);
                }
            }
        }
        else
        {
            foreach (size_t i, member; EnumMembers!BinType)
            {
                if (x == i) {
                    return Bin!BinType(member);
                }
            }
            assert(0, "EnumAxis.bin: index is out of range");
        }
    }
}

// Exercise both enum lookup fallbacks and their bounds checks.
version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.traits: EnumMembers;

    enum Fraction : double { half = 0.5, oneAndHalf = 1.5, twoAndHalf = 2.5 }
    enum Large
    {
        v00 = 100, v01, v02, v03, v04, v05, v06, v07, v08, v09,
        v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
        v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
        v30, v31, v32, v33, v34, v35, v36, v37, v38, v39,
        v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50
    }
    static assert(EnumMembers!Large.length == 51);

    void check(E)(E invalid)
    {
        EnumAxis!(size_t, E) axis;
        foreach (size_t i, member; EnumMembers!E)
        {
            assert(axis.index(member) == i);
            assert(axis.bin(i) == Bin!E(member));
            assert(axis.bin!i() == Bin!E(member));
        }
        assertThrown!AssertError(axis.index(invalid));
        assertThrown!AssertError(axis.bin(axis.N_bin));
        assertThrown!AssertError(axis.bin(axis.N_bin + 1));
        static assert(!__traits(compiles, axis.bin!(EnumMembers!E.length)()));
    }

    check!Fraction(cast(Fraction) 1.0);
    check!Large(cast(Large) 99);

    // The switch path must reject the one-past-end index at the same guard.
    enum Small { first, second }
    EnumAxis!(size_t, Small) small;
    assertThrown!AssertError(small.bin(small.N_bin));
    static assert(!__traits(compiles, small.bin!2()));
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    enum Foo
    {
        A,
        B,
        C
    }
    EnumAxis!(size_t, Foo) enumAxis;

    assert(enumAxis.index(Foo.A) == 0);
    assert(enumAxis.index(Foo.B) == 1);
    assert(enumAxis.index(Foo.C) == 2);

    assert(enumAxis.bin(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin(2) == Bin!Foo(Foo.C));

    assert(enumAxis.bin!(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin!(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin!(2) == Bin!Foo(Foo.C));
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    enum Foo : string
    {
        A = "Z",
        B = "Y",
        C = "X"
    }
    EnumAxis!(size_t, Foo) enumAxis;

    assert(enumAxis.index(Foo.A) == 0);
    assert(enumAxis.index(Foo.B) == 1);
    assert(enumAxis.index(Foo.C) == 2);

    assert(enumAxis.bin(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin(2) == Bin!Foo(Foo.C));

    assert(enumAxis.bin!(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin!(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin!(2) == Bin!Foo(Foo.C));
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    enum Foo
    {
        A = 0,
        B = 1,
        C = 3
    }
    EnumAxis!(size_t, Foo) enumAxis;

    assert(enumAxis.index(Foo.A) == 0);
    assert(enumAxis.index(Foo.B) == 1);
    assert(enumAxis.index(Foo.C) == 2);

    assert(enumAxis.bin(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin(2) == Bin!Foo(Foo.C));

    assert(enumAxis.bin!(0) == Bin!Foo(Foo.A));
    assert(enumAxis.bin!(1) == Bin!Foo(Foo.B));
    assert(enumAxis.bin!(2) == Bin!Foo(Foo.C));
}

/++
Factory function to produce $(LREF EnumAxis)

Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins

See_also:
    $(LREF EnumAxis)
+/
EnumAxis!(CountType, BinType) enumAxis(CountType, BinType)()
{
    return EnumAxis!(CountType, BinType)();
}

/// ditto
EnumAxis!(DefaultCountType, BinType) enumAxis(BinType)()
{
    return .enumAxis!(DefaultCountType, BinType)();
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    enum Foo
    {
        A,
        B,
        C
    }
    auto x0 = enumAxis!(size_t, Foo);
    auto x1 = enumAxis!Foo;

    static assert(is(typeof(x0) == EnumAxis!(size_t, Foo)));
    static assert(is(typeof(x1) == EnumAxis!(DefaultCountType, Foo)));
}

/++
Axis similar to EnumAxis, but allows for overflow for when a string is passed
that does not match with enum members of `BinT`.

Params:
    CountT = the type that is used to count in histogram bins
    BinT = the type of the values that are compared in histogram bins
    axisOptions = options

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF TransformAxis),
    $(LREF EnumAxis),
    $(LREF VariableAxis)
+/
struct CategoryAxis(CountT, BinT, AxisOptions axisOptions)
    if (is(BinT == enum) &&
        EnumMembers!BinT.length == NoDuplicates!(EnumMembers!BinT).length)
{
    import std.traits: isSomeString;

    ///
    EnumAxis!(CountT, BinT) enumAxis;

    ///
    alias CountType = CountT;

    ///
    alias BinType = BinT;

    ///
    alias options = axisOptions;

    ///
    CountType N_bin()() const
    {
        return enumAxis.N_bin;
    }

    ///
    CountType index()(BinType value) const
    {
        return enumAxis.index(value);
    }

    ///
    CountType index(A)(A value) const
        if (isSomeString!A)
    {
        import mir.conv: to;
        try
        {
            return this.index(value.to!BinType);
        }
        catch (Exception e)
        {
            assert(0, "CategoryAxis.index: string value does not convert to enum, index is invalid here, increment overflow instead");
        }
    }

    ///
    Bin!BinType bin(size_t x)() const
    {
        return enumAxis.bin!x;
    }

    ///
    Bin!BinType bin()(size_t x) const
    {
        return enumAxis.bin(x);
    }

    ///
    bool isOverflow()(BinType value) const
    {
        return false;
    }

    ///
    bool isOverflow(A : const(char)[])(A value) const
    {
        import mir.conv: to;
        try
        {
            BinType x = value.to!BinType;
            return false;
        }
        catch (Exception e)
        {
            return true;
        }
    }

    ///
    bool isOverflow(A : const(char))(A value) const
    {
        import mir.conv: to;
        try
        {
            BinType x = value.to!BinType;
            return false;
        }
        catch (Exception e)
        {
            return true;
        }
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    enum Foo
    {
        A,
        B,
        C
    }
    CategoryAxis!(size_t, Foo, AxisOptions()) categoryAxis;

    assert(categoryAxis.index(Foo.A) == 0);
    assert(categoryAxis.index(Foo.B) == 1);
    assert(categoryAxis.index(Foo.C) == 2);

    assert(categoryAxis.index("A") == 0);
    assert(categoryAxis.index("B") == 1);
    assert(categoryAxis.index("C") == 2);

    assert(categoryAxis.N_bin == 3);

    assert(categoryAxis.bin(0) == Bin!Foo(Foo.A));
    assert(categoryAxis.bin(1) == Bin!Foo(Foo.B));
    assert(categoryAxis.bin(2) == Bin!Foo(Foo.C));

    assert(categoryAxis.bin!(0) == Bin!Foo(Foo.A));
    assert(categoryAxis.bin!(1) == Bin!Foo(Foo.B));
    assert(categoryAxis.bin!(2) == Bin!Foo(Foo.C));

    assert(!categoryAxis.isOverflow(Foo.B));
	assert(!categoryAxis.isOverflow("B"));
	assert(categoryAxis.isOverflow("D"));
}

// Check that assert thrown when string input does not match enum
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;

    enum Foo
    {
        A,
        B,
        C
    }
    CategoryAxis!(size_t, Foo, AxisOptions()) categoryAxis;

    assertThrown!AssertError(categoryAxis.index("D"));
}

/++
Factory function to produce $(LREF CategoryAxis)

Params:
    CountType = the type that is used to count in histogram bins
    BinType = the type of the values that are compared in histogram bins
    axisOptions = options

See_also:
    $(LREF CategoryAxis)
+/
CategoryAxis!(CountType, BinType, axisOptions)
    categoryAxis(CountType, BinType, AxisOptions axisOptions = AxisOptions())()
{
    return CategoryAxis!(CountType, BinType, axisOptions)();
}

/// ditto
CategoryAxis!(DefaultCountType, BinType, axisOptions)
    categoryAxis(BinType, AxisOptions axisOptions = AxisOptions())()
{
    return .categoryAxis!(DefaultCountType, BinType, axisOptions)();
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    enum Foo
    {
        A,
        B,
        C
    }

    auto x0 = categoryAxis!(size_t, Foo, AxisOptions());
    auto x1 = categoryAxis!(Foo, AxisOptions());
    auto x2 = categoryAxis!Foo;

    static assert(is(typeof(x0) == CategoryAxis!(size_t, Foo, AxisOptions())));
    static assert(is(typeof(x1) == CategoryAxis!(DefaultCountType, Foo, AxisOptions())));
    static assert(is(typeof(x2) == CategoryAxis!(DefaultCountType, Foo, AxisOptions())));
}

/++
Axis for non-equidistant data.

Params:
    CountT = the type that is used to count in histogram bins
    Iterator = iterator type for the bin boundaries
    axisOptions = options

See_also:
    $(LREF AxisOptions),
    $(LREF IntegralAxis),
    $(LREF RegularAxis),
    $(LREF TransformAxis),
    $(LREF EnumAxis),
    $(LREF CategoryAxis)
+/
struct VariableAxis(CountT, Iterator, AxisOptions axisOptions)
{
    import mir.primitives: DeepElementType;
    import mir.ndslice.slice: Slice, SliceKind;

private:
    Slice!(Iterator) _payload;

public:

    ///
    alias CountType = CountT;

    ///
    alias BinType = DeepElementType!(Slice!(Iterator));

    ///
    alias options = axisOptions;

    /++
    Boundaries must define at least one bin, be strictly increasing, and have a
    bin count representable by CountType. Keep shared boundaries unchanged
    while using the axis, including through external aliases.
    +/
    this(It, SliceKind kind)(Slice!(It, 1LU, kind) slice)
    {
        assert(slice.length >= 2, "VariableAxis.this: at least two boundaries required");
        assert(slice.length - 1 <= CountType.max,
            "VariableAxis.this: bin count does not fit CountType");
        assert(strictlyIncreasing(slice),
            "VariableAxis.this: boundaries must be strictly increasing");
        import core.lifetime: move;
        _payload = move(slice);
    }

    private static bool strictlyIncreasing(S)(ref S slice)
    {
        foreach (i; 1 .. slice.length)
            if (!(slice[i - 1] < slice[i]))
                return false;
        return true;
    }

    /++
    Copy the axis handle with read-only access to its break values.
    Reference-counted iterators retain ownership; borrowed iterators stay borrowed.
    +/
    auto lightConst()() const @property
    {
        import mir.qualifier: LightConstOf;
        // Reuse already validated boundaries without scanning them on each view.
        VariableAxis!(CountType, LightConstOf!Iterator, axisOptions) result;
        result._payload = _payload.lightConst;
        return result;
    }

    ///
    CountType N_bin()() const
    {
        return cast(CountType) _payload.length - 1;
    }

    ///
    BinType low()() const
    {
        return _payload[0];
    }

    ///
    BinType high()() const
    {
        return _payload[$ - 1];
    }

    ///
    bool isUnderflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed &&
                   !axisOptions.isCircular) {
            return x <= low();
        } else {
            return x < low();
        }
    }

    ///
    bool isOverflow()(BinType x) const
    {
        static if (axisOptions.isRightClosed ||
                   axisOptions.isCircular) {
            return x > high();
        } else {
            return x >= high();
        }
    }

    ///
    CountType index()(BinType x)
    {
        import mir.stat.descriptive.histogram.traits: checkOverUnderFlow;

        checkOverUnderFlow!(BinType, axisOptions)(x, low(), high());

        // Search a borrowed slice while this axis retains the backing storage.
        // This avoids copying reference-counted iterators inside Phobos's
        // SortedRange, whose slicing path is not DIP1000-safe for those iterators.
        static if (!axisOptions.isRightClosed) {
            static if (axisOptions.isCircular) {
                if (x == high()) {
                    return cast(CountType) 0;
                }
            }
            import std.range: assumeSorted;
            return cast(CountType)
                (_payload.lightScope.assumeSorted!("a <= b").lowerBound(x).length - 1);
        } else {
            static if (axisOptions.isCircular) {
                if (x == low()) {
                    return cast(CountType) (N_bin() - 1);
                }
            }
            import std.range: assumeSorted;
            return cast(CountType)
                (_payload.lightScope.assumeSorted!("a < b").lowerBound(x).length - 1);
        }
    }

    /++
    Return the two boundary values for a bin.
    Numeric boundaries are independent snapshots: changing the returned bin
    does not change the axis or its boundary storage. For custom boundary types,
    copying a value does not deep-copy any references it contains.
    +/
    auto bin()(size_t x) const
    {
        import std.traits: Unqual;
        assert(x < N_bin, "VariableAxis.bin: index is out of range");
        return Bin!(Unqual!BinType)(_payload[x], _payload[x + 1]);
    }
}

// Variable-axis indices use CountType for both interval conventions.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;

    auto breaks = [0.0, 1.0, 3.0, 4.0].sliced;
    auto left = VariableAxis!(uint, double*, AxisOptions())(breaks);
    static assert(is(typeof(left.index(0.5)) == uint));
    assert(left.index(0.0) == 0u);
    assert(left.index(1.0) == 1u);
    assert(left.index(2.0) == 1u);
    assert(left.index(3.0) == 2u);

    auto right = VariableAxis!(uint, double*, AxisOptions(IsRightClosed(true)))(breaks);
    static assert(is(typeof(right.index(0.5)) == uint));
    assert(right.index(1.0) == 0u);
    assert(right.index(2.0) == 1u);
    assert(right.index(3.0) == 1u);
    assert(right.index(4.0) == 2u);
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(double)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = i + 2.0;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(double), AxisOptions())(counts.asSlice);
    assert(variableAxis.N_bin == (counts.length - 1));
    assert(variableAxis.low == 2.0);
    assert(variableAxis.high == 12.0);

    assert(!variableAxis.isOverflow(5.0));
    assert(!variableAxis.isUnderflow(5.0));
    assert(variableAxis.isOverflow(13.0));
    assert(variableAxis.isUnderflow(1.0));

    assert(variableAxis.index(2.0) == 0);
    assert(variableAxis.index(2.5) == 0);
    assert(variableAxis.index(3.0) == 1);
    assert(variableAxis.index(11.5) == 9);

    assert(variableAxis.bin(0).low == 2.0);
    assert(variableAxis.bin(0).high == 3.0);
    assert(variableAxis.bin(1).low == 3.0);
    assert(variableAxis.bin(1).high == 4.0);
    assert(variableAxis.bin(9).low == 11.0);
    assert(variableAxis.bin(9).high == 12.0);
}

// With isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(double)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = i + 2.0;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(double), AxisOptions(true))(counts.asSlice);

    assert(variableAxis.index(2.5) == 0);
    assert(variableAxis.index(3.0) == 0);
    assert(variableAxis.index(3.5) == 1);
    assert(variableAxis.index(4.0) == 1);
    assert(variableAxis.index(4.5) == 2);
    assert(variableAxis.index(5.0) == 2);
    assert(variableAxis.index(5.5) == 3);
    assert(variableAxis.index(6.0) == 3);
    assert(variableAxis.index(12.0) == 9);
    assert(variableAxis.index(11.5) == 9);
}

// Some more tests
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(double)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = i + 2.0;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(double), AxisOptions())(counts.asSlice);

    assert(variableAxis.index(3.5) == 1);
    assert(variableAxis.index(4.0) == 2);
    assert(variableAxis.index(4.5) == 2);
    assert(variableAxis.index(5.0) == 3);
    assert(variableAxis.index(5.5) == 3);
    assert(variableAxis.index(6.0) == 4);
}

// integral test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(int)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = cast(int) i + 2;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(int), AxisOptions())(counts.asSlice);

    assert(variableAxis.index(2) == 0);
    assert(variableAxis.index(4) == 2);
    assert(variableAxis.index(5) == 3);
    assert(variableAxis.index(6) == 4);
}

// integral test, isRightClosed = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(int)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = cast(int) i + 2u;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(int), AxisOptions(true))(counts.asSlice);

    assert(variableAxis.index(4) == 1);
    assert(variableAxis.index(5) == 2);
    assert(variableAxis.index(6) == 3);
    assert(variableAxis.index(12) == 9);
}

// integral test, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(int)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = cast(int) i + 2u;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(int), AxisOptions(IsCircular(true)))(counts.asSlice);

    assert(variableAxis.index(2) == 0);
    assert(variableAxis.index(5) == 3);
    assert(variableAxis.index(12) == 0);
}

// integral test, isRightClosed = true, isCircular = true
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(int)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = cast(int) i + 2u;
        i++;
    }
    auto variableAxis = VariableAxis!(size_t, RCI!(int), AxisOptions(IsRightClosed(true), IsCircular(true)))(counts.asSlice);

    assert(variableAxis.index(2) == 9);
    assert(variableAxis.index(5) == 2);
    assert(variableAxis.index(12) == 9);
}

/++
Factory function to produce $(LREF VariableAxis) object

Params:
    CountType = the type that is used to count in histogram bins
    Iterator = the type of the values that are compared in histogram bins
    axisOptions = options

See_also:
    $(LREF VariableAxis)
+/
template variableAxis(CountType, Iterator, AxisOptions axisOptions = AxisOptions())
{
    import core.lifetime: move;
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
    +/
    VariableAxis!(CountType, Iterator, axisOptions)
        variableAxis(size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        return VariableAxis!(CountType, Iterator, axisOptions)(slice.move);
    }
}

/++
Params:
    Iterator = the type of the values that are compared in histogram bins
    axisOptions = options
+/
template variableAxis(Iterator, AxisOptions axisOptions = AxisOptions())
{
    import core.lifetime: move;
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
    +/
    VariableAxis!(DefaultCountType, Iterator, axisOptions)
        variableAxis(size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        return .variableAxis!(DefaultCountType, Iterator, axisOptions)(slice.move);
    }
}

/++
Params:
    axisOptions = options
+/
template variableAxis(AxisOptions axisOptions = AxisOptions())
{
    import core.lifetime: move;
    import mir.ndslice.slice: Slice, SliceKind;

    /++
    Params:
        slice = slice
    +/
    VariableAxis!(DefaultCountType, Iterator, axisOptions)
        variableAxis(Iterator, size_t N, SliceKind kind)(Slice!(Iterator, N, kind) slice)
    {
        return .variableAxis!(DefaultCountType, Iterator, axisOptions)(slice.move);
    }
}

/// Example
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.traits: DefaultCountType;

    import mir.rc.array;

    size_t len = 11;
    auto counts = mininitRcarray!(double)(len);
    size_t i = 0;
    while (i < len)
    {
        counts[i] = i + 2.0;
        i++;
    }

    auto x0 = variableAxis!(size_t, RCI!(double), AxisOptions())(counts.asSlice);
    auto x1 = variableAxis!(RCI!(double))(counts.asSlice);
    auto x2 = variableAxis(counts.asSlice);

    static assert(is(typeof(x0) == VariableAxis!(size_t, RCI!(double), AxisOptions())));
    static assert(is(typeof(x1) == VariableAxis!(DefaultCountType, RCI!(double), AxisOptions())));
    static assert(is(typeof(x2) == VariableAxis!(DefaultCountType, RCI!(double), AxisOptions())));
}


// RC-backed lookups remain safe under DIP1000 and retain their break storage.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.allocation: rcslice;
    import mir.rc.array: RCI;

    auto makeAxis(bool rightClosed)()
    {
        // Only the returned axis retains this allocation after the call.
        auto breaks = rcslice!double([0.0, 1.0, 3.0, 6.0]);
        return VariableAxis!(uint, RCI!double, AxisOptions(rightClosed))(breaks);
    }

    void check(Axis)(ref Axis axis) @safe pure nothrow @nogc
    {
        static if (Axis.options.isRightClosed)
        {
            assert(axis.index(1.0) == 0);
            assert(axis.index(3.0) == 1);
            assert(axis.index(6.0) == 2);
        }
        else
        {
            assert(axis.index(0.0) == 0);
            assert(axis.index(1.0) == 1);
            assert(axis.index(3.0) == 2);
        }
        assert(axis.index(0.5) == 0);
        assert(axis.index(2.0) == 1);
        assert(axis.index(5.0) == 2);
        assert(axis.low == 0 && axis.high == 6 && axis.N_bin == 3);
    }

    auto left = makeAxis!false();
    auto right = makeAxis!true();
    check(left);
    check(right);
}

// Valid observations near endpoints must always map to ordinary bins.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    import mir.ndslice.allocation: rcslice;

    static foreach (rightClosed; [false, true])
    static foreach (circular; [false, true])
    {{
        alias A = RegularAxis!(uint, double, AxisOptions(rightClosed, true, true, circular));
        auto axis = A(2, -1.0, 1.0);
        // Exact hexadecimal literals represent the adjacent doubles inside each end.
        assert(axis.index(0x1.fffffffffffffp-1) == 1);
        assert(axis.index(-0x1.fffffffffffffp-1) == 0);
        assert(axis.index(0.0) == (rightClosed ? 0 : 1));
        static immutable uint[4] zero = [0, 0, 0, 0];
        auto counts = rcslice!uint(zero[]);
        auto h = HistogramAccumulator!(typeof(counts), A)(counts, axis);
        h.put(0x1.fffffffffffffp-1);
        h.put(-0x1.fffffffffffffp-1);
        assert(h.counts[1] == 1 && h.counts[2] == 1);
        assert(h.underflow == 0 && h.overflow == 0);
        if (circular)
        {
            assert(axis.index(1.0) == (rightClosed ? 1 : 0));
            assert(axis.index(-1.0) == (rightClosed ? 1 : 0));
        }
    }}
    // An identity transform exercises the same endpoint rounding through delegation.
    alias Transformed = TransformAxis!(uint, double, "a", "a", AxisOptions());
    auto transformed = Transformed(2, -1.0, 1.0);
    assert(transformed.index(0x1.fffffffffffffp-1) == 1);
    // Underflow during normalization at the lower end of a right-closed axis.
    auto tiny = RegularAxis!(uint, double, AxisOptions(true))(2, 0.0, 4.0);
    assert(tiny.index(0x0.0000000000001p-1022) == 0);
}

// Integral endpoints must not truncate the count or overflow during addition.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    alias WideCount = IntegralAxis!(ulong, uint, AxisOptions());
    assertThrown!AssertError(WideCount(0x1_0000_0001UL, 0u));
    // A representable count can still overflow when added to the lower bound.
    assertThrown!AssertError(WideCount(2, uint.max - 1));
    auto unsignedLimit = WideCount(uint.max, 0u);
    assert(unsignedLimit.high == uint.max);
    assert(unsignedLimit.index(uint.max - 1) == uint.max - 1);

    alias Signed = IntegralAxis!(uint, int, AxisOptions());
    assertThrown!AssertError(Signed(cast(uint) int.max + 1, 0));
    assertThrown!AssertError(Signed(2, int.max - 1));
    auto signedLimit = Signed(2, int.max - 2);
    assert(signedLimit.high == int.max);
    assert(signedLimit.index(int.max - 1) == 1);
    // Negative lower bounds remain valid, including at the signed minimum.
    auto negative = Signed(2, int.min);
    assert(negative.high == int.min + 2);
    assert(negative.index(int.min + 1) == 1);
    auto crossingZero = Signed(2, -1);
    assert(crossingZero.high == 1);
    assert(crossingZero.index(0) == 1);
}

// Reject malformed axes at construction, before indexing or allocation.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.ndslice.slice: sliced;

    alias I = IntegralAxis!(uint, double, AxisOptions());
    alias R = RegularAxis!(uint, double, AxisOptions());
    assertThrown!AssertError(I(0, 0.0));
    assertThrown!AssertError(I(2, double.nan));
    assertThrown!AssertError(I(2, double.infinity));
    assertThrown!AssertError(R(0, 0.0, 1.0));
    assertThrown!AssertError(R(2, 1.0, 1.0));
    assertThrown!AssertError(R(2, double.nan, 1.0));
    assertThrown!AssertError(R(2, 0.0, double.infinity));
    alias T = TransformAxis!(uint, double, "a", "a", AxisOptions());
    assertThrown!AssertError(T(0, 0.0, 1.0));
    alias V = VariableAxis!(uint, double*, AxisOptions());
    foreach (breaks; [cast(double[]) [], [0.0], [0.0, 0.0],
                      [0.0, 2.0, 1.0], [0.0, double.nan, 2.0]])
        assertThrown!AssertError(V(breaks.sliced));
    auto many = new double[257];
    foreach (i, ref value; many) value = i;
    alias Small = VariableAxis!(ubyte, double*, AxisOptions());
    assertThrown!AssertError(Small(many.sliced));
    assert(V([0.0, 1.0].sliced).N_bin == 1);
}

// Extreme endpoint observations retain their bins and flow classification.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    import mir.ndslice.allocation: rcslice;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T smallest = nextUp(T(0));
        // Start at a power of two so these four upward ULPs have equal spacing.
        const T large = nextUp(T.max / 4);
        const T narrowHigh = nextUp(nextUp(nextUp(nextUp(large))));
        T[2][7] bounds = [
            [T(-1), T(1)], [T(0), T(2)],
            [T(0), smallest * 4], [-T.min_normal, T.min_normal],
            [large, narrowHigh], [T(0), T.max / 2], [-T.max / 4, T.max / 4]];
        foreach (interval; bounds)
        {
            static foreach (right; [false, true])
            static foreach (circular; [false, true])
            {{
                alias A = RegularAxis!(uint, T, AxisOptions(right, true, true, circular));
                auto axis = A(2, interval[0], interval[1]);
                const T insideLow = nextUp(interval[0]);
                const T insideHigh = nextDown(interval[1]);
                assert(axis.index(insideLow) == 0);
                assert(axis.index(insideHigh) == 1);
                assert(axis.isUnderflow(nextDown(interval[0])));
                assert(axis.isOverflow(nextUp(interval[1])));
                assert(axis.isUnderflow(-T.infinity));
                assert(axis.isOverflow(T.infinity));
                if (circular)
                {
                    assert(axis.index(interval[0]) == (right ? 1 : 0));
                    assert(axis.index(interval[1]) == (right ? 1 : 0));
                }
                else if (right)
                {
                    assert(axis.isUnderflow(interval[0]));
                    assert(axis.index(interval[1]) == 1);
                }
                else
                {
                    assert(axis.index(interval[0]) == 0);
                    assert(axis.isOverflow(interval[1]));
                }
                static immutable uint[4] zero = [0, 0, 0, 0];
                auto counts = rcslice!uint(zero[]);
                auto h = HistogramAccumulator!(typeof(counts), A)(counts, axis);
                h.put(insideLow, insideHigh, -T.infinity, T.infinity);
                assert(h.counts[1] == 1 && h.counts[2] == 1);
                assert(h.underflow == 1 && h.overflow == 1);
            }}
        }
        // On a zero-based axis, adjacent values around the exact interior edge
        // do not lose precision through subtraction of a nonzero lower bound.
        auto left = RegularAxis!(uint, T, AxisOptions())(2, T(0), T(2));
        auto right = RegularAxis!(uint, T, AxisOptions(true))(2, T(0), T(2));
        assert(left.index(nextDown(T(1))) == 0);
        assert(left.index(T(1)) == 1);
        assert(left.index(nextUp(T(1))) == 1);
        assert(right.index(nextDown(T(1))) == 0);
        assert(right.index(T(1)) == 0);
        assert(right.index(nextUp(T(1))) == 1);
        auto tiny = RegularAxis!(uint, T, AxisOptions(true))(2, T(0), T(4));
        assert(tiny.index(smallest) == 0);
    }}
}

// Reject invalid extreme configurations and NaN observations before updating counts.
version(mir_stat_test)
pure
unittest
{
    import std.meta: AliasSeq;
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        alias A = RegularAxis!(uint, T, AxisOptions(false, true, true));
        assertThrown!AssertError(A(2, -T.max, T.max));
        assertThrown!AssertError(A(2, T(0), T.infinity));
        assertThrown!AssertError(A(2, -T.infinity, T(0)));
        assertThrown!AssertError(A(2, T.nan, T(1)));
        auto axis = A(2, T(-1), T(1));
        assertThrown!AssertError(axis.index(T.nan));
        auto h = HistogramAccumulator!(uint[], A)([0u, 0u, 0u, 0u], axis);
        assertThrown!AssertError(h.put(T.nan));
        assert(h.counts == [0u, 0u, 0u, 0u]);
        assert(h.underflow == 0 && h.overflow == 0);
    }}
}

// Independently scan public bin descriptions to check boundary lookup. This
// deliberately does not use the normalized candidate or the binary-search helper.
version(mir_stat_test)
private void checkBoundaryMembership(Axis)(ref Axis axis) @safe pure nothrow @nogc
{
    import std.math: nextUp, nextDown;
    import mir.ndslice.allocation: mininitRcslice;
    import mir.stat.descriptive.histogram.accumulator: HistogramAccumulator;
    alias T = Axis.BinType;
    const n = cast(size_t) axis.N_bin;
    import mir.stat.descriptive.histogram.traits: storageExtent, includeUnderflow;
    const extent = storageExtent(axis);
    auto counts = mininitRcslice!uint(extent);
    auto expectedCounts = mininitRcslice!uint(extent);
    foreach (i; 0 .. extent) { counts[i] = 0; expectedCounts[i] = 0; }
    auto histogram = HistogramAccumulator!(typeof(counts), Axis)(counts, axis);
    assert(axis.bin(0).low == axis.low);
    assert(axis.bin(n - 1).high == axis.high);
    foreach (i; 0 .. n)
    {
        auto bin = axis.bin(i);
        assert(bin.low < bin.high);
        if (i + 1 < n)
            assert(bin.high == axis.bin(i + 1).low);
        foreach (edge; [bin.low, bin.high])
        foreach (x; [nextDown(edge), edge, nextUp(edge)])
        {
            if (axis.isUnderflow(x) || axis.isOverflow(x))
                continue;
            size_t expected = n;
            static if (Axis.options.isCircular)
            {
                static if (Axis.options.isRightClosed)
                {
                    if (x == axis.low) expected = n - 1;
                }
                else if (x == axis.high) expected = 0;
            }
            if (expected == n)
                foreach (j; 0 .. n)
                {
                    auto interval = axis.bin(j);
                    static if (Axis.options.isRightClosed)
                        const contains = interval.low < x && x <= interval.high;
                    else
                        const contains = interval.low <= x && x < interval.high;
                    if (contains) { expected = j; break; }
                }
            assert(expected < n);
            assert(axis.index(x) == expected);
            histogram.put(x);
            ++expectedCounts[expected + includeUnderflow!Axis];
        }
    }
    assert(counts == expectedCounts);
    assert(histogram.underflow == 0 && histogram.overflow == 0);
}

// Wide and narrow intervals, neighboring representable values, and both closures.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (right; [false, true])
    static foreach (circular; [false, true])
    {{
        alias A = RegularAxis!(uint, T, AxisOptions(right, true, true, circular));
        T[2][4] intervals = [[T(-1), T(1)], [T(0.1), T(1.1)],
            [-T.max / 4, T.max / 4], [-T.min_normal, T.min_normal]];
        foreach (n; [2u, 3u, 10u, 100u])
        foreach (bounds; intervals)
        {
            auto axis = A(n, bounds[0], bounds[1]);
            checkBoundaryMembership(axis);
        }
        // Only a few values are representable in these spans. The bins still
        // have distinct edges; their interiors need not contain another value.
        const T small = nextUp(T(0));
        auto subnormal = A(4, T(0), small * 4);
        checkBoundaryMembership(subnormal);
        const T large = nextUp(T.max / 4);
        auto narrow = A(4, large, nextUp(nextUp(nextUp(nextUp(large)))));
        checkBoundaryMembership(narrow);

        // Direct regression for cancellation at an exactly representable edge.
        auto zeroCrossing = A(2, T(-1), T(1));
        assert(zeroCrossing.index(nextDown(T(0))) == 0);
        assert(zeroCrossing.index(T(0)) == (right ? 0 : 1));
        assert(zeroCrossing.index(nextUp(T(0))) == 1);
    }}
}

// Forward-transform estimates are checked against inverse-transformed edges.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: log10, sqrt;
    static foreach (T; AliasSeq!(float, double, real))
    static foreach (right; [false, true])
    static foreach (circular; [false, true])
    {{
        alias Log = TransformAxis!(uint, T, (T x) => cast(T) log10(x),
            (T x) => cast(T)(T(10) ^^ x), AxisOptions(right, true, true, circular));
        auto logarithmic = Log(20, T(1), T(1.0e12));
        checkBoundaryMembership(logarithmic);
        alias Root = TransformAxis!(uint, T, (T x) => cast(T) sqrt(x),
            (T x) => x * x, AxisOptions(right, true, true, circular));
        auto squareRoot = Root(20, T(0), T(1.0e12));
        checkBoundaryMembership(squareRoot);
    }}
}

// Reject collapsed rounded grids, including collapse caused by the inverse transform.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;
    import std.math: nextUp, sqrt;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        alias A = RegularAxis!(uint, T, AxisOptions());
        assertThrown!AssertError(A(4, T(1), nextUp(T(1))));
        const T small = nextUp(T(0));
        assertThrown!AssertError(A(2, T(0), small));
        // A single bin needs only its two distinct endpoints.
        auto one = A(1, T(0), small);
        assert(one.index(T(0)) == 0);
        assert(one.bin(0).high == small);
        alias Root = TransformAxis!(uint, T, (T x) => cast(T) sqrt(x),
            (T x) => x * x, AxisOptions());
        assertThrown!AssertError(Root(2, T(0), small));
    }}
    alias Constant = TransformAxis!(uint, double, "a", "0.0", AxisOptions());
    alias Reversed = TransformAxis!(uint, double, "a", "1.0 - a", AxisOptions());
    alias Invalid = TransformAxis!(uint, double, "a", "double.nan", AxisOptions());
    assertThrown!AssertError(Constant(4, 0.0, 1.0));
    assertThrown!AssertError(Reversed(4, 0.0, 1.0));
    assertThrown!AssertError(Invalid(4, 0.0, 1.0));
}

// Unit-width grids must reject rounded-away steps in every floating-point type.
version(mir_stat_test)
pure
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (right; [false, true])
    {{
        alias A = IntegralAxis!(uint, T, AxisOptions(right));
        // At this power of two, the spacing above it is two rather than one.
        const T limit = T(2) ^^ T.mant_dig;
        assertThrown!AssertError(A(4, limit));
        assertThrown!AssertError(A(4, -limit - T(4)));
        assertThrown!AssertError(A(4, limit - T(2)));
        assertThrown!AssertError(A(4, T.max));
        assertThrown!AssertError(A(4, T.infinity));
        assertThrown!AssertError(A(4, -T.infinity));
        assertThrown!AssertError(A(4, T.nan));
    }}
}

// Check public bin edges and neighboring values, including fractional origins
// where subtraction can round an observation onto the wrong side of an edge.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;

    static foreach (T; AliasSeq!(float, double, real))
    static foreach (right; [false, true])
    static foreach (circular; [false, true])
    {{
        alias A = IntegralAxis!(uint, T, AxisOptions(right, true, true, circular));
        const T limit = T(2) ^^ T.mant_dig;
        // These grids touch the precision limit without crossing into the
        // region where adjacent unit steps collapse.
        auto positive = A(4, limit - T(4));
        auto negative = A(4, -limit);
        checkBoundaryMembership(positive);
        checkBoundaryMembership(negative);
        foreach (low; [T(0.1), T(-0.1), nextUp(T(0)), nextDown(T(0))])
        {
            auto fractional = A(8, low);
            checkBoundaryMembership(fractional);
        }
    }}
}

// Rejected template instantiations must produce false, not a hard error while
// overload resolution considers a candidate with an incompatible bin type.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log10;
    static assert(isTransformFunction!(log10, double));
    static assert(!isTransformFunction!(log10, uint));
    static assert(!isTransformFunction!(log10, string));
    template notAFunction(T) { enum notAFunction = 1; }
    static assert(!isTransformFunction!(notAFunction, double));
    static assert(isTransformFunction!("a * 2", double));
    static assert(!isTransformFunction!("a.missingMember", double));
}

// Copied numeric bins are independent of boundary qualifiers and storage lifetime.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import std.meta: AliasSeq;

    static foreach (T; AliasSeq!(float, double, real))
    {{
        static foreach (Q; AliasSeq!(T, const(T), immutable(T)))
        {{
            Q[3] edges = [0, 1, 3];
            alias A = VariableAxis!(uint, Q*, AxisOptions());
            auto axis = A(edges[].sliced);
            auto bin = axis.bin(1);
            static assert(is(typeof(bin) == Bin!T));
            assert(bin.low == 1 && bin.high == 3);
            bin.low = 2;
            assert(edges[1] == 1 && axis.bin(1).low == 1);
            const reader = axis;
            assert(reader.bin(1) == Bin!T(1, 3));
        }}
        static Bin!T fromLocal() @safe pure nothrow @nogc
        {
            T[3] edges = [0, 1, 3];
            const axis = VariableAxis!(uint, T*, AxisOptions())(edges[].sliced);
            return axis.bin(1);
        }
        assert(fromLocal() == Bin!T(1, 3));
    }}
}

// Both the first invalid index and an extreme index are rejected.
version(mir_stat_test)
@system pure
unittest
{
    import mir.ndslice.slice: sliced;
    import std.exception: assertThrown;
    import core.exception: AssertError;
    double[3] edges = [0, 1, 3];
    auto axis = VariableAxis!(uint, double*, AxisOptions())(edges[].sliced);
    assertThrown!AssertError(axis.bin(2));
    assertThrown!AssertError(axis.bin(size_t.max));
}
