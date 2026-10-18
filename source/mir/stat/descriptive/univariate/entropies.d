/++
Entropy sum and its accumulator.

These APIs compute `sum(x * ln(x))` using the natural logarithm, with a zero
contribution for `x == 0`. They do not normalize inputs or negate the sum.
For probabilities that sum to one, the result is the negative of Shannon
entropy in nats.

Use $(REF entropy, mir, stat, descriptive, univariate, entropies) for a range or array, or $(REF EntropyAccumulator, mir, stat, descriptive, univariate, entropies)
to accumulate contributions through `put`.

Import `mir.stat.descriptive.univariate.entropies` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(SCRIPT inhibitQuickIndex = 1;)

$(DIVC quickindex,
$(BOOKTABLE,
$(TR $(TH Category) $(TH Declarations))
    $(TR $(TD Function) $(TD
        $(REF entropy, mir, stat, descriptive, univariate, entropies)$(NBSP)
    ))
    $(TR $(TD Accumulator) $(TD
        $(REF EntropyAccumulator, mir, stat, descriptive, univariate, entropies)$(NBSP)
    ))
))

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

module mir.stat.descriptive.univariate.entropies;

public import mir.math.sum: Summation;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: fmamath;
import mir.math.sum: Summator, ResolveSummationType;
import mir.ndslice.slice: isConvertibleToSlice, isSlice, Slice, SliceKind;
import std.traits: isIterable, isMutable;

import mir.stat.descriptive.univariate.internal.types: statType, meanType;

/++
Accumulates `sum(x * ln(x))` without normalization or negation.

Inputs must be nonnegative; zero contributes zero. See $(LREF entropy) for
the sign convention and interpretation for probability inputs.
+/
struct EntropyAccumulator(T, Summation summation)
{
    import mir.math.internal.xlogy: xlog;
    import mir.primitives: hasShape;
    import std.traits: isIterable;

    ///
    Summator!(T, summation) summator;
    /// Returns the accumulated sum of `x * ln(x)`.
    F entropy(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) summator.sum;
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        static if (hasShape!Range)
        {
            import mir.ndslice.topology: as, map;

            summator.put(r.as!T.map!xlog);
        }
        else
        {
            foreach(x; r)
            {
                summator.put(xlog(cast(T)x));
            }
        }
    }

    ///
    void put()(T x)
    {
        summator.put(xlog(x));
    }

    ///
    void put(U)(EntropyAccumulator!(U, summation) e)
    {
        summator.put(e.summator.sum);
    }
}

/// test basic functionality
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    EntropyAccumulator!(double, Summation.pairwise) x;
    x.put([0.1, 0.2, 0.3].sliced);
    assert(x.entropy.approxEqual(-0.913338));
    x.put(0.4);
    assert(x.entropy.approxEqual(-1.279854));
}

// test floats
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    EntropyAccumulator!(float, Summation.pairwise) x;
    x.put([0.1, 0.2, 0.3].sliced);
    assert(x.entropy.approxEqual(-0.913338));
    x.put(0.4);
    assert(x.entropy.approxEqual(-1.279854));
}

// test put EntropyAccumulator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto a = [1.0, 2, 3,  4,  5,  6].sliced;
    auto b = [7.0, 8, 9, 10, 11, 12].sliced;

    auto x = a / 78.0;
    auto y = b / 78.0;

    EntropyAccumulator!(double, Summation.pairwise) m0;
    m0.put(x);
    assert(m0.entropy.approxEqual(-0.800844));
    EntropyAccumulator!(double, Summation.pairwise) m1;
    m1.put(y);
    assert(m1.entropy.approxEqual(-1.526653));
    m0.put(m1);
    assert(m0.entropy.approxEqual(-2.327497));
}

/++
If `T` is a floating point type, this is an alias to the unqualified type.
If `T` is not a floating point type, this will alias a `double` type if `T`
is summable and implicitly convertible to a floating point type.
+/
package(mir)
template entropyType(T)
{
    import mir.math.sum: sumType;

    alias U = sumType!T;
    alias entropyType = statType!(U, false);
}

/++
Computes `sum(x * ln(x))`, using the natural logarithm.

Inputs must be nonnegative. Each zero contributes zero, following the
limit of `x * ln(x)` as `x` approaches zero from above. Negative inputs
violate an assertion when assertions are enabled.

The input is not normalized, and the sum is not negated. For probabilities
that sum to one, negate the result to obtain Shannon entropy in nats.
Values greater than one are also accepted, so arbitrary nonnegative inputs
can produce a positive result. For example, `[0.5, 0.5]` produces `-ln(2)`,
while `[1, 1]` produces zero rather than normalizing the two values.
By default, if `F` is not a floating point type, then the result will have a
`double` type if `F` is implicitly convertible to a floating point type.
Params:
    F = controls type of output
    summation = algorithm for summing the individual entropy values (default: Summation.appropriate)
Returns:
    The sum of `x * ln(x)` over the input, with a floating point result
See_also:
    $(MATHREF sum, Summation)
+/
template entropy(F, Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath entropyType!Range entropy(Range)(Range r)
        if (isIterable!Range)
    {
        alias G = typeof(return);
        EntropyAccumulator!(G, ResolveSummationType!(summation, Range, G)) entropyAccumulator;
        entropyAccumulator.put(r.move);
        return entropyAccumulator.entropy;
    }

    /++
    Params:
        ar = values
    +/
    @fmamath entropyType!F entropy(scope const F[] ar...)
    {
        alias G = typeof(return);
        EntropyAccumulator!(G, ResolveSummationType!(summation, const(G)[], G)) entropyAccumulator;
        entropyAccumulator.put(ar);
        return entropyAccumulator.entropy;
    }
}

///
template entropy(Summation summation = Summation.appropriate)
{
    import core.lifetime: move;
    import std.traits: isIterable;

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath entropyType!Range entropy(Range)(Range r)
        if (isIterable!Range)
    {
        alias F = typeof(return);
        return .entropy!(F, summation)(r.move);
    }

    /++
    Params:
        ar = values
    +/
    @fmamath entropyType!T entropy(T)(scope const T[] ar...)
    {
        alias F = typeof(return);
        return .entropy!(F, summation)(ar);
    }
}

/// ditto
template entropy(F, string summation)
{
    mixin("alias entropy = .entropy!(F, Summation." ~ summation ~ ");");
}

/// ditto
template entropy(string summation)
{
    mixin("alias entropy = .entropy!(Summation." ~ summation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    assert(entropy([0.166667, 0.333333, 0.50]).approxEqual(-1.011404));

    assert(entropy!float([0.05, 0.1, 0.15, 0.2, 0.25, 0.25].sliced(3, 2)).approxEqual(-1.679648));

    static assert(is(typeof(entropy!float([0.166667, 0.333333, 0.50])) == float));
}

/// Entropy of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    double[] a = [1.0, 2, 3,  4,  5,  6, 7, 8, 9, 10, 11, 12];
    a[] /= 78.0;

    auto x = a.sliced;
    assert(x.entropy.approxEqual(-2.327497));
}

/// Entropy of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;

    double[] a = [1.0, 2, 3,  4,  5,  6, 7, 8, 9, 10, 11, 12];
    a[] /= 78.0;

    auto x = a.fuse;
    assert(x.entropy.approxEqual(-2.327497));
}

/// Column entropy of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, map;

    double[][] a = [
        [1.0, 2, 3,  4,  5,  6],
        [7.0, 8, 9, 10, 11, 12]
    ];
    a[0][] /= 78.0;
    a[1][] /= 78.0;

    auto x = a.fuse;
    auto result = [-0.272209, -0.327503, -0.374483, -0.415678, -0.452350, -0.485273];

    // Use byDim or alongDim with map to compute entropy of row/column.
    assert(x.byDim!1.map!entropy.all!approxEqual(result));
    assert(x.alongDim!0.map!entropy.all!approxEqual(result));

    // FIXME
    // Without using map, computes the entropy of the whole slice
    // assert(x.byDim!1.entropy == x.sliced.entropy);
    // assert(x.alongDim!0.entropy == x.sliced.entropy);
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat;

    auto a = [1, 1e100, 1, 1e100].sliced;

    auto x = a * 10_000;

    assert(x.entropy!"kbn".approxEqual(4.789377e106));
    assert(x.entropy!"kb2".approxEqual(4.789377e106));
    assert(x.entropy!"precise".approxEqual(4.789377e106));
    assert(x.entropy!(double, "precise").approxEqual(4.789377e106));
}

/++
For integral slices, pass output type as template parameter to ensure output
type is correct.
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    auto x = [3, 1, 1, 2, 4, 4,
              2, 7, 5, 1, 2, 3].sliced;

    auto y = x.entropy;
    assert(y.approxEqual(43.509472));
    static assert(is(typeof(y) == double));

    assert(x.entropy!float.approxEqual(43.509472f));
}

/// Arbitrary entropy
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;

    assert(entropy(0.25, 0.25, 0.25, 0.25).approxEqual(-1.386294));
    assert(entropy!float(0.25, 0.25, 0.25, 0.25).approxEqual(-1.386294));
}

// Dynamic array / UFCS
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: approxEqual;

    assert(entropy([0.25, 0.25, 0.25, 0.25]).approxEqual(-1.386294));
    assert([0.25, 0.25, 0.25, 0.25].entropy.approxEqual(-1.386294));
}

// Check type of alongDim result
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto y = x.alongDim!1.map!entropy;
    assert(y.all!approxEqual([1.386294, 8.841014]));
    static assert(is(entropyType!(typeof(y)) == double));
}

// @nogc test
version(mir_stat_test)
@safe pure @nogc nothrow
unittest
{
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;

    static immutable x = [1.0 / 78,  2.0 / 78,  3.0 / 78,  4.0 / 78,
                          5.0 / 78,  6.0 / 78,  7.0 / 78,  8.0 / 78,
                          9.0 / 78, 10.0 / 78, 11.0 / 78, 12.0 / 78];

    assert(x.sliced.entropy.approxEqual(-2.327497));
    assert(x.sliced.entropy!float.approxEqual(-2.327497));
}
