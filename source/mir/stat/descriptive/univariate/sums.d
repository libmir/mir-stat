/++
Weighted sums and their accumulator.

Use `wsum(values, weights)` to sum observation-weight products without
normalization, or $(LREF WSummator) to accumulate them through `put`.
For ordinary sums and summation algorithms, see $(GMREF mir-algorithm, mir, math, sum).
For weighted means, see $(REF wmean, mir, stat, descriptive, univariate, means).

Import `mir.stat.descriptive.univariate.sums` directly, or use
$(MREF mir,stat,descriptive,univariate) for the complete univariate API.

$(BOOKTABLE,
$(TR $(TH Task) $(TH Declaration))
$(TR $(TD Weighted sum) $(TD $(LREF wsum)))
$(TR $(TD Accumulator) $(TD $(LREF WSummator)))
)

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2022 Mir Stat Authors.

Macros:
SUBREF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, $1)$(NBSP)
SUB2REF = $(REF_ALTTEXT $(TT $2), $2, mir, stat, descriptive, $1)$(NBSP)
MATHREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, math, $1)$(NBSP)
NDSLICEREF = $(GREF_ALTTEXT mir-algorithm, $(TT $2), $2, mir, ndslice, $1)$(NBSP)
+/
module mir.stat.descriptive.univariate.sums;

public import mir.math.sum: Summation;
import mir.math.sum: ResolveSummationType, Summator;

/++
Output range for $(LREF wsum).

Merging another accumulator through `put` adds its rounded weighted sum.
Low-order corrections retained by compensated or precise summation are not
transferred. Later cancellation can therefore give a different result from
inserting all observations into one accumulator, even when both accumulators
use the same summation algorithm.

When those corrections matter, pass the observation and weight batches to one
accumulator instead of merging already accumulated results.
+/
struct WSummator(T, Summation summation, U = T)
{
    import mir.ndslice.slice: isConvertibleToSlice, isSlice, kindOf;
    import std.range.primitives: isInputRange;
    import std.traits: isIterable;

    ///
    Summator!(T, summation) wsummator;

    // Match scalar put: convert each operand before multiplying.
    private static auto weightedProduct(A, B)(A a, B b)
    {
        return cast(T) a * cast(U) b;
    }

    ///
    F wsum(F = T)() const @safe @property pure nothrow @nogc
    {
        return cast(F) wsummator.sum;
    }

    ///
    void put(Slice1, Slice2)(Slice1 s, Slice2 w)
        if (isSlice!Slice1 && isSlice!Slice2)
    {
        static assert (Slice1.N == Slice2.N, "s and w must have the same number of dimensions");
        static assert (kindOf!Slice1 == kindOf!Slice2, "s and w must have the same kind");

        import mir.ndslice.slice: Contiguous;
        import mir.ndslice.topology: zip, map;
        // String lambdas enable fast math and can discard NaN * 0. Keep the
        // same numeric multiplication and conversions as scalar insertion.
        alias product = weightedProduct;

        assert(s._lengths == w._lengths, "WMeanAcumulator.put: both slices must have the same lengths");

        static if (kindOf!Slice1 != Contiguous && Slice1.N > 1) {
            assert(s.strides == w.strides, "WMeanAccumulator.put: cannot put canonical and universal slices when strides do not match");
            auto combine = s.zip!true(w);
        } else {
            auto combine = s.zip!false(w);
        }

        auto combine2 = combine.map!product;
        wsummator.put(combine2);
    }

    ///
    void put(SliceLike1, SliceLike2)(SliceLike1 s, SliceLike2 w)
        if (isConvertibleToSlice!SliceLike1 && !isSlice!SliceLike1 &&
            isConvertibleToSlice!SliceLike2 && !isSlice!SliceLike2)
    {
        import mir.ndslice.slice: toSlice;
        this.put(s.toSlice, w.toSlice);
    }

    ///
    void put(Range)(Range r)
        if (isIterable!Range)
    {
        wsummator.put(r);
    }

    ///
    void put(RangeA, RangeB)(RangeA r, RangeB w)
        if (isInputRange!RangeA && !isConvertibleToSlice!RangeA &&
            isInputRange!RangeB && !isConvertibleToSlice!RangeB)
    {
        while (!r.empty && !w.empty)
        {
            this.put(r.front, w.front);
            r.popFront;
            w.popFront;
        }
        assert(r.empty && w.empty, "WSummator.put: lengths must match");
    }

    ///
    void put()(T x, U w)
    {
        wsummator.put(x * w);
    }

    ///
    void put()(T x)
    {
        wsummator.put(x);
    }

    /// Adds the other accumulator's rounded weighted sum; see the precision limitation above.
    void put(F = T, G = U)(WSummator!(F, summation, G) wm)
    {
        wsummator.put(cast(T) wm.wsummator);
    }
}

/// Insert successive batches into one accumulator to retain summation corrections.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    WSummator!(double, Summation.kbn) accumulator;
    accumulator.put([0x1p54, 1.0], [1.0, 1.0]);
    accumulator.put([-0x1p54], [1.0]);
    assert(accumulator.wsum == 1);
}
///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    x.put([0.0, 1, 2, 3, 4].sliced, [1, 2, 3, 4, 5].sliced);
    x.wsum.should == 40;
    x.put(5, 6);
    x.wsum.should == 70;
}

// dynamic array test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    x.put([0.0, 1, 2, 3, 4], [1, 2, 3, 4, 5]);
    x.wsum.should == 40;
}

// static array test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    static immutable y = [0.0, 1, 2, 3, 4];
    static immutable w = [1, 2, 3, 4, 5];
    x.put(y, w);
    x.wsum.should == 40;
}

// 2-d slice test
version(mir_stat_test)
@safe pure
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.fuse: fuse;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = [
        [0.0, 1, 2],
        [3.0, 4, 5]
    ].fuse;
    auto w = [
        [1.0, 2, 3],
        [4.0, 5, 6]
    ].fuse;
    x.put(y, w);
    x.wsum.should == 70;
}

// universal slice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: iota, universal;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota(6).universal;
    auto w = iota([6], 1).universal;
    x.put(y, w);
    x.wsum.should == 70;
}

// canonical slice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: canonical, iota;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota(6).canonical;
    auto w = iota([6], 1).canonical;
    x.put(y, w);
    x.wsum.should == 70;
}

// 2-d universal slice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: iota, universal;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota([2, 3]).universal;
    auto w = iota([2, 3], 1).universal;
    x.put(y, w);
    x.wsum.should == 70;
}

// 2-d canonical slice test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.topology: canonical, iota;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota([2, 3]).canonical;
    auto w = iota([2, 3], 1).canonical;
    x.put(y, w);
    x.wsum.should == 70;
}

/// Assume no weights, like Summator
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    x.put([0.0, 1, 2, 3, 4].sliced);
    x.wsum.should == 10;
    x.put(5);
    x.wsum.should == 15;
}

// dynamic array test, assume no weights
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    x.put([0.0, 1, 2, 3, 4]);
    x.wsum.should == 10;
}

// static array test, assume no weights
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    static immutable y = [0.0, 1, 2, 3, 4];
    x.put(y);
    x.wsum.should == 10;
}

// Adding WSummators
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;

    double[] x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25];
    double[] y = [2.0, 7.5, 5.0, 1.0, 1.5, 0.0];

    WSummator!(float, Summation.pairwise) m0;
    m0.put(x);
    WSummator!(float, Summation.pairwise) m1;
    m1.put(y);
    m0.put(m1);
    m0.wsum.should == 29.25;
}

// repeat test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.ndslice.slice: slicedField;
    import mir.ndslice.topology: iota, repeat;
    import mir.test: should;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota(6);
    auto w = repeat(1.0, 6).slicedField;
    x.put(y, w);
    x.wsum.should == 15;
}

// range test without shape
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: Summation;
    import mir.test: should;
    import std.range: iota;

    WSummator!(double, Summation.pairwise) x;
    auto y = iota(6);
    auto w = iota(1, 7);
    x.put(y, w);
    x.wsum.should == 70;
}

// complex test
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: Summation;
    import mir.complex;
    import mir.test: should;
    alias C = Complex!double;

    WSummator!(C, Summation.pairwise, double) x;
    x.put([C(1, 3), C(2), C(3)]);
    x.wsum.should == C(6, 3);
}

/++
Computes the weighted sum of the input.

Weights are algebraic multipliers: signed weights are permitted and no
normalization or finite/nonnegative-weight validation is performed.
Zero-weight observations are still multiplied, so a floating-point NaN or
infinity multiplied by zero contributes NaN. Unlike weighted quantiles,
zero weights do not exclude observations.

Params:
    F = controls type of output
    summation = algorithm for calculating sums (default: Summation.appropriate)
    G = controls the type of weights
Returns:
    The weighted sum of all the elements in the input

See_also:
    $(MATHREF sum, Summation)
+/
template wsum(F, Summation summation = Summation.appropriate, G = F)
{
    import mir.math.common: fmamath;
    import mir.math.sum: sumType;
    import mir.ndslice.slice: isConvertibleToSlice;
    import std.traits: isIterable;

    /++
    Params:
        s = slice-like
        w = weights
    +/
    @fmamath sumType!F wsum(SliceA, SliceB)(SliceA s, SliceB w)
        if (isConvertibleToSlice!SliceA && isConvertibleToSlice!SliceB)
    {
        import core.lifetime: move;

        alias H = typeof(return);
        WSummator!(H, ResolveSummationType!(summation, SliceA, H), G) wsum;
        wsum.put(s.move, w.move);
        return wsum.wsum;
    }

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath sumType!F wsum(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias H = typeof(return);
        WSummator!(H, ResolveSummationType!(summation, Range, H), G) wsum;
        wsum.put(r.move);
        return wsum.wsum;
    }
}

/// ditto
template wsum(Summation summation = Summation.appropriate)
{
    import mir.math.common: fmamath;
    import mir.math.sum: sumType;
    import mir.ndslice.slice: isConvertibleToSlice;
    import std.traits: isIterable;

    /++
    Params:
        s = slice-like
        w = weights
    +/
    @fmamath sumType!SliceA wsum(SliceA, SliceB)(SliceA s, SliceB w)
        if (isConvertibleToSlice!SliceA && isConvertibleToSlice!SliceB)
    {
        import core.lifetime: move;
        import mir.math.sum: sumType;

        alias F = typeof(return);
        return .wsum!(F, summation, sumType!SliceB)(s.move, w.move);
    }

    /++
    Params:
        r = range, must be finite iterable
    +/
    @fmamath sumType!Range wsum(Range)(Range r)
        if (isIterable!Range)
    {
        import core.lifetime: move;

        alias F = typeof(return);
        return .wsum!(F, summation, F)(r.move);
    }
}

/// ditto
template wsum(F, string summation, G = F)
{
    mixin("alias wsum = .wsum!(F, Summation." ~ summation ~ ", G);");
}

/// ditto
template wsum(string summation)
{
    mixin("alias wsum = .wsum!(Summation." ~ summation ~ ");");
}

///
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.math.common: approxEqual;
    import mir.ndslice.slice: sliced;
    import mir.test: should;
    alias C = Complex!double;

    wsum([1, 2, 3], [1, 2, 3]).should == (1 + 4 + 9);
    wsum([C(1, 3), C(2), C(3)], [1, 2, 3]).should == C((1 + 4 + 9), 3);

    wsum!float([0, 1, 2, 3, 4, 5].sliced(3, 2), [1, 2, 3, 4, 5, 6].sliced(3, 2)).should == 70;

    static assert(is(typeof(wsum!float([1, 2, 3], [1, 2, 3])) == float));
}

/// Signed weights contribute algebraically; zero weights do not filter NaNs.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.math: isNaN;
    assert(wsum([2.0, 8.0], [-1.0, 2.0]) == 14);
    assert(wsum([2.0, 8.0], [0.0, 0.0]) == 0);
    assert(wsum([double.nan, 8.0], [0.0, 1.0]).isNaN);
}

// Optimized batch multiplication must not fold NaN * 0 or infinity * 0 to zero.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import std.meta: AliasSeq;
    import std.math: isNaN;
    static foreach (T; AliasSeq!(float, double, real))
        static foreach (mode; [Summation.naive, Summation.pairwise])
        {
            assert(wsum!(T, mode)([T.nan, T(8)], [T(0), T(1)]).isNaN);
            assert(wsum!(T, mode)([T.infinity, T(8)], [T(0), T(1)]).isNaN);
        }
}

/// If weights are not provided, then behaves like sum
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice.slice: sliced;
    import mir.test: should;
    alias C = Complex!double;

    wsum([1.0, 2, 3]).should == 6;
    wsum([C(1, 3), C(2), C(3)]).should == C(6, 3);

    wsum!float([0, 1, 2, 3, 4, 5].sliced(3, 2)).should == 15;

    static assert(is(typeof(wsum!float([1, 2, 3])) == float));
}

/// Weighted sum of vector
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: iota, map;
    import mir.test: should;

    auto x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
              2.0, 7.5, 5.0, 1.0, 1.5, 0.0].sliced;
    auto w = iota([12], 1);

    x.wsum.should == 29.25;
    x.wsum(w).should == 203;
}

/// Weighted sum of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: iota;
    import mir.test: should;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto w = iota([2, 6], 1);

    x.wsum.should == 29.25;
    x.wsum(w).should == 203;
}

/// Column sum of matrix
version(mir_stat_test)
@safe pure
unittest
{
    import mir.algorithm.iteration: all;
    import mir.math.common: approxEqual;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.topology: alongDim, byDim, iota, map, universal;

    auto x = [
        [0.0, 1.0, 1.5, 2.0, 3.5, 4.25],
        [2.0, 7.5, 5.0, 1.0, 1.5, 0.0]
    ].fuse;
    auto w = iota([2], 1).universal;
    auto result = [4, 16, 11.5, 4, 6.5, 4.25];

    // Use byDim or alongDim with map to compute sum of row/column.
    assert(x.byDim!1.map!(a => a.wsum(w)).all!approxEqual(result));
    assert(x.alongDim!0.map!(a => a.wsum(w)).all!approxEqual(result));

    // FIXME
    // Without using map, computes the sum of the whole slice
    // assert(x.byDim!1.wsum(w) == x.sliced.wsum);
    // assert(x.alongDim!0.wsum(w) == x.sliced.wsum);
}

/// Can also set algorithm or output type
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: repeat, universal;
    import mir.test: should;

    //Set sum algorithm (also for weights) or output type

    auto a = [1, 1e100, 1, -1e100].sliced;

    auto x = a * 10_000;
    auto w1 = [1, 1, 1, 1].sliced;
    auto w2 = [0.25, 0.25, 0.25, 0.25].sliced;

    x.wsum!"kbn"(w1).should == 20_000;
    x.wsum!"kbn"(w2).should == 20_000 / 4;
    x.wsum!"kb2"(w1).should == 20_000;
    x.wsum!"precise"(w1).should == 20_000;
    x.wsum!(double, "precise")(w1).should == 20_000;

    auto y = uint.max.repeat(3);
    y.wsum!ulong([1, 1, 1].sliced.universal).should == 12884901885;
}

/++
wsum works for complex numbers and other user-defined types
+/
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.complex;
    import mir.ndslice.slice: sliced;
    import mir.test: should;
    alias C = Complex!double;

    auto x = [C(1.0, 2), C(2, 3), C(3, 4), C(4, 5)].sliced;
    auto w = [1, 2, 3, 4].sliced;
    x.wsum(w).should == C(30, 40);
}

/// Compute weighted sum tensors along specified dimention of tensors
version(mir_stat_test)
@safe pure
unittest
{
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: alongDim, as, iota, map, universal;
    /++
      [[0,1,2],
       [3,4,5]]
     +/
    auto x = [
        [0, 1, 2],
        [3, 4, 5]
    ].fuse.as!double;
    auto w = [
        [1, 2, 3],
        [4, 5, 6]
    ].fuse;
    auto w1 = [1, 2].sliced.universal;
    auto w2 = [1, 2, 3].sliced;

    assert(x.wsum(w) == 70);

    auto m0 = [(0 + 6), (1 + 8), (2 + 10)];
    assert(x.alongDim!0.map!(a => a.wsum(w1)) == m0);
    assert(x.alongDim!(-2).map!(a => a.wsum(w1)) == m0);

    auto m1 = [(0 + 2 + 6), (3 + 8 + 15)];
    assert(x.alongDim!1.map!(a => a.wsum(w2)) == m1);
    assert(x.alongDim!(-1).map!(a => a.wsum(w2)) == m1);
}

// test chaining
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;
    [1.0, 2, 3, 4].wsum.should == 10;
}

// additional alongDim tests
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.sum: sumType;
    import mir.ndslice.topology: iota, alongDim, map;

    auto x = iota([2, 2], 1);
    auto w = iota([2], 2);
    auto y = x.alongDim!1.map!(a => a.wsum(w));
    static assert(is(sumType!(typeof(y)) == long));
}

// @nogc test
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.test: should;

    static immutable x = [0.0, 1.0, 1.5, 2.0, 3.5, 4.25,
                          2.0, 7.5, 5.0, 1.0, 1.5, 0.0];
    static immutable w = [1.0, 2, 3,  4,  5,  6,
                            7, 8, 9, 10, 11, 12];

    x.wsum.should == 29.25;
    x.wsum(w).should == 203.0;
}

// Batch products use accumulator operand types, just as scalar insertion does.
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.stat.descriptive.univariate.means: wmean, WMeanAccumulator, AssumeWeights;
    static foreach (method; [Summation.naive, Summation.pairwise])
    {{
        int[] values = [50_000, 50_000];
        int[] weights = [50_000, 50_000];
        assert(wsum!(double, method)(values, weights) == 5_000_000_000.0);
        assert((wmean!(double, method, AssumeWeights.primary, double, method)(values, weights)) == 50_000);

        static foreach (assumption; [AssumeWeights.primary, AssumeWeights.sumToOne])
        {{
            WMeanAccumulator!(double, method, assumption) batch, scalar;
            batch.put(values, weights);
            foreach (i; 0 .. values.length)
                scalar.put(values[i], weights[i]);
            assert(batch.wsum == scalar.wsum);
            assert(batch.wmean == scalar.wmean);
        }}

        // The inputs are finite floats, but their product exceeds float.max.
        float[] large = [1e20f];
        WSummator!(double, method) batchSum, scalarSum;
        batchSum.put(large, large);
        scalarSum.put(large[0], large[0]);
        assert(batchSum.wsum == scalarSum.wsum);
        assert(batchSum.wsum > 0.999e40 && batchSum.wsum < 1.001e40);

        static foreach (assumption; [AssumeWeights.primary, AssumeWeights.sumToOne])
        {{
            WMeanAccumulator!(double, method, assumption) batch, scalar;
            batch.put(large, large);
            scalar.put(large[0], large[0]);
            assert(batch.wsum == scalar.wsum);
            assert(batch.wmean == scalar.wmean);
        }}
    }}
}
