/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution).

Outcomes are zero-based indices into $(D p). Each entry is a finite
probability in $(D [0, 1]), and the entries must sum approximately to one,
as checked by $(D approxEqual). The functions do not normalize the supplied
probabilities. Empty probability sequences and out-of-range observations
violate the contracts.

CDF and CCDF use the order of the category indices. This ordering is chosen
by the caller; it need not have an intrinsic meaning for category labels.
The overloads accept a one-dimensional slice or an array/variadic list of
built-in floating-point probabilities.

Finite expressions are subject to floating-point rounding. Contract violations
are not prescribed return values: disabling contracts does not make an invalid
input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2023 Mir Stat Authors.

+/

module mir.stat.distribution.categorical;

import mir.algorithm.iteration: all;
import mir.internal.utility: isFloatingPoint;
import mir.math.common: approxEqual;
import mir.math.sum: elementType, sumType, sum;
import mir.ndslice.slice: Slice, SliceKind;
import std.traits: CommonType;

/++
Computes the Categorical probability mass function (PMF).

Returns the point probability $(D P(X == x) = p[x]).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p[x] == 0)) $(TD $(D 0)))
    $(TR $(TD $(D p[x] == 1)) $(TD $(D 1)))
    $(TR $(TD $(D x >= p.length)) $(TD Contract violation))
)

Params:
    x = zero-based category index at which to evaluate PMF
    p = category probabilities, each in $(D [0, 1]), summing approximately to one

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution)
+/
@safe pure nothrow @nogc
elementType!(Slice!(Iterator, 1, kind)) categoricalPMF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (isFloatingPoint!(elementType!(Slice!(Iterator, 1, kind))))
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[x];
}

/// ditto
@safe pure nothrow @nogc
T categoricalPMF(T)(const size_t x, scope const T[] p...)
    if (isFloatingPoint!T)
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[x];
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [0.1, 0.5, 0.4];
    auto p = x.sliced;

    0.categoricalPMF(p).shouldApprox == 0.1;
    1.categoricalPMF(p).shouldApprox == 0.5;
    2.categoricalPMF(p).shouldApprox == 0.4;
}

/// Can also use dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: shouldApprox;

    double[] p = [0.1, 0.5, 0.4];

    0.categoricalPMF(p).shouldApprox == 0.1;
    1.categoricalPMF(p).shouldApprox == 0.5;
    2.categoricalPMF(p).shouldApprox == 0.4;
}

/// Normalize nonnegative weights before passing them as category probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.sum: sum;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [1.0, 5, 4];
    auto p = x.sliced;
    auto q = p / sum(p);

    0.categoricalPMF(q).shouldApprox == 0.1;
    1.categoricalPMF(q).shouldApprox == 0.5;
    2.categoricalPMF(q).shouldApprox == 0.4;
}

/++
Computes the Categorical cumulative distribution function (CDF).

Returns $(D P(X <= x)) by summing entries from index zero through $(D x).
At the final category the result is the supplied total, which may differ
slightly from one; it is not replaced with an exact one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == 0)) $(TD $(D p[0])))
    $(TR $(TD $(D x == p.length - 1)) $(TD Sum of all entries in $(D p)))
    $(TR $(TD $(D x >= p.length)) $(TD Contract violation))
)

Params:
    x = zero-based category index at which to evaluate CDF
    p = category probabilities, each in $(D [0, 1]), summing approximately to one

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution)
+/
@safe pure nothrow @nogc
sumType!(Slice!(Iterator, 1, kind)) categoricalCDF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (isFloatingPoint!(elementType!(Slice!(Iterator, 1, kind))))
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[0 .. (x + 1)].sum;
}

/// ditto
@safe pure nothrow @nogc
T categoricalCDF(T)(const size_t x, scope const T[] p...)
    if (isFloatingPoint!T)
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[0 .. (x + 1)].sum;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [0.1, 0.5, 0.4];
    auto p = x.sliced;

    0.categoricalCDF(p).shouldApprox == 0.1;
    1.categoricalCDF(p).shouldApprox == 0.6;
    2.categoricalCDF(p).shouldApprox == 1.0;
}

/// Can also use dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: shouldApprox;

    double[] p = [0.1, 0.5, 0.4];

    0.categoricalCDF(p).shouldApprox == 0.1;
    1.categoricalCDF(p).shouldApprox == 0.6;
    2.categoricalCDF(p).shouldApprox == 1.0;
}

/++
Computes the Categorical complementary cumulative distribution function (CCDF).

Returns the probability of a category strictly greater than `x`.

Sums entries after $(D x) directly, avoiding subtraction of a rounded CDF
from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == p.length - 1)) $(TD $(D 0)))
    $(TR $(TD $(D p.length == 1) and $(D x == 0)) $(TD $(D 0)))
    $(TR $(TD $(D x >= p.length)) $(TD Contract violation))
)

Params:
    x = zero-based category index at which to evaluate CCDF
    p = category probabilities, each in $(D [0, 1]), summing approximately to one

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution)
+/
@safe pure nothrow @nogc
sumType!(Slice!(Iterator, 1, kind)) categoricalCCDF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (isFloatingPoint!(elementType!(Slice!(Iterator, 1, kind))))
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[x + 1 .. $].sum;
}

/// ditto
@safe pure nothrow @nogc
T categoricalCCDF(T)(const size_t x, scope const T[] p...)
    if (isFloatingPoint!T)
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    return p[x + 1 .. $].sum;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [0.1, 0.5, 0.4];
    auto p = x.sliced;

    0.categoricalCCDF(p).shouldApprox == 0.9;
    1.categoricalCCDF(p).shouldApprox == 0.4;
    2.categoricalCCDF(p).shouldApprox == 0.0;
}

/// Can also use dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: shouldApprox;

    double[] p = [0.1, 0.5, 0.4];

    0.categoricalCCDF(p).shouldApprox == 0.9;
    1.categoricalCCDF(p).shouldApprox == 0.4;
    2.categoricalCCDF(p).shouldApprox == 0.0;
}

// CDF and CCDF partition the probability, including at the final category.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.common: approxEqual;
    double[4] p = [0.125, 0, 0.375, 0.5];
    foreach (x; 0 .. p.length)
    {
        assert(approxEqual(categoricalCDF(x, p[]) + categoricalCCDF(x, p[]), 1.0));
        assert(approxEqual(categoricalCDF(x, p[].sliced) + categoricalCCDF(x, p[].sliced), 1.0));
    }
    assert(categoricalCCDF(3, p[]) == 0);
    double[1] singleton = [1.0];
    assert(categoricalCCDF(0, singleton[]) == 0);
}

/++
Computes the Categorical inverse cumulative distribution function (InvCDF).

Returns the first category whose accumulated probability is at least $(D q).
Equality selects that category. Accumulation is subject to floating-point
rounding. If the scan never reaches $(D q) because the total is slightly
below it, the function returns the last category with positive probability.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D q == 0)) $(TD $(D 0), even when $(D p[0] == 0)))
    $(TR $(TD $(D q == 1)) $(TD First category whose accumulated probability reaches one; otherwise the last positive-probability category))
    $(TR $(TD $(D q < 0), $(D q > 1), or NaN) $(TD Contract violation))
)

Params:
    q = cumulative probability in $(D [0, 1])
    p = category probabilities, each in $(D [0, 1]), summing approximately to one

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution)
+/
@safe pure nothrow @nogc
size_t categoricalInvCDF(T, Iterator, SliceKind kind)(const T q, scope const Slice!(Iterator, 1, kind) p)
    if (isFloatingPoint!(CommonType!(T, elementType!(Slice!(Iterator, 1, kind)))))
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    CommonType!(T, elementType!(typeof(p))) s = 0.0;
    size_t lastPositive;
    foreach (i; 0 .. p.length) {
        s += p[i];
        if (q <= s)
            return i;
        if (p[i] > 0)
            lastPositive = i;
    }
    // Rounding can leave the cumulative probability just below one.
    return lastPositive;
}

/// ditto
@safe pure nothrow @nogc
size_t categoricalInvCDF(T)(const T q, scope const T[] p...)
    if (isFloatingPoint!T)
    in (q >= 0, "q must be greater than or equal to 0")
    in (q <= 1, "q must be less than or equal to 1")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    T s = 0.0;
    size_t lastPositive;
    foreach (i; 0 .. p.length) {
        s += p[i];
        if (q <= s)
            return i;
        if (p[i] > 0)
            lastPositive = i;
    }
    // Rounding can leave the cumulative probability just below one.
    return lastPositive;
}

// A rounded cumulative sum must not advance past the last category or select
// trailing categories with zero probability.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    double[12] p = 0;
    p[0 .. 10] = 0.1;
    assert(categoricalInvCDF(1.0, p[0 .. 10]) == 9);
    assert(categoricalInvCDF(1.0, p[0 .. 10].sliced) == 9);
    assert(categoricalInvCDF(1.0, p[]) == 9);
    assert(categoricalInvCDF(1.0, p[].sliced) == 9);
    assert(categoricalInvCDF(1.0 - double.epsilon / 2, p[]) == 9);
    assert(categoricalInvCDF(0.0, p[]) == 0);
    assert(categoricalInvCDF(0.25, p[]) == 2);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.test: should;

    static immutable x = [0.1, 0.5, 0.4];
    auto p = x.sliced;

    categoricalInvCDF(0.0, p).should == 0;
    categoricalInvCDF(0.1, p).should == 0;
    categoricalInvCDF(0.2, p).should == 1;
    categoricalInvCDF(0.3, p).should == 1;
    categoricalInvCDF(0.4, p).should == 1;
    categoricalInvCDF(0.5, p).should == 1;
    categoricalInvCDF(0.6, p).should == 1;
    categoricalInvCDF(0.7, p).should == 2;
    categoricalInvCDF(0.8, p).should == 2;
    categoricalInvCDF(0.9, p).should == 2;
    categoricalInvCDF(1.0, p).should == 2;
}

/// Can also use dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.test: should;

    double[] p = [0.1, 0.5, 0.4];

    categoricalInvCDF(0.5, p).should == 1;
}

/// Zero-probability categories preserve indices but do not add cumulative mass.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    double[4] p = [0.0, 0.25, 0.75, 0.0];
    assert(categoricalPMF(0, p[]) == 0);
    assert(categoricalCDF(1, p[]) == 0.25);
    assert(categoricalCCDF(1, p[]) == 0.75);
    assert(categoricalInvCDF(0.0, p[]) == 0);
    assert(categoricalInvCDF(0.25, p[]) == 1);
    assert(categoricalInvCDF(1.0, p[]) == 2);
}

/++
Computes the Categorical log probability mass function (LPMF).

Returns the natural logarithm of $(D p[x]).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p[x] == 0)) $(TD $(D -T.infinity) for the probability element type $(D T)))
    $(TR $(TD $(D p[x] == 1)) $(TD $(D 0)))
    $(TR $(TD $(D x >= p.length)) $(TD Contract violation))
)

Params:
    x = zero-based category index at which to evaluate LPMF
    p = category probabilities, each in $(D [0, 1]), summing approximately to one

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution)
+/
@safe pure nothrow @nogc
elementType!(Slice!(Iterator, 1, kind)) categoricalLPMF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (isFloatingPoint!(elementType!(Slice!(Iterator, 1, kind))))
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    import mir.math.common: log;
    return x.categoricalPMF(p).log;
}

/// ditto
@safe pure nothrow @nogc
T categoricalLPMF(T)(const size_t x, scope const T[] p...)
    if (isFloatingPoint!T)
    in (x < p.length, "x must be less than the length of p")
    in (p.sum.approxEqual(1.0), "p must sum to 1")
    in (p.all!("a >= 0"), "p must be greater than or equal to 0")
    in (p.all!("a <= 1"), "p must be less than or equal to 1")
{
    import mir.math.common: log;
    return x.categoricalPMF(p).log;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: log;
    import mir.ndslice.slice: sliced;
    import mir.test: shouldApprox;

    static immutable x = [0.1, 0.5, 0.4];
    auto p = x.sliced;

    0.categoricalLPMF(p).shouldApprox == log(0.1);
    1.categoricalLPMF(p).shouldApprox == log(0.5);
    2.categoricalLPMF(p).shouldApprox == log(0.4);
}

/// Can also use dynamic array
version(mir_stat_test)
@safe pure nothrow
unittest
{
    import mir.math.common: log;
    import mir.test: shouldApprox;

    double[] p = [0.1, 0.5, 0.4];

    0.categoricalLPMF(p).shouldApprox == log(0.1);
    1.categoricalLPMF(p).shouldApprox == log(0.5);
    2.categoricalLPMF(p).shouldApprox == log(0.4);
}
