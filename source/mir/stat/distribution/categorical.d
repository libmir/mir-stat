/++
This module contains algorithms for the $(LINK2 https://en.wikipedia.org/wiki/Categorical_distribution, Categorical Distribution).

Outcomes are zero-based indices into $(D p). Each entry is a finite
probability in $(D [0, 1]), and the total must be within $(D 2^^-20)
(approximately $(D 9.54e-7)) of one, inclusive. This tolerance is the same
for built-in and $(D Fp) types and does not depend on the category count.
Built-ins apply $(D approxEqual) to their computed sum; $(D Fp) validates
the exact sum of the represented inputs. Acceptance can therefore differ
at the tolerance boundary. The functions do not normalize the supplied
probabilities. Empty probability sequences and out-of-range observations
violate the contracts.

CDF and CCDF use the order of the category indices. This ordering is chosen
by the caller; it need not have an intrinsic meaning for category labels.
The overloads accept a one-dimensional slice or an array/variadic list of
built-in floating-point or $(D Fp) probabilities. Fp results have the input
probability width; an Fp inverse query must have the same width as the entries.

The tolerance permits small discrepancies in supplied totals; it is not an
accuracy tolerance for the calculations. Fp CDF and CCDF round their exact
prefix or suffix sum once to nearest, ties to even. The inverse compares
exact prefix sums with the represented query. LPMF uses guarded logarithms
without a universal correct-rounding guarantee. PMF returns the supplied entry.

An accepted total slightly different from one is preserved: the final CDF
can be below or above one. If no prefix reaches an inverse query, the result
is the last positive-probability category. With an excessive total, a prefix
can reach one before the final positive category. Callers wanting normalized
probabilities must normalize explicitly; neither type family does this silently.

Fp summation uses fixed-capacity stack storage without reordering the input.
For N entries and E distinct represented exponents, summation takes O(N*E)
work and inverse search takes O(N*E*log(N+1)) work. Validation also sums all
entries when contracts are enabled. Storage depends on the Fp width, not N.

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
import std.traits: CommonType, Unqual;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid;
import mir.stat.internal.fp_categorical: categoricalSum, categoricalInverse, categoricalValid;

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

/// ditto
@safe pure nothrow @nogc
Unqual!(elementType!(Slice!(Iterator, 1, kind))) categoricalPMF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (is(Unqual!(elementType!(Slice!(Iterator, 1, kind))) == Fp!bits, uint bits))
    in (x < p.length, "x must be less than the length of p")
    in (categoricalValid!(typeof(elementType!(Slice!(Iterator, 1, kind)).init.coefficient).sizeof * 8)(p), "entries must be probabilities with total within 2^-20 of one")
{
    return p[x];
}

/// ditto
@safe pure nothrow @nogc
T categoricalPMF(T)(const size_t x, scope const T[] p...)
    if (is(T == Fp!bits, uint bits))
{
    import mir.ndslice.slice: sliced;
    return categoricalPMF(x, p.sliced);
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

/// ditto
@safe pure nothrow @nogc
Unqual!(elementType!(Slice!(Iterator, 1, kind))) categoricalCDF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (is(Unqual!(elementType!(Slice!(Iterator, 1, kind))) == Fp!bits, uint bits))
    in (x < p.length, "x must be less than the length of p")
    in (categoricalValid!(typeof(elementType!(Slice!(Iterator, 1, kind)).init.coefficient).sizeof * 8)(p), "entries must be probabilities with total within 2^-20 of one")
{
    enum bits = typeof(elementType!(typeof(p)).init.coefficient).sizeof * 8;
    return categoricalSum!bits(p, 0, x + 1).value;
}

/// ditto
@safe pure nothrow @nogc
T categoricalCDF(T)(const size_t x, scope const T[] p...)
    if (is(T == Fp!bits, uint bits))
{
    import mir.ndslice.slice: sliced;
    return categoricalCDF(x, p.sliced);
}

/// A slightly deficient total is accepted but is not silently normalized.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fp: Fp;
    alias T = Fp!128;
    T[3] p = [T(0.5), T(0.5 - 0x1p-21), T(0)];
    assert(categoricalCDF(2, p[]) == T(1 - 0x1p-21));
    assert(categoricalPMF(0, p[]) == T(0.5));
    assert(categoricalInvCDF(T(1), p[]) == 1);
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

/// ditto
@safe pure nothrow @nogc
Unqual!(elementType!(Slice!(Iterator, 1, kind))) categoricalCCDF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (is(Unqual!(elementType!(Slice!(Iterator, 1, kind))) == Fp!bits, uint bits))
    in (x < p.length, "x must be less than the length of p")
    in (categoricalValid!(typeof(elementType!(Slice!(Iterator, 1, kind)).init.coefficient).sizeof * 8)(p), "entries must be probabilities with total within 2^-20 of one")
{
    enum bits = typeof(elementType!(typeof(p)).init.coefficient).sizeof * 8;
    return categoricalSum!bits(p, x + 1, p.length).value;
}

/// ditto
@safe pure nothrow @nogc
T categoricalCCDF(T)(const size_t x, scope const T[] p...)
    if (is(T == Fp!bits, uint bits))
{
    import mir.ndslice.slice: sliced;
    return categoricalCCDF(x, p.sliced);
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
Equality selects that category. Built-in accumulation is subject to floating-point
rounding; Fp compares exact prefix sums, including when a rounded CDF equals q
but the exact sum has not reached it. If the scan never reaches $(D q) because the total is slightly
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

/// ditto
@safe pure nothrow @nogc
size_t categoricalInvCDF(T, Iterator, SliceKind kind)(const T q, scope const Slice!(Iterator, 1, kind) p)
    if (is(T == Fp!bits, uint bits) && is(Unqual!(elementType!(Slice!(Iterator, 1, kind))) == T))
    in (fpProbabilityValid(q), "q must be finite and between zero and one")
    in (categoricalValid!(typeof(elementType!(Slice!(Iterator, 1, kind)).init.coefficient).sizeof * 8)(p), "entries must be probabilities with total within 2^-20 of one")
{
    return categoricalInverse(q, p);
}

/// ditto
@safe pure nothrow @nogc
size_t categoricalInvCDF(T)(const T q, scope const T[] p...)
    if (is(T == Fp!bits, uint bits))
{
    import mir.ndslice.slice: sliced;
    return categoricalInvCDF(q, p.sliced);
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

/// ditto
@safe pure nothrow @nogc
Unqual!(elementType!(Slice!(Iterator, 1, kind))) categoricalLPMF(Iterator, SliceKind kind)(const size_t x, scope const Slice!(Iterator, 1, kind) p)
    if (is(Unqual!(elementType!(Slice!(Iterator, 1, kind))) == Fp!bits, uint bits))
    in (x < p.length, "x must be less than the length of p")
    in (categoricalValid!(typeof(elementType!(Slice!(Iterator, 1, kind)).init.coefficient).sizeof * 8)(p), "entries must be probabilities with total within 2^-20 of one")
{
    enum bits = typeof(elementType!(typeof(p)).init.coefficient).sizeof * 8;
    alias T = Unqual!(elementType!(typeof(p)));
    if (!p[x].coefficient) return T(true, long.max, typeof(T.init.coefficient).init);
    import mir.stat.internal.fp_probability: fpProbabilityLess;
    if (!fpProbabilityLess(p[x], T(1))) return T(0);
    import mir.stat.internal.fp_probability_log: ProbabilityLog;
    auto result = ProbabilityLog!bits.probability(p[x]).finish!bits;
    result.sign = !!result.coefficient;
    return result;
}

/// ditto
@safe pure nothrow @nogc
T categoricalLPMF(T)(const size_t x, scope const T[] p...)
    if (is(T == Fp!bits, uint bits))
{
    import mir.ndslice.slice: sliced;
    return categoricalLPMF(x, p.sliced);
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

// Shared exact inputs use the same outcomes for built-in and Fp probabilities.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    import mir.math.common: fabs, log;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128)) {{
        T[4] p = [T(0.125), T(0), T(0.375), T(0.5)];
        const double[4] cdf = [0.125, 0.125, 0.5, 1];
        const double[4] ccdf = [0.875, 0.875, 0.5, 0];
        foreach (i; 0 .. p.length) {
            assert(categoricalPMF(i, p[]) == p[i]);
            assert(categoricalCDF(i, p[]) == T(cdf[i]));
            assert(categoricalCCDF(i, p[]) == T(ccdf[i]));
            assert(categoricalPMF(i, p[].sliced) == p[i]);
            assert(categoricalCDF(i, p[].sliced) == T(cdf[i]));
            assert(categoricalCCDF(i, p[].sliced) == T(ccdf[i]));
        }
        assert(categoricalInvCDF(T(0), p[]) == 0);
        assert(categoricalInvCDF(T(0.125), p[]) == 0);
        assert(categoricalInvCDF(T(0.25), p[]) == 2);
        assert(categoricalInvCDF(T(0.5), p[]) == 2);
        assert(categoricalInvCDF(T(1), p[]) == 3);
        assert(categoricalInvCDF(T(0.5), p[].sliced) == 2);
        assert(categoricalInvCDF!T(T(0.5), p[]) == 2);
        assert(categoricalInvCDF!T(T(0.5), p[].sliced) == 2);
        assert(categoricalInvCDF!T(T(0.25), T(0.25), T(0.75)) == 0);
        assert(categoricalLPMF(1, p[]) == T(-double.infinity));
        enum tolerance = is(T == float) ? 4e-7 : 4e-15;
        assert(fabs(cast(double)categoricalLPMF(2, p[]) - log(0.375)) < tolerance);
        assert(fabs(cast(double)categoricalLPMF(2, p[].sliced) - log(0.375)) < tolerance);
        assert(categoricalPMF(1, T(0.25), T(0.75)) == T(0.75));
        assert(categoricalCDF(0, T(0.25), T(0.75)) == T(0.25));
        assert(categoricalCCDF(0, T(0.25), T(0.75)) == T(0.75));
        assert(categoricalInvCDF(T(0.25), T(0.25), T(0.75)) == 0);
        assert(categoricalLPMF(0, T(1)) == T(0));
        T[3] zeros = [T(0), T(1), T(0)];
        assert(categoricalInvCDF(T(0), zeros[]) == 0);
        assert(categoricalInvCDF(T(1), zeros[]) == 1);
        // Same tolerance, no normalization, and the same deficient-total fallback.
        T[3] deficient = [T(0.5), T(0.5 - 0x1p-21), T(0)];
        assert(categoricalCDF(2, deficient[]) == T(1 - 0x1p-21));
        assert(categoricalInvCDF(T(1), deficient[]) == 1);
        T[3] excessive = [T(0.5), T(0.5), T(0x1p-21)];
        assert(categoricalCDF(2, excessive[]) == T(1 + 0x1p-21));
        assert(categoricalInvCDF(T(1), excessive[]) == 1);
    }}
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import mir.ndslice.slice: sliced;
    static foreach (bits; [64, 128]) {{
        alias T = Fp!bits;
        alias U = UInt!bits;
        enum half = T(false, -cast(long)bits, U(1) << (bits - 1));
        enum below = T(false, -cast(long)bits - 1, U.max);
        enum quarterGap = T(false, -cast(long)bits - 2, U(1));
        T[4] p = [below, quarterGap, half, quarterGap];
        assert(categoricalCDF(1, p[]) == half);
        assert(categoricalInvCDF(half, p[]) == 2); // Rounded CDF alone is insufficient.
        assert(categoricalInvCDF(below, p[]) == 0);
        const T[4] qualified = p;
        static assert(is(typeof(categoricalCDF(1, qualified[])) == T));
        assert(categoricalCDF(1, qualified[].sliced) == half);
        assert(categoricalInvCDF(half, qualified[].sliced) == 2);
        const T query = half;
        assert(categoricalInvCDF!T(query, qualified[]) == 2);
        assert(categoricalInvCDF!T(query, qualified[].sliced) == 2);
        static assert(!__traits(compiles, categoricalInvCDF(Fp!(bits * 2)(0.5), p[])));
        immutable T[2] fixed = [T(0.25), T(0.75)];
        assert(categoricalPMF(1, fixed[]) == T(0.75));
        assert(categoricalCCDF(0, fixed[].sliced) == T(0.75));
        const tiny = T(false, long.min, U(1));
        T[2] suffix = [T(1), tiny];
        assert(categoricalCDF(1, suffix[]) == T(1));
        assert(categoricalCCDF(0, suffix[]) == tiny);
        assert(categoricalInvCDF(T(1), suffix[]) == 0);
        // Both tolerance boundaries are inclusive, checked before rounding.
        T[1] lower = [T(1 - 0x1p-20)];
        T[2] upper = [T(1), T(0x1p-20)];
        assert(categoricalValid!bits(lower[]) && categoricalValid!bits(upper[]));
        T[3] above = [T(1), T(0x1p-20), tiny];
        assert(!categoricalValid!bits(above[]));
        auto smaller = lower[0];
        smaller.coefficient -= 1;
        T[2] under = [smaller, T(false, -cast(long)bits - 2, U(3))];
        assert(!categoricalValid!bits(under[]));
    }}
}

version(mir_stat_test)
unittest
{
    import core.exception: AssertError;
    import std.exception: assertThrown;
    alias T = Fp!64;
    bool contractsEnabled;
    void probe() in { contractsEnabled = true; } do {}
    probe();
    if (contractsEnabled) {
        T[2] valid = [T(0.25), T(0.75)];
        T[0] empty;
        assertThrown!AssertError(categoricalInvCDF(T(0), empty[]));
        foreach (bad; [T(-0.1), T(1.1), T(double.infinity), T(double.nan)]) {
            T[2] invalid = [bad, T(0.5)];
            assertThrown!AssertError(categoricalPMF(0, invalid[]));
            assertThrown!AssertError(categoricalCDF(0, invalid[]));
            assertThrown!AssertError(categoricalCCDF(0, invalid[]));
            assertThrown!AssertError(categoricalLPMF(0, invalid[]));
            assertThrown!AssertError(categoricalInvCDF(T(0.5), invalid[]));
            assertThrown!AssertError(categoricalInvCDF(bad, valid[]));
        }
        T[2] wrongTotal = [T(0.25), T(0.25)];
        assertThrown!AssertError(categoricalCDF(0, wrongTotal[]));
        assertThrown!AssertError(categoricalPMF(2, valid[]));
        assertThrown!AssertError(categoricalCDF(2, valid[]));
        assertThrown!AssertError(categoricalCCDF(2, valid[]));
        assertThrown!AssertError(categoricalLPMF(2, valid[]));
    }
}


version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: strided, reversed;
    alias T = Fp!64;
    T[6] backing = [T(0.125), T(-1), T(0.375), T(-1), T(0.5), T(-1)];
    auto p = backing[].sliced.strided!0(2);
    assert(categoricalPMF(1, p) == T(0.375));
    assert(categoricalCDF(1, p) == T(0.5));
    assert(categoricalCCDF(1, p) == T(0.5));
    assert(categoricalInvCDF(T(0.5), p) == 1);
    assert(categoricalLPMF(1, p) == categoricalLPMF(1, T(0.125), T(0.375), T(0.5)));
    const backwards = p.reversed!0;
    assert(categoricalCDF(1, backwards) == T(0.875));
    assert(categoricalCCDF(1, backwards) == T(0.125));
    assert(categoricalInvCDF(T(0.5), backwards) == 0);
    assert(backing[1] == T(-1));
}
