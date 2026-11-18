/++
This module contains algorithms for the continuous $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution).

These functions describe a continuous distribution on the closed interval
$(D [lower, upper]), with $(D lower < upper). The forward functions require
$(D x) to lie in that interval; they do not extend the density or probabilities
outside it. The inverse CDF accepts both probability endpoints.

The one-argument overloads also accept $(D Fp!bits) on the unit interval.
The PDF and LPDF are exactly one and zero. The CDF and quantile preserve the
represented input, including at the exponent floor. The strict CCDF rounds
the exact binary complement to the result width (nearest, ties to even).
These operations use no hardware floating-point intermediates. Widths are
positive multiples of the native word size supported by $(D Fp).

The special-value tables assume finite distribution parameters in the stated
domain. Finite expressions are subject to floating-point rounding and range
limits. Contract violations are not prescribed return values: disabling
contracts does not make an invalid input supported.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2022-3 Mir Stat Authors.

+/

module mir.stat.distribution.uniform;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityValid, fpProbabilityComplement;
import mir.stat.internal.fp_uniform: uniformPointValid;
import mir.internal.utility: isFloatingPoint;

/++
Computes the uniform probability density function (PDF).

Returns a density, which may exceed one; it is not a point probability.

The table expresses the density in real arithmetic; the implementation also
handles finite bounds whose difference overflows the floating-point type.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D 1 / (upper - lower))))
    $(TR $(TD $(D x == upper)) $(TD $(D 1 / (upper - lower))))
    $(TR $(TD Outside $(D [lower, upper]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate PDF
    lower = finite lower support boundary
    upper = finite upper support boundary, greater than lower

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformPDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    // Opposite finite bounds can have an unrepresentable full width.
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return T(0.5) / (upper * T(0.5) - lower * T(0.5));
    return 1.0L / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformPDF == 1);
    assert(0.5.uniformPDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformPDF(1.0, 3.0).approxEqual(0.5));
}

/// The density includes both bounds; it can exceed one on a narrow interval.
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    assert(uniformPDF(0.0, 0.0, 0.5) == 2.0);
    assert(uniformPDF(0.5, 0.0, 0.5) == 2.0);
}

/// On a finite closed $(D Fp) interval the density is the reciprocal of the
/// exact width, rounded to nearest, ties to even. A sufficiently wide interval
/// can have a density that rounds to zero.
@safe pure nothrow @nogc
T uniformPDF(T)(const T x, const T lower, const T upper = T(1))
    if (is(T == Fp!size, size_t size))
    in (uniformPointValid(x, lower, upper), "require finite lower <= x <= upper and lower < upper")
{
    import mir.stat.internal.fp_uniform: uniformRatio;
    return uniformRatio(T(0), T(1), lower, upper);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {
        assert(uniformPDF(T(-1), T(-3), T(5)) == T(0.125));
    }
    static foreach (bits; [64u, 128u]) {{
        alias T = Fp!bits;
        const tiny = T(false, long.min, UInt!bits(1));
        auto reciprocal = T(1);
        reciprocal.exponent = long.max - bits + 2;
        assert(uniformPDF(tiny, T(0), tiny) == reciprocal);
        const huge = T(false, long.max - 1, UInt!bits.max);
        assert(uniformPDF(T(0), T(0), huge) == T(0));
        assert(uniformPDF(T(0), T(-1)) == T(0.5));
    }}
}

/// Unit interval with a represented $(D Fp) input. Returns exactly one, including at both support endpoints.
@safe pure nothrow @nogc
T uniformPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(x), "x must be finite and in [0, 1]")
{
    return T(1);
}

// Shared dyadic inputs have exact results in every tested representation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {{
        foreach (i; 0 .. 9)
        {
            const x = T(i / 8.0);
            assert(uniformPDF(x) == T(1));
        }
    }}
}

/++
Computes the uniform cumulative distribution function (CDF).

Returns the lower-tail probability $(D P(X <= x)).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D 0)))
    $(TR $(TD $(D x == upper)) $(TD $(D 1)))
    $(TR $(TD Outside $(D [lower, upper]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CDF
    lower = finite lower support boundary
    upper = finite upper support boundary, greater than lower

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformCDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (x * T(0.5) - lower * T(0.5))
            / (upper * T(0.5) - lower * T(0.5));
    return (x - lower) / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformCDF == 0.5);
    assert(0.5.uniformCDF(0.0, 1.5).approxEqual(1.0 / 3));
    assert(2.5.uniformCDF(1.0, 3.0).approxEqual(3.0 / 4));
}

/// With finite $(D Fp) bounds, rounds $(D (x-lower)/(upper-lower)) to
/// nearest, ties to even. Both differences use the represented inputs.
@safe pure nothrow @nogc
T uniformCDF(T)(const T x, const T lower, const T upper = T(1))
    if (is(T == Fp!size, size_t size))
    in (uniformPointValid(x, lower, upper), "require finite lower <= x <= upper and lower < upper")
{
    import mir.stat.internal.fp_uniform: uniformRatio;
    return uniformRatio(lower, x, lower, upper);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {
        assert(uniformCDF(T(-1), T(-3), T(5)) == T(0.25));
    }
    static foreach (bits; [64u, 128u]) {{
        alias T = Fp!bits;
        const tiny = T(false, long.min, UInt!bits(1));
        assert(uniformCDF(tiny, T(0), T(2)) == T(0)); // Halfway to the smallest result.
        // The repeating binary expansion of 1/3 rounds upward at these even widths.
        auto thirds = UInt!bits.max;
        thirds /= UInt!bits(3);
        thirds = (thirds << 1) + 1;
        assert(uniformCDF(T(1), T(0), T(3)) == T(false, -long(bits)-1, thirds));
        const three = T(false, long.min, UInt!bits(3));
        assert(uniformCDF(three, T(0), T(2)) == T(false, long.min, UInt!bits(2)));
        const upper = T(false, long.max - 1, UInt!bits.max);
        const lower = T(true, upper.exponent, upper.coefficient);
        assert(uniformCDF(T(0), lower, upper) == T(0.5));
        assert(uniformCDF(lower, lower, upper) == T(0));
        assert(uniformCDF(upper, lower, upper) == T(1));
        assert(uniformCDF(T(0), T(-1)) == T(0.5));
    }}
}

/// Unit interval with a represented $(D Fp) input. Returns the represented input exactly, including at the exponent floor.
@safe pure nothrow @nogc
T uniformCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(x), "x must be finite and in [0, 1]")
{
    return x;
}

// Shared dyadic inputs have exact results in every tested representation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {{
        foreach (i; 0 .. 9)
        {
            const x = T(i / 8.0);
            assert(uniformCDF(x) == x);
        }
    }}
}

// Preserve the smallest positive represented input.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static if (size_t.sizeof == 4)
        alias widths = AliasSeq!(32u, 64u, 96u, 128u, 256u);
    else
        alias widths = AliasSeq!(64u, 128u, 256u);
    static foreach (bits; widths)
    {{
        alias T = Fp!bits;
        const tiny = T(false, long.min, UInt!bits(1));
        assert(uniformCDF(tiny) == tiny);
    }}
}

/++
Computes the uniform complementary cumulative distribution function (CCDF).

Returns the upper-tail probability $(D P(X > x)). Use this function directly
when the upper tail is small, rather than subtracting the CDF from one.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D 1)))
    $(TR $(TD $(D x == upper)) $(TD $(D 0)))
    $(TR $(TD Outside $(D [lower, upper]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate CCDF
    lower = finite lower support boundary
    upper = finite upper support boundary, greater than lower

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformCCDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (upper * T(0.5) - x * T(0.5))
            / (upper * T(0.5) - lower * T(0.5));
    return (upper - x) / width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformCCDF == 0.5);
    assert(0.5.uniformCCDF(0.0, 1.5).approxEqual(2.0 / 3));
    assert(2.5.uniformCCDF(1.0, 3.0).approxEqual(1.0 / 4));
}

/// For finite $(D Fp) bounds, rounds the strict tail $(D (upper-x)/(upper-lower))
/// directly to nearest, ties to even, including subnormal probabilities.
@safe pure nothrow @nogc
T uniformCCDF(T)(const T x, const T lower, const T upper = T(1))
    if (is(T == Fp!size, size_t size))
    in (uniformPointValid(x, lower, upper), "require finite lower <= x <= upper and lower < upper")
{
    import mir.stat.internal.fp_uniform: uniformRatio;
    return uniformRatio(x, upper, lower, upper);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {
        assert(uniformCCDF(T(-1), T(-3), T(5)) == T(0.75));
    }
    static foreach (bits; [64u, 128u]) {{
        alias T = Fp!bits;
        auto lower = T(-1);
        lower.exponent += 20_000;
        auto tail = T(1);
        tail.exponent -= 20_000;
        // 1/(2^20000+1) rounds to 2^-20000 even though the CDF rounds to one.
        assert(uniformCCDF(T(0), lower) == tail);
        assert(uniformCDF(T(0), lower) == T(1));
        assert(uniformCCDF(lower, lower) == T(1));
        assert(uniformCCDF(T(1), lower) == T(0));
    }}
}

/// Unit interval with a represented $(D Fp) input. Rounds the exact binary complement to nearest, ties to even.
@safe pure nothrow @nogc
T uniformCCDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(x), "x must be finite and in [0, 1]")
{
    return fpProbabilityComplement(x);
}

// Shared dyadic inputs have exact results in every tested representation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {{
        foreach (i; 0 .. 9)
        {
            const x = T(i / 8.0);
            assert(uniformCCDF(x) == T((8 - i) / 8.0));
        }
    }}
}

// Exact dyadic references at the exponent floor and on either side of the
// midpoint between one and its predecessor. The latter spacing is 2^-bits.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static if (size_t.sizeof == 4)
        alias widths = AliasSeq!(32u, 64u, 96u, 128u, 256u);
    else
        alias widths = AliasSeq!(64u, 128u, 256u);
    static foreach (bits; widths)
    {{
        alias T = Fp!bits;
        const tiny = T(false, long.min, UInt!bits(1));
        assert(uniformCCDF(tiny) == T(1));

        UInt!bits ones;
        ones.data[] = size_t.max;
        const predecessor = T(false, -long(bits), ones);
        auto spacing = T(1);
        spacing.exponent -= bits;
        assert(uniformCCDF(predecessor) == spacing);

        auto midpoint = spacing;
        midpoint.exponent--;
        assert(uniformCCDF(midpoint) == T(1));
        const below = T(false, midpoint.exponent - 1, ones);
        assert(uniformCCDF(below) == T(1));
        auto above = midpoint;
        above.coefficient += 1;
        assert(uniformCCDF(above) == predecessor);
    }}
}

/++
Computes the uniform inverse cumulative distribution function (InvCDF)

Returns the quantile at cumulative probability $(D p).

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D p == 0)) $(TD $(D lower)))
    $(TR $(TD $(D p == 1)) $(TD $(D upper)))
    $(TR $(TD $(D p < 0) or $(D p > 1) or NaN) $(TD Contract violation))
)

Params:
    p = value to evaluate InvCDF
    lower = finite lower support boundary
    upper = finite upper support boundary, greater than lower

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformInvCDF(T)(const T p, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (p >= 0, "p must be greater than or equal to 0")
    in (p <= 1, "p must be less than or equal to 1")
    in (lower < upper, "lower must be less than upper")
{
    if (p == 0) return lower;
    if (p == 1) return upper;
    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return (1 - p) * lower + p * upper;
    return lower + p * width;
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual;
    assert(0.5.uniformInvCDF == 0.5);
    assert((1.0 / 3).uniformInvCDF(0.0, 1.5).approxEqual(0.5));
    assert((3.0 / 4).uniformInvCDF(1.0, 3.0).approxEqual(2.5));
}

/// Unit interval with a represented $(D Fp) input. Returns the represented probability exactly, including at the exponent floor.
@safe pure nothrow @nogc
T uniformInvCDF(T)(const T p)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(p), "p must be finite and in [0, 1]")
{
    return p;
}

// Shared dyadic inputs have exact results in every tested representation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {{
        foreach (i; 0 .. 9)
        {
            const x = T(i / 8.0);
            assert(uniformInvCDF(x) == x);
        }
    }}
}

// Preserve the smallest positive represented input.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static if (size_t.sizeof == 4)
        alias widths = AliasSeq!(32u, 64u, 96u, 128u, 256u);
    else
        alias widths = AliasSeq!(64u, 128u, 256u);
    static foreach (bits; widths)
    {{
        alias T = Fp!bits;
        const tiny = T(false, long.min, UInt!bits(1));
        assert(uniformInvCDF(tiny) == tiny);
    }}
}

/++
Computes the uniform log probability density function (LPDF)

Returns the natural logarithm of the density. A negative value is valid;
it is not a probability.

The table expresses the density in real arithmetic; the implementation also
handles finite bounds whose difference overflows the floating-point type.

Special values:
$(TABLE
    $(TR $(TH Input) $(TH Result))
    $(TR $(TD $(D x == lower)) $(TD $(D -log(upper - lower))))
    $(TR $(TD $(D x == upper)) $(TD $(D -log(upper - lower))))
    $(TR $(TD Outside $(D [lower, upper]) or NaN) $(TD Contract violation))
)

Params:
    x = value to evaluate LPDF
    lower = finite lower support boundary
    upper = finite upper support boundary, greater than lower

See_also:
    $(LINK2 https://en.wikipedia.org/wiki/Continuous_uniform_distribution, Uniform Distribution)
+/
@safe pure nothrow @nogc
T uniformLPDF(T)(const T x, const T lower = 0, const T upper = 1)
    if (isFloatingPoint!T)
    in (x >= lower, "x must be greater than or equal to lower bound")
    in (x <= upper, "x must be less than or equal to upper bound")
    in (lower < upper, "lower must be less than upper")
{
    import mir.math.common: log;
    import mir.math.constant: LN2;

    const T width = upper - lower;
    if (width == T.infinity && lower > -T.infinity && upper < T.infinity)
        return -log(upper * T(0.5) - lower * T(0.5)) - T(LN2);
    return -log(width);
}

///
version(mir_stat_test)
@safe pure nothrow @nogc
unittest
{
    import mir.math.common: approxEqual, log;
    assert(0.5.uniformLPDF == 0);
    assert(0.5.uniformLPDF(0.0, 1.5).approxEqual(-log(1.5)));
    assert(1.5.uniformLPDF(1.0, 3.0).approxEqual(-log(2.0)));
}

/// Evaluates $(D -log(upper-lower)) with guard precision. Near unit width
/// the exact displacement from one preserves cancellation-sensitive results.
/// The logarithm has no universal correct-rounding guarantee.
@safe pure nothrow @nogc
T uniformLPDF(T)(const T x, const T lower, const T upper = T(1))
    if (is(T == Fp!size, size_t size))
    in (uniformPointValid(x, lower, upper), "require finite lower <= x <= upper and lower < upper")
{
    import mir.stat.internal.fp_uniform: uniformLogDensity;
    return uniformLogDensity(lower, upper);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.bignum.fixed: UInt;
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {
        assert(uniformLPDF(T(-1), T(-1), T(0)) == T(0));
    }
    alias T = Fp!64;
    alias U = UInt!64;
    const tiny = T(false, long.min, U(1));
    const negTiny = T(true, long.min, U(1));
    assert(uniformLPDF(T(1), tiny) == tiny);
    assert(uniformLPDF(T(1), negTiny) == negTiny);
    // Independent rounded references for -log(3), 2^63*log(2), and the
    // negative log of twice the largest represented finite value.
    assert(uniformLPDF(T(1), T(-2)) == T(true, -63, U(0x8c9f53d5681854bbUL)));
    assert(uniformLPDF(tiny, T(0), tiny) == T(false, -1, U(0xb17217f7d1cf79acUL)));
    const huge = T(false, long.max - 1, U.max);
    const negHuge = T(true, huge.exponent, huge.coefficient);
    assert(uniformLPDF(T(0), negHuge, huge) == T(true, -1, U(0xb17217f7d1cf7a03UL)));
}

/// Unit interval with a represented $(D Fp) input. Returns exactly zero, including at both support endpoints.
@safe pure nothrow @nogc
T uniformLPDF(T)(const T x)
    if (is(T == Fp!size, size_t size))
    in (fpProbabilityValid(x), "x must be finite and in [0, 1]")
{
    return T(0);
}

// Shared dyadic inputs have exact results in every tested representation.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, real, Fp!64, Fp!128))
    {{
        foreach (i; 0 .. 9)
        {
            const x = T(i / 8.0);
            assert(uniformLPDF(x) == T(0));
        }
    }}
}

// Finite bounds remain usable even when their difference overflows.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.math.common: approxEqual, log;
    import mir.math.constant: LN2;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T m = T.max;
        assert(uniformCDF(T(0), -m, m) == T(0.5));
        assert(uniformCCDF(T(0), -m, m) == T(0.5));
        // Identity also verifies positive zero and avoids a DMD real equality
        // code-generation issue when these calls are inlined on Linux.
        assert(uniformCDF(-m, -m, m) is T(0));
        assert(uniformCDF(m, -m, m) == 1);
        assert(uniformCCDF(-m, -m, m) == 1);
        assert(uniformCCDF(m, -m, m) is T(0));
        assert(uniformPDF(T(0), -m, m) > 0);
        assert(approxEqual(uniformPDF(T(0), -m, m), T(0.5)/m,
            T(8)*T.epsilon, T(0)));
        assert(approxEqual(uniformLPDF(T(0), -m, m), -log(m)-T(LN2),
            T(8)*T.epsilon, T(0)));
        assert(uniformInvCDF(T(0), -m, m) == -m);
        assert(uniformInvCDF(T(1), -m, m) == m);
        assert(uniformInvCDF(T(0.5), -m, m) is T(0));
        // Unequal bounds prevent a symmetry-only implementation from passing.
        const T lower = -m, upper = m*T(0.5);
        foreach (p; [T(0.25), T(0.5), T(0.75)])
        {
            const T x = uniformInvCDF(p, lower, upper);
            assert(x >= lower && x <= upper);
            assert(approxEqual(uniformCDF(x, lower, upper), p, T(8)*T.epsilon));
            assert(approxEqual(uniformCCDF(x, lower, upper), 1-p, T(8)*T.epsilon));
        }
        // Exact inverse endpoints must not depend on cancellation in the width.
        assert(uniformInvCDF(T(1), -m*T(0.5), T(1)) == 1);
    }}
}

// Narrow intervals can round interior quantiles to a bound, but must retain
// exact probability endpoints, bounded results, and monotonicity.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import std.math: nextUp, nextDown;
    static foreach (T; AliasSeq!(float, double, real))
    {{
        const T tiny = nextUp(T(0));
        const T[2][7] intervals = [[T(0), tiny], [tiny, 2*tiny], [-tiny, tiny],
            [T(1), nextUp(T(1))], [-T.max, T.max],
            [-T.max, T.max/T(2)], [-T.max/T(2), T(-1)]];
        foreach (bounds; intervals)
        {
            const T lower = bounds[0], upper = bounds[1];
            assert(uniformCDF(lower, lower, upper) == 0);
            assert(uniformCDF(upper, lower, upper) == 1);
            assert(uniformCCDF(lower, lower, upper) == 1);
            assert(uniformCCDF(upper, lower, upper) == 0);
            T previous = lower;
            foreach (p; [T(0), tiny, T.epsilon/T(2), T(0.25), T(0.5),
                T(0.75), nextDown(T(1)), T(1)])
            {
                const T x = uniformInvCDF(p, lower, upper);
                assert(x >= lower && x <= upper);
                assert(x >= previous);
                previous = x;
            }
            assert(uniformInvCDF(T(0), lower, upper) == lower);
            assert(uniformInvCDF(T(1), lower, upper) == upper);
        }
    }}
}
