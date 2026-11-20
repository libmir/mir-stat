/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_laplace;

import mir.bignum.fp: Fp;
import mir.stat.internal.fp_probability: fpProbabilityComplement, fpProbabilityLess;
import mir.stat.internal.fp_scaled_probability: ProbabilityScaled, exponentBias, scaledDecay;

@safe pure nothrow @nogc:

// Construct a normalized half even when a double significand would not fit.
private Fp!bits half(uint bits)()
{
    auto value = Fp!bits(1);
    value.exponent -= 1;
    return value;
}

// Near the median, the first two decay terms can reach a rounding midpoint.
// Keep the cubic term (of order 2^(-3*bits)) when deciding the final side.
private alias Decay(uint bits) = ProbabilityScaled!(3 * bits + 192);

// Align the two coefficients at the smaller exponent before taking the
// absolute difference. Across a larger gap even a full-width low coefficient
// contributes less than a working ulp relative to a one-bit high coefficient.
private auto distance(uint work, uint bits)(Fp!bits x, Fp!bits location)
{
    alias S = ProbabilityScaled!work;
    if (!x.coefficient)
        return S.from(location);
    if (!location.coefficient)
        return S.from(x);
    const high = x.exponent >= location.exponent ? x : location;
    const low = x.exponent >= location.exponent ? location : x;
    const gap = cast(ulong)high.exponent - cast(ulong)low.exponent;
    if (gap >= work + bits)
        return S.from(high);
    // The exact sum fits even when both inputs have full coefficients.
    enum capacity = work + 2 * bits + 64;
    const a = high.coefficient.toSize!capacity << cast(size_t)gap;
    const b = low.coefficient.toSize!capacity;
    const c = high.sign != low.sign ? a + b : a >= b ? a - b : b - a;
    return S.from(Fp!capacity(false, low.exponent, c));
}

// Compare represented values, including signed zero and unnormalized coefficients.
private int position(uint bits)(Fp!bits x, Fp!bits location)
{
    if (x.sign != location.sign)
    {
        if (!x.coefficient && !location.coefficient)
            return 0;
        return x.sign ? -1 : 1;
    }
    if (fpProbabilityLess(x, location))
        return x.sign ? 1 : -1;
    if (fpProbabilityLess(location, x))
        return x.sign ? -1 : 1;
    return 0;
}

// For distinct finite points, retain the separate exponent until the caller
// has applied its normalization. At z >= 2^65 even division by the smallest
// scale (2^long.min) leaves less than half the smallest output quantum.
private auto halfDecay(uint bits, uint work)(Fp!bits x, Fp!bits location, ProbabilityScaled!work scale)
{
    alias S = ProbabilityScaled!work;
    const z = distance!work(x, location).divide(scale);
    if (z.exponent >= exponentBias + 66)
        return S(Fp!work(0));
    auto tail = scaledDecay(z);
    tail.exponent -= 1;
    return tail;
}

package(mir.stat) Fp!bits laplaceTail(bool upper, uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(x.sign == upper ? 1 : 0);
    const side = position(x, location);
    if (!side)
        return half!bits;
    const smallTail = upper != (side < 0);
    alias S = Decay!bits;
    auto tail = halfDecay(x, location, S.from(scale));
    if (!tail.fraction.coefficient)
        return T(smallTail ? 0 : 1);
    if (smallTail)
        return tail.finish!bits;
    if (tail.exponent + bits + 1 <= exponentBias)
        return T(1);
    tail.fraction.exponent -= cast(long)cast(ulong)(exponentBias - tail.exponent);
    return cast(T) fpProbabilityComplement(tail.fraction);
}
