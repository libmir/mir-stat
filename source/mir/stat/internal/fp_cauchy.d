/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: Mir Stat Authors.
+/
module mir.stat.internal.fp_cauchy;

import mir.bignum.fixed: UInt;
import mir.bignum.fp: Fp;
import mir.stat.internal.fp_location_scale: addOffset, distance, position;
import mir.stat.internal.fp_pi: guardedPi;
import mir.stat.internal.fp_shape_transform: ShapeTransform;

package(mir.stat) @safe pure nothrow @nogc
Fp!bits cauchyDensity(bool logarithmic, uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    alias T = Fp!bits;
    alias B = ShapeTransform!bits;
    alias S = B.S;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return logarithmic ? T(true, long.max, UInt!bits(0)) : T(0);

    // f = scale / (pi * (scale^2 + (x-location)^2)). Keep the
    // exponents separate through both squaring and final division.
    const width = S.from(scale);
    auto squares = width.multiply(width);
    if (position(x, location))
    {
        const offset = distance!(B.work)(x, location);
        squares = squares.add(offset.multiply(offset));
    }
    const density = width.divide(squares.multiply(S.from(guardedPi!(B.work)())));
    static if (logarithmic)
        return cast(T)B.logarithm(density);
    else
        return addOffset!bits(T(0), density, false);
}

private struct CauchyTrig(uint bits)
{
    alias B = ShapeTransform!bits;
    enum work = B.work;
    alias S = B.S;
    alias I = UInt!(work + 64);

    @safe pure nothrow @nogc:

    // Fixed-point series use work fractional bits. Keep the leading factor
    // separately scaled so tiny angles survive below this absolute grid.
    static I squareOnGrid(S x)
    {
        import mir.stat.internal.fp_scaled_probability: exponentBias;
        const square = x.multiply(x);
        const gap = exponentBias - square.exponent;
        if (gap >= UInt!128(work))
            return I(0);
        return square.fraction.coefficient.toSize!(work + 64) >> cast(size_t)gap;
    }

    static I multiplyOnGrid(I a, I b)
    {
        auto product = a.toSize!(2 * work + 128);
        product *= b;
        return (product >> work).toSize!(work + 64);
    }

    static I divideWord(I value, uint divisor)
    {
        static if (size_t.sizeof == 4)
            value.view /= divisor;
        else
            value /= UInt!64(divisor);
        return value;
    }

    // atan(x)/pi, for 0 < x <= 1. Above 1/2 reduce around pi/4
    // so successive terms shrink by a factor of at least four.
    static S atanFraction(S x)
    {
        import mir.stat.internal.fp_scaled_probability: exponentBias;
        alias W = Fp!work;
        const one = S.from(W(1));
        S quarter = one;
        quarter.exponent -= 2;
        const reduce = x.exponent >= exponentBias;
        if (reduce)
        {
            const diff = B.combine(B.Term(one, false, true), B.Term(x, true, true));
            if (!diff.present)
                return quarter;
            x = diff.value.divide(one.add(x));
        }
        const square = squareOnGrid(x);
        I power = I(1) << work;
        auto sum = power;
        for (uint n = 1; ; ++n)
        {
            power = multiplyOnGrid(power, square);
            const term = divideWord(power, 2 * n + 1);
            if (!term) break;
            if (n & 1) sum -= term;
            else sum += term;
        }
        auto factor = S.from(W(sum));
        factor.exponent -= work;
        const angle = x.multiply(factor).divide(S.from(guardedPi!work()));
        return reduce ? B.combine(B.Term(quarter, false, true),
            B.Term(angle, true, true)).value : angle;
    }
}

package(mir.stat) @safe pure nothrow @nogc
Fp!bits cauchyTail(bool upper, uint bits)(Fp!bits x, Fp!bits location, Fp!bits scale)
{
    import mir.stat.internal.fp_location_scale: half;
    alias T = Fp!bits;
    alias A = CauchyTrig!bits;
    alias S = A.S;
    if (x.isNaN)
        return x;
    if (x.isInfinity)
        return T(x.sign == upper ? 1 : 0);
    const side = position(x, location);
    if (!side)
        return half!bits();
    const small = upper == (side > 0);
    const offset = distance!(A.work)(x, location);
    const width = S.from(scale);
    const far = offset.exponent > width.exponent ||
        (offset.exponent == width.exponent && offset.fraction.coefficient >= width.fraction.coefficient);
    if (far)
    {
        const tail = A.atanFraction(width.divide(offset));
        return small ? tail.finish!bits : addOffset!bits(T(1), tail, true);
    }
    return addOffset!bits(half!bits(), A.atanFraction(offset.divide(width)), small);
}
