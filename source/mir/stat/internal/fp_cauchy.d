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

    // tan(pi*p) for 0 < p < 1/4. The cosine Taylor term of order 2n
    // divided by 2n+1 is the corresponding term of sin(x)/x.
    static S tangent(S probability)
    {
        const angle = probability.multiply(S.from(guardedPi!work()));
        const square = squareOnGrid(angle);
        I term = I(1) << work;
        I cosine = term, sinc = term;
        for (uint n = 1; ; ++n)
        {
            term = divideWord(divideWord(multiplyOnGrid(term, square), 2 * n - 1), 2 * n);
            if (!term) break;
            const sineTerm = divideWord(term, 2 * n + 1);
            if (n & 1)
            {
                cosine -= term;
                sinc -= sineTerm;
            }
            else
            {
                cosine += term;
                sinc += sineTerm;
            }
        }
        return angle.multiply(S.from(Fp!work(sinc))).divide(S.from(Fp!work(cosine)));
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

package(mir.stat) @safe pure nothrow @nogc
Fp!bits cauchyQuantile(uint bits)(Fp!bits p, Fp!bits location, Fp!bits scale)
{
    import mir.stat.internal.fp_location_scale: half;
    alias T = Fp!bits;
    alias A = CauchyTrig!bits;
    alias S = A.S;
    if (!p.coefficient || !position(p, T(1)))
        return T(!p.coefficient, long.max, UInt!bits(0));
    const median = half!bits();
    const side = position(p, median);
    if (!side)
        return location;

    const lower = side < 0;
    auto quartile = T(lower ? 1 : 3);
    quartile.exponent -= 2;
    const relative = position(p, quartile);
    const width = S.from(scale);
    if (!relative)
        return addOffset!bits(location, width, lower);

    // Use cotangent near endpoints and tangent around the median.
    // Apply scale before narrowing an offset outside the output range.
    const outer = lower == (relative < 0);
    const argument = outer ? (lower ? S.from(p) : distance!(A.work)(T(1), p))
                           : distance!(A.work)(p, median);
    const tangent = A.tangent(argument);
    const offset = outer ? width.divide(tangent) : width.multiply(tangent);
    return addOffset!bits(location, offset, lower);
}
