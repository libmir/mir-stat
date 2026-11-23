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
