/++
Internal Stirling correction shared by logarithmic beta and rising products.

Callers combine the leading log-gamma terms before evaluation to avoid
cancellation. This helper supplies only the remaining correction at z >= 16,
using the reciprocal argument so callers need not form a potentially
unrepresentable endpoint.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.math.internal.log_gamma;

// Stirling's correction at z>=16, evaluated with r=1/z. The first omitted
// term is below 2e-23 at z=16, below double and 80-bit real precision.
package(mir.math) @safe pure nothrow @nogc
T logGammaStirlingCorrection(T)(const T r)
{
    const T r2 = r * r;
    return r * (T(1) / 12 + r2 * (-T(1) / 360 + r2 * (T(1) / 1260
        + r2 * (-T(1) / 1680 + r2 * (T(1) / 1188 + r2 * (-T(691) / 360360
        + r2 * (T(1) / 156 + r2 * (-T(3617) / 122400
        + r2 * (T(43867) / 244188)))))))));
}
