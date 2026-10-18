/++
Internal centered gamma log-density calculation.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.gamma;

// Requires finite a >= 32 and abs(d) < 1/2, with d = (x/scale-a)/a.
// Keep the displacement separate so callers can retain integer shape bits.
package(mir.stat)
@safe pure nothrow @nogc
T centeredGammaLPDF(T)(const T a, const T d, const T logScale)
{
    pragma(inline, false);
    import mir.math.common: log;
    import mir.stat.constant: LOGSQRT2PI;
    const T t = d / (2 + d);
    const T t2 = t * t;
    // |t| < 1/3 here. Each extra term reduces the truncation error by
    // at least nine; choose enough terms for the significand of T.
    enum terms = (T.mant_dig + 2) / 3 + 2;
    T polynomial = T(1) / (2 * terms + 1);
    static foreach (j; 0 .. terms - 1)
        polynomial = T(1) / (2 * (terms - j - 1) + 1) + t2 * polynomial;
    const T sum = t * t2 * polynomial;
    // Reuse log(1+d)=2*(t+sum); no second logarithm is needed.
    const T r = 1 / a, r2 = r * r;
    // At a>=32 the first omitted Stirling term is below 1e-24.
    const T correction = r * (T(1)/12 + r2 * (-T(1)/360 + r2 * (T(1)/1260
        + r2 * (-T(1)/1680 + r2 * (T(1)/1188 + r2 * (-T(691)/360360 + r2 / 156))))));
    return a * (-d * t + 2 * sum) - 2 * (t + sum) - T(0.5) * log(a)
        - logScale - T(LOGSQRT2PI) - correction;
}
