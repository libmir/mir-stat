/++
Internal allocation-free triangular matrix/vector operations.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.math.internal.linearAlgebra.triangularVector;

import mir.math.internal.linearAlgebra.triangular: Triangle;
import mir.math.internal.linearAlgebra.products: Scalar, nativeSlice, blasMatrix, blasVector;

// Callers establish square shape, matching vector length, writable disjoint
// vector elements, and a nonoverlapping read-only triangular matrix.
package(mir) void triangularVectorInPlace(bool solve, Triangle triangle, A, X)(
    scope auto ref A a, scope auto ref X x)
{
    static assert(triangle == Triangle.upper || triangle == Triangle.lower);
    if (!x.length) return;
    static if (is(typeof({ import mir.blas; })) && nativeSlice!A && nativeSlice!X &&
        (is(Scalar!A == float) || is(Scalar!A == double)))
    {
        if (blasMatrix(a) && blasVector(x))
        {
            import mir.blas: trsv, trmv, Uplo;
            import cblas: Diag;
            auto ap = a.lightScope.lightConst;
            auto xp = x.lightScope;
            enum uplo = triangle == Triangle.upper ? Uplo.Upper : Uplo.Lower;
            static if (solve) trsv(uplo, Diag.NonUnit, ap, xp);
            else trmv(uplo, Diag.NonUnit, ap, xp);
            return;
        }
    }
    // Solve in dependency order; multiply in the opposite order to preserve
    // every source element until its last use. No temporary vector is needed.
    enum reverse = solve ? triangle == Triangle.upper : triangle == Triangle.lower;
    foreach (step; 0 .. x.length)
    {
        auto i = reverse ? x.length - 1 - step : step;
        auto value = solve ? x[i] : Scalar!A(0);
        auto first = triangle == Triangle.upper ? i + (solve ? 1 : 0) : 0;
        auto last = triangle == Triangle.upper ? x.length : i + (solve ? 0 : 1);
        foreach (k; first .. last)
            static if (solve) value -= a[i,k] * x[k];
            else value += a[i,k] * x[k];
        static if (solve) x[i] = value / a[i,i];
        else x[i] = value;
    }
}
