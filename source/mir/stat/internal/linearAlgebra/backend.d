/++
Optional backend availability and storage eligibility checks.
These helpers do not allocate or perform matrix operations.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.backend;

import mir.stat.internal.linearAlgebra.types: Scalar;
import mir.internal.utility: realType;

package enum hasBlas = is(typeof({ import mir.blas; }));
package enum hasLapack = is(typeof({ import mir.lapack; }));

package enum blasNumber(T) = is(realType!T == float) || is(realType!T == double);
// BLAS needs native pointers. Computed ndslice iterators use the native D kernel.
package enum nativeSlice(A) = is(typeof(A.init.lightScope._iterator) : const(Scalar!A)*);

// BLAS requires representable dimensions and nonoverlapping rows or columns.
/// Whether matrix dimensions and positive strides can be represented by BLAS without overlap.
package bool blasMatrix(A)(scope auto ref A a)
{
    import cblas: blasint;
    return a.length!0 <= blasint.max && a.length!1 <= blasint.max &&
        a._stride!0 > 0 && a._stride!1 > 0 &&
        a._stride!0 <= blasint.max && a._stride!1 <= blasint.max &&
        ((a._stride!1 == 1 && a._stride!0 >= a.length!1) ||
         (a._stride!0 == 1 && a._stride!1 >= a.length!0));
}

/// Whether a vector has a positive stride and dimensions representable by BLAS.
package bool blasVector(A)(scope auto ref A a)
{
    import cblas: blasint;
    return a.length <= blasint.max && a._stride!0 > 0 && a._stride!0 <= blasint.max;
}
