/++
Structural matrix policies and shared numeric type constraints.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.types;

import mir.ndslice.slice: isSlice;
import mir.internal.utility: isComplex;
import std.traits: Unqual, isFloatingPoint;

/// Selected triangle of the logical matrix view.
enum TrianglePolicy
{
    /// Read entries on or above the diagonal.
    upper,
    /// Read entries on or below the diagonal.
    lower,
}

/// Whether diagonal entries come from storage or are implicitly one.
enum DiagonalPolicy
{
    /// Read the stored diagonal values.
    nonUnit,
    /// Use one without reading stored diagonal values.
    unit,
}

package alias Scalar(A) = Unqual!(typeof(A.init._iterator[0]));
package enum numericSlice(A) = isSlice!A && (A.N == 1 || A.N == 2) &&
    (isFloatingPoint!(Scalar!A) || isComplex!(Scalar!A));
