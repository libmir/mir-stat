/++
Structural matrix policies and shared numeric type constraints.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.types;

import mir.ndslice.slice: isSlice;
import mir.internal.utility: isComplex;
import std.traits: Unqual, isFloatingPoint, isMutable;

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

import mir.ndslice.topology: unpack, universal;
import mir.ndslice.dynamic: transposed;

/// Strip descriptor constness while preserving const elements and retained ownership.
package(mir.stat) template ObservationDescriptor(A)
{
    static if (isMutable!A) alias ObservationDescriptor = A;
    else alias ObservationDescriptor = typeof(A.init.lightConst);
}

/// A rectangular one-dimensional collection of real observation vectors.
/// alongDim/byDim produce this structure; arbitrary arrays of vectors do not.
package(mir.stat) enum isObservationSelection(A) = isSlice!A && A.N == 1 &&
    __traits(compiles, ObservationDescriptor!A.init.unpack) &&
    isSlice!(typeof(A.init[0])) && typeof(A.init[0]).N == 1 &&
    numericSlice!(typeof(ObservationDescriptor!A.init.unpack)) &&
    isFloatingPoint!(Scalar!(typeof(ObservationDescriptor!A.init.unpack)));

/// Matrix descriptor with coordinates first and selected observations second.
package(mir.stat) alias ObservationColumns(A) = typeof(ObservationDescriptor!A.init.unpack.universal.transposed);

/// Recover a column-observation matrix without copying or dropping ownership.
/// Selection order, lengths, and strides are preserved, including empty selections.
package(mir.stat) ObservationColumns!A observationColumns(A)(return scope A selection)
    if (isObservationSelection!A)
{
    static if (isMutable!A) return selection.unpack.universal.transposed;
    else return selection.lightConst.unpack.universal.transposed;
}
