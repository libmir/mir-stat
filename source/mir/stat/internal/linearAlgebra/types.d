/++
Structural matrix policies and shared numeric type constraints.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.linearAlgebra.types;

import mir.ndslice.slice: isSlice, toSlice;
import mir.internal.utility: isComplex;
import std.traits: Unqual, isFloatingPoint, isMutable;

/// Representation of the supplied covariance or precision matrix.
enum CovarianceRepresentation
{
    /// Symmetric positive-definite covariance S. Factorized once per call, including a whole batch.
    covariance,
    /// Covariance factor reused without factorization: upper U with S = U.transpose * U, or lower L with S = L * L.transpose.
    covarianceCholesky,
    /// Symmetric positive-definite precision matrix P = inverse(S).
    inverseCovariance,
    /// Upper R with P = R.transpose * R, or lower Q with P = Q * Q.transpose.
    inverseCovarianceCholesky,
}

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

private import mir.stat.internal.linearAlgebra.views: isStructuralView, DiagonalInfo, diagonalView;

// Borrow ndslice storage directly; array inputs use the ordinary slice view.
// The public call retains the input owners for the lifetime of these views.
/// Borrow an ndslice or array as a view tied to the input lifetime; callers retain owning inputs.
package(mir.stat) auto asView(A)(return scope auto ref A value)
{
    static if (DiagonalInfo!A.valid) return diagonalView(value.borrowStorage);
    else static if (isSlice!A) return value.lightScope;
    else return value.toSlice;
}

/// Borrowed ndslice descriptor used for array and slice inputs.
private alias View(A) = typeof(A.init.asView);
/// Unqualified floating-point element type of an observation or workspace.
package(mir.stat) template Element(A)
{
    static if (DiagonalInfo!A.valid) alias Element = Unqual!(typeof(A.init[0,0]));
    else alias Element = Unqual!(typeof(View!A.init._iterator[0]));
}
/// Whether an input can be borrowed as a real floating-point vector.
package(mir.stat) enum vector(A) = is(typeof(A.init.asView)) && View!A.N == 1 && isFloatingPoint!(Element!A);
/// Whether an input can be borrowed as a real floating-point matrix.
package(mir.stat) enum matrix(A) = !isStructuralView!A && is(typeof(A.init.asView)) && View!A.N == 2 && isFloatingPoint!(Element!A);
/// Whether observation, center, and metric have supported ranks and matching element types.
package(mir.stat) enum compatible(X, C, M) = vector!X && vector!C && (matrix!M || DiagonalInfo!M.valid) &&
    is(Element!X == Element!C) && is(Element!X == Element!M);

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
