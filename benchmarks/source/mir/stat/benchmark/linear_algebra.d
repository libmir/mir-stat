/++
Reproducible linear algebra comparisons using the shared benchmark scheduler.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.benchmark.linear_algebra;

import mir.stat.benchmark.cases: Result, Transform;
import mir.math.internal.benchmark: benchmarkPrepared, BenchmarkSchedule;
import mir.stat.internal.linearAlgebra;
import mir.stat.distance;
import mir.ndslice.topology: universal;
import mir.ndslice.dynamic: transposed;
import std.algorithm: canFind, count;
import std.exception: enforce;
import std.math: abs, isFinite;

package(mir.stat):
enum algebraNames = ["triangularLeft", "triangularRight", "quadratic", "mahalanobis"];

/// List the algorithms available for a validated benchmark family.
string[] algebraAlgorithms(string name)
{
    switch (name)
    {
    case "triangularLeft", "triangularRight": return ["dense", "triangular"];
    case "quadratic": return ["composed", "general", "symmetric", "singleTriangle", "generalFast", "symmetricFast"];
    case "mahalanobis": return ["covarianceGC", "factorGC", "precisionGC", "precisionFactorGC",
        "covarianceWorkspace", "factorWorkspace", "precisionWorkspace", "precisionFactorWorkspace"];
    default: throw new Exception("Unknown linear algebra benchmark: " ~ name);
    }
}

/// Reject unsupported sizes, transformations, or algorithms before allocating benchmark inputs.
void validateAlgebra(string name, size_t iterations, size_t size, Transform transform,
    string[] selected, size_t rounds)
{
    enforce(iterations > 0 && rounds > 0, "Iterations and rounds must be positive");
    enforce(size > 0 && size <= 4096, "Matrix order must be between 1 and 4096");
    enforce(transform == Transform.none, "Linear algebra benchmarks require --transform=none");
    auto choices = algebraAlgorithms(name);
    foreach (choice; selected)
        enforce(choices.canFind(choice), "Unknown algorithm for " ~ name ~ ": " ~ choice);
}

// Experimental flattened reduction: deliberately kept out of the library until
// its cancellation/overflow tradeoffs have been evaluated against row reductions.
/// Experimental single-triangle reduction; may overflow where row-wise cancellation remains finite.
private double singleTriangle(V, M)(V x, M a)
{
    import mir.math.sum: Summator, Summation;
    Summator!(double, Summation.kbn) total;
    foreach (i; 0 .. x.length)
    {
        total.put(x[i] * (a[i,i] * x[i]));
        foreach (j; i + 1 .. x.length)
        {
            auto term = x[i] * (a[i,j] * x[j]);
            total.put(term);
            total.put(term);
        }
    }
    return total.sum;
}

/// Run a benchmark family with shared seeded inputs and return checked per-round results.
Result[] runAlgebra(string name, size_t iterations, size_t size, ulong seed,
    string[] selected, size_t rounds, size_t warmup, bool transpose = false)
{
    validateAlgebra(name, iterations, size, Transform.none, selected, rounds);
    switch (name)
    {
    case "triangularLeft": return measure!0(name, iterations, size, seed, selected, rounds, warmup, transpose);
    case "triangularRight": return measure!1(name, iterations, size, seed, selected, rounds, warmup, transpose);
    case "quadratic": return measure!2(name, iterations, size, seed, selected, rounds, warmup, transpose);
    case "mahalanobis": return measure!3(name, iterations, size, seed, selected, rounds, warmup, transpose);
    default: assert(0);
    }
}

/// Prepare only required buffers, rotate enabled algorithms, and validate results outside timing.
private Result[] measure(int family)(string name, size_t iterations, size_t n, ulong seed,
    string[] selected, size_t rounds, size_t warmup, bool transpose)
{
    import mir.ndslice.allocation: uninitSlice;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;
    auto rng = Mt19937_64(seed);
    auto normal = NormalVariable!double(0, 1);
    auto labels = algebraAlgorithms(name);
    /// Whether the request includes a named algorithm.
    bool needs(string label) { return !selected.length || selected.canFind(label); }
    const needFactor = family == 3 && (needs("factorGC") || needs("factorWorkspace"));
    const needPrecisionFactor = family == 3 && (needs("precisionFactorGC") || needs("precisionFactorWorkspace"));
    const needPrecision = family == 3 && (needPrecisionFactor || needs("precisionGC") || needs("precisionWorkspace"));
    const needA = family != 3 || needFactor || needs("covarianceGC") || needs("covarianceWorkspace");
    /// Allocate only matrices required by the selected family and algorithms.
    auto matrix(bool needed)
    {
        auto result = uninitSlice!double(needed ? n : 0, needed ? n : 0).universal;
        return transpose ? result.transposed : result;
    }
    auto a = matrix(needA);
    auto dense = matrix(family < 2);
    auto b = matrix(family < 2);
    auto output = matrix(family < 2);
    auto precision = matrix(needPrecision);
    auto factor = matrix(needFactor);
    auto precisionFactor = matrix(needPrecisionFactor);
    auto factorWork = matrix(family == 3 && needs("covarianceWorkspace"));
    auto referenceOutput = matrix(family < 2);
    auto x = uninitSlice!double(family >= 2 ? n : 0);
    auto center = uninitSlice!double(family == 3 ? n : 0);
    auto work = uninitSlice!double((family == 2 && (needs("composed") || needs("generalFast") || needs("symmetricFast"))) || (family == 3 &&
        (needs("covarianceWorkspace") || needs("factorWorkspace") ||
         needs("precisionWorkspace") || needs("precisionFactorWorkspace"))) ? n : 0);
    auto u = uninitSlice!double(n);
    // Fixed well-conditioned SPD metric S=I+u*u', with analytic inverse
    // P=I-u*u'/(1+u'u). This avoids using the measured routines as references.
    double norm = 0;
    foreach (i; 0 .. n)
    {
        u[i] = normal(rng) / n;
        norm += u[i]*u[i];
        static if (family == 3) center[i] = 0;
    }
    foreach (i; 0 .. n) foreach (j; 0 .. n)
    {
        auto value = (i == j ? 1.0 : 0.0) + u[i]*u[j];
        static if (family < 2)
        {
            dense[i,j] = j >= i ? value : 0;
            a[i,j] = j >= i ? value : double.nan;
        }
        else if (needA) a[i,j] = value;
        if (needPrecision) precision[i,j] = (i == j ? 1.0 : 0.0) - u[i]*u[j]/(1+norm);
        // Preserve the existing seeded input stream without keeping an unused B.
        auto sample = normal(rng);
        static if (family < 2) b[i,j] = sample;
    }
    static if (family == 3)
    {
        if (needFactor) choleskyDecomposeInto(a, factor);
        if (needPrecisionFactor) choleskyDecomposeInto(precision, precisionFactor);
    }
    // All callbacks read identical inputs and overwrite independent scratch.
    /// Time a dense product into reusable output; all entries are checked after timing.
    double denseProduct()
    {
        static if (family == 0) mtimesInto(dense,b,output);
        else mtimesInto(b,dense,output);
        return output[0,0];
    }
    /// Time the structured product into reusable output without reading ignored entries.
    double triangularProduct()
    {
        static if (family == 0) mtimesTriangularInto(a,b,output);
        else mtimesTriangularRightInto(b,a,output);
        return output[0,0];
    }
    /// Evaluate a quadratic form through a matrix-vector product and dot product.
    double composed() { mtimesInto(a,x,work); return mtimes(x,work); }
    /// Evaluate the general compensated quadratic form.
    double general() { return quadraticForm(x,a); }
    /// Evaluate the compensated quadratic form through a symmetric view.
    double symmetric() { return quadraticForm(x,symmetricView(a)); }
    /// Evaluate a workspace quadratic form with explicit fast arithmetic.
    double generalFast() { import mir.math.sum: Summation; return quadraticFormWithWorkspace!(Summation.fast)(x,a,work); }
    /// Evaluate a symmetric workspace quadratic form with explicit fast arithmetic.
    double symmetricFast() { import mir.math.sum: Summation; return quadraticFormWithWorkspace!(Summation.fast)(x,symmetricView(a),work); }
    /// Measure the experimental flattened reduction on the shared inputs.
    double single() { return singleTriangle(x,a); }
    /// Measure squared distance including GC scratch allocation and covariance factorization.
    double covarianceGC() { return mahalanobisSquared(x,center,a); }
    /// Measure squared distance from a covariance factor with GC scratch.
    double factorGC() { return mahalanobisSquared!(MahalanobisMatrix.cholesky)(x,center,factor); }
    /// Measure squared distance from a precision matrix with GC scratch.
    double precisionGC() { return mahalanobisSquared!(MahalanobisMatrix.inverseCovariance)(x,center,precision); }
    /// Measure squared distance from a precision factor with GC scratch.
    double precisionFactorGC() { return mahalanobisSquared!(MahalanobisMatrix.inverseCovarianceCholesky)(x,center,precisionFactor); }
    /// Measure covariance factorization and squared distance with reusable buffers.
    double covarianceWorkspace() { return mahalanobisSquaredWithWorkspace(x,center,a,work,factorWork); }
    /// Measure squared distance using a covariance factor and reusable workspace.
    double factorWorkspace() { return mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.cholesky)(x,center,factor,work); }
    /// Measure squared distance using a precision matrix and reusable workspace.
    double precisionWorkspace() { return mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovariance)(x,center,precision,work); }
    /// Measure squared distance using a precision factor and reusable workspace.
    double precisionFactorWorkspace() { return mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovarianceCholesky)(x,center,precisionFactor,work); }
    import std.meta: AliasSeq;
    static if (family < 2) alias functions = AliasSeq!(denseProduct, triangularProduct);
    else static if (family == 2) alias functions = AliasSeq!(composed, general, symmetric, single, generalFast, symmetricFast);
    else alias functions = AliasSeq!(covarianceGC, factorGC, precisionGC, precisionFactorGC,
        covarianceWorkspace, factorWorkspace, precisionWorkspace, precisionFactorWorkspace);
    bool[functions.length] enabled;
    foreach (i, label; labels) enabled[i] = !selected.length || selected.canFind(label);
    auto active = enabled[].count(true);
    double reference;
    /// Generate shared inputs and independent reference results outside the timed operation.
    void prepare()
    {
        static if (family < 2)
        {
            foreach (i; 0 .. n) foreach (j; 0 .. n) b[i,j] = normal(rng);
            foreach (i; 0 .. n) foreach (j; 0 .. n)
            {
                real value = 0;
                foreach (k; 0 .. n)
                    static if (family == 0) value += cast(real)dense[i,k]*b[k,j];
                    else value += cast(real)b[i,k]*dense[k,j];
                referenceOutput[i,j] = cast(double)value;
            }
            reference = referenceOutput[0,0];
        }
        else
        {
            real xx = 0, ux = 0;
            foreach (i; 0 .. n) { x[i] = normal(rng); xx += cast(real)x[i]*x[i]; ux += cast(real)u[i]*x[i]; }
            static if (family == 2) reference = cast(double)(xx + ux*ux);
            else reference = cast(double)(xx - ux*ux/(1+norm));
        }
    }
    /// Reject nonfinite results and discrepancies beyond the benchmark tolerance.
    void verify(double value, double expected)
    {
        import std.format: format;
        enforce(isFinite(value) && abs(value-expected) <= 1e-10*(1+abs(expected)),
            format("%s: result %.17g differs from reference %.17g", name, value, expected));
    }
    /// Check the scalar result and every matrix output entry outside timing.
    void check(size_t index, double value)
    {
        verify(value,reference);
        static if (family < 2)
            foreach (i; 0 .. n) foreach (j; 0 .. n) verify(output[i,j],referenceOutput[i,j]);
    }
    Result[] results;
    foreach (round; 0 .. rounds)
    {
        double[functions.length] values;
        auto first = round % active;
        auto elapsed = benchmarkPrepared!functions(iterations,values,enabled,
            BenchmarkSchedule(round == 0 ? warmup : 0,first),&prepare,&check);
        size_t ordinal;
        foreach (i, label; labels) if (enabled[i])
        {
            results ~= Result(name ~ (transpose ? "_transposed" : "_rowMajor"),label,values[i],elapsed[i],
                Transform.none,round+1,(ordinal+active-first)%active+1);
            ++ordinal;
        }
    }
    return results;
}

unittest
{
    import std.exception: assertThrown;
    foreach (name; algebraNames) foreach (transpose; [false,true])
    {
        auto rows = runAlgebra(name,2,3,5489,null,2,1,transpose);
        assert(rows.length == 2*algebraAlgorithms(name).length);
        foreach (row; rows) assert(isFinite(row.value));
    }
    foreach (name; algebraNames)
        foreach (algorithm; algebraAlgorithms(name))
            assert(runAlgebra(name,1,3,5489,[algorithm],1,0).length == 1);
    assertThrown!Exception(validateAlgebra("quadratic",1,0,Transform.none,null,1));
    assertThrown!Exception(validateAlgebra("quadratic",1,2,Transform.center,null,1));
    assertThrown!Exception(validateAlgebra("quadratic",1,2,Transform.none,["unknown"],1));
    auto rows = runAlgebra("quadratic",1,1,5489,["singleTriangle","general"],2,0);
    assert(rows.length == 4 && rows[0].position == 1 && rows[2].position == 2);
}

// Flattening a quadratic form can overflow individual contributions even when
// row-wise cancellation keeps the existing API's answer finite.
unittest
{
    import mir.ndslice.slice: sliced;
    double[2] values = [2,2];
    double m = double.max/3;
    double[4] matrix = [m,-m,-m,m];
    auto x = values[].sliced;
    auto a = matrix[].sliced(2,2);
    assert(quadraticForm(x,symmetricView(a)) == 0);
    assert(!isFinite(singleTriangle(x,a)));
}

// A selected quadratic algorithm needs one matrix, not every family's buffers.
unittest
{
    import core.memory: GC;
    enum order = 128;
    auto before = GC.stats().allocatedInCurrentThread;
    auto rows = runAlgebra("quadratic",1,order,5489,["general"],1,0);
    auto bytes = GC.stats().allocatedInCurrentThread - before;
    assert(rows.length == 1);
    // Allow ample vector, scheduler, and runtime overhead without accepting
    // the former nine-matrix setup. This is an allocation budget, not a timer.
    assert(bytes < 3 * order * order * double.sizeof);
}
