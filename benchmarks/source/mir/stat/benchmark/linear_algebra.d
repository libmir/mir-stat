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
enum algebraNames = ["crossprod", "tcrossprod", "triangularLeft", "triangularRight", "quadratic", "mahalanobis", "quadraticBatch", "mahalanobisBatch", "symmetricVector", "symmetricLeft", "symmetricRight", "symmetricColumn", "symmetricRow"];

/// List the algorithms available for a validated benchmark family.
string[] algebraAlgorithms(string name)
{
    switch (name)
    {
    case "crossprod", "tcrossprod": return ["general", "specialized"];
    case "symmetricVector": return ["general", "symmetric", "packed"];
    case "symmetricLeft", "symmetricRight", "symmetricColumn", "symmetricRow": return ["general", "symmetric", "packed", "generalMixed", "symmetricMixed", "packedMixed", "workspaceInput", "workspaceOutput"];
    case "quadraticBatch": return ["repeated", "batch", "repeatedFast", "batchFast"];
    case "mahalanobisBatch": return ["repeatedFactor", "batchFactor", "repeatedPrecision", "batchPrecision",
        "repeatedPrecisionFast", "batchPrecisionFast", "repeatedPrecisionFactor", "batchPrecisionFactor"];
    case "triangularLeft", "triangularRight": return ["dense", "triangular", "packed"];
    case "quadratic": return ["composed", "general", "symmetric", "singleTriangle", "generalFast", "symmetricFast", "packedSymmetric", "packedFast"];
    case "mahalanobis": return ["covarianceGC", "factorGC", "precisionGC", "precisionFactorGC",
        "covarianceWorkspace", "factorWorkspace", "precisionWorkspace", "precisionFactorWorkspace", "precisionFastGC", "precisionFastWorkspace"];
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
    case "crossprod": return measureCrossProduct!true(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "tcrossprod": return measureCrossProduct!false(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "symmetricColumn": return measureSymmetric!(false, false, true)(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "symmetricRow": return measureSymmetric!(false, true, true)(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "symmetricVector": return measureSymmetric!(true, false)(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "symmetricLeft": return measureSymmetric!(false, false)(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "symmetricRight": return measureSymmetric!(false, true)(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "quadraticBatch": return measureBatch!false(name,iterations,size,seed,selected,rounds,warmup,transpose);
    case "mahalanobisBatch": return measureBatch!true(name,iterations,size,seed,selected,rounds,warmup,transpose);
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
    const needPrecision = family == 3 && (needPrecisionFactor || needs("precisionGC") || needs("precisionWorkspace") || needs("precisionFastGC") || needs("precisionFastWorkspace"));
    const needA = family < 2 ? needs("triangular") : family == 2 ?
        (needs("composed") || needs("general") || needs("symmetric") || needs("singleTriangle") ||
         needs("generalFast") || needs("symmetricFast")) :
        (needFactor || needs("covarianceGC") || needs("covarianceWorkspace"));
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
    auto work = uninitSlice!double((family == 2 && (needs("composed") || needs("generalFast") || needs("symmetricFast") || needs("packedFast"))) || (family == 3 &&
        (needs("covarianceWorkspace") || needs("factorWorkspace") ||
         needs("precisionWorkspace") || needs("precisionFactorWorkspace") || needs("precisionFastWorkspace"))) ? n : 0);
    import mir.ndslice.topology: stairs;
    auto packedData = uninitSlice!double((family < 2 && needs("packed")) ||
        (family == 2 && (needs("packedSymmetric") || needs("packedFast"))) ? n*(n+1)/2 : 0);
    auto packed = packedData.stairs!"-"(packedData.length ? n : 0);
    auto productWork = uninitSlice!double(family == 3 && needs("precisionFastWorkspace") ? n : 0);
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
            if (needA) a[i,j] = j >= i ? value : double.nan;
        }
        else if (needA) a[i,j] = value;
        if (packedData.length && j >= i) packed[i][j-i] = value;
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
    /// Multiply directly from packed stairs without expanding the triangle.
    double packedProduct()
    {
        static if (family == 0) mtimesInto(packed,b,output);
        else mtimesInto(b,packed,output);
        return output[0,0];
    }
    /// Evaluate compensated rows from packed symmetric storage.
    double packedSymmetric() { return quadraticForm(x,symmetricView(packed)); }
    /// Evaluate packed symmetric BLAS or its native D fallback.
    double packedFast() { import mir.math.sum: Summation; return quadraticFormWithWorkspace!(Summation.fast)(x,symmetricView(packed),work); }
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
    /// Include two GC vector allocations in the fast precision calculation.
    double precisionFastGC()
    {
        import mir.math.sum: Summation;
        return mahalanobisSquared!(MahalanobisMatrix.inverseCovariance, TrianglePolicy.upper, Summation.fast)(x,center,precision);
    }
    /// Reuse both centered-input and matrix-vector buffers for fast precision arithmetic.
    double precisionFastWorkspace()
    {
        import mir.math.sum: Summation;
        return mahalanobisSquaredWithWorkspace!(MahalanobisMatrix.inverseCovariance, TrianglePolicy.upper, Summation.fast)(x,center,precision,work,productWork);
    }
    import std.meta: AliasSeq;
    static if (family < 2) alias functions = AliasSeq!(denseProduct, triangularProduct, packedProduct);
    else static if (family == 2) alias functions = AliasSeq!(composed, general, symmetric, single, generalFast, symmetricFast, packedSymmetric, packedFast);
    else alias functions = AliasSeq!(covarianceGC, factorGC, precisionGC, precisionFactorGC,
        covarianceWorkspace, factorWorkspace, precisionWorkspace, precisionFactorWorkspace, precisionFastGC, precisionFastWorkspace);
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

// Packed-only quadratic runs must not retain a dense copy of their metric.
unittest
{
    import core.memory: GC;
    enum order = 128;
    auto before = GC.stats().allocatedInCurrentThread;
    auto rows = runAlgebra("quadratic",1,order,5489,["packedFast"],1,0);
    auto bytes = GC.stats().allocatedInCurrentThread - before;
    assert(rows.length == 1);
    // Half a matrix plus vectors/scheduler fits comfortably below a full matrix.
    assert(bytes < order * order * double.sizeof);
}

/// Compare 32 observation columns per invocation, using identical reusable buffers and policies.
private Result[] measureBatch(bool distance)(string name, size_t iterations, size_t n, ulong seed,
    string[] selected, size_t rounds, size_t warmup, bool transpose)
{
    import mir.ndslice.allocation: uninitSlice;
    import mir.math.sum: Summation;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;
    import std.meta: AliasSeq;
    enum columns = 32;
    auto labels = algebraAlgorithms(name);
    /// Whether a buffer's consumers are selected.
    bool needs(string label) { return !selected.length || selected.canFind(label); }
    /// Allocate an optional metric in the requested layout.
    auto matrix(bool needed) {
        auto a = uninitSlice!double(needed ? n : 0, needed ? n : 0).universal;
        return transpose ? a.transposed : a;
    }
    /// Allocate a batch in row-major or transposed storage without changing logical shape.
    auto batchMatrix(bool needed = true) {
        auto a = uninitSlice!double(needed ? (transpose ? columns : n) : 0, needed ? (transpose ? n : columns) : 0).universal;
        return transpose ? a.transposed : a;
    }
    auto metric = matrix(!distance || needs("repeatedPrecision") || needs("batchPrecision") ||
        needs("repeatedPrecisionFast") || needs("batchPrecisionFast") || needs("repeatedPrecisionFactor") || needs("batchPrecisionFactor"));
    auto factor = matrix(distance && (needs("repeatedFactor") || needs("batchFactor")));
    auto precisionFactor = matrix(distance && (needs("repeatedPrecisionFactor") || needs("batchPrecisionFactor")));
    auto x = batchMatrix(); auto work = batchMatrix();
    auto product = batchMatrix(distance && (needs("repeatedPrecisionFast") || needs("batchPrecisionFast")));
    auto result = uninitSlice!double(columns);
    auto reference = uninitSlice!double(columns);
    auto u = uninitSlice!double(n); auto center = uninitSlice!double(n); center[] = 0;
    auto rng = Mt19937_64(seed); auto normal = NormalVariable!double(0, 1);
    real norm = 0;
    foreach (i;0 .. n) { u[i] = normal(rng)/n; norm+=cast(real)u[i]*u[i]; }
    foreach (i;0 .. n) foreach (j;0 .. n) {
        auto covariance = (i == j ? 1.0 : 0.0)+u[i]*u[j];
        if (metric.length) metric[i, j] = distance ? (i == j ? 1.0 : 0.0)-u[i]*u[j]/cast(double)(1+norm) : covariance;
        if (factor.length) factor[i, j] = covariance;
    }
    if (factor.length) choleskyDecomposeInPlace(factor);
    if (precisionFactor.length) choleskyDecomposeInto(metric, precisionFactor);
    /// Run the same arithmetic policy as either one batch or repeated column calls.
    double run(bool batch, MahalanobisMatrix form = MahalanobisMatrix.inverseCovariance, Summation summation = Summation.kbn)()
    {
        auto xc = x.transposed; auto wc = work.transposed;
        static if (!distance) {
            static if (batch) quadraticFormBatchInto!summation(x, symmetricView(metric), result, work);
            else foreach (j;0 .. columns) result[j] = quadraticFormWithWorkspace!summation(xc[j], symmetricView(metric), wc[j]);
        } else {
            static if (form == MahalanobisMatrix.cholesky) auto m = factor;
            else static if (form == MahalanobisMatrix.inverseCovarianceCholesky) auto m = precisionFactor;
            else auto m = metric;
            static if (batch) {
                static if (form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast)
                    mahalanobisSquaredBatchWithWorkspace!(form, TrianglePolicy.upper, summation)(x, center, m, result, work, product);
                else mahalanobisSquaredBatchWithWorkspace!(form, TrianglePolicy.upper, summation)(x, center, m, result, work);
            } else {
                auto pc = product.transposed;
                foreach (j;0 .. columns) {
                    static if (form == MahalanobisMatrix.inverseCovariance && summation == Summation.fast)
                        result[j] = mahalanobisSquaredWithWorkspace!(form, TrianglePolicy.upper, summation)(xc[j], center, m, wc[j], pc[j]);
                    else result[j] = mahalanobisSquaredWithWorkspace!(form)(xc[j], center, m, wc[j]);
                }
            }
        }
        return result[0];
    }
    static if (!distance) alias functions = AliasSeq!(run!false, run!true,
        run!(false, MahalanobisMatrix.inverseCovariance, Summation.fast), run!(true, MahalanobisMatrix.inverseCovariance, Summation.fast));
    else alias functions = AliasSeq!(run!(false, MahalanobisMatrix.cholesky), run!(true, MahalanobisMatrix.cholesky),
        run!false, run!true, run!(false, MahalanobisMatrix.inverseCovariance, Summation.fast), run!(true, MahalanobisMatrix.inverseCovariance, Summation.fast),
        run!(false, MahalanobisMatrix.inverseCovarianceCholesky), run!(true, MahalanobisMatrix.inverseCovarianceCholesky));
    bool[functions.length] enabled;
    foreach (i, label;labels) enabled[i] = needs(label);
    auto active = enabled[].count(true);
    /// Refresh inputs and the analytic rank-one reference outside timing.
    void prepare() {
        foreach (j;0 .. columns) {
            real xx = 0, ux = 0;
            foreach (i;0 .. n) { x[i, j] = normal(rng); xx+=cast(real)x[i, j]*x[i, j]; ux+=cast(real)u[i]*x[i, j]; }
            reference[j] = cast(double)(distance ? xx-ux*ux/(1+norm) : xx+ux*ux);
        }
    }
    /// Validate every observation result against an independent analytic reference.
    void check(size_t index, double value) {
        foreach (j;0 .. columns) enforce(isFinite(result[j]) && abs(result[j]-reference[j])<=1e-10*(1+abs(reference[j])),"Incorrect batch benchmark result");
    }
    Result[] results;
    foreach (round;0 .. rounds) {
        double[functions.length] values;
        auto first = round%active;
        auto elapsed = benchmarkPrepared!functions(iterations, values, enabled, BenchmarkSchedule(round == 0 ? warmup : 0, first), &prepare, &check);
        size_t ordinal;
        foreach (i, label;labels) if (enabled[i]) {
            results~=Result(name~(transpose ? "_transposed" : "_rowMajor"), label, values[i], elapsed[i], Transform.none, round+1, (ordinal+active-first)%active+1);
            ++ordinal;
        }
    }
    return results;
}

/// Compare general, symmetric and packed products, including opposite output layout.
/// Matrix cases have 32 right-hand sides (or 32 rows for a right-side metric).
private Result[] measureSymmetric(bool vector, bool right, bool singleColumn = false)(string name, size_t iterations,
    size_t n, ulong seed, string[] selected, size_t rounds, size_t warmup, bool transpose)
{
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.topology: stairs;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;
    import std.meta: AliasSeq;
    enum columns = vector || singleColumn ? 1 : 32;
    auto labels = algebraAlgorithms(name);
    /// Whether an algorithm is included in this run.
    bool needs(string label) { return !selected.length || selected.canFind(label); }
    bool needDense = needs("general") || needs("symmetric");
    bool needPacked = needs("packed");
    bool needOutput = needDense || needPacked;
    bool needMixed;
    static if (!vector)
    {
        needDense = needDense || needs("generalMixed") || needs("symmetricMixed") || needs("workspaceInput") || needs("workspaceOutput");
        needPacked = needPacked || needs("packedMixed");
        needMixed = needs("generalMixed") || needs("symmetricMixed") || needs("packedMixed") || needs("workspaceInput") || needs("workspaceOutput");
    }
    auto a = uninitSlice!double(needDense ? n : 0, needDense ? n : 0).universal;
    if (transpose) a = a.transposed;
    auto packed = uninitSlice!double(needPacked ? n*(n+1)/2 : 0);
    auto pv = symmetricView(packed.stairs!"-"(needPacked ? n : 0));
    auto u = uninitSlice!double(n);
    auto rng = Mt19937_64(seed);
    auto normal = NormalVariable!double(0, 1);
    foreach (i; 0 .. n) u[i] = normal(rng)/n;
    size_t offset;
    foreach (i; 0 .. n) foreach (j; 0 .. n)
    {
        auto value = (i == j ? 1.0 : 0.0) + u[i]*u[j];
        if (needDense) a[i,j] = value;
        if (needPacked && j >= i) packed[offset++] = value;
    }
    static if (vector)
    {
        auto b = uninitSlice!double(n);
        auto c = uninitSlice!double(n);
        auto reference = uninitSlice!double(n);
    }
    else
    {
        /// Allocate only selected output layouts, preserving logical product shape.
        auto matrix(bool needed, bool colMajor)
        {
            auto value = uninitSlice!double(needed ? (colMajor ? columns : n) : 0,
                needed ? (colMajor ? n : columns) : 0).universal;
            return colMajor ? value.transposed : value;
        }
        // Right products transpose these working descriptors before the call.
        enum flipForRight = right;
        auto b = matrix(true, transpose != flipForRight);
        auto c = matrix(needOutput, transpose != flipForRight);
        auto mixed = matrix(needMixed, transpose == flipForRight);
        auto workInput = matrix(needs("workspaceInput"), transpose == flipForRight);
        auto workOutput = matrix(needs("workspaceOutput"), transpose != flipForRight);
        auto reference = matrix(true, false);
    }
    /// Dispatch the selected representation into a shared preallocated output.
    double run(int representation, bool mixedLayout = false)()
    {
        static if (mixedLayout) auto output = mixed;
        else auto output = c;
        static if (representation == 0) auto metric = a;
        else static if (representation == 1) auto metric = symmetricView(a);
        else auto metric = pv;
        static if (right) mtimesInto(b.transposed, metric, output.transposed);
        else mtimesInto(metric, b, output);
        static if (vector) return output[0];
        else return output[0,0];
    }
    /// Include layout conversion in timing while reusing caller workspace.
    double runWorkspace(bool copyInput)()
    {
        static if (copyInput) auto work = workInput;
        else auto work = workOutput;
        static if (right) mtimesInto(b.transposed, symmetricView(a), mixed.transposed, work.transposed);
        else mtimesInto(symmetricView(a), b, mixed, work);
        return mixed[0,0];
    }
    static if (vector) alias functions = AliasSeq!(run!0, run!1, run!2);
    else alias functions = AliasSeq!(run!0, run!1, run!2, run!(0,true), run!(1,true), run!(2,true),
        runWorkspace!true, runWorkspace!false);
    bool[functions.length] enabled;
    foreach (i, label; labels) enabled[i] = needs(label);
    auto active = enabled[].count(true);
    /// Generate shared inputs and an independent rank-one analytic reference outside timing.
    void prepare()
    {
        foreach (j; 0 .. columns)
        {
            real dot = 0;
            foreach (i; 0 .. n)
            {
                auto value = normal(rng);
                static if (vector) b[i] = value;
                else b[i,j] = value;
                dot += cast(real)u[i]*value;
            }
            foreach (i; 0 .. n)
            {
                static if (vector) reference[i] = cast(double)(b[i] + u[i]*dot);
                else reference[i,j] = cast(double)(b[i,j] + u[i]*dot);
            }
        }
    }
    /// Check every output entry after each timed call, including mixed layouts.
    void check(size_t index, double value)
    {
        static if (vector) auto output = c;
        else auto output = index < 3 ? c : mixed;
        foreach (j; 0 .. columns) foreach (i; 0 .. n)
        {
            static if (vector) { auto actual = output[i]; auto expected = reference[i]; }
            else { auto actual = output[i,j]; auto expected = reference[i,j]; }
            enforce(isFinite(actual) && abs(actual-expected) <= 1e-10*(1+abs(expected)),
                "Incorrect symmetric product benchmark result");
        }
    }
    Result[] results;
    foreach (round; 0 .. rounds)
    {
        double[functions.length] values;
        auto first = round % active;
        auto elapsed = benchmarkPrepared!functions(iterations, values, enabled,
            BenchmarkSchedule(round == 0 ? warmup : 0, first), &prepare, &check);
        size_t ordinal;
        foreach (i, label; labels) if (enabled[i])
        {
            results ~= Result(name ~ (transpose ? "_transposed" : "_rowMajor"), label, values[i], elapsed[i],
                Transform.none, round+1, (ordinal+active-first)%active+1);
            ++ordinal;
        }
    }
    return results;
}

/// Compare one-triangle self-products with general multiplication using identical reusable output.
private Result[] measureCrossProduct(bool cross)(string name, size_t iterations,
    size_t n, ulong seed, string[] selected, size_t rounds, size_t warmup, bool transpose)
{
    import mir.ndslice.allocation: uninitSlice;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;
    enum contraction = 32;
    enum labels = ["general", "specialized"];
    const inputRows = cross ? contraction : n;
    const inputColumns = cross ? n : contraction;
    auto a = uninitSlice!double(transpose ? inputColumns : inputRows,
        transpose ? inputRows : inputColumns).universal;
    if (transpose) a = a.transposed;
    static if (cross) auto rows = a.transposed;
    else auto rows = a;
    auto output = uninitSlice!double(n,n).universal;
    if (transpose) output = output.transposed;
    auto reference = uninitSlice!double(n,n);
    auto rng = Mt19937_64(seed);
    auto normal = NormalVariable!double(0,1);
    /// Refresh logical row vectors and independently accumulate their products in extended precision.
    void prepare() {
        foreach (i;0..n) foreach (k;0..contraction) rows[i,k]=normal(rng);
        foreach (i;0..n) foreach (j;0..n) {
            real sum = 0;
            foreach (k;0..contraction) sum+=cast(real)rows[i,k]*rows[j,k];
            reference[i,j]=cast(double)sum;
        }
    }
    /// Ordinary matrix multiplication provides the performance baseline.
    double general() { mtimesInto(rows,rows.transposed,output); return output[0,0]; }
    /// Include triangle mirroring inside the specialized operation's timed interval.
    double specialized() {
        static if(cross) crossprodInto(a,output);
        else tcrossprodInto(a,output);
        return output[0,0];
    }
    /// Validate every output entry outside the timed region.
    void check(size_t index, double value) {
        foreach(i;0..n) foreach(j;0..n)
            enforce(isFinite(output[i,j]) && abs(output[i,j]-reference[i,j])<=1e-10*(1+abs(reference[i,j])),
                "Incorrect cross-product benchmark result");
    }
    bool[2] enabled;
    foreach(i,label;labels) enabled[i]=!selected.length || selected.canFind(label);
    auto active=enabled[].count(true);
    Result[] results;
    foreach(round;0..rounds) {
        double[2] values;
        auto first=round%active;
        auto elapsed=benchmarkPrepared!(general,specialized)(iterations,values,enabled,
            BenchmarkSchedule(round==0 ? warmup : 0,first),&prepare,&check);
        size_t ordinal;
        foreach(i,label;labels) if(enabled[i]) {
            results~=Result(name~(transpose ? "_transposed" : "_rowMajor"),label,values[i],elapsed[i],
                Transform.none,round+1,(ordinal+active-first)%active+1);
            ++ordinal;
        }
    }
    return results;
}

// Check both cross-product orientations and benchmark scheduling.
unittest
{
    foreach(name; ["crossprod", "tcrossprod"])
    foreach(transpose; [false,true])
        assert(runAlgebra(name,1,3,5489,[],2,1,transpose).length==4);
}
