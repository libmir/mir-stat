/++
Multivariate normal density benchmarks with a shared prepared covariance factor.
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.benchmark.multivariate_normal;

package(mir.stat) auto measureDensity(size_t iterations, size_t n, size_t columns, ulong seed,
    string[] selected, size_t rounds, size_t warmup, bool transpose)
{
    import mir.math.internal.benchmark: benchmarkPrepared, BenchmarkSchedule;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.topology: universal;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;
    import mir.stat.benchmark.cases: Result, Transform;
    import mir.stat.benchmark.linear_algebra: algebraAlgorithms;
    import mir.stat.distribution.multivariate_normal;
    import mir.stat.covariance_views: covarianceCholeskyView;
    import std.algorithm: canFind, count;
    import std.conv: to;
    import std.exception: enforce;
    import std.math: abs, isFinite, log, PI;
    import std.meta: AliasSeq;

    enforce(columns > 0 && columns <= 4096, "Observation count must be between 1 and 4096");
    auto matrix(size_t rows, size_t cols)
    {
        auto m = uninitSlice!double(transpose ? cols : rows, transpose ? rows : cols).universal;
        return transpose ? m.transposed : m;
    }
    auto factor = matrix(n, n);
    auto x = matrix(n, columns);
    auto work = matrix(n, columns);
    auto result = uninitSlice!double(columns);
    auto center = uninitSlice!double(n);
    auto logReference = uninitSlice!double(columns);
    auto pdfReference = uninitSlice!double(columns);
    // Upper bidiagonal U, with covariance U' U. Generate x = mean + U' z.
    factor[] = 0;
    real logScale = -0.5L * n * log(2 * PI);
    foreach (i; 0 .. n)
    {
        center[i] = 0.25;
        factor[i, i] = 1 + 0.125 * (i % 3);
        logScale -= log(cast(real) factor[i, i]);
        if (i + 1 < n) factor[i, i + 1] = 0.25;
    }
    auto metric = covarianceCholeskyView(factor);
    auto rng = Mt19937_64(seed);
    auto normal = NormalVariable!double(0, 1);
    void prepare()
    {
        import std.math: referenceExp = exp;
        foreach (j; 0 .. columns)
        {
            double previous = 0;
            real squares = 0;
            foreach (i; 0 .. n)
            {
                auto z = normal(rng);
                x[i, j] = center[i] + factor[i, i] * z + 0.25 * previous;
                squares += cast(real) z * z;
                previous = z;
            }
            auto expected = logScale - 0.5L * squares;
            logReference[j] = cast(double) expected;
            pdfReference[j] = cast(double) referenceExp(expected);
        }
    }
    double scalar(bool logarithmic)()
    {
        foreach (j; 0 .. columns)
        {
            static if (logarithmic)
                result[j] = multivariateNormalLPDFWithWorkspace(x.transposed[j], center, metric, work.transposed[j]);
            else
                result[j] = multivariateNormalPDFWithWorkspace(x.transposed[j], center, metric, work.transposed[j]);
        }
        return result[0];
    }
    double batch(bool logarithmic)()
    {
        static if (logarithmic)
            multivariateNormalLPDFBatchWithWorkspace(x, center, metric, result, work);
        else
            multivariateNormalPDFBatchWithWorkspace(x, center, metric, result, work);
        return result[0];
    }
    alias functions = AliasSeq!(scalar!true, batch!true, scalar!false, batch!false);
    auto labels = algebraAlgorithms("multivariateNormal");
    bool[functions.length] enabled;
    foreach (i, label; labels) enabled[i] = !selected.length || selected.canFind(label);
    const active = enabled[].count(true);
    void check(size_t index, double firstValue)
    {
        const logarithmic = index < 2;
        foreach (j; 0 .. columns)
        {
            const expected = logarithmic ? logReference[j] : pdfReference[j];
            const tolerance = logarithmic ? 1e-11 * (1 + abs(expected)) :
                1e-11 * expected + double.min_normal * double.epsilon;
            enforce(isFinite(result[j]) && abs(result[j] - expected) <= tolerance,
                "Incorrect multivariate normal benchmark result for " ~ labels[index]);
        }
    }
    Result[] results;
    auto name = "multivariateNormal_" ~ columns.to!string ~ (transpose ? "_transposed" : "_rowMajor");
    foreach (round; 0 .. rounds)
    {
        double[functions.length] values;
        const first = round % active;
        auto elapsed = benchmarkPrepared!functions(iterations, values, enabled,
            BenchmarkSchedule(round == 0 ? warmup : 0, first), &prepare, &check);
        size_t ordinal;
        foreach (i, label; labels) if (enabled[i])
        {
            results ~= Result(name, label, values[i], elapsed[i], Transform.none,
                round + 1, (ordinal + active - first) % active + 1);
            ++ordinal;
        }
    }
    return results;
}
