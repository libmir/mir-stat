/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.benchmark.cases;

import core.time: Duration;
import mir.math.internal.benchmark: benchmarkRandom, benchmarkRandom2;
import mir.math.sum: Summation;
import mir.stat.descriptive.univariate: skewness, kurtosis, SkewnessAlgo, KurtosisAlgo;
import mir.stat.descriptive.multivariate: covariance, correlation, CovarianceAlgo, CorrelationAlgo;
import std.meta: AliasSeq;
import std.traits: EnumMembers;

package(mir.stat):

enum caseNames = ["skewness", "kurtosis", "covariance", "correlation"];

struct Result
{
    string statistic;
    string algorithm;
    double value;
    Duration elapsed;
}

void validateCase(string name, size_t iterations, size_t size)
{
    import std.algorithm: canFind;
    import std.exception: enforce;
    enforce(caseNames.canFind(name), "Unknown benchmark case: " ~ name);
    enforce(iterations > 0, "Benchmark needs at least one iteration");
    auto minimum = name == "kurtosis" ? 4 : name == "skewness" ? 3 : 2;
    enforce(size >= minimum, "Input size is too small for " ~ name);
}

// Cases return results; presentation and command-line handling belong to the runner.
Result[] runCase(string name, size_t iterations, size_t size)
{
    validateCase(name, iterations, size);
    switch (name)
    {
    case "skewness": return measure!(skewness, SkewnessAlgo)(name, iterations, size);
    case "kurtosis": return measure!(kurtosis, KurtosisAlgo)(name, iterations, size);
    case "covariance": return measure!(covariance, CovarianceAlgo)(name, iterations, size);
    case "correlation": return measure!(correlation, CorrelationAlgo)(name, iterations, size);
    default: assert(0);
    }
}

private template algorithms(alias operation, Choices...)
{
    alias algorithms = AliasSeq!();
    static foreach (choice; Choices)
        algorithms = AliasSeq!(algorithms, operation!(double, choice, Summation.fast));
}

private Result[] measure(alias operation, Choices)(string name, size_t iterations, size_t size)
{
    alias choices = EnumMembers!Choices;
    alias functions = algorithms!(operation, choices);
    double[functions.length] values;
    static if (is(Choices == CorrelationAlgo))
    {
        import mir.ndslice.slice: Slice;
        // Standardize outside timing, as required by the assumption-based algorithms.
        auto times = benchmarkRandom2!functions(iterations, size, values,
            &standardize!(Slice!(double*)));
    }
    else static if (is(Choices == CovarianceAlgo))
        auto times = benchmarkRandom2!functions(iterations, size, values);
    else
        auto times = benchmarkRandom!functions(iterations, size, values);

    Result[] results;
    static foreach (i, choice; choices)
        results ~= Result(name, __traits(identifier, choice), values[i], times[i]);
    return results;
}

private void standardize(R)(R x, R y)
{
    import mir.stat.transform: zscore;
    auto zx = x.zscore;
    auto zy = y.zscore;
    foreach (i; 0 .. x.length)
    {
        x[i] = zx[i];
        y[i] = zy[i];
    }
}

@system
unittest
{
    import std.exception: assertThrown;
    import std.math: isFinite;
    foreach (name; caseNames)
    {
        auto results = runCase(name, 2, 32);
        auto expected = name == "skewness" ? EnumMembers!SkewnessAlgo.length
            : name == "kurtosis" ? EnumMembers!KurtosisAlgo.length
            : name == "covariance" ? EnumMembers!CovarianceAlgo.length
            : EnumMembers!CorrelationAlgo.length;
        assert(results.length == expected);
        foreach (result; results)
        {
            assert(result.statistic == name);
            assert(result.algorithm.length > 0);
            assert(result.value.isFinite);
        }
        assertThrown!Exception(runCase(name, 0, 32));
        assertThrown!Exception(runCase(name, 1, 0));
    }
    assertThrown!Exception(runCase("unknown", 1, 32));
    assertThrown!Exception(runCase("skewness", 1, 2));
    assertThrown!Exception(runCase("kurtosis", 1, 3));
    assertThrown!Exception(runCase("correlation", 1, 1));
    assertThrown!Exception(runCase("covariance", 1, 1));
}
