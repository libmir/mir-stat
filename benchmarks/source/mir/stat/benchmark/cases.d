/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.benchmark.cases;

import core.time: Duration;
import mir.math.internal.benchmark: benchmarkPrepared;
import mir.math.sum: Summation;
import mir.stat.descriptive.univariate: skewness, kurtosis, SkewnessAlgo, KurtosisAlgo;
import mir.stat.descriptive.multivariate: covariance, correlation, CovarianceAlgo, CorrelationAlgo;
import std.meta: AliasSeq;
import std.traits: EnumMembers;

package(mir.stat):

enum functionNames = ["skewness", "kurtosis", "covariance", "correlation"];

enum Transform { none, center, standardize }

// An empty selection represents all applicable algorithms.
string[] parseAlgorithms(string text) @safe pure
{
    import std.algorithm: canFind;
    import std.array: split;
    import std.string: strip;
    import std.exception: enforce;
    text = text.strip;
    if (text == "all")
        return null;
    string[] selected;
    foreach (part; text.split(","))
    {
        auto name = part.strip;
        enforce(name.length > 0, "Algorithm names must not be empty");
        enforce(name != "all", "Use all alone, not with individual algorithms");
        enforce(!selected.canFind(name), "Duplicate algorithm: " ~ name);
        selected ~= name;
    }
    enforce(selected.length > 0, "Select at least one algorithm or all");
    return selected;
}

private bool applicable(string algorithm, Transform transform) @safe pure nothrow @nogc
{
    return (algorithm != "assumeZeroMean" || transform != Transform.none) &&
        (algorithm != "assumeStandardized" || transform == Transform.standardize);
}

private string[] algorithmNames(Choices)() @safe pure
{
    string[] names;
    static foreach (choice; EnumMembers!Choices)
        names ~= __traits(identifier, choice);
    return names;
}

private string[] availableAlgorithms(string name) @safe pure
{
    switch (name)
    {
    case "skewness": return algorithmNames!SkewnessAlgo;
    case "kurtosis": return algorithmNames!KurtosisAlgo;
    case "covariance": return algorithmNames!CovarianceAlgo;
    case "correlation": return algorithmNames!CorrelationAlgo;
    default: assert(0, "Function must be validated first");
    }
}

struct Result
{
    string functionName;
    string algorithm;
    double value;
    Duration elapsed;
    Transform transform;
}

void validateFunction(string name, size_t iterations, size_t size, Transform transform = Transform.none, string[] selected = null)
{
    import std.algorithm: canFind;
    import std.exception: enforce;
    enforce(functionNames.canFind(name), "Unknown benchmark function: " ~ name);
    enforce(transform == Transform.none || transform == Transform.center ||
        transform == Transform.standardize, "Unknown transform");
    enforce(iterations > 0, "Benchmark needs at least one iteration");
    auto minimum = name == "kurtosis" ? 4 : name == "skewness" ? 3 : 2;
    enforce(size >= minimum, "Input size is too small for " ~ name);
    auto available = availableAlgorithms(name);
    foreach (i, algorithm; selected)
    {
        enforce(available.canFind(algorithm), "Unknown algorithm for " ~ name ~ ": " ~ algorithm);
        enforce(applicable(algorithm, transform), "Algorithm " ~ algorithm ~ " is incompatible with the selected transform");
        enforce(!selected[0 .. i].canFind(algorithm), "Duplicate algorithm: " ~ algorithm);
    }
}

// Cases return results; presentation and command-line handling belong to the runner.
Result[] runFunction(string name, size_t iterations, size_t size, ulong seed = 5489, Transform transform = Transform.none, string[] selected = null)
{
    validateFunction(name, iterations, size, transform, selected);
    switch (name)
    {
    case "skewness": return measureInput!(skewness, SkewnessAlgo)(name, iterations, size, seed, transform, selected);
    case "kurtosis": return measureInput!(kurtosis, KurtosisAlgo)(name, iterations, size, seed, transform, selected);
    case "covariance": return measureInput!(covariance, CovarianceAlgo)(name, iterations, size, seed, transform, selected);
    case "correlation": return measureInput!(correlation, CorrelationAlgo)(name, iterations, size, seed, transform, selected);
    default: assert(0);
    }
}

private template algorithms(alias operation, Choices...)
{
    alias algorithms = AliasSeq!();
    static foreach (choice; Choices)
        algorithms = AliasSeq!(algorithms, operation!(double, choice, Summation.fast));
}

private Result[] measureInput(alias operation, Choices)(string name, size_t iterations,
    size_t size, ulong seed, Transform transform, string[] selected)
{
    final switch (transform)
    {
    case Transform.none:
        return measure!(operation, Choices, Transform.none)(name, iterations, size, seed, selected);
    case Transform.center:
        return measure!(operation, Choices, Transform.center)(name, iterations, size, seed, selected);
    case Transform.standardize:
        return measure!(operation, Choices, Transform.standardize)(name, iterations, size, seed, selected);
    }
}

private Result[] measure(alias operation, Choices, Transform mode)(string name, size_t iterations, size_t size, ulong seed, string[] selected)
{
    import mir.ndslice.allocation: stdcUninitSlice, stdcFreeSlice;
    import mir.ndslice.slice: Slice;
    import mir.random.engine.mersenne_twister: Mt19937_64;
    import mir.random.variable: NormalVariable;

    import std.meta: Filter;
    template included(alias choice)
    {
        enum included = applicable(__traits(identifier, choice), mode);
    }
    alias choices = Filter!(included, EnumMembers!Choices);
    alias functions = algorithms!(operation, choices);
    import std.algorithm: canFind;
    bool[functions.length] enabled;
    static foreach (i, choice; choices)
        enabled[i] = selected.length == 0 || selected.canFind(__traits(identifier, choice));
    enum paired = is(Choices == CorrelationAlgo) || is(Choices == CovarianceAlgo);
    auto x = stdcUninitSlice!double(size);
    scope(exit) x.stdcFreeSlice;
    static if (paired)
    {
        auto y = stdcUninitSlice!double(size);
        scope(exit) y.stdcFreeSlice;
    }
    // A local, explicitly selected engine avoids thread-local RNG state.
    auto generator = Mt19937_64(seed);
    auto normal = NormalVariable!double(0, 1);
    double reference;
    void prepare()
    {
        foreach (i; 0 .. size)
        {
            // Use nonzero location and nonunit scale for the general case.
            x[i] = 2 + 2 * normal(generator);
            static if (paired)
                y[i] = -1 + 0.5 * (x[i] - 2) + 3 * normal(generator);
        }
        static if (mode == Transform.standardize)
        {
            standardize(x);
            static if (paired)
                standardize(y);
        }
        else static if (mode == Transform.center)
        {
            center(x);
            static if (paired)
                center(y);
        }
        // Reference calculations use measured centering/variance, never an
        // assumeZeroMean or assumeStandardized shortcut.
        static if (is(Choices == SkewnessAlgo) || is(Choices == KurtosisAlgo))
            reference = operation!(double, Choices.threePass, Summation.precise)(x);
        else
            reference = operation!(double, Choices.twoPass, Summation.precise)(x, y);
    }
    import std.conv: to;
    auto inputLabel = mode.to!string;
    string[functions.length] labels;
    static foreach (i, choice; choices)
        labels[i] = name ~ ("/" ~ inputLabel ~ "/") ~ __traits(identifier, choice);
    void check(size_t index, double value)
    {
        checkResult(labels[index], value, reference);
    }

    // All algorithms see the same prepared buffers, through read-only views.
    Slice!(const(double)*) inputX = x;
    double[functions.length] values;
    static if (paired)
    {
        Slice!(const(double)*) inputY = y;
        auto times = benchmarkPrepared!functions(iterations, values, enabled, &prepare, &check, inputX, inputY);
    }
    else
        auto times = benchmarkPrepared!functions(iterations, values, enabled, &prepare, &check, inputX);

    Result[] results;
    static foreach (i, choice; choices)
        if (enabled[i])
            results ~= Result(name, __traits(identifier, choice), values[i], times[i], mode);
    return results;
}

// Mixed tolerance handles statistics near zero as well as order-one values.
// This is a benchmark guard for these normal input distributions, not a general accuracy
// contract for the statistical algorithms. Reject nonfinite values explicitly.
private void checkResult(string label, double value, double reference) @safe pure
{
    import std.math: isFinite, abs;
    import std.format: format;
    enum tolerance = 1e-10;
    if (!(value.isFinite && reference.isFinite &&
        abs(value - reference) <= tolerance + tolerance * abs(reference)))
        throw new Exception(format("%s: result %.17g differs from reference %.17g", label, value, reference));
}

private void center(R)(R x)
{
    import mir.stat.descriptive.univariate: mean;
    auto average = x.mean!(double, Summation.precise);
    foreach (ref value; x)
        value -= average;
}

private void standardize(R)(R x)
{
    import mir.stat.transform: zscore;
    auto scores = x.zscore;
    foreach (i; 0 .. x.length)
        x[i] = scores[i];
}

@system
unittest
{
    import std.exception: assertThrown;
    import std.math: isFinite;
    foreach (name; functionNames)
    foreach (mode; EnumMembers!Transform)
    {
        auto results = runFunction(name, 2, 32, 5489, mode);
        auto expected = name == "skewness" ? EnumMembers!SkewnessAlgo.length
            : name == "kurtosis" ? EnumMembers!KurtosisAlgo.length
            : name == "covariance" ? EnumMembers!CovarianceAlgo.length
            : EnumMembers!CorrelationAlgo.length;
        if (mode == Transform.none)
            expected -= name == "correlation" ? 2 : 1;
        else if (name == "correlation" && mode == Transform.center)
            --expected;
        assert(results.length == expected);
        foreach (result; results)
        {
            assert(result.functionName == name && result.transform == mode);
            if (mode == Transform.none)
                assert(result.algorithm != "assumeZeroMean" && result.algorithm != "assumeStandardized");
            if (mode == Transform.center)
                assert(result.algorithm != "assumeStandardized");
            assert(result.algorithm.length > 0);
            assert(result.value.isFinite);
        }
        assertThrown!Exception(runFunction(name, 0, 32));
        assertThrown!Exception(runFunction(name, 1, 0));
    }
    assertThrown!Exception(runFunction("unknown", 1, 32));
    assertThrown!Exception(runFunction("correlation", 1, 32, 5489, cast(Transform) 99));
    assertThrown!Exception(runFunction("skewness", 1, 2));
    assertThrown!Exception(runFunction("kurtosis", 1, 3));
    assertThrown!Exception(runFunction("correlation", 1, 1));
    assertThrown!Exception(runFunction("covariance", 1, 1));
}

// Replaying a seed reproduces values, not elapsed time. Check every supported
// case and its minimum input size as well as a larger input.
@system
unittest
{
    import std.math: abs;
    foreach (name; functionNames)
    {
        size_t minimum = name == "kurtosis" ? 4 : name == "skewness" ? 3 : 2;
        foreach (size; [minimum, size_t(32)])
        foreach (seed; [0UL, 5489UL, ulong.max])
        foreach (mode; EnumMembers!Transform)
        {
            auto first = runFunction(name, 3, size, seed, mode);
            auto second = runFunction(name, 3, size, seed, mode);
            foreach (i; 0 .. first.length)
            {
                // Same algorithm, compiler and seed replay identical operations.
                assert(first[i].value == second[i].value);
                assert(abs(first[i].value - first[0].value) <=
                    2e-10 * (1 + abs(first[0].value)));
            }
        }
    }
    auto firstSeed = runFunction("covariance", 3, 32, 1);
    auto secondSeed = runFunction("covariance", 3, 32, 2);
    assert(firstSeed[0].value != secondSeed[0].value);
}

@safe pure
unittest
{
    import std.exception: assertThrown;
    checkResult("zero", 1e-12, 0);
    checkResult("relative", 1000 + 1e-8, 1000);
    assertThrown!Exception(checkResult("incorrect", 0.1, 0));
    assertThrown!Exception(checkResult("nan", double.nan, 0));
    assertThrown!Exception(checkResult("infinity", double.infinity, double.infinity));
    assertThrown!Exception(checkResult("invalid reference", 0, double.nan));
}

@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.math.sum: sum;
    import mir.math.common: approxEqual;
    import std.math: abs;
    double[4] x = [10, 12, 15, 19];
    double[4] y = [9, 8, 11, 14];
    center(x[].sliced);
    assert(abs(x[].sum) <= 32 * double.epsilon);
    double centeredSquares = 0;
    foreach (value; x)
        centeredSquares += value * value;
    assert(centeredSquares.approxEqual(46.0)); // Centering does not scale variance.
    standardize(x[].sliced);
    standardize(y[].sliced);
    assert(abs(x[].sum) <= 32 * double.epsilon);
    assert(abs(y[].sum) <= 32 * double.epsilon);
    double xx = 0, yy = 0;
    foreach (i; 0 .. x.length)
    {
        xx += x[i] * x[i];
        yy += y[i] * y[i];
    }
    assert(xx.approxEqual(3.0) && yy.approxEqual(3.0));
}

@safe pure
unittest
{
    import std.exception: assertThrown;
    assert(parseAlgorithms("all").length == 0);
    assert(parseAlgorithms(" twoPass, assumeStandardized ") == ["twoPass", "assumeStandardized"]);
    foreach (invalid; ["", " ", ",", "twoPass,", "all,twoPass", "twoPass,twoPass"])
        assertThrown!Exception(parseAlgorithms(invalid));
}

@system
unittest
{
    import std.algorithm: find;
    import std.exception: assertThrown;
    foreach (name; functionNames)
    foreach (transform; EnumMembers!Transform)
    {
        auto all = runFunction(name, 2, 32, 123, transform);
        auto selected = runFunction(name, 2, 32, 123, transform, ["twoPass", "online"]);
        assert(selected.length == 2);
        foreach (result; selected)
        {
            assert(result.algorithm == "twoPass" || result.algorithm == "online");
            auto matching = all.find!(other => other.algorithm == result.algorithm);
            assert(matching.length > 0 && result.value == matching[0].value);
        }
    }
    auto single = runFunction("correlation", 2, 32, 123, Transform.standardize, ["assumeStandardized"]);
    assert(single.length == 1 && single[0].algorithm == "assumeStandardized");
    auto shortcuts = runFunction("correlation", 2, 32, 123, Transform.standardize,
        ["assumeStandardized", "assumeZeroMean"]);
    assert(shortcuts.length == 2);
    foreach (transform; [Transform.none, Transform.center])
        assertThrown!Exception(runFunction("correlation", 1, 32, 123, transform, ["assumeStandardized"]));
    assertThrown!Exception(runFunction("correlation", 1, 32, 123, Transform.none, ["assumeZeroMean"]));
    assertThrown!Exception(runFunction("skewness", 1, 32, 123, Transform.standardize, ["assumeStandardized"]));
    assertThrown!Exception(runFunction("correlation", 1, 32, 123, Transform.none, ["unknown"]));
    assertThrown!Exception(runFunction("correlation", 1, 32, 123, Transform.none, ["online", "online"]));
}
