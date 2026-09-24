/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.benchmark.runner;

import mir.stat.benchmark.cases: functionNames, validateFunction, runFunction, Transform, parseAlgorithms;

int main(string[] args)
{
    import std.getopt: getopt, defaultGetoptPrinter;
    import std.exception: enforce;
    import std.stdio: stderr, writefln;
    try
    {
        string selected = "all";
        string transformName = "none";
        string algorithmList = "all";
        size_t iterations = 10_000;
        size_t size = 1_000;
        ulong seed = 5489;
        auto options = getopt(args,
            "transform", "none (default), center, or standardize (sample z-scores)", &transformName,
            "algorithms", "Comma-separated algorithm names, or all (default)", &algorithmList,
            "seed", "Random seed (default 5489)", &seed,
            "function", "all, skewness, kurtosis, covariance, or correlation", &selected,
            "iterations", "Number of iterations per algorithm (default 10000)", &iterations,
            "size", "Observations per input (default 1000)", &size);
        if (options.helpWanted)
        {
            defaultGetoptPrinter("mir-stat statistical benchmarks", options.options);
            return 0;
        }
        enforce(args.length == 1, "Unexpected positional arguments");
        import std.conv: to;
        auto transform = transformName.to!Transform;
        auto algorithms = parseAlgorithms(algorithmList);
        auto names = selected == "all" ? functionNames[] : [selected];
        // Validate the whole request before starting any benchmark.
        foreach (name; names)
            validateFunction(name, iterations, size, transform, algorithms);
        writefln("Iterations: %s; input size: %s; seed: %s; summation: fast", iterations, size, seed);
        foreach (name; names)
        {
            foreach (result; runFunction(name, iterations, size, seed, transform, algorithms))
                writefln("%s [%s] / %s: value=%s, elapsed=%s",
                    result.functionName, result.transform, result.algorithm, result.value, result.elapsed);
        }
        return 0;
    }
    catch (Exception e)
    {
        stderr.writeln(e.msg);
        return 1;
    }
}
