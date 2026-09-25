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
        string outputFormat = "text";
        size_t rounds = 10;
        size_t warmup = 10;
        size_t iterations = 10_000;
        size_t size = 1_000;
        ulong seed = 5489;
        auto options = getopt(args,
            "rounds", "Measured rounds (default 10)", &rounds,
            "warmup", "Untimed iterations before the first round (default 10)", &warmup,
            "format", "text (default) or csv", &outputFormat,
            "transform", "none (default), center, or standardize (sample z-scores)", &transformName,
            "algorithms", "Comma-separated algorithm names, or all (default)", &algorithmList,
            "seed", "Random seed (default 5489)", &seed,
            "function", "all, skewness, kurtosis, covariance, or correlation", &selected,
            "iterations", "Iterations per algorithm per round (default 10000)", &iterations,
            "size", "Observations per input (default 1000)", &size);
        if (options.helpWanted)
        {
            defaultGetoptPrinter("mir-stat statistical benchmarks", options.options);
            return 0;
        }
        enforce(args.length == 1, "Unexpected positional arguments");
        enforce(outputFormat == "text" || outputFormat == "csv", "Unknown output format");
        import std.conv: to;
        auto transform = transformName.to!Transform;
        auto algorithms = parseAlgorithms(algorithmList);
        auto names = selected == "all" ? functionNames[] : [selected];
        // Validate the whole request before starting any benchmark.
        foreach (name; names)
            validateFunction(name, iterations, size, transform, algorithms, rounds, warmup);
        import mir.stat.benchmark.report: csvHeader, csvRow;
        import std.stdio: writeln;
        if (outputFormat == "csv")
            writeln(csvHeader);
        else
            writefln("Rounds: %s; iterations per round: %s; warm-up: %s; input size: %s; seed: %s; summation: fast",
                rounds, iterations, warmup, size, seed);
        foreach (name; names)
        {
            foreach (result; runFunction(name, iterations, size, seed, transform, algorithms, rounds, warmup))
            {
                if (outputFormat == "csv")
                    writeln(csvRow(result, iterations, size, seed, warmup));
                else
                    writefln("%s [%s] / %s: round=%s, position=%s, value=%s, elapsed=%s",
                        result.functionName, result.transform, result.algorithm,
                        result.round, result.position, result.value, result.elapsed);
            }
        }
        return 0;
    }
    catch (Exception e)
    {
        stderr.writeln(e.msg);
        return 1;
    }
}
