/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.benchmark.runner;

import mir.stat.benchmark.cases: caseNames, validateCase, runCase;

int main(string[] args)
{
    import std.getopt: getopt, defaultGetoptPrinter;
    import std.exception: enforce;
    import std.stdio: stderr, writefln;
    try
    {
        string selected = "all";
        size_t iterations = 10_000;
        size_t size = 1_000;
        auto options = getopt(args,
            "case", "all, skewness, kurtosis, covariance, or correlation", &selected,
            "iterations", "Number of iterations per algorithm (default 10000)", &iterations,
            "size", "Observations per input (default 1000)", &size);
        if (options.helpWanted)
        {
            defaultGetoptPrinter("mir-stat statistical benchmarks", options.options);
            return 0;
        }
        enforce(args.length == 1, "Unexpected positional arguments");
        auto names = selected == "all" ? caseNames[] : [selected];
        // Validate the whole request before starting any benchmark.
        foreach (name; names)
            validateCase(name, iterations, size);
        writefln("Iterations: %s; input size: %s; summation: fast", iterations, size);
        foreach (name; names)
        {
            foreach (result; runCase(name, iterations, size))
                writefln("%s / %s: value=%s, elapsed=%s",
                    result.statistic, result.algorithm, result.value, result.elapsed);
        }
        return 0;
    }
    catch (Exception e)
    {
        stderr.writeln(e.msg);
        return 1;
    }
}
