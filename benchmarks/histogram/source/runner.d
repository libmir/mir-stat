// License: Apache-2.0
module mir.stat.benchmark.histogram_runner;

import mir.math.internal.benchmark;

// Only the declaration is visible while optimizing the benchmark caller.
extern(C) ulong mirStatHistogramObserve(const(uint)* counts, size_t length);

// Forward to the opaque observer without retaining the count storage.
private void observe(const(uint)* counts, size_t length)
{
    mirStatHistogramObserve(counts, length);
}

// Each invocation measures one allocation/axis case in a fresh process.
int main(string[] args)
{
    import std.getopt : getopt, defaultGetoptPrinter;
    import std.exception : enforce;
    import std.stdio : writeln, stderr;
    import std.conv : to;
    try
    {
        string allocationName = "gc";
        bool variableAxis;
        size_t observations = 1024;
        size_t iterations = 10000;
        auto options = getopt(args,
            "allocation", "gc, rc, or custom", &allocationName,
            "variable-axis", "Use variable boundaries instead of a regular axis", &variableAxis,
            "size", "Number of observations (default 1024)", &observations,
            "iterations", "Number of constructions (default 10000)", &iterations);
        if (options.helpWanted)
        {
            defaultGetoptPrinter("Histogram construction benchmark", options.options);
            return 0;
        }
        enforce(args.length == 1, "Unexpected positional arguments");
        auto selected = allocationName.to!HistogramAllocation;
        HistogramConstructionTiming result;
        static foreach (allocation; [HistogramAllocation.gc, HistogramAllocation.rc, HistogramAllocation.custom])
            if (selected == allocation)
            {
                if (variableAxis)
                    result = benchmarkHistogramConstruction!(allocation, true)(observations, iterations, &observe);
                else
                    result = benchmarkHistogramConstruction!(allocation, false)(observations, iterations, &observe);
            }
        writeln("allocation=", selected, " variableAxis=", variableAxis,
            " size=", observations, " iterations=", iterations, " checksum=", result.checksum);
        writeln("construction=", result.timing.loop.elapsed,
            " gcTime=", result.timing.loop.gcTime, " collections=", result.timing.loop.collections);
        writeln("finalCollection=", result.timing.finalCollection.elapsed,
            " gcTime=", result.timing.finalCollection.gcTime,
            " collections=", result.timing.finalCollection.collections);
        writeln("total=", result.timing.elapsed);
        return 0;
    }
    catch (Exception e)
    {
        stderr.writeln(e.msg);
        return 1;
    }
}
