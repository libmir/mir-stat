/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.
+/

module mir.stat.benchmark.report;

import mir.stat.benchmark.cases: Result;

package(mir.stat):

enum csvHeader = "function,transform,algorithm,round,position,iterations,size,seed,warmup,mean_value,elapsed_ns";

// Identifiers in Result come from the validated function/algorithm enumerations.
string csvRow(Result result, size_t iterations, size_t size, ulong seed, size_t warmup) @safe pure
{
    import std.format: format;
    return format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%.17g,%s",
        result.functionName, result.transform, result.algorithm, result.round,
        result.position, iterations, size, seed, warmup, result.value,
        result.elapsed.total!"nsecs");
}

@safe pure
unittest
{
    import core.time: nsecs;
    import mir.stat.benchmark.cases: Transform;
    auto result = Result("correlation", "twoPass", -0.5, nsecs(1200), Transform.center, 2, 1);
    assert(csvRow(result, 10, 32, 5489, 3) == "correlation,center,twoPass,2,1,10,32,5489,3,-0.5,1200");
    result.value = 1.0 / 3;
    result.elapsed = nsecs(1_000_000_100);
    assert(csvRow(result, 10, 32, ulong.max, 0) ==
        "correlation,center,twoPass,2,1,10,32,18446744073709551615,0,0.33333333333333331,1000000100");
}
