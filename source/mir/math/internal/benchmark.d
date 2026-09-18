/++
License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)

Authors: John Michael Hall

Copyright: 2026 Mir Stat Authors.

+/

module mir.math.internal.benchmark;

import core.time;
import std.traits: isMutable;

package(mir)
template benchmarkValues(fun...)
{
    Duration[fun.length] benchmarkValues(T)(size_t n, out T[fun.length] values)
    {
        import std.datetime.stopwatch: StopWatch, AutoStart;
        import std.exception: enforce;

        enforce(n > 0, "Benchmark needs at least one iteration");
        Duration[fun.length] result;
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 0 .. n) {
                values[i] += fun[i]();
            }
            result[i] = sw.peek();
            values[i] /= n;
        }

        return result;
    }
}

package(mir)
template benchmarkRandom(fun...)
{
    Duration[fun.length] benchmarkRandom(T)(size_t n, size_t m, out T[fun.length] values)
        if (isMutable!T)
    {
        import mir.ndslice.allocation: stdcFreeSlice, stdcUninitSlice;
        import mir.random.engine: Random, threadLocalPtr;
        import mir.random.variable: NormalVariable;
        import std.datetime.stopwatch: StopWatch, AutoStart;
        import std.exception: enforce;

        enforce(n > 0, "Benchmark needs at least one iteration");

        Random* gen = threadLocalPtr!Random;
        auto rv = NormalVariable!T(0, 1);

        Duration[fun.length] result;
        auto r = stdcUninitSlice!T(m);
        scope(exit) r.stdcFreeSlice;
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 0 .. n) {
                sw.stop();
                foreach (ref e; r)
                    e = rv(gen);
                sw.start();
                values[i] += fun[i](r);
            }
            result[i] = sw.peek();
            values[i] /= n;
        }
        return result;
    }
}

package(mir)
template benchmarkRandom2(fun...)
{
    Duration[fun.length] benchmarkRandom2(T)(size_t n, size_t m, out T[fun.length] values)
        if (isMutable!T)
    {
        import mir.ndslice.allocation: stdcFreeSlice, stdcUninitSlice;
        import mir.random.engine: Random, threadLocalPtr;
        import mir.random.variable: NormalVariable;
        import std.datetime.stopwatch: StopWatch, AutoStart;
        import std.exception: enforce;

        enforce(n > 0, "Benchmark needs at least one iteration");

        Random* gen = threadLocalPtr!Random;
        auto rv = NormalVariable!T(0, 1);

        Duration[fun.length] result;
        auto r1 = stdcUninitSlice!T(m);
        scope(exit) r1.stdcFreeSlice;
        auto r2 = stdcUninitSlice!T(m);
        scope(exit) r2.stdcFreeSlice;
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 0 .. n) {
                sw.stop();
                foreach (size_t k; 0 .. m) {
                    r1[k] = rv(gen);
                    r2[k] = r1[k] + rv(gen);
                }
                sw.start();
                values[i] += fun[i](r1, r2);
            }
            result[i] = sw.peek();
            values[i] /= n;
        }
        return result;
    }
}

version (mir_stat_test)
@safe
unittest
{
    import std.exception: assertThrown;

    size_t firstCalls, secondCalls;
    double first() { ++firstCalls; return 6.0; }
    double second() { ++secondCalls; return 12.0; }
    double[2] values;
    foreach (n; [1, 3])
    {
        firstCalls = secondCalls = 0;
        benchmarkValues!(first, second)(n, values);
        assert(firstCalls == n && secondCalls == n);
        assert(values == [6.0, 12.0]);
    }
    firstCalls = secondCalls = 0;
    assertThrown!Exception(benchmarkValues!(first, second)(0, values));
    assert(firstCalls == 0 && secondCalls == 0);
}

// These helpers require mir-random, supplied by the unittest-perf configuration.
version (mir_stat_test_benchmark)
@system
unittest
{
    import std.exception: assertThrown;

    size_t calls;
    double[1] values;
    alias oneInput = (r) {
        assert(r.length == 4);
        ++calls;
        return 6.0;
    };
    alias twoInputs = (r1, r2) {
        assert(r1.length == 4 && r2.length == 4);
        ++calls;
        return 12.0;
    };
    foreach (n; [1, 3])
    {
        calls = 0;
        benchmarkRandom!oneInput(n, 4, values);
        assert(calls == n && values[0] == 6.0);
        calls = 0;
        benchmarkRandom2!twoInputs(n, 4, values);
        assert(calls == n && values[0] == 12.0);
    }
    calls = 0;
    assertThrown!Exception(benchmarkRandom!oneInput(0, 4, values));
    assertThrown!Exception(benchmarkRandom2!twoInputs(0, 4, values));
    assert(calls == 0);

    // Exercise unwinding after buffer allocation when the callback throws.
    double fail() { throw new Exception("benchmark callback"); }
    alias throwingOne = (r) => fail();
    alias throwingTwo = (r1, r2) => fail();
    assertThrown!Exception(benchmarkRandom!throwingOne(1, 4, values));
    assertThrown!Exception(benchmarkRandom2!throwingTwo(1, 4, values));
}

// Collector counters cover the process, so use a quiet, single-workload process.
// GC time is diagnostic: it is included in elapsed, not additional to it.
package(mir) struct BenchmarkPhase
{
    Duration elapsed;
    Duration gcTime;
    ulong collections;
}

package(mir) struct BenchmarkTiming
{
    BenchmarkPhase loop;
    BenchmarkPhase finalCollection;

    Duration elapsed() const pure @safe nothrow @nogc
    {
        return loop.elapsed + finalCollection.elapsed;
    }
}

// Measure exactly iterations calls. Perform input setup, warmup and any initial
// collection before calling this helper. Automatic GC retains its normal policy.
// Optional final collection is measured separately: callers can report deferred
// reclamation without hiding it inside construction time. No printing or GC
// configuration changes occur here. Record compiler flags, affinity and runtime
// GC options alongside results; do not pin a parallel collector to one CPU when
// measuring its normal behavior.
package(mir) BenchmarkTiming benchmarkWithGCStatistics(F)(size_t iterations,
    scope F operation, bool collectAfter = false)
{
    import core.memory: GC;
    import std.datetime.stopwatch: StopWatch, AutoStart;
    import std.exception: enforce;

    enforce(iterations > 0, "Benchmark needs at least one iteration");
    BenchmarkTiming result;
    auto before = GC.profileStats();
    auto timer = StopWatch(AutoStart.yes);
    foreach (i; 0 .. iterations)
        operation();
    timer.stop();
    auto after = GC.profileStats();
    result.loop = BenchmarkPhase(timer.peek(),
        after.totalCollectionTime - before.totalCollectionTime,
        after.numCollections - before.numCollections);

    if (collectAfter)
    {
        before = GC.profileStats();
        timer.reset();
        timer.start();
        GC.collect();
        timer.stop();
        after = GC.profileStats();
        result.finalCollection = BenchmarkPhase(timer.peek(),
            after.totalCollectionTime - before.totalCollectionTime,
            after.numCollections - before.numCollections);
    }
    return result;
}

version (mir_stat_test)
@safe
unittest
{
    import std.exception: assertThrown;

    size_t calls;
    auto timing = benchmarkWithGCStatistics(3, () { ++calls; });
    assert(calls == 3);
    assert(timing.finalCollection == BenchmarkPhase.init);
    assert(timing.elapsed == timing.loop.elapsed);
    assertThrown!Exception(benchmarkWithGCStatistics(0, () { ++calls; }));
    assert(calls == 3);

    timing = benchmarkWithGCStatistics(2, () { ++calls; }, true);
    assert(calls == 5);
    assert(timing.elapsed == timing.loop.elapsed + timing.finalCollection.elapsed);
    // Do not assert durations or GC counts: those depend on runtime settings.
}
