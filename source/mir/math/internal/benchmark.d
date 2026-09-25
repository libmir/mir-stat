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

// Warm up once before measurement, and rotate only among enabled algorithms.
package(mir) struct BenchmarkSchedule
{
    size_t warmup;
    size_t firstAlgorithm;
}

// Prepare one dataset per iteration and check every result outside timing.
// Each function receives the same arguments; callers should provide read-only
// views when sharing input buffers across algorithms.
package(mir)
template benchmarkPrepared(fun...)
{
    Duration[fun.length] benchmarkPrepared(T, Prepare, Check, Args...)(
        size_t n, out T[fun.length] values, const bool[fun.length] enabled,
        BenchmarkSchedule schedule, scope Prepare prepare, scope Check check, Args args)
    {
        import std.datetime.stopwatch: StopWatch;
        import std.exception: enforce;
        enforce(n > 0, "Benchmark needs at least one iteration");
        size_t[fun.length] active;
        size_t count;
        foreach (i, include; enabled)
            if (include)
                active[count++] = i;
        enforce(count > 0, "Benchmark needs at least one selected algorithm");
        auto first = schedule.firstAlgorithm % count;
        values[] = 0;
        StopWatch[fun.length] watches;
        void iteration(bool timed)()
        {
            prepare();
            foreach (offset; 0 .. count)
            {
                auto index = active[(first + offset) % count];
                // Dispatch before starting the clock, preserving direct calls.
                foreach (i, operation; fun)
                {
                    if (index == i)
                    {
                        static if (timed)
                            watches[i].start();
                        auto value = operation(args);
                        static if (timed)
                        {
                            watches[i].stop();
                            values[i] += value;
                        }
                        check(i, value);
                    }
                }
            }
        }
        foreach (i; 0 .. schedule.warmup)
            iteration!false();
        foreach (i; 0 .. n)
            iteration!true();
        Duration[fun.length] elapsed;
        foreach (i, ref value; values)
        {
            value /= n;
            // Preserve native clock ticks across calls; Duration conversion
            // rounds to 100 ns, so convert only once per algorithm.
            elapsed[i] = watches[i].peek();
        }
        return elapsed;
    }
}

version(mir_stat_test)
@safe
unittest
{
    import std.exception: assertThrown;
    size_t preparations, checks, calls;
    int[1] data;
    void prepare() { data[0] = cast(int) ++preparations; }
    int first(const(int)[] input)
    {
        assert(checks == 2 * (preparations - 1));
        ++calls;
        return input[0];
    }
    int second(const(int)[] input)
    {
        assert(checks == 2 * preparations - 1);
        ++calls;
        return input[0];
    }
    void check(size_t index, double value)
    {
        assert(index == checks % 2);
        assert(value == preparations);
        ++checks;
    }
    double[2] values;
    benchmarkPrepared!(first, second)(3, values, [true, true], BenchmarkSchedule.init, &prepare, &check, cast(const(int)[]) data[]);
    assert(preparations == 3 && checks == 6 && calls == 6);
    assert(values == [2.0, 2.0]);
    assertThrown!Exception(benchmarkPrepared!(first, second)(0, values, [true, true], BenchmarkSchedule.init, &prepare, &check, data[]));
    assert(preparations == 3 && checks == 6 && calls == 6);

    void failPreparation() { throw new Exception("preparation failed"); }
    assertThrown!Exception(benchmarkPrepared!(first, second)(1, values, [true, true], BenchmarkSchedule.init, &failPreparation, &check, data[]));
    assert(calls == 6);

    preparations = checks = calls = 0;
    void failCheck(size_t index, double value) { throw new Exception("incorrect result"); }
    assertThrown!Exception(benchmarkPrepared!(first, second)(3, values, [true, true], BenchmarkSchedule.init, &prepare, &failCheck, data[]));
    assert(preparations == 1 && calls == 1);

    preparations = checks = calls = 0;
    int failOperation(const(int)[] input) { throw new Exception("operation failed"); }
    assertThrown!Exception(benchmarkPrepared!(failOperation, second)(1, values, [true, true], BenchmarkSchedule.init, &prepare, &check, data[]));
    assert(preparations == 1 && checks == 0 && calls == 0);
}

version(mir_stat_test)
@safe
unittest
{
    import std.exception: assertThrown;
    size_t preparations, checks, skippedCalls;
    void prepare() { ++preparations; }
    int skipped() { ++skippedCalls; return -1; }
    int selected() { return 12; }
    void check(size_t index, double value)
    {
        assert(index == 1 && value == 12);
        ++checks;
    }
    double[2] values;
    auto elapsed = benchmarkPrepared!(skipped, selected)(3, values, [false, true], BenchmarkSchedule.init, &prepare, &check);
    assert(preparations == 3 && checks == 3 && skippedCalls == 0);
    assert(values == [0.0, 12.0]);
    assert(elapsed[0] == Duration.zero);
    assertThrown!Exception(benchmarkPrepared!(skipped, selected)(3, values, [false, false], BenchmarkSchedule.init, &prepare, &check));
    assert(preparations == 3 && checks == 3 && skippedCalls == 0);
}

version(mir_stat_test)
@safe
unittest
{
    import std.exception: assertThrown;
    size_t preparations, checks, skippedCalls;
    size_t[10] order;
    void prepare() { ++preparations; }
    double first() { return cast(double) preparations; }
    double skipped() { ++skippedCalls; return -1; }
    double last() { return 10.0 * preparations; }
    void check(size_t index, double value)
    {
        order[checks++] = index;
        assert(value == (index == 0 ? 1.0 : 10.0) * preparations);
    }
    double[3] values;
    // Rotate across the two enabled entries, ignoring the disabled middle one.
    benchmarkPrepared!(first, skipped, last)(3, values, [true, false, true],
        BenchmarkSchedule(2, 1), &prepare, &check);
    assert(preparations == 5 && checks == 10 && skippedCalls == 0);
    assert(order == [2, 0, 2, 0, 2, 0, 2, 0, 2, 0]);
    assert(values == [4.0, 0.0, 40.0]); // Warm-up values 1 and 2 were excluded.

    preparations = checks = 0;
    assertThrown!Exception(benchmarkPrepared!(first, skipped, last)(0, values,
        [true, false, true], BenchmarkSchedule(2, 1), &prepare, &check));
    assert(preparations == 0 && checks == 0);
    void rejectWarmup(size_t index, double value)
    {
        ++checks;
        throw new Exception("warm-up check failed");
    }
    assertThrown!Exception(benchmarkPrepared!(first, skipped, last)(3, values,
        [true, false, true], BenchmarkSchedule(2, 1), &prepare, &rejectWarmup));
    assert(preparations == 1 && checks == 1);
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
    Duration[fun.length] benchmarkRandom2(T, Prepare = typeof(null))(
        size_t n, size_t m, out T[fun.length] values, scope Prepare prepare = null)
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
                // Prepare inputs required by specialized algorithms outside timing.
                static if (!is(Prepare == typeof(null)))
                    prepare(r1, r2);
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

// These helpers require mir-random, supplied by the unittest-benchmark configuration.
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

    import mir.ndslice.slice: Slice;
    size_t preparations;
    void prepare(Slice!(double*) x, Slice!(double*) y)
    {
        ++preparations;
        x[] = 6.0;
        y[] = 12.0;
    }
    alias preparedInput = (x, y) {
        ++calls;
        assert(x[0] == 6.0 && y[0] == 12.0);
        return x[0] + y[0];
    };
    benchmarkRandom2!preparedInput(3, 4, values, &prepare);
    assert(preparations == 3 && calls == 3 && values[0] == 18.0);
    assertThrown!Exception(benchmarkRandom2!preparedInput(0, 4, values, &prepare));
    assert(preparations == 3 && calls == 3);
    static void failPreparation(Slice!(double*) x, Slice!(double*) y)
    {
        throw new Exception("benchmark preparation");
    }
    assertThrown!Exception(benchmarkRandom2!preparedInput(1, 4, values, &failPreparation));
    assert(calls == 3);
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
