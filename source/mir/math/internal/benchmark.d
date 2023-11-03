module mir.math.internal.benchmark;

import core.time;
import std.traits: isMutable;

package(mir)
template benchmarkValues(fun...)
{
    Duration[fun.length] benchmarkValues(T)(size_t numberSimulations, out T[fun.length] values)
    {
        import std.datetime.stopwatch: StopWatch, AutoStart;
        Duration[fun.length] result;
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 1 .. numberSimulations) {
                values[i] += fun[i]();
            }
            result[i] = sw.peek();
            values[i] /= numberSimulations;
        }

        return result;
    }
}

// handles 1-dimension case
package(mir)
template benchmarkRandom(fun...)
{
    Duration[fun.length] benchmarkRandom(T)(size_t numberSimulations, size_t m, out T[fun.length] values)
        if (isMutable!T)
    {
        import mir.ndslice.allocation: stdcFreeSlice, stdcUninitSlice;
        import mir.random.engine: Random, threadLocalPtr;
        import mir.random.variable: NormalVariable;
        import std.datetime.stopwatch: StopWatch, AutoStart;

        Random* gen = threadLocalPtr!Random;
        auto rv = NormalVariable!T(0, 1);

        Duration[fun.length] result;
        auto r = stdcUninitSlice!T(m);
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 1 .. numberSimulations) {
                sw.stop();
                foreach (ref e; r)
                    e = rv(gen);
                sw.start();
                values[i] += fun[i](r);
            }
            result[i] = sw.peek();
            values[i] /= numberSimulations;
        }
        r.stdcFreeSlice;
        return result;
    }
}

// handles 2-dimension case
package(mir)
template benchmarkRandom2(fun...)
{
    Duration[fun.length] benchmarkRandom2(T)(size_t numberSimulations, size_t m, out T[fun.length] values)
        if (isMutable!T)
    {
        import mir.ndslice.allocation: stdcFreeSlice, stdcUninitSlice;
        import mir.random.engine: Random, threadLocalPtr;
        import mir.random.variable: NormalVariable;
        import std.datetime.stopwatch: StopWatch, AutoStart;

        Random* gen = threadLocalPtr!Random;
        auto rv = NormalVariable!T(0, 1);

        Duration[fun.length] result;
        auto r1 = stdcUninitSlice!T(m);
        auto r2 = stdcUninitSlice!T(m);
        auto sw = StopWatch(AutoStart.yes);

        foreach (i, unused; fun) {
            values[i] = 0;
            sw.reset();
            foreach (size_t j; 1 .. numberSimulations) {
                sw.stop();
                foreach (size_t k; 0 .. m) {
                    r1[k] = rv(gen);
                    r2[k] = r1[k] + rv(gen);
                }
                sw.start();
                values[i] += fun[i](r1, r2);
            }
            result[i] = sw.peek();
            values[i] /= numberSimulations;
        }
        r1.stdcFreeSlice;
        r2.stdcFreeSlice;
        return result;
    }
}

// handles N-dimension case with additional inputs for mahalanobis
package(mir)
template benchmarkRandomN_mahal(bool[] usePrecision, bool[] useCholesky, fun...)
{
    import mir.ndslice;
    import mir.ndslice.slice: Slice;
    import mir.rc.array: RCI;

    Duration[fun.length] benchmarkRandomN_mahal(T)(size_t numberSimulations, size_t m, size_t n, Slice!(RCI!T, 1) mu, Slice!(RCI!T, 2) sigma, ref Slice!(RCI!T, 2) values)
        if (isMutable!T)
    {
        import mir.math.internal.lubeck2: mlinverse;
        import mir.math.internal.linearAlgebra: choleskyDecompose;
        import mir.ndslice.allocation: stdcFreeSlice, stdcUninitSlice, mininitRcslice;
        import mir.random.engine: Random, threadLocalPtr;
        import mir.random.ndvariable: multivariateNormalVariable;
        import mir.rc.array: RCArray;
        import std.datetime.stopwatch: StopWatch, AutoStart;

        auto muScope = mu.lightScope;
        auto sigmaScope = sigma.lightScope;
        auto invSigma = mlinverse(sigma);
        auto invSigmaScope = invSigma.lightScope;
        auto cholesky = choleskyDecompose(sigma);
        auto choleskyScope = cholesky.lightScope;
        auto invCholesky = choleskyDecompose(invSigma);
        auto invCholeskyScope = invCholesky.lightScope;

        Random* gen = threadLocalPtr!Random;
        auto sigmaCopy = stdcUninitSlice!T(n, n);
        foreach (size_t i; 0 .. n) {
            foreach (size_t j; i .. n) {
                sigmaCopy[i, j] = sigma[i, j];
                if (i != j) {
                    sigmaCopy[j, i] = sigmaCopy[i, j];
                }
            }
        }
        auto rv = multivariateNormalVariable!T(muScope, sigmaCopy, false);

        Duration[fun.length] result;
        auto r = stdcUninitSlice!T(m, n);
        auto temp1 = mininitRcslice!T(m);
        auto temp2 = RCArray!T(n);
        auto sw = StopWatch(AutoStart.yes);
        // Make sure initially zeros
        foreach (size_t i; 0 .. fun.length) {
            foreach (size_t k; 0 .. m) {
                values[i, k] = 0.0;
            }
        }
        foreach (i, unused; fun) {
            sw.reset();
            foreach (size_t j; 1 .. numberSimulations) {
                sw.stop();
                foreach (size_t k; 0 .. m) {
                    rv(gen, temp2[]);
                    foreach (size_t l; 0 .. n) {
                        r[k, l] -= temp2[l] - muScope[l];
                    }
                }
                sw.start();
                static if (usePrecision[i] == true) {
                    static if (useCholesky[i] == false) {
                        temp1 = fun[i](r, invSigmaScope);
                    } else {
                        temp1 = fun[i](r, invCholeskyScope);
                    }
                } else {
                    static if (useCholesky[i] == false) {
                        temp1 = fun[i](r, sigmaScope);
                    } else {
                        temp1 = fun[i](r, choleskyScope);
                    }
                }
                foreach (size_t k; 0 .. m) {
                    values[i, k] += temp1[k];
                }
            }
            result[i] = sw.peek();
            foreach (size_t k; 0 .. m) {
                values[i, k] /= numberSimulations;
            }
        }
        r.stdcFreeSlice;
        sigmaCopy.stdcFreeSlice;
        return result;
    }
}
