/++
Cached rank selection on a caller-owned Mir slice.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
+/
module mir.stat.descriptive.univariate.internal.partition;

/++ Selection policy for the internal cached rank selector. +/
package(mir.stat) enum PartitionPivotStrategy
{
    /// Partition around middle-slot values and retain intermediate pivots.
    midpoint,
    /// Use mir-algorithm's Phobos-derived rank-guided sampling and retain its established pivots.
    rankGuided,
    /// Choose rank guidance near an unresolved interval's ends, once per search.
    adaptive,
}

/++
The strategy is chosen at compile time; the default preserves midpoint selection.
Adaptive selection uses rank guidance only when the initial unresolved interval has
at least 128 elements and the target lies in its outer 1/16 at either end. These
are internal tuning choices, not quantile-definition options.

Rank is a zero-based position in the flattened slice, as for partitionAt.
Record the requested rank and the pivots established by this helper's partition
loop. When it delegates an interval to partitionAt, only the requested rank is
recorded from that call. Later calls leave every marked position untouched.

Provide a zeroed bitmap with one bit per element (round up to whole ulong
words) to remember every rank. A shorter bitmap remembers only the ranks it
can represent; an empty bitmap uses the selected strategy without recording or
reusing pivots. Bitmap storage must not overlap the slice elements. Reuse the
bitmap only with the same workspace, flattened order, and comparison. Clear it after independently
modifying the data.
+/
package(mir.stat) void partitionAtWithCache(alias less = "a < b",
    PartitionPivotStrategy strategy = PartitionPivotStrategy.midpoint, S)(
    S workspace, size_t rank, scope ulong[] established) @safe pure nothrow @nogc
    if (!is(typeof(less) == PartitionPivotStrategy))
{
    static assert(strategy == PartitionPivotStrategy.midpoint ||
        strategy == PartitionPivotStrategy.rankGuided ||
        strategy == PartitionPivotStrategy.adaptive, "Unknown partition pivot strategy");
    import core.bitop: bsf, bsr;
    import mir.ndslice.sorting: partitionAt, pivotPartition;
    import mir.ndslice.topology: flattened;
    import mir.qualifier: lightScope;
    scope auto flat = workspace.lightScope.flattened;
    import mir.ndslice.iterator: FlattenedIterator;
    import mir.ndslice.slice: IteratorOf, slicedField;
    static if (is(IteratorOf!(typeof(flat)) == FlattenedIterator!Args, Args...))
    {
        // Address noncontiguous matrices from the original flattened view.
        // Subintervals then move an index rather than shifting the underlying
        // multidimensional iterator backward across row boundaries.
        scope auto values = flat.slicedField;
    }
    else alias values = flat;
    assert(rank < values.length);
    const word = rank / 64;
    const mask = 1UL << (rank % 64);
    if (word < established.length && (established[word] & mask)) return;
    size_t lo = 0, hi = values.length;
    if (established.length)
    {
        // Nearest fixed rank below the target, including the end of a short bitmap.
        size_t i = word < established.length ? word : established.length - 1;
        ulong bits = established[i];
        if (i == word) bits &= mask - 1;
        for (;;)
        {
            if (bits) { lo = i * 64 + bsr(bits) + 1; break; }
            if (!i) break;
            bits = established[--i];
        }
        if (word < established.length)
        {
            i = word;
            bits = established[i] & ~(mask | (mask - 1));
            for (;;)
            {
                if (bits) { hi = i * 64 + bsf(bits); break; }
                if (++i == established.length) break;
                bits = established[i];
            }
        }
    }
    assert(lo <= rank && rank < hi && hi <= values.length);

    static if (strategy == PartitionPivotStrategy.rankGuided)
    {
        import mir.stat.descriptive.univariate.internal.rank_partition: partitionAtRecorded;
        if (hi - lo > 1)
            partitionAtRecorded!less(values[lo .. hi], rank - lo, established, lo);
    }
    else
    {
        static if (strategy == PartitionPivotStrategy.adaptive)
        {
            // Choose once for this target, after restricting to cached boundaries.
            // Do not switch policy partway through the midpoint partition loop.
            const initialWidth = hi - lo;
            if (initialWidth >= 128 &&
                (rank - lo < initialWidth / 16 || hi - 1 - rank < initialWidth / 16))
            {
                import mir.stat.descriptive.univariate.internal.rank_partition: partitionAtRecorded;
                partitionAtRecorded!less(values[lo .. hi], rank - lo, established, lo);
                if (word < established.length) established[word] |= mask;
                return;
            }
        }
        // Limit the extra partition passes if midpoint values repeatedly make
        // poor pivots. At either endpoint partitionAt already uses a linear scan.
        size_t budget = hi - lo;
        while (hi - lo > 1 && rank != lo && rank != hi - 1)
        {
            const width = hi - lo;
            if (width > budget) break;
            const pivot = lo + pivotPartition!less(values[lo .. hi], width / 2);
            if (pivot / 64 < established.length)
                established[pivot / 64] |= 1UL << (pivot % 64);
            if (rank == pivot) return;
            if (rank < pivot) hi = pivot;
            else lo = pivot + 1;
            // Charge a full pass when a split keeps more than three quarters.
            // Balanced splits have geometrically decreasing total cost.
            if (hi - lo > width - width / 4) budget -= width;
        }
        if (hi - lo > 1)
            partitionAt!less(values[lo .. hi], rank - lo);
    }
    if (word < established.length) established[word] |= mask;
}

/// Strategy-first spelling; comparator-first calls remain supported.
package(mir.stat) void partitionAtWithCache(PartitionPivotStrategy strategy,
    alias less = "a < b", S)(S workspace, size_t rank, scope ulong[] established)
    @safe pure nothrow @nogc
{
    partitionAtWithCache!(less, strategy)(workspace, rank, established);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    int[5] x = [10, 0, 30, 20, 40];
    ulong[1] marks;
    partitionAtWithCache(x[].sliced, 2, marks[]);
    assert(x[2] == 20 && x[3] == 30);
    assert((marks[0] & 0b1100) == 0b1100);
    auto before = x;
    partitionAtWithCache(x[].sliced, 3, marks[]);
    assert(x == before && marks[0] == 0b1100);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.sorting: sort;
    int[143] input;
    foreach (i, ref v; input) v = cast(int)((19*i+13)%29);
    auto expected = input;
    sort!("a > b")(expected[].sliced);
    foreach (words; 0 .. 4)
    {
        auto x = input;
        ulong[3] bits;
        foreach (i; 0 .. x.length)
        {
            const rank = (i*17+64)%x.length;
            partitionAtWithCache!("a > b")(x[].sliced, rank, bits[0 .. words]);
            assert(x[rank] == expected[rank]);
            foreach (j; 0 .. x.length)
                if (j/64 < words && (bits[j/64] & (1UL << (j%64))))
                    assert(x[j] == expected[j]);
        }
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    int[10] data = [10, -1, 0, -1, 30, -1, 20, -1, 40, -1];
    auto values = data[].sliced.stride(2);
    ulong[1] established;
    partitionAtWithCache(values, 2, established[]);
    assert(values[2] == 20 && values[3] == 30);
    partitionAtWithCache(values, 0, established[]);
    partitionAtWithCache(values, 4, established[]);
    foreach (i; 0 .. 5)
        assert(data[2*i+1] == -1);
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.dynamic: transposed;
    int[12] data = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    auto matrix = data[].sliced(4, 3).transposed; // 3 rows of 4; strided
    ulong[1] marks;
    partitionAtWithCache(matrix, 7, marks[]);
    assert(matrix[1, 3] == 8 && (marks[0] & (1UL << 7)));
    partitionAtWithCache(matrix, 2, marks[]);
    assert(matrix[0, 2] == 3 && matrix[1, 3] == 8);
    bool[12] seen;
    foreach (v; data)
    {
        assert(v >= 1 && v <= 12 && !seen[v - 1]);
        seen[v - 1] = true;
    }
}

// Verify every remembered position, not just the requested one. Sample pivots
// must not be recorded until they have been expanded to the full interval.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.sorting: sort;
    alias P = PartitionPivotStrategy;
    static foreach (strategy; [P.midpoint, P.rankGuided, P.adaptive])
    foreach (n; [1, 2, 3, 12, 13, 64, 127, 128, 257])
    foreach (pattern; 0 .. 5)
    foreach (words; [0, 1, 5])
    {
        int[257] data, expected;
        foreach (i; 0 .. n)
            data[i] = pattern == 0 ? cast(int)((i * 173 + 19) % 257) :
                pattern == 1 ? 7 : pattern == 2 ? cast(int)(i % 5) :
                pattern == 3 ? cast(int)i : cast(int)(n - i);
        expected[] = data[];
        sort!("a > b")(expected[0 .. n].sliced);
        ulong[5] cache;
        foreach (j; 0 .. n)
        {
            // Visit every rank in a nonsequential order to exercise intermediate pivots.
            const rank = (j * 73 + n / 2) % n;
            partitionAtWithCache!("a > b", strategy)(data[0 .. n].sliced,
                rank, cache[0 .. words]);
            assert(data[rank] == expected[rank]);
            foreach (i; 0 .. n)
                if (i / 64 < words && (cache[i / 64] & (1UL << (i % 64))))
                    assert(data[i] == expected[i]);
        }
    }
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    alias P = PartitionPivotStrategy;
    static foreach (strategy; [P.midpoint, P.rankGuided, P.adaptive])
    {{
        int[5] data = [10, 0, 30, 20, 40];
        ulong[1] cache;
        partitionAtWithCache!strategy(data[].sliced, 2, cache[]);
        assert((cache[0] & 0b1100) == 0b1100);
        auto saved = data;
        const marks = cache;
        partitionAtWithCache!(strategy, "a < b")(data[].sliced, 3, cache[]);
        assert(data == saved && cache == marks);
    }}
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stride;
    import mir.ndslice.dynamic: transposed;
    alias P = PartitionPivotStrategy;
    static foreach (strategy; [P.midpoint, P.rankGuided, P.adaptive])
    {{
        int[512] storage;
        foreach (i; 0 .. 256)
        {
            storage[2 * i] = 255 - cast(int)i;
            storage[2 * i + 1] = -1;
        }
        auto strided = storage[].sliced.stride(2);
        ulong[4] cache;
        partitionAtWithCache!(strategy, (int a, int b) => a < b)(strided, 5, cache[]);
        partitionAtWithCache!strategy(strided, 200, cache[]);
        assert(strided[5] == 5 && strided[200] == 200);
        foreach (i; 0 .. 256) assert(storage[2 * i + 1] == -1);

        int[256] storage2;
        foreach (i, ref value; storage2) value = cast(int)((i * 173) % 256);
        auto matrix = storage2[].sliced(16, 16).transposed;
        cache[] = 0;
        partitionAtWithCache!(strategy, "a > b")(matrix, 5, cache[]);
        partitionAtWithCache!("a > b", strategy)(matrix, 200, cache[]);
        assert(matrix[0, 5] == 250 && matrix[12, 8] == 55);
    }}
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import std.meta: AliasSeq;
    import mir.ndslice.slice: sliced;
    alias P = PartitionPivotStrategy;
    static foreach (T; AliasSeq!(ulong, double))
    static foreach (strategy; [P.midpoint, P.rankGuided, P.adaptive])
    {{
        T[129] values;
        foreach (i, ref value; values)
        {
            static if (is(T == ulong)) value = (1UL << 63) + (i * 37) % 129;
            else value = cast(double)((i * 37) % 129) / 4 - 16;
        }
        ulong[3] cache;
        foreach (rank; [64, 0, 128, 3, 120, 17])
        {
            partitionAtWithCache!strategy(values[].sliced, rank, cache[]);
            static if (is(T == ulong)) assert(values[rank] == (1UL << 63) + rank);
            else assert(values[rank] == cast(double)rank / 4 - 16);
        }
    }}
}

// An empty bitmap changes storage only: the first search must perform the same
// selection as a fresh full bitmap, including adaptive choices near the ends.
version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    alias P = PartitionPivotStrategy;
    static foreach (strategy; [P.midpoint, P.rankGuided, P.adaptive])
    static foreach (less; ["a < b", "a > b"])
    foreach (rank; [0, 5, 128, 250, 256])
    {
        int[257] uncached, cached;
        foreach (i, ref value; uncached) value = cast(int)((i * 173 + 19) % 257);
        cached = uncached;
        ulong[5] cache;
        ulong[1] sentinel = [ulong.max];
        partitionAtWithCache!(strategy, less)(uncached[].sliced, rank, sentinel[0 .. 0]);
        partitionAtWithCache!(less, strategy)(cached[].sliced, rank, cache[]);
        assert(uncached == cached);
        assert(uncached[rank] == (less == "a < b" ? rank : 256 - rank));
        assert(sentinel[0] == ulong.max);
    }

    // Omitting the strategy and explicitly requesting midpoint agree without storage.
    int[5] implicit = [10, 0, 30, 20, 40];
    auto explicit = implicit;
    partitionAtWithCache(implicit[].sliced, 2, null);
    partitionAtWithCache!(P.midpoint)(explicit[].sliced, 2, null);
    assert(implicit == explicit);
}
