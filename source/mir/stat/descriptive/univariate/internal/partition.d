/++
Cached rank selection on a caller-owned Mir slice.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
+/
module mir.stat.descriptive.univariate.internal.partition;

/++
Rank is a zero-based position in the flattened slice, as for partitionAt.
Record the requested rank and the pivots established by this helper's partition
loop. When it delegates an interval to partitionAt, only the requested rank is
recorded from that call. Later calls leave every marked position untouched.

Provide a zeroed bitmap with one bit per element (round up to whole ulong
words) to remember every rank. A shorter bitmap remembers only the ranks it
can represent; an empty bitmap delegates to ordinary partitionAt. Bitmap
storage must not overlap the slice elements. Reuse the bitmap only with the
same workspace, flattened order, and comparison. Clear it after independently
modifying the data.
+/
package(mir.stat) void partitionAtCache(alias less = "a < b", S)(S workspace, size_t rank,
    scope ulong[] established) @safe pure nothrow @nogc
{
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
    if (!established.length)
    {
        partitionAt!less(values, rank);
        return;
    }

    const word = rank / 64;
    const mask = 1UL << (rank % 64);
    if (word < established.length && (established[word] & mask)) return;
    size_t lo = 0, hi = values.length;
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
    assert(lo <= rank && rank < hi && hi <= values.length);

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
    if (word < established.length) established[word] |= mask;
}

version(mir_stat_test)
@safe pure nothrow @nogc unittest
{
    import mir.ndslice.slice: sliced;
    int[5] x = [10, 0, 30, 20, 40];
    ulong[1] marks;
    partitionAtCache(x[].sliced, 2, marks[]);
    assert(x[2] == 20 && x[3] == 30);
    assert((marks[0] & 0b1100) == 0b1100);
    auto before = x;
    partitionAtCache(x[].sliced, 3, marks[]);
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
            partitionAtCache!("a > b")(x[].sliced, rank, bits[0 .. words]);
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
    partitionAtCache(values, 2, established[]);
    assert(values[2] == 20 && values[3] == 30);
    partitionAtCache(values, 0, established[]);
    partitionAtCache(values, 4, established[]);
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
    partitionAtCache(matrix, 7, marks[]);
    assert(matrix[1, 3] == 8 && (marks[0] & (1UL << 7)));
    partitionAtCache(matrix, 2, marks[]);
    assert(matrix[0, 2] == 3 && matrix[1, 3] == 8);
    bool[12] seen;
    foreach (v; data)
    {
        assert(v >= 1 && v <= 12 && !seen[v - 1]);
        seen[v - 1] = true;
    }
}
