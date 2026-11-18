/++
Rank-guided selection with intermediate pivot recording.

Adapted from mir-algorithm 3.22.5, mir/ndslice/sorting.d, whose selection
implementation derives from Phobos std.algorithm.sorting.topN. Keep the sampling
and partition expansion logic aligned with that source when applying fixes.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Copyright: 2020 Ilia Ki, Kaleidic Associates Advisory Limited, Symmetry Investments
Authors: Andrei Alexandrescu (Phobos), Ilia Ki (API, rework, Mir adaptation), Mir Stat Authors
+/
module mir.stat.descriptive.univariate.internal.rank_partition;
import mir.ndslice.slice;
import mir.ndslice.sorting: medianOf, pivotPartitionImpl;
package(mir.stat) template partitionAtRecorded(alias less = "a < b")
{
    import mir.functional: naryFun;

    static if (__traits(isSame, naryFun!less, less))
    {
        /++
        Params:
            slice = n-dimensional slice
            nth = The index of the element that should be in sorted position after the
                function is finished.
            established = Bitmap belonging to the caller's full observation workspace.
            origin = Start of slice within that workspace.

        The caller records nth after return, including endpoint/one-element cases.
        +/
        void partitionAtRecorded(Iterator, size_t N, SliceKind kind)
            (Slice!(Iterator, N, kind) slice, size_t nth, scope ulong[] established, size_t origin) @trusted nothrow @nogc
        {
            import mir.qualifier: lightScope;
            import core.lifetime: move;
            import mir.ndslice.topology: flattened;

            assert(slice.elementCount > 0, "partitionAt: slice must have elementCount greater than 0");
            assert(nth < slice.elementCount, "partitionAt: nth must be less than the elementCount of the slice");

            // The caller supplies an in-bounds unresolved interval. Iterators
            // remain inside it; origin maps recorded positions into its bitmap.
            bool useSampling = true;
            auto flattenedSlice = slice.move.flattened;
            auto frontI = flattenedSlice._iterator.lightScope;
            auto lastI = frontI + (flattenedSlice.length - 1);
            partitionAtImpl!less(frontI, lastI, nth, useSampling, established, origin);
        }
    }
    else
        alias partitionAtRecorded = .partitionAtRecorded!(naryFun!less);
}


private @trusted pure nothrow @nogc
void partitionAtImpl(alias less, Iterator)(
    Iterator loI,
    Iterator hiI,
    size_t n,
    bool useSampling, scope ulong[] established = null, size_t origin = 0)
{
    assert(loI <= hiI, "partitionAtImpl: frontI must be less than or equal to lastI");

    import mir.utility: swapStars;
    import mir.functional: reverseArgs;

    auto firstI = loI;
    Iterator pivotI;
    size_t len;

    for (;;) {
        len = hiI - loI + 1;

        if (len <= 1) {
            break;
        }

        if (n == 0) {
            pivotI = loI;
            foreach (i; 1 .. len) {
                if (less(loI[i], *pivotI)) {
                    pivotI = loI + i;
                }
            }
            swapStars(loI + n, pivotI);
            break;
        }

        if (n + 1 == len) {
            pivotI = loI;
            foreach (i; 1 .. len) {
                if (reverseArgs!less(loI[i], *pivotI)) {
                    pivotI = loI + i;
                }
            }
            swapStars(loI + n, pivotI);
            break;
        }

        if (len <= 12) {
            pivotI = loI + len / 2;
            pivotPartitionImpl!less(loI, hiI, pivotI);
        } else if (n * 16 <= (len - 1) * 7) {
            pivotI = partitionAtPartitionOffMedian!(less, false)(loI, hiI, n, useSampling);
            // Quality check
            if (useSampling)
            {
                auto pivot = pivotI - loI;
                if (pivot < n)
                {
                    if (pivot * 4 < len)
                    {
                        useSampling = false;
                    }
                }
                else if ((len - pivot) * 8 < len * 3)
                {
                    useSampling = false;
                }
            }
        } else if (n * 16 >= (len - 1) * 9) {
            pivotI = partitionAtPartitionOffMedian!(less, true)(loI, hiI, n, useSampling);
            // Quality check
            if (useSampling)
            {
                auto pivot = pivotI - loI;
                if (pivot < n)
                {
                    if (pivot * 8 < len * 3)
                    {
                        useSampling = false;
                    }
                }
                else if ((len - pivot) * 4 < len)
                {
                    useSampling = false;
                }
            }
        } else {
            pivotI = partitionAtPartition!less(loI, hiI, n, useSampling);
            // Quality check
            if (useSampling) {
                auto pivot = pivotI - loI;
                if (pivot * 9 < len * 2 || pivot * 9 > len * 7)
                {
                    // Failed - abort sampling going forward
                    useSampling = false;
                }
            }
        }

        // Only full-interval pivots are globally established. Recursive sample
        // selection receives no bitmap: those positions may move during expansion.
        const fixedRank = origin + (pivotI - firstI);
        if (fixedRank / 64 < established.length)
            established[fixedRank / 64] |= 1UL << (fixedRank % 64);

        if (n < (pivotI - loI)) {
            hiI = pivotI - 1;
        } else if (n > (pivotI - loI)) {
            n -= (pivotI - loI + 1);
            loI = pivotI;
            ++loI;
        } else {
            break;
        }
    }
}


private @trusted pure nothrow @nogc
Iterator partitionAtPartition(alias less, Iterator)(
    ref Iterator frontI,
    ref Iterator lastI,
    size_t n,
    bool useSampling)
{
    size_t len = lastI - frontI + 1;

    assert(len >= 9 && n < len, "partitionAtPartition: length must be longer than 9 and n must be less than r.length");

    size_t ninth = len / 9;
    size_t pivot = ninth / 2;
    // Position subrange r[loI .. hiI] to have length equal to ninth and its upper
    // median r[loI .. hiI][$ / 2] in exactly the same place as the upper median
    // of the entire range r[$ / 2]. This is to improve behavior for searching
    // the median in already sorted ranges.
    auto loI = frontI;
    loI += len / 2 - pivot;
    auto hiI = loI;
    hiI += ninth;

    // We have either one straggler on the left, one on the right, or none.
    assert(loI - frontI <= lastI - hiI + 1 || lastI - hiI <= loI - frontI + 1, "partitionAtPartition: straggler check failed for loI, len, hiI");
    assert(loI - frontI >= ninth * 4, "partitionAtPartition: loI - frontI >= ninth * 4");
    assert((lastI + 1) - hiI >= ninth * 4, "partitionAtPartition: (lastI + 1) - hiI >= ninth * 4");

    // Partition in groups of 3, and the mid tertile again in groups of 3
    if (!useSampling) {
        auto loI_ = loI;
        loI_ -= ninth;
        auto hiI_ = hiI;
        hiI_ += ninth;
        p3!(less, Iterator)(frontI, lastI, loI_, hiI_);
    }
    p3!(less, Iterator)(frontI, lastI, loI, hiI);

    // Get the median of medians of medians
    // Map the full interval of n to the full interval of the ninth
    pivot = (n * (ninth - 1)) / (len - 1);
    if (hiI > loI) {
        auto hiI_minus = hiI;
        --hiI_minus;
        partitionAtImpl!less(loI, hiI_minus, pivot, useSampling);
    }

    auto pivotI = loI;
    pivotI += pivot;

    return expandPartition!less(frontI, lastI, loI, pivotI, hiI);
}


private @trusted pure nothrow @nogc
Iterator partitionAtPartitionOffMedian(alias less, bool leanRight, Iterator)(
    ref Iterator frontI,
    ref Iterator lastI,
    size_t n,
    bool useSampling)
{
    size_t len = lastI - frontI + 1;

    assert(len >= 12, "partitionAtPartitionOffMedian: len must be greater than 11");
    assert(n < len, "partitionAtPartitionOffMedian: n must be less than len");
    auto _4 = len / 4;
    auto leftLimitI = frontI;
    static if (leanRight)
        leftLimitI += 2 * _4;
    else
        leftLimitI += _4;
    // Partition in groups of 4, and the left quartile again in groups of 3
    if (!useSampling)
    {
        auto leftLimit_plus_4 = leftLimitI;
        leftLimit_plus_4 += _4;
        p4!(less, leanRight)(frontI, lastI, leftLimitI, leftLimit_plus_4);
    }
    auto _12 = _4 / 3;
    auto loI = leftLimitI;
    loI += _12;
    auto hiI = loI;
    hiI += _12;
    p3!less(frontI, lastI, loI, hiI);

    // Get the median of medians of medians
    // Map the full interval of n to the full interval of the ninth
    auto pivot = (n * (_12 - 1)) / (len - 1);
    if (hiI > loI) {
        auto hiI_minus = hiI;
        --hiI_minus;
        partitionAtImpl!less(loI, hiI_minus, pivot, useSampling);
    }
    auto pivotI = loI;
    pivotI += pivot;
    return expandPartition!less(frontI, lastI, loI, pivotI, hiI);
}


private @trusted
void p3(alias less, Iterator)(
    Iterator frontI,
    Iterator lastI,
    Iterator loI,
    Iterator hiI)
{
    assert(loI <= hiI && hiI <= lastI, "p3: loI must be less than or equal to hiI and hiI must be less than or equal to lastI");
    immutable diffI = hiI - loI;
    Iterator lo_loI;
    Iterator hi_loI;
    for (; loI < hiI; ++loI)
    {
        lo_loI = loI;
        lo_loI -= diffI;
        hi_loI = loI;
        hi_loI += diffI;
        assert(lo_loI >= frontI, "p3: lo_loI must be greater than or equal to frontI");
        assert(hi_loI <= lastI, "p3: hi_loI must be less than or equal to lastI");
        medianOf!less(lo_loI, loI, hi_loI);
    }
}


private @trusted
template p4(alias less, bool leanRight)
{
    void p4(Iterator)(
        Iterator frontI,
        Iterator lastI,
        Iterator loI,
        Iterator hiI)
    {
        assert(loI <= hiI && hiI <= lastI, "p4: loI must be less than or equal to hiI and hiI must be less than or equal to lastI");

        immutable diffI = hiI - loI;
        immutable diffI2 = diffI * 2;

        Iterator lo_loI;
        Iterator hi_loI;

        static if (leanRight)
            Iterator lo2_loI;
        else
            Iterator hi2_loI;

        for (; loI < hiI; ++loI)
        {
            lo_loI = loI - diffI;
            hi_loI = loI + diffI;

            assert(lo_loI >= frontI, "p4: lo_loI must be greater than or equal to frontI");
            assert(hi_loI <= lastI, "p4: hi_loI must be less than or equal to lastI");

            static if (leanRight) {
                lo2_loI = loI - diffI2;
                assert(lo2_loI >= frontI, "lo2_loI must be greater than or equal to frontI");
                medianOf!(less, leanRight)(lo2_loI, lo_loI, loI, hi_loI);
            } else {
                hi2_loI = loI + diffI2;
                assert(hi2_loI <= lastI, "hi2_loI must be less than or equal to lastI");
                medianOf!(less, leanRight)(lo_loI, loI, hi_loI, hi2_loI);
            }
        }
    }
}


private @trusted
template expandPartition(alias less)
{
    Iterator expandPartition(Iterator)(
        ref Iterator frontI,
        ref Iterator lastI,
        ref Iterator loI,
        ref Iterator pivotI,
        ref Iterator hiI)
    {
        import mir.algorithm.iteration: all;

        assert(frontI <= loI, "expandPartition: frontI must be less than or equal to loI");
        assert(loI <= pivotI, "expandPartition: loI must be less than or equal pivotI");
        assert(pivotI < hiI, "expandPartition: pivotI must be less than hiI");
        assert(hiI <= lastI, "expandPartition: hiI must be less than or equal to lastI");

        foreach(x; loI .. (pivotI + 1))
            assert(!less(*pivotI, *x), "expandPartition: loI .. (pivotI + 1) failed test");
        foreach(x; (pivotI + 1) .. hiI)
            assert(!less(*x, *pivotI), "expandPartition: (pivotI + 1) .. hiI failed test");

        import mir.utility: swapStars;
        import mir.algorithm.iteration: all;
        // We work with closed intervals!
        --hiI;

        auto leftI = frontI;
        auto rightI = lastI;
        loop: for (;; ++leftI, --rightI)
        {
            for (;; ++leftI)
            {
                if (leftI == loI) break loop;
                if (!less(*leftI, *pivotI)) break;
            }
            for (;; --rightI)
            {
                if (rightI == hiI) break loop;
                if (!less(*pivotI, *rightI)) break;
            }
            swapStars(leftI, rightI);
        }

        foreach(x; loI .. (pivotI + 1))
            assert(!less(*pivotI, *x), "expandPartition: loI .. (pivotI + 1) failed less than test");
        foreach(x; (pivotI + 1) .. (hiI + 1))
            assert(!less(*x, *pivotI), "expandPartition: (pivotI + 1) .. (hiI + 1) failed less than test");
        foreach(x; frontI .. leftI)
            assert(!less(*pivotI, *x), "expandPartition: frontI .. leftI failed less than test");
        foreach(x; (rightI + 1) .. (lastI + 1))
            assert(!less(*x, *pivotI), "expandPartition: (rightI + 1) .. (lastI + 1) failed less than test");

        auto oldPivotI = pivotI;

        if (leftI < loI)
        {
            // First loop: spend r[loI .. pivot]
            for (; loI < pivotI; ++leftI)
            {
                if (leftI == loI) goto done;
                if (!less(*oldPivotI, *leftI)) continue;
                --pivotI;
                assert(!less(*oldPivotI, *pivotI), "expandPartition: less check failed");
                swapStars(leftI, pivotI);
            }
            // Second loop: make leftI and pivot meet
            for (;; ++leftI)
            {
                if (leftI == pivotI) goto done;
                if (!less(*oldPivotI, *leftI)) continue;
                for (;;)
                {
                    if (leftI == pivotI) goto done;
                    --pivotI;
                    if (less(*pivotI, *oldPivotI))
                    {
                        swapStars(leftI, pivotI);
                        break;
                    }
                }
            }
        }

        // First loop: spend r[lo .. pivot]
        for (; hiI != pivotI; --rightI)
        {
            if (rightI == hiI) goto done;
            if (!less(*rightI, *oldPivotI)) continue;
            ++pivotI;
            assert(!less(*pivotI, *oldPivotI), "expandPartition: less check failed");
            swapStars(rightI, pivotI);
        }
        // Second loop: make leftI and pivotI meet
        for (; rightI > pivotI; --rightI)
        {
            if (!less(*rightI, *oldPivotI)) continue;
            while (rightI > pivotI)
            {
                ++pivotI;
                if (less(*oldPivotI, *pivotI))
                {
                    swapStars(rightI, pivotI);
                    break;
                }
            }
        }

    done:
        swapStars(oldPivotI, pivotI);


        foreach(x; frontI .. (pivotI + 1))
            assert(!less(*pivotI, *x), "expandPartition: frontI .. (pivotI + 1) failed test");
        foreach(x; (pivotI + 1) .. (lastI + 1))
            assert(!less(*x, *pivotI), "expandPartition: (pivotI + 1) .. (lastI + 1) failed test");
        return pivotI;
    }
}
