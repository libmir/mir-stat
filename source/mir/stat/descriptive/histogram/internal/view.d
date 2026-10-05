/++
Internal helpers for histogram views and row traversal.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.view;

import mir.ndslice.slice: isSlice;
import mir.qualifier: lightConst;
import mir.stat.descriptive.histogram.traits: isAxis;
import std.traits: isArray, isDynamicArray, hasElaborateDestructor;
import std.meta: allSatisfy;

// DMD 2.111/2.112 can return a stale stack address when a const RC slice row
// is passed directly to a recursive call, e.g. total(storage[i]). Naming a
// scoped row first avoids the miscompile. The expression compiles on affected
// versions, so __traits(compiles) cannot detect this runtime code-generation bug.
// This is a DMD backend workaround, not a frontend-version requirement for LDC
// or GDC. Keep the original value-passing path elsewhere: naming the row changes
// auto ref deduction and can prevent vectorization of histogram merges.
version (DigitalMars)
    private enum needsConstRowWorkaround = __VERSION__ >= 2111 && __VERSION__ < 2113;
else
    private enum needsConstRowWorkaround = false;

package(mir.stat.descriptive.histogram) enum needsScopedSliceRow(S) =
    needsConstRowWorkaround && isSlice!S && hasElaborateDestructor!S;

package(mir.stat.descriptive.histogram) template JointArrayInfo(Storage)
{
    static if (isArray!Storage)
    {
        alias Child = JointArrayInfo!(typeof(Storage.init[0]));
        enum rank = 1 + Child.rank;
        alias Element = Child.Element;
    }
    else
    {
        enum rank = 0;
        alias Element = Storage;
    }
}

private template supportsAxisBin(Axis)
{
    enum supportsAxisBin = __traits(compiles, {
        const Axis axis;
        auto readOnlyAxis = lightConst(axis);
        auto description = (const typeof(readOnlyAxis)).init.bin(size_t.init);
    });
}

package(mir.stat.descriptive.histogram) template supportsBinView(Storage, Axis...)
{
    static if (isDynamicArray!Storage)
        private enum supportedStorage = JointArrayInfo!Storage.rank == Axis.length;
    else static if (isSlice!Storage)
        private enum supportedStorage = Storage.N == Axis.length;
    else
        private enum supportedStorage = false;

    static if (supportedStorage && Axis.length > 0 && allSatisfy!(isAxis, Axis))
        enum supportsBinView = allSatisfy!(supportsAxisBin, Axis) && __traits(compiles, {
            const Storage storage;
            auto readOnlyCounts = lightConst(storage);
        });
    else
        enum supportsBinView = false;
}
