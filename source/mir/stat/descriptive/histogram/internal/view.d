/++
Internal helpers for histogram view constraints.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.descriptive.histogram.internal.view;

import mir.ndslice.slice: isSlice;
import mir.primitives: DeepElementType;
import mir.qualifier: lightConst;
import mir.stat.descriptive.histogram.traits: isAxis;
import std.traits: isArray, isDynamicArray, isNumeric;
import std.meta: allSatisfy;

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
        private enum supportedStorage = JointArrayInfo!Storage.rank == Axis.length &&
            isNumeric!(JointArrayInfo!Storage.Element);
    else static if (isSlice!Storage)
        private enum supportedStorage = Storage.N == Axis.length &&
            isNumeric!(DeepElementType!Storage);
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
