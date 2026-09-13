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
import std.traits: isDynamicArray, isNumeric;

package(mir.stat.descriptive.histogram) template supportsBinView(Storage, Axis)
{
    static if (isDynamicArray!Storage)
        private enum supportedStorage = true;
    else static if (isSlice!Storage)
        private enum supportedStorage = Storage.N == 1;
    else
        private enum supportedStorage = false;

    static if (supportedStorage && isAxis!Axis)
        enum supportsBinView = __traits(compiles, {
            const Storage storage;
            const Axis axis;
            auto readOnlyCounts = lightConst(storage);
            auto readOnlyAxis = lightConst(axis);
            auto description = (const typeof(readOnlyAxis)).init.bin(size_t.init);
            auto count = (const typeof(readOnlyCounts)).init[size_t.init];
            static assert(isNumeric!(DeepElementType!Storage));
        });
    else
        enum supportsBinView = false;
}
