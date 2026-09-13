/++
Internal detection of compiler escape checking for borrowed views.

License: $(HTTP www.apache.org/licenses/LICENSE-2.0, Apache-2.0)
Authors: John Michael Hall
Copyright: 2026 Mir Stat Authors.
+/
module mir.stat.internal.borrow;

// Detect enforcement of the specific escape rule required by borrowed views.
// Test compiler behavior rather than a user-defined version or compiler release:
// DIP1000 may eventually become the default. Keep detection isolated here.
private struct BorrowEscapeProbe
{
    int value;
    int* borrow() return @safe { return &value; }
}

package(mir.stat) enum hasBorrowEscapeChecking = !__traits(compiles, () @safe {
    BorrowEscapeProbe owner;
    auto pointer = owner.borrow();
    return pointer;
});

// Called only when escape checking is unavailable, making the caller infer
// @system. This is deliberately not @trusted: unchecked borrowing needs review.
package(mir.stat) void uncheckedBorrow() @system pure nothrow @nogc {}
