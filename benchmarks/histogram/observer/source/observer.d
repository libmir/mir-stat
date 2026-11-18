// License: Apache-2.0
module histogram_observer;

// Separate compilation keeps the count buffer observable to the caller.
// Do not build this library with LTO or retain the borrowed pointer.
extern(C) ulong mirStatHistogramObserve(const(uint)* counts, size_t length)
{
    ulong total;
    foreach (i; 0 .. length)
        total += counts[i];
    return total;
}
