---
name: d-compiler-investigation
description: Investigate suspected D compiler bugs, ICEs/assertions, misleading or context-dependent diagnostics, compiler-version regressions, DMD/LDC disagreements, optimization-dependent miscompiles, and cases needing a reduced reproducer. Includes DustMite reduction and D-native compiler diagnostics.
---

# D Compiler Investigation

Treat the compiler as a component that can fail. Do not distort otherwise-correct source code around suspicious behavior before classifying and reducing the problem.

## Capture and classify

Record the exact compiler and version, full command or DUB configuration, target, optimization/debug/unittest/BetterC flags, preview/transition/version identifiers, import paths, and the first useful diagnostic/assertion. Re-run the exact failing command before changing anything.

Classify the case as one of: ordinary source error; hidden speculative/template diagnostic; compiler regression; ICE/assertion/segfault; compiler or version disagreement; optimization-dependent miscompile/runtime failure; pathological compile-time behavior.

## Use D-native diagnostics first

Consult the installed compiler's help because flags vary by compiler and version. When supported, prefer targeted diagnostics over speculative source edits:

- speculative-error reporting for failures hidden by constraints or `__traits(compiles)`;
- template-instantiation listing for unexpected instantiation or overload selection;
- string-mixin output for diagnostics originating in generated source;
- GC-allocation reporting for unexpected `@nogc` failures;
- compile-time tracing for semantic, CTFE, template-instantiation, or codegen blowups;
- dependency or verbose output for module/import sensitivity;
- preview/transition toggles for language-change regressions;
- assembly or IR output for optimization and code-generation failures.

For DMD, useful options may include `-verrors=spec`, `-vtemplates=list-instances`, `-mixin`, `-vgc`, `-ftime-trace`, `-deps`, `-v`, `-preview=...`, `-transition=...`, and `-vasm`. Verify support with the installed compiler rather than assuming a particular version.

## Differential test

Change one variable at a time. When useful compare:

- current compiler with a previous known-good version;
- DMD with LDC;
- optimized with unoptimized builds;
- debug with release builds;
- relevant preview or transition settings;
- minimized compiler flags against the original invocation.

A difference narrows the fault; it does not by itself prove a compiler bug.

## Reduce early; use DustMite

Use DustMite when manual reduction stops producing a small reproducer quickly, especially when:

- unrelated edits or declaration/import order change the failure;
- the failure exists only in a large source context;
- the compiler crashes or asserts;
- a compiler upgrade introduced the issue;
- compilers or compiler versions disagree unexpectedly;
- the diagnostic points far from the apparent cause;
- optimization is required to reproduce incorrect behavior;
- the failure depends on templates, mixins, CTFE, or semantic-analysis ordering in a way that is difficult to isolate manually.

Create a self-contained reduction directory and a deterministic test command/oracle. **The oracle must preserve the specific bug, not merely compilation failure.**

Examples:

- ICE: preserve the assertion, crash signature, or other distinctive failure;
- diagnostic regression: match the relevant diagnostic while rejecting unrelated syntax or type errors;
- miscompile: compile and run, then test the incorrect observable result;
- compiler disagreement: encode the intended difference in compilation success or runtime result;
- compile-time regression: preserve a measurable threshold or timeout only when it reliably distinguishes the problem.

Verify the oracle manually before starting DustMite. After reduction, verify that the minimized case still demonstrates the original behavior under the original compiler configuration.

## Inspect the reduced case

Use the reduced reproducer to identify the language/compiler mechanism involved: overload resolution, template constraint evaluation, semantic ordering, CTFE, string mixins, qualifier conversion, lifetime analysis, attribute inference, optimizer/codegen behavior, or another subsystem.

Before concluding that the compiler is wrong, check whether the reduced source violates the D specification or relies on implementation-defined behavior.

## Finish the investigation

Choose the narrowest correct outcome:

- fix invalid or overly fragile source code;
- add a local, documented compiler workaround when necessary;
- report a compiler bug with the reduced reproducer and exact version/flags;
- bisect compiler versions or commits when regression identification is valuable;
- make no source change when the compiler diagnostic is correct.

If a workaround is necessary, keep it narrowly scoped, document the affected compiler/version range, and add a regression test. Remove the workaround when supported compilers no longer require it; retain useful regression coverage.
