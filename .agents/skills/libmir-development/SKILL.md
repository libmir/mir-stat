---
name: libmir-development
description: Routine implementation, review, refactoring, testing, and D-language debugging in libmir. Use for normal libmir work involving templates, ndslice, qualifiers, attributes, BetterC, API compatibility, or repository conventions.
---

# Libmir Development

Work from the checkout, not assumptions. Inspect the target module and nearby tests before changing code. Reuse findings from the current task; inspect analogous Mir code, `dub.*`, and CI/compiler configurations only when relevant or when they have changed.

Check whether another available skill is relevant to the problem and use it when helpful. No companion skill is required; reuse findings and completed checks when switching.

## Preserve Mir contracts

Prefer the smallest change that preserves existing public behavior and supported configurations. Treat these as deliberate unless source/tests show otherwise:

- `@safe`, `@nogc`, `nothrow`, `pure`, BetterC/runtime independence;
- public names, overload availability, template constraints, return/reference semantics;
- lazy/view semantics and zero-copy topology operations;
- generic support for qualifiers, ranks, layouts, and strides;
- supported compiler/configuration matrix.

Do not weaken attributes, broaden constraints, add allocations, materialize views, or specialize away generic support merely to make a case compile.

## D-specific debugging

For template/overload failures, determine the concrete instantiation before editing:

1. exact template arguments and qualifiers;
2. lvalue/rvalue and `ref`/`auto ref` behavior;
3. selected constraint / `static if` branch;
4. viable overloads and conversion ranking;
5. inferred/required attributes and lifetime rules.

Useful temporary probes include `pragma(msg, ...)`, `static assert(is(...))`, and `__traits(compiles, ...)`. When speculative compilation hides the real error, compile the expression directly or use the installed DMD's speculative-error diagnostic option if supported. For string mixins, dump the expansion with the compiler's supported mixin-output option. Remove temporary probes before completion unless they become regression tests.

For a constraint change, test the affected acceptance dimensions: intended accepted/rejected types and relevant qualifiers, lvalues/rvalues, or neighboring overloads. Do not mechanically exercise every qualifier or layout. Fix the narrow semantic cause rather than making the constraint unconditional.

## Safety, lifetime, and allocation

At pointer/FFI or `@trusted` boundaries, explicitly establish ownership, extent, alignment, mutability, and lifetime. Keep `@trusted` regions minimal and make the safety invariant evident from surrounding code/tests.

If `@nogc` unexpectedly fails, use compiler allocation diagnostics when supported instead of guessing. Check hidden allocations from closures/delegates, temporary dynamic arrays, formatting/error paths, conversions, and convenience APIs.

Do not silence a `scope`/escape/safety error until you understand whether the compiler is proving a real lifetime violation. Preserve `return scope`/reference behavior when it is part of the API.

## ndslice

For slice/view changes, reason in terms of rank, shape, strides/layout, aliasing, and lifetime. Do not assume contiguity unless the type/constraint establishes it. A transpose/reshape/topology operation should remain a view when that is the existing contract. Assignment/copy logic must respect the repository's overlap semantics.

Fast paths may require contiguous/layout-specific preconditions, but keep a correct generic path when the public algorithm accepts strided/general slices.

## Verification

Use the repository's actual commands/configurations. Start with the smallest relevant unittest/compile test, then broaden checks according to the change and explicit user requirements. Test the affected compiler/configuration and another supported compiler for template, attribute, safety, or frontend-sensitive changes. Reuse completed checks unless subsequent changes invalidate them; do not mechanically run every configuration.

For a suspected compiler regression, ICE, context-dependent diagnostic, DMD/LDC disagreement, or optimization miscompile, capture the exact compiler version and failing command, then reduce the case while preserving the specific failure. Distinguish invalid source from a compiler fault before adding a workaround.
