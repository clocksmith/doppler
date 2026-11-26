---
name: doppler-kernel-reviewer
description: Review a named Doppler kernel and wrapper against the existing style guides when a concrete kernel path or diff is supplied.
---

# Doppler Kernel Review

## Prerequisites

- Identify the kernel JS wrapper, declaration, WGSL sources, and review diff.
- Open the guides in the "Kernels or shaders" row of the
  [AGENTS.md owner table](../../AGENTS.md#style-guides).

## Procedure

1. Trace config/rules to the execution-v1 dispatch identity and wrapper/WGSL files.
2. Run the mechanical checks:

   ```bash
   node skills/doppler-kernel-reviewer/scripts/lint-kernel.js <kernel.js>
   node skills/doppler-kernel-reviewer/scripts/lint-kernel.js <kernel.wgsl>
   node --check <kernel.js>
   ```

3. Apply `skills/doppler-kernel-reviewer/rules/checklist.md`.
4. Verify JSON owns selection, JS owns orchestration, and WGSL owns deterministic
   arithmetic/memory transforms.
5. Report findings by severity with exact file and line references.

## Validation

The lint and syntax commands pass or every failure is reported, every checklist item
has an evidence reference, and performance findings cite a comparable benchmark or
profiling receipt rather than inference from shader text.

## Stop Conditions

Stop if the kernel or governing execution identity is not named. Do not change style
guides, kernels, manifests, or runtime policy during a review unless the user separately
requests implementation.

## Outputs

A review report containing findings, evidence, unresolved questions, and the exact
commands run.

## Side Effects

Read-only unless the user separately authorizes fixes. Style-guide authorship is not
part of this skill.
