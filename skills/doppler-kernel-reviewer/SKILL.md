---
name: doppler-kernel-reviewer
description: Review kernels against DOPPLER style guide and propose style guide updates.
---

# Kernel Reviewer

This skill helps you review DOPPLER kernels (WGSL, JS wrappers, .d.ts) against the official style guides. It also supports proposing changes to the style guides themselves.

## Mandatory Style Guides

Read these before non-trivial kernel review or kernel-wrapper edits:
- `docs/style/general-style-guide.md`
- `docs/style/javascript-style-guide.md`
- `docs/style/wgsl-style-guide.md`

Also read `docs/style/config-style-guide.md` when the review touches rule selection, dtype policy, or kernel-path metadata.

## Developer Guide Routing

When the review turns into implementation guidance, also open:
- `docs/developer-guides/README.md`

Common routes:
- activation-specific implementation work: `docs/developer-guides/10-activation-implementation.md`
- new kernel or kernel-variant work: `docs/developer-guides/11-wgsl-kernel.md`
- attention-kernel changes: `docs/developer-guides/13-attention-variant.md`
- cache/layout changes that require kernel compatibility work: `docs/developer-guides/15-kvcache-layout.md`

## Plane Contract (Review Invariant)

See also: `docs/style/general-style-guide.md#invariants-quick-reference` (execution plane contract).

- JSON rules + config assets own kernel selection and feature toggles.
- Execution-v1 manifests own explicit dispatch identities; reviewers should reject implicit legacy fallback assumptions.
- JS wrappers own orchestration (validation, binding/pipeline setup, dispatch lifecycle).
- WGSL owns deterministic arithmetic and memory transforms only.
- A review must flag any ad-hoc, implicit behavior branching in JS or WGSL that bypasses rule assets/config resolution.

## Workflows

### 1. Review Kernel

**Goal**: Verify that a specific kernel complies with `docs/style/*.md`.

**Steps**:
1. Read kernel files:
   - `src/gpu/kernels/<name>.js`
   - `src/gpu/kernels/<name>.d.ts`
   - WGSL sources referenced by the JS wrapper (typically under `src/gpu/kernels/`)
2. Run mechanical checks:
   - `node skills/doppler-kernel-reviewer/scripts/lint-kernel.js src/gpu/kernels/<name>.js`
   - `node skills/doppler-kernel-reviewer/scripts/lint-kernel.js src/gpu/kernels/<name>.wgsl`
3. Run syntax check for wrapper JS:
   - `node --check src/gpu/kernels/<name>.js`
4. Perform manual checklist review:
   - `skills/doppler-kernel-reviewer/rules/checklist.md`
   - `docs/style/general-style-guide.md`
   - `docs/style/javascript-style-guide.md`
   - `docs/style/wgsl-style-guide.md`
5. Report findings ordered by severity with concrete file references.

### 2. Update Style Guide

**Goal**: Propose a change to the style guides to improve clarity or support new patterns.

**Steps**:
1. Read the relevant guide (for WGSL: `docs/style/wgsl-style-guide.md`).
2. Propose a concrete diff in the style guide file.
3. Justify the change based on runtime correctness, portability, or maintainability.
4. If the guide changes, update checklist/lint heuristics in this skill to stay aligned.

## Resources
- `rules/checklist.md`: Condensed style rules.
- `scripts/lint-kernel.js`: Automated regex checks.
- `docs/developer-guides/README.md`: extension playbook routing.
