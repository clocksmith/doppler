---
name: doppler-kernel-reviewer
description: Review kernels against DOPPLER style guide and propose style guide updates.
---

# Kernel Reviewer

This skill helps you review DOPPLER kernels (WGSL, JS wrappers, .d.ts) against the official style guides. It also supports proposing changes to the style guides themselves.

## Workflows

### 1. Review Kernel

**Goal**: Verify that a specific kernel complies with `docs/style/*.md`.

**Steps**:
1. Read kernel files:
   - `src/gpu/kernels/<name>.js`
   - `src/gpu/kernels/<name>.d.ts`
   - WGSL sources referenced by the JS wrapper (typically under `src/gpu/kernels/`)
2. Run mechanical checks:
   - `node .claude/skills/doppler-kernel-reviewer/scripts/lint-kernel.js src/gpu/kernels/<name>.js`
   - `node .claude/skills/doppler-kernel-reviewer/scripts/lint-kernel.js src/gpu/kernels/<name>.wgsl`
3. Run syntax check for wrapper JS:
   - `node --check src/gpu/kernels/<name>.js`
4. Perform manual checklist review:
   - `.claude/skills/doppler-kernel-reviewer/rules/checklist.md`
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
