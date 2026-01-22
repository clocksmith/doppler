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
1.  **Read Files**: Use `view_file` to read the kernel's source files. Usually:
    -   `src/gpu/kernels/<name>.js`
    -   `src/gpu/kernels/<name>.d.ts`
    -   `src/gpu/kernels/<name>.wgsl` (and variants)
2.  **Run Linter**: Run `node .claude/skills/doppler-kernel-reviewer/scripts/lint-kernel.js <kernel-js-path>` to check for common mechanical errors (padding, JSDoc, etc.).
3.  **Manual Check**: Verify the code against `.claude/skills/doppler-kernel-reviewer/rules/checklist.md`.
4.  **Report**: Provide a list of violations (if any) or a confirmation of compliance.

### 2. Update Style Guide

**Goal**: Propose a change to the style guides to improve clarity or support new patterns.

**Steps**:
1.  **Read Guide**: Read the relevant style guide (e.g., `docs/style/WGSL_STYLE_GUIDE.md`).
2.  **Propose Diff**: Create a diff block showing the proposed change.
3.  **Justify**: Explain *why* the change is improved (e.g., "Improves alignment safety" or "Reduces boilerplate").

## Resources
- `rules/checklist.md`: Condensed style rules.
- `scripts/lint-kernel.js`: Automated regex checks.
