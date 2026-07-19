# WGSL Writer V3 Execution Qualification Plan

## Decision

WGSL Writer V3 is Doppler's planned general shader-program experiment. Its
output unit is an executable package, not a WGSL string. This public document
owns the technical plan and claim boundary. It does not encode private portfolio
priority or authorize V3 implementation, training, productization, or
capability claims.

## Current Ground Truth

V2 established seed-confirmed development evidence for complete 1-D elementwise
f32 compute shaders under explicit interface contracts. It also established
exact selected-adapter completion parity between Transformers/PEFT and Doppler.

V2 did not establish arbitrary resource design, render pipelines, textures,
multi-pass execution, general shader authorship, or external promotion. Doppler
now contains the V3 executable-package schema, validator, planner, Chromium
executor, fixtures, and oracles. The four mechanics-reference packages passed
identity-bound AMD/Vulkan Chromium qualification on 2026-07-18; see
[`status/wgsl-writer-v3-reference-qualification-2026-07-18.md`](status/wgsl-writer-v3-reference-qualification-2026-07-18.md).
The executable capability corpus and its semantic oracles remain unmaterialized,
so training and every capability or product claim remain blocked.

Canonical predecessor evidence:

- [`status/wgsl-writer-v2-result-2026-07-14.md`](status/wgsl-writer-v2-result-2026-07-14.md)
- [`status/wgsl-writer-v2-result-2026-07-14.json`](status/wgsl-writer-v2-result-2026-07-14.json)

## Win Condition

Given a natural-language specification and an explicit host contract, a
promoted Doppler artifact returns:

1. complete WGSL modules;
2. resource and pipeline declarations;
3. dispatch or draw commands;
4. verified output artifacts; and
5. a receipt binding the request, package, execution, device, outputs, and
   cleanup result.

The claim remains limited to the shader families admitted by the promotion
contract. Compilation alone is never semantic success.

## Work Order

### 1. Freeze the executable package

Add one versioned schema that represents:

- WGSL modules and entry points;
- buffers, textures, samplers, bind groups, and initialization bytes;
- compute and render pipelines, overrides, vertex layouts, and target formats;
- compute dispatches, render passes, vertex draws, and indexed draws;
- declared readbacks, rendered outputs, and comparison policies; and
- explicit cleanup obligations.

The initial implementation should have one validator and one Chromium WebGPU
executor. Unknown fields, unsupported feature combinations, missing bindings,
resource-limit violations, and incomplete output declarations fail closed with
stable reason codes.

Implemented source boundaries:

- `src/config/schema/wgsl-author-package.schema.json`
- `tools/lib/wgsl-author-package.js`
- `tools/lib/wgsl-author-execution-plan.js`
- `tools/lib/wgsl-author-browser-executor.js`
- `tools/run-wgsl-author-v3-reference.js`
- `tests/tooling/wgsl-author-package.test.js`
- `tests/tooling/wgsl-author-execution-plan.test.js`
- `tests/tooling/wgsl-author-browser-executor.test.js`
- `tests/tooling/wgsl-author-reference.test.js`

Runtime policy remains JSON-owned; JavaScript owns orchestration and cleanup;
WGSL owns shader math.

### 2. Qualify four reference packages — complete

Run these hand-authored packages through Chromium WebGPU before model output is
allowed into the evaluator:

1. compute addition;
2. procedural rendering;
3. indexed rendering; and
4. compute-to-render composition.

Every module must compile. Every resource, bind group, and pipeline must
instantiate. Every command must execute. Compute readbacks must match a CPU
oracle under pinned tolerances. Rendered output must match a frozen raster
oracle with explicit format, comparison mask, and tolerance. Multi-pass state
must preserve declared dependencies.

Each receipt binds package and module hashes, browser build, adapter identity,
device capabilities and limits, execution plan, outputs, oracle revision,
validation result, and resource cleanup status.

### 3. Make the capability catalog executable

Materialize family-disjoint calibration, checkpoint-selection,
seed-confirmation, and promotion roles across:

- compute, render, and compute-to-render families;
- storage and uniform buffers;
- sampled and storage textures;
- pipeline overrides and workgroup variation;
- vertex and index inputs;
- output readback and raster comparison; and
- declared resource-limit failures.

Depth/stencil, blending, mip levels, multisampling, indirect commands, and
queries stay blocked until each has a package-schema extension, executor path,
reference fixture, semantic oracle, and failure tests.

### 4. Seal semantic verification

Blocking evidence for every eligible task includes:

- package validation, module compilation, and pipeline creation;
- dispatch or draw execution;
- CPU numerical or raster-oracle agreement;
- buffer bounds, immutable-input, and resource-contract checks;
- deterministic replay;
- task-valid metamorphic transformations;
- alternate shapes, workgroups, or draw dimensions; and
- historical regression execution.

A valid empty output is permitted only when the task contract declares it.
Malformed, missing, duplicate, or unknown outputs invalidate the package.

### 5. Run matched SAME-R lanes

Training begins only after the package executor, reference suite, populations,
metrics, and ranker are frozen. Compare matched lanes for:

- unchanged base generation;
- V2 seed-47 adapter initialization;
- complete-package SFT;
- curated execution-failure repair; and
- a count-matched control.

Freeze model identity, tokenizer, prompts, row and token budgets, optimizer,
adapter shape, update count, evaluation looks, seeds, and family partitions.
Select on semantic execution success first, followed by edit size, policy
compliance, compilation, and resource correctness. V2 may initialize a lane;
its result does not transfer into the V3 claim.

### 6. Prove the exact Doppler artifact

The selected candidate must pass:

1. PEFT-to-Doppler import and adapter activation;
2. tokenizer and prompt-token parity;
3. first-token logit and top-k checks;
4. selected-token and completion parity;
5. execution of the exact generated packages in Chromium WebGPU; and
6. exact hosted base, adapter, tokenizer, and package identity checks.

Promotion evaluates the hosted browser artifact, not a trainer checkpoint.
Receipts bind source revisions, corpus licenses, evaluator revisions, rejected
attempts, artifact URLs and hashes, device identity, and the one-use promotion
decision. Gamma records SAME-R training lineage; Doppler owns the V3 capability
contract, browser verifier, and promotion boundary.

### 7. Add a product surface only after promotion

The eventual command accepts a specification plus host contract and returns the
validated package, verified outputs, and receipt. Its public name is not frozen.
Adding it requires one normalized command contract with explicit browser and
Node behavior.

The surface must never:

- equate compilation with semantic correctness;
- invent omitted host bindings or resource policy;
- auto-apply an unverified package;
- silently downgrade unsupported features; or
- claim shader families outside the promoted capability envelope.

## Authorization Boundary

This plan defines the evidence required for WGSL Writer V3. Private work order
and portfolio priority are intentionally outside this public repository. No
external project artifact or result transfers into the V3 capability claim.
