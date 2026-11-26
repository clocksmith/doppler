# Doppler Developer Guides

Task-oriented guide map for extending Doppler.

Use this directory when the question is not "how does Doppler work?" but
"how do I add X without missing the contract checks?"

Reference lookup:
- General and JavaScript invariants: `docs/style/general-style-guide.md` and
  `docs/style/javascript-style-guide.md`.
- Shader contracts: `docs/style/wgsl-style-guide.md`.
- Command surfaces: `docs/style/command-interface-design-guide.md`.

Use the sections relevant to the question at hand. These guides are references,
not a reading gate before implementation.

Normative references still live elsewhere:
- `docs/style/*.md` define coding and contract rules
- `docs/config.md` defines config/runtime behavior
- `docs/conversion-runtime-contract.md` defines conversion-owned vs runtime-owned fields
- `docs/kernel-testing-design.md` defines kernel verification expectations

## How To Use This Directory

1. Start with the smallest guide that matches your change.
2. Identify the contracts and verification relevant to the changed behavior.
   Choose the implementation order from the actual code and observed failures.
3. If your change spans multiple areas, use a composite guide and then open the referenced atomic guides.

Every guide in this directory should answer the same four questions:
- what to touch
- in what order to touch it
- how to verify it
- what the common misses are

## Start Here If You Want To Add X

Atomic guides (numbered) describe one extension point; composite guides compose
several atomic guides.

| Goal | Start with | Kind | Blast radius |
| --- | --- | --- | --- |
| Understand ownership before changing config, kernels, registries, or hosted models | [config-source-of-truth.md](config-source-of-truth.md) | reference | source-of-truth map for conversion, runtime, harness, kernel, model, hosted registry, and evidence policy |
| Run a first verify, local conversion, or benchmark from the repository | [library-engineering-workflows.md](library-engineering-workflows.md) | reference | CLI commands only |
| Add a model end to end with research, debug, verify, and bench discipline | [model-onboarding-playbook.md](model-onboarding-playbook.md) | composite | conversion, runtime, debug, verify, and bench planning |
| Tune runtime behavior without code changes | [01-runtime-profile.md](01-runtime-profile.md) | atomic | JSON only |
| Put an existing chat format on a model | [02-assign-chat-template.md](02-assign-chat-template.md) | atomic | JSON only |
| Add a new model family contract | [03-model-family-config.md](03-model-family-config.md) | atomic | migration note: explicit conversion config, no family registry |
| Convert a checkpoint with an existing family | [04-conversion-config.md](04-conversion-config.md) | atomic | JSON only |
| Publish a verified artifact | [05-promote-model-artifact.md](05-promote-model-artifact.md) | atomic | catalog metadata, external storage, HF publication |
| Migrate model artifacts to source/weight/manifest identity | [historical artifact identity migration](https://github.com/clocksmith/doppler/blob/4b28d7831f12884c53e3a4c6e3470ca0dea9775f/docs/developer-guides/artifact-identity-migration-plan.md) | migration | retained migration evidence |
| Compose existing kernels differently | [06-kernel-path-config.md](06-kernel-path-config.md) | atomic | execution graph + transforms, no new WGSL |
| Export a portable Doe/Cerebras model program | [../integration/program-bundle.md](../integration/program-bundle.md) | reference | Program Bundle schema, exporter, transcript gate |
| Add a new config/manifest field | [07-manifest-runtime-field.md](07-manifest-runtime-field.md) | atomic | schema + merge + parser + tests |
| Add a new built-in chat formatter | [08-chat-template-formatter.md](08-chat-template-formatter.md) | atomic | JS + type declarations + tests |
| Add a new sampler or sampling knob | [09-sampling-strategy.md](09-sampling-strategy.md) | atomic | schema + runtime + tests |
| Add a new activation | [10-activation-implementation.md](10-activation-implementation.md) | atomic | WGSL + runtime + config + tests |
| Add a new kernel or kernel variant | [11-wgsl-kernel.md](11-wgsl-kernel.md) | atomic | WGSL + wrapper/selection + tests |
| Add a new top-level command | [12-command-surface.md](12-command-surface.md) | atomic | both runners + command contract + docs |
| Add a new attention mechanism | [13-attention-variant.md](13-attention-variant.md) | atomic | deep pipeline + kernels + tests (existing transformer pipeline only) |
| Add a new quantization format | [14-quantization-format.md](14-quantization-format.md) | atomic | converter + loader + kernels + tests |
| Add a new KV-cache layout | [15-kvcache-layout.md](15-kvcache-layout.md) | atomic | cache internals + attention integration |
| Optimize a correct GPU kernel path | [16-kernel-performance-optimization.md](16-kernel-performance-optimization.md) | atomic | WGSL + runtime profile + GPU tests + benchmark evidence |
| Add a new model family that still fits an existing pipeline | [composite-model-family.md](composite-model-family.md) | composite | cross-cutting within an existing pipeline family |
| Add a new pipeline family or new `modelType` ecosystem | [composite-pipeline-family.md](composite-pipeline-family.md) | composite | full vertical slice across a new pipeline family |

Feature notes for specific runtime paths:
[hyper-connection-mix-branch.md](hyper-connection-mix-branch.md),
[mapped-expert-upload-cache.md](mapped-expert-upload-cache.md),
[routed-expert-tracing.md](routed-expert-tracing.md),
[verified-piece-gather.md](verified-piece-gather.md).

## Composite Guides

These are not numbered because they compose multiple atomic guides.

### [model-onboarding-playbook.md](model-onboarding-playbook.md)

Use when adding a model and you need the full plan before implementation:

- source research
- conversion/runtime ownership split
- execution-graph strategy
- deterministic verify/debug ladder
- benchmark discipline
- promotion gate

This is the front door for model onboarding. After it narrows the scope, switch
to [composite-model-family.md](composite-model-family.md) or
[composite-pipeline-family.md](composite-pipeline-family.md) plus the atomic
guides that match the concrete work.

### [composite-model-family.md](composite-model-family.md)

Use when adding a new model family that still fits an existing pipeline family.

Usually composes:
- [02-assign-chat-template.md](02-assign-chat-template.md)
- [03-model-family-config.md](03-model-family-config.md)
- [04-conversion-config.md](04-conversion-config.md)
- [06-kernel-path-config.md](06-kernel-path-config.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [08-chat-template-formatter.md](08-chat-template-formatter.md)
- [10-activation-implementation.md](10-activation-implementation.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)

### [composite-pipeline-family.md](composite-pipeline-family.md)

Use when adding a fundamentally new pipeline family or new `modelType`
ecosystem.

Usually composes:
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [12-command-surface.md](12-command-surface.md)
- [15-kvcache-layout.md](15-kvcache-layout.md)

And also requires:
- pipeline registry wiring
- public API/export wiring
- package entrypoint updates
- API docs and export-inventory sync

## Required Shape For Each Guide

Use [_template.md](_template.md) when writing the individual guides.

Every guide should include:
- Goal
- When to use this guide
- Blast radius
- Required touch points
- Recommended order
- Verification
- Common misses
- Related guides
- Canonical references

## Cross-Cutting Rules

These apply to every guide in this directory:

- Fail fast, not silent. Unsupported capability or invalid config must throw an actionable error.
- JSON is the behavior contract. Runtime-visible decisions belong in manifests, config assets, and rule maps, not hidden JS branching.
- Run/record parity matters for kernel work. If a wrapper supports both paths, they must use the same resolved constants and behavior.
- No ad hoc runtime logging. Use the debug system in runtime code and keep direct `console.*` limited to allowed entrypoints.
- Contract updates are same-change work. If behavior changes, update schema, config, docs, and tests together.
