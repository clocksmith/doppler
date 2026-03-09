# Doppler Developer Guides

Task-oriented guide map for extending Doppler.

Use this directory when the question is not "how does Doppler work?" but
"how do I add X without missing the contract checks?"

Prerequisite:
- Read `docs/style/general-style-guide.md` and `docs/style/javascript-style-guide.md` before any non-trivial edit.
- For WGSL work, also read `docs/style/wgsl-style-guide.md`.
- For command-surface work, also read `docs/style/command-interface-design-guide.md`.

Normative references still live elsewhere:
- `docs/style/*.md` define coding and contract rules
- `docs/config.md` defines config/runtime behavior
- `docs/conversion-runtime-contract.md` defines conversion-owned vs runtime-owned fields
- `docs/kernel-testing-design.md` defines kernel verification expectations

## How To Use This Directory

1. Start with the smallest guide that matches your change.
2. Follow its required touch points and verification steps.
3. If your change spans multiple areas, use a composite guide and then open the referenced atomic guides.

Every guide in this directory should answer the same four questions:
- what to touch
- in what order to touch it
- how to verify it
- what the common misses are

## Start Here If You Want To Add X

| Goal | Start with | Notes |
| --- | --- | --- |
| Tune runtime behavior without code changes | [01-runtime-preset.md](01-runtime-preset.md) | JSON-only runtime preset work |
| Put an existing chat format on a model | [02-assign-chat-template.md](02-assign-chat-template.md) | Uses an existing built-in formatter |
| Add a new model preset | [03-model-preset.md](03-model-preset.md) | Already-supported runtime behavior only |
| Convert a checkpoint with an existing family | [04-conversion-config.md](04-conversion-config.md) | Existing preset + existing kernel paths |
| Publish a verified artifact | [05-promote-model-artifact.md](05-promote-model-artifact.md) | Curated metadata + external storage + HF |
| Compose existing kernels differently | [06-kernel-path-preset.md](06-kernel-path-preset.md) | New execution identity, no new WGSL |
| Add a new config/manifest field | [07-manifest-runtime-field.md](07-manifest-runtime-field.md) | Schema + merge + parser + tests |
| Add a new built-in chat formatter | [08-chat-template-formatter.md](08-chat-template-formatter.md) | New formatter function and registry key |
| Add a new sampler or sampling knob | [09-sampling-strategy.md](09-sampling-strategy.md) | Sampling pipeline work |
| Add a new activation | [10-activation-implementation.md](10-activation-implementation.md) | WGSL + runtime wiring + config |
| Add a new kernel or kernel variant | [11-wgsl-kernel.md](11-wgsl-kernel.md) | Selection/wrapper/testing work |
| Add a new top-level command | [12-command-surface.md](12-command-surface.md) | Browser/Node parity contract |
| Add a new attention mechanism | [13-attention-variant.md](13-attention-variant.md) | Existing transformer pipeline only |
| Add a new quantization format | [14-quantization-format.md](14-quantization-format.md) | Converter + loader + kernels |
| Add a new KV-cache layout | [15-kvcache-layout.md](15-kvcache-layout.md) | Deep cache and attention integration |
| Add a new model family that still fits an existing pipeline | [composite-model-family.md](composite-model-family.md) | Composes several atomic guides |
| Add a new pipeline family or new `modelType` ecosystem | [composite-pipeline-family.md](composite-pipeline-family.md) | Largest extension path |

## Journey Map

| Guide | Kind | Blast Radius |
| --- | --- | --- |
| [01-runtime-preset.md](01-runtime-preset.md) | atomic | JSON only |
| [02-assign-chat-template.md](02-assign-chat-template.md) | atomic | JSON only |
| [03-model-preset.md](03-model-preset.md) | atomic | JSON + loader registry |
| [04-conversion-config.md](04-conversion-config.md) | atomic | JSON only |
| [05-promote-model-artifact.md](05-promote-model-artifact.md) | atomic | metadata + publication workflow |
| [06-kernel-path-preset.md](06-kernel-path-preset.md) | atomic | JSON + registry |
| [07-manifest-runtime-field.md](07-manifest-runtime-field.md) | atomic | schema + merge + parser + tests |
| [08-chat-template-formatter.md](08-chat-template-formatter.md) | atomic | JS + type declarations + tests |
| [09-sampling-strategy.md](09-sampling-strategy.md) | atomic | schema + runtime + tests |
| [10-activation-implementation.md](10-activation-implementation.md) | atomic | WGSL + runtime + config + tests |
| [11-wgsl-kernel.md](11-wgsl-kernel.md) | atomic | WGSL + wrapper/selection + tests |
| [12-command-surface.md](12-command-surface.md) | atomic | both runners + command contract + docs |
| [13-attention-variant.md](13-attention-variant.md) | atomic | deep pipeline + kernels + tests |
| [14-quantization-format.md](14-quantization-format.md) | atomic | converter + loader + kernels + tests |
| [15-kvcache-layout.md](15-kvcache-layout.md) | atomic | cache internals + attention integration |
| [composite-model-family.md](composite-model-family.md) | composite | cross-cutting within an existing pipeline family |
| [composite-pipeline-family.md](composite-pipeline-family.md) | composite | full vertical slice across a new pipeline family |

## Atomic Guides

These are the numbered guides. Each should describe one extension point.

1. [01-runtime-preset.md](01-runtime-preset.md)
   Add a runtime preset without changing runtime code.
2. [02-assign-chat-template.md](02-assign-chat-template.md)
   Assign an existing built-in chat template to a model preset.
3. [03-model-preset.md](03-model-preset.md)
   Add a model preset for an already-supported schema/behavior set.
4. [04-conversion-config.md](04-conversion-config.md)
   Add a conversion config for an existing supported family.
5. [05-promote-model-artifact.md](05-promote-model-artifact.md)
   Promote a locally verified RDRR artifact into curated metadata and hosted publication.
6. [06-kernel-path-preset.md](06-kernel-path-preset.md)
   Add or register a kernel-path preset using existing kernels.
7. [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
   Add a new manifest/runtime contract field.
8. [08-chat-template-formatter.md](08-chat-template-formatter.md)
   Add a new built-in chat formatter type.
9. [09-sampling-strategy.md](09-sampling-strategy.md)
   Add a new sampling strategy or sampling knob.
10. [10-activation-implementation.md](10-activation-implementation.md)
    Add a new activation implementation.
11. [11-wgsl-kernel.md](11-wgsl-kernel.md)
    Add a new WGSL kernel or kernel variant.
12. [12-command-surface.md](12-command-surface.md)
    Add a new top-level Doppler command.
13. [13-attention-variant.md](13-attention-variant.md)
    Add a new attention variant inside the existing transformer pipeline.
14. [14-quantization-format.md](14-quantization-format.md)
    Add a new quantization format end to end.
15. [15-kvcache-layout.md](15-kvcache-layout.md)
    Add a new KV-cache layout or inference memory-management strategy.

## Composite Guides

These are not numbered because they compose multiple atomic guides.

### [composite-model-family.md](composite-model-family.md)

Use when adding a new model family that still fits an existing pipeline family.

Usually composes:
- [02-assign-chat-template.md](02-assign-chat-template.md)
- [03-model-preset.md](03-model-preset.md)
- [04-conversion-config.md](04-conversion-config.md)
- [06-kernel-path-preset.md](06-kernel-path-preset.md)
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
- JSON is the behavior contract. Runtime-visible decisions belong in manifests, presets, and rule maps, not hidden JS branching.
- Run/record parity matters for kernel work. If a wrapper supports both paths, they must use the same resolved constants and behavior.
- No ad hoc runtime logging. Use the debug system in runtime code and keep direct `console.*` limited to allowed entrypoints.
- Contract updates are same-change work. If behavior changes, update schema, config, docs, and tests together.

## Suggested Build Order

Write these first:
1. `07-manifest-runtime-field.md`
2. `11-wgsl-kernel.md`
3. `12-command-surface.md`
4. `03-model-preset.md`
5. `04-conversion-config.md`
6. `06-kernel-path-preset.md`

Then add:
- `08-chat-template-formatter.md`
- `09-sampling-strategy.md`
- `10-activation-implementation.md`
- `13-attention-variant.md`
- `14-quantization-format.md`
- `15-kvcache-layout.md`

Composite guides should come last, once the atomic guides exist.
