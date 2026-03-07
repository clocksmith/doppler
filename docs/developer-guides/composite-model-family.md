# Add a New Model Family Within an Existing Pipeline

## Goal

Onboard a new model family that still fits an existing Doppler pipeline family.

## When To Use This Guide

- The model still belongs to an existing runtime family such as the transformer text pipeline.
- You expect to compose existing extension points rather than invent a whole new pipeline ecosystem.

## Blast Radius

- Composite, cross-cutting work inside an existing pipeline family

## Required Touch Points

- A new model preset
- At least one conversion config
- Usually a chat-template decision and one or more kernel-path decisions
- Loader, runtime, or schema updates only if the family actually requires them
- End-to-end verification and optional promotion work

## Recommended Order

1. Confirm the family really fits an existing pipeline. If not, stop and use [composite-pipeline-family.md](composite-pipeline-family.md).
2. Start with [03-model-preset.md](03-model-preset.md).
3. Add the initial conversion recipe with [04-conversion-config.md](04-conversion-config.md).
4. Choose either [02-assign-chat-template.md](02-assign-chat-template.md) or [08-chat-template-formatter.md](08-chat-template-formatter.md).
5. Add [06-kernel-path-preset.md](06-kernel-path-preset.md) if the family needs different execution identity.
6. Add [07-manifest-runtime-field.md](07-manifest-runtime-field.md), [10-activation-implementation.md](10-activation-implementation.md), [11-wgsl-kernel.md](11-wgsl-kernel.md), [13-attention-variant.md](13-attention-variant.md), or [14-quantization-format.md](14-quantization-format.md) only when the family genuinely needs them.
7. Run convert, verify, debug, and browser verification before any promotion.
8. If the artifact is going to be reused or published, finish with [05-promote-model-artifact.md](05-promote-model-artifact.md).

## Verification

- `npm run onboarding:check:strict`
- Run a full convert plus verify or debug path
- Run browser verification for any family that touches kernels, attention, or cache behavior
- Compare output against the reference stack you use for that family
- Get human review on coherence before promotion

## Common Misses

- Treating a new family as if it needs a new pipeline when the existing pipeline already fits.
- Updating loader or runtime paths before the preset and manifest contract are correct.
- Skipping browser verification because Node looked fine.
- Publishing or cataloging the family before deterministic output has been reviewed.

## Related Guides

- [03-model-preset.md](03-model-preset.md)
- [04-conversion-config.md](04-conversion-config.md)
- [05-promote-model-artifact.md](05-promote-model-artifact.md)
- [06-kernel-path-preset.md](06-kernel-path-preset.md)
- [07-manifest-runtime-field.md](07-manifest-runtime-field.md)
- [08-chat-template-formatter.md](08-chat-template-formatter.md)
- [10-activation-implementation.md](10-activation-implementation.md)
- [11-wgsl-kernel.md](11-wgsl-kernel.md)
- [13-attention-variant.md](13-attention-variant.md)
- [14-quantization-format.md](14-quantization-format.md)

## Canonical References

- `src/inference/README.md`
- [../style/general-style-guide.md](../style/general-style-guide.md)
- [../model-promotion-playbook.md](../model-promotion-playbook.md)
