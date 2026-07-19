---
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - webgpu
  - wgsl
---

# Doppler WGSL Repair V12 adapters

This directory is the technical source of truth for Doppler adapter artifact
identity. `catalog.json` binds each weight file to its SHA-256, exact training
base, exact RDRR runtime base, deployment manifest, immutable origins, and
evidence receipts.

The Clocksmith Hugging Face mirror uses this layout:

```text
adapters/
  seed11/
    adapter_model.safetensors
    adapter_config.json
    source-peft-config.json
    source-training-manifest.json
    training-export-receipt.json
    runtime-adapter-manifest.json
  seed29/
  seed47/
```

Seeds 11 and 47 are preserved replication artifacts. Seed 29 is the selected,
semantically confirmed repair candidate. None of the three is promoted for
production, authorized as a general WGSL writer, or valid for a different base
model. The historical training manifests name the BF16 training checkpoint;
the deployment manifests bind the same bytes to the verified Doppler RDRR
runtime identity without rewriting historical receipts.

The deployed `adapter_config.json` names the immutable upstream
`Qwen/Qwen3.5-9B` revision. `source-peft-config.json` preserves the trainer's
machine-local source path for audit history; it is not a load contract.

Hugging Face is the immutable custody and PEFT-interchange origin. A private
Reploid deployment uses a generation-pinned GCS object as its signed primary
origin; short-lived delivery URLs are authorization material and never artifact
identity. Caches are keyed by the adapter weight SHA-256.

Run `npm run adapter-artifacts:check` for metadata verification or
`npm run adapter-artifacts:verify-local` to hash all local weight bytes.
