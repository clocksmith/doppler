# Subsystem Support Matrix

Auto-generated from `src/config/support-tiers/subsystems.json`.
Run `npm run support:subsystems:sync` after editing the subsystem-tier registry.

Updated at: 2026-08-22

## Summary

- Tier 1 subsystems: 8
- Experimental subsystems: 9
- Internal-only subsystems: 2

## Tier Meanings

- `tier1`: current public support contract and claimable mainline behavior.
- `experimental`: checked-in and sometimes exported, but not part of the canonical quickstart or demo-default proof path.
- `internal-only`: repo machinery that should not be treated as part of the public product contract.

## README-Facing Claims

| Claim | Visibility | Tier | Anchors | Notes |
| --- | --- | --- | --- | --- |
| Hosted browser demo workspace | `primary` | `tier1` | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [demo/index.html](../demo/index.html) | Canonical hosted proof surface for browser-native local text inference. |
| OpenAI-compatible localhost server | `primary` | `tier1` | [README.md](../README.md), [docs/cli.md](cli.md), [src/cli/doppler-serve.js](../src/cli/doppler-serve.js) | Compatibility bridge for existing OpenAI-style apps and eval workflows. |
| Quickstart CLI | `primary` | `tier1` | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [src/cli/doppler-quickstart.js](../src/cli/doppler-quickstart.js) | Canonical zero-install proof path for local text inference from the terminal. |
| Root app-facing facade | `primary` | `tier1` | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/capsule-runtime.js](../src/capsule-runtime.js) | Preferred Capsule-native application surface with explicit trust and injected execution ports. |
| Text inference runtime | `primary` | `tier1` | [README.md](../README.md), [docs/architecture.md](architecture.md), [docs/pipeline-contract.md](pipeline-contract.md) | Mainline runtime contract for verified browser and Node text generation workloads. |
| Generation subpath | `secondary` | `tier1` | [docs/api/generation.md](api/generation.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/generation/index.js](../src/generation/index.js) | Supported advanced surface for direct text-pipeline construction. |
| RDRR artifact runtime | `secondary` | `tier1` | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/formats/rdrr/manifest.js](../src/formats/rdrr/manifest.js) | Primary artifact format for the verified quickstart and hosted model path. |
| Tooling subpath | `secondary` | `tier1` | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.js](../src/tooling-exports.js) | Supported advanced surface for the tier1 command contract, diagnostics, storage, and registry tooling. Some helper groups exported on the same subpath remain experimental and are classified separately below. |
| LoRA hot-swap | `primary` | `experimental` | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/experimental/adapters/lora-loader.js](../src/experimental/adapters/lora-loader.js) | User-visible adapter swap capability that remains outside the canonical quickstart and demo-default proof path. |
| Bun WebGPU host runtime | `secondary` | `experimental` | [GOALS.md](../GOALS.md), [docs/bun-product-qualification.md](bun-product-qualification.md), [docs/release-matrix.md](release-matrix.md) | Bun can execute the Node command surface with native WebGPU, but it remains experimental until the three-workload qualification policy has current semantic receipts, explicit human promotion, and synchronized support and release projections. |
| Direct-source runtime inputs | `secondary` | `experimental` | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/tooling/source-artifact-adapter.js](../src/tooling/source-artifact-adapter.js) | Available direct-source path for safetensors, gguf, .tflite, .task, and .litertlm inputs, but not part of the quickstart or canonical demo proof. |
| Legacy manifest-loading compatibility facade | `secondary` | `experimental` | [docs/api/compat.md](api/compat.md), [src/index.js](../src/index.js), [src/client/doppler-api.js](../src/client/doppler-api.js) | Explicit compatibility surface for legacy manifest loading and provider-oriented application APIs. |

## Tier 1 Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Hosted browser demo workspace | `demo` | yes | yes | no | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [demo/index.html](../demo/index.html) | Canonical hosted proof surface for browser-native local text inference. |
| OpenAI-compatible localhost server | `cli` | yes | no | no | [README.md](../README.md), [docs/cli.md](cli.md), [src/cli/doppler-serve.js](../src/cli/doppler-serve.js) | Compatibility bridge for existing OpenAI-style apps and eval workflows. |
| Quickstart CLI | `cli` | yes | no | no | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [src/cli/doppler-quickstart.js](../src/cli/doppler-quickstart.js) | Canonical zero-install proof path for local text inference from the terminal. |
| Root app-facing facade | `api` | yes | no | yes | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/capsule-runtime.js](../src/capsule-runtime.js) | Preferred Capsule-native application surface with explicit trust and injected execution ports. |
| Text inference runtime | `runtime` | yes | yes | no | [README.md](../README.md), [docs/architecture.md](architecture.md), [docs/pipeline-contract.md](pipeline-contract.md) | Mainline runtime contract for verified browser and Node text generation workloads. |
| Generation subpath | `api` | yes | no | yes | [docs/api/generation.md](api/generation.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/generation/index.js](../src/generation/index.js) | Supported advanced surface for direct text-pipeline construction. |
| RDRR artifact runtime | `format` | yes | yes | no | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/formats/rdrr/manifest.js](../src/formats/rdrr/manifest.js) | Primary artifact format for the verified quickstart and hosted model path. |
| Tooling subpath | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.js](../src/tooling-exports.js) | Supported advanced surface for the tier1 command contract, diagnostics, storage, and registry tooling. Some helper groups exported on the same subpath remain experimental and are classified separately below. |

## Experimental Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA hot-swap | `runtime` | yes | no | no | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/experimental/adapters/lora-loader.js](../src/experimental/adapters/lora-loader.js) | User-visible adapter swap capability that remains outside the canonical quickstart and demo-default proof path. |
| Bun WebGPU host runtime | `runtime` | yes | no | no | [GOALS.md](../GOALS.md), [docs/bun-product-qualification.md](bun-product-qualification.md), [docs/release-matrix.md](release-matrix.md) | Bun can execute the Node command surface with native WebGPU, but it remains experimental until the three-workload qualification policy has current semantic receipts, explicit human promotion, and synchronized support and release projections. |
| Direct-source runtime inputs | `format` | yes | no | no | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/tooling/source-artifact-adapter.js](../src/tooling/source-artifact-adapter.js) | Available direct-source path for safetensors, gguf, .tflite, .task, and .litertlm inputs, but not part of the quickstart or canonical demo proof. |
| Legacy manifest-loading compatibility facade | `api` | yes | no | yes | [docs/api/compat.md](api/compat.md), [src/index.js](../src/index.js), [src/client/doppler-api.js](../src/client/doppler-api.js) | Explicit compatibility surface for legacy manifest loading and provider-oriented application APIs. |
| P2P and shard distribution runtime | `runtime` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/distribution/shard-delivery.js](../src/experimental/distribution/shard-delivery.js), [src/experimental/distribution/p2p-control-plane.js](../src/experimental/distribution/p2p-control-plane.js) | Strategic distribution lane that exists in the repo but is not part of the public tier1 runtime claim set. |
| Runtime hotswap bundle system | `runtime` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/hotswap/runtime.js](../src/experimental/hotswap/runtime.js), [src/experimental/hotswap/model-swap.js](../src/experimental/hotswap/model-swap.js) | Checked-in experimental runtime machinery, not part of the public package contract. |
| Tooling operator commands | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [src/tooling/node-command-runner.js](../src/tooling/node-command-runner.js), [src/experimental/training/index.js](../src/experimental/training/index.js) | Node-only `diagnose`, `lora`, and `distill` operator flows share the tooling contract surface, but they are still experimental research/operator lanes rather than tier1 product behavior. |
| Training and distillation surfaces | `runtime` | yes | no | no | [docs/training-handbook.md](training-handbook.md), [src/experimental/training/GUIDE.md](../src/experimental/training/GUIDE.md), [src/experimental/training/index.js](../src/experimental/training/index.js) | Repo-supported research and operator lane, including distillation workflows that are not exported by the SFT/LoRA training subpath. |
| Training API subpath | `api` | yes | no | yes | [docs/api/training.md](api/training.md), [docs/training-handbook.md](training-handbook.md), [src/training.js](../src/training.js) | Public experimental SFT/LoRA library; native full-graph support is capability-gated and packed Q4K bases require an external trainer. |

## Internal-only Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Bridge and extension integration layer | `integration` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/bridge/index.js](../src/experimental/bridge/index.js), [src/experimental/bridge/extension-client.js](../src/experimental/bridge/extension-client.js) | Repo integration machinery, not part of the public package or hosted demo contract. |
| Browser import and conversion helpers | `browser` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/browser/browser-converter.js](../src/experimental/browser/browser-converter.js), [src/experimental/browser/tensor-source-http.js](../src/experimental/browser/tensor-source-http.js) | Repo-internal browser import plumbing that supports the demo and tooling flows. |
