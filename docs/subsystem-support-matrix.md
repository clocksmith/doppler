# Subsystem Support Matrix

Auto-generated from `src/config/support-tiers/subsystems.json`.
Run `npm run support:subsystems:sync` after editing the subsystem-tier registry.

Updated at: 2026-04-13

## Summary

- Tier 1 subsystems: 9
- Experimental subsystems: 12
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
| Root app-facing facade | `primary` | `tier1` | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/index.js](../src/index.js) | Preferred public surface for application authors. |
| Text inference runtime | `primary` | `tier1` | [README.md](../README.md), [docs/architecture.md](architecture.md), [docs/pipeline-contract.md](pipeline-contract.md) | Mainline runtime contract for verified browser and Node text generation workloads. |
| Generation subpath | `secondary` | `tier1` | [docs/api/generation.md](api/generation.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/generation/index.js](../src/generation/index.js) | Supported advanced surface for direct text-pipeline construction. |
| Loaders subpath | `secondary` | `tier1` | [docs/api/loaders.md](api/loaders.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/loaders/index.js](../src/loaders/index.js) | Supported advanced surface for explicit loading flows beyond the root facade. |
| RDRR artifact runtime | `secondary` | `tier1` | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/formats/rdrr/manifest.js](../src/formats/rdrr/manifest.js) | Primary artifact format for the verified quickstart and hosted model path. |
| Tooling subpath | `secondary` | `tier1` | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.js](../src/tooling-exports.js) | Supported advanced surface for the tier1 command contract, diagnostics, storage, and registry tooling. Some helper groups exported on the same subpath remain experimental and are classified separately below. |
| LoRA hot-swap | `primary` | `experimental` | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/experimental/adapters/lora-loader.js](../src/experimental/adapters/lora-loader.js) | User-visible adapter swap capability that remains outside the canonical quickstart and demo-default proof path. |
| Direct-source runtime inputs | `secondary` | `experimental` | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/tooling/source-artifact-adapter.js](../src/tooling/source-artifact-adapter.js) | Available direct-source path for safetensors, gguf, .tflite, .task, and .litertlm inputs, but not part of the quickstart or canonical demo proof. |
| Multi-model orchestration | `secondary` | `experimental` | [docs/architecture.md](architecture.md), [docs/api/orchestration.md](api/orchestration.md), [src/inference/multi-model-network.js](../src/inference/multi-model-network.js) | Available for advanced experimentation, but not part of the tier1 demo or quickstart contract. |
| Orchestration subpath | `secondary` | `experimental` | [docs/api/orchestration.md](api/orchestration.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/experimental/orchestration/index.js](../src/experimental/orchestration/index.js) | Mixed advanced surface for adapters, routers, heads, and multi-model helpers; available, but not part of the mainline proof story. |

## Tier 1 Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Hosted browser demo workspace | `demo` | yes | yes | no | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [demo/index.html](../demo/index.html) | Canonical hosted proof surface for browser-native local text inference. |
| OpenAI-compatible localhost server | `cli` | yes | no | no | [README.md](../README.md), [docs/cli.md](cli.md), [src/cli/doppler-serve.js](../src/cli/doppler-serve.js) | Compatibility bridge for existing OpenAI-style apps and eval workflows. |
| Quickstart CLI | `cli` | yes | no | no | [README.md](../README.md), [docs/getting-started.md](getting-started.md), [src/cli/doppler-quickstart.js](../src/cli/doppler-quickstart.js) | Canonical zero-install proof path for local text inference from the terminal. |
| Root app-facing facade | `api` | yes | no | yes | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/index.js](../src/index.js) | Preferred public surface for application authors. |
| Text inference runtime | `runtime` | yes | yes | no | [README.md](../README.md), [docs/architecture.md](architecture.md), [docs/pipeline-contract.md](pipeline-contract.md) | Mainline runtime contract for verified browser and Node text generation workloads. |
| Generation subpath | `api` | yes | no | yes | [docs/api/generation.md](api/generation.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/generation/index.js](../src/generation/index.js) | Supported advanced surface for direct text-pipeline construction. |
| Loaders subpath | `api` | yes | no | yes | [docs/api/loaders.md](api/loaders.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/loaders/index.js](../src/loaders/index.js) | Supported advanced surface for explicit loading flows beyond the root facade. |
| RDRR artifact runtime | `format` | yes | yes | no | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/formats/rdrr/manifest.js](../src/formats/rdrr/manifest.js) | Primary artifact format for the verified quickstart and hosted model path. |
| Tooling subpath | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.js](../src/tooling-exports.js) | Supported advanced surface for the tier1 command contract, diagnostics, storage, and registry tooling. Some helper groups exported on the same subpath remain experimental and are classified separately below. |

## Experimental Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA hot-swap | `runtime` | yes | no | no | [README.md](../README.md), [docs/api/root.md](api/root.md), [src/experimental/adapters/lora-loader.js](../src/experimental/adapters/lora-loader.js) | User-visible adapter swap capability that remains outside the canonical quickstart and demo-default proof path. |
| Direct-source runtime inputs | `format` | yes | no | no | [docs/rdrr-format.md](rdrr-format.md), [docs/architecture.md](architecture.md), [src/tooling/source-artifact-adapter.js](../src/tooling/source-artifact-adapter.js) | Available direct-source path for safetensors, gguf, .tflite, .task, and .litertlm inputs, but not part of the quickstart or canonical demo proof. |
| Multi-model orchestration | `runtime` | yes | no | no | [docs/architecture.md](architecture.md), [docs/api/orchestration.md](api/orchestration.md), [src/inference/multi-model-network.js](../src/inference/multi-model-network.js) | Available for advanced experimentation, but not part of the tier1 demo or quickstart contract. |
| Orchestration subpath | `api` | yes | no | yes | [docs/api/orchestration.md](api/orchestration.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/experimental/orchestration/index.js](../src/experimental/orchestration/index.js) | Mixed advanced surface for adapters, routers, heads, and multi-model helpers; available, but not part of the mainline proof story. |
| Diffusion API subpath | `api` | yes | no | yes | [docs/api/diffusion.md](api/diffusion.md), [src/experimental/diffusion/index.js](../src/experimental/diffusion/index.js) | Shipped advanced surface for diffusion/image workflows, outside the main tier1 local text inference story. |
| Energy API subpath | `api` | yes | no | yes | [docs/api/energy.md](api/energy.md), [src/experimental/energy/index.js](../src/experimental/energy/index.js) | Shipped specialized surface, but not part of the mainline README or demo-default claim set. |
| P2P and shard distribution runtime | `runtime` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/distribution/shard-delivery.js](../src/experimental/distribution/shard-delivery.js), [src/experimental/distribution/p2p-control-plane.js](../src/experimental/distribution/p2p-control-plane.js) | Strategic distribution lane that exists in the repo but is not part of the public tier1 runtime claim set. |
| Runtime hotswap bundle system | `runtime` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/hotswap/runtime.js](../src/experimental/hotswap/runtime.js), [src/experimental/hotswap/model-swap.js](../src/experimental/hotswap/model-swap.js) | Checked-in experimental runtime machinery, not part of the public package contract. |
| Tooling browser import helpers | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.shared.js](../src/tooling-exports.shared.js) | Browser conversion and file-picker helpers ship on the tooling subpath for advanced workflows, but they are not part of the tier1 quickstart or demo-default contract. |
| Tooling operator commands | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [src/tooling/node-command-runner.js](../src/tooling/node-command-runner.js), [src/experimental/training/index.js](../src/experimental/training/index.js) | Node-only `diagnose`, `lora`, and `distill` operator flows share the tooling contract surface, but they are still experimental research/operator lanes rather than tier1 product behavior. |
| Tooling P2P helpers | `api` | yes | no | yes | [docs/api/tooling.md](api/tooling.md), [docs/api/advanced-root-exports.md](api/advanced-root-exports.md), [src/tooling-exports.shared.js](../src/tooling-exports.shared.js) | P2P and distribution helpers are exported for advanced tooling flows, but they remain outside the current tier1 local text inference story. |
| Training and distillation surfaces | `runtime` | yes | no | no | [docs/training-handbook.md](training-handbook.md), [src/experimental/training/README.md](../src/experimental/training/README.md), [src/experimental/training/index.js](../src/experimental/training/index.js) | Repo-supported research and operator lane, but not part of the current tier1 app/runtime contract. |

## Internal-only Surfaces

| Subsystem | Scope | User-facing | Demo default | Exported | Anchors | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Bridge and extension integration layer | `integration` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/bridge/index.js](../src/experimental/bridge/index.js), [src/experimental/bridge/extension-client.js](../src/experimental/bridge/extension-client.js) | Repo integration machinery, not part of the public package or hosted demo contract. |
| Browser import and conversion helpers | `browser` | no | no | no | [docs/architecture.md](architecture.md), [src/experimental/browser/browser-converter.js](../src/experimental/browser/browser-converter.js), [src/experimental/browser/tensor-source-http.js](../src/experimental/browser/tensor-source-http.js) | Repo-internal browser import plumbing that supports the demo and tooling flows. |
