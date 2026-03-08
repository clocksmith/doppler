# Tooling API

## Purpose

Browser-safe and Node tooling surface for command runners, diagnostics, registry helpers, storage access, and the shared command contract used by the CLI and browser relay surfaces.

## Import Path

```js
import {
  normalizeToolingCommandRequest,
  ensureCommandSupportedOnSurface,
  runBrowserCommand,
  runNodeCommand,
} from '@simulatte/doppler/tooling';
```

## Audience

Tool builders, harness code, demos, and consumers that need browser/CLI command parity or direct access to the normalized command contract.

## Stability

Public, but tooling-oriented rather than app-facing.

## Primary Exports

- shared command contract helpers
- browser command runner helpers
- Node command runner helpers
- storage registry and shard-manager helpers
- runtime and config inspection helpers

The generated export inventory is the authoritative symbol list for this surface because the tooling subpath is broad.

## Core Behaviors

- shared browser/Node command contract
- explicit fail-closed surface support via `ensureCommandSupportedOnSurface(...)`
- `runBrowserCommand(...)` and `runNodeCommand(...)` return success envelopes on success and throw normalized command errors on failure
- browser-conditioned shared exports for browser bundles, with Node-only helpers available on the default Node import path for the same subpath
- intended for harnesses, diagnostics, demos, registry/storage workflows, and command runners

## Command Families

Canonical tooling commands:

- `convert`
- `debug`
- `bench`
- `verify`
- `lora`
- `distill`

Operator command families:

- `lora.action`: `run|eval|watch|export|compare|quality-gate|activate`
- `distill.action`: `run|stage-a|stage-b|eval|watch|compare|quality-gate|subsets`

Important surface rules:

- `lora` and `distill` normalize through the same command API as the harnessed commands
- browser surfaces currently reject `lora` and `distill`
- `runBrowserCommand(...)` is appropriate for browser-safe commands such as `verify`, `debug`, and `bench`
- `runBrowserCommand(...)` also supports `convert` when the caller injects `options.convertHandler(request)`; CLI browser relay still rejects `convert`
- `runNodeCommand(...)` is the canonical operator execution path for `lora` and `distill`
- current Node operator surfaces reject `runtimePreset`, `runtimeConfigUrl`, and `runtimeConfig` because those runtime inputs are not consumed by `lora`/`distill`
- `configChain` is a harness/browser-URL runtime input, not part of the normalized tooling command request contract
- browser-conditioned imports of `@simulatte/doppler/tooling` resolve the browser-safe shared tooling entry and do not expose `runNodeCommand(...)` or `runBrowserCommandInNode(...)`

## Command Contract Summary

| Command | Required request fields | Notes |
| --- | --- | --- |
| `convert` | `request.inputDir`, `request.convertPayload.converterConfig` | CLI/browser relay: Node-only. Direct `runBrowserCommand(...)`: supported with injected `convertHandler` |
| `verify` | `request.suite` plus `request.modelId` except `kernels` | `request.modelUrl` is optional when `request.modelId` is present |
| `debug` | `request.modelId` | `request.modelUrl` is optional when `request.modelId` is present |
| `bench` | `request.modelId` | `request.modelUrl` is optional when `request.modelId` is present; `bench + workloadType="training"` intentionally allows `modelId: null` |
| `distill` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |
| `lora` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |

Operator-action notes:
- `distill.watch`, `distill.compare`, `distill.quality-gate` require `runRoot`
- `distill.eval` accepts `checkpointPath` or replays finalized checkpoints already present in the run root
- `lora.watch`, `lora.compare`, `lora.quality-gate` require `runRoot`
- `lora.eval` and `lora.export` accept `checkpointPath` or reuse finalized checkpoints in the run root
- `lora.activate` is part of the command contract, but the current Node runner rejects it and points activation to the browser provider/runtime surface

Supported surfaces:
- `convert`: `--surface auto|node`
- `debug`, `bench`, `verify`: `--surface auto|node|browser`
- `lora`, `distill`: `--surface auto|node` in practice; `--surface browser` is rejected

CLI notes:
- operator runs are workload-first; prefer `workloadPath` over ad hoc request fields
- `--surface auto` for `lora` and `distill` does not downgrade to browser
- `lora` and `distill` reject `runtimePreset`, `runtimeConfigUrl`, and `runtimeConfig` on the current Node operator surface
- run-root artifacts live under `reports/training/<kind>/<workload-id>/<timestamp>/`
- `run_contract.json` and `workload.lock.json` are written for every operator run
- `convert` does not take `modelId`; set `output.modelBaseId` in the converter config
- `convert` rejects `runtimePreset`, `runtimeConfigUrl`, `runtimeConfig`, and `configChain` because the convert runner does not consume runtime config
- explicit `convertPayload.execution.useGpuCast=true` is fail-closed; if Node WebGPU is unavailable or GPU casting fails, conversion errors instead of silently falling back to CPU
- `loadMode="memory"` is Node-only and requires local filesystem model data
- prefer immutable Hugging Face revisions for reproducible hosted runs

## Symbol Groups

### Command contract and runners

- `normalizeToolingCommandRequest(...)`
- `ensureCommandSupportedOnSurface(...)`
- `normalizeBrowserCommand(...)`
- `runBrowserCommand(...)`
- `normalizeNodeCommand(...)`
- `runNodeCommand(...)`
- `normalizeNodeBrowserCommand(...)`
- `runBrowserCommandInNode(...)`
- `TOOLING_COMMANDS`
- `TOOLING_SURFACES`
- `TOOLING_SUITES`

### Config and preset helpers

- `listPresets(...)`
- `detectPreset(...)`
- `resolvePreset(...)`
- `getRuntimeConfig()`
- `setRuntimeConfig(...)`

### Storage and manifest helpers

- shard-manager exports
- storage-registry exports
- manifest parsing helpers

### Device and diagnostics helpers

- `initDevice(...)`
- `getDevice()`
- `isWebGPUAvailable()`
- browser suite helpers

## Minimal Example

```js
import { normalizeToolingCommandRequest, runBrowserCommand } from '@simulatte/doppler/tooling';

const request = normalizeToolingCommandRequest({
  command: 'verify',
  suite: 'inference',
  modelId: 'gemma-3-270m-it-wq4k-ef16-hf16',
});

const result = await runBrowserCommand(request);
console.log(result.ok);
```

## Advanced Example

```js
import { normalizeToolingCommandRequest, runNodeCommand } from '@simulatte/doppler/tooling';

const request = normalizeToolingCommandRequest({
  command: 'distill',
  action: 'subsets',
  workloadPath: 'tools/configs/training-workloads/distill-translategemma-tiny.json',
});

const result = await runNodeCommand(request);
console.log(result.result?.subset?.subsetManifestPath);
```

## Contract Notes

- operator commands are workload-first; use `workloadPath` or `runRoot`
- `watch`, `compare`, and `quality-gate` are run-root-driven actions
- `eval` and `export` can operate from an explicit checkpoint path or from finalized checkpoints already present in the run root
- behavior-changing training/eval policy belongs in workload JSON, not in ad hoc command flags

## Code Pointers

- tooling export surface: [src/tooling-exports.d.ts](../../src/tooling-exports.d.ts)
- shared tooling exports: [src/tooling-exports.shared.d.ts](../../src/tooling-exports.shared.d.ts)
- command contract: [src/tooling/command-api.js](../../src/tooling/command-api.js)
- command contract types: [src/tooling/command-api.d.ts](../../src/tooling/command-api.d.ts)
- Node runner: [src/tooling/node-command-runner.js](../../src/tooling/node-command-runner.js)
- browser runner: [src/tooling/browser-command-runner.js](../../src/tooling/browser-command-runner.js)

## Related Surfaces

- [API Docs Index](index.md)
- [Generated export inventory](reference/exports.md)
