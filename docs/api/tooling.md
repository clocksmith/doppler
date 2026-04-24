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
} from 'doppler-gpu/tooling';
```

## Audience

Tool builders, harness code, demos, and consumers that need browser/CLI command parity or direct access to the normalized command contract.

## Stability

Public and tooling-oriented rather than app-facing.
This subpath is mixed-tier: the command/storage/registry core is `tier1`, while
Node operator flows on the same export surface remain `experimental`.
Browser import helpers and P2P helpers now live on
`doppler-gpu/tooling-experimental`. See [Subsystem Support Matrix](../subsystem-support-matrix.md).

## Primary Exports

- tier1 command contract helpers
- tier1 browser and Node command runners for the canonical verify/debug/bench flows
- tier1 storage registry and shard-manager helpers
- tier1 runtime and config inspection helpers
- Program Bundle validator plus Node-only exporter/checker/parity helpers
- experimental Node operator flows for `diagnose`, `lora`, and `distill`

The generated export inventory is the authoritative symbol list for this surface because the tooling subpath is broad and still mixes promoted helpers with experimental operator flows.

## Core Behaviors

- shared browser/Node command contract
- explicit fail-closed surface support via `ensureCommandSupportedOnSurface(...)`
- `runBrowserCommand(...)` and `runNodeCommand(...)` return success envelopes on success and throw normalized command errors on failure
- browser-conditioned shared exports for browser bundles, with Node-only helpers available on the default Node import path for the same subpath
- intended for harnesses, diagnostics, demos, registry/storage workflows, and command runners
- browser-conditioned imports expose Program Bundle validation only; Node imports also expose `exportProgramBundle(...)`, `writeProgramBundle(...)`, `loadProgramBundle(...)`, `checkProgramBundleFile(...)`, and `checkProgramBundleParity(...)`

## Command Families

Canonical tooling commands:

- `convert`
- `refresh-integrity`
- `debug`
- `bench`
- `verify`

Experimental operator commands:

- `diagnose`
- `lora`
- `distill`

Maintenance/export path:

- `program-bundle` is a CLI/tool script for exporting `doppler.program-bundle/v1`.
  It is intentionally outside `normalizeToolingCommandRequest(...)` because it
  reads local artifacts and does not have browser command semantics.
- `program-bundle:reference` is the one-click proof lane for fresh bundles. It
  runs a bounded `verify`, writes the returned report locally, and calls the
  Program Bundle exporter with that report.

Program Bundle parity:

- use `verify` with `workload="inference"` and `workloadType="program-bundle"`.
- set `programBundle` or `programBundlePath`.
- set optional `parityProviders`, for example `["browser-webgpu", "node:webgpu", "node:doe-gpu"]`.
- default `programBundleParityMode` is `contract`; `execute` runs the Node/WebGPU replay path when the provider is available.

Operator command families:

- `lora.action`: `run|eval|watch|export|compare|quality-gate|activate`
- `distill.action`: `run|stage-a|stage-b|eval|watch|compare|quality-gate|subsets`

Important surface rules:

- `diagnose`, `lora`, and `distill` normalize through the same command API as the harnessed commands
- browser surfaces currently reject `diagnose`, `lora`, and `distill`
- `runBrowserCommand(...)` is appropriate for browser-safe commands such as `verify`, `debug`, and `bench`
- `runBrowserCommand(...)` also supports `convert` when the caller injects `options.convertHandler(request)`; CLI browser relay still rejects `convert`
- `runNodeCommand(...)` is the canonical operator execution path for `diagnose`, `lora`, and `distill`
- current Node operator surfaces reject `runtimeProfile`, `runtimeConfigUrl`, and `runtimeConfig` because those runtime inputs are not consumed by `lora`/`distill`
- `configChain` is a harness/browser-URL runtime input, not part of the normalized tooling command request contract
- browser-conditioned imports of `doppler-gpu/tooling` resolve the browser-safe shared tooling entry and do not expose `runNodeCommand(...)` or `runBrowserCommandInNode(...)`
- browser-conditioned imports can validate a Program Bundle but cannot export one because exporting reads local files

## Command Contract Summary

| Command | Required request fields | Notes |
| --- | --- | --- |
| `convert` | `request.inputDir`, `request.convertPayload.converterConfig` | CLI/browser relay: Node-only. Direct `runBrowserCommand(...)`: supported with injected `convertHandler` |
| `refresh-integrity` | `request.modelDir` | Node-only. Rebuilds `manifest.integrityExtensions` from local shard bytes; rejects runtime config inputs |
| `verify` | `request.workload` plus `request.modelId` except `kernels` and `workloadType="program-bundle"` | `request.workload` is required and may be `embedding` for embedding-model correctness checks; `request.modelUrl` is optional when `request.modelId` is present; `request.inferenceInput` is available for request-owned inference payloads when `workload="inference"`; Program Bundle parity requires `programBundle` or `programBundlePath` |
| `debug` | `request.workload` plus `request.modelId` | `request.workload` is required and may be `inference` or `embedding`; `request.inferenceInput` is available when `workload="inference"` |
| `bench` | `request.workload` plus `request.modelId` | `request.workload` is required and may be `inference`, `embedding`, `training`, `diffusion`, or `energy`; `workload="training"` intentionally allows `modelId: null`; `request.inferenceInput` is available when `workload="inference"` |
| `diagnose` | `request.workload` plus `request.modelId` | Node-only. Uses the same workload family as `debug` and fails closed on browser |
| `distill` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |
| `lora` | `request.action` plus `request.workloadPath` or `request.runRoot` | Node-only today; browser fails closed |

Operator-action notes:
- `distill.watch`, `distill.compare`, `distill.quality-gate` require `runRoot`
- `distill.eval` accepts `checkpointPath` or replays finalized checkpoints already present in the run root
- `lora.watch`, `lora.compare`, `lora.quality-gate` require `runRoot`
- `lora.eval` and `lora.export` accept `checkpointPath` or reuse finalized checkpoints in the run root
- `lora.activate` is part of the command contract, but the current Node runner rejects it and points activation to the browser runtime surface

Supported surfaces:
- `convert`: `--surface auto|node`
- `refresh-integrity`: `--surface auto|node`
- `debug`, `bench`, `verify`: `--surface auto|node|browser`
- `diagnose`, `lora`, `distill`: `--surface auto|node` in practice; `--surface browser` is rejected

CLI notes:
- operator runs are workload-first; prefer `workloadPath` over ad hoc request fields
- `--surface auto` for `lora` and `distill` does not downgrade to browser
- `--config` accepts inline JSON, file path, or URL for all commands
- `--runtime-config` supports inline JSON, file path, or URL for `verify`, `debug`, and `bench`
- `diagnose` is Node-only and uses the same normalized request contract as `debug`
- `lora` and `distill` reject `runtimeProfile`, `runtimeConfigUrl`, and `runtimeConfig` on the current Node operator surface
- run-root artifacts live under `reports/training/<kind>/<workload-id>/<timestamp>/`
- `run_contract.json` and `workload.lock.json` are written for every operator run
- `convert` does not take `modelId`; set `output.modelBaseId` in the converter config
- `convert` rejects `runtimeProfile`, `runtimeConfigUrl`, `runtimeConfig`, and `configChain` because the convert runner does not consume runtime config
- `refresh-integrity` rejects `runtimeProfile`, `runtimeConfigUrl`, `runtimeConfig`, and `configChain` because it only restamps local artifact metadata
- explicit `convertPayload.execution.useGpuCast=true` is fail-closed; if Node WebGPU is unavailable or GPU casting fails, conversion errors instead of silently falling back to CPU
- `loadMode="memory"` is Node-only and requires local filesystem model data; direct-source loads default to hash verification when `runtime.loading.shardCache.verifyHashes` is not overridden, and Node source loads now fail closed when `runtime.loading.memoryManagement.budget` says the projected resident footprint is too large
- `request.inferenceInput` is the shared request-owned inference payload. It currently supports:
  - `prompt`: string or structured prompt object/array for text inference
  - `maxTokens`: request-local max token override
  - `image`: one of `url`, `pixels`, or `pixelDataBase64` for image-to-text flows
  - `softTokenBudget`: per-request visual token budget when `image` is present
- raw image payloads require `image.width` and `image.height`; `image.pixels` / `image.pixelDataBase64` must contain RGB or RGBA bytes
- browser relay can rewrite a local `file://` `request.inferenceInput.image.url` onto the relay-owned static server; direct browser/Node surfaces otherwise require a decodable URL in the active runtime
- a persisted direct-source manifest can be written with `node tools/materialize-source-manifest.js <source-dir-or-gguf>` and then loaded through `loadMode="http"` from a `file://` or hosted manifest root
- in browser runs, a persisted direct-source artifact can now be cache-primed into OPFS and later reopened with `loadMode="opfs"` when the request includes `modelId` plus the explicit hosted `modelUrl`
- debug runs and any run with active investigation instrumentation persist a `debugSnapshot` in the run result/report so probe and trace logs survive beyond live console output
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
- `buildManifestIntegrityFromModelDir(...)` (Node import path only)
- `refreshManifestIntegrity(...)` (Node import path only)

### Program Bundle

- `validateProgramBundle(...)`
- `PROGRAM_BUNDLE_SCHEMA_ID`
- `PROGRAM_BUNDLE_SCHEMA_VERSION`
- `PROGRAM_BUNDLE_HOST_JS_SUBSET`
- `exportProgramBundle(...)` (Node import path only)
- `writeProgramBundle(...)` (Node import path only)
- `loadProgramBundle(...)` (Node import path only)
- `checkProgramBundleFile(...)` (Node import path only)
- `checkProgramBundleParity(...)` (Node import path only)
- `PROGRAM_BUNDLE_PARITY_SCHEMA_ID`
- `TOOLING_SURFACES`
- `TOOLING_WORKLOADS`

### Config helpers

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
- browser workload helpers

## Minimal Example

```js
import { normalizeToolingCommandRequest, runBrowserCommand } from 'doppler-gpu/tooling';

const request = normalizeToolingCommandRequest({
  command: 'verify',
  workload: 'embedding',
  modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
});

const result = await runBrowserCommand(request);
console.log(result.ok);
```

## Regular vs Embedding Examples

```js
const regularDebug = normalizeToolingCommandRequest({
  command: 'debug',
  workload: 'inference',
  modelId: 'gemma-3-270m-it-q4k-ehf16-af32',
});

const embeddingDebug = normalizeToolingCommandRequest({
  command: 'debug',
  workload: 'embedding',
  modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
});
```

## Image-to-Text Example

```js
const imageVerify = normalizeToolingCommandRequest({
  command: 'verify',
  workload: 'inference',
  modelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
  inferenceInput: {
    prompt: 'Describe the image in one short sentence.',
    maxTokens: 16,
    softTokenBudget: 140,
    image: {
      width: 1,
      height: 1,
      pixels: [255, 255, 255, 255],
    },
  },
});
```

## Advanced Example

```js
import { normalizeToolingCommandRequest, runNodeCommand } from 'doppler-gpu/tooling';

const request = normalizeToolingCommandRequest({
  command: 'distill',
  action: 'subsets',
  workloadPath: 'src/experimental/training/workload-packs/distill-translategemma-tiny.json',
});

const result = await runNodeCommand(request);
console.log(result.result?.subset?.subsetManifestPath);
```

## Contract Notes

- operator commands are workload-first; use `workloadPath` or `runRoot`
- `watch`, `compare`, and `quality-gate` are run-root-driven actions
- `eval` and `export` can operate from an explicit checkpoint path or from finalized checkpoints already present in the run root
- behavior-changing training/eval policy belongs in workload JSON, not in ad hoc command flags
- browser conversion/file-picker helpers and P2P helpers were split to `doppler-gpu/tooling-experimental`

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
