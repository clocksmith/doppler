# Using `doppler-gpu` as a library

This guide is for applications that embed `doppler-gpu` as an npm dependency (Vite, Rollup, webpack, esbuild, or any modern ESM bundler). For CLI usage and first-party tooling, see `AGENTS.md`.

## Minimum requirements

- Node **>=22** (for JSON import attributes used internally by the rule registry).
- Bundlers that understand JSON module imports (`with { type: 'json' }`). Vite 5+, Rollup 4+, esbuild 0.17+, webpack 5+ all qualify.
- WebGPU in the runtime environment (browser-side: `GPUAdapter`).

## Tree-shaking

`doppler-gpu` ships with `"sideEffects": ["**/*.wgsl", "./src/gpu/device.js"]`. Bundlers will drop any imported-but-unreferenced export. Two consequences:

1. Prefer the narrowest subpath export for your use case (see below). The `doppler-gpu/tooling` barrel re-exports from storage, config, device, converter, and inference surfaces; a narrower import avoids dragging all of those into your bundle.
2. WGSL files are marked with side effects because they are currently loaded at runtime via fetch (see WGSL section). If you preseed them (recommended for bundled builds) you can also drop `**/*.wgsl` from your consumer's `sideEffects` consideration.

## Recommended imports

```js
// Runtime pipeline (the thing that actually runs inference)
import { createPipeline } from "doppler-gpu/generation";

// Narrow tooling exports (prefer over the mega barrel)
import { initDevice, registerShaderSources } from "doppler-gpu/tooling/device";
import {
  openModelStore,
  loadManifestFromStore,
  ensureModelCached,
} from "doppler-gpu/tooling/storage";
import { parseManifest } from "doppler-gpu/tooling/manifest";

// Per-family static metadata (no runtime weight — kilobyte-scale)
import { KNOWN_MODELS, resolveHfBaseUrl } from "doppler-gpu/models/qwen3";
```

Available narrow entry points:

- `doppler-gpu` — the `doppler` client and provider factory.
- `doppler-gpu/generation` — `createPipeline`, `InferencePipeline`, `EmbeddingPipeline`.
- `doppler-gpu/tooling/device` — GPU init, capabilities, shader-source preseeding.
- `doppler-gpu/tooling/storage` — OPFS shard manager, registry, inventory, quota, cache.
- `doppler-gpu/tooling/manifest` — RDRR manifest parsing + schema defaults.
- `doppler-gpu/tooling` — mega barrel (back-compat; pulls the union of the narrow entry points).
- `doppler-gpu/models/{qwen3,gemma3,gemma4,embeddinggemma}` — family pointers.

## Program Bundles

`doppler-gpu/tooling` exports `validateProgramBundle(...)` on both browser and
Node imports. The Node import path also exports `exportProgramBundle(...)`,
`writeProgramBundle(...)`, `loadProgramBundle(...)`, and
`checkProgramBundleFile(...)` for producing `doppler.program-bundle/v1`
artifacts from local manifests and browser reference reports.

Program Bundle export is for toolchains such as Doe/Cerebras lowering. Normal
browser applications should load RDRR manifests directly unless they need that
portable execution-graph artifact.

## Shader loading

`doppler-gpu` ships 179 WGSL kernels in `src/gpu/kernels/*.wgsl`. By default, the runtime fetches them at runtime from `KERNEL_BASE_PATH` (derived from `globalThis.__DOPPLER_KERNEL_BASE_PATH__` or `import.meta.url`). This requires a same-origin HTTP endpoint serving the `.wgsl` files.

For bundled builds, preseed the shader cache with bundler-inlined strings:

```js
// Vite / Rollup with @rollup/plugin-url / ?raw:
import { registerShaderSources } from "doppler-gpu/tooling/device";

const shaderSources = import.meta.glob(
  "/node_modules/doppler-gpu/src/gpu/kernels/*.wgsl",
  { as: "raw", eager: true }
);
registerShaderSources(shaderSources);
```

After `registerShaderSources` runs, the runtime bypasses the HTTP fetch path entirely and reads from memory. You no longer need to serve `/doppler/src/gpu/kernels/*` from your app's origin.

## Model artifacts

Model shards (hundreds of MB to several GB) are **not** in the npm package; they live on Hugging Face at `Clocksmith/rdrr`. Use one of:

- `resolveHfBaseUrl(modelId)` from the family module → fetch manifest + shards directly from HF at runtime (OPFS-cached).
- Self-host a copy and pass a `file://` or HTTPS `modelUrl` to the pipeline.
- The CLI's auto-resolution (external-drive root) works only for CLI consumers; library consumers must pass an explicit URL.

## Worker bundles (iife)

The rule registry previously used top-level `await` which broke worker bundles targeting iife. That was removed in 0.4.3. Consumers building classic workers no longer need `format: 'esm'` workarounds.

## Migration checklist for existing consumers

1. Replace `fetch("/doppler/src/inference/pipelines/text.js").then(import)` patterns with `import { createPipeline } from "doppler-gpu/generation"`.
2. Delete same-origin `/doppler/src/*` proxy middleware (only `/doppler/src/gpu/kernels/*` is needed, and only if you did not preseed).
3. If you served `/doppler/models/<id>/*` from a filesystem copy, switch to `resolveHfBaseUrl` or your own mirror.
4. Bump your peer dep to `doppler-gpu >= 0.4.3`.
