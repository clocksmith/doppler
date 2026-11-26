# Advanced Export Map

## Purpose

Choose the right dedicated subpath when embedding `dopplerx` as a library
(Vite, Rollup, webpack, esbuild, or any modern ESM bundler), and map the older
broad root surface to the current subpaths. For CLI usage, see
[the CLI reference](../cli.md).

## Audience

Advanced consumers migrating from the older catch-all root barrel, or bundling
lower-level surfaces into an application.

## Stability

Public guidance only. Subpath support tiers are defined in
[Subsystem Support Matrix](../subsystem-support-matrix.md). The exact export
list per subpath is the [generated export inventory](reference/exports.md);
[API Docs](index.md) classifies each surface.

## Subpath Split

| Subpath | Use for | API contract |
| --- | --- | --- |
| `dopplerx` | signed Capsule execution with explicit ports | [Doppler Run API](root.md) |
| `dopplerx/host` | Capsule execution with Doppler's browser or Node host ports | [Doppler Run API](root.md) |
| `dopplerx/compat` | the former manifest-loading facade (`dr`, `doppler`) | [Compatibility API](compat.md) |
| `dopplerx/generation` | direct pipeline construction (`createPipeline`, `InferencePipeline`, `EmbeddingPipeline`) | [Generation API](generation.md) |
| `dopplerx/training` | SFT/LoRA orchestration and backward primitives | [Training API](training.md) |
| `dopplerx/tooling/device` | GPU init, capabilities, shader-source preseeding | [export inventory](reference/exports.md) |
| `dopplerx/tooling/storage` | OPFS shard manager, registry, inventory, quota, cache | [export inventory](reference/exports.md) |
| `dopplerx/tooling/manifest` | RDRR manifest parsing and schema defaults | [export inventory](reference/exports.md) |
| `dopplerx/tooling` | command runners and the union of the tooling surfaces | [Tooling API](tooling.md) |

## Migration Rule

- New application code executes signed Capsules through `dopplerx` or
  `dopplerx/host`. The root has no `dr`; `dr` is only in `dopplerx/compat`, an
  experimental compatibility facade.
- `doppler-gpu/loaders`, `doppler-gpu/orchestration`, and
  `doppler-gpu/tooling-experimental` were removed in `doppler-gpu` 0.7.0 (the
  package's former name); see the [changelog](../../CHANGELOG.md) for
  replacements.
- Manifest and OPFS storage helpers import from `dopplerx/tooling/manifest` and
  `dopplerx/tooling/storage`.
- Treat `dopplerx/tooling` as a mixed-tier export mainly because operator
  command flows remain experimental.
- Check the subsystem support matrix before treating every exported subpath as
  equally promoted.

## Bundling

- Runtime: Node 22 (what CI and qualified runs use). Source modules use JSON
  import attributes (`with { type: 'json' }`), so the bundler must understand
  JSON module imports.
- WebGPU must be available in the runtime environment.
- `package.json` declares `"sideEffects": ["**/*.wgsl", "./src/gpu/device.js"]`,
  so bundlers drop imported-but-unreferenced exports. Prefer the narrowest
  subpath: `dopplerx/tooling` re-exports storage, config, device, converter,
  and inference helpers, and a narrower import avoids pulling all of them in.

```js
import { createPipeline } from 'dopplerx/generation';
import { initDevice, registerShaderSources } from 'dopplerx/tooling/device';
import {
  openModelStore,
  loadManifestFromStore,
  ensureModelCached,
} from 'dopplerx/tooling/storage';
import { parseManifest } from 'dopplerx/tooling/manifest';
```

## Shader Loading

The package ships its WGSL kernels in `src/gpu/kernels/`. By default the runtime
fetches them from a base path derived from `globalThis.__DOPPLER_KERNEL_BASE_PATH__`
or the module's `import.meta.url`, which needs an HTTP endpoint serving the
`.wgsl` files.

For bundled builds, preseed the shader cache with bundler-inlined strings
before any pipeline init:

```js
// Vite: import.meta.glob with raw strings.
import { registerShaderSources } from 'dopplerx/tooling/device';

const shaderSources = import.meta.glob(
  '/node_modules/dopplerx/src/gpu/kernels/*.wgsl',
  { as: 'raw', eager: true }
);
registerShaderSources(shaderSources);
```

Preseeded sources are keyed by file name, and the runtime reads them from
memory instead of fetching. You then no longer need to serve the kernel
directory from your app's origin.

## Model Artifacts

Model shards are not in the npm package. Hosted artifacts are listed in the
shipped `models/catalog.json` (Hugging Face `clocksmith/rdrr`); catalog lanes
marked local-only must come from your own artifact location. Library consumers
pass an explicit model URL; the CLI's external-root resolution
(`$DOPPLER_EXTERNAL_MODELS_ROOT`) is CLI-only.

## Program Bundles

`dopplerx/tooling` exports `validateProgramBundle(...)` on browser and Node
imports. The Node import also exports `exportProgramBundle(...)`,
`writeProgramBundle(...)`, `loadProgramBundle(...)`, and
`checkProgramBundleFile(...)`. See [Program Bundle](../integration/program-bundle.md).

## Code Pointers

- package export map: [package.json](../../package.json)
- Capsule root: [src/capsule-runtime.js](../../src/capsule-runtime.js)
- compatibility facade: [src/index.js](../../src/index.js)
- shader preseeding: [src/gpu/kernels/shader-cache.js](../../src/gpu/kernels/shader-cache.js)

## Related Surfaces

- [API Docs](index.md)
- [Generated export inventory](reference/exports.md)
- [Doppler Run API](root.md)
- [Compatibility API](compat.md)
