# Doppler Rig and Doppler Run

Doppler Rig prepares models. A Capsule binds the signed implementation. Doppler
Run verifies and executes it. These names replace Forge and Runtime as component
names; compiler, runtime, and host remain useful technical descriptions.

## Current interfaces

- Preparation command: `npm run rig:capsule -- --config <path>` or
  `node tools/rig-model-capsule.js --config <path>` from the repository.
- Preparation implementation: `rigModelCapsule()` and `buildRigOptions()` in
  `src/tooling/model-capsule-rig.js`; compiler entry: `runRigPipeline()`.
- Execution import: `doppler-gpu/run`, with the same exports as `doppler-gpu`.
- Injected execution core: `createDopplerRun()` and `RUN_CORE_VERSION`.
- Execution types: `DopplerRun`, `DopplerRunSession`, and `RunPorts`.
- Applications can continue using `openCapsule()` through `doppler-gpu/host`.

Rig is repository preparation tooling. This rename does not add a Rig compiler
or signing authority to the browser execution package.

## Compatibility

`doppler-gpu/runtime`, `createDopplerRuntime`, `RUNTIME_CORE_VERSION`, and their
existing types remain aliases of the same implementation. The `forge:*` npm
scripts, `tools/forge-model-capsule.js`, and old preparation exports continue to
forward to Rig. The old and new entrypoints have identical execution semantics.
These aliases do not restore the former Pack APIs or schemas.

Serialized schema IDs, `forgeVersion`, `runtimeConfig`, execution identities,
model IDs, and configuration paths retain their spelling. Existing module paths
that appear in retained evidence remain usable. Ordinary runtime terminology,
including Node runtime and runtime configuration, is not a component brand.

Signed Capsules, historical reports, and published browser/Node starter archives
retain their exact bytes. New source names are not a new qualified application
release; adopting rebuilt package bytes requires separate installed acceptance.
