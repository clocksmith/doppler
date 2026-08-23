# Pack Runtime API

## Purpose

`doppler-gpu` is the Pack-native production entrypoint. It validates a signed
Doppler Pack, verifies every artifact and reachable WGSL module, selects one
already-qualified TargetPlan for the observed device, binds resources, and
executes the declared commands. It does not load unsigned manifests, infer a
model family, choose kernels, or supply a development signing key.

`doppler-gpu/runtime` is an exact alias of the same entrypoint.

## Import path

```js
import {
  DOPPLER_VERSION,
  RUNTIME_CORE_VERSION,
  createDopplerRuntime,
  createFetchPackArtifactStore,
  openPack,
} from 'doppler-gpu';
```

## Required authority

The application must inject all four execution ports:

- `device`: the concrete WebGPU resource and capability adapter.
- `artifactStore`: reads and hashes Pack artifacts.
- `trustedSigners`: the explicit signer-ID to public-JWK trust map.
- `programFactory`: loads generic execution mechanisms for the selected plan.

No port has a behavior-changing default. Missing ports fail before Pack
validation. The compatibility facade's `modelLoadOptions` are prohibited on
Pack execution because they could rewrite signed execution policy.

## Minimal example

```js
import { createFetchPackArtifactStore, openPack } from 'doppler-gpu';

const packUrl = new URL('./model.pack.json', import.meta.url).href;
const pack = await (await fetch(packUrl)).json();
const artifactStore = createFetchPackArtifactStore(packUrl);

const session = await openPack(pack, {
  device,
  artifactStore,
  trustedSigners: new Map([[signerId, signerPublicKey]]),
  programFactory,
});

const { text, tokenIds } = await session.generateText(generationOptions);
console.log(text, tokenIds, session.selectedTargetPlanDigest);
await session.close();
```

`generationOptions` contains only SessionPlan values admitted by the Pack,
including prompt tokens, output limit, sampling tuple, stop policy, and abort
signal. It cannot change graph topology, precision, fusion, kernel selection,
KV layout, or memory strategy.

## TargetPlan v2 initialization gate

For a `doppler.target-plan/v2` target, `programFactory` must return a program
whose `getInitialExecutionIdentity()` reports the resolved graph, reachable
kernel closure, dtype lane, fusion set, KV layout, memory policy, execution-plan
digest, and runtime-engine identity. Runtime compares that canonical identity
with the signed TargetPlan before creating the resource binder or dispatching
prefill. Any mismatch fails closed.

Newly forged ModelIR v2 targets require
`doppler.initial-execution-identity/v2`. Its signed `programLoadPolicy` contains
only the fully resolved runtime `session` and `compute` JSON required to create
the declared program. Public `modelLoadOptions` remain prohibited. Runtime
applies this Pack-owned policy before loading the mature execution mechanism,
then independently observes and compares the complete identity before prefill.
Identity v1 remains accepted only for compatibility with already frozen Pack v0
targets.

## Architecture boundary

JSON declares semantic and execution policy. JavaScript validates, binds, and
orchestrates. WGSL computes only declared tensor operations. Every reachable
shader is content-addressed; changing shader bytes changes TargetPlan and Pack
identity.

## Code pointers

- [Pack runtime entrypoint](../../src/pack-runtime.js)
- [Runtime composition root](../../src/client/runtime/composition-root.js)
- [Initial execution identity](../../src/config/initial-execution-identity.js)
- [Compatibility API](compat.md)
