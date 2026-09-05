# Electron Pack integration

This example composes the shipped `doppler-gpu/host` and `doppler-gpu/electron`
exports. It is integration code, not a customer application, model-quality
qualification, or evidence that a production release was adopted.

## Main and preload

Call `registerDocumentSearchReleaseMain()` with the application's durable
compare-and-swap store, release-decision and revocation signature verifiers,
and `ipcMain`. Call `exposeDocumentSearchReleaseBridge()` in preload to expose
the narrow `dopplerRelease` API. Its `resolveCurrent()` invokes the main-process
coordinator's fail-closed authorization check; `status()` is for inspection and
must not be used to bypass that check.

The main-process application must authenticate IPC senders and implement its
customer-authorization policy. A digest is a reference to authorization, not
proof that an arbitrary renderer is authorized to activate a release. The
example does not replace those application boundaries.

## Renderer

```js
import { createDocumentSearchHostRenderer } from './renderer.js';

const renderer = createDocumentSearchHostRenderer(window.dopplerRelease, {
  trustedSigners,
  acceptedTargetPlanDigests,
});

const receipt = await renderer.rerank({
  application: applicationBinding,
  query,
  documents,
  options: { benchmark: false },
}, { signal: abortController.signal });
```

`doppler-gpu/host` composes the existing WebGPU provider, verified artifact store,
and Doppler program adapter. It does not choose trusted publishers or accepted
TargetPlans for the application. Browser bundles select the browser export;
Node selects the native JavaScript host export. Bun remains experimental.
URL artifacts use HTTP by default; an explicit `artifactStore` also works with
a URL or a supplied Pack object. No Doe, Poolday, registry, or signing service is
required. Model data still has to be acquired, and the host must supply WebGPU.

For custom host composition, the existing
`createDocumentSearchRenderer(releaseState, runtimePorts)` remains available.
Its ports are explicit:

- `device` exposes the chosen WebGPU device and its observed profile.
- `packSource.fetchPack(path)` resolves the authorized immutable Pack reference.
- `artifactStore` supplies the Pack's exact artifact bytes and hash/size checks.
- `trustedSigners` pins the signing authorities the application accepts.
- `programFactory` instantiates the selected, qualified TargetPlan. It must not
  infer another plan or load a different model. Pack Runtime validates the
  signed artifact closure and, for TargetPlan v2, the initial execution identity.

Keep these ports in the renderer or its explicitly configured worker. Do not
send WebGPU devices, functions, or signing private keys through the release IPC.
The main process remains responsible for release state, not model execution.

`applicationBinding` comes from the application's own pinned revision, workload,
and oracle contract. It contains `applicationId`, `applicationRevision`,
`applicationRevisionDigest`, `workload: { id, digest }`, and
`oracle: { id, digest }`. Do not copy the binding out of an untrusted Pack to
make a mismatch pass. Positional `rerank(query, documents)` is not supported.

`renderer.rerank(request, openOptions)` owns its session and closes it on success
and failure. It rechecks current-release authorization after loading and before
returning the result, rejects an opened Pack with a different identity, and
preserves the primary error if cleanup also fails. Cancellation prevents a
cancelled result from being returned and closes any completed load; it does not
promise preemption of an already-submitted GPU command or a non-cooperative
program factory.

For multiple calls on an explicitly retained session, `renderer.openCurrent()`
avoids reloading the model. The caller must close that session and manage
invalidation on activation, revocation, or device loss. Its authorization check
occurs at open; it is not a continuously authorized lease. No implicit session
cache or automatic fallback is installed by this example.

## Verification

```bash
npm run production-release:check
npm run package:smoke
node tools/check-packed-package.js --retain /absolute/path/to/new-evaluation-bundle
```

The release check includes one connected, signed-fixture episode: install,
explicit activation, rerank, rejected candidate, second release, restart,
revocation, and customer-requested rollback. It uses a synthetic program/device
and simulated IPC. The package smoke installs a tarball into an isolated
consumer, copies this example, executes the signed-Pack contract, and compiles
consumer types using only public package imports. Neither command proves
physical Electron inference, Qwen quality, external operation, or revenue.

`--retain` preserves the exact tarball, npm file inventory, installed consumer,
source commit and dirty-state declaration, command outputs, and terminal receipt
on success or failure. The destination must not exist. Verify the tarball's
SHA-256 against `receipt.json`; rerun `node electron-smoke.js` from the retained
`consumer/` directory without rebuilding Doppler. The consumer contains only a
signed synthetic Pack, not the Qwen model. Its type check records the repository's
TypeScript invocation separately from the runtime installation. Dirty source is
declared, not represented as reproducible from the commit alone.

## Physical component probe

With the pinned Electron version installed, a working desktop display, and the
actual Qwen artifacts available locally, run:

```bash
node tools/probe-electron-reranker.js \
  tools/policies/electron-qwen-component-probe.json \
  /absolute/path/to/retained-package-bundle \
  /absolute/path/to/qwen-3-reranker-0-6b-q4k-ehf16-af32 \
  /absolute/path/to/new-physical-observation
```

The checked-in policy selects Linux Electron and an AMD physical adapter. It
pins inputs, the top-document acceptance rule, and warmup/sample counts. The
probe rejects software adapters, hashes the real model artifacts, verifies the
retained tarball hash, and serves the installed package on loopback. It retains
the actual process arguments, raw reports, failures, and cleanup outcome.
Electron/Playwright are probe-host tools; they are not included in the runtime
tarball. The recorded launch arguments expose any host sandbox switches.

The verify and benchmark commands each load the model. Warm samples describe
inference after the benchmark's own load, not a warm application restart.
Loopback HTTP does not establish internet installation performance or a cold OS
cache. A passing component probe is not signed-Pack execution or an incumbent
comparison. The synthetic Pack smoke and this physical component observation
must not be combined into an unobserved end-to-end claim. That proof still needs
the qualified Qwen Pack and a connected physical application episode.

## Source-qualified Pack evaluation

The repository tools `capture-reranker-source-reference.py`,
`qualify-reranker-electron.js`, and `build-reranker-evaluation-pack.js` retain
separate source-reference, model-qualification, build, and Pack-execution
artifacts. See [the evaluation workflow](../../docs/integration/reranker-evaluation.md).
The Python reference is build/evaluation tooling, not a runtime dependency.

Rerank qualification binds the actual input tokens, yes/no logits, scoring
policy, numerical tolerances, and exact ranking. A generation transcript or a
passing top-document check alone cannot qualify it. An evaluation Pack is not a
catalog promotion or evidence of external adoption.
