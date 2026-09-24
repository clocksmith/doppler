# Installed local document search

This reference application composes qualified embedding and reranking Capsules
through the installed `doppler-gpu/host` API. `search.js` owns text/Markdown
indexing, cosine candidate retrieval, and candidate-to-document mapping. Document
parsing and index policy stay outside the inference runtime.

This is the canonical implementation for Doppler's
[next product increment](../../docs/goals.md#next-product-increment-copy-and-run-local-search):
a complete copy-and-run local-search application. That contract requires reusable
sessions and indexes, application-owned cancellation/disposal, complete pinned
setup, honest progress, and clean-environment physical acceptance. The engineering
build below is retained for reproduction; it does not yet establish that beginner
onboarding is complete. Do not create a second search engine in the
[one-shot capability example](../capsule-capabilities/README.md).

Initial completion uses embeddings and reranking, not generation. The runnable
delivery must include complete pinned descriptors, accepted implementations,
publisher trust, and release metadata for the developer to inspect and adopt.
It must not require private signing keys. The application retains both healthy
sessions across queries and rebuilds incompatible indexes from preserved documents.

Local processing, cancellation/reuse, cleanup, and offline reopening are scoped
acceptance requirements, not automatic guarantees. Cancellation does not interrupt
submitted GPU commands; cleanup does not guarantee immediate physical reclamation.
Offline reopening needs retained application/runtime files, metadata, models, and
the index; it cannot discover unseen revocations. Declare host/provider and memory
requirements, and test network/logging behavior across the complete application.

## Quick Start (Copy and Run)

### 1. Prerequisites

- Chromium with the declared models' WebGPU features, including `shader-f16`.
- Node.js 18+ to install the frozen package and host the application.
- Disk and GPU capacity for both resident models, plus retained artifacts and the
  document index. A retention-cache budget is not a total memory limit.

`shard-sources.json` pins all 32 exact Capsule shards to immutable public URLs.
The [public source audit](../../artifacts/document-search-delivery-2026-09-23/public-source-audit.json)
verifies every downloaded size and hash. Model identities are unchanged. This
establishes acquisition completeness; installed real-model acceptance remains
separate and must pass before this candidate is presented as supported.

Before downloading, the page shows each model's size, `shader-f16` and `subgroups`
requirements, storage needs, and unavailable sources. Both models remain resident;
feature availability alone does not guarantee enough GPU or system memory.

### 2. Install and Start
```sh
cd examples/document-search
npm ci --omit=optional
npm start
```
The entire directory can be copied outside the repository, including `vendor/`.
The lockfile pins the included runtime archive by integrity; no private signing
keys or developer-local paths are needed. `--omit=optional` omits native Node GPU
providers because this starter executes models in the browser. `npm ci` rejects
dependency/lock mismatches rather than rewriting the lock.

Installation and prestart run `prepare.js`: it checks archive integrity, inventories
the installed runtime and application assets, and generates `application-assets.js`
and `build-receipt.json`. The server only resolves runtime files from this starter's
`node_modules/doppler-gpu`, never `../../src` or a manually copied runtime tree.
The server starts at `http://127.0.0.1:8080/index.html`.

### 3. Usage Walkthrough
1. Open `http://127.0.0.1:8080/index.html` in a WebGPU-enabled browser.
2. Check **"Keep these model releases for offline use"**.
3. Click **"Install models"** to download and verify the pinned embedding and reranker weights.
4. Under **Documents & Search**, click **Choose Files** and select files from `samples/` (e.g., `sourdough.md`, `tire.txt`, `eclipse.md`, `solar.txt`, `git.md`, `starter-motor.txt`).
5. Click **"Save document index"** to index your documents locally into OPFS.
6. Type a query into the search box (e.g., *"How do I change a flat tire?"* or *"How does wild yeast fermentation work?"*) and press Enter or click **"Search"**.
7. After installation and indexing succeed, close models and the browser, stop the
   server, disconnect networking, and reopen the same URL/profile. Choose **Open
   installed models**. Model search acceptance must establish that documents and
   queries remain local; application-shell caching alone does not establish this.

## Architecture and Contracts

`controller.js` coordinates model lifecycle, session retention, and tracked operations.
Cancellation buttons follow controller state, including indexing. Explicit **Close
models** awaits active work and releases sessions; `controller.dispose()` additionally
rejects future work, prevents late state publication, and reports cleanup failures.
Do not rely on page-unload callbacks to finish asynchronous cleanup.

`document-store.js` commits documents and vectors in one integrity-checked snapshot,
using asynchronous atomic file replacement. Interrupted staging, failed writes,
or cancellation before close preserve the previous snapshot. Starting atomic close
is the commit point: later cancellation does not claim that a completed publication
was rolled back. Disposal waits for it but never publishes a late in-memory index.
SyncAccessHandle/in-place writes are not supported for this application contract.
Legacy installations are read without mixing independently saved document/index files.

`document-import.js` gives documents independent IDs; equal filenames do not collide.
Unchanged retained imports keep their IDs. Content hashes are recomputed for embedding
reuse rather than accepted from untrusted import metadata.
`search.js` owns document hashing, cosine candidate retrieval, and reranking without framework dependencies.
`installation.js` manages local artifact caching in OPFS, checksum verification, and rollback protection.
`service-worker.js` caches application assets for complete offline availability.

From the repository, `node tools/check-document-search-starter.js` copies the starter
outside the checkout, runs frozen `npm ci`, verifies every served asset, and reopens
the actual application shell with the server stopped and browser networking disabled.
Its retained receipt explicitly says `physicalExecution: false`. UI cancellation
tests use synthetic model ports and are separate from GPU qualification. A final
model receipt must bind this installed archive, generated manifest, signed model
identities, environment, fixture, and probe; no source copying after qualification.

## Engineering build and reproduction

First retain a passing installed-package smoke bundle and source-qualified
embedding and reranking Capsule builds. The build tool accepts a JSON file:

Create the package consumer outside the repository and any directory whose
ancestors contain `node_modules`. Node otherwise resolves ancestor dependencies,
which defeats the standalone-provider isolation check. Install repository
dependencies with `npm ci` and Chromium with `npx playwright install chromium`.
On Linux, install Chromium system dependencies with
`npx playwright install-deps chromium` when the host does not already supply them.

```sh
TMPDIR=/var/tmp node tools/check-packed-package.js --retain /var/tmp/doppler-package-evidence
```

```json
{
  "packageBundlePath": "/absolute/installed-package-bundle",
  "outputDir": "/absolute/new-document-search-application",
  "previousApplicationDir": null,
  "models": [
    { "role": "embedding", "capsuleRoot": "/absolute/embedding-capsule" },
    { "role": "reranker", "capsuleRoot": "/absolute/reranker-capsule" }
  ],
  "search": {
    "dimension": 1024,
    "candidateCount": 3,
    "queryPrefix": "",
    "documentPrefix": ""
  },
  "loading": { "maxRetainedArtifactBytes": null },
  "storage": {
    "opfsRootDir": "doppler-document-search",
    "useSyncAccessHandle": false,
    "maxConcurrentHandles": 1
  }
}
```

```sh
node tools/build-document-search-app.js /absolute/build-config.json
```

Use `previousApplicationDir: null` only for the initial application. For later
builds, supply the previous application directory with its retained release
history. An unchanged release reuses its signed events. Changed application
bindings append an event; known denial, rollback, and fork errors reject the
build. Preserve any newer checkpoint or release history observed by deployed
applications; a build cannot discover unseen revocations from an offline copy.

The builder checks the archived package receipt, copies installed package files,
migrates the supplied current v2 Capsules into separate v3 artifacts, and signs
application bindings using the supplied local evaluation authorities. It does
not rewrite the input Capsules, publish a package, or update another repository.
Keep private signing custody outside the generated application directory.

The search geometry and prefixes must match the qualified embedding contract.
`maxRetainedArtifactBytes` controls the runtime's verified-file retention budget;
`null` preserves unlimited retention. A smaller budget may increase reads and
verification work. It does not cap total process or GPU memory.

## Use

Serve the generated directory over HTTPS or localhost. For repository-local use:

```sh
node --input-type=module <<'JS'
import { createStaticFileServer } from './src/tooling/node-browser-command-runner.js';
const server = await createStaticFileServer({
  rootDir: '/absolute/new-document-search-application', host: '127.0.0.1', port: 8080,
});
console.log(server.baseUrl + '/index.html');
JS
```

1. Open the application and explicitly choose retained local use.
2. Install the models, select text or Markdown files, and save the document index.
3. Search, close the browser, and reopen the same address and browser profile
   with networking disabled. Choose **Open installed models**.

Normal opening rejects damaged files. **Repair damaged model files** explicitly
reacquires missing or damaged artifacts after signature and release checks.
It preserves valid artifacts and release checkpoints. Cancelled or failed
installation does not commit a completed installation record, and retries wait
for outstanding writes to settle. Storage errors remain visible to the user.

An incompatible embedding identity invalidates the old vector index. Retained
text remains available to **Rebuild retained index**; damaged retained text or
index data is rejected. Browser-managed storage remains subject to eviction or
user deletion. Offline retained use cannot observe new remote revocations.

Device loss rejects active sessions. Close them and explicitly reopen the
installed models; pipeline cleanup preserves an acquired replacement device and
cannot restore a device already known to be lost.

## Qualification

`tools/qualify-document-search.js` runs the frozen
`tests/fixtures/document-search-unseen.json` corpus through actual installed
models, retains BM25 results, closes the browser and server, and repeats search
offline in the same profile. Its `recovery` option exercises cancellation,
Chromium origin quota exhaustion, physical cached-byte corruption and repair,
index invalidation, and device destruction/reopening. The index-upgrade fault
injects an incompatible prior binding; it is distinct from qualifying a new
embedding release.

`tools/qualify-document-search-update.js` accepts `priorQualification`,
`nextApplicationDir`, and `outputDir` paths. It copies the prior browser profile,
explicitly updates the application service worker, installs a different qualified
embedding Capsule, checks that old release checkpoints remain intact, rejects
the stale index, and rebuilds from retained text. It then closes the browser and
server and repeats search offline. Context-level request observations include
service workers; online reembedding and search must make zero network requests.
This is local application acceptance, separate from publication or external use.

`tools/compare-document-search-retention.js` compares passing installations using
an explicit interleaved order and fresh browser processes. It records renderer
RSS, artifact counters, opening and first-query costs, and unchanged query
results. Source qualification, application quality, memory measurements, and
external adoption are separate evidence categories. The local fixture is a
bounded regression corpus, not evidence of general retrieval superiority.

For the separately qualified Node reranker lane, install the retained
`webgpu-0.4.0.tgz` into the package bundle's `consumer` directory with
`npm install /absolute/webgpu-0.4.0.tgz --ignore-scripts --omit=optional --offline
--no-audit --no-fund`. Retain the resulting lockfile and install log. The source
Capsule directory must include `distribution/capsule-v3.json`,
`current-open-options.json`, and `release-checkpoints.json`. Its accepted plans
must explicitly qualify Node; a browser-qualified plan alone fails on Node.

`tools/qualify-installed-reranker-node.js` accepts one JSON config path:

```json
{
  "packageBundlePath": "/absolute/isolated-package-bundle",
  "capsuleRoot": "/absolute/node-qualified-reranker-capsule",
  "deniedRoot": "/absolute/preserved-revoked-reranker-capsule",
  "referencePath": "/absolute/frozen-reranker-reference.json",
  "outputDir": "/absolute/new-node-qualification",
  "requiredVendor": "amd",
  "providerCreateArgs": ["backend=vulkan", "enable-dawn-features=allow_unsafe_apis"],
  "repeatRuns": 2
}
```

The probe uses the installed runtime, disables network acquisition, checks the
unchanged source oracle, destroys and reopens the physical device, and verifies
the preserved denial and rollback rejection. It copies the release ledger into
the new output directory before checkpoint writes. It requires the denied
fixture's `retained-open-options.json`, `recovery-release-events.json`,
`recovery-checkpoint.json`, and signed distribution metadata. Input ledgers and
historical qualification receipts remain immutable.
