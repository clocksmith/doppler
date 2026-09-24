# Document search engineering and qualification

The beginner path is in [README.md](README.md). These instructions are for
maintainers reproducing model releases and installed-application evidence.

## Qualify the distributable

Run the clean consumer and physical acceptance together from the repository:

```sh
node tools/check-document-search-starter.js /absolute/new-consumer /absolute/qualification-config.json
```

The configuration supplies `fixturePath`, `requiredVendor`, `channel`,
`launchArgs`, `temporaryDirectory`, `timeoutMs`, and `recovery: true`. Use
`tests/fixtures/document-search-unseen.json` unchanged. The runner sets the
application and evidence directories after copying the starter and running
frozen `npm ci`; it never substitutes checkout runtime files. Omitting the
configuration runs the separately labeled shell-only check.

The physical path serves the copied starter through its own `server.js`, checks
archive/lock/asset/Capsule identity before and after execution, retains both model
sessions, and compares the retained corpus online and after restarting offline
with the server stopped. It exercises unchanged-document reuse, query/index
cancellation after GPU submission, superseded queries, interrupted atomic saves,
explicit closure, corrupted-model repair, index invalidation, and device loss.
Native boundary fault injection is identified in the receipt. A successful local
evaluation is not independent adoption.

Receipts include browser/hardware/probe identity, filesystem type, installation,
artifact I/O and verification, indexing, query, reopening, cancellation, and
closure observations. Filesystem backing and other workload conditions constrain
timing interpretation; no historical throughput result qualifies this package.

## Audit and publish exact missing artifacts

`tools/check-document-search-sources.js` accepts a JSON configuration containing
`applicationDir` and `receiptPath`. A `localRoot` audits the matching Capsule tree;
adding `stagingDir` stages only missing shard bytes under content-addressed paths
and includes model licenses. It checks signatures, identities, every declared
artifact, and source-map sizes/hashes before staging.

Upload the audited staging directory to `clocksmith/rdrr`, then run the checker
without `localRoot` or `stagingDir`, supplying the returned 40-character commit
as `revision`. The checker downloads and verifies every shard and only fills
missing URLs after all bytes pass. It never promotes a model, changes its
identity, or rewrites signed release history. Run `prepare.js` afterward to
regenerate the requirements and asset manifest.

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
