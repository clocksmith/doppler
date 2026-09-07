# Installed local document search

This reference application composes qualified embedding and reranking Capsules
through the installed `doppler-gpu/host` API. `search.js` owns text/Markdown
indexing, cosine candidate retrieval, and candidate-to-document mapping. Document
parsing and index policy stay outside the inference runtime.

`installation.js` connects the existing OPFS backend to the Capsule artifact
store interface. Runtime metadata verification and application plan approval
precede artifact acquisition. Model bytes, signed metadata, explicit retained-use
decisions, and monotonically advancing release checkpoints survive browser
restart. The service worker retains the installed application and runtime files.

## Build

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
