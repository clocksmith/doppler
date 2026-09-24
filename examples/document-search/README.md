# Installed local document search

This reference application composes qualified embedding and reranking Capsules
through the installed `doppler-gpu/host` API. `search.js` owns text/Markdown
indexing, cosine candidate retrieval, and candidate-to-document mapping. Document
parsing and index policy stay outside the inference runtime.

This is the canonical implementation for Doppler's
[next product increment](../../docs/goals.md#next-product-increment-copy-and-run-local-search):
a complete copy-and-run local-search application. That contract requires reusable
sessions and indexes, application-owned cancellation/disposal, complete pinned
setup, honest progress, and clean-environment physical acceptance. The [engineering build](ENGINEERING.md) is retained for reproduction; installed application evidence and limitations are recorded below. Do not create a second search engine in the
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

## Copy and run

### 1. Requirements and supported configuration

This starter is qualified on **Linux, Chrome 146.0.7680.177, AMD Radeon 8060S
Graphics (RADV STRIX_HALO)**, with `shader-f16`, `subgroups`, and a maximum
buffer size of at least 64 MiB. The tested host has 122 GiB usable system RAM;
a minimum RAM or GPU-memory configuration has not been established. Both models
and their working buffers must fit simultaneously. Other devices and browsers
need separate qualification.

- Node.js 18+ installs the frozen dependency and runs the local server; inference
  executes in Chromium, without a Node GPU provider.
- Model acquisition is **2,145,462,810 bytes**: 1,200,631,926 for embeddings and
  944,830,884 for reranking. Allow additional space for runtime/application files,
  browser overhead, documents, indexes, and temporary writes.
- Keep the same browser profile and origin to reopen offline. Browser storage
  can be evicted or deleted; offline use cannot discover new revocations.

All 32 shards have immutable public URLs and passed full downloaded size/hash
checks. The model identities and trusted publishers are unchanged. Before
loading, the page displays sizes, required features, storage quota, and missing
sources; unsupported hardware or missing model files prevent installation.

The physical acceptance used a memory-backed Linux profile (`tmpfs`) because the
machine's disk was full. It proves browser-process restart, not survival across
machine reboot; use persistent storage for retained documents in ordinary use.
Disk-backed installation was rejected when space ran out. No durability or
performance claim for another filesystem follows from this result.

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

For the tested Linux GPU path, launch Chrome with a separate profile:

```sh
google-chrome --user-data-dir=/absolute/path/to/search-profile \
  --enable-unsafe-webgpu --enable-webgpu-developer-features \
  --disable-dawn-features=disallow_unsafe_apis --ignore-gpu-blocklist \
  --use-angle=vulkan --enable-features=Vulkan --disable-vulkan-surface \
  http://127.0.0.1:8080/index.html
```

Use an existing writable parent for the profile directory and keep that directory
for later offline use. The acceptance profile was under `/dev/shm`; it is temporary
across machine reboot. The page checks the features before acquiring model bytes.

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

## Measured application behavior

The [installed acceptance receipt](https://github.com/clocksmith/doppler/blob/main/artifacts/document-search-delivery-2026-09-23/browser-qualification.json)
binds the vendored archive, all application assets, both signed model identities,
browser, hardware, reference corpus, and qualification probe. All six reference
queries passed online and offline, with identical rankings and scores. Both
models stayed resident during search; indexing and search made no observed
network requests. This bounded corpus does not establish general retrieval
quality or superiority to another library.

| Measurement on the declared host | Observed |
| --- | ---: |
| Initial model installation and preparation, fresh browser profile | 128,184 ms |
| Indexing six reference documents | 1,871 ms |
| First query with both models loaded | 967 ms |
| Median of the next five queries | 906 ms |
| Offline model reopening | 49,182 ms |

These are observations from one run using `tmpfs`, not promised latency. The UI
shows installation and query timings on your machine. Cancellation, unchanged
index reuse, superseded queries, interrupted saves, corruption repair, index
rebuilding, explicit closure, and device-loss recovery passed in the same run.
Submitted GPU commands are allowed to finish after cancellation; cancelled results
are suppressed and healthy sessions remain reusable.

The inference dependency is the included `doppler-gpu-0.6.2.tgz`, pinned by its
lockfile and SHA-256 `625b8ed8c8e86c28a5b49e1e5ce43969885acd3df309f456eba719d3eeaa5539`.
This starter release does not publish or qualify a different npm archive.
Node model execution, packaged Electron, and Bun are outside this browser release.

Maintainers can reproduce installation and qualification using
[ENGINEERING.md](ENGINEERING.md). Independent integration and a second revision
remain evidence to collect; this internally operated reference is not adoption.
