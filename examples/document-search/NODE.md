# Local document search in Node

[Download the Node starter 0.1.1](https://huggingface.co/clocksmith/rdrr/resolve/eafe756a6b11d19eafc9e030b720a2c92e63b2c7/document-search/node/releases/0.1.1/c0a5c9097725842067aa04808861bebcda2c5ba7bd296b891731dee25f470939/doppler-node-document-search-0.1.1.tgz)
and verify SHA-256 `c0a5c9097725842067aa04808861bebcda2c5ba7bd296b891731dee25f470939` before extracting it.
The [maintenance receipt](../../artifacts/document-search-maintenance-2026-09-26/README.md)
binds the archive to installed execution, models, hardware, and probes.
Its signed metadata permits fresh installation before **2026-09-28
00:59:12.562 UTC**. Later fresh installations need a new deliverable with renewed
signed metadata. Already accepted installations retain explicit offline use.

This application uses the same document controller and search implementation as
the browser starter. Embedding and reranking stay loaded together in interactive
mode. Documents and queries stay on this machine; installation downloads the
exact signed model artifacts.

The qualified configuration is Linux x64, Node 22.22.1, `webgpu` 0.4.0, and AMD
Radeon 8060S with RADV Mesa 26.0.3. The GPU needs `shader-f16`, subgroups, and the
buffer limits declared by both Capsules. Qualification used a 122 GiB host;
minimum system memory and other GPUs are not established. Requested GPU buffer
sizes are not measurements of physical residency. Electron and Bun are separate.

Download size is approximately 2.15 GB for the two models, plus this application
and npm dependencies. Allow at least 3 GB of free private storage for models,
atomic replacement files, and a small corpus; larger corpora need additional
space. Persistent-disk Node reopening, lifecycle recovery, and offline execution
passed in the [loading qualification](../../artifacts/document-search-diagnosis-2026-09-26/README.md).
Machine-reboot persistence and smaller-memory hardware remain unqualified. Each process
opening verifies the retained artifacts and prepares both models on the GPU.

From the extracted application directory:

```sh
npm ci --omit=optional --no-audit --no-fund
node node.js install ./search-data
node node.js index ./search-data ./documents.json
node node.js interactive ./search-data
```

Create `documents.json` with your own text or Markdown:

```json
[
  { "id": "notes", "title": "Project notes", "mediaType": "text/plain", "text": "The text you want to search." }
]
```

Keep each document's `id` stable across revisions. Indexing the same content again
reuses its embedding. Interactive mode accepts one query per line, reports its
measured duration, and keeps both models loaded. EOF closes the application.
`node node.js search ./search-data "your query"` runs one query and closes.

Ctrl-C requests cancellation. Already submitted GPU work must finish before
cleanup; cancellation is not immediate. Failed saves preserve the previous
document/index snapshot. After a device loss, close and reopen the application.
For a corrupt installation, run `node node.js repair ./search-data`; repair
verifies retained bytes and reacquires damaged artifacts. Downloads require
networking, while reopening and searching retained documents work offline.

Use a dedicated WebGPU process and one process per storage directory. The runner
rejects an existing provider rather than taking ownership of its device.
A crash leaves `.document-search.lock`
with its owner PID; confirm that process has stopped before removing the lock.
Uncommitted `*.pending` files can then be removed. Keep the directory private and
do not modify it while the application is running.

For integration, import `createNodeDocumentSearch` from `./node.js`, call
`app.controller.openRetained()` (or `install()` initially), then use
`indexDocuments()` and `search()`. Always `await app.close()` in `finally`.
Existing release checkpoints and explicit retained-local authorization remain
part of the installation; offline use cannot discover unseen revocations.
