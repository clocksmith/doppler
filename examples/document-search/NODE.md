# Local document search in Node

[Download the standalone Node archive](https://huggingface.co/clocksmith/rdrr/resolve/fa3325d41d43bf7ec3a9dcd39e77587fefabdd89/document-search/node/releases/0.1.0/caa1f395d8a0943d9cd62442f87860a1ccbe6e5abb2eb3b5ff54375e4b1e1a6e/doppler-node-document-search-0.1.0.tgz)
and verify SHA-256 `caa1f395d8a0943d9cd62442f87860a1ccbe6e5abb2eb3b5ff54375e4b1e1a6e` before extracting it.
The [acceptance record](../../artifacts/document-search-maintenance-2026-09-25/README.md#node)
binds that archive to its installed runtime, models, hardware, and probes.

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
space. Persistent disk and reboot qualification remain outstanding. Each process
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
