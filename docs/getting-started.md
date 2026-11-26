# Doppler Getting Started

Doppler is a JavaScript library for local model execution through WebGPU.
Start with a published package and an obtainable model; model preparation,
qualification, and benchmarking are separate engineering workflows.

## Published first run

This guide pins `dopplerx@0.2.0`, the current release.
Source revisions after it, including the unpublished 0.2.0 candidate, are in
the [changelog](../CHANGELOG.md); published versions are in the
[npm package history](https://www.npmjs.com/package/dopplerx?activeTab=versions).
The pinned quickstart includes the
`gemma3-270m` alias for the hosted Gemma 3 270M Q4K model:

```sh
npx --yes --package dopplerx@0.2.0 dopplerx --help
npx --yes --package dopplerx@0.2.0 dopplerx --list-models
npx --yes --package dopplerx@0.2.0 dopplerx --model gemma3-270m --prompt "Describe WebGPU briefly"
```

Use Node 22 (what CI and qualified runs use) and a working WebGPU provider; the
package declares no `engines` minimum and older Node versions are untested. The published package has
optional native provider dependencies; JavaScript does not remove host GPU,
driver, or native installation requirements. A provider error needs an explicit
supported configuration, not a silent CPU or cloud fallback. Browser use requires
a WebGPU-capable browser; browser and Node qualification are separate.

The package's registry pins model revision
`a8591b20bce7c22d75becde1315482e76ff85fc9`. The six hosted weight shards total
399,357,184 bytes, plus tokenizer, manifest, package, and cache overhead.
This guide makes no fresh physical inference, minimum-memory, or
offline-reopening qualification claim. See the [model evidence](model-support-matrix.md) for exact support
scope. First execution downloads model assets. Retain required assets before
expecting offline operation.

## Package name

The package was published as `doppler-gpu` up to 0.7.x (last 0.7.8) and is now
`dopplerx`, whose version numbering restarts at 0.1. `doppler-gpu` remains a
separate npm package; new installs and imports use `dopplerx`. The `dopplerx`
package still ships a `doppler-gpu` bin alias for the quickstart. Older text
that names `doppler-gpu` versions or subpaths describes the former package.

## Embed the published compatibility path

Install the exact release:

```sh
npm install --save-exact dopplerx@0.2.0
```

The shortest embedding path is `dopplerx/compat`, the manifest-loading
[compatibility facade](api/compat.md). It is tier `experimental` in the
[subsystem support matrix](subsystem-support-matrix.md); the primary
application surface is the [signed Run API](api/root.md) (`dopplerx/host`),
which opens signed Capsules under application-supplied trust.

```js
import { dr } from 'dopplerx/compat';

const session = await dr.open('gemma3-270m');
try {
  console.log(await session.generate('Describe WebGPU briefly'));
} finally {
  await session.close();
}
```

Keep sessions open across requests when appropriate and close them at
application shutdown. The [signed Run API](api/root.md), [choice scoring](api/choice-scoring.md), and
[resident partitions](distribution/resident-partition-execution.md) have their
own version, model, and host qualification boundaries. Consult those contracts
before using them. Applications own prompts, policy, trust, updates, and
presentation; Doppler owns loading and execution lifecycle.

## Retained search starter

The [0.1.1 search starter](../examples/document-search/README.md) demonstrates
embedding/reranking with resident sessions and application-owned indexing. Its
fresh-install signed eligibility ended September 28, 2026, 00:55:40.930 UTC.
Treat it as a versioned example until metadata and delivery are renewed; do not
change signed historical bytes or imply that existing installed acceptance
qualifies fresh installation. Its [Node guide](../examples/document-search/NODE.md)
and [retained receipts](../artifacts/document-search-maintenance-2026-09-26/README.md)
have separate host scopes.

## Next steps

- [API index](api/index.md): select operations qualified for your model and host.
- [Current priorities](../GOALS.md#current-priorities): one maintained implementation.
- [Engineering workflows](developer-guides/library-engineering-workflows.md):
  repository setup, conversion, verification, and benchmark commands.
- [Operations](operations.md): diagnostics and failure handling.
- [Performance and sizing](performance-sizing.md): measured limits and host requirements.
