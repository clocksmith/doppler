# Doppler Getting Started

Doppler is a JavaScript library for local model execution through WebGPU.
Start with a published package and an obtainable model; model preparation,
qualification, and benchmarking are separate engineering workflows.

## Published first run

The npm registry advertises `doppler-gpu@0.6.1`. Its quickstart includes the
`gemma3-270m` alias for the hosted Gemma 3 270M Q4K model:

```sh
npx --yes --package doppler-gpu@0.6.1 doppler-gpu --help
npx --yes --package doppler-gpu@0.6.1 doppler-gpu --list-models
npx --yes --package doppler-gpu@0.6.1 doppler-gpu --model gemma3-270m --prompt "Describe WebGPU briefly"
```

Use Node 20 or newer and a working WebGPU provider. The published package has
optional native provider dependencies; JavaScript does not remove host GPU,
driver, or native installation requirements. A provider error needs an explicit
supported configuration, not a silent CPU or cloud fallback. Browser use requires
a WebGPU-capable browser; browser and Node qualification are separate.

The package's registry pins model revision
`a8591b20bce7c22d75becde1315482e76ff85fc9`. The six hosted weight shards total
399,357,184 bytes, plus tokenizer, manifest, package, and cache overhead.
Package retrieval and those artifact endpoints were checked for this guide;
that check is not a fresh physical inference, minimum-memory, or offline-reopening
qualification. See the [model evidence](model-support-matrix.md) for exact support
scope. First execution downloads model assets. Retain required assets before
expecting offline operation.

## Embed the published compatibility path

Install the exact release:

```sh
npm install --save-exact doppler-gpu@0.6.1
```

```js
import { dr } from 'doppler-gpu/compat';

const session = await dr.open('gemma3-270m');
try {
  console.log(await session.generate('Describe WebGPU briefly'));
} finally {
  await session.close();
}
```

This is the published manifest-loading compatibility API. Keep sessions open
across requests when appropriate and close them at application shutdown.
The newer [signed Run API](api/root.md), [choice scoring](api/choice-scoring.md),
and [resident partitions](distribution/resident-partition-execution.md) describe
source contracts with their own release and qualification status. Do not assume
those exports or operations exist in npm 0.6.1. Applications own prompts, policy,
trust, updates, and presentation; Doppler owns loading and execution lifecycle.

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
