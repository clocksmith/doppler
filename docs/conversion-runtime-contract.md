# Conversion Runtime Contract

Canonical contract for conversion-time fields vs runtime-overridable fields.

Use this file as the source of truth when deciding whether a setting requires
re-conversion or can be changed per-run.

## Scope

- Conversion input: `request.convertPayload.converterConfig`
- Manifest output: `manifest.json`
- Runtime overlay: `runtime.inference.*`

## Single Sources of Truth

- Converter schema defaults: `src/config/schema/converter.schema.js`
- Conversion behavior: `src/converter/core.js`, `src/converter/quantization-info.js`
- Kernel-path and execution runtime behavior:
  - `docs/rdrr-format.md` (artifact contract)
  - `docs/style/config-style-guide.md` (runtime overlay rules)
  - `src/inference/pipelines/text/execution-v1.js` (strict runtime compile)

## Field Matrix

| Field | Contract owner | Persisted in manifest | Runtime-overridable | Re-convert required to change |
| --- | --- | --- | --- | --- |
| explicit conversion config fields | Conversion | Direct (`inference.*` authored in config) | No | Yes |
| `quantization.weights` | Conversion | Yes (`quantization`, tensor dtypes/layout) | No | Yes |
| `quantization.embeddings` | Conversion | Yes | No | Yes |
| `quantization.lmHead` | Conversion | Yes | No | Yes |
| `output.textOnly` | Conversion | Indirect (tensor set emitted) | No | Yes |
| `manifest.hashAlgorithm` | Conversion | Yes (`hashAlgorithm`, shard hashes) | No | Yes |
| `quantization.computePrecision` | Conversion-authored runtime default | Yes (`quantizationInfo.compute`) | Yes (through runtime/session policy) | No (for runtime behavior) |
| `inference.defaultKernelPath` | Conversion-authored runtime default | Yes | Yes (`runtime.inference.kernelPath`) | No (for runtime behavior) |
| `inference.session.decodeLoop.*` | Conversion-authored batching policy | Yes | Via `runtime.inference.session` | No (for runtime behavior) |
| `output.fast` | Reserved converter config flag | No active effect in current converter path | n/a | n/a |

Notes:

- Model ID suffixes are naming only; runtime policy is driven by manifest/runtime config.
- Storage dtype changes always require re-conversion.
- Runtime overlay for execution is strict: only `runtime.inference.session` is accepted.

## Runtime Precedence

Kernel-path resolution (low to high):

1. `manifest.inference.defaultKernelPath`
2. `runtime.inference.kernelPath`
3. per-run pipeline context override (internal runner context)

`null` is a valid "no explicit kernel path" result. Runtime must not invent an
implicit `'auto'` kernel path.

Execution compile order:

1. Start from `manifest.inference.session` and `manifest.inference.execution`
2. Expand compact tuples into resolved steps
3. Build inline kernel path and layer pipeline from resolved steps
4. Merge `session` into `runtime.inference.session`

Execution dtype ownership:

- config-selected kernel paths must already match runtime `activationDtype`
  and `kvDtype`
- manifest/model-selected kernel paths may seed those runtime dtypes only when
  the resolved session does not already specify conflicting values; conflicts fail closed

## Why this file exists

Other docs may describe subsets of this behavior for local context, but they
should link here rather than redefine the contract in multiple places.
