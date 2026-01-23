# Browser vs CLI Capabilities

This document summarizes current capabilities and known gaps between the
browser runtime and the original Node/CLI runtime. References are provided for
implementation context.

## Capability Matrix

| Area | Browser Runtime | Original Runtime (Node/CLI) | Key References |
|---|---|---|---|
| Conversion | In-app GGUF/safetensors -> RDRR; remote URLs with HTTP range + download-first fallback; tokenizer.json only; Q4_K_M col layout blocked; converter HTTP knobs in schema and UI. | Node converter CLI with filesystem access and CLI entrypoints for GGUF/safetensors conversion. | Browser: `src/browser/browser-converter.js:69`, `src/browser/browser-converter.js:84`, `src/browser/browser-converter.js:134`, `src/browser/browser-converter.js:173`, `src/browser/browser-converter.js:321`, `src/browser/tensor-source-http.js:25`, `src/browser/tensor-source-download.js:216`, `src/config/schema/converter.schema.js:25`, `demo/index.html:337` · Node: `src/converter/node-converter/index.js:7` |
| Loading | OPFS/IDB shard storage via shard manager; OPFS backend support with sync access handle when available. | CLI drives browser harness for loading via tests/harness page; same loader/storage path executed in browser context. | Browser: `src/storage/shard-manager.js:46`, `src/storage/backends/opfs-store.js:27` · CLI harness: `cli/runners/inference.js:4` |
| Inference | WebGPU pipeline in browser; suites run via browser harness; UI-triggered diagnostics. | CLI runs suites (test/bench/debug) via runners and harness URLs for automation. | Browser: `src/inference/pipeline.js:482`, `src/inference/browser-harness.js:440`, `demo/diagnostics-controller.js:112` · CLI: `cli/runners/index.js:4`, `cli/runners/inference.js:6` |
| Tooling | Diagnostics UI with intent-gated runs + report export; optional browser tool runner for workspace tools. | CLI tooling intent enforcement and tool registry for scripts/bench tools. | Browser: `demo/diagnostics-controller.js:36`, `src/storage/reports.js:48`, `demo/tooling-controller.js:6` · CLI: `cli/index.js:350`, `cli/tools/registry.js:7` |

## Parity Goals and Gaps

### Conversion
- Browser gap: `tokenizer.model` is intentionally unsupported; browser conversion requires `tokenizer.json`.
  - Reference: `src/browser/browser-converter.js:173`
- Browser gap: column-wise Q4_K_M quantization is blocked due to streaming limits.
  - Reference: `src/browser/browser-converter.js:321`, `src/browser/quantization.js:181`
- Node/CLI advantage: full local filesystem and CLI conversion path for automation.
  - Reference: `src/converter/node-converter/index.js:7`

### Loading
- Browser: OPFS/IDB only, subject to storage quotas and persistence prompts.
  - Reference: `src/storage/shard-manager.js:46`
- Node/CLI: loads via browser harness in automation runs; no direct filesystem loader in Node.
  - Reference: `cli/runners/inference.js:4`

### Inference
- Browser: primary runtime for WebGPU inference and interactive runs.
  - Reference: `src/inference/pipeline.js:482`
- Node/CLI: orchestration and automation via harness URLs; enables CI/test workflows.
  - Reference: `cli/runners/index.js:4`, `cli/runners/inference.js:6`

### Tooling
- Browser: diagnostics runs with intent enforcement and report export; UI-driven.
  - Reference: `demo/diagnostics-controller.js:36`, `src/storage/reports.js:48`
- Node/CLI: tool registry and CLI-only scripts for kernel and manifest tooling.
  - Reference: `cli/tools/registry.js:7`

## Notes
- The browser conversion path already supports remote URLs via HTTP range, with
  automatic download-first fallback into OPFS/IDB when range is unavailable.
  - Reference: `src/browser/tensor-source-http.js:25`, `src/browser/tensor-source-download.js:216`
- Runtime config intent enforcement is applied in both CLI and browser diagnostics.
  - Reference: `cli/index.js:350`, `demo/diagnostics-controller.js:36`
