# Pipeline Contract and Implementation Boundaries

This is a top-down, contract-first view of Doppler’s **actual** pipelines.
It is intentionally not force-fitted to any arbitrary fixed number of steps; each
boundary is only where input/output semantics change.

```mermaid
flowchart TD
  subgraph Entry["1) Command & Contract Boundary"]
    A["External caller<br/>(doppler CLI / browser runner / benchmark / test)"]
    B["src/tooling/command-api.js<br/>request parse + validation"]
    A --> B
  end

  subgraph Dispatch["2) Surface Dispatch"]
    C["tools/doppler-cli.js<br/>--surface + mode routing"]
    D["src/tooling/node-command-runner.js"]
    E["src/tooling/node-browser-command-runner.js<br/>Playwright relay + command-runner.html"]
    F["src/tooling/browser-command-runner.js<br/>in-browser execution"]
    B --> C --> D
    C --> E --> F
  end

  subgraph Conversion["3) Conversion Command (offline artifact path)"]
    CVT["doppler convert"]
    G["converter runtime<br/>src/converter/**/*.js"]
    H["src/browser/browser-converter.js<br/>shared manifest builder + conversion helpers"]
    I["Output: RDRR manifest + shards + tokenizer assets"]
    C -->|if command=convert| CVT --> G --> H --> I
  end

  subgraph Runtime["4) Runtime Inference Path"]
    J["doppler debug/bench/test-model"]
    K["Artifact resolution + model metadata<br/>(src/storage/shard-manager.js, docs/formats.md)"]
    L["Manifest preflight + schema validation<br/>(src/formats/rdrr, config merge)"]
    M["Config assembly<br/>src/config/ + runtime overrides"]
    N["Loader init + tensor binding<br/>src/loader/doppler-loader.js"]
    O["Token pipeline + prompt shape<br/>src/inference/pipelines/text.js, tokenizer"]
    P["Prefill<br/>src/inference/pipelines/text/generator-steps.js"]
    Q["Decode loop + KV cache update<br/>src/inference/pipelines/text/generator.js"]
    R["Sampling & stopping<br/>src/inference/pipelines/text/sampling.js"]
    S["Materialization<br/>result + metadata + traces"]
    C -->|if infer/bench/test| J
    J --> K --> L --> M --> N --> O --> P --> Q --> R --> S
  end
```

## Boundary Map (I/O + Implementation)

| # | Boundary | Input | Implementation (main) | Output |
|---|---|---|---|---|
| 1 | Command normalization | Raw CLI/web call, flags | `tools/doppler-cli.js`, `src/tooling/command-api.js` | Canonical request + runtime intent |
| 2 | Surface dispatch | Canonical request + mode | `src/tooling/node-command-runner.js`, `src/tooling/node-browser-command-runner.js`, `src/tooling/browser-command-runner.js` | Surface-specific execution |
| 3 | Conversion preflight | Source path + converter config | `src/converter/*.js`, `src/browser/browser-converter.js` | RDRR artifact set (manifest + shards) |
| 4 | Artifact resolution | `modelId`/`modelUrl` + runtime intent | `tools/doppler-cli.js`, `src/storage/shard-manager.js` | Manifest URI + shard/cached assets |
| 5 | Manifest/config contract merge | Manifest + config overrides | `src/config/**/*.js`, `src/config/runtime.js` | Resolved config with source metadata |
| 6 | Model loading | Resolved manifest + config | `src/loader/doppler-loader.js`, `src/formats/rdrr` | GPU-ready weights + kv/tensor locations |
| 7 | Prompt shaping | Raw prompt + tokenizer config | `src/inference/tokenizer.js`, `src/inference/pipelines/text/*.js` | Token IDs + generation options |
| 8 | Prefill | Full prompt token sequence + KV-empty state | `src/inference/pipelines/text/generator-steps.js` | Seeded KV cache + first logits |
| 9 | Decode step | Prefill state + step options | `src/inference/pipelines/text/generator.js`, `src/inference/pipelines/text/layer.js` | Next token + updated cache/state |
| 10 | Output materialization | Stream + trace metadata | CLI/test/bench handlers + debug trace modules | Final response JSON/text + perf/safety metadata |

## Why this map is honest to current behavior

- Conversion is not in the inference loop; it is a separate command path that produces
  artifacts consumed later.
- Loader and inference are separated by contract boundaries (request -> artifact ->
  merged config -> loaded tensors -> execution).
- There is a single runtime contract across browser/node runners; surface differences are
  dispatch/transport details, not separate semantics.
- Prefill and decode share kernel modules but differ in attention/cache behavior and
  prompt-length assumptions.
