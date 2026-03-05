# Pipeline Contract and Implementation Boundaries

Contract-first view of Doppler runtime boundaries.

```mermaid
flowchart TD
  subgraph Entry["1) Command and Contract Boundary"]
    A["External caller"]
    B["src/tooling/command-api.js\nrequest parse + validation"]
    A --> B
  end

  subgraph Dispatch["2) Surface Dispatch"]
    C["tools/doppler-cli.js\n--config + --surface routing"]
    D["src/tooling/node-command-runner.js"]
    E["src/tooling/node-browser-command-runner.js"]
    F["src/tooling/browser-command-runner.js"]
    B --> C --> D
    C --> E --> F
  end

  subgraph Conversion["3) Conversion Command"]
    CVT["doppler convert"]
    G["src/converter/**/*.js"]
    H["src/browser/browser-converter.js"]
    I["RDRR artifacts"]
    C -->|if convert| CVT --> G --> H --> I
  end

  subgraph Runtime["4) Runtime Inference Path"]
    J["doppler debug/bench/verify"]
    K["artifact resolution"]
    L["manifest preflight + schema validation"]
    M["config assembly"]
    N["loader init + tensor binding"]
    O["token pipeline + prompt shaping"]
    P["prefill"]
    Q["decode loop + KV cache"]
    R["sampling + stopping"]
    S["result materialization"]
    C -->|if debug/bench/verify| J
    J --> K --> L --> M --> N --> O --> P --> Q --> R --> S
  end
```

## Boundary map

| # | Boundary | Input | Implementation | Output |
| --- | --- | --- | --- | --- |
| 1 | Command normalization | raw CLI/web call | `tools/doppler-cli.js`, `src/tooling/command-api.js` | canonical request + intent |
| 2 | Surface dispatch | request + mode | node/browser runners | surface-specific execution |
| 3 | Conversion path | source path + conversion config | converter modules | RDRR artifacts |
| 4 | Artifact resolution | `modelId`/`modelUrl` | storage tooling | manifest URI + shard source |
| 5 | Config merge | manifest + runtime override | `src/config/**` | resolved config |
| 6 | Model loading | resolved manifest/config | `src/loader/**` | GPU-ready tensors + cache state |
| 7 | Prompt shaping | prompt + tokenizer config | `src/inference/**` | token ids + generation options |
| 8 | Prefill | prompt tokens + empty KV | text pipeline steps | seeded KV + first logits |
| 9 | Decode step | current state + step options | text pipeline loop | next token + updated state |
| 10 | Output materialization | stream + traces | command handlers | response + metadata |

## Plane interpretation

- JSON plane: contract and policy (`manifest`, presets, rule assets)
- JS plane: orchestration and validation
- WGSL plane: deterministic arithmetic only

Any unresolved selection path is a contract error before dispatch.

## Related

- System model: [architecture.md](architecture.md)
- Inference implementation details: [../src/inference/README.md](../src/inference/README.md)
- Format contracts: [rdrr-format.md](rdrr-format.md)
