# DOPPLER Harness Interface Style Guide

Design rules for the browser harness and demo diagnostics UI.
DOPPLER is browser-only; all command surfaces emit runtime config.

---

## Goals

- **Config-first control**: runtime tunables live in config, not UI toggles.
- **Reproducibility**: same config + model + inputs produces the same run.
- **Minimal surface area**: harness modes are stable; add new ones sparingly.
- **Safe defaults**: avoid destructive or implicit behavior changes.

---

## Vocabulary

- **Harness mode**: `runtime.shared.harness.mode` (`kernels`, `inference`, `training`, `bench`, `simulation`).
- **Runtime preset**: named runtime config in `src/config/presets/runtime/`.
- **Runtime config**: merged config passed into the harness.
- **Runtime override**: `runtimeConfig` (JSON) or `runtimeConfigUrl` (URL).
- **Tooling intent**: `runtime.shared.tooling.intent` (`verify`, `investigate`, `calibrate`).

---

## Command Model (Browser)

### Intent Mapping

| Intent | Harness Mode | Use Case |
|--------|--------------|----------|
| `verify` | `kernels`, `inference`, `training` | Correctness validation |
| `investigate` | `inference` | Debugging, tracing, profiling |
| `calibrate` | `bench` | Performance baselines |

### Required Config Fields

- `runtime.shared.harness` (mode, autorun, modelId, skipLoad)
- `runtime.shared.tooling.intent`
- `runtime.inference` or `runtime.loading` overrides as needed

The harness must not mutate config values or add implicit defaults for model,
mode, or intent. If a tunable changes, it must be captured in config.

---

## URL Contract

The harness accepts **only** these URL parameters:

- `runtimePreset` (preset id)
- `runtimeConfig` (JSON-encoded runtime config)
- `runtimeConfigUrl` (URL to a runtime config JSON file)
- `configChain` (JSON-encoded config chain for debugging)

No per-field URL overrides are allowed.

Examples:

```
http://localhost:8080/tests/harness.html?runtimePreset=debug
http://localhost:8080/tests/harness.html?runtimeConfig=...JSON...
http://localhost:8080/tests/harness.html?runtimeConfigUrl=/configs/bench.json
```

---

## Logging and Output

- Runtime code uses the debug module, not `console.*`.
- Harness and demo entry points may print to the console.
- Use the DevTools filter or `DOPPLER.printLogSummary()` to inspect output.

Benchmark results should be saved as JSON under `tests/results/` when captured.

---

## See Also

- `command-interface-design-guide.md`
- `config-style-guide.md`
- `../testing.md`
