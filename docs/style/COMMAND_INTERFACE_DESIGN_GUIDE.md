# DOPPLER Command Interface Design Guide

Design rules for the command interface, independent of the UI surface.
This applies to the CLI and any future browser-based command console.

---

## Goals

- **Config-only control**: commands are defined by config, not flags.
- **Explicit exit conditions**: every command declares how it ends.
- **Stable intent mapping**: one verb maps to one intent cluster.
- **Isolation when measuring**: benchmarks run without observer effects.
- **Deterministic verification**: tests are repeatable and reproducible.

---

## Intent Clusters

DOPPLER groups tool actions into three architectural clusters. Each cluster
has a distinct exit condition and is enforced in config.

### A) Verification (Gatekeeper)

- **Intent**: `verify`
- **Exit condition**: Boolean pass/fail + diagnostics
- **Examples**: test suites, correctness checks
- **Rules**:
  - Deterministic inputs
  - No interactive sessions
  - Diagnostics only on failure

### B) Investigation (Microscope)

- **Intent**: `investigate`
- **Exit condition**: Artifact or stream (trace, profile, logs)
- **Examples**: debug sessions, profiling, tracing
- **Rules**:
  - Observer effect is acceptable
  - Interactive or long-running sessions allowed
  - Artifacts must be saved or streamed

### C) Calibration (Yardstick)

- **Intent**: `calibrate`
- **Exit condition**: Scalar metrics (latency, tok/s, VRAM)
- **Examples**: benchmarks, load tests
- **Rules**:
  - Isolation required
  - No profilers, traces, probes, or debug instrumentation
  - Results must be comparable to baselines

---

## Config Contract (Single Source of Truth)

All command interfaces emit a config object with these fields:

- `cli.command` (run/test/bench/debug/convert/tool)
- `cli.suite` (optional, depending on command)
- `model` (required when the suite needs a model)
- `runtime.shared.tooling.intent` (verify/investigate/calibrate)

Commands are rejected if:
- `cli.command` is missing
- `runtime.shared.tooling.intent` is missing (for test/bench/debug)
- intent does not match the command
- calibrate intent enables tracing/profiling/probes

---

## Intent Mapping

| Command | Intent | Notes |
|---------|--------|-------|
| `test` | `verify` | Deterministic correctness gate |
| `debug` | `investigate` | Traces, profiling, probes allowed |
| `bench` | `calibrate` | Baseline metrics only |

Bench runs that enable profiling must switch intent to `investigate`; calibrate
must keep profiling/tracing off.

Maintenance commands (`convert`, `tool`) are config-only but do not require a
tooling intent unless they run harnessed workloads.

If a command needs mixed behavior, split it into two runs with two configs.
Do not overload a single run.

---

## Interface Rules

- The interface never mutates config values.
- The interface never supplies hidden defaults for model/command/suite.
- The interface cannot override runtime tunables (prompt, sampling, batching).
- All behavior changes are encoded in config and captured in artifacts.

---

## Artifacts and Outputs

- **Verification**: structured error log and exit code.
- **Investigation**: trace files, profiler output, or live stream.
- **Calibration**: JSON metrics + optional HTML report.

Outputs must be consistent across CLI and browser UIs.

---

## Cross-Surface Parity

Any new command capability must be:
1) Defined in config schema,
2) Validated in runtime config validation,
3) Available in both CLI and browser command surfaces.

Do not add UI-only switches.

---

## References

- `docs/style/CLI_STYLE_GUIDE.md`
- `docs/style/CONFIG_STYLE_GUIDE.md`
- `docs/style/BENCHMARK_STYLE_GUIDE.md`
