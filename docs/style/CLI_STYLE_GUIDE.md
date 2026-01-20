# DOPPLER Command Interface Style Guide

Design rules for the DOPPLER command interface. This guide covers both the
terminal CLI and a future browser-based command interface. Treat them as two
front-ends for the same command model and config pipeline.

---

## Goals

- **Config-first control**: runtime tunables live in config, not flags or UI.
- **Reproducibility**: same config + model + inputs produces the same run.
- **Single command model**: CLI and browser interface behave identically.
- **Minimal surface area**: commands are stable; add new verbs sparingly.
- **Safe defaults**: avoid destructive or implicit behavior changes.

---

## Vocabulary

- **Command**: top-level verb (e.g. `bench`, `test`, `debug`).
- **Suite**: sub-scope inside a command (e.g. `inference`, `kernels`).
- **Preset**: named runtime config in `src/config/presets/runtime/`.
- **Config ref**: preset name or config file path.
- **Runtime config**: merged config passed into the harness.
- **Harness**: browser runner that executes the command workload.

---

## Command Model

### Core verbs

Commands should map to intent and be stable:

- `debug`: interactive or trace-focused runs
- `test`: correctness validation
- `bench`: performance measurement

If new verbs are added, they must:
- map to an existing subsystem boundary,
- reuse the same harness pipeline,
- remain config-driven.

### Structure

Preferred format:

```
command <suite> [options]
```

Examples (conceptual, npm-agnostic):

```
doppler bench inference --config bench --model gemma-3-1b
doppler test kernels --config ci
doppler debug inference --config debug
```

The browser command interface should expose the same fields: command, suite,
config ref, model id, and harness options.

---

## Config Ownership (Non-Negotiable)

Runtime tunables are config-only. Command interfaces must not override:
- prompt selection
- max tokens
- sampling (temperature/topK/topP)
- trace categories or log levels
- warmup/timed runs

Allowed options should only select or load config:
- `--config <preset|file>`
- `--model <id>`
- harness-only toggles (e.g. `--headed`)

If a user needs to change a tunable, they must supply a config or preset. This
rule applies equally to CLI flags and browser UI controls.

---

## Output and Artifacts

- Commands must emit a compact, human-readable summary.
- Full results must be stored as JSON artifacts (bench/test harnesses).
- Browser UI should expose the same artifacts and paths as the CLI.

Benchmark output must include:
- model id
- prompt class
- run counts
- tok/s for prefill and decode
- GPU submit counts
- buffer pool stats

See `BENCHMARK_STYLE_GUIDE.md` for output schema and baseline rules.

---

## Validation and Errors

- Invalid configs fail fast with a clear error message.
- Use error codes where available (e.g. `DOPPLER_*`).
- Do not silently mutate or fill missing tunables in interface logic.
- Prefer “config is invalid” over “fallback to defaults”.

---

## Browser Interface Requirements

The browser command UI must:
- remain non-blocking (runs in a worker or async task queue),
- provide progress events (model load, warmup, run count),
- support cancellation with graceful cleanup,
- preserve logs and artifacts for the current run.

The browser UI must not add per-field overrides. It should only:
- pick a preset or config file,
- set the model id,
- choose harness flags (headed, timeout, profile dir).

---

## Logging

- Runtime code uses the debug module, not `console.*`.
- Command entry points may print directly to the terminal/UI.
- Browser UI should display the same log categories as CLI output.

---

## Compatibility and Parity

The command interface is a thin shell. Do not encode logic in CLI/UI that
should live in runtime config or the harness. Any new option must be:

1) in schema defaults, and
2) configurable via config files, and
3) surfaced in both CLI and browser UI.

If parity breaks, prefer fixing the config system, not adding UI exceptions.
