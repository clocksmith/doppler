# Doppler Onboarding Tooling

This page defines the code-first, machine-checkable onboarding path for models,
conversion configs, kernel paths, and runtime behavior presets.

## Why this exists

New capabilities should be:

- Deterministic and scriptable.
- Fail fast on cross-file consistency issues.
- Easy to trace back to concrete artifacts.

`tools/onboarding-tooling.js` implements:

1. `check` – a full consistency sweep across conversion configs, model presets,
   kernel paths, compare configs, and harness metric contracts.
2. `scaffold` – generated stubs for recurring onboarding tasks.

## check mode

Command:

```bash
node tools/onboarding-tooling.js check [--root <repo-root>] [--strict] [--json]
```

### What is validated

- Model presets:
  - JSON shape and required IDs.
  - `extends` references to known presets with cycle detection.
  - Kernel path IDs referenced by `inference.kernelPaths`.
  - Registration visibility via `src/config/loader.js`.
- Runtime presets:
  - JSON shape and runtime shape checks.
  - ID uniqueness and `extends` chain/cycle validation across nested directories.
  - Missing `runtime` is flagged for visibility.
- Conversion configs:
  - Conversion output target format (`output.modelBaseId`, `output.baseDir`).
  - Preset lookup via `presets.model`.
  - Default kernel path references.
- Kernel path registry:
  - `registry.json` entry uniqueness.
  - `file` / `aliasOf` correctness.
  - Kernel path JSON load + step schema (`op`, `kernel`).
  - Step kernel file existence in `src/gpu/kernels/`.
- Compare harness contracts:
  - `compare-engines.config.json` profile entries and ids.
  - `compare-metrics.json` metric IDs.
  - Required metric coverage in `benchmarks/vendors/harnesses/*.json`.

### Output

- default: human-readable grouped issues.
- `--json`: emits a structured report:
  - `status`
  - `issues`
  - `summary` (errors/warnings counts)
  - `metadata`:
    - check counts
    - compare-profile coverage
    - kernel status coverage
- Compare checks also validate:
  - each compare profile maps to a known model preset or conversion model ID
  - every compare metric appears in both harness path maps

### Exit behavior

- Any error => exit code 1.
- `--strict` also treats warnings as failures.

## scaffold mode

Command:

```bash
node tools/onboarding-tooling.js scaffold --kind <model|conversion|kernel|behavior> --id <id> [flags]
```

### Shared flags

- `--id <id>` required.
- `--output <file>` optional explicit output path.
- `--force` overwrite existing files.
- `--status-reason <text>` optional kernel registry reason (`kernel` scaffold only).

### model scaffold

```bash
node tools/onboarding-tooling.js scaffold --kind model --id my-new-model
```

Generates:

`src/config/presets/models/my-new-model.json`

Includes an extendable starter preset with placeholder detection fields.

The command also prints copy/paste registration hints for
`src/config/loader.js` (preset import + detection-order placement).

### conversion scaffold

```bash
node tools/onboarding-tooling.js scaffold \
  --kind conversion --id my-new-model \
  --family gemma3 --preset gemma3 \
  --base-dir models/local --default-kernel-path gemma3-f16-fused-f16a-online
```

Generates:

`tools/configs/conversion/gemma3/my-new-model.json`

Includes output model/preset binding and default kernel path.

### kernel scaffold

```bash
node tools/onboarding-tooling.js scaffold --kind kernel --id gemma3-f32-audit-path
```

Generates:

`src/config/presets/kernel-paths/gemma3-f32-audit-path.json`

The command also prints a registry entry snippet that can be pasted into
`src/config/presets/kernel-paths/registry.json`.

### behavior scaffold

```bash
node tools/onboarding-tooling.js scaffold --kind behavior --id super-flash-memory-15-0
```

Generates:

`src/config/presets/runtime/modes/super-flash-memory-15-0.json`

Includes a runtime preset scaffold (`name`, `description`, `runtime.inference`).

## Canonical model workflow (adaptable to any model)

Use this when validating a conversion config end-to-end:

`--surface auto` is transport resolution only:
- Node runner attempt first for harnessed commands; browser relay is used only when Node
  WebGPU transport is unavailable.
- Command intent and contract remain unchanged across transports.

1. Convert the source into RDRR.
2. Run an inference correctness check (`verify`).
3. Run a performance benchmark.

```bash
# Required:
INPUT_PATH=/path/to/source/model
CONVERSION_CONFIG=tools/configs/conversion/<family>/<model-variant>.json
MODEL_ID=gemma-3-1b-it-wf16-ef16-hf16-f16

# 1) Convert
node tools/doppler-cli.js convert --config "{
  \"request\": {
    \"inputDir\": \"$INPUT_PATH\",
    \"convertPayload\": {\"converterConfig\": $(cat "$CONVERSION_CONFIG")}
  }
}"

# 2) Correctness check
node tools/doppler-cli.js verify --config "{
  \"request\": {\"suite\": \"inference\", \"modelId\": \"$MODEL_ID\", \"runtimePreset\": \"modes/debug\"},
  \"run\": {\"surface\": \"auto\"}
}" --json

# 3) Benchmark
node tools/doppler-cli.js bench --config "{
  \"request\": {\"modelId\": \"$MODEL_ID\"},
  \"run\": {\"surface\": \"browser\", \"bench\": {\"save\": true}}
}" --json
```

Notes:

- `request.runtimePreset` is optional on `bench` and `verify`; include only when you want an explicit runtime profile.
- `bench` defaults to browser, headless mode, and cache warm.
- Use the same pattern for any model family: swap `MODEL_ID` + `CONVERSION_CONFIG`.

Example multi-variant loop for Gemma-3:

```bash
for cfg in \
  tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f16.json \
  tools/configs/conversion/gemma3/gemma-3-1b-it-wf16-ef16-hf16-f32.json \
  tools/configs/conversion/gemma3/gemma-3-1b-it-wq4k-ef16-hf16-f16.json \
  tools/configs/conversion/gemma3/gemma-3-1b-it-wq4k-ef16-hf16-f32.json \
  tools/configs/conversion/gemma3/gemma-3-270m-it-wf16-ef16-hf16-f16.json \
  tools/configs/conversion/gemma3/gemma-3-270m-it-wf16-ef16-hf16-f32.json \
  tools/configs/conversion/gemma3/gemma-3-270m-it-wq4k-ef16-hf16-f16.json \
  tools/configs/conversion/gemma3/gemma-3-270m-it-wq4k-ef16-hf16-f32.json; do
  MODEL_ID="$(jq -r '.output.modelBaseId' "$cfg")"
  node tools/doppler-cli.js convert --config "{\"request\":{\"inputDir\":\"$INPUT_PATH\",\"convertPayload\":{\"converterConfig\":$(cat "$cfg")}}}"
  node tools/doppler-cli.js verify --config "{\"request\":{\"suite\":\"inference\",\"modelId\":\"$MODEL_ID\",\"runtimePreset\":\"modes/debug\"},\"run\":{\"surface\":\"auto\"}}" --json
  node tools/doppler-cli.js bench --config "{\"request\":{\"modelId\":\"$MODEL_ID\"},\"run\":{\"surface\":\"browser\",\"bench\":{\"save\":true}}}" --json
done
```

## Rebuilding only a manifest (no shard rewrite)

If you only changed conversion metadata (kernel path, session defaults, etc.) and want to avoid
re-quantizing/repacking shards, regenerate only `manifest.json`:

```bash
node tools/refresh-converted-manifest.js models/local/<model-id> \
  --config tools/configs/conversion/gemma3/gemma-3-270m-it-wf16-ef16-hf16-f16.json
```

Options:

- `--manifest <path>`: explicit manifest path (defaults to `<model-dir>/manifest.json`).
- `--model-id <id>`: override `manifest.modelId` in the refreshed file.
- `--skip-shard-check`: do not validate shard files exist before writing.
- `--dry-run`: validate and print what would change without writing.

## CLI examples

- Validate all onboarding artifacts:

```bash
npm run onboarding:check
```

- Fail on warnings too:

```bash
npm run onboarding:check:strict
```

- Create a behavior stub for a model-level experiment:

```bash
npm run onboarding:scaffold -- scaffold --kind behavior --id super-flash-memory-15-0
```

## Status checklist for onboarding work

- [x] Config contracts are machine-validated.
- [x] Cross-references are checkable before runtime changes.
- [x] Common onboarding artifacts can be scaffolded consistently.
- [x] Runtime preset validation is now part of the check flow.
- [x] Loader registration hints are emitted by the scaffold command.
- [x] Add strict CI gate integration (`onboarding-tooling` workflow).
