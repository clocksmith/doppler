# Doppler CLI Reference

The canonical CLI entrypoint is `src/cli/doppler-cli.js`.

The `dopplerx` package ships four bins:

| Bin | Source | Purpose |
| --- | --- | --- |
| `doppler` | `src/cli/doppler-cli.js` | Contract-driven tooling CLI (this page). |
| `dopplerx` | `src/cli/doppler-quickstart.js` | First-run quickstart for local generation; see [getting started](getting-started.md). |
| `doppler-gpu` | `src/cli/doppler-quickstart.js` | Alias of `dopplerx`, kept from the package's former name. |
| `doppler-serve` | `src/cli/doppler-serve.js` | OpenAI-compatible localhost server. |

The command list below is complete; it follows `TOOLING_COMMAND_SET` in
`src/tooling/command-api-constants.js` plus the CLI-only commands dispatched in
`src/cli/doppler-cli.js`. `doppler --help` lists the same commands with their flags.

## Command Surface

```bash
node src/cli/doppler-cli.js <command> --config <request> [flags]
```

- `--config` is required for command-runner commands (`release` also accepts flags). CLI-only commands take their own flags; `program-bundle` also accepts `--config`.
- `--help` prints CLI usage.
- `--pretty` prints human-readable output.
- JSON output is default; `--json` is accepted for explicit automation.

### Supported commands

Command-runner commands (shared browser/Node contract in `src/tooling/command-api.js`):

| Command | Workload intent | Notes |
| --- | --- | --- |
| `bench` | `calibrate` | Requires `request.workload`. |
| `debug` | `investigate` | Requires `request.workload`. |
| `verify` | `verify` | Requires `request.workload` (except legacy `kernels` shape). |
| `diagnose` | `investigate` | Node-only operator-diff investigation command. |
| `convert` | `convert` | Node-only command. |
| `refresh-integrity` | maintenance | Node-only; recomputes `integrityExtensions` for a local RDRR model directory (`request.modelDir`; optional `manifestPath`, `blockSize`, `dryRun`, `skipShardCheck`). Rejects runtime-input fields. |
| `release` | production eligibility | Node-only; emits signed `eligible` or `blocked` evidence and never activates. |
| `lora` | operator lifecycle | Node-only command. |
| `distill` | operator lifecycle | Node-only command. |

CLI-only commands (dispatched in `src/cli/doppler-cli.js`, outside the command-runner contract):

| Command | Purpose | Notes |
| --- | --- | --- |
| `profiles` | discovery | Lists checked-in runtime profile IDs; no workload is executed. |
| `program-bundle` | maintenance/export | Artifact exporter; see [Program Bundle Export](#program-bundle-export). |
| `onboard inspect` | source intake | `--source <checkpoint-dir> --out <dir> [--family-intake <json>]`; writes `source-intake.json`, a conversion-config skeleton, and a contract-test plan before conversion. See [onboarding tooling](onboarding-tooling.md). |
| `boundary` | boundary evidence | Actions `capture`, `source-capsule`, `token-evidence`, `compare`; each requires `--out`. See the [evidence loop](evidence-loop.md). |
| `intake` | post-conversion validation | `--convert-config <path\|json> [--manifest <path>] [--model-dir <dir>] [--skip-convert] [--out <report.json>]`; exits nonzero when the report has blockers. |
| `bundle` | one-command Program Bundle | `--manifest <path> --out <dir>` (reference capture plus export; `--surface node\|browser`) or `--convert-config <path\|json> --out <dir>`. See [Program Bundles](integration/program-bundle.md#one-command-composition-doppler-bundle). |

## `--config` inputs

`--config` is polymorphic and can be provided as one of:

- **Inline JSON object**
  - `--config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"}}'`
- **File path**
  - `--config ./configs/cli-request.json`
- **URL**
  - `--config https://example.org/configs/cli-request.json`

The parsed object is contract-validated by `resolveConfigEnvelope()`:
- Either top-level object has `request` + optional `run`,
  or an old-style object where all fields belong to `request`.
- Runtime override fields can be set in either:
  - top-level `runtimeProfile`, `runtimeConfigUrl`, and `runtimeConfig`, or
  - inside `request`.
- `runtimeProfile`/`runtimeConfig`/`runtimeConfigUrl` should be defined in exactly one location.

## Model resolution

- The CLI resolves `request.modelId` from the external RDRR root
  `$DOPPLER_EXTERNAL_MODELS_ROOT/rdrr/`. When that variable is unset, the root
  is the first existing of `/Volumes/models`, `/Volumes/models2`, and
  `/media/$USER/models` (see `src/tooling/hf-registry/catalog-io.js`).
- For a model outside that root, set the request-level `request.modelUrl` to a
  `file://` path.
- Run `doppler profiles --json` to discover checked-in runtime profile IDs before
  choosing an investigation, verify or calibration profile.

## `--runtime-config` inputs

This flag is a compatibility alias for setting runtime overrides.
For harnessed commands, it accepts the same input shapes as `--config`.

If you use `--runtime-config`, do not put runtime override fields inside `--config` at the same time.

For harnessed commands (`bench`, `debug`, `diagnose`, `verify`) the same polymorphic formats are accepted:

- inline JSON object
- file path
- URL

Example:

```bash
node src/cli/doppler-cli.js bench \
  --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"}}' \
  --runtime-config '{"inference":{"sampling":{"temperature":0,"topK":1}}}' \
  --json
```

## `profiles` and `--runtime-profile`

Use `profiles` to discover checked-in runtime profile IDs before selecting a
verify, debug, diagnose, or bench runtime:

Discovery lists the profiles available in the installation. The npm package
includes general runtime defaults and the retained Qwen comparison profiles with
their ancestors. One-off probes, diagnostic campaigns, simulation settings, and
machine-specific deployment examples remain in the repository; `package.json`
owns their publication exclusions. Use a repository checkout for those profiles.

```bash
node src/cli/doppler-cli.js profiles --json
node src/cli/doppler-cli.js profiles --pretty
```

`--runtime-profile <id>` is a convenience alias for setting
`request.runtimeProfile` before command normalization:

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}' \
  --runtime-profile profiles/verbose-trace \
  --json
```

For profile IDs under `src/config/runtime/profiles/`, the `profiles/` prefix may
be omitted:

```bash
node src/cli/doppler-cli.js verify \
  --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32"}}' \
  --runtime-profile production \
  --json
```

`--runtime-profile` is intentionally narrow. It cannot be combined with
`--runtime-config`, `runtimeProfile`, `runtimeConfigUrl`, or `runtimeConfig`
inside `--config`; use one runtime input path per command.

## `--surface` and execution intent

- `--surface auto` (default) tries Node first; falls back to browser relay for supported harnessed commands.
- `--surface node` forces Node execution.
- `--surface browser` forces headless Chromium relay.

`convert`, `refresh-integrity`, `release`, `diagnose`, `lora`, and `distill` reject `--surface browser`.
`convert`, `refresh-integrity`, `lora`, and `distill` reject runtime-input fields in the Node operator surface.

Command-level surface support:

- `bench`, `debug`, `verify`: `auto|node|browser`
- `convert`, `refresh-integrity`: `auto|node` (`browser` is rejected)
- `release`: `auto|node` (`browser` is rejected)
- `lora`, `distill`: `auto|node` (`browser` is rejected)
- `diagnose`: `auto|node` (`browser` is rejected)
- `program-bundle`: no `--surface`; reads declared files and writes a JSON artifact
- `profiles`: no `--surface`; reads checked-in runtime config metadata only
- `onboard`, `boundary`, `intake`: no `--surface`; read and write local files
- `bundle`: `node|browser` for the `--manifest` form

## Production release command

The `doppler` bin ships in the `dopplerx` package. Use either an installed
binary or the explicit npm package form (pin the version you qualified, as the
[release action](../.github/actions/doppler-release/action.yml) does with
`dopplerx@$DOPPLER_VERSION`):

```bash
doppler release --manifest doppler-release.json \
  --out .doppler-release/evidence \
  --capsule-trusted-signers trust/capsule-signers.json \
  --fleet-trusted-signers trust/fleet-signers.json \
  --fleet-receipts evidence/windows.json,evidence/macos.json \
  --signing-private-key "$RELEASE_PRIVATE_JWK_PATH" \
  --signing-public-key "$RELEASE_PUBLIC_JWK_PATH" \
  --signing-authority doppler-release-authority

npx --package dopplerx@<version> doppler release --config release-command.json
```

The default CLI phase is `decide`. Customer-operated Windows and macOS agents
run the explicit qualification phase:

```bash
doppler release --action qualify \
  --manifest doppler-release.json \
  --target windows-x64-nvidia-webgpu \
  --device-identity evidence/windows-device.json \
  --out .doppler-release/windows \
  --capsule-trusted-signers trust/capsule-signers.json \
  --signing-private-key "$FLEET_PRIVATE_JWK_PATH" \
  --signing-public-key "$FLEET_PUBLIC_JWK_PATH" \
  --signing-authority customer-windows-agent
```

`decide` verifies the Capsule and every exact-device fleet receipt, then writes a
retained `candidate.capsule.json`, `release-decision.json`,
`known-exclusions.json`, `rollback-target.json`, and
`revocation-configuration.json`. A blocked decision also writes and retains
`failure-bundle.json`. A failed qualification or blocked decision still prints
its JSON result and exits nonzero. Neither phase deploys, activates, rolls back,
mutates the customer repository, or invents a fleet receipt. The customer
activation system must separately verify and act on an eligible decision.

## Program Bundle Export

`program-bundle` exports `doppler.program-bundle/v1` from a manifest plus a
browser/WebGPU reference report. It is intentionally a maintenance/export path,
not a shared browser command.

```bash
node src/cli/doppler-cli.js program-bundle --config '{
  "manifestPath": "models/local/gemma-3-270m-it-q4k-ehf16-af32/manifest.json",
  "referenceReportPath": "tests/fixtures/reports/gemma-3-270m-it-q4k-ehf16-af32/2026-03-18T13-33-38.973Z.json",
  "conversionConfigPath": "src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json",
  "outputPath": "examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json"
}'
```

The exporter fails if the reference report lacks prompt/output/token identity,
if a reachable WGSL kernel is undeclared, or if a kernel digest drifts from the
checked-in digest registry.

For a fresh proof run plus export, use the bounded reference lane:

```bash
npm run program-bundle:reference -- --manifest models/local/gemma-4-e2b-it-q4k-ehf16-af32-int4ple/manifest.json \
  --conversion-config src/config/conversion/gemma4/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.json \
  --out examples/program-bundles/gemma-4-e2b-it-q4k-ehf16-af32-int4ple.program-bundle.json \
  --surface browser --prompt "The color of the sky is" --max-tokens 8
```

That command runs a bounded `verify`, writes the actual returned report under
`reports/program-bundles/`, and exports the bundle from that report. A manual
receipt without transcript metrics is not accepted as a Program Bundle source.

## Program Bundle Parity

Program Bundle parity uses the normal `verify` command so browser and Node
surfaces share the same request contract. The export/check maintenance command
above still only writes or validates bundle files.

```bash
node src/cli/doppler-cli.js verify --config '{
  "request": {
    "workload": "inference",
    "workloadType": "program-bundle",
    "programBundlePath": "examples/program-bundles/gemma-3-270m-it-q4k-ehf16-af32.program-bundle.json",
    "parityProviders": ["browser-webgpu", "node:webgpu", "node:doe-gpu"],
    "programBundleParityMode": "contract"
  }
}' --json
```

There is no default parity mode or provider list. `contract` validates the
closed bundle and probes selected providers without claiming execution. Use
`"programBundleParityMode":"execute"` for replay. Provider results separate
schema validity, provider availability, execution, and transcript matching.

## Output format and error envelope

JSON output is the default. `--json` is accepted as an explicit no-op for scripts.
Use `--pretty` for human-readable summaries.
JSON success and failure return envelopes:

- Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
- Error: `{ ok: false, schemaVersion: 1, surface, request, error }`

For automation, use the default JSON output or pass `--json` explicitly.

## OpenAI-compatible server receipts

`doppler-serve` accepts `include_receipt: true` on non-streaming
`POST /v1/chat/completions` requests. A successful local runtime receipt sets
`resolutionStatus` to `resolved` and carries `logicalModelId`,
`resolvedArtifactVariantId`, and `resolvedExecutionId` in `resolution`.
The logical ID preserves the model string requested by the client even when it
resolves through a canonical registry ID or explicit artifact URL.

If execution fails before those identities exist, the diagnostic receipt sets
`resolutionStatus` to `unavailable` and records
`resolutionUnavailableReason: execution-failed-before-resolution`. The server
does not synthesize artifact or execution identity from catalog labels.

## Common launch patterns

### Hosted model (auto surface)

```bash
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"qwen3-0.8b"}}' --json
```

### Local rebuilt artifact

```bash
node src/cli/doppler-cli.js verify --config '{
  "request": {
    "workload": "inference",
    "modelId": "gemma-3-270m-it-q4k-ehf16-af32",
    "modelUrl": "file:///tmp/gemma-3-270m-it-q4k-ehf16-af32"
  },
  "run": {
    "surface": "node"
  }
}' --json
```

### Convert then debug with runtime override

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"modelId":"qwen3-0.8b","workload":"inference"},"run":{"surface":"node"}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":8}}}' \
  --json
```

## Useful validation notes

- If `--config` is omitted: `command requires --config <path|url|json>.`
- For bad URLs in `--config` or `--runtime-config`, errors include HTTP or fetch diagnostics.
- For malformed JSON: error reports `Invalid --config` / `Invalid --runtime-config`.
