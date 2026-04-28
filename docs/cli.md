# Doppler CLI Reference

The canonical CLI entrypoint is `src/cli/doppler-cli.js`.

For the npm-facing quickstart path, use `npx doppler-gpu`. That bin is a thin
first-run surface for local generation. The `doppler` CLI below is the
contract-driven tooling surface for `verify`, `debug`, `bench`, `convert`, and
operator workflows. It also exposes Node-only maintenance and investigation
paths such as `program-bundle` and `diagnose`.
It also exposes `profiles`, a read-only discovery command for checked-in
runtime profile IDs.

## Command Surface

```bash
node src/cli/doppler-cli.js <command> --config <request> [flags]
```

- `--config` is required for workload commands. `profiles` does not take a config.
- `--help` prints CLI usage.
- `--pretty` prints human-readable output.
- JSON output is default; `--json` is accepted for explicit automation.

### Supported commands

| Command | Workload intent | Notes |
| --- | --- | --- |
| `bench` | `calibrate` | Requires `request.workload`. |
| `debug` | `investigate` | Requires `request.workload`. |
| `verify` | `verify` | Requires `request.workload` (except legacy `kernels` shape). |
| `diagnose` | `investigate` | Node-only operator-diff investigation command. |
| `convert` | `convert` | Node-only command. |
| `profiles` | discovery | Lists checked-in runtime profile IDs; no workload is executed. |
| `lora` | operator lifecycle | Node-only command. |
| `distill` | operator lifecycle | Node-only command. |
| `program-bundle` | maintenance/export | Node-only artifact exporter; outside the browser/Node command-runner contract. |

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

`convert`, `diagnose`, `lora`, and `distill` reject `--surface browser`.
`convert`, `lora`, and `distill` reject runtime-input fields in the Node operator surface.

Command-level surface support:

- `bench`, `debug`, `verify`: `auto|node|browser`
- `convert`: `auto|node` (`browser` is rejected)
- `lora`, `distill`: `auto|node` (`browser` is rejected)
- `diagnose`: `auto|node` (`browser` is rejected)
- `program-bundle`: no `--surface`; reads declared files and writes a JSON artifact
- `profiles`: no `--surface`; reads checked-in runtime config metadata only

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
    "parityProviders": ["browser-webgpu", "node:webgpu", "node:doe-gpu"]
  }
}' --json
```

The default parity mode is `contract`. It validates the reference browser
transcript, plans Node/Dawn replay, and checks Doe.js availability without
requiring the optional package to be installed. Use
`"programBundleParityMode":"execute"` to run the Node/WebGPU replay provider.

## Output format and error envelope

JSON output is the default. `--json` is accepted as an explicit no-op for scripts.
Use `--pretty` for human-readable summaries.
JSON success and failure return envelopes:

- Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
- Error: `{ ok: false, schemaVersion: 1, surface, request, error }`

For automation, use the default JSON output or pass `--json` explicitly.

## Common launch patterns

### Hosted model (auto surface)

```bash
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"gemma3-270m"}}' --json
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
  --config '{"request":{"modelId":"gemma3-270m","workload":"inference"},"run":{"surface":"node"}}' \
  --runtime-config '{"inference":{"generation":{"maxTokens":8}}}' \
  --json
```

## Useful validation notes

- If `--config` is omitted: `command requires --config <path|url|json>.`
- For bad URLs in `--config` or `--runtime-config`, errors include HTTP or fetch diagnostics.
- For malformed JSON: error reports `Invalid --config` / `Invalid --runtime-config`.
