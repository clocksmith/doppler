# Doppler CLI Reference

The canonical CLI entrypoint is `src/cli/doppler-cli.js`.

For the npm-facing quickstart path, use `npx doppler-gpu`. That bin is a thin
first-run surface for local generation. The `doppler` CLI below is the
contract-driven tooling surface for `verify`, `debug`, `bench`, `convert`, and
operator workflows.

## Command Surface

```bash
node src/cli/doppler-cli.js <command> --config <request> [flags]
```

- `--config` is required for every command.
- `--help` prints CLI usage.
- `--pretty` prints human-readable output.
- JSON output is default (omit `--pretty`).

### Supported commands

| Command | Workload intent | Notes |
| --- | --- | --- |
| `bench` | `calibrate` | Requires `request.workload`. |
| `debug` | `investigate` | Requires `request.workload`. |
| `verify` | `verify` | Requires `request.workload` (except legacy `kernels` shape). |
| `convert` | `convert` | Node-only command. |
| `lora` | operator lifecycle | Node-only command. |
| `distill` | operator lifecycle | Node-only command. |

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

For harnessed commands (`bench`, `debug`, `verify`) the same polymorphic formats are accepted:

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

## `--surface` and execution intent

- `--surface auto` (default) tries Node first; falls back to browser relay for supported harnessed commands.
- `--surface node` forces Node execution.
- `--surface browser` forces headless Chromium relay.

`convert`, `lora`, and `distill` currently reject runtime-input fields in the Node operator surface.

Command-level surface support:

- `bench`, `debug`, `verify`: `auto|node|browser`
- `convert`: `auto|node` (`browser` is rejected)
- `lora`, `distill`: `auto|node` (`browser` is rejected)

## Output format and error envelope

Without `--json`, output is designed for humans (`--pretty`).
With `--json`, success and failure return JSON envelopes:

- Success: `{ ok: true, schemaVersion: 1, surface, request, result }`
- Error: `{ ok: false, schemaVersion: 1, surface, request, error }`

If you need traceability for automation, add `--json` on the command line.

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
