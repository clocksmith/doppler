# Doppler Config

Runtime-facing config contract for inference and command execution.

Implementation internals for the config subsystem live in
[../src/config/README.md](../src/config/README.md).

## Canonical ownership

- Conversion vs runtime field ownership:
  [conversion-runtime-contract.md](conversion-runtime-contract.md)
- Command/surface behavior contract:
  [style/command-interface-design-guide.md](style/command-interface-design-guide.md)
- Merge and schema conventions:
  [style/config-style-guide.md](style/config-style-guide.md)

## Runtime config model

Doppler resolves runtime behavior from explicit runtime inputs composed in order
(each later input overrides the previous):

1. **Runtime profile** — named profile via `request.runtimeProfile` (e.g. `profiles/production`)
2. **Runtime config URL** — remote/local JSON via `request.runtimeConfigUrl`
3. **Explicit runtime override** — inline object via `request.runtimeConfig`

`configChain` is an ordered list of runtime config refs that composes ahead of
the three inputs above on surfaces that support it.

Command metadata is not a runtime-config layer. `command`, `workload`,
`modelId`, and command intent stay in the explicit command/suite request and
must not be injected into `runtime.shared.*` by the runners.

CLI flag inputs:

- `--config` accepts inline JSON, a file path, or URL for all commands.
- `--runtime-config` is supported by harnessed commands (`bench`, `debug`, `verify`)
  and accepts the same input shapes (`JSON`, `path`, `URL`).

See `src/tooling/command-runner-shared.js:applyRuntimeInputs()` for the
merge implementation.

Behavior-changing execution choices must be present in the manifest or explicit
runtime config before execution. No hidden runtime fallbacks or command-owned
runtime rewrites are allowed.

## Multimodal config contract

Multimodal encoder behavior is config-owned, not runtime-inferred.

- `config.vision_config.vision_architecture` is required for image/video-capable models.
- `config.audio_config.audio_architecture` is required for audio-capable models.
- Vision preprocessing fields such as normalization, pixel limits, patch sizing,
  and merge sizing must be present explicitly in config or explicit runtime
  model overrides.
- Runtime does not infer multimodal architecture from `model_type`, mirrored
  top-level manifest fields, or missing tunables.

If a multimodal manifest is missing those fields, the runtime must fail fast and
the converter/config must be fixed instead of patching around the artifact.

## Kernel path contract

Kernel paths are explicit execution plans selected by ID.

Canonical ID registry:
- `src/config/kernel-paths/registry.json`

Path definitions:
- `src/config/kernel-paths/*.json`

Kernel-path resolution precedence is canonical in
[`conversion-runtime-contract.md`](conversion-runtime-contract.md).
This file only relies on that contract:
- `kernelPath` remains the only supported kernel-selection override surface
- internal runner-owned per-run context may still apply the highest-precedence
  override after manifest and runtime config resolution

`kernelPath` is the only supported kernel-selection override surface.
Do not use legacy `kernelPlan`.

Kernel-path dtype contract:
- config-selected kernel paths must already match the resolved runtime
  `activationDtype` and `kvcache.kvDtype`
- manifest/model-selected kernel paths may seed those runtime dtypes only while
  the runtime values are still at global defaults; conflicting runtime overrides
  fail closed

## Runtime boundaries

Runtime config may tune execution policy (for example batching, sampling,
intent, diagnostics), but may not rewrite conversion-owned storage facts
(quantization layout, emitted tensor set, shard hashing policy).

If a change requires different storage artifacts, reconvert the model.

For investigate/verify text-generation runs, `runtime.shared.tooling.diagnostics="always"`
enables operator-timeline capture without switching the harness mode to
`diagnose`. The automatic policy is intended for compare-like debug lanes: it
emits low-overhead operator records while preserving the resolved execution
plan and batching behavior unless the caller explicitly requests a heavier
diagnostics payload.

## Command/runtime boundary

Harnessed command runs (`verify`, `debug`, `bench`) must preserve explicit
command context:
- `command`
- `workload`
- `modelId` when the workload requires one
- normalized command intent

That context belongs to the command request / suite options, not to the runtime
config merge chain. Workload-specific model ID enforcement lives in
`src/tooling/command-api-family-normalizers.js`.

## Common usage

Set runtime profile only:

```bash
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"inference","modelId":"gemma-3-270m-it-q4k-ehf16-af32","runtimeProfile":"profiles/production"},"run":{"surface":"auto"}}' --json
```

Override kernel path explicitly:

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"modelId":"gemma-3-270m-it-q4k-ehf16-af32"},"run":{"surface":"auto"}}' \
  --runtime-config '{"inference":{"kernelPath":"gemma3-q4k-dequant-f32a-online"}}' \
  --json
```

Load a request from a URL:

```bash
node src/cli/doppler-cli.js verify --config https://example.org/doppler/verify-gemma.json --json
```

## Change checklist

When adding new runtime-visible behavior:

1. update schema/validation (`src/config/schema/*`)
2. update merge/loader wiring
3. update rule assets or registry data when selection behavior changes
4. add regression coverage under `tests/config` and/or integration tests
5. update this file and the relevant style guide section

## Related

- [conversion-runtime-contract.md](conversion-runtime-contract.md)
- [rdrr-format.md](rdrr-format.md)
- [operations.md](operations.md)
- [testing.md](testing.md)
