# Runtime Config Assets

Runtime config JSON files fall into three categories:

- Profile wrappers: files with `id`, `name`, lifecycle metadata, and a
  `runtime` object. These are loadable through `runtimeProfile`,
  `runtimeConfigUrl`, or `configChain`.
- Runtime overlays: profile wrappers whose `runtime` object is intentionally
  small and meant to be composed with another profile.
- Non-profile policy assets: files with their own `$schema`, such as
  `diagnostics/drift-policies.json`. These are loaded by specific subsystems
  and are not runtime profiles.

Rules:

- Discover checked-in profile wrappers with
  `node src/cli/doppler-cli.js profiles --json`.
- Do not add string `runtime.inference.kernelPath` IDs. Runtime `kernelPath` is
  either `null` or an inline execution-v1-derived object.
- Keep model-owned behavior in conversion config and the manifest execution
  graph, not runtime profiles.
- If a non-profile asset is added under this tree, add an explicit schema and
  update onboarding checks so it is not treated as a runtime profile.
