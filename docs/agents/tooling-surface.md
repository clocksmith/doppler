# Tooling Surface: Public vs Internal

## Three Layers

| Layer | Location | Ships in package? | When agents need it |
|-------|----------|-------------------|---------------------|
| **Public CLI** | `src/cli/doppler-cli.js` | Yes (bin entry) | Using CLI commands |
| **Command infrastructure** | `src/tooling/` | Partial (`./tooling` export) | Editing runtime/harness code |
| **Dev scripts** | `tools/` | No (except `tools/convert-safetensors-node.js`) | Repo maintenance only |

## Public Package (what ships to npm)

Exports:
- `"."` — `src/index.js` (main library API)
- `"./provider"` — `src/client/doppler-provider.js`
- `"./tooling"` — `src/tooling-exports.js` (command runners, harness API)
- `"./internal"` — `src/index-internal.js`
- `"./generation"`, `"./diffusion"`, `"./energy"` — pipeline-specific exports

Binary: `doppler` → `src/cli/doppler-cli.js`

## Internal Dev Tools (never shipped)

Everything in `tools/` except `convert-safetensors-node.js`:
- Test runners, coverage reporters
- Benchmark comparison scripts (`compare-engines.js`, `bench-text-decode-paths.js`)
- Model conversion helpers (`refresh-converted-manifest.js`, `onboarding-tooling.js`)
- Sync/check scripts (`sync-model-support-matrix.js`, `check-contract-artifacts.js`)
- Agent parity verification (`verify-agent-parity.js`, `verify-agent-freshness.js`)

## Rules

- `tools/` scripts may import from `src/` directly (they run in the repo, not as a package consumer).
- Published package consumers use only the documented exports.
- Skills should invoke CLI commands (`npm run debug`, `npm run bench`) rather than importing `src/tooling/` internals.
- New dev scripts go in `tools/`. New commands go in `src/tooling/command-api.js` + CLI.
