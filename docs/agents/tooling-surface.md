# Tooling Surface: Public vs Internal

## Three Layers

| Layer | Location | Ships in package? | When agents need it |
|-------|----------|-------------------|---------------------|
| **Public CLI** | `src/cli/doppler-cli.js` | Yes (bin entry) | Using CLI commands |
| **Command infrastructure** | `src/tooling/` | Partial (`./tooling` export) | Editing runtime/harness code |
| **Dev scripts** | `tools/` | No (except `tools/convert-safetensors-node.js`) | Repo maintenance only |

## Documented Package Surface

Exports: see the generated [export inventory](../api/reference/exports.md) and
[API Docs](../api/index.md). The root `dopplerx` export is the signed Capsule
runtime (`src/capsule-runtime.js`); `dopplerx/compat` (`src/index.js`) is the
legacy manifest-loading facade.

Binaries:
- `doppler` → `src/cli/doppler-cli.js` (tooling CLI)
- `dopplerx` → `src/cli/doppler-quickstart.js`
- `doppler-gpu` → `src/cli/doppler-quickstart.js` (former package name, same quickstart)
- `doppler-serve` → `src/cli/doppler-serve.js`

## Internal Dev Tools (never shipped)

Everything in `tools/` except `convert-safetensors-node.js`:
- Test runners, coverage reporters
- Benchmark comparison scripts (`compare-engines.js`, `bench-text-decode-paths.js`)
- Model conversion helpers (`refresh-converted-manifest.js`, `onboarding-tooling.js`)
- Sync/check scripts (`sync-model-support-matrix.js`, `check-contract-artifacts.js`)
- Agent parity verification (`verify-agent-parity.js`)

## Rules

- `tools/` scripts may import from `src/` directly (they run in the repo, not as a package consumer).
- Published package consumers use only the documented exports.
- New docs should center signed Capsule execution from the root surface; legacy manifest loading belongs under `dopplerx/compat`.
- The command list is in [the CLI reference](../cli.md#supported-commands); the shared browser/Node command contract is in [Tooling API](../api/tooling.md#command-contract-summary). CLI-local commands such as `program-bundle` and `profiles` are outside that contract.
- Agents should run `node src/cli/doppler-cli.js profiles --json` before guessing runtime profile names.
- Skills should invoke CLI commands (`npm run debug`, `npm run bench`) rather than importing `src/tooling/` internals.
- New dev scripts go in `tools/`. New commands go in `src/tooling/command-api.js` + CLI.

## Repository operator entrypoint: kernel-path builder

`src/tooling/kernel-path-builder/index.js` is an intentionally retained, repository-only
operator entrypoint. Import `buildKernelPathBuilderArtifacts` with explicit
`configEntries`, `manifestEntries` and `registryEntries` to produce an execution
index, proposals and a Markdown report. It does not install a runtime path or qualify
a proposed kernel. It is not a package export; applications should not deep-import it.
`tests/tooling/kernel-path-builder-entry.test.js` checks the entrypoint contract.

The `reports/` ignore rules control admission of new files, not retention of existing
tracked evidence. Tracked receipts may be referenced by contracts or historical claims.
Review those references and supersession before archiving; never delete a receipt solely
because an ignore pattern would exclude a new file at the same path.
