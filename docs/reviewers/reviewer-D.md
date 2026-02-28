# Reviewer D Assignment (Configs, Conversion, Loader/Storage, Plumbing)

See central process in `docs/reviewers/coordination.md` before starting.

## Scope
- Own:
  - `src/config/**`
  - `src/converter/**`
  - `src/loader/**`
  - `src/storage/**`
  - `src/client/**`
  - `src/bridge/**`
  - `src/browser/**`
  - `src/adapters/**`
  - `src/formats/**`
  - `src/tooling/**`
  - `src/hotswap/**`
  - `src/debug/**` (except `src/debug/perf.js` and `src/debug/config.js` if reviewer C takes full test-tooling block, coordinate first)
  - `src/bootstrap.js`

## Primary objectives
1. Ensure policy/config files are the source of truth for conversion/loading/runtime modes.
2. Remove legacy/compat token assumptions by routing behavior through explicit schema-driven decisions.
3. Keep loader/storage paths deterministic and fail-fast on capability/schema mismatch.

## Mandatory checks per file
- Replace fallback-driven behavior for model/runtime selection with explicit config keys and schema checks.
- Ensure manifest/loader assertions are strict; no ambiguous fallback in conversion path.
- Remove runtime non-determinism in loader/storage defaults.
- If converter or schema changes are made, verify paired docs/schema updates are in the same edit scope.

## Deliverables
- For every assigned file, update these tracking artifacts:
  - `docs/tracking/execution-plane-review-queue.md`
  - `docs/tracking/execution-plane-review-queue.js-runtime.md`
  - `docs/tracking/execution-plane-audit.md`
  - `docs/tracking/execution-plane-audit.json`
  - `docs/tracking/execution-plane-audit.csv`
  - `docs/tracking/execution-plane-review-queue.json`
  - `docs/tracking/execution-plane-review-queue.js-runtime.json`
- Add schema/manifest/document updates alongside any runtime-visible config or field changes.

## Hard constraints
- No silent fallback in conversion/loader semantics.
- Do not migrate browser/node runtime behavior.
- Keep command surfaces compatible: any policy update that affects command surface must be mirrored in `src/tooling/command-api.js`.

## Stop criteria
- Stop when all assigned folders under `src/config`, `src/converter`, `src/loader`, `src/storage`, `src/bridge`, `src/browser`, `src/tooling`, `src/tooling`, `src/client`, `src/adapters`, and `src/formats` are marked `reviewed` in both queues.
