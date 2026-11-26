# Testing Runbook

Canonical guide for running Doppler tests and for what CI runs. Changes are
committed and pushed straight to `main` ([AGENTS.md](../AGENTS.md#style-guides));
there is no pull-request or pre-merge gate.

## Quick reference

| Action | How | When |
| --- | --- | --- |
| Focused test | `node tools/run-node-tests.js <file>` | while editing the loader, pipeline or kernel that file covers |
| Fast source checks | `npm run check:fast` | before every push (about 22 s, no GPU) |
| Contract tests and triggered checks | the contract tests for the changed behavior ([AGENTS.md](../AGENTS.md#style-guides)) and `npm run check:changed` | before pushing to `main` |
| Repository acceptance | `npm run check:green` | release, hardware work, or a change that justifies it |
| Behavior checks only | `npm run check:behavior` | includes the whole unit suite, which opens a GPU device |
| Full check sweep | `npm run check:all` | release preparation or gate-policy changes |
| Node tests | `npm run test:unit` | all non-kernel test directories; opens a GPU device |
| Kernel correctness | `tests/harness.html?workload=kernels` | after kernel changes |
| Inference smoke | `tests/harness.html?workload=inference` | after pipeline changes |
| Training harness | `tests/harness.html?workload=training` | training work |
| Coverage report | `npm run test:coverage` | on demand; not run by CI |

## Discover and target checks

`npm run` lists the commands from `package.json`; `npm pkg get scripts` provides
the machine-readable inventory. Named checks remain independently runnable.

Both Node runners use the `suites` in
[`tools/policies/test-coverage-policy.json`](../tools/policies/test-coverage-policy.json):
`unit` discovers every test directory except `tests/kernels` and the files listed
in the `unit-webgpu` and `unit-browser` suites, which run separately through
`npm run test:unit:webgpu` and `npm run test:unit:browser`. `gpu` selects
`tests/kernels`, and `all` discovers the complete tree. These are selection groups,
not hardware evidence classes. Hardware-conditional tests report their own skips.
Coverage-specific exclusions and thresholds remain separate policy fields.

Use `node tools/run-node-tests.js --suite unit --list` to inspect the exact JSON
file list without executing it. Pass a file or directory to run a focused check.
Overlapping roots run each file once. `.pending.test.js` files are omitted from
directory discovery unless `--include-pending` is supplied; explicitly naming a
pending file still runs it.

## Repository gate

[`tools/policies/check-gate-policy.json`](../tools/policies/check-gate-policy.json)
splits the read-only checks into two sets, run by `tools/run-check-gate.js`:

- `behavior`: unit tests, declaration types, the browser import graph, package
  export parity, and the command surface contract. These run on every change.
- `triggered`: mirror, drift, freshness and policy checks. Each runs only when
  a changed file matches its inputs: the declared globs, repository paths named
  inside its `referencedBy` JSON files, or the local import closure of the tools
  and tests its npm script runs.

Changed files are the commits since the merge base with `origin/main` plus
staged, unstaged and untracked files. Pass `--base <ref>` to compare against
another ref; an unknown ref fails instead of selecting nothing.

| Command | Runs |
| --- | --- |
| `npm run check:green` | behavior checks plus the triggered checks for the current change |
| `npm run check:behavior` | behavior checks only |
| `npm run check:changed` | triggered checks for the current change only |
| `npm run check:all` | every behavior and triggered check |

Add `--list` to `node tools/run-check-gate.js` to print the selection and the
matching file for each triggered check without running anything. Editing the
gate policy or `tools/run-check-gate.js` selects every triggered check. A new
check belongs in the policy with inputs that cover every file it reads.
`check:green` before this split ran every check; that sweep is now `check:all`.

The gate expands its read-only npm check chain through the existing Node
runner's `--scripts` mode. Successful overlapping tests run once within that
invocation's consecutive test batches, each in a separate process. Intervening
commands invalidate test success; a failed prerequisite stops the chain. There
is no cross-invocation success cache. Stateful shell programs and npm lifecycle
hooks retain npm execution. Generate source before running the read-only chain.

## Evidence identity and custody

Historical reports and vendor samples are evidence, not duplicate implementations.
Keep rejected observations and their original hashes. Tests replaying historical
observations must use their recorded inputs, not require today's filesystem to
remain unchanged. Local-only artifacts need explicit custody and reproduction
instructions; an absent archive is not independent reproduction.

The runtime's `src/formats/canonical-hash.js` and repository evidence's
`tools/lib/canonical-json.js` have distinct established serialization contracts:
locale-based versus UTF-16 key ordering, and different undefined-value handling.
They must not be interchanged to remove apparent duplication. Golden byte/hash
fixtures in `tests/tooling/evidence-json-contracts.test.js` preserve both; changing
either requires an explicit identity migration, not a generic helper refactor.

## Browser harness

Use the [test harness guide](../tests/GUIDE.md) for command context, runtime
configuration, and browser relay examples. Command workload and mode are not
runtime configuration fields.

## CI notes

The automatic CI workflow is `Smoke`
([check-green.yml](../.github/workflows/check-green.yml)). It runs on every push
to `main` (and on pull requests and manual dispatch) on Node 22, with no browser,
GPU provider, or model download. Its `smoke` job runs, in order:

1. `npm ci --ignore-scripts --omit=optional --no-audit --no-fund`
2. four contract tests: `tests/inference/execution-backend-contract.test.js`,
   `tests/config/runtime-transform-ownership.test.js`,
   `tests/config/error-taxonomy.test.js`, `tests/config/public-error-contract.test.js`
3. `npm run typecheck:source`
4. the Capsule tests through `node tools/run-node-tests.js` (`tests/capsule/capsule-v2.test.js`,
   `capsule-v3.test.js`, `capsule-retained-local-use.test.js`,
   `tests/runtime/capsule-artifact-source.test.js`, `capsule-load-options.test.js`)
5. `tests/integration/execution-graph-transforms.test.js`
6. `npm run check:fast`, `npm run style:guides:check`, `npm run style:guides:test`
7. `npm run package:smoke`: packs and installs the library offline, checks public
   imports and declarations, and exercises synthetic consumer contracts.

A separate `privacy-protocol-experiment` job runs `npm test --prefix tools/privacy-demo`.
Reproduce a step locally after the same `npm ci` command; because CI installs
with `--omit=optional`, no Node test there can open a GPU device.

- CI does not run `check:green`, the whole unit suite, or coverage. A green
  `Smoke` run does not mean every test passed or qualify numerical correctness.
- No workflow has a `schedule:` trigger; there is no nightly tier.
- [Manual runtime validation](../.github/workflows/manual-runtime-validation.yml)
  and [installed consumers](../.github/workflows/installed-consumers.yml) run on
  demand; [doppler-release.yml](../.github/workflows/doppler-release.yml) is the
  reusable release-eligibility workflow. Physical GPU and model-artifact
  qualification require their declared hardware and inputs; hosted software
  rendering is not physical GPU evidence.
- Node coverage policy is defined in `tools/policies/test-coverage-policy.json`.

## Related

- Debug workflow and boundary-diff protocol: [debug-playbook.md](debug-playbook.md)
- Debug report template: [debug-investigation-template.md](debug-investigation-template.md)
- Test harness details: [../tests/GUIDE.md](../tests/GUIDE.md)
- Kernel coverage details: [../tests/kernels/GUIDE.md](../tests/kernels/GUIDE.md)
- Kernel benchmark baselines: [../tests/kernels/benchmarks.md](../tests/kernels/benchmarks.md)
- Kernel test design guidance: [kernel-testing-design.md](kernel-testing-design.md)
- Inference matrix strategy: [inference-test-matrix.md](inference-test-matrix.md)
- Kernel override policy (canonical): [WGSL style guide](style/wgsl-style-guide.md#kernel-compatibility-and-overrides)
