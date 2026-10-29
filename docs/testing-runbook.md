# Testing Runbook

Canonical guide for running Doppler tests.

## Quick reference

| Action | How | When |
| --- | --- | --- |
| Repository acceptance | `npm run check:green` | source, contract, or integration changes |
| Behavior checks only | `npm run check:behavior` | quick local loop |
| Full check sweep | `npm run check:all` | release preparation or gate-policy changes |
| Node tests | `npm run test:unit` | all non-kernel test directories |
| Kernel correctness | `tests/harness.html` mode `kernels` | after kernel changes |
| Inference smoke | `tests/harness.html` mode `inference` | after pipeline changes |
| Training harness | `tests/harness.html` mode `training` | training work |
| Coverage gate | `npm run test:coverage` | before merge |

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

- The [automatic CI workflow](../.github/workflows/check-green.yml) installs
  Chromium and runs the browser kernel and demo/offline-PWA contracts after
  the Node gates. Its SwiftShader kernel lane is not physical GPU qualification.
- Physical GPU and model-artifact qualification remain separately scoped runs.
- Node coverage policy is defined in `tools/policies/test-coverage-policy.json`.

## Related

- Debug workflow and boundary-diff protocol: [debug-playbook.md](debug-playbook.md)
- Debug report template: [debug-investigation-template.md](debug-investigation-template.md)
- Test harness details: [../tests/GUIDE.md](../tests/GUIDE.md)
- Kernel coverage details: [../tests/kernels/GUIDE.md](../tests/kernels/GUIDE.md)
- Kernel benchmark baselines: [../tests/kernels/benchmarks.md](../tests/kernels/benchmarks.md)
- Kernel test design guidance: [kernel-testing-design.md](kernel-testing-design.md)
- Kernel override policy (canonical): [operations.md#kernel-overrides--compatibility](operations.md#kernel-overrides--compatibility)
