# Testing Runbook

Canonical guide for running Doppler tests.

## Quick reference

| Action | How | When |
| --- | --- | --- |
| Repository acceptance | `npm run check:green` | source, contract, or integration changes |
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
`unit` discovers every test directory except `tests/kernels`, `gpu` selects that
directory, and `all` discovers the complete tree. These are selection groups,
not hardware evidence classes. Hardware-conditional tests report their own skips.
Coverage-specific exclusions and thresholds remain separate policy fields.

Use `node tools/run-node-tests.js --suite unit --list` to inspect the exact JSON
file list without executing it. Pass a file or directory to run a focused check.
Overlapping roots run each file once. `.pending.test.js` files are omitted from
directory discovery unless `--include-pending` is supplied; explicitly naming a
pending file still runs it.

`check:green` expands its read-only npm check chain through the existing Node
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
