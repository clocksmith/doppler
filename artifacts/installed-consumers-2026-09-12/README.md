# Installed-consumer baseline, 2026-09-12

Runtime source baseline: Doppler `6734945b60f9f943194ef21fec6cf93e32f6cded`.
Reploid consumer: `477670567d13df536157815560baf8c987948331`.
Fixture generators are the files committed with this record; their original
working-tree state and runner digest are preserved in each source-state.json.
The remote workflow records the committed fixture-generation revision.

Two distinct runtime archives were tested; neither version strings nor passing
fixtures make them interchangeable:

| Archive | SHA-256 | Installed contracts |
| --- | --- | --- |
| Published npm 0.6.1 | 96d2699a3e677815890f804459c023bea4fb2e2a1a59d157b5b6a04c9e573d5f | standalone + Reploid pass |
| Built from this source baseline | 0d1792d355417e828e690a9dae5b117a2ff5159a569fe422c061631302396761 | standalone + Reploid pass |

Receipts record archive integrity, environment, and exact signed fixture model
descriptors. Generated fixtures are application test data, excluded from the
runtime package. Both installed consumers use public exports, with injected
programs for repeatable contract tests. These passes are not GPU qualification.
The designated cross-repository workflow requires Reploid acceptance and shares
one archive with the standalone consumer. Missing consumer setup fails.

Physical browser results here use the **published** archive, Chrome
146.0.7680.177, physical AMD rdna-3, Linux. The original Qwen3 4B generation run
lost its GPU device during prefill. The short-prompt diagnostic instead failed
because a selected fused FFN shader was outside the verified Capsule closure;
it did not establish the cause of device loss. Both failures remain preserved.
The subsequent full-prompt run matched all 16 frozen source-reference tokens.
Its embedding/reranker attempts failed before execution because the application
had not supplied persistReleaseCheckpoint. Corrected application wiring is
being exercised separately. Each physical receipt includes model descriptors,
settings, Capsule roots, plan identities, outputs or exact failure boundaries.

The runtime closure synchronization updates the current generated inventory for
pre-existing src/debug/stats.js changes. It does not rewrite historical model
manifests. The package payload limit was stale by 1,580 bytes at baseline; its
exact 1,784-file inventory is retained in candidate/npm-pack.json. Consumer
fixtures add zero runtime files.

Reproduce an installed contract bundle:

```sh
node tools/check-packed-package.js --archive /absolute/archive.tgz --retain /absolute/new-bundle
DOPPLER_TEST_CONSUMER=/absolute/new-bundle/consumer node /absolute/reploid/tests/fixtures/doppler-installed-generation.js
```

Run physical cases using tools/check-installed-capabilities.js and an explicit
configuration from a retained receipt, with its model artifact roots mounted.
The runner preserves each operation's failure and continues the declared cases.
No npm publication, dependency upgrade, or production deployment is claimed.

Component: doppler.repository-tooling; doppler.tests; doppler
Intent: preserved
Acceptance evidence: retained receipts; check-packed-package.js; Reploid test:doppler-consumer; physical receipts
Boundary effects: Reploid installed-consumer tests; public host example
