# Signed resident producer boundary

This is synthetic producer acceptance, not physical model qualification. The
ordinary Rig command previously rejected resident partition reports as missing
generation results (`before.log`). It now preserves admitted partition records
and signs the verified piece index alongside the original evidence and frozen
comparison contract (`after.log`). No physical passing report was created.

Use the existing command after a trusted physical producer has derived and
retained a passing comparison:

```
node tools/rig-model-capsule.js --program-bundle BUNDLE.json \
  --manifest manifest.json --model-dir MODEL_DIRECTORY \
  --initial-identity initial-identity.json \
  --qualification-report resident-node.json \
  --qualification-report resident-browser.json \
  --resident-contract frozen-resident-contract.json \
  --piece-index pieces.json --release-manifest release.json \
  --signing-private-key PRIVATE_JWK --signing-public-key PUBLIC_JWK \
  --signing-authority AUTHORITY --out OUTPUT/capsule.json --json
```

The frozen `doppler.resident-qualification-contract/v1` contains
`referenceHash`, `evaluatorHash`, `tolerance`, and `comparedSteps`. The independent
F64 authority and unchanged 0.001 remain the current application requirement;
this code does not select a new authority or generate its reference.

Each `doppler.resident-partition-qualification/v1` report contains
`operation: residentPartition`, passed status, model ID, current manifest
identity, execution graph hash, initial execution identity, physical surface,
complete canonical partition plan/hash and index. Its `comparison` binds the
canonical contract hash, reference/capture/evaluator hashes, exact tolerance,
`complete: true`, and ordered `steps` with index and derived max absolute error.
A changed F32/two-F32 session identity, model, graph, plan, bounds, count, failed
step or incomplete comparison rejects. This is structural admission of evidence
from an explicitly trusted producer: hashes and self-reported error numbers do
not independently establish that the producer computed the comparison correctly.
A raw whole-model distribution lacking partition/runtime binding is insufficient.

The normal existing Program Bundle generation/source-truth requirements remain.
Manifest equality uses the current bundle, so legitimate shader-manifest changes
receive new identities while weight artifacts retain their own byte identities.

`--piece-index` verifies the index against exact manifest bytes, validates range
coverage through the existing verified reader, and reads/verifies each original
piece from source files before signing. Missing, corrupt, undeclared and
mismatched source bytes reject. The signed `program.pieceIndexArtifactId` points
to the retained exact index bytes; no full-model runtime download was added.

Validation: tooling end-to-end fixture with Node/index0 and browser/index1,
existing forge stages/qualification/source-truth tests, resident signed opening
and corruption tests, strict TypeScript, source style, architecture and runtime
closure checks. All are CPU/synthetic checks. Generated dependency view refresh
is coordinated with the concurrent numerical source checkpoint.

Next physical boundary: obtain a qualifying numerical/runtime archive and actual
partition evidence, then sign and open the exact artifact. Reploid's existing
catalog needs Capsule, executable identity, accepted target digests, source-file
closure and release/admission policy. Rig model paths use `artifacts/model/`;
host custody adapters must resolve those against manifest-relative piece paths.
No production catalog, accepted runtime pin, release authority or deployment changed.

## Physical report production (prepared, not executed)

`produce-physical-reports.js` converts retained cooperative split captures into
this existing report schema. It independently decodes every actual F32 logit and
compares it with the approved F64 vector for all 55 ordered prefixes. It rejects
changed tokens, generation policy, missing vectors, nonfinite values, failed
cleanup, changed budgets, and a host assignment different from the predeclared
two-machine placement. Raw `matches` and error summaries do not grant a pass.

Before running the physical controller, retain a separate expectations JSON and
its SHA-256. `physical-producer-preparation.json` binds the already frozen
reference and archive-8 inputs and lists the missing setup fields. It is
intentionally not executable: the observed application bundle, exact partition
plan, provider descriptors, directional placements, and final producer SHA must
be frozen before reading the split outcomes. Archive 8's first resident opening
subsequently exposed a lifecycle initialization failure; a corrected standard
archive requires a new preparation binding and numerical evidence. Do not reuse
archive-8 identities for those changed bytes.

After a successful physical run, create a controller-input JSON with
`expectations`, `capture`, `reference`, `manifest`, `graph`, `run`, `archive`, and
`evaluator`, each `{ "path": "/absolute/retained/file", "hash": "sha256:..." }`,
and `requestContracts`, an ordered array of three such bindings. Then run:

```
node artifacts/signed-resident-producer-20261012/produce-physical-reports.js \
  /ABS/physical-report-inputs.json /ABS/NEW_REPORT_DIRECTORY
```

The expectations file contains the fields in `fixedExpectations` plus the listed
pre-run fields. `producerHash` binds the reviewed producer itself. The tool
extracts the hash-bound standard archive into its own temporary directory,
imports identity, canonical-hash, and existing Rig-admission helpers from that
exact package closure, verifies observer factory/helper hashes, and removes the
temporary directory. It neither imports live runtime code nor overwrites prior
reports. A passing conversion emits `contract.json`, `partition-0.json`, and
`partition-1.json`; this command does not sign or deploy anything.

The source-bound CDP observer captures the actual resolved session at resident
opening. Each report preserves that observed identity. In archive 8's canonical
registry path, signed execution adds no registry extension; the exact source
trace is `archive8-signed-identity-review.json`. Custom registries are a different
identity and are not covered. Different provider identities must not be edited
into agreement: existing Rig admission requires every attached report to match
the selected initial identity. Signed opening must still observe and compare its
actual program identity through ordinary admission, retain any rejection and
settlement, and execute the applicable physical journey.

The CPU tests are synthetic producer controls, not physical model evidence.
`physical-producer-tests.log` retains the focused result. No resident report has
been minted from future or fabricated model outcomes. Model answer quality,
actual signed opening, selective acquisition, and release eligibility remain
separate gates.

The direct numerical fixture mode is now explicit. Its pre-run expectations use
`deliveryMode: direct-fixture`, `testDeliveryHash`, and exact `installedFiles`.
The CLI input supplies `delivery` (the frozen snapshot binding) instead of `run`;
an application-controller envelope is rejected in this mode. The fixture checks
all listed source files and every installed archive member before loading the
model, then records that observed test-delivery identity. These are not deployed
application bytes. Browser versions and directional host assignments are frozen
as well as adapter descriptions. The ordinary controller mode is unchanged.

Archive 9's two expected placements are frozen in
`archive9-mac-amd-expectations.json` and `archive9-amd-mac-expectations.json`,
before either split loads weights. The raw provider and public-plan preflights,
complete snapshot file list, and its SHA-256 are retained alongside them. The
observed empty adapter device identifier is normalized to `unknown` exactly as
the bound Doppler package does; raw observations remain unchanged. There are
2,001 test-delivery files and 1,974 installed archive files in this snapshot.
`physical-producer-tests-direct.log` retains 32 passing CPU controls. Root's
independent read-only review accepted the direct-mode change before execution;
the earlier `review.json` continues to bind its earlier exact source, not these
later bytes. Actual split results remain required.
