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
