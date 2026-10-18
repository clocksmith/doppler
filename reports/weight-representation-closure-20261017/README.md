# Representation qualification: hosted closure and BLAKE3 correction

Component: `doppler.tests`, `doppler.runtime-source.storage`, `doppler.docs`.
Intent preserved. Storage still owns byte integrity, not model substitution.
Boundary effects: standard BLAKE3 is corrected; old nonstandard manifest digests
are rejected, not migrated. No model files, public services, or release metadata
were changed. This is not a performance candidate or a new model qualification.

## Hosted gate

[Run 39085130244](https://github.com/clocksmith/doppler/actions/runs/39085130244)
passed at `bbcf7ffd7a9dd49ba850e5c2c2a8ff36158ffe32`. Behavior (643 files),
Node WebGPU (27 files), browser units (2 files), packed-package and demo/PWA
stages all executed. The software adapter was Mesa llvmpipe 25.2.8, LLVM 20.1.2.
The native training round executed; this is software Vulkan evidence, not
physical GPU training qualification.

The initial bootstrap failure is retained in the
[original campaign](../weight-representation-20261017/README.md). Installing
Lavapipe in `1d7781c1` exposed a later SIGSEGV after the projection stability
test printed success (run 39083585069). Commit `bbcf7ffd` drains queued work
before releasing the projection buffers and drains/destroys the device in an
`after` hook. The affected test passed on M3 Metal, AMD-hosted software Vulkan,
and hosted CI. No test, assertion, budget, or timeout was weakened.

Green stages are not a full physical matrix. [Remaining skips](remaining-skips.json)
preserve every logged optional model, platform, and local-evidence skip, including
Qwen linear attention, prepared sequence, partition, speculative, Apple-only
kernel, and local training-evidence lanes. Do not treat those as newly executed.
M3 27B remains unprovisioned; its partial copy is not a qualified model.

## Legacy BLAKE3 diagnosis and correction

The old compressor XORed four state words with IV constants, unlike BLAKE3, and
root output omitted the final XOR mixing. Its output is a nonstandard legacy
digest. The correction passes all 35 unkeyed 32-byte
[official vectors](../../tests/storage/blake3-vectors.json), including block,
chunk, and tree boundaries. The old implementation passed zero of those vectors.
Each vector is also tested with seven incremental update sizes and repeated
finalization. Historical digests remain rejection fixtures.

[The unchanged first 2B shard](blake3-shard-control.json) is 67,108,864 bytes:

- Its manifest and frozen old runtime agree on legacy digest
  `7f38a80c6ba6593361224d8041712c18cb78151c1697cc5de7562cf53a2dbb4a`.
- Corrected JavaScript and the retained independent Python BLAKE3 control agree
  on standard digest
  `ed436f5e41bb514f0ecc2b49a8eae00a8b0985d470e649d3b1265eba54347f2d`.
- Its SHA-256 remains
  `e40985f8ab9f9d321fe46c309edf79d39291156826c9a77903caed4fadf8138c`.

This reconciles the discrepancy; it does not upgrade old manifests or signatures.
The original campaign's independent, commit-pinned SHA-256 piece inventories
remain its byte authority. See the [migration contract](../../docs/rdrr-format.md#shard-hash-algorithms-and-legacy-blake3).
Strict standard verification rejects legacy manifest hashes. An intentionally
created successor artifact needs a separate destination, independent byte
verification, new bound identities, and qualification before release.

## Validation and reproduction

The hash change passed five focused test files, source style/unused checks, and
installed-package smoke: 1,961 files, 3,487,845 packed bytes, 14,154,864 unpacked
bytes. Existing limits were retained. Installed synthetic generation controls
and all 32 public export declarations passed. The hosted green run above precedes
this hash correction; it must not be relabeled as CI evidence for the correction.

```sh
node tools/run-node-tests.js tests/storage/blake3.test.js tests/storage/artifact-storage-context.test.js tests/runtime/resident-piece-blake3.test.js tests/runtime/capsule-artifact-verification-reuse.test.js tests/tooling/program-bundle-exporter.test.js
npm run source:style:check
npm run package:smoke
```

Storage tests exercise standard digest acceptance and historical digest rejection
through both range-read and streaming verification, repeated failure, and
unchanged manifest/byte state. Unknown hash selectors reject explicitly.

[Raw evidence](raw-evidence.tar.gz) contains hosted logs/status, before/after
vector results, focused physical teardown checks, the independent shard control,
and local validation logs. [SHA256SUMS](SHA256SUMS) pins the bundle and control
files. Extract into this report directory to inspect them.

## Next work, still open

Do not extend representation analysis: available 2B/27B profiles remain memory
neutral. Freeze a new baseline before the separate reusable-binding experiment;
keep the rejected four-token delivery experiment. Test scratch allocation only
after binding results, and exact prefix reuse afterward. No WAN or LoRA work
is part of that optimization experiment.

The missing M3 27B lane, optional skipped physical checks, 27B matching F16
answer-quality controls, and independent application retention remain open.
Reploid quality, ordinary deployed journey, and different-contributor replacement
remain separate product work; community LoRA follows quality and replacement.
