# Receipt supersession index and errata

Receipts under `artifacts/` and `reports/` are frozen: their file inventories
are hash-checked, so a later correction or closure never edits the earlier
receipt. This index records where each correction lives. Read a receipt
together with the entries that name it.

- "X is superseded by Y" means Y corrects, closes or replaces a conclusion that
  X states. X stays valid as a record of what was observed at the time.
- The [errata](#errata) list receipt summaries that contradict their own
  retained data. The data files are authoritative.

Paths are repository-relative. Each entry was checked against the named files
on 2026-11-26.

## Flash-Next numerics

- [flash-next-qualification-20261105](../../reports/flash-next-qualification-20261105/README.md)
  is superseded by
  [flash-next-qualification-20261106](../../reports/flash-next-qualification-20261106/README.md),
  then [20261107](../../reports/flash-next-qualification-20261107/README.md) and
  [20261109](../../reports/flash-next-qualification-20261109/README.md): each
  continues the same failing long-input gate (HC product rounding, then a
  separate F32-KV candidate, then source-order controls). None qualifies the
  model.
- [flash-next-qualification-20261109](../../reports/flash-next-qualification-20261109/README.md)
  is superseded by
  [flash-next-origin-20261109](../../reports/flash-next-origin-20261109/README.md):
  the GDN softplus repair cuts the 2,304-token F32-KV check from 13 violations
  to 1.
- [flash-next-origin-20261109](../../reports/flash-next-origin-20261109/README.md)
  is superseded by
  [flash-next-route-products-20261110](../../reports/flash-next-route-products-20261110/README.md):
  separately rounded MoE route products take the same check from 1 violation to
  0 under the unchanged `quantized-v1` policy (one long-prefill check on AMD
  Node/WebGPU, not model qualification).
- [numerical-discontinuity-20261109](../../reports/numerical-discontinuity-20261109/README.md)
  is superseded by the origin and route-products reports above. Its
  `OPERATOR-FOLLOWUP.md` (line 89) and `LAYER25-PREROUTER-WITNESS.md` (lines 16
  and 165) say the final 13 violations remain open; they were closed for that
  check by the softplus and route-product repairs.
- [flash-next-reference-controls-20261110](../../reports/flash-next-reference-controls-20261110/README.md)
  is superseded by
  [flash-next-forced-routing-20261112](../../reports/flash-next-forced-routing-20261112/README.md):
  reference controls ruled out CPU-reference noise for the 2,187-token
  `inventory` failure; forcing the CPU expert selection makes the GPU pass,
  attributing the failure to routing flips.
- [flash-next-routing-aware-fresh-20261112 README](../../reports/flash-next-routing-aware-fresh-20261112/README.md)
  ("INCOMPLETE, GPU steps on hold") is superseded by its
  [RESUME.md](../../reports/flash-next-routing-aware-fresh-20261112/RESUME.md):
  the GPU steps finished; the forced-indices comparisons, CPU greedy rounds 2
  to 7, the same-input audit and the evaluator remain. Both files record policy
  hash `sha256:7a63c40c…`. Since commit `ef4137a46` (2026-11-15) changed
  `tools/policies/routing-aware-acceptance.json`, the policy on main hashes to
  `sha256:ac3efb02605531dfc99a801ba00bf965b61babe180cb42de6b0f137f46eec679`
  (`node tools/evaluate-routing-aware-acceptance.js --policy-hash`), so the
  frozen run is bound to the earlier policy bytes.

## Qwen 2026-10-17 chain

- [qwen-competition-20261017](../../reports/qwen-competition-20261017/README.md)
  and
  [qwen-service-qualification-20261017](../../reports/qwen-service-qualification-20261017/README.md)
  ("Transformers.js had lower warm first-token latency in every lane") are
  superseded by
  [qwen-sampling-correction-20261017](../../reports/qwen-sampling-correction-20261017/README.md):
  its [comparison table](../../reports/qwen-sampling-correction-20261017/comparison-table.md)
  has Doppler ahead on Vulkan 2B TTFT (for example warm 64-token input,
  156.9 ms vs 241.0 ms). It is a later candidate, so it reports a new
  measurement rather than restating the old one.
- [qwen-prepared-execution-followup-20261017](../../reports/qwen-prepared-execution-followup-20261017/README.md)
  (AMD 0.8B loss under the default profile) is superseded by
  [qwen-profile-correction-20261017](../../reports/qwen-profile-correction-20261017/README.md):
  the Linux comparison lane now selects the existing Qwen decode fusions, and
  Doppler wins decode and complete-generation latency in all eight rows. The
  default-profile loss itself stays valid.
- [weight-representation-20261017](../../reports/weight-representation-20261017/README.md)
  (hosted CI gate open, standard-BLAKE3 discrepancy not repaired) is superseded
  by
  [weight-representation-closure-20261017](../../reports/weight-representation-closure-20261017/README.md):
  hosted run 39085130244 passed at `bbcf7ffd` and BLAKE3 was corrected. Only the
  M3 27B lane remains open.
- [qwen-lm-head-optimization-20261019](../../reports/qwen-lm-head-optimization-20261019/README.md)
  (M3 control request -42.3%, acceptance evidence listed) is superseded by
  [qwen-lm-head-closure-20261021](../../reports/qwen-lm-head-closure-20261021/README.md):
  the AMD half fails promotion with no complete-request gain, and the kernel
  change was removed from the shared default (kept as `candidate.patch`).

## Qwen 3.6 35B and 27B

- [qwen36-35b-startup-20261101](../../reports/qwen36-35b-startup-20261101/README.md)
  prefill comparison is superseded by
  [qwen36-35b-prefill-tile16-20261102](../../reports/qwen36-35b-prefill-tile16-20261102/README.md):
  the native adapter's prefill counter covers 27 vs 28 (or 2286 vs 2287) input
  tokens and a different logit boundary, so earlier native-vs-Doppler prefill
  latencies from that adapter are not directly comparable.
- [qwen36-35b-native-moe-registers-20261101](../../reports/qwen36-35b-native-moe-registers-20261101/README.md)
  and [qwen36-35b-prefill-20261101](../../reports/qwen36-35b-prefill-20261101/README.md)
  (256-token prefill chunks in the candidate profile) are superseded by the
  tile16 report: `profiles/qwen3-6-35b-a3b-q4_0-candidate` now uses 2048-token
  chunks with the sixteen-query tile.
- The [qwen-shared-performance diagnostic](../../reports/qwen-shared-performance-20261102/diagnostic-20261103/README.md)
  ("Native MTP for 35B MoE is not implemented") is superseded by
  [native-mtp-20261103](../../reports/qwen-shared-performance-20261102/native-mtp-20261103/README.md):
  draft chains, batched verification and recurrent recovery now execute in a
  frozen experiment (not a serving release or speed claim).

## Choice scoring

- [choice-scoring](../../reports/choice-scoring/README.md) (archive 0.6.4) is
  superseded by
  [choice-scoring-successor-20261022](../../reports/choice-scoring-successor-20261022/README.md)
  (via the superseded
  [20261021 candidate](../../reports/choice-scoring-successor-20261021/README.md)).
  Its CPU reference step runs `tests/integration/choice-scoring-reference.py`,
  which is no longer in the tree (scripts kept at commit 3e485ca93).

## Packages

- [package-release-20261111/HANDOFF.md](../../reports/package-release-20261111/HANDOFF.md)
  (instructions for `doppler-gpu` 0.7.1, the package's former name) is
  superseded by
  [package-release-0.7.2](../../reports/package-release-0.7.2/README.md)
  (`doppler-gpu` 0.7.2). Its
  `release-status.json` records a `knownDefect`: installed `doppler` and
  `doppler-serve` exit 0 without output through npm bin links, fixed in 0.7.3
  ([release-status.json](../../reports/package-release-0.7.3/release-status.json)
  records the bin-link install check).
- [dopplerx-0.1.5-qualification](../../reports/dopplerx-0.1.5-qualification/README.md)
  is bound to the PR #15 tarball `ecb2a13e0823ea46…`, which was never
  published: npm lists only `dopplerx` 0.0.0-stage and 0.1.7. It does not
  qualify npm `dopplerx@0.1.7` or the 0.2.0 candidate.

## Artifacts

- [reference-target-compatibility-20261011](../../artifacts/reference-target-compatibility-20261011/README.md)
  ("reference decision is pending explicit approval") is superseded by
  [whole-model-f64-complete-20261011](../../artifacts/whole-model-f64-complete-20261011/README.md):
  the independent F64 contract was approved on 2026-10-11 (`approval.json`,
  commit `0796f6eff`). Approval grants no candidate a pass.
- [address-redaction-20261013](../../artifacts/address-redaction-20261013/README.md)
  is superseded by the `address-redaction-20261014` follow-up, which covered
  `user@host` login forms the first pass missed. That folder moved to the
  maintainers' store on 2026-10-29; see
  [external-evidence-index.json](external-evidence-index.json). The later
  [private-detail sanitization](../../artifacts/private-detail-sanitization-20261030/README.md)
  widened the scope again.
- The Qwen 0.8B Capsule chain in `artifacts/qwen35-08b-capsule-intake-20261012/`:
  [signed-candidate](../../artifacts/qwen35-08b-capsule-intake-20261012/signed-candidate/README.md)
  (archive `102fd0c4…`) is superseded by
  [signed-0fad-candidate](../../artifacts/qwen35-08b-capsule-intake-20261012/signed-0fad-candidate/README.md)
  (archive `0fadac51…`), which is superseded by
  [signed-closure-candidate](../../artifacts/qwen35-08b-capsule-intake-20261012/signed-closure-candidate/README.md)
  (corrected manifest `709894a7`, same 0fad archive).
- [model-repairs-20261109/gemma-repeatability](../../artifacts/model-repairs-20261109/gemma-repeatability/README.md)
  is superseded in scope by
  [qwen-softmax-reduction](../../artifacts/model-repairs-20261109/qwen-softmax-reduction/README.md):
  the earlier packet's snapshot never regenerated `rule-bundle.json`, so it
  observed only the unfused path; the follow-up retests the fused path.
- [ordered-partitions-20261013/three-stage-delivery-preparation](../../artifacts/ordered-partitions-20261013/three-stage-delivery-preparation/README.md)
  ("the prepared await-device.lost harness is a separate unrun diagnostic") is
  superseded by
  [intel-host-paired-control/await-lost-replay](../../artifacts/ordered-partitions-20261013/intel-host-paired-control/await-lost-replay/README.md),
  where the four installed paired-cast controls passed on intel-host.

## Errata

Each entry names the summary line and the data it contradicts.

- `artifacts/document-search-diagnosis-2026-09-26/README.md:86-87` says
  identifier queries "rank correctly". In
  [component-matrix.json](../../artifacts/document-search-diagnosis-2026-09-26/component-matrix.json)
  both pairs with the MiniLM reranker rank 4/5 identifier queries correctly:
  `version` is an identifier query, and the same README reports it wrong.
- `artifacts/selective-materialization-20261012/README.md:29` reports 22/51 for
  "Retain source recurrent RMS epsilon with ordinary mapped stores".
  [interaction-result-summary.json](../../artifacts/selective-materialization-20261012/interaction-result-summary.json)
  (`retain-recurrent-rms-epsilon-only`) has 18 of 51 prefixes over tolerance;
  the maximum `0.003513270518` matches.
- `artifacts/qwen38-flash-next-20261013/README.md:91-95` says norm vectors are
  exact and projection classes have 0.01% to 0.05% inexact values, with "a few
  hundred" flushed per class. In
  [storage-precision-scan.json](../../artifacts/qwen38-flash-next-20261013/storage-precision-scan.json)
  the attention and MLP `hc_norm` vectors have 7 and 4 inexact values, the
  inexact share reaches 0.247% (`ple.conv1d.weight`), and up to 1,737 values
  flush to zero (`embed_tokens.weight`). The 91-value example is correct.
- `artifacts/buffer-retirement-2026-10-05/manifest.json:13,15` give digests
  `3a791df1…` and `ebccd9c7…` for `corrected-generation-reopen.log` and
  `corrected-generation.log`. The retained `.log.gz` files (whose own digests
  match) decompress to `4e4d437a4e5924bd7151b43e5ccc0939a7e4318f0e71be6b00c48457c9e88d7b`
  and `a76834a48a41e88a0799e94c0bb97a617b21461cd95372dc39a3888eea72e2d8`.
- `reports/numerical-discontinuity-20261109/LAYER25-PREROUTER-WITNESS.md:105-109`
  (section 3.2) uses `diffRms` 2.411e-6 / 2.589e-6 and z = -1.547 / -0.643 for
  positions 1717 / 2271.
  [layer0-to-24-drift-trace-receipt.json](../../reports/numerical-discontinuity-20261109/layer0-to-24-drift-trace-receipt.json)
  records 1.860e-6 / 1.998e-6, expected dot σ 2.103e-6 / 2.379e-6 and
  z = -2.005 / -0.832; section 3.1 of the same file uses the JSON values.
- `reports/flash-next-qualification-20261107/README.md:123` says nine focused
  JS tests pass;
  [development-tests.log](../../reports/flash-next-qualification-20261107/development-tests.log)
  reports `# tests 7`, `# pass 7`.
- `reports/package-release-0.7.2/README.md:25` says projection calls took
  204–861 ms. The 36 `stackProjectionMs` samples in
  [browser-projection.json](../../reports/package-release-0.7.2/browser-projection.json)
  range from 192.1 to 987.7 ms.
- `reports/demo-live-default-20261108/README.md:13-16` says the physical run
  validates the activation-change stream.
  [physical-16-token.json](../../reports/demo-live-default-20261108/physical-16-token.json)
  records `activationChanges: 0` and an empty `lastActivation`; the 71
  visible changes are `colorChanges`.
- `reports/dopplerx-0.1.5-qualification/README.md:38` says temperature
  calibration "reduced held-out log-loss and ECE to 0". In
  [mars-metal-js-capabilities.json](../../reports/dopplerx-0.1.5-qualification/mars-metal-js-capabilities.json)
  the temperature (0.101957) was fitted on 6 cases that were all already
  correct, which drives calibrated probabilities to 1/0. Zero loss on 6
  separable held-out cases shows a degenerate fit, not a calibration gain.
