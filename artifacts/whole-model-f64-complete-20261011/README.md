# Complete frozen-prefix numerical disposition

**The user explicitly approved this independent numerical contract on 2026-10-11.**
[Approval](approval.json) binds the unchanged proposal and all three contracts.
Reploid now compares against these F64 values without rounding them to F32; runtime
precision and 0.001 remain unchanged. Approval grants no candidate a pass.

The following records describe the earlier diagnostic execution, when reference
authority was still pending. Their original identities and failed results remain
unchanged, including the historical fixture and standard 0.6.27 archive.

The three contracts were frozen before execution. They supply the exact prompt
tokens and frozen continuation tokens for sequences of 20, 22 and 234 tokens,
covering 2, 2 and 51 output prefixes. The existing independent F64 equations are
unchanged. The sole tool change permits omission of the optional sequential F32
comparator, whose retained evidence covers only eight prefixes. The existing
complete F32 comparator is still included.

## Independent calculation and scope

The calculation independently decodes deployed Q4_K weights and widens the exact
decoded values, F16 weights and F32 weights into F64. Recurrence, attention,
normalization and projections evolve from their own inputs. K/V values round to
nearest-even F16 at the declared storage boundaries, then widen for subsequent
arithmetic. No Doppler hidden state, cached values or logits enter the
calculation. Historical tokens are supplied only as explicit prefix inputs;
the calculation is not forced to select the expected next token.

All 18 deployed shard identities are checked through the retained piece index.
The documented legacy manifest BLAKE3 mismatch remains a limitation; this work
neither fixes that metadata nor authorizes a signed model release. The source
configuration, manifest, piece index, historical fixture and comparison inputs
are hash-bound in each contract.

Execution used one BLAS/OMP thread. Extending the original four-prefix causal
calculation to all 51 prefixes changes those first four F64 vectors by at most
4.27e-14. All 55 greedy tokens agree across the independently calculated F64,
historical, source F32 and tested Doppler outputs. Token agreement does not
substitute for the full-vector numerical requirement.

## Physical controls

Every fresh physical replay uses the existing producer harness with
`DOPPLER_FORENSIC_FULL_PREFIXES=1`, exact archive bytes, matching shader pins and
the unchanged frozen tokens. Each covers all 55 prefixes plus two first-request
reuse controls. Reuse outputs must match exactly. There is no operator readback
capture in these executions; native pipeline selection observations retain
executed entry points. The host is Linux with Chrome 146.0.7680.177.

This is an unsplit numerical diagnostic using the harness's existing
6,000,000,000-byte allowance. It is **not** acceptance of the 1,420,000,000-byte
contributor ceiling, distributed placement, cancellation, task quality or Mac.

The controlled archive chain changes exactly one shader at each step:

1. Standard `9280ead5`: the unchanged delivered archive.
2. F16-only `74cb0329`: only `matmul_f16w_f32a_tiled.wgsl` changes.
3. Combined `cd50bcc0`: only `fused_matmul_q4_widetile.wgsl` additionally changes.

The newer source diagnostic `307104de` also contains subsequent public-contract,
sampling and selective-acquisition work. Its separate complete replay prevents
attributing an older capture to that source. `archive-differences.json` records
every changed package member. Fresh standard output exactly reproduces all 55
retained pre-repair `e0f558cf` outputs.

Comparison files retain every prefix, maximum/RMS difference, count exceeding
0.001, token identity and selected/worst-error values. They distinguish the
active historical fixture from investigative independent F32 and F64 targets.
The complete candidate matrix and conditional recommendation are in
`disposition.json`.

## Reference decision

The F64 calculation provides an independently evolved, more precise evaluation
of the declared equations with explicit F16 cache boundaries. It is a meaningful
candidate reference **if** the intended contract is final-logit accuracy against
that evaluation. It is not automatically the prescribed behavior of every
F32 implementation: F32 reassociation and pre-storage rounding can change which
F16 cache value is stored, and subsequent state evolves from that value.

Conversely, an F32 reference requires a declared evaluation and rounding policy;
the label “F32” does not specify reduction order or fused operations. The two
previous independent F32 implementations already differed beyond 0.001 on a
selected prefix. A reference must be chosen for the intended numerical meaning,
not because a candidate happens to pass it. Neither reference choice supplies
full acceptance for the tested candidates. No captured-output fixture proposal
is activated, no threshold is relaxed, and `main_subgroup` is not restored.

The appropriate disposition is to keep numerical acceptance open, explicitly
review the evaluation contract, and retain candidate arithmetic as unqualified
until its complete outcomes support the intended contract. Historical failures
remain intact. This diagnostic does not prove the residual error unavoidable or
identify every upstream cause.

## Reproduction and retention

From the repository root, execute each frozen contract with the existing tool:

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 tools/qwen35-f64-reference.py \
  --contract artifacts/whole-model-f64-complete-20261011/request-0-contract.json \
  --out reports/local/whole-model-f64-complete-20261011/request-0.json
```

Repeat for request 1 and 2. Each `*-replay-binding.json` records the exact GPU
command, archive and manifest. The combined/current diagnostic uses the existing
decompressed `../q4-accumulation-repair-20261011/diagnostic-manifest.json.gz`
beside links to unchanged model assets, never overwriting the deployed manifest.

`compare.py` verifies the retained independent vector shards, source/contract
hashes, capture identity, full coverage and exact reuse before comparison. Pass
`--capture`, `--binding`, `--label` and `--out` for the selected replay. The default
binding/label are the current-source diagnostic.

Full F64 vectors and GPU captures remain under
`reports/local/whole-model-f64-complete-20261011/`; committed indexes/receipts bind
their exact paths and hashes. The 109.8 MB compressed F64 vectors are deliberately
excluded from Git history. Committed comparison rows and logs preserve every
failed prefix without inflating ordinary clones. Independently rerunning the
frozen contracts reconstructs the vectors; local shards group ten output rows.

Validation: eight existing independent-equation tests passed, including
source-implementation recurrence checks. Full physical replay scope and shader
pin assertions passed separately; numerical failures remain failures.

Component: `doppler.repository-tooling`, `doppler.tests`. Intent: preserved.
Boundary effects: none; no runtime, kernel, manifest or active reference changed
in this evidence batch.
