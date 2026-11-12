# Routing-aware acceptance for routed-MoE models

Policy: [`doppler.routing-aware-acceptance/v1`](../tools/policies/routing-aware-acceptance.json)
([schema](../src/config/schema/routing-aware-acceptance.schema.json)).
Evaluator: [`tools/evaluate-routing-aware-acceptance.js`](../tools/evaluate-routing-aware-acceptance.js).
Tests: [`tests/tooling/routing-aware-acceptance.test.js`](../tests/tooling/routing-aware-acceptance.test.js).

**Status: proposed.** Every limit is marked `proposed` until the owner approves
it. A proposed policy can never produce `pass`. Flash-Next stays **BLOCKED**
until a fresh unforced run passes the approved policy.

## Why a separate gate

On the 2,187-token `inventory` prompt, the unforced Flash-Next run fails
`doppler.boundary-tolerance/quantized-v1` (30,664 violations). When the
CPU-reference expert sets are forced, it passes with 0 violations
([forced-routing control](../reports/flash-next-forced-routing-20261112/README.md)).
The failure comes from discrete expert-selection flips. A simple margin gate,
"every flip is a near-tie", cannot work: 9,754 token-layer expert sets differ,
but only 326 CPU rows have a rank-10/11 gap below 1e-4. Most differences are a
downstream cascade from one earlier flip. This gate attributes that cascade to
its root and leaves quantized-v1 unchanged.

## Rules

All rules must hold. Any missing, unreadable or hash-mismatched input fails the
gate.

1. **Identity and freshness.**
   - The policy must be approved.
   - Every artifact (unforced run, forced control, same-input checks and
     task-quality results) carries the same tuple: model ID, artifact manifest
     hash, precision (runtime profile, weights, activations, KV cache and router
     logits), backend (surface, API and binding) and device (vendor and
     architecture, plus the exact adapter string).
   - The tuple must be one of the policy's `qualificationTuples`. A pass
     qualifies only that tuple, recorded in `qualifiedTuple`. It does not
     transfer to other tuples.
   - The evidence binds the canonical policy hash and the pinned prompt-set
     hash. It starts after `approval.approvedAtUtc`.
2. **Routing attribution, for each pinned case.**
   - Inputs: the executed GPU expert indices, read back after top-k (not inferred
     from logits), and the CPU reference's own router logits and expert sets.
   - A divergence is a token-layer whose executed expert set differs from the
     CPU set. Slot order is ignored.
   - Causal cone: divergence (L, t) is downstream of (L0, t0) when L > L0 and
     t >= t0.
   - A root is a divergence that is not downstream of any other divergence.
     Every other divergence is a cascade.
   - At most `maxIndependentRootsPerCase` roots are allowed (proposed: 1).
   - Every root must be a near-tie.
   - **Near-tie margin.** Take the CPU reference's router logits for that
     token-layer. Dropped experts are those the CPU selected but the GPU did not
     execute. Added experts are those the GPU executed outside the CPU set.
     - `flipDisplacement = max cpuLogit[dropped] - min cpuLogit[added]`
     - It is a near-tie when
       `flipDisplacement <= min(absoluteCeiling, envelopeMultiplier * envelope[L])`.
     - `envelope[L]` is the largest GPU-vs-CPU router-logit difference at layer L
       in the forced-indices control, on the same case and tuple. That is the
       GPU's rounding drift on the CPU routing path.
     - Proposed values: multiplier 2, because two logits each moved by at most the
       envelope can only swap within twice it; ceiling 1e-3.
     - The rank-k/k+1 margin is also reported.
3. **Required controls, for each case.**
   - Forced-indices control: 0 executed selection differences and 0 quantized-v1
     violations on the final logits. Every GPU own-choice difference recorded
     under forcing is itself a near-tie.
   - Observation neutrality: the instrumented unforced prefill logits are
     byte-identical to an uninstrumented run.
   - Same-input operator checks under quantized-v1 have 0 violations. Embedding,
     final norm and LM head are checked at model level. Hyper-connection,
     routed-MoE and shared-expert checks cover every layer. The gated-delta and
     full-attention checks together cover every layer.
4. **Task quality.** These limits are fixed in the policy before testing and
   hashed into receipts. All values are **PROPOSED** for owner approval.
   - The pinned [prompt set](../tools/data/flash-next-routing-acceptance-prompts-v1.json)
     has 22 fresh raw-text cases: 12 arithmetic, 4 copy and 6 long-context
     retrieval cases of about 1,500 to 4,096 tokens. The long cases place the
     queried bin at depths from 8% to 92%. None was used in earlier Flash-Next
     work.
   - Distribution metrics use teacher-forced prefill, CPU F32 reference versus
     GPU, at positions 63, 127, … and the last position:
     - Top-1 agreement must be 100% where the CPU top-1 lead is at least 1 logit.
       The retained failure moved no logit more than 0.467.
     - Top-1 agreement must be at least 97% overall.
     - Mean KL(cpu‖gpu) must be at most 0.001 nats, and p99 at most 0.01 nats.
   - Exact-answer metrics use greedy decoding for at most 8 new tokens. The
     answer is the first whitespace token with trailing punctuation removed.
     - The GPU may miss at most 1 item that the CPU answers correctly.
     - The net item loss may be at most 1.
     - At least 90% of items must have token-identical CPU and GPU continuations.
   - These metrics compare the GPU with the CPU reference on the same artifact.
     They do not set an absolute quality bar.

The evaluator recomputes every aggregate from per-row and per-item records and
does not trust supplied totals.

## Evidence shape

`doppler.routing-aware-acceptance-evidence/v1`, with paths relative to the
evidence file:

```json
{
  "schema": "doppler.routing-aware-acceptance-evidence/v1",
  "policySha256": "sha256:…", "promptSetSha256": "sha256:…", "runStartedAtUtc": "…",
  "tuple": { "modelId": "…", "artifactManifestSha256": "…", "precision": {…}, "backend": {…},
             "device": { "vendor": "amd", "architecture": "rdna3", "adapter": "…" } },
  "cases": [{
    "caseId": "ledger-00", "numLayers": 48, "numTokens": 2301, "numExperts": 512, "topK": 10,
    "cpu": { "fixture": "cpu/ledger-00/fixture.json" },
    "unforced": { "mode": "observe", "selectionSource": "gpu-executed-readback", "tuple": {…},
                  "prefillLogitsSha256": "…", "uninstrumentedPrefillLogitsSha256": "…",
                  "layers": [{ "layer": 0, "experts": { "file": "…/gpu-experts-0.u32", "sha256": "…" } }] },
    "forcedIndices": { "mode": "indices", "completed": true, "tuple": {…},
                       "executedSelectionDifferingTokenLayers": 0,
                       "logits": { "policyId": "doppler.boundary-tolerance/quantized-v1", "count": 248320, "outsidePolicy": 0 },
                       "layers": [{ "layer": 0, "routerLogitsMaxAbsDiffVsReference": 2.7e-5,
                                    "ownSelectionDifferences": [{ "token": 5, "experts": [ … ] }] }] }
  }],
  "sameInputOperators": { "tuple": {…}, "policyId": "doppler.boundary-tolerance/quantized-v1",
                          "checks": [{ "family": "routed-moe", "caseId": "ledger-00", "layers": [0, 1, …], "count": 0, "outsidePolicy": 0 }] },
  "taskQuality": { "file": "task-quality.json", "sha256": "…" }
}
```

The CPU fixture is the existing `doppler.flash-next-routing-selection-fixture/v1`
format, written by
[prepare-routing-fixture.py](../reports/flash-next-forced-routing-20261112/prepare-routing-fixture.py).
Task-quality results use `doppler.routing-aware-task-quality/v1`:

- `caseTokenCounts`
- `distributionRows`: `{caseId, position, cpuTop1, gpuTop1, cpuTop1MarginLogits, klCpuGpuNats}`
- `answers`: `{caseId, cpuTokens, gpuTokens, cpuText, gpuText}`

## Fresh-test procedure (after owner approval only)

1. **Approve.** The owner edits the policy:
   - Set `approval.status`, `nearTie.status` and `taskQuality.status` to
     `approved`, and fill in `approvedBy` and `approvedAtUtc`.
   - Adjust any limits.
   - Commit the policy. Record the output of
     `node tools/evaluate-routing-aware-acceptance.js --policy-hash` as the
     run's `policySha256`.
   - Changing any limit after this point changes the hash and invalidates
     runs.
2. **Freeze inputs.** Tokenize the pinned prompt set with the artifact
   tokenizer, without special tokens. Write the input receipt before any CPU or
   GPU execution. Check each case's token bounds.
3. **CPU reference.** Use the pinned producer
   (`cpuReference.producerSha256`). For every case, write:
   - all-position router rows, converted to fixtures with
     `prepare-routing-fixture.py`;
   - logits at the sampled positions;
   - greedy continuations.

   It never reads GPU output.
4. **Unforced GPU run.** Run each case on the tuple's runtime profile, with
   `tools/run-flash-next-forced-routing.js`, `mode: "observe"` and
   `recordSelections: true`. This writes `selections/gpu-experts-<L>.u32`.
   Also run each case once without instrumentation, to check observation
   neutrality.
5. **Forced-indices control.** Run each case with `mode: "indices"`. Score the
   final logits with
   [compare-forced-routing.py](../reports/flash-next-forced-routing-20261112/compare-forced-routing.py)
   under unchanged quantized-v1. Take the router-logit envelope and
   `ownSelectionDifferences` from `control.json`.
6. **Same-input operator checks.** Use the
   [same-input audit](../reports/flash-next-new-prompts-20261110/audit-same-input-operators.py)
   method on one pinned long case, extended to all 48 layers, final norm and LM
   head. Record quantized-v1 violation counts for each family.
7. **Task quality.** Run teacher-forced prefill at the sampled positions and
   greedy decoding on the GPU. Compare with the CPU reference, and write the
   task-quality file.
8. **Evaluate.** Run
   `node tools/evaluate-routing-aware-acceptance.js --evidence EVIDENCE.json --out VERDICT.json`.
   Exit code 0 means `pass` for the recorded tuple only.

Only step 8 decides. Do not tune between steps. A failed run stays failed. A
new attempt needs a new policy version or a new approved prompt set.

## Gaps this procedure depends on

- Same-input checks for Layers 25 to 47, final norm and LM head do not exist
  yet. The retained audit covers Layers 0 to 24 at one position.
- Step 7 needs a GPU teacher-forced sampled-position logit capture and a greedy
  decode harness that bind this policy hash. Neither is written yet.
- The selection dump (`recordSelections`) and per-token own-selection records
  are new. They have unit tests only and have not run on a GPU.
