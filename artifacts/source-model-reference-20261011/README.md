# Captured attention history and source recurrence controls

Item 7 of the [canonical checklist](https://github.com/clocksmith/reploid/blob/main/TODO.md)
remains open. No runtime arithmetic, accepted reference, precision policy, or
0.001 threshold changed. No new package was built or deployed.

Component: `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved. Boundary effects: observation and independent offline
diagnostics only; no observed tensor is fed back into execution.

## Hypotheses and results

The selected failure is request 2, decode step 3. With unchanged source-model
computation, its maximum logit difference remains 0.0014121532440185547.

**Attention-local error:** reconstruct all six full-attention layers' complete
K/V histories from captured GPU projections and rotary outputs, apply the
declared F16 storage rounding, and compute attention in float64 using each
captured GPU query. All six outputs agree within 0.000000397. This control does
not establish an attention-kernel defect. The reconstructed histories differ
from independently evolved source-model caches; those differences are retained
separately. They are reconstructions from pre-cache observations, not direct
readbacks of the physical cache allocation.

**Inherited recurrence ordering:** run the independent Transformers sequential
gated delta recurrence for prefill in place of its chunked recurrence. This
diagnostic keeps F32 arithmetic, the deployed weights and F16 attention caches.
The selected logit difference falls to 0.0011813640594482422, still above 0.001.
This explains some sensitivity but does not resolve numerical acceptance or
justify replacing the reference. It is not a proposed runtime substitution.

`attention-history.json` and `prefill-order-control.json` retain eight prefix
results, actual-operand comparisons, environment versions and identity hashes.
Full raw reports, including logits, stay at their hash-bound local paths.

**Linear-operation error:** `linear-history.json` replays layer 0's complete
captured QKV, gate and decay projections through independent float64 causal
convolution, gated delta recurrence and output normalization. It starts from
zero state and uses the deployed convolution, decay and normalization weights.
Across 187 tokens / 382,976 output values, maximum difference is
0.0000008118932455; the final token differs by at most 0.0000001691524057.
This control does not establish a local layer-0 recurrence defect. It does not
measure the other recurrent layers or read back the GPU's internal state.
The full-model logit failure remains unchanged. Its executed tool is retained
separately as `qwen35-linear-history-reference.py.gz`.

## Physical capture

`physical-capture.json` identifies the standard 0.6.27 archive and a new physical
browser capture on Linux. All declared shader pins matched. Observation changes
produced zero difference from the uninstrumented logits. The run completed 55
shared prefixes and the existing first-request reuse control. Execution
completion is not historical-reference acceptance.

The initial expanded capture failed at browser-automation IPC with
`ERR_STRING_TOO_LONG`; `all-attention-gpu.log.gz` preserves it. Observations now
transfer one operator row at a time, with numeric captures encoded as F32 bytes
instead of decimal arrays. `all-attention-gpu-binary.log.gz` records the successful
rerun. The original baseline setup failure is also retained. Executed diagnostic
and capture sources are retained compressed; their hashes match the receipts.

## Reproduction

Use the dependencies and deployed source configuration identified by the
[preceding control](../source-model-reference-20261010/README.md). Write a target
file containing `{"index":2,"step":3}` and create the output directory first.

```bash
DOPPLER_FORENSIC_FULL_PREFIXES=1 node tests/integration/frozen-reference-producer-replay.js \
  artifacts/partition-contracts-20261010/doppler-gpu-0.6.27.tgz \
  models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  reports/local/source-model-reference-20261011/all-attention-gpu-binary.json \
  current reports/local/frozen-reference-producer/decode-target.json

PYTHONPATH=/tmp/doppler-reference-deps python3 tools/qwen35-deployed-reference.py \
  --model models/local/qwen-3-5-0-8b-q4k-ehaf16 \
  --source-config artifacts/source-model-reference-20261010/source-config.json \
  --capture reports/local/frozen-reference-producer/doppler-current026-full-linux.json.gz \
  --reference ../reploid/tests/fixtures/distributed-reference.json.gz \
  --piece-index ../reploid/self/config/model-pieces/qwen-0-8b-pieces.json \
  --piece-index-identity sha256:18adb1f08f4e694d357a27f7a5d67ea57a7d50c5e82aa7fda099084b445974b7 \
  --prefixes 8 --threads 8 \
  --boundary-capture reports/local/source-model-reference-20261011/all-attention-gpu-binary.json \
  --out reports/local/source-model-reference-20261011/final-default-comparison.json
```

For the recurrence-order control, add `--linear-prefill source-recurrent` and use
a different output path. The normal default remains `source-chunk`.

Acceptance evidence: both diagnostic invocations completed; their numerical
verdicts remain failed. `PYTHONPATH=/tmp/doppler-reference-deps python3
tests/integration/qwen35-deployed-reference-test.py` passes constant-attention,
binary-encoding equivalence, missing-history rejection and nonfinite-input
rejection controls. These are synthetic diagnostic tests, not model acceptance.
Python compilation, JavaScript syntax, CATSCAN and diff checks pass.
The recurrence extension also passes a scalar two-token closed-form reference
and rejects incomplete projection history (`diagnostic-checks-linear.json`).
Reproduce the linear control using the same default command and a separate
output path; no additional physical capture is required.


## All recurrent layers

`all-recurrence.json` extends the actual-operand float64 replay to all 18 recurrent
layers for the same request 2 / decode step 3 capture. Each replay spans all 187
tokens from zero state. Maximum local error across the layers is
0.0000092796421356 (layer 6); the largest final-token error is
0.0000021167316851 (layer 22). This does not establish a recurrence defect.
The independent full-model logit discrepancy remains 0.0014121532440185547,
so item 7 is still open. No numerical reference or runtime arithmetic changed.

Executed source is Doppler `32e7920c`. The hash-bound raw observation and report
remain in `reports/local/source-model-reference-20261011/all-recurrence/`;
`all-recurrence-gpu.log.gz` retains physical completion and historical comparison
failures. Synthetic controls additionally check distinct weights in different
layers and reject incomplete later-layer histories.

## Feed-forward layers and final projection

`all-ffn.json` checks all 24 feed-forward layers and the final projection using
the captured inputs, independently decoded deployed weights and float64
equations. The largest local feed-forward difference is 0.0000013279863708
(layer 23); the final projection differs by 0.0000075189989683. These results
do not identify a local feed-forward or final-projection defect. The complete
source-model discrepancy remains 0.0014121532440185547, above 0.001.

The physical capture used `bac4ad37` and the unchanged standard archive.
`all-ffn-gpu.log.gz` retains completion and failed historical comparisons.
The first offline comparison rejected metadata-only recurrence observations;
`all-ffn-missing-capture-failure.log.gz` preserves that failure. The diagnostic
now selects only captured recurrence outputs, with a synthetic regression.
`qwen35-all-ffn-reference.py.gz` preserves the executed corrected source; the
receipt binds the complete raw report and GPU observation by SHA-256.

## Cache rounding sensitivity

`cache-rounding.json` retains independent source values immediately before F16
cache storage and compares them with captured GPU pre-storage values. In the
first full-attention layer, maximum value-projection difference grows from
0.0000052452087402 before storage to 0.0009765625 after rounding. Of 661 different
stored values, 611 are adjacent F16 numbers. Maximum key difference grows from
0.0000316575169563 to 0.001953125.

Holding each captured query fixed, substituting the independently evolved cache
changes float64 attention output by up to 0.0001005140493 in layer 3 and
0.0019243072713 in layer 23. This is an offline counterfactual; no observed value
is fed into either model execution. It demonstrates amplification through cache
rounding, not a complete attribution of the final logit difference or a qualified
reference replacement. Complete-model error remains 0.0014121532440185547.

Eleven synthetic checks include an exact F16-midpoint crossing and constant-query
cache substitution with a known output difference. No runtime, model precision,
acceptance tolerance or protected fixture changed.

The two retained independent F32 source recurrence implementations also disagree
with each other by 0.0010614395141601562 at the selected decode prefix
(`source-order-difference.json`). They use the same weights, tokens and F16 cache
policy. This demonstrates reference-order sensitivity; it does not authorize
relaxing 0.001 or treating either implementation as an approved replacement.

## Existing 2B model task-quality control

`model-2b-source-generation.json` independently replays the fresh 208-through-219
counting task after the existing 2B model passed Reploid's six-case quality suite.
All 48 generated token IDs match the physical unsplit browser result, including
EOS. The source configuration comes from the source revision pinned in that
model's `origin.json`; its URL and SHA-256 are retained. Deployed weights and
tokenizer are checked against the 2B piece index.

This is a scoped generation control for an already-integrated model. It neither
replaces the small-model numerical reference nor establishes distributed
capacity, complete application acceptance or deployment.

## Independent reproduction of the incomplete larger-model answer

`model-2b-counting-source.json` reports the original request to count from one
to two hundred, using the deployed 2B weights, captured tokenizer inputs and
unchanged application generation settings. Independent source equations match
all 372 Doppler output tokens, including EOS after 120. Full source output,
input control and invocation log are retained compressed. This establishes
that the incomplete answer reproduces outside Doppler; it does not qualify the
answer, clear full-model numerical acceptance or justify changing protected
requirements. No runtime or reference changed.
