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
