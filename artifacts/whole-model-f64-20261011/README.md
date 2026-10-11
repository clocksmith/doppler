# Independent whole-model F64 selected-prefix control

The unchanged 0.001 check passes the first three selected prefixes and fails the
fourth for **every** retained implementation. This diagnostic does not qualify a
release or activate a reference. No production arithmetic, model bytes, package,
GPU run, or acceptance tolerance changed.

| Implementation compared with independent F64 | Fourth-prefix maximum error | Logits over 0.001 |
| --- | ---: | ---: |
| Current Doppler | 0.001528939508 | 148 |
| Historical fixture | 0.001334912061 | 67 |
| Independent F32 source, chunk recurrence | 0.001220126168 | 9 |
| Independent F32 source, sequential recurrence | 0.001091038326 | 8 |

All four greedy tokens agree: 16, 198, 17, 198. Each vector contains 248,320
unmasked logits. Token agreement does not override numerical failures.

## Frozen evaluation

[contract.json](contract.json) was written and its SHA-256 recorded before the
first execution: `ff3766e12cbb22e9f3d15cf729f1943aa8821823daa788aabae62ad33f50b73b`.
It binds the exact deployed manifest, verified piece index, source configuration,
184 prompt token IDs, three frozen continuation tokens, and retained comparison
identities. No captured activation, cache, recurrent state, or output enters the
independently evolved computation.

The reference uses NumPy F64 model equations, sequential zero-initialized gated
delta recurrence, F64 normalization and RoPE, and explicit nearest-even F16 K/V
storage. Deployed Q4_K bytes are independently decoded with `gguf`; separately
quantized input/output embeddings remain separate. Weights and token IDs are the
only model inputs. The four prefix outputs come from one causal 187-token pass.
F64 is a diagnostic evaluation policy, not a change to production precision.

SHA-256 pieces authenticate deployed shards. The retained legacy manifest shard
hash limitation remains documented in the preceding source-model evidence; this
run does not claim that the legacy hash implementation was repaired.

## Validation and cache boundary

Eight synthetic checks cover F16 tie-even rounding and overflow, causal grouped
attention, RoPE, normalization offsets, zero-state recurrence, complete hybrid
model prefix causality, and an independently implemented Transformers delta-layer
comparison. The latter is a synthetic equation check, not the physical model gate.

Repeating the selected control with one instead of four BLAS threads produced an
identical report, including all layer hashes, cache hashes, and logits. Retaining
first-attention operands for a post-computation comparison also leaves those
results identical. The original executed tool is retained in
`initial-executed-tool.py.gz`; the current tool adds only that observer.

Existing GPU observations reveal 566 different stored K values and 554 V values
out of 95,744 each at layer 3. The first K difference occurs at position 0, head 1,
dimension 3; the first V difference at position 0, head 0, dimension 21. Layer-3
normalized inputs already differ by up to 0.00002370065 from independently evolved
F64 inputs. This is an upstream arithmetic difference, not evidence that the
physical cache stored the wrong value. The preceding physical-cache experiment
verified actual storage and reads; the preceding midpoint audit distinguishes
inherited differences from local rounding on each implementation's own operands.

## Disposition

Neither a captured Doppler fixture nor either F32 source is presently an
independently justified replacement that makes the selected failure pass. The
historical fixture also fails the F64 target. Keep reference activation separate
from the desired independent numerical contract and from release acceptance.

The precise unresolved decision is which independently specified evaluation and
rounding contract owns the 0.001 accuracy requirement. If the F64-equation/F16-cache
target is adopted, the selected failure remains an implementation-accuracy
requirement; reference correction cannot erase it. If an F32 operational reference
is intended, its reduction, recurrence, normalization, and rounding policy must be
specified independently, rather than chosen for agreement with captured Doppler.
This evidence does not establish a one-sided local defect or authorize another
speculative kernel patch. No additional 55-prefix campaign was run.

## Reproduction

From the Doppler root, with the pinned model and retained inputs available:

```sh
PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
python3 tools/qwen35-f64-reference.py \
  --contract artifacts/whole-model-f64-20261011/contract.json \
  --out reports/local/whole-model-f64-20261011/cache-comparison.json \
  --boundary-capture reports/local/source-model-reference-20261011/all-attention-gpu-binary.json

PYTHONPATH=/tmp/doppler-reference-deps OPENBLAS_NUM_THREADS=4 \
python3 tests/integration/qwen35-f64-reference-test.py
```

The initial runs omit `--boundary-capture`; the stability run uses one BLAS/OMP
thread. [review.json](review.json) contains compact comparisons and run hashes;
`comparison.json.gz` retains the full independent F64 vectors. All three model
invocations completed. Their failing numerical comparisons remain retained.

Component: `doppler.repository-tooling`, `doppler.tests`.
Intent: preserved.
Acceptance evidence: eight synthetic tests, three retained selected-prefix runs,
Python compilation, and `git diff --check`.
Boundary effects: none; no runtime, public contract, package, or active reference changes.
