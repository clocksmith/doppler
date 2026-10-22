# Issue-triage disagreements: quantization, not GPU numerics (2026-10-22)

In the [application comparison](../issue-triage-comparison-20261022/README.md), Doppler (Q4K
reranker) and Transformers.js (fp16 export of the same checkpoint) chose different routes on 24 of
150 test issues. These cases answer the localization question for this workload: does Doppler's
GPU execution diverge from a reference, and which side matches full precision?

## Method

- Issues: the 24 disagreements plus 16 agreements chosen by SHA-256 of their index
  ([select-and-build.js](select-and-build.js)). Prompts were rebuilt exactly as the application
  builds them. All 40 used the 1,600-character budget, 3 route prompts each.
- Full-precision oracle: the FP32 ONNX export of `Qwen/Qwen3-Reranker-0.6B@e61197ed`. Its
  `model.onnx` (`7c2db33a…`) and `model.onnx_data` (`9ce4eefb…`) match the
  [2026-09-07 acquisition record](../reranker-comparison/20260907/README.md). It ran in
  Transformers.js 4.2.0 on onnxruntime-node CPU ([fp32-oracle.js](fp32-oracle.js)).
- Same-weights reference: `tests/integration/choice-scoring-reference.py` (Transformers
  Qwen3ForCausalLM, eager F32 on CPU) over the Q4_K-dequantized weights that Doppler runs.
- Doppler and Transformers.js logits come from the comparison's runs, unchanged. All values are
  in [logits.json](logits.json); issue text is not included.

## Results

| 40 issues × 3 routes | Max \|yes/no logit\| difference | Route choices equal |
|---|---|---|
| Doppler GPU vs CPU F32 on the same Q4K weights | median 0.004, max 0.015 | 40/40 |
| CPU Q4K vs FP32 oracle (route score) | median 0.65, max 3.52 | 18/40 (2/24 disagreements) |
| Transformers.js fp16 vs FP32 oracle (route score) | median 0.11, max 0.47 | 37/40 (21/24 disagreements) |

Doppler executes its Q4K model faithfully: its error against an F32 reference on identical weights
is the same size as the qualified 0.0127 bound. The gap to full precision comes from Q4K
quantization of the attention projections in this artifact (the FFN stays F16). On the
disagreements, the full-precision route was correct on 10 of 24, Doppler's on 7 and
Transformers.js's on 13.

## Consequence

The owner's ladder asked for a rare-ambiguity fallback if recomputing candidate logits at high
precision restored the winner. Here, the final projection is not where the error enters. The
deviation is spread through the network by quantized weights, and a Q4K artifact holds no
unquantized weights to recompute from. A better Doppler answer for this workload needs a less
lossy artifact. The F16 conversion in the [decision benchmark](../decision-benchmark-20261022/contract.json)
measures that directly. No runtime change follows from this report.

Component: `doppler` (evidence).
Intent: preserved.
Acceptance evidence: this directory.
Boundary effects: none.
