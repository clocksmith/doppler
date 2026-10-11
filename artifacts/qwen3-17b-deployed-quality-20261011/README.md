# Isolated Qwen3-1.7B preparation

This artifact records diagnostic preparation, not an accepted application,
published model, signed release, or physical memory qualification. The original
source thinking-policy screen remains separately retained in
`../qwen3-17b-application-policy-20261011`.

The existing Rig converter prepared all 311 tensors into 24 SHA-256-verified
shards containing 1,590,331,392 bytes. The manifest digest is
`bc7ede46274d8c18ebb882bb30a87b7a31ccbf3e87afb691a63c476fa0cdcf38`.
The exact recipe, source state, converter output, manifest and shard verification
are retained here. Source geometry and RoPE theta remain those of pinned
Qwen/Qwen3-1.7B revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.

Layer matrices and the separate output head use Q4_K; the input embedding uses
F16. The head is 175,030,272 bytes and the embedding is 622,329,856 bytes.
Although the original checkpoint ties these endpoint weights, this deployed
precision profile has distinct tensors. The offline evaluator must load both
exactly and must never retie the quantized head to the F16 embedding. The prepared
tokenizer JSON is semantically identical to the original source tokenizer JSON.

`suite.json` freezes the exact manifest and the previously frozen original-source
tasks, settings, seeds, natural boundary/EOS requirements, 4,096 generated-token,
8,192 total-token and 16,000 raw UTF-16 character limits. Its SHA-256 is
`34ca02e794f8067c08bb19a0d1dbc84e39617557eeeff9432d004a459e338537`.
Reusing these cases assesses the conversion against the selected source policy;
it is not a new independent holdout or proof of GPU sampling parity.

The additive `--deployed` mode in `tools/source-model-task-quality.py` uses the
same source architecture, evaluator and tokenizer validation. Its offline loader
uses independent `gguf.dequantize` Q4_K decoding, verifies every shard and tensor,
and rejects missing, malformed or nonfinite data. Synthetic loader tests preserve
distinct embedding/head values and reject eight invalid-artifact controls.
`loader-validation.json` additionally records strict independent decoding of all
311 actual prepared tensors: all are finite, shapes match the original source
architecture, and head/embedding storage and values remain distinct. This load
check performs no generation or GPU work.

The exact manifest's 11/17-layer split plus 8,240 combined reserved F16 cache
positions has lower bounds of 1,305,114,112 and 1,230,246,400 bytes. The remaining
114,885,888 and 189,753,600 bytes must accommodate all excluded materialization,
workspaces, pools, alignment and in-flight resources. These calculations in
`capacity-lower-bound.json` cannot establish physical acceptance.
The 8,240-position example means two 24-token prompts with 4,096 output tokens
reserved for each. It does not mean two full 8,192-token reservations fit.
Longer concurrent prompts require their actual reservations and may correctly
reject under the unchanged limit; repeated 1,588-token controls are separate.

Original-source quality completed 14 of 14 cases before the first deployed
invocation. That independent deployed-weight run now uses the exact frozen suite
and manifest; its result remains pending. No GPU execution or model promotion
is recorded here. Measured partition memory, exact deployed-weight quality,
numerical comparison and model-owned final-answer decoding remain separate gates.
The standard 0.6.27 archive, catalog and signed records are unchanged.

Component: `doppler.repository-tooling`. Intent: preserved. Boundary effects:
none; offline evaluation cannot change a running model's computation.
