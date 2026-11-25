# External reference checks

Doppler is JavaScript-only. It no longer ships the PyTorch/Transformers debug
scripts that used to live here (`hf_embed_check.py`, `hf_layer_out.py`,
`hf_attn_debug.py`, `hf_weights.py`, `hf_rope_check.py`,
`hf_qwen35_linear_attn_debug.py`) or the boundary provider adapter
`tools/qwen35-hf-boundary-reference.py`. They remain at commit
[`3e485ca93`](https://github.com/clocksmith/doppler/tree/3e485ca9303ee51131884b6242ce4a7fe4753c24/src/debug/reference)
for anyone who wants to run them in their own Python environment.

## Comparing Doppler with an upstream framework

1. Run the original SafeTensors checkpoint in the upstream framework, outside
   this repository, at a pinned framework and model revision.
2. Emit the observed boundary tensors as a `doppler.boundary-provider-capture/v1`
   document (the retired adapter above shows one way to do it).
3. Bind the capture with `doppler boundary source-capsule` and compare it with
   the canonical boundary workflow. The comparison and all acceptance decisions
   run in JavaScript.

Retained reference outputs under `artifacts/`, `reports/` and
`tests/fixtures/` are the golden inputs for the JavaScript comparers; they do
not need the framework to be installed.

## Isolating a divergence

1. Find the first layer whose output diverges from the reference capture.
2. Check embeddings and their scaling.
3. Check attention at the diverging layer, then RoPE if Q/K differ after
   position 0.
4. Compare dequantized weight values for the affected tensors.

The [debug protocol](../../../docs/agents/debug-protocol.md) covers Doppler's
own trace and probe tooling for each step.
