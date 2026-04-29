# Gemma 4 31B F16 Compute Target

## Scope

This note tracks the experimental all-f16 compute target for
`gemma-4-31b-it-text-q4k-ehf16-af32`.

The lane keeps the existing Q4K weight pack and changes execution policy through
`profiles/gemma4-31b-f16-activations-probe` plus the
`useGemma431BTextF16Activations` execution-v1 capability transform.

## Runtime Contract

- Runtime profile:
  `src/config/runtime/profiles/gemma4-31b-f16-activations-probe.json`
- Capability rule:
  `src/rules/inference/capability-transforms.rules.json`
- Transform:
  `src/config/transforms/execution-graph-transforms.js`
- Model graph source:
  `src/config/conversion/gemma4/gemma-4-31b-it-text-q4k-ehf16-af32.json`

The transform applies only when the runtime requests f16 activation, math,
accumulation, output, and KV cache dtypes, and the adapter reports both
`shader-f16` and subgroups.

## Kernel Routing

Expected f16 routing:

- Decode Q4 projections:
  `fused_matmul_q4_multicol_f16a.wgsl`
- Prefill Q4 projections:
  `fused_matmul_q4_widetile_f16a.wgsl`
- Decode attention:
  `attention_decode_online_f16.wgsl`
- Sliding prefill attention:
  `attention_small_f16.wgsl`
- Full-attention prefill:
  `attention_head512_f16.wgsl`
- Final norm, lm_head, sampling:
  f16 utility kernels

`*_f16kv` attention kernels are not f16 activation kernels. They mean f32
activations with f16 KV and should not appear in the resolved f16 lane.

## Remaining Shader Work

The selected Gemma 4 31B f16 route now uses f16 activation storage and f16
internal arithmetic across the model path:

- Q4K projection reductions:
  `fused_matmul_q4_multicol_f16a.wgsl`,
  `fused_matmul_q4_widetile_f16a.wgsl`
- Attention reductions and softmax:
  `attention_decode_online_f16.wgsl`, `attention_small_f16.wgsl`,
  `attention_head512_f16.wgsl`
- Utility math:
  `rmsnorm_f16.wgsl`, `rope_f16.wgsl`, `gelu_f16.wgsl`,
  `residual_f16.wgsl`, `matmul_f16_tiled.wgsl`,
  `matmul_gemv_subgroup_f16a.wgsl`, `sample_f16.wgsl`

Uniform ABI fields and packed-weight decode helpers still carry f32-compatible
host encodings where the surrounding API requires it, but the selected shader
entrypoints cast those values into the f16 lane before arithmetic.

## Decode Evidence

On the AMD RADV WebGPU surface used for this branch, the f16 decode path now
passes the probe profile's `maxTokens=8` cap and a 16-token override that stops
after nine generated tokens. The 16-token browser/WebGPU debug sample generated
`The sky is a clear, bright blue.` with `activationDtype=f16`, `kvDtype=f16`,
`batchSize=3`, and `readbackInterval=1`.

## Claim Boundary

This branch makes the execution plan compile and route to f16 kernels under an
all-f16 session contract, with the probe profile capped to the passing decode
batch geometry.
It does not by itself publish a catalog claim or replace the existing af32 model
identity. Promotion requires a parity receipt captured through the profile.

## Probe Command

```bash
node src/cli/doppler-cli.js debug \
  --config '{"request":{"workload":"inference","modelId":"gemma-4-31b-it-text-q4k-ehf16-af32"},"run":{"surface":"browser","browser":{"channel":"chrome","headless":true,"console":true}}}' \
  --runtime-profile profiles/gemma4-31b-f16-activations-probe \
  --json
```
