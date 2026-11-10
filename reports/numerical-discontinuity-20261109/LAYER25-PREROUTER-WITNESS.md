# Layer-25 pre-router witness, physical WebGPU counterfactual, and Layers 0..24 drift trace

This extends the frozen diagnostics in `08c12ad6` and `8115ff46` without
modifying any of their files. It covers:

1. **Offline constructive `0-ULP` WGSL witnesses** across all 27 Layer-25 stages
   ([`layer25-prerouter-witness-results.json`](layer25-prerouter-witness-results.json),
   `newGpuInvocations: 0`).
2. **Fresh physical WebGPU execution of the substituted-input counterfactuals**
   (`R0_physical_gpu`, `R1_physical_gpu`, `R2_physical_gpu`) on `Jupiter`'s
   AMD RDNA-3 GPU ([`layer25-physical-gpu-counterfactual-receipt.json`](layer25-physical-gpu-counterfactual-receipt.json),
   `newGpuInvocations: 56`).
3. **Backward drift trace and implementation-vs-variation separation across Layers `0..25`**
   ([`layer0-to-24-drift-trace-receipt.json`](layer0-to-24-drift-trace-receipt.json)).

**Flash-Next remains unqualified (`13` final-logit violations unchanged).**

## 1. Offline constructive `0-ULP` WGSL operator witnesses across Layer 25

The offline analyzer ([`tools/analyze-layer25-prerouter-witness.js`](../../tools/analyze-layer25-prerouter-witness.js),
backed by [`tools/layer25-wgsl-witness-core.js`](../../tools/layer25-wgsl-witness-core.js))
verifies the SHA-256 identity of `manifest.json`, the GPU stage observation
(`f32kv-layer25-stages-20261107.json`), the GPU router observation
(`f32kv-router-subgroup-20261107.json`), the GPU/CPU routing comparison
(`f32kv-router-subgroup-routing-20261107.json`), both CPU reference traces
(`trace.json` and `receipt.json`), all 56 binary tensor files, and all 18
Layer-25 non-expert weight slices before performing any arithmetic.

Every captured Layer-25 GPU stage is reproduced **100% bit-for-bit (`0 ULP`)**
by its constructive F32 WGSL operator witness:

| Operator family | Stages witnessed | Kernel / arithmetic contract | Position `[1717, 2271, 2303]` (`0 ULP`) | Full 2,304-token sequence (`0 ULP`) |
| --- | --- | --- | ---: | ---: |
| Hyper-connection norm | `attn_hc.norm`, `mlp_hc.norm` | `4 x 2560` stream groups, 256-thread tree FMA sum-of-squares, `divideRefined`, `inverseRootRefined`, left-associated `(x * invRms) * (1 + w)` | `61,440 / 61,440` | — |
| Wave64 subgroup matmul | `attn_hc.{down,up,logits}`, `mlp_hc.{down,up,logits}`, `router25.logits` | [`matmul_f32_row_subgroup.wgsl`](../../src/gpu/kernels/matmul_f32_row_subgroup.wgsl): `vec4<f32>` lane FMA at stride `256`, 6-stage wave64 DPP binary tree (`stride = 1, 2, 4, 8, 16, 32`), `((x + y) + z) + w` | `64,920 / 64,920` | `1,179,648 / 1,179,648` (`router25.logits`) |
| Hyper-connection activations & mixing | `attn_hc.{lowRank,mix,branchInput,gate,injected}`, `mlp_hc.{lowRank,mix,branchInput,gate}`, `mlp_hc.input`, `router25.input` | `siluWgsl(divideRefined(x, 4))`, `sigmoidRefined(x)`, `divideRefined(sum_s(norm_s * mix_s), 4)`, `2 * sigmoidRefined(divideRefined(x, 4))`, `stream + branch * gate` | `140,184 / 140,184` | — |
| Linear-attention tiled matmuls (`N >= 64`) | `linear.attn.{qkv_proj,linear_z_proj}`, `attn_hc.branchOutput` (`out_proj`) | [`matmul_f16w_f32a_tiled.wgsl`](../../src/gpu/kernels/matmul_f16w_f32a_tiled.wgsl): 16-element K-tile FMA into `tile = 0.0`, then `acc += tile` | `56,832 / 56,832` | `37,748,736 / 37,748,736` (`qkv_proj` + `z_proj`) |
| Linear-attention sequential matmuls (`N = 48 < 64`) | `linear.attn.{linear_a_proj,linear_b_proj}` | [`matmul_f16w_f32a.wgsl`](../../src/gpu/kernels/matmul_f16w_f32a.wgsl): sequential FMA `0..K-1` | `288 / 288` | `221,184 / 221,184` (`a_proj` + `b_proj`) |
| GatedDeltaNet conv1d + recurrent core | `linear.attn.linear_core_out` | [`gated_delta_conv.wgsl`](../../src/gpu/kernels/gated_delta_conv.wgsl) (4-tap FMA + `siluWgsl`) + [`gated_delta_recurrent_sigmoid_gate.wgsl`](../../src/gpu/kernels/gated_delta_recurrent_sigmoid_gate.wgsl) (`roundTiesToEven` `expRefined`/`logRefined`, `sigmoidRefined` gate) | `18,432 / 18,432` | `14,155,776 / 14,155,776` |
| **Chained end-to-end Layer-25 witness** | `attn_hc.input -> ... -> mlp_hc.branchInput -> router25.logits` | All 21 Layer-25 operators chained from `attn_hc.input` without resetting to intermediate GPU captures | `7,680 / 7,680` (`branchInput`), `1,536 / 1,536` (`logits`) | — |

Across all 27 stage checks at `[1717, 2271, 2303]`, **357,456 of 357,456** values
match at `0 ULP`. Across the full 2,304-token sequence, **14,155,776 of 14,155,776**
`linear_core_out` values and **1,179,648 of 1,179,648** `router25.logits` values
match at `0 ULP`. Seven operator fault controls and one capture-identity negative
control are verified in [layer25-prerouter-witness-results.json](layer25-prerouter-witness-results.json)
and [layer25-prerouter-witness-negative-result.json](layer25-prerouter-witness-negative-result.json).

## 2. Offline JS counterfactuals (`A0..A6`, `R0..R2`) and fresh physical WebGPU execution (`R0_physical_gpu..R2_physical_gpu`)

In [layer25-prerouter-witness-results.json](layer25-prerouter-witness-results.json)
(`newGpuInvocations: 0`), `A0..A6` and `R0..R2` are evaluated using the offline
JavaScript F64 reference and JS WGSL witnesses. To verify the substituted-input
counterfactuals on the actual hardware, [`tools/run-layer25-physical-gpu-counterfactual.js`](../../tools/run-layer25-physical-gpu-counterfactual.js)
dispatches the 9 production WGSL shaders (`hyper_connection_norm.wgsl`,
`hyper_connection_activation.wgsl`, `hyper_connection_branch.wgsl`,
`hyper_connection_inject.wgsl`, `matmul_f32_row_subgroup.wgsl`,
`matmul_f16w_f32a_tiled.wgsl`, `matmul_f16w_f32a.wgsl`, `gated_delta_conv.wgsl`,
`gated_delta_recurrent_sigmoid_gate.wgsl`) on `Jupiter`'s physical AMD RDNA-3
GPU via Chromium WebGPU (`newGpuInvocations: 56`), recorded in
[layer25-physical-gpu-counterfactual-receipt.json](layer25-physical-gpu-counterfactual-receipt.json):

| Counterfactual | Execution surface | Layer-24 `attn_hc.input` | Layer-25 `attn_hc` / `inject` / `mlp_hc` | Layer-25 `linear_attn` | Pos `1717` exact / WGSL margin (`493 - 330`) | Pos `2271` exact / WGSL margin (`311 - 411`) | Match vs JS witness (`0 ULP`) | Top-10 match with CPU (`1717 / 2271 / 2303`) |
| --- | --- | --- | --- | --- | ---: | ---: | :---: | :---: |
| `A0_pure_cpu` | Offline F64 | CPU | CPU (F64) | CPU (F64) | `+1.514e-6` / `+1.907e-6` | `+1.134e-6` / `+9.537e-7` | — | `10/10`, `10/10`, `10/10` |
| `A1_gpu_layer24_stream_with_cpu_linear_branch` | Offline F64 | **GPU** | CPU (F64) | CPU `branchOutput` | **`-4.562e-6`** / **`-4.768e-6`** | `+6.852e-7` / `+4.768e-7` | — | `9/10`, `10/10`, `10/10` |
| `A2_gpu_layer24_stream_with_spliced_linear_attn` | Offline F64 + JS WGSL | **GPU** | CPU (F64) | Spliced `linear_attn` | **`-2.783e-6`** / **`-2.861e-6`** | **`-6.970e-7`** / **`-4.768e-7`** | — | `9/10`, `9/10`, `10/10` |
| `R0_physical_gpu_all_layer25` | **Physical AMD RDNA-3 GPU** | **GPU** | **Physical GPU WGSL** | **Physical GPU WGSL** | **`-2.702e-6`** / **`-2.384e-6`** | **`-8.463e-7`** / **`-4.768e-7`** | `22 / 22` stages (`0 ULP`) | `9/10`, `9/10`, `10/10` |
| `R1_physical_gpu_cpu_stream_and_branch_into_gpu_hc` | **Physical AMD RDNA-3 GPU** | **CPU** | **Physical GPU WGSL** | CPU `branchOutput` | **`+1.465e-6`** / **`+1.431e-6`** | **`+9.970e-7`** / **`+4.768e-7`** | `7,680 / 7,680` & `1,536 / 1,536` (`0 ULP`) | `10/10`, `10/10`, `10/10` |
| `R2_physical_gpu_cpu_layer24_stream_into_all_gpu_layer25` | **Physical AMD RDNA-3 GPU** | **CPU** | **Physical GPU WGSL** | **Physical GPU WGSL** | **`+1.902e-6`** / **`+2.384e-6`** | **`+9.789e-7`** / **`+9.537e-7`** | `7,680 / 7,680` & `1,536 / 1,536` (`0 ULP`) | `10/10`, `10/10`, `10/10` |

## 3. Backward drift trace and implementation-vs-variation separation across Layers `0..25`

[`tools/analyze-layer0-to-24-drift-trace.js`](../../tools/analyze-layer0-to-24-drift-trace.js)
analyzes all `570` GPU prefix stage tensors (`stages.json`), all `26` GPU/CPU
router captures (`L = 0..25`, `2,304` tokens), all `234` HC/router weight
slices across Layers `0..25`, and the CPU 4-stream checkpoints (`C = 0, 3, 4, 8, 12, 16, 20, 24, 25`),
producing [layer0-to-24-drift-trace-receipt.json](layer0-to-24-drift-trace-receipt.json):

### 3.1 Separating implementation compliance from arithmetic variation (Layers `0..25`)

1. **`0-ULP` WGSL specification compliance across all 26 layers:** Every non-PLE
   captured stage, stream-continuity link (`L{l}.attn_hc.input == L{l-1}.mlp_hc.injected`
   for `l = 2..25`), and router projection across `L = 0..25` matches its
   constructive WGSL witness at **`0 ULP` (`569 / 569` tensor checks,
   `8,635,104 / 8,635,104` F32 values)**. The sole `L1.attn_hc.input != L0.mlp_hc.injected`
   transition is Layer 1's per-layer embedding (`model.language_model.layers.1.ple.*`)
   addition.
2. **Same-input single-block F32 vs F64 rounding variation (`~8e-8`) vs multi-layer trajectory drift (`1.07e-6 -> 3.69e-6`):**
   Evaluating the independent F64 reference (`cpuHcForwardF64`, `cpuHcInjectF64`)
   on the **exact same captured GPU inputs** at every layer `L = 0..25` yields a
   local single-block relative RMS difference of `5.9e-8` to `1.18e-7` (`max 1–2 ULP`).
   By contrast, cumulative GPU-vs-CPU `router.input` (`mlp_hc.branchInput`)
   relative RMS grows across depth from `2.81e-6` (full sequence) / `1.07e-6`
   (`pos 1717, 2271` at `L0`) to `3.69e-6` (full sequence) / `1.86e-6` (`pos 1717`)
   and `2.00e-6` (`pos 2271`) at `L25`. Over `95%` of the difference entering
   `router25` is accumulated multi-layer F32 trajectory drift rather than local
   Layer-25 operator rounding.

### 3.2 Why Layer 25 (`pos 1717` and `pos 2271`) is the first routing flip (`59,904` token-layer census)

At Layer 25, the critical expert-pair weight difference norms are
$\|w_{493}^{(25)} - w_{330}^{(25)}\|_2 = 1.1305$ and
$\|w_{311}^{(25)} - w_{411}^{(25)}\|_2 = 1.1904$. An isotropic activation drift
with `diffRms = 2.411e-6` (`pos 1717`) and `2.589e-6` (`pos 2271`) induces an
expected dot-product standard deviation of:

- $\sigma_{\text{dot},1717} = 2.411 \times 10^{-6} \times 1.1305 = 2.725 \times 10^{-6}$ (observed shift: `-4.216e-6`, $z = -1.547\,\sigma$)
- $\sigma_{\text{dot},2271} = 2.589 \times 10^{-6} \times 1.1904 = 3.081 \times 10^{-6}$ (observed shift: `-1.980e-6`, $z = -0.643\,\sigma$)

Across all `26 * 2,304 = 59,904` token-layer routing decisions in `L = 0..25`,
**zero** tokens in `L13..L24` (`27,648` decisions) have a CPU rank-10 vs rank-11
margin $\le 2.0 \times 10^{-6}$. Layer 25 is the first layer after Layer 12 with
any sub-$2 \times 10^{-6}$ (`1–2 ULP`) tie, and its only two such tokens are
**`pos 1717` (`1.907e-6`, `2 ULP`)** and **`pos 2271` (`9.537e-7`, `1 ULP`)**—both
smaller than $1\,\sigma_{\text{dot}}$.

### 3.3 Exact 4-stream checkpoint interval attribution (`C = 0, 3, 4, 8, 12, 16, 20, 24, 25`)

Grafting the 4-stream drift $\Delta\text{streams}_C = \text{streams}_C^{\text{GPU}} - \text{streams}_C^{\text{CPU}}$
onto `cpu_L25.attn_hc.input` isolates where the flip-inducing drift originates:

| Interval / Checkpoint | 4-stream increment RMS | Cumulative prefix `pos 1717` exact margin (`cpuBr` / `gpuBr`) | Cumulative prefix `pos 2271` exact margin (`cpuBr` / `gpuBr`) | Leave-interval-out `pos 1717` (`gpuBr`) | Leave-interval-out `pos 2271` (`gpuBr`) | Leave-interval-out top-10 (`gpuBr`: `1717 / 2271 / 2303`) |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `C = L0` (entry) | `0.000e+0` | `+1.465e-6` / `+3.306e-6` | `+9.970e-7` / `-4.752e-7` | — | — | — |
| `L0..L2` (`C = L3`) | `1.950e-8` | `+1.740e-6` / `+3.603e-6` | `+1.488e-6` / `+2.349e-8` | `-2.961e-6` (flip) | `-1.134e-6` (flip) | `9/10`, `9/10`, `10/10` |
| `L3` (`C = L4`, QSA) | `8.279e-9` | `+1.521e-6` / `+3.330e-6` | `+1.527e-6` / `-1.289e-8` | `-2.463e-6` (flip) | `-9.146e-7` (flip) | `9/10`, `9/10`, `10/10` |
| `L4..L7` (`C = L8`) | `2.509e-8` | `+1.442e-6` / `+3.363e-6` | `+1.682e-6` / `+1.884e-7` | `-2.642e-6` (flip) | `-9.854e-7` (flip) | `9/10`, `9/10`, `10/10` |
| `L8..L11` (`C = L12`) | `4.422e-8` | `+1.524e-6` / `+3.436e-6` | `+2.204e-6` / `+7.425e-7` | `-2.739e-6` (flip) | `-1.379e-6` (flip) | `9/10`, `9/10`, `10/10` |
| `L12..L15` (`C = L16`) | `2.617e-8` | `+8.195e-7` / `+2.716e-6` | `+1.757e-6` / `+2.652e-7` | `-1.925e-6` (flip) | `-3.108e-7` (flip) | `9/10`, `9/10`, `10/10` |
| `L16..L19` (`C = L20`) | `3.436e-8` | `-1.778e-7` / `+1.783e-6` | `+1.406e-6` / `-8.220e-8` | `-1.791e-6` (flip) | `-4.303e-7` (flip) | `9/10`, `9/10`, `10/10` |
| **`L20..L23` (`C = L24`)** | **`1.228e-7`** | **`-2.895e-6`** / **`-1.024e-6`** | **`-3.564e-7`** / **`-1.808e-6`** | **`+4.678e-8` (no flip)** | **`+1.092e-6` (no flip)** | **`10/10`, `10/10`, `10/10`** |
| `L24` (`C = L25`) | `5.666e-8` | `-4.505e-6` / `-2.702e-6` | `+6.227e-7` / `-8.463e-7` | `-1.024e-6` (flip) | `-1.808e-6` (flip) | `9/10`, `9/10`, `10/10` |

- **Cumulative drift through `C = L16` (`L0..L15`) is benign:** Both positions
  maintain positive margins and `10/10` top-10 agreement with CPU under both
  CPU and GPU Layer-25 `branchOutput`.
- **`Interval L20..L23` (`3 linear + L23 QSA full_attention`) is the dominant driver:**
  It has the largest 4-stream drift increment (`rms = 1.228e-7`), pushes `pos 1717`
  down by `-2.72e-6` and `pos 2271` down by `-1.76e-6`, and is the **only** single
  interval whose removal (`drop_L20..L23_gpuBranch`) restores positive margins and
  `10/10` CPU top-10 expert selection at all three positions (`1717`, `2271`, `2303`).
  Within `L20..L23`, `L23` (`QSA full_attention`) produces the largest single-layer
  jump in `mlp_hc.branchInput` relative RMS (`+5.43e-7`, from `2.72e-6` at `L22`
  to `3.27e-6` at `L23`) and swings `pos 2271` `dot(w311 - w411)` from `+6.62e-6`
  at `L22` to `-3.32e-6` at `L23`.
- **Suffix-only confirmation:** Zeroing all drift before `L20` (`suffix L20..L24`
  into GPU Layer 25) still flips both positions (`-1.203e-6`, `-1.142e-6`),
  whereas zeroing all drift before `L24` (`suffix L24` into GPU Layer 25) flips
  neither (`+1.648e-6`, `+6.375e-7`, `10/10` at all positions).

## 4. Complete-network qualification gate status

Operator-level precision candidates have been tested through the complete
48-layer network against the unchanged `quantized-v1` gate in
[`reports/flash-next-qualification-20261107/README.md`](../flash-next-qualification-20261107/README.md)
and [`reports/flash-next-qualification-20261109/README.md`](../flash-next-qualification-20261109/README.md):

- Compensated router reduction (`router_f32_ Utilities`): `13` logit violations, `maxAbs = 0.109003`.
- Source-order `inject_gate` (`exp -> add -> reciprocal`): `13` logit violations, `maxAbs = 0.109030`.
- Source-order `inject_gate` + `stream_mix`: `13` logit violations, `maxAbs = 0.109012`.
- Four-accumulator Q/K dot & materialized softmax: `50,893` / `50,895` logit violations.

Reproducing captured GPU arithmetic or isolating the `L20..L23` trajectory drift
does not change the qualification gate: **the 13 final-logit violations remain
open failures.**

## 5. Reproduction

```sh
node --test tools/numerical-discontinuity-core.test.js \
  tools/dyadic-diagnostic.test.js \
  tools/layer25-wgsl-witness-core.test.js

# On Jupiter (physical AMD RDNA-3 WebGPU counterfactual + Layers 0..24 drift trace):
node tools/run-layer25-physical-gpu-counterfactual.js \
  reports/numerical-discontinuity-20261109/layer25-prerouter-witness-protocol.json \
  /run/media/x/models/rdrr/qwen-3-8-flash-next-text-q4k-ehf16-af32 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/qualify \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-layer25-trace-20261107 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-routing-trace-20261107 \
  src/gpu/kernels \
  reports/numerical-discontinuity-20261109/layer25-physical-gpu-counterfactual-receipt.json

node tools/analyze-layer0-to-24-drift-trace.js \
  reports/numerical-discontinuity-20261109/layer0-to-24-drift-trace-protocol.json \
  /run/media/x/models/rdrr/qwen-3-8-flash-next-text-q4k-ehf16-af32 \
  /home/x/deco/doppler/.cache/flash-next-prefix-e-20261109 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/qualify \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-layer25-trace-20261107 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-routing-trace-20261107 \
  reports/numerical-discontinuity-20261109/layer0-to-24-drift-trace-receipt.json
```

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
physical AMD RDNA-3 WebGPU counterfactual execution (`56` GPU dispatches, `0 ULP`
vs offline witness and retained captures), `569 / 569` `0-ULP` WGSL checks across
Layers `0..25`, and 4-stream interval attribution. Boundary effects: diagnostic
tooling and evidence only; no runtime or qualification change.
