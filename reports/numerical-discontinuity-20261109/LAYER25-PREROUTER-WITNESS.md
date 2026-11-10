# Layer-25 pre-router WGSL witness and bidirectional counterfactual attribution

This extends the frozen diagnostics in `08c12ad6` and `8115ff46` without
modifying any of their files. It witnesses the complete Flash-Next Layer-25
pre-router and router execution path (`attn_hc -> linear_attn -> inject -> mlp_hc -> router`)
against the frozen F32-KV stage and router captures on `Jupiter`.
**Flash-Next remains unqualified (`13` final-logit violations unchanged).**

## 1. Constructive `0-ULP` WGSL operator witnesses across all 27 Layer-25 stages

The analyzer ([`tools/analyze-layer25-prerouter-witness.js`](../../tools/analyze-layer25-prerouter-witness.js),
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
match at `0 ULP`.

## 2. Fault and identity negative controls

Seven operator fault controls and one capture-identity negative control are
verified in [layer25-prerouter-witness-results.json](layer25-prerouter-witness-results.json)
and [layer25-prerouter-witness-negative-result.json](layer25-prerouter-witness-negative-result.json):

1. `corrupted_output_1ulp`: flipping 1 ULP on one element per row of `attn_hc.norm` is rejected on `3 / 3` rows.
2. `norm_missing_weight_offset`: scaling by `w` instead of `(1 + w)` in `attn_hc.norm` is rejected on `3 / 3` rows.
3. `norm_right_associated`: right-associated `x * (invRms * (1 + w))` in `attn_hc.norm` is rejected on `3 / 3` rows.
4. `subgroup_top_down_reduction`: top-down (`32, 16, 8, 4, 2, 1`) instead of bottom-up (`1, 2, 4, 8, 16, 32`) wave64 reduction in `attn_hc.down` is rejected on `3 / 3` rows.
5. `matmul_seq_instead_of_tiled16`: sequential `0..K-1` FMA instead of 16-element K-tile FMA in `linear.attn.qkv_proj` is rejected on `3 / 3` rows.
6. `conv1d_unfused_mul_add`: unfused `acc + (x * w)` instead of `fma(x, w, acc)` in `gated_delta_conv` is rejected on `3 / 3` rows.
7. `gdn_wrong_silu_gate`: `silu` instead of `sigmoid` norm-gate activation in `gated_delta_recurrent` is rejected on `3 / 3` rows.
8. `Incorrect stagesObservation SHA-256`: corrupting an expected SHA-256 in the protocol exits with code `1` before creating an output file.

## 3. Bidirectional counterfactual attribution (`A0..A6` and `R0..R2`)

Both exact integer dyadic dot products and the physical `matmul_f32_row_subgroup.wgsl`
wave64 router witness are evaluated across forward (`GPU -> CPU`) and reverse
(`CPU -> GPU`) counterfactuals at positions `1717` (`493` vs `330`), `2271`
(`311` vs `411`), and `2303` (control):

| Counterfactual | Layer-24 `attn_hc.input` | Layer-25 `attn_hc` / `inject` / `mlp_hc` | Layer-25 `linear_attn` | Pos `1717` exact margin (`493 - 330`) | Pos `2271` exact margin (`311 - 411`) | Top-10 match with CPU (`1717 / 2271 / 2303`) |
| --- | --- | --- | --- | ---: | ---: | :---: |
| `A0_pure_cpu` | CPU | CPU (F64) | CPU (F64) | `+1.514252e-6` | `+1.133674e-6` | `10/10`, `10/10`, `10/10` |
| `A1_gpu_layer24_stream_with_cpu_linear_branch` | **GPU** | CPU (F64) | CPU `branchOutput` | **`-4.561852e-6` (flip)** | `+6.852351e-7` | `9/10`, `10/10`, `10/10` |
| `A2_gpu_layer24_stream_with_spliced_linear_attn` | **GPU** | CPU (F64) | **Spliced `branchInput` -> `linear_attn`** | **`-2.782866e-6` (flip)** | **`-6.969921e-7` (flip)** | `9/10`, `9/10`, `10/10` |
| `A5_gpu_attn_hc_and_linear_attn` | **GPU** | `attn_hc`: GPU; `inject` + `mlp_hc`: CPU (F64) | **GPU** | **`-2.705240e-6` (flip)** | **`-7.823672e-7` (flip)** | `9/10`, `9/10`, `10/10` |
| `A6_all_gpu_layer25` (`= R0`) | **GPU** | **GPU WGSL** | **GPU WGSL** | **`-2.701794e-6` (flip)** | **`-8.463161e-7` (flip)** | `9/10`, `9/10`, `10/10` |
| `R1_cpu_stream_and_branch_into_gpu_wgsl_hc` | **CPU** | **GPU WGSL** | CPU `branchOutput` | **`+1.465292e-6` (no flip)** | **`+9.970442e-7` (no flip)** | `10/10`, `10/10`, `10/10` |
| `R2_cpu_layer24_stream_into_all_gpu_wgsl_layer25` | **CPU** | **GPU WGSL** | **GPU WGSL** | **`+1.901814e-6` (no flip)** | **`+9.788917e-7` (no flip)** | `10/10`, `10/10`, `10/10` |

### Attribution conclusions

1. **No Layer-25 operator bug:** Every Layer-25 operator from `attn_hc.input` through `router25.logits` executes its declared WGSL kernel at `0 ULP`.
2. **Forward sufficiency (`A1`, `A2`):** Substituting the incoming Layer-24 stream state (`gpu_attn_hc.input`) at `[1717, 2271, 2303]` while keeping `attn_hc`, `inject`, and `mlp_hc` in F64 CPU reference arithmetic already flips `1717` (`493 -> 330`) via the residual/gate path alone (`A1`), and flips **both** `1717` (`493 -> 330`) and `2271` (`311 -> 411`) once the resulting `attn_hc.branchInput` is propagated through `linear_attn` (`A2`).
3. **Reverse necessity (`R1`, `R2`):** Feeding the CPU Layer-24 state (`cpu_attn_hc.input`) into **100% physical GPU WGSL Layer-25 operators** (`attn_hc -> linear_attn -> inject -> mlp_hc -> router25`) eliminates both flips and preserves `10/10` top-10 expert selection at `1717`, `2271`, and `2303` under both exact dyadic dots and wave64 subgroup router logits.
4. **Scope boundary:** This proves that Layer-25's operators are bit-exact to their WGSL specification and that incoming Layer-24 state drift (`rmsDiff = 1.581e-7` at `1717`, `1.308e-7` at `2271`) is both necessary and sufficient for the two Layer-25 routing flips. It does **not** prove that the cumulative drift across Layers `0..24` is permitted under a model qualification contract.

## 4. Reproduction

```sh
node --test tools/numerical-discontinuity-core.test.js \
  tools/dyadic-diagnostic.test.js \
  tools/layer25-wgsl-witness-core.test.js

# On Jupiter (with frozen captures under /run/media/x/models/):
node tools/analyze-layer25-prerouter-witness.js \
  reports/numerical-discontinuity-20261109/layer25-prerouter-witness-protocol.json \
  /run/media/x/models/rdrr/qwen-3-8-flash-next-text-q4k-ehf16-af32 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/qualify \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-layer25-trace-20261107 \
  /run/media/x/models/qwen38-flash-next-de4b8e4d/reference/f32kv-routing-trace-20261107 \
  /tmp/layer25-prerouter-witness-results.json
```

Component: `doppler.repository-tooling`. Intent: preserved. Acceptance evidence:
`0-ULP` constructive WGSL operator witnesses across all 27 Layer-25 stages, 7
operator fault controls, identity negative control, and bidirectional router
counterfactuals (`10/10` unit tests passing). Boundary effects: read-only
analysis of frozen Flash-Next captures; no runtime or qualification change.
