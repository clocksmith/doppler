# Kernel-Path Builder Report

## Summary

- Models indexed: 11
- Registry kernel paths: 22
- Config sources: 12
- Artifact manifest sources: 0
- Exact matches: 4
- Verified proposals: 8
- Unverified proposals: 4
- Skipped: 0

## Proposal Stats

- Proposal records: 12
- New kernel-path ids: 8
- Existing kernel-path ids reused: 4

## Models

### gemma-3-1b-it-f16-af32

- Sources: `src/config/conversion/gemma3/gemma-3-1b-it-f16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:7a886d21`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather.wgsl | main | sha256:777991fb6e4b3b506e4493b47ee998afe541924ddd7c04e1eadf4cb7fd719ef8 | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_small_f16kv.wgsl | main | sha256:7e3acc24b8b45294d18052c759319fe8a202d4ebfc4b653d4b904b245ae7e5c9 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-f16-fused-f32a-online` (0 mismatches)
- `gemma3-q4k-dequant-f32a-small-attn` (0 mismatches)
- `gemma3-f16-fused-f32a-online-streamingprefill` (1 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.

Proposal:

```json
{
  "kind": "existing",
  "selectedKernelPathId": "gemma3-f16-fused-f32a-online",
  "path": {
    "id": "gemma3-f16-fused-f32a-online",
    "name": "Gemma 3 F16 (F32 activations, online)",
    "description": "F16 weights with F32 activations, online decode attention, and explicit gate/up matmul kernel overrides.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_small_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "preLayer": [
      {
        "op": "embed",
        "kernel": "gather.wgsl",
        "entry": "main",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "op": "final_norm",
        "kernel": "rmsnorm.wgsl",
        "entry": "main"
      },
      {
        "op": "lm_head",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "entry": "main_multicol",
        "weights": "lm_head",
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        }
      },
      {
        "op": "lm_head_prefill",
        "kernel": "matmul_f16w_f32a_tiled.wgsl",
        "entry": "main",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "op": "sample",
        "kernel": "sample.wgsl",
        "entry": "sample_single_pass"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma3-f16-fused-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### gemma-3-1b-it-q4k-ehf16-af32

- Sources: `src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:8ac75329`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32a-online` (2 mismatches)
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
- `gemma3-q4k-dequant-f32w-f32a-online` (2 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `lfm2-q4k-dequant-f32a-online` (2 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "gemma3-q4k-dequant-f32a-online",
    "name": "Derived kernel path for gemma-3-1b-it-q4k-ehf16-af32",
    "description": "Generated from gemma-3-1b-it-q4k-ehf16-af32 execution-v1 graph.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma3-q4k-dequant-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### gemma-3-270m-it-q4k-ehf16-af32

- Sources: `src/config/conversion/gemma3/gemma-3-270m-it-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:7a886d21`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather.wgsl | main | sha256:777991fb6e4b3b506e4493b47ee998afe541924ddd7c04e1eadf4cb7fd719ef8 | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_small_f16kv.wgsl | main | sha256:7e3acc24b8b45294d18052c759319fe8a202d4ebfc4b653d4b904b245ae7e5c9 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32a-small-attn` (0 mismatches)
- `gemma3-f16-fused-f32a-online` (0 mismatches)
- `gemma3-f16-fused-f32a-online-streamingprefill` (1 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.

Proposal:

```json
{
  "kind": "existing",
  "selectedKernelPathId": "gemma3-q4k-dequant-f32a-small-attn",
  "path": {
    "id": "gemma3-q4k-dequant-f32a-small-attn",
    "name": "Gemma 3 Q4K Dequant (F32 activations, small-attn prefill)",
    "description": "Q4K dequantized to F16 with F32 activations. Same as gemma3-q4k-dequant-f32a-online but uses attention_small_f16kv.wgsl for prefill (diagnostic variant).",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_small_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "preLayer": [
      {
        "op": "embed",
        "kernel": "gather.wgsl",
        "entry": "main",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "op": "final_norm",
        "kernel": "rmsnorm.wgsl",
        "entry": "main"
      },
      {
        "op": "lm_head",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "entry": "main_multicol",
        "weights": "lm_head",
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        }
      },
      {
        "op": "lm_head_prefill",
        "kernel": "matmul_f16w_f32a_tiled.wgsl",
        "entry": "main",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "op": "sample",
        "kernel": "sample.wgsl",
        "entry": "sample_single_pass"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma3-q4k-dequant-f32a-small-attn",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### gemma-4-moe-q4k-ehf16-af32

- Sources: `src/config/conversion/gemma4/gemma-4-moe-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:a767e5ec`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | ffn | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | ffn | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32w-f32a-online` (2 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `gemma3-q4k-dequant-f32a` (3 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
- `gemma3-q4k-dequant-f32a-nosubgroups` (3 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "gemma4-q4k-dequant-f32a-online",
    "name": "Derived kernel path for gemma-4-moe-q4k-ehf16-af32",
    "description": "Generated from gemma-4-moe-q4k-ehf16-af32 execution-v1 graph.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "ffn"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "ffn"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma4-q4k-dequant-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### google-embeddinggemma-300m-q4k-ehf16-af32

- Sources: `src/config/conversion/embeddinggemma/google-embeddinggemma-300m-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f32`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:859cce56`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather.wgsl | main | sha256:777991fb6e4b3b506e4493b47ee998afe541924ddd7c04e1eadf4cb7fd719ef8 | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_small.wgsl | main | sha256:6752ddd7ab53e6235c9b5b1a9515141c0d111df7fac9f4c0d7a38f9943490ed4 | - | all |
| decode | decode | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_small.wgsl | main | sha256:6752ddd7ab53e6235c9b5b1a9515141c0d111df7fac9f4c0d7a38f9943490ed4 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |

Closest matches:
- `embeddinggemma-q4k-dequant-f32a` (0 mismatches)
- `embeddinggemma-f16-f32a` (0 mismatches)
- `embeddinggemma-f32-f32a` (2 mismatches)
  - decode steps differ. Hint: Update the decode steps to match the resolved execution graph step-for-step.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "existing",
  "selectedKernelPathId": "embeddinggemma-q4k-dequant-f32a",
  "path": {
    "id": "embeddinggemma-q4k-dequant-f32a",
    "name": "EmbeddingGemma Q4K Dequant (F32 activations)",
    "description": "Q4K weights dequantized to F16 with F32 activations for embedding stability.",
    "activationDtype": "f32",
    "decode": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_small.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_small.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_f16w_f32a.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "preLayer": [
      {
        "op": "embed",
        "kernel": "gather.wgsl",
        "entry": "main",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "op": "final_norm",
        "kernel": "rmsnorm.wgsl",
        "entry": "main"
      }
    ],
    "sampling": []
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "embeddinggemma-q4k-dequant-f32a",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### gpt-oss-20b-f16-xmxfp4

- Sources: `src/config/conversion/gpt-oss-20b-f16-xmxfp4.json`
- Lowering: `inline-kernel-path-error`
- Session: activation=`f16`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:8020bad4`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | ffn | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | ffn | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32w-f32a-online` (3 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `gemma2-f16-f16a` (4 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
  - sampling steps differ. Hint: Update the sampling steps to match the resolved execution graph step-for-step.
- `gemma2-q4k-dequant-f16a` (4 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
  - sampling steps differ. Hint: Update the sampling steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "conversion-f16-f16a-online",
    "name": "Derived kernel path for gpt-oss-20b-f16-xmxfp4",
    "description": "Generated from gpt-oss-20b-f16-xmxfp4 execution-v1 graph.",
    "activationDtype": "f16",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "ffn"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "ffn"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": false,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": false
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": false
      }
    ],
    "errors": [
      "[Execution] Inline kernelPath attention kernel \"attention_decode_online_f16kv.wgsl\" requires activationDtype=\"f32\" and kvcache.kvDtype=\"f16\", but resolved activationDtype=\"f16\" and kvcache.kvDtype=\"f16\".",
      "[ExecutionPlan] Missing finiteness fallback kernel path mapping for \"conversion-f16-f16a-online\". Add an explicit rule in src/rules/inference/kernel-path.rules.json."
    ],
    "roundTripShapeMatches": true,
    "compiledPlan": null
  }
}
```

### janus-pro-1b-text-q4k-ehaf16

- Sources: `src/config/conversion/janus/janus-pro-1b-text-q4k-ehaf16.json`
- Lowering: `inline-kernel-path-error`
- Session: activation=`f16`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:1c198d79`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32a-online` (3 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
- `gemma3-q4k-dequant-f32w-f32a-online` (3 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `lfm2-q4k-dequant-f32a-online` (3 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "janus-q4k-dequant-f16a-online",
    "name": "Derived kernel path for janus-pro-1b-text-q4k-ehaf16",
    "description": "Generated from janus-pro-1b-text-q4k-ehaf16 execution-v1 graph.",
    "activationDtype": "f16",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": false,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": false
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": false
      }
    ],
    "errors": [
      "[Execution] Inline kernelPath attention kernel \"attention_decode_online_f16kv.wgsl\" requires activationDtype=\"f32\" and kvcache.kvDtype=\"f16\", but resolved activationDtype=\"f16\" and kvcache.kvDtype=\"f16\".",
      "[ExecutionPlan] Missing finiteness fallback kernel path mapping for \"janus-q4k-dequant-f16a-online\". Add an explicit rule in src/rules/inference/kernel-path.rules.json."
    ],
    "roundTripShapeMatches": true,
    "compiledPlan": null
  }
}
```

### lfm2.5-1.2b-instruct-q4k-ehf16-af32

- Sources: `src/config/conversion/lfm2/lfm2.5-1.2b-instruct-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:6c1e5c5d`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_small_f16kv.wgsl | main | sha256:7e3acc24b8b45294d18052c759319fe8a202d4ebfc4b653d4b904b245ae7e5c9 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a_tiled.wgsl | main | sha256:e94ae5374e8b43dd48b663eff59a45c822c3784d5702a1145266b6ffd15ba78c | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `lfm2-q4k-dequant-f32a-online` (0 mismatches)
- `gemma3-q4k-dequant-f32a-online` (1 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
- `gemma3-q4k-dequant-f32a-small-attn` (1 mismatches)
  - preLayer steps differ. Hint: Update the preLayer steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "existing",
  "selectedKernelPathId": "lfm2-q4k-dequant-f32a-online",
  "path": {
    "id": "lfm2-q4k-dequant-f32a-online",
    "name": "LFM2 Q4K Dequant (F32 activations, fast prefill)",
    "description": "LFM2-tuned Q4K path: F32 activations with tiled prefill matmul and small-kernel prefill attention.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "entry": "main_vec4",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "op": "input_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "q_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "op": "k_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "op": "v_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "op": "rope_q",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "rope_k",
          "kernel": "rope.wgsl",
          "entry": "main"
        },
        {
          "op": "attention",
          "kernel": "attention_small_f16kv.wgsl",
          "entry": "main"
        },
        {
          "op": "o_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "op": "attn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        },
        {
          "op": "post_attn_norm",
          "kernel": "rmsnorm.wgsl",
          "entry": "main"
        },
        {
          "op": "gate_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "op": "up_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "op": "activation",
          "kernel": "gelu.wgsl",
          "entry": "main",
          "constants": {
            "HAS_GATE": true
          }
        },
        {
          "op": "down_proj",
          "kernel": "matmul_f16w_f32a_tiled.wgsl",
          "entry": "main",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "op": "ffn_residual",
          "kernel": "residual.wgsl",
          "entry": "main"
        }
      ]
    },
    "preLayer": [
      {
        "op": "embed",
        "kernel": "gather_f16.wgsl",
        "entry": "main",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "op": "final_norm",
        "kernel": "rmsnorm.wgsl",
        "entry": "main"
      },
      {
        "op": "lm_head",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "entry": "main_multicol",
        "weights": "lm_head",
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        }
      },
      {
        "op": "lm_head_prefill",
        "kernel": "matmul_f16w_f32a_tiled.wgsl",
        "entry": "main",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "op": "sample",
        "kernel": "sample.wgsl",
        "entry": "sample_single_pass"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "lfm2-q4k-dequant-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### qwen-3-5-0-8b-q4k-ehaf16

- Sources: `src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json`
- Lowering: `execution-graph-only`
- Session: activation=`f16`, kv=`f16`, layout=`--`, batch=`4`, readback=`1`
- Execution signature: `execv1:386c4fe0`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `qwen3-q4k-dequant-f32a-online` (4 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.
- `gemma3-q4k-dequant-f32a-online` (4 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.
- `gemma3-q4k-dequant-f32w-f32a-online` (4 mismatches)
  - Activation dtype differs. Hint: Set activationDtype to "f16" or choose kernels compatible with "f32".
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "qwen3-q4k-dequant-f16a-online",
    "name": "Derived kernel path for qwen-3-5-0-8b-q4k-ehaf16",
    "description": "Generated from qwen-3-5-0-8b-q4k-ehaf16 execution-v1 graph.",
    "activationDtype": "f16",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": false,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": false
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [
      "[Execution] Inline kernelPath attention kernel \"attention_decode_online_f16kv.wgsl\" requires activationDtype=\"f32\" and kvcache.kvDtype=\"f16\", but resolved activationDtype=\"f16\" and kvcache.kvDtype=\"f16\"."
    ],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "qwen3-q4k-dequant-f16a-online",
      "primaryActivationDtype": "f16",
      "fallbackPlanId": "finiteness_fallback",
      "fallbackKernelPathId": "qwen3-q4k-dequant-f32a-online"
    }
  }
}
```

### qwen-3-5-2b-q4k-ehaf16

- Sources: `src/config/conversion/qwen3/qwen-3-5-2b-q4k-ehaf16.json`
- Lowering: `execution-graph-only`
- Session: activation=`f16`, kv=`f16`, layout=`--`, batch=`4`, readback=`1`
- Execution signature: `execv1:2620b8b0`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:257973dd762541d8a2c4b16ad8ad1d071199199e81e477ab644dce915961a989 | embed_tokens | all |
| decode | decode | input_norm | rmsnorm_f16.wgsl | main | sha256:ce2f200529dc5a3b5985922bad7d7d1fb96785f15ef3e4e235321053e9fe9e59 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope_f16.wgsl | main | sha256:c0f3d933864fec40a424de79adac52bec54402351680fe419df1bf58235ab683 | - | all |
| decode | decode | rope_k | rope_f16.wgsl | main | sha256:c0f3d933864fec40a424de79adac52bec54402351680fe419df1bf58235ab683 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual_f16.wgsl | main | sha256:6dbc8ef2a8ac57117bd41aabc6979ddd7fe2f363b80b761ba2b8f7826d7e02b9 | - | all |
| decode | decode | post_attn_norm | rmsnorm_f16.wgsl | main | sha256:ce2f200529dc5a3b5985922bad7d7d1fb96785f15ef3e4e235321053e9fe9e59 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu_f16.wgsl | main | sha256:aa10873ced2bf9b3e80f39c22d68745f812dbfc96b2afe04f52f9d0589cfac46 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup_f16a.wgsl | main_vec4 | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual_f16.wgsl | main | sha256:6dbc8ef2a8ac57117bd41aabc6979ddd7fe2f363b80b761ba2b8f7826d7e02b9 | - | all |
| prefill | prefill | input_norm | rmsnorm_f16.wgsl | main | sha256:ce2f200529dc5a3b5985922bad7d7d1fb96785f15ef3e4e235321053e9fe9e59 | - | all |
| prefill | prefill | q_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope_f16.wgsl | main | sha256:c0f3d933864fec40a424de79adac52bec54402351680fe419df1bf58235ab683 | - | all |
| prefill | prefill | rope_k | rope_f16.wgsl | main | sha256:c0f3d933864fec40a424de79adac52bec54402351680fe419df1bf58235ab683 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual_f16.wgsl | main | sha256:6dbc8ef2a8ac57117bd41aabc6979ddd7fe2f363b80b761ba2b8f7826d7e02b9 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm_f16.wgsl | main | sha256:ce2f200529dc5a3b5985922bad7d7d1fb96785f15ef3e4e235321053e9fe9e59 | - | all |
| prefill | prefill | gate_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu_f16.wgsl | main | sha256:aa10873ced2bf9b3e80f39c22d68745f812dbfc96b2afe04f52f9d0589cfac46 | - | all |
| prefill | prefill | down_proj | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual_f16.wgsl | main | sha256:6dbc8ef2a8ac57117bd41aabc6979ddd7fe2f363b80b761ba2b8f7826d7e02b9 | - | all |
| postLayer | both | final_norm | rmsnorm_f16.wgsl | main | sha256:ce2f200529dc5a3b5985922bad7d7d1fb96785f15ef3e4e235321053e9fe9e59 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup_f16a.wgsl | main_multicol | sha256:d0a981cdfeb21a40da0c0fe82a0f2f08b99f0bd2d0354d4bd8ca432855fd79e5 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16.wgsl | main | sha256:7e9f23f326b5750fc7783a1959334c237aff4a23da6f87176608c5c4eb4fc278 | lm_head | all |
| postLayer | both | sample | sample_f16.wgsl | sample_single_pass | sha256:fde7ed164f7a0ce7f1adb0366cdf9b798d04f0608110430a632e9a1539c63b9b | - | all |

Closest matches:
- `qwen3-q4k-dequant-f16a-online` (1 mismatches)
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.
- `gemma3-q4k-dequant-f16a-online` (3 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.
- `gemma3-f16-fused-f16a-online` (4 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
  - Custom runtime layers bypass raw kernel-path lowering. Hint: Keep the proposal partial and preserve the custom runtime facts for the bypassed layers instead of forcing a registry path to claim ownership of them.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "qwen3-q4k-dequant-f16a-online",
    "name": "Derived kernel path for qwen-3-5-2b-q4k-ehaf16",
    "description": "Generated from qwen-3-5-2b-q4k-ehaf16 execution-v1 graph.",
    "activationDtype": "f16",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm_f16.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope_f16.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope_f16.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual_f16.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm_f16.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu_f16.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup_f16a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual_f16.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm_f16.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope_f16.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope_f16.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual_f16.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm_f16.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu_f16.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual_f16.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm_f16.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup_f16a.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample_f16.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": false,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": false
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [
      "[Execution] Inline kernelPath attention kernel \"attention_decode_online_f16kv.wgsl\" requires activationDtype=\"f32\" and kvcache.kvDtype=\"f16\", but resolved activationDtype=\"f16\" and kvcache.kvDtype=\"f16\"."
    ],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "qwen3-q4k-dequant-f16a-online",
      "primaryActivationDtype": "f16",
      "fallbackPlanId": "finiteness_fallback",
      "fallbackKernelPathId": "qwen3-q4k-dequant-f32a-online"
    }
  }
}
```

### translategemma-4b-1b-enes-q4k-ehf16-af32

- Sources: `src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:8ac75329`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32a-online` (2 mismatches)
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
- `gemma3-q4k-dequant-f32w-f32a-online` (2 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `lfm2-q4k-dequant-f32a-online` (2 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "gemma3-q4k-dequant-f32a-online",
    "name": "Derived kernel path for translategemma-4b-1b-enes-q4k-ehf16-af32",
    "description": "Generated from translategemma-4b-1b-enes-q4k-ehf16-af32 execution-v1 graph.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma3-q4k-dequant-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```

### translategemma-4b-it-q4k-ehf16-af32

- Sources: `src/config/conversion/gemma3/translategemma-4b-it-q4k-ehf16-af32.json`
- Lowering: `inline-kernel-path`
- Session: activation=`f32`, kv=`f16`, layout=`--`, batch=`--`, readback=`--`
- Execution signature: `execv1:8ac75329`

| Section | Phase | Op | Kernel | Entry | Digest | Weights | Layers |
| --- | --- | --- | --- | --- | --- | --- | --- |
| preLayer | both | embed | gather_f16.wgsl | main | sha256:a4829f4067091c98ad6ebbc9b0744cdd5bbcd4fbf6092b2f7cc7f1098695860f | embed_tokens | all |
| decode | decode | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | q_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.q_proj | all |
| decode | decode | k_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.k_proj | all |
| decode | decode | v_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.v_proj | all |
| decode | decode | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| decode | decode | attention | attention_decode_online_f16kv.wgsl | main | sha256:4c5d8c92a0a111af716d6b46b9559446807c086027445c7fefa150202f43dae4 | - | all |
| decode | decode | o_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.self_attn.o_proj | all |
| decode | decode | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| decode | decode | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| decode | decode | gate_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.gate_proj | all |
| decode | decode | up_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.up_proj | all |
| decode | decode | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| decode | decode | down_proj | matmul_gemv_subgroup.wgsl | main_vec4 | sha256:3cee3bed453b40c5564a751d2a917649e10ad52f5268e77cbfecfcee34780457 | layer.{L}.mlp.down_proj | all |
| decode | decode | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | input_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | q_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.q_proj | all |
| prefill | prefill | k_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.k_proj | all |
| prefill | prefill | v_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.v_proj | all |
| prefill | prefill | rope_q | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | rope_k | rope.wgsl | main | sha256:4c803ad5e0dd065d5572c7aecc1def277c43884dcc02f22a9676914c10111400 | - | all |
| prefill | prefill | attention | attention_streaming_f16kv.wgsl | main | sha256:b337a2dcf7b40431a733d1726eee8bf23504136fd5f915bec057fb59f3ea7480 | - | all |
| prefill | prefill | o_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.self_attn.o_proj | all |
| prefill | prefill | attn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| prefill | prefill | post_attn_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| prefill | prefill | gate_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.gate_proj | all |
| prefill | prefill | up_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.up_proj | all |
| prefill | prefill | activation | gelu.wgsl | main | sha256:a9007ea08aaff98f9be08f1e0490a6bcf252883eac5513de876ab9ce918865e6 | - | all |
| prefill | prefill | down_proj | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | layer.{L}.mlp.down_proj | all |
| prefill | prefill | ffn_residual | residual.wgsl | main | sha256:f1abd88c959c5d8dd27b9353d487e37b2a96850ed9d90c365212e260399cc2a7 | - | all |
| postLayer | both | final_norm | rmsnorm.wgsl | main | sha256:f516b3e4bde2015f2a207c3ca5b8c9820c7809fa8f8d0786f90c568e0f1ac077 | - | all |
| postLayer | both | lm_head | matmul_gemv_subgroup.wgsl | main_multicol | sha256:96c38c15e6fed0d7efdc5cd094db5843a8e8ddfe01eee3bc7322fa555dacf3d0 | lm_head | all |
| postLayer | both | lm_head_prefill | matmul_f16w_f32a.wgsl | main | sha256:027a8f1cd9713cbe0b0ada160bd175e0542bb90896ad85441b023522d9a1befc | lm_head | all |
| postLayer | both | sample | sample.wgsl | sample_single_pass | sha256:4412357e84113ee2f1bc0dc8bf89e314c2ab482c89c14ca016ea9949d16a9d0c | - | all |

Closest matches:
- `gemma3-q4k-dequant-f32a-online` (2 mismatches)
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.
- `gemma3-q4k-dequant-f32w-f32a-online` (2 mismatches)
  - decode kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - prefill steps differ. Hint: Update the prefill steps to match the resolved execution graph step-for-step.
- `lfm2-q4k-dequant-f32a-online` (2 mismatches)
  - prefill kernels assume different device capabilities. Hint: Choose a registry path whose subgroup and attention-kernel assumptions match the execution graph, or synthesize a new path id for this capability mix.
  - postLayer steps differ. Hint: Update the postLayer steps to match the resolved execution graph step-for-step.

Proposal:

```json
{
  "kind": "proposed",
  "selectedKernelPathId": null,
  "path": {
    "id": "gemma3-q4k-dequant-f32a-online",
    "name": "Derived kernel path for translategemma-4b-it-q4k-ehf16-af32",
    "description": "Generated from translategemma-4b-it-q4k-ehf16-af32 execution-v1 graph.",
    "activationDtype": "f32",
    "kvDtype": "f16",
    "decode": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_decode_online_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main_vec4",
          "kernel": "matmul_gemv_subgroup.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "prefill": {
      "steps": [
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "input_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "q_proj",
          "weights": "layer.{L}.self_attn.q_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "k_proj",
          "weights": "layer.{L}.self_attn.k_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "v_proj",
          "weights": "layer.{L}.self_attn.v_proj"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_q"
        },
        {
          "entry": "main",
          "kernel": "rope.wgsl",
          "op": "rope_k"
        },
        {
          "entry": "main",
          "kernel": "attention_streaming_f16kv.wgsl",
          "op": "attention"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "o_proj",
          "weights": "layer.{L}.self_attn.o_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "attn_residual"
        },
        {
          "entry": "main",
          "kernel": "rmsnorm.wgsl",
          "op": "post_attn_norm"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "gate_proj",
          "weights": "layer.{L}.mlp.gate_proj"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "up_proj",
          "weights": "layer.{L}.mlp.up_proj"
        },
        {
          "constants": {
            "HAS_GATE": true
          },
          "entry": "main",
          "kernel": "gelu.wgsl",
          "op": "activation"
        },
        {
          "entry": "main",
          "kernel": "matmul_f16w_f32a.wgsl",
          "op": "down_proj",
          "weights": "layer.{L}.mlp.down_proj"
        },
        {
          "entry": "main",
          "kernel": "residual.wgsl",
          "op": "ffn_residual"
        }
      ]
    },
    "preLayer": [
      {
        "entry": "main",
        "kernel": "gather_f16.wgsl",
        "op": "embed",
        "weights": "embed_tokens"
      }
    ],
    "postLayer": [
      {
        "entry": "main",
        "kernel": "rmsnorm.wgsl",
        "op": "final_norm"
      },
      {
        "constants": {
          "MULTICOL_COLS_PER_WG": 64,
          "MULTICOL_THREADS_PER_COL": 4
        },
        "entry": "main_multicol",
        "kernel": "matmul_gemv_subgroup.wgsl",
        "op": "lm_head",
        "weights": "lm_head"
      },
      {
        "entry": "main",
        "kernel": "matmul_f16w_f32a.wgsl",
        "op": "lm_head_prefill",
        "weights": "lm_head"
      }
    ],
    "sampling": [
      {
        "entry": "sample_single_pass",
        "kernel": "sample.wgsl",
        "op": "sample"
      }
    ]
  },
  "verification": {
    "ok": true,
    "checks": [
      {
        "id": "kernelPathContract",
        "ok": true
      },
      {
        "id": "sessionCompatibility",
        "ok": true
      },
      {
        "id": "roundTripShape",
        "ok": true
      },
      {
        "id": "executionPlanCompile",
        "ok": true
      }
    ],
    "errors": [],
    "roundTripShapeMatches": true,
    "compiledPlan": {
      "primaryPlanId": "primary",
      "primaryKernelPathId": "gemma3-q4k-dequant-f32a-online",
      "primaryActivationDtype": "f32",
      "fallbackPlanId": null,
      "fallbackKernelPathId": null
    }
  }
}
```
