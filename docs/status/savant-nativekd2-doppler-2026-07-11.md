# Savant NativeKD2 Doppler Receipt - 2026-07-11

## Scope

This receipt covers the Gamma NativeKD2 single-checkpoint EN/ES student at
`checkpoint-000025`, converted to Doppler RDRR, verified on AMD Radeon 8060S
with RADV/Mesa, measured against the manifest baseline, and evaluated on the
128-row bidirectional WMT13 set used by Gamma.

The evidence here is local constructive evidence. It is not a hosted release,
catalog support claim, cross-engine comparison, or demo publication claim.

## Artifact

- Model ID: `translategemma-4b-1b-enes-q4k-ehf16-af32`
- Source checkpoint:
  `gamma/projects/distillation/translation/runs/translategemma4b_es_en_gemma3_1b_savant_nativekd2_balanced_sft010_kd010_lr1e6_steps200_20260710/mixed/checkpoint-000025`
- Conversion config:
  `src/config/conversion/gemma3/translategemma-4b-1b-enes-q4k-ehf16-af32.json`
- Local RDRR size: 1,025,493,248 bytes across 16 shards
- Weight policy: Q4K projections, F16 embeddings/head, F32 compute, F16 KV
- Conversion contract: pass, 340 tensors, layer-pattern checks 8/8
- Integrity: every declared shard SHA-256 matched
- Strict onboarding check: 0 errors, 0 warnings

The deterministic smoke prompts produced:

| Direction | Input | Output |
| --- | --- | --- |
| EN to ES | `The weather is nice today.` | `El clima está agradable hoy.` |
| ES to EN | `El clima está agradable hoy.` | `The weather is nice today.` |

## Winning Runtime Probe

The retained-Q4K and Gemma fusion settings are packaged as the experimental,
model-scoped profile
`profiles/translategemma-1b-savant-q4k-rdna3-throughput-probe`.
The profile preserves the manifest 4x1 sequential decode cadence and enables:

- `retainQ4KMaterialization`
- `useWideTileQ4KPrefill`
- `useSandwichRMSNormPairFusion`
- `usePostFfnNextInputRMSNormPairFusion`
- `useFusedQKVSplitQKNormRoPE`

The profile is scoped to the measured RDNA3/Vulkan lane. It must not be treated
as Apple Metal evidence.

## Fixed-Prompt Paired Result

The packaged profile was measured on both sides of a fresh manifest-baseline
run. Each receipt uses two warmups and three timed runs with identical prompt,
sampling, token budget, model artifact, surface, and host.

| Lane | Decode tok/s | Prefill tok/s | Total run | Decode record ops | Used memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manifest baseline | 29.14 | 156.22 | 3,007.49 ms | 24,768 | 2,144.12 MiB |
| Packaged profile, first | 36.55 | 223.43 | 2,314.49 ms | 15,480 | 2,511.66 MiB |
| Packaged profile, repeat | 36.36 | 224.14 | 2,320.97 ms | 15,480 | 2,511.66 MiB |

Relative to the paired baseline, the two profile receipts show:

- decode throughput: +25.43% and +24.78%
- prefill throughput: +43.02% and +43.48%
- total-run latency: -23.04% and -22.83%
- recorded decode operations: -37.5%
- used memory: +17.14%, or about 367.54 MiB

All three receipts generated the same 70-token translation.

## Constructive WMT13 Quality And Parity

The quality runner keeps one pipeline loaded per lane, applies the resolved
runtime context throughout the lane, unloads between lanes, records per-row
token and timing hashes, and computes SacreBLEU 2.6.0 BLEU and chrF.

| Artifact/lane | BLEU | chrF | EN-ES BLEU | ES-EN BLEU |
| --- | ---: | ---: | ---: | ---: |
| Source BF16 checkpoint | 34.1896 | 59.9388 | 32.5398 | 33.9087 |
| Doppler Q4K baseline | 31.9149 | 58.2124 | 30.6799 | 31.1268 |
| Doppler Q4K optimized | 31.9149 | 58.2124 | 30.6799 | 31.1268 |

Important boundaries:

- Optimized and baseline Doppler outputs match exactly on 128/128 rows.
- Both lanes repeat exact token hashes on 8/8 determinism rows.
- No row stopped at the maximum-token cap.
- Q4K is 2.2748 BLEU and 1.7264 chrF below the source BF16 checkpoint.
  Demo or public evidence must use the measured Q4K score for this artifact,
  not inherit the BF16 checkpoint score.

Across the 128 constructive rows, the optimized lane's medians improve from
31.90 to 43.69 decode tok/s (+36.95%), from 138.28 to 198.77 prefill tok/s
(+43.74%), and from 1,023.94 to 743.19 ms total latency (-27.42%).

The canonical receipt is
`benchmarks/vendors/results/savant-nativekd2/quality-wmt13/savant-nativekd2-wmt13-enes-128_20260711T012704Z.receipt.json`.

## Probe Decisions

| Probe | Decode tok/s | Prefill tok/s | Total run | Decision |
| --- | ---: | ---: | ---: | --- |
| Manifest baseline | 30.08 | 155.15 | 2,900.57 ms | baseline |
| Batch 8 / readback 8 sequential | 26.77 | 151.18 | 3,187.51 ms | reject: over-execution and slower decode |
| Retain Q4K only | 36.46 | 39.74 | 4,119.82 ms | reject: prefill collapse |
| Retain Q4K + wide prefill | 36.32 | 138.29 | 2,565.45 ms | useful intermediate |
| Fusion bundle only | 35.23 | 155.29 | 2,586.05 ms | lower-memory option |
| Retain + wide + fusion bundle | 43.26 | 143.36 | 2,245.78 ms | exploratory peak; package and repeat |
| Full bundle + 4x4 overlapped cadence | 35.98 | 222.74 | 2,344.87 ms | reject: slower decode than manifest cadence |

The exploratory 43.26 tok/s result is retained as a receipt, but the repeated
packaged-profile values are the primary fixed-prompt evidence.

## Demo And Promotion Boundary

The student is not in the active Doppler demo. It is not in
`models/catalog.json`, and the current RDRR artifact exists only in the local
model directory.

The current publish-tier audit is also incomplete in the active demo:

- Tier 1: Qwen 3.5 0.8B is visible; Qwen 3 Embedding 0.6B and Qwen 3
  Reranker 0.6B are explicitly hidden.
- Tier 2: Qwen 3.5 2B and Gemma 4 E2B are visible.
- Translate: the 4B teacher is verified and cataloged but not demo-visible;
  the 1B Savant student is not cataloged.

Before demo/catalog promotion, the remaining gates are:

1. Human review of the exact Q4K translation artifact and its 31.9149 BLEU / 58.2124 chrF evidence.
2. Per-layer numerical comparison for the fusion bundle before any manifest-default promotion.
3. Support-registry entry, hosted artifact identity, and catalog sync.
4. Frozen translate-compare evidence using the real demo-facing model ID.
5. Active browser demo smoke with the promoted student artifact.
