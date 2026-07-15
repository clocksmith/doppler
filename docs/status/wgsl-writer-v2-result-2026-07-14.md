# WGSL writer v2 result (2026-07-14)

## Decision

Doppler now has seed-confirmed development evidence for one narrow complete-
shader capability: writing a 1-D elementwise f32 WGSL compute shader from a
natural-language operation specification and an explicit interface contract.
Selected seed 47 also has exact completion parity between Transformers/PEFT and
Doppler's F16 Qwen runtime.

This is not external promotion, a general WGSL-writer result, or authorization
for a writer product or CLI. The populations are constructed and family-
disjoint, but the externally custodied one-use promotion population remains
unmaterialized.

## Training

All three base-initialized Qwen 3.5 9B rank-32 LoRA runs consumed all 720 rows
exactly once under the frozen recipe.

| Seed | Distinct rows | Final loss | Mean loss | PEFT tree SHA-256 |
|---:|---:|---:|---:|---|
| 11 | 720/720 | 0.000007883 | 0.0167738 | `fc8ff4a3bb25773024f673d4729b242fd70ee78ac9f399b9403cafce3675b986` |
| 29 | 720/720 | 0.000044650 | 0.0181288 | `d40e7f2f85fb8ea88387e7beb6fd1ae2fb83245cafb5e9e303527b81e5fbcb9c` |
| 47 | 720/720 | 0.000027080 | 0.0176386 | `aa8ad92a5e57c04d359fa2a7095f40eda6f85728f4e1ef90f4a9e8e5633d200d` |

Loss and export completion were admission evidence only. They did not select a
seed or establish capability.

The first seed-11 launch under Transformers 4.57.6 failed its Qwen 3.5 version
preflight before training and remains preserved as a blocked, non-capability
attempt. The admitted runs used the isolated Transformers 5.13.1 executor with
PyTorch 2.12.1+rocm7.2; the blocked dependency lineage was not resumed.

## Disjoint semantic gates

Every candidate received one greedy submission per role. Chromium WebGPU
compilation, dispatch, CPU-oracle agreement, bounds canaries, workgroup/shape
variation, input permutations, and the response envelope were blocking.

| Role | Seed 11 | Seed 29 | Seed 47 | Authority |
|---|---:|---:|---:|---|
| Calibration | 16/16 | 14/16 | 15/16 | none |
| Checkpoint selection | 9/16 | 10/16 | 11/16 | selected seed 47 |
| Seed confirmation | 16/16 | 15/16 | 15/16 | confirmation |

All three seeds cleared the frozen 50% per-seed confirmation floor. The mean
confirmation semantic pass rate was 95.83%, above the frozen 75% requirement,
and the selected seed confirmed at 93.75%. All confirmation shaders compiled
and obeyed the response contract.

## Doppler parity

The pre-selection parity policy bound calibration row 0, the exact Qwen source
revision, the Doppler F16 RDRR manifest, and the selected seed-47 export.

- Prompt token IDs: exact, 650/650.
- Adapter completion token IDs and text: exact, 247/247.
- Adapter first-token logit cosine similarity: 0.9996032.
- Adapter top-10 token overlap: 10/10.
- Adapter-delta cosine similarity: 0.9983933.
- Source, tokenizer, RDRR, PEFT weights, Doppler manifest/weights, and adapter
  activation identity checks: all passed.

Base completions diverged after token 68, which is permitted by the frozen base
thresholds; base first-token identity, prompt tokens, top-k overlap, and logit
similarity passed. The selected adapter's full completion was exact.

Canonical lightweight evidence:

- Terminal result JSON: SHA-256
  `8994172a5f725e9eb830573bafa9a03a2cb195922bbab6cd8e21ab2e913bd5fc`;
  internal receipt hash
  `63e39007971d75835bace7608fbe92079b2a696bec25bf4c5844f126714835bc`.
- Seed-selection receipt: SHA-256
  `09df9c7444025ef749718c75afc8fbad877608f88289a6999f3544621f649cee`.
- Seed-confirmation receipt: SHA-256
  `91f2387db1535d6d75705749a659e2288619bba2236ce5d10054ff4249693e11`.
- Selected-adapter parity receipt: SHA-256
  `bb1c9e0463b86c21a11a71fd6cb5642a3f63691f2044f5bb5a76253c4e750cba`.

## Preserved mechanics defect

The first checkpoint-selection receipts were rejected by the frozen ranker.
Failed semantic outputs could produce an infinite relative error; the hashing
layer encoded that value explicitly, while raw JSON serialization silently
stored `null`. The three candidate receipt hashes therefore did not reproduce.

The invalid receipts remain under
`reports/training/wgsl-writer/doppler-wgsl-writer-v2/attempts/checkpoint-selection-nonfinite-json-invalid`.
Canonical non-finite serialization and a regression test were added. The exact
PEFT completion receipt was reused, Chromium semantic verification was rerun,
and every candidate summary was byte-for-byte equivalent as structured data.
No model output was regenerated and the ranking outcome did not change.

## Artifact and claim boundary

The three PEFT adapters and Doppler exports are hash-bound and present on this
machine. The adapter binaries are not stored in Git and have no immutable
external URLs, so they must be preserved before this machine is retired.

The result covers 1-D elementwise f32 shaders under explicit bindings, uniform
layout, dispatch, bounds, and arithmetic requirements. It does not cover
arbitrary algorithms, binding design from prose alone, textures, atomics,
subgroups, multi-pass pipelines, naturally occurring specifications, or broad
WGSL authorship. External promotion requires an independently custodied,
one-use population and the exact Doppler artifact. Productization remains
blocked until that gate passes.
