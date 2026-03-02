# TranslateGemma Distill v1 (Practical)

This document defines the supported distillation scope in Doppler for `translategemma`.

## Claim Boundary

Doppler supports a practical two-stage distill workflow:

1. `stage_a`: KD-oriented stage (`objective="kd"`)
2. `stage_b`: triplet-oriented stage (`objective="triplet"`) with mandatory stage_a dependency
3. Distill batches are built from live teacher/student model logits (no synthetic KD/triplet hint fallback)

It does not claim architecture-paper parity or leaderboard SOTA.

## CLI Cookbook

Stage A verify:

```bash
node tools/doppler-cli.js test-model \
  --suite training \
  --surface node \
  --training-schema-version 1 \
  --training-stage stage_a \
  --teacher-model-id translategemma-4b-it-wq4k-ef16-hf16 \
  --student-model-id gemma-3-1b-it-wq4k-ef16-hf16 \
  --distill-dataset-id en-es \
  --distill-dataset-path /abs/path/translate_distill_pairs_en_es_2way.train.jsonl \
  --distill-language-pair en-es \
  --json
```

Stage B verify (requires stage A artifact):

```bash
node tools/doppler-cli.js test-model \
  --suite training \
  --surface node \
  --training-schema-version 1 \
  --training-stage stage_b \
  --teacher-model-id translategemma-4b-it-wq4k-ef16-hf16 \
  --student-model-id gemma-3-1b-it-wq4k-ef16-hf16 \
  --distill-dataset-path /abs/path/translate_distill_pairs_en_es_2way.train.jsonl \
  --stagea-artifact <path/to/distill_stage_a_manifest.json> \
  --stagea-artifact-hash <hash> \
  --json
```

Distill bench helper:

```bash
node tools/run-distill-bench.mjs --surface node --workload tiny --out-dir bench/out/distill --distill-dataset-path /abs/path/translate_distill_pairs_en_es_2way.train.jsonl
node tools/run-distill-bench.mjs --surface node --workload medium --out-dir bench/out/distill --distill-dataset-path /abs/path/translate_distill_pairs_en_es_2way.train.jsonl
```

Node runtime note:

- `--distill-dataset-path` is Node-only right now.
- Optional WebGPU module override:
  `DOPPLER_NODE_WEBGPU_MODULE=../fawn/nursery/webgpu-core`

## Artifact + Provenance

Distill run directories contain:

- `metrics.ndjson`
- `distill_stage_a_manifest.json` or `distill_stage_b_manifest.json`

Manifest fields include:

- `distillContractHash`
- `distillResolvedConfig`
- `runtimeDump`
- `lossSummary`
- `stageADependency` and lineage parent links for `stage_b`

## Training Metrics Contract

Distill objectives use:

- `objective="kd"` with `loss_kd` and `distill_stage="stage_a"`
- `objective="triplet"` with `loss_triplet` and `distill_stage="stage_b"`
