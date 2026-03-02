# UL v1 (Practical) in Doppler

This document defines the supported UL v1 scope in Doppler.

## Claim Boundary

Doppler currently supports a **UL-inspired practical two-stage pipeline**.

It does **not** claim paper-equivalent Unified Latents implementation parity or
paper-scale SOTA results (FID/FVD/PSNR).

## Supported Stages

1. `stage1_joint`
- Noisy latent batch preparation with deterministic seed handling.
- Stage1 objective term emission and machine-parseable metrics.
- Manifest + NDJSON artifact generation.

2. `stage2_base`
- Stage1 artifact dependency checks (hash + contract linkage + stage type).
- Freeze-map-aware optimizer targeting.
- Stage2 manifest with lineage linkage to stage1.

## CLI Cookbook

Stage1 verify:

```bash
node tools/doppler-cli.js test-model \
  --suite training \
  --surface node \
  --training-schema-version 1 \
  --training-stage stage1_joint \
  --json
```

Stage2 verify (using an existing stage1 artifact):

```bash
node tools/doppler-cli.js test-model \
  --suite training \
  --surface node \
  --training-schema-version 1 \
  --training-stage stage2_base \
  --stage1-artifact <path/to/ul_stage1_manifest.json> \
  --stage1-artifact-hash <hash> \
  --json
```

Training-calibrate bench path:

```bash
node tools/doppler-cli.js bench \
  --workload-type training \
  --surface node \
  --training-schema-version 1 \
  --training-bench-steps 2 \
  --training-stage stage1_joint \
  --json
```

Helper scripts:

```bash
node tools/run-ul-bench.mjs --surface node --out-dir bench/out/ul
node tools/run-ul-bench.mjs --surface node --workload tiny --out-dir bench/out/ul
node tools/run-ul-bench.mjs --surface node --workload medium --out-dir bench/out/ul
node tools/compare-ul-runs.mjs --left <manifest-a.json> --right <manifest-b.json>
node tools/verify-training-provenance.mjs --self-test
```

Workload contracts:

- `tools/configs/training-workloads/ul-training-tiny.json`
- `tools/configs/training-workloads/ul-training-medium.json`

Deterministic CI fixture:

- `tests/fixtures/training/ul-tiny-ci-dataset.json`
- `seed=1337`, `expectedStepCount=2`

## Artifact Fields

UL run directories contain:

- `metrics.ndjson`: step-level training metrics report entries.
- `ul_stage1_manifest.json` or `ul_stage2_manifest.json`.

Manifest highlights:

- `manifestHash`: deterministic content hash (stable view).
- `manifestContentHash`: deterministic content hash alias for compatibility.
- `manifestFileHash` (runtime result field): hash of serialized file bytes.
- `ulContractHash`: stage-link contract identity.
- `runtimeDump`: resolved UL execution knobs for the run.
- `buildProvenance`: runtime/build/schema provenance block.
- `lineage` and `stage1Dependency`: parent linkage for stage2.
- `latentDataset`: stage1 latent-cache record path/hash/summary used for stage2 gating.

## Failure Taxonomy

Common hard-fail cases:

- `UL stage2 requires training.ul.stage1Artifact.`
- `UL stage2 artifact hash mismatch: ...`
- `UL stage2 requires stage1_joint artifact, got "...".`
- `UL stage2 contract mismatch: expected ..., got ...`
- `training metrics: ...` schema validation failures for malformed step payloads.

## Reproducibility Notes

- Deterministic manifest hash is content-based and excludes volatile fields
  such as timestamp/run id.
- File hash remains available for strict artifact byte-level gating.
