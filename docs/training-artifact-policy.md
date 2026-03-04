# Training Artifact Policy

Canonical policy for training artifact naming, hashing, lineage, and
deterministic behavior.

## Artifact conventions

UL run directories typically follow:

- `reports/training/ul/<stage>_<timestamp>/`

Common outputs:

- `metrics.ndjson`
- `ul_stage1_manifest.json` or `ul_stage2_manifest.json`
- stage-specific payloads (for example latents in stage1)

Distill run directories typically include:

- `metrics.ndjson`
- `distill_stage_a_manifest.json` or `distill_stage_b_manifest.json`

Deterministic workload packs are published in:

- `tools/configs/training-workloads/*.json`
- `tools/configs/training-workloads/registry.json`

## Hashing policy

- Hash function: SHA-256 hex for lineage/provenance linkage.
- `manifestContentHash`/`manifestHash`: deterministic normalized content hash.
- `manifestFileHash`: exact serialized file bytes hash.
- Content hash and file hash are intentionally distinct and both can be useful.

## Lineage requirements

- Stage2 UL artifacts must reference and validate stage1 dependency hashes.
- Stage B distill artifacts must reference and validate stage A dependency hashes.
- Reports should include explicit artifact references and matching manifest hashes
  when available.
- Published workload packs include deterministic `baselineReportId` values derived
  from workload content hashes. Claim publication artifacts must reference those ids.

## Deterministic timestamp behavior

- Timestamps are allowed for auditability and run tracking.
- Deterministic content hashes must exclude volatile timestamp/run-id fields.

## Validation tools

Use provenance verification for manifests/reports:

```bash
node tools/verify-training-provenance.mjs --manifest <manifest.json> [--stage1-manifest <manifest.json>]
node tools/verify-training-provenance.mjs --report <report.json>
node tools/verify-training-workload-packs.mjs --registry tools/configs/training-workloads/registry.json
node tools/publish-training-report-ids.mjs --registry tools/configs/training-workloads/registry.json --out reports/training/report-ids/latest.json
```
