# Training Lineage Requirements

End-to-end lineage requirements across training outputs.

| Artifact type | Required lineage fields | Hash requirement | Notes |
|---|---|---|---|
| Suite report (`suite=training`) | `lineage.training.ulArtifacts[]` when UL artifacts exist | Each linked artifact hash must match manifest hash | Report must preserve run timestamp and model id |
| Bench report (`suite=bench`, `workloadType=training`) | `metrics.ulArtifacts[]` and `lineage.training.ulArtifacts[]` | Manifest hashes and paths must be coherent | Training metrics report entries must validate schema |
| Distill stage_a manifest | `manifestHash`, `distillContractHash`, `lossSummary` | Content hash must be deterministic | `metrics.stepMetricsPath` must resolve and validate |
| Distill stage_b manifest | `stageADependency`, `lineage.parentManifestHash`, `lineage.parentContractHash` | Stage_a link hash must match | Stage_b must fail on hash/contract/stage mismatch |
| UL stage1 manifest | `manifestHash`, `ulContractHash`, `latentDataset` | Content hash must be deterministic | `metrics.stepMetricsPath` must resolve and validate |
| UL stage2 manifest | `stage1Dependency`, `lineage.parentManifestHash`, `lineage.parentContractHash` | Stage1 link hash must match | Stage2 must fail on hash/contract/stage mismatch |
| Checkpoint (when exported) | `metadata.checkpointHash`, `metadata.lineage.checkpointKey`, `metadata.lineage.previousCheckpointHash`, `metadata.lineage.sequence` | Checkpoint hash must be stable over payload + metadata hash fields | Resume override trail must include mismatches + reason |

## Coherence Utility

Use:

```bash
node tools/verify-training-provenance.mjs --manifest <manifest.json> [--stage1-manifest <manifest.json>]
node tools/verify-training-provenance.mjs --report <report.json> [--checkpoint <checkpoint.json>]
```

The utility validates manifest shape, metrics payload schema, stage-link contracts, and report/checkpoint lineage coherence.
