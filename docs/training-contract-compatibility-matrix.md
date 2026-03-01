# Training Contract Compatibility Matrix

Browser/Node/CLI compatibility matrix for training fields.

| Field | Browser runner | Node runner | CLI flag |
|---|---|---|---|
| `trainingTests` | supported | supported | `--training-tests` |
| `trainingStage` | supported | supported | `--training-stage` |
| `trainingConfig` | supported | supported | `--training-config-json` |
| `stage1Artifact` | supported | supported | `--stage1-artifact` |
| `stage1ArtifactHash` | supported | supported | `--stage1-artifact-hash` |
| `ulArtifactDir` | supported | supported | `--ul-artifact-dir` |
| `trainingSchemaVersion` | supported (pinned=1) | supported (pinned=1) | `--training-schema-version` |
| `trainingBenchSteps` | supported (`bench` + training workload) | supported (`bench` + training workload) | `--training-bench-steps` |
| `workloadType` | supported | supported | `--workload-type` |

## Command Scope Rules

1. Training fields are legal only for:
- `test-model --suite training`
- `bench --workload-type training`

2. Unsupported combinations fail fast in command normalization.

3. Suite routing is fail-closed with `unsupported_suite` metadata.
