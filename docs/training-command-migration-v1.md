# Training Command Migration v1

This note documents training command payload shape changes.

## New fields

1. `trainingSchemaVersion` (required semantics for training flows)
- Defaulted to `1` for `suite="training"` and `bench + workloadType="training"`.
- Any non-`1` value is rejected.

2. `trainingBenchSteps`
- Optional positive integer for training bench runs.
- Used only on calibrate path (`bench --workload-type training`).

## Existing field scope reminder

Training-only fields remain valid only when:

- `suite="training"`, or
- `command="bench"` with `workloadType="training"`.

All other usage is fail-closed.
