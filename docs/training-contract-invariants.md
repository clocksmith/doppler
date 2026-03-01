# Training Contract Invariants

This file is the canonical invariants block for training command flows.

## Verify vs Calibrate

1. `verify` intent is `test-model --suite training`.
- Exit condition: pass/fail suite summary with per-test diagnostics.
- Output contract: `suite="training"` and `metrics.testsRun/selectedTests/availableTests`.

2. `calibrate` intent is `bench --workload-type training`.
- Exit condition: comparable latency/throughput metrics with per-step training metrics report.
- Output contract: `suite="bench"` and `metrics.workloadType="training"`.

3. Verify and calibrate are not interchangeable.
- `verify` cannot silently emit bench semantics.
- `calibrate` cannot silently reuse verify-only result shape.

## Fail-Closed Invariants

1. Unknown suite values must fail with `code="unsupported_suite"` and structured metadata:
- `requestedSuite`
- `allowedSuites`
- `command`
- `surface`

2. Training-only fields are fail-closed.
- Allowed only for `suite="training"` or `bench + workloadType="training"`.
- Other combinations must throw validation errors.

3. Training schema version is pinned.
- `trainingSchemaVersion` must be `1` for training flows.
- Any other value is rejected.

4. Auto-surface fallback is blocked for training flows.
- CLI `--surface auto` may not silently downgrade training commands from node to browser.
