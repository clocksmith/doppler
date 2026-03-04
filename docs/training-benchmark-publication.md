# Training Benchmark Publication Process

Publication process for training/distillation benchmark and quality claims.

## Inputs

- Contract gate status (`npm run ci:training:contract`)
- Workload registry validation (`npm run training:workloads:verify`)
- Report-id publication artifact (`npm run training:report-ids:publish`)
- Distill quality-gate outputs when applicable

## Publication bundle

Each claim publication must include:

1. Workload pack id/path/hash
2. Report id
3. Claim boundary statement
4. Surface and runtime metadata
5. Quality-gate bundle (distill claims)

## Required process

1. Run gates and collect artifacts.
2. Map every claim to report id + workload hash.
3. Publish benchmark summary with explicit boundary language.
4. Link machine-readable artifacts for replay and verification.

## Rejection conditions

- Missing report id or workload hash.
- Workload pack not present in workload registry.
- Distill claim without quality-gate artifacts.
- Contract gate failures in release window.
