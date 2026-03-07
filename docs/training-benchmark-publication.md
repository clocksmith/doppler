# Training Benchmark Publication Process

Publication process for training and distillation benchmark or quality claims.

## Inputs

- contract-gate status (`npm run ci:training:contract`)
- workload-registry validation (`npm run training:workloads:verify`)
- report-id publication artifact (`npm run training:report-ids:publish`)
- run-root compare and quality-gate artifacts for claimable outputs

## Publication Bundle

Each claim publication must include:

1. workload-pack ID, path, and hash
2. report ID
3. claim-boundary statement
4. surface and runtime metadata
5. compare report and quality-gate report when the claim is about a trained output rather than a raw harness lane

## Required Process

1. Run gates and collect artifacts.
2. Map every claim to report ID, workload hash, and run root.
3. Verify that the selected checkpoint or export has matching eval coverage.
4. Publish benchmark or quality summary with explicit boundary language.
5. Link machine-readable artifacts for replay and verification.

## Rejection Conditions

- missing report ID or workload hash
- workload pack not present in the workload registry
- claimable lora or distill output without a corresponding quality-gate artifact
- claimable checkpoint or export without matching eval artifacts
- contract-gate failures in the release window
