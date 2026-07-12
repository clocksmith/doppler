# Training Handbook

Canonical operational handbook for Doppler training and distillation.

## Scope

- first-class workload-driven `lora` and `distill` operator commands
- legacy `verify` / `bench` training harness lanes for contract gates and calibration
- deterministic run roots, artifact lineage, and claim traceability
- eval, checkpoint watch, compare, and quality-gate workflows

This is operational guidance, not a paper-equivalent SOTA claim surface.

## Primary Commands

### Contract and release gates

```bash
npm run ci:training:contract
npm run training:contract:delta -- --out ./reports/training/contract-delta/latest.json
npm run training:workloads:verify
npm run training:report-ids:publish -- --out ./reports/training/report-ids/latest.json
```

### First-class operator commands

```bash
node src/cli/doppler-cli.js distill --config '{"request":{"action":"subsets","workloadPath":"src/experimental/training/workload-packs/distill-translategemma-tiny.json"}}'

node src/cli/doppler-cli.js distill --config '{"request":{"action":"run","workloadPath":"src/experimental/training/workload-packs/distill-translategemma-tiny.json"}}'

node src/cli/doppler-cli.js distill --config '{"request":{"action":"watch","runRoot":"reports/training/distill/distill-translategemma-tiny/2026-03-07T00-00-00.000Z","stopWhenIdle":true}}'

node src/cli/doppler-cli.js lora --config '{"request":{"action":"run","workloadPath":"src/experimental/training/workload-packs/lora-toy-tiny.json"}}'

node src/cli/doppler-cli.js lora --config '{"request":{"action":"export","runRoot":"reports/training/lora/lora-toy-tiny/2026-03-07T00-00-00.000Z"}}'
```

`tools/distillation.js` and `tools/lora.js` are thin wrappers over `src/cli/doppler-cli.js`.
The canonical request contract lives in `src/tooling/command-api.js`.

### Legacy harness lanes

```bash
node src/cli/doppler-cli.js verify --config '{"request":{"workload":"training","modelId":"gemma-3-270m-it-q4k-ehf16-af32"}}' --json

node src/cli/doppler-cli.js bench --config '{"request":{"workload":"training","modelId":"gemma-3-270m-it-q4k-ehf16-af32","workloadType":"training","trainingSchemaVersion":1}}' --json
```

Use these lanes for contract verification, calibration, and legacy training harness coverage.
They are not replacements for the workload-first `lora` / `distill` operator lifecycle.

## Command Surface

### `distill`

Actions:
- `run`
- `stage-a`
- `stage-b`
- `eval`
- `watch`
- `compare`
- `quality-gate`
- `subsets`

Action requirements:
- `run`, `stage-a`, `subsets`: `workloadPath` or `runRoot`
- `stage-b`: `workloadPath` or `runRoot`; provide `stageArtifact` when a validated Stage A artifact is not already discoverable from the run root
- `eval`: `workloadPath` or `runRoot`, plus either `checkpointPath` or finalized checkpoints under the run root
- `watch`, `compare`, `quality-gate`: `runRoot`

### `lora`

Actions:
- `run`
- `eval`
- `watch`
- `export`
- `compare`
- `quality-gate`
- `activate`

Action requirements:
- `run`: `workloadPath` or `runRoot`
- `eval`, `export`: `workloadPath` or `runRoot`, plus either `checkpointPath` or finalized checkpoints under the run root
- `watch`, `compare`, `quality-gate`: `runRoot`
- `activate`: part of the command contract, but the current Node operator runner fails closed and directs activation to the browser provider/runtime surface

## Current Implementation Notes

- `lora` and `distill` are currently Node-only in the command API and fail closed on browser surfaces.
- `lora run` executes the built-in `baseModelId="training-toy"` /
  `datasetFormat="toy_linear_classification_jsonl"` fixture, the native
  full-graph causal-LM runner for `gemma-3-270m-it-f16-af32`, and the
  provider-backed causal-LM LoRA contract for registered q4k Gemma/Qwen
  `text-pairs` workloads.
- The LoRA runner support contract is executable via
  `LORA_RUNNER_SUPPORT_CONTRACT`, `getLoraRunnerCompatibility()`, and
  `assertLoraRunnerCompatibility()` from
  `src/experimental/training/lora-pipeline.js`. Registered Columbo base-model
  ids include `gemma-3-270m-it-f16-af32`, `gemma4-e2b-it`,
  `gemma-4-e2b-it-q4k-ehf16-af32`,
  `gemma-4-e2b-it-q4k-ehf16-af32-int4ple`,
  `qwen-3-5-0-8b-q4k-ehaf16`, `qwen-3-5-2b-q4k-ehaf16`,
  `qwen-3-6-27b-q4k-ehaf16`, and `qwen-3-6-27b-q4k-eaf16`.
  The `text-pairs` dataset mapper and loader accept `{prompt, completion}`,
  `{source, target}`, and `{input, output}` rows. Gemma/Qwen causal-LM
  workloads (`datasetFormat="text-pairs"`, `taskType="text_generation"`) are
  supported by Doppler's causal-LM LoRA runner/export path. The native internal
  runner loads the f16 base model, tokenizes text-pair rows, trains LoRA
  tensors, and writes verified external safetensors adapter packages. Registered
  q4k bases require `runLoraPipeline({ causalLmTrainer })` or
  `lora.trainer.modulePath`; the internal full-graph runner does not train
  packed q4k base weights. Causal-LM workloads must declare
  `training.batchSize=1`, `lora.maxLength` or `lora.sequenceLength`, and
  `lora.joinWith`. A browser/Dream trainer must return named LoRA `lora_a` /
  `lora_b` tensors for every requested target module.
- `exportLoRAAdapter({ weightsFormat: "safetensors" })` returns a Doppler
  adapter manifest with `weightsPath`, `weightsSize`, `checksum`, and
  `checksumAlgorithm`, plus the external safetensors bytes. `lora run` writes
  checkpoint exports as `<checkpoint>.adapter.manifest.json`,
  `<checkpoint>.adapters.safetensors`, and `<checkpoint>.export.json`.
- `lora run` writes base-model eval reports before training for causal-LM
  workloads, then writes adapter eval reports with `baseline`, loss delta, and
  `qualityClaim` fields. Workload eval datasets can declare
  `quality.requireImprovement=true`; `quality-gate` fails if the required
  baseline comparison is missing or the adapter does not improve on the
  declared metric.
- Provider-backed q4k Gemma/Qwen trainers may return `evalReports`; Doppler
  materializes those reports under `eval/`, adds them to the scoreboard, and
  includes them in compare/quality-gate artifacts. This is the supported path
  for q4k student receipts until a native packed-q4k trainer exists.
- Code-agent held-out eval datasets can declare `agentEval` gates. These gates
  require a matching `training_eval_report.agentEval` receipt with passing row
  checks before `quality-gate` passes. The gate covers JS patching, WGSL review,
  manifest/config review, Reploid VFS/status/tool-loop behavior, patch
  application evidence, and no-hallucinated-files/tools checks.
- `tools/run-agent-heldout-eval.js` converts held-out candidate completions into
  a normal `training_eval_report`; when rows require patch evidence it verifies
  unified diffs with `git apply --check` against the explicit `--patch-root`.
- Host-session teacher selection for Doppler JavaScript and WGSL uses the
  versioned policy in `tools/policies/host-teacher-qualification-policy.json`
  and the hidden split bank in `tools/teacher-qualification/`. Verify the bank
  with `npm run training:teachers:verify`. Run qualification and accepted-label
  export with an explicit model identity, for example
  `npm run training:teachers:qualify -- --teacher codex=<full-model-id> --with-labels`.
  `claude -p` is also supported through
  `--teacher claude=<full-model-id>`.
- Host qualification runs in disposable pinned source snapshots. The actual
  git diff, allowed-path and command audits, exact original-source recovery,
  executable checks, CLI version, model id, evaluator harness hash, and
  policy/task hashes are written to a machine receipt. Provider prose and exit
  status alone are not evidence.
  Qualification tasks are never exported, only passing `label` repairs become
  text pairs, and `student_holdout` tasks remain reserved for student eval.
- GLM/Qwen/other teacher traces can be normalized into LoRA `text-pairs` with
  `tools/build-teacher-trace-text-pairs.js`. Trace rows preserve
  `schemaVersion`, `teacherModelId`, `studentBaseModelId`, `taskKind`,
  `policyId` / `sourcePolicyId`, `sourceFiles`, `generationParams`,
  `license`, provenance, and `gepaCandidateId` metadata.
- Fresh teacher traces from an OpenAI-compatible GLM/Qwen endpoint can be
  generated with `tools/generate-teacher-traces-openai-compatible.js`; it
  requires an explicit `--base-url`, `--model`, and `--api-key-env` and fails
  closed when credentials are missing.
- Reploid GEPA frontier exports can be converted into teacher traces with
  `tools/build-gepa-teacher-traces.js`. Doppler treats GEPA as a prompt/policy
  frontier source; the resulting `sourcePolicyId` and `gepaCandidateId` values
  travel with the training rows and adapter lineage.
- `distill` resolves KD and triplet workload stages into the internal `stage_a` / `stage_b` runner contract.
- Distillation workloads that declare `objective="sft"` or `trainingStage="sft"` must provide `distill.sftLora`; those stages route through the registered causal-LM LoRA `text-pairs` runner/export path and still write a distill stage manifest under the run root.
- Distillation translation eval is currently implemented for `studentGraphMode="transformer_full"` only.

## Verifier-Guided Optimizer Boundary

V8 qualified teacher repairs and trained them with completion-masked SFT.
Passing executable checks made the data execution-verified; it did not turn
that optimizer into RLVR.

V9 introduced grouped on-policy rollouts, policy/reference token log-probabilities,
reward vectors, advantages, DPO, and a clipped GRPO update through the Gamma
ROCm protocol. Qwen 3.5 9B is provisioned at exact revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. V10 completed seed-11 SFT and
raised family-disjoint public compiler-repair pass@1 from 8.36% to 88.29%.
That is a narrow SFT capability result, not RLVR, semantic kernel correctness,
or promotion. Base and adapted rollouts use the same task order, sample count,
sampling parameters, and per-sample seeds. Generation receipts record KV-cache,
sample-batch, and log-probability-batch behavior, and resumable prefixes fail
closed on any model, policy, dataset, sampling, or training mismatch.

DPO is skipped when verifier-derived data contains no preference pair. GRPO is
skipped when every group has zero advantages. Executing optimizer steps on an
all-zero learning signal is not RLVR evidence. The primary GRPO implementation
filters zero-advantage samples, seed-shuffles
the remaining samples, accumulates the declared microsteps against one frozen
rollout policy, and performs exactly one optimizer update with zero stale-policy
updates. V11 derived DPO and GRPO inputs only from the diagnostic split and
reserved the public split for one policy comparison. Twelve groups supplied
96 nonzero-advantage samples; 11 groups had constructive verifier variance and
one varied only on exact-reference match. The one-update GRPO policy raised
public pass@1 from 88.29% to 94.98%, with 20 paired wins and zero losses. The
400-step DPO policy trained on 11 pairs regressed to 36.79% pass@1 and is
rejected. This is one-seed compiler-repair capability evidence, not semantic
kernel correctness or promotion. See the
[V11 optimizer result](status/wgsl-repair-v11-2026-07-12.md).

These optimizer surfaces, and any future minimum-risk, process-supervision,
CISPO, or on-policy distillation lane, must follow the
[Verifier-Guided and RLVR Training Contract](rlvr-training-contract.md) and
materialize the complete artifact chain before making a training claim. The
[V8 rejection](status/wgsl-student-replay-v8-2026-07-11.md) and
[V10 SFT result](status/wgsl-repair-v10-2026-07-12.md) remain separate.

V12 corrects the data-lane schedule before running the curation ablation. The
old 800-step dataset order exposed byte-identical prefixes in anchor,
external20, and random20. The V12 workloads seed/hash-order all 1,200 rows and
consume each full lane once, recording the order hash. Evaluation is split by
the input-visible broken-span length into 64-token short and 640-token long
strata, then recombined over the original denominator. See the
[V12 controlled-lane design](status/wgsl-repair-v12-design-2026-07-12.md).

Legacy Distill Studio helpers are compatibility tooling only:
- `tools/distill-studio-mvp.js`
- `tools/distill-studio-diagnostics.js`
- `tools/distill-studio-quality-gate.js`

New operator behavior must be documented under the `distill` surface, not under Distill Studio naming.

## Workload Packs

Canonical workload packs live under:
- `src/experimental/training/workload-packs/lora-*.json`
- `src/experimental/training/workload-packs/distill-*.json`

Each pack is the source of truth for:
- `schemaVersion`, `kind`, `id`, `description`, `claimBoundary`, `seed`
- `baseModelId`, `studentModelId`, `teacherModelId`
- `datasetId`, `datasetPath`, `evalDatasets`
- `evalDatasets[].agentEval` for code-agent held-out promotion gates:
  `suiteId`, `categories`, `minPassRate`, `requirePatchApplies`,
  `requireNoHallucinatedFiles`, `requireNoHallucinatedTools`, `allowedFiles`,
  and `allowedTools`
- `trainingSchemaVersion`, `checkpointEvery`, `selectionMetric`, `selectionGoal`, `surfaceSupport`
- `training.optimizer`, `training.batchSize`, `training.accumSteps`, `training.steps`, `training.precision`, `training.gradientClipping`
- pipeline-specific fields in `lora.*` or `distill.*`

LoRA-specific fields include:
- `datasetFormat`
- `taskType`
- `baseModelRef`
- `baseModelRevision`
- `maxLength` / `sequenceLength`
- `joinWith`
- `adapter.rank`
- `adapter.alpha`
- `adapter.dropout`
- `adapter.targetModules`
- `freeze`
- `export`
- `activation`
- `trainer` for provider-backed causal-LM LoRA runs:
  `trainer.modulePath`, `trainer.exportName`, and optional `trainer.runnerId`

Distill-specific fields include:
- `stagePlan`
- `studentGraphMode`
- `temperature`
- `alphaKd`
- `alphaCe`
- `tripletMargin`
- `sourceLangs`
- `targetLangs`
- `pairAllowlist`
- `strictPairContract`
- `subsetSpec`

Rule: if it changes behavior, it belongs in the workload pack.

## Run Roots and Artifacts

Deterministic run roots:
- `reports/training/lora/<workload-id>/<timestamp>/`
- `reports/training/distill/<workload-id>/<timestamp>/`

Each run root contains:
- `run_contract.json`
- `workload.lock.json`
- `logs/`
- `checkpoints/`
- `eval/`
- `scoreboard/`
- `exports/`
- `compare/`
- `quality-gate/`

Checkpoint readiness is explicit:
- checkpoint metadata is written separately from raw state
- finalized checkpoints are marked with `checkpoint.complete.json`
- watchers evaluate only finalized checkpoints

Canonical artifact classes:
- `training_run_contract`
- `training_checkpoint`
- `training_eval_report`
- `training_scoreboard`
- `training_compare_report`
- `training_quality_gate`
- `lora_adapter_manifest`
- `distill_stage_manifest`
- `subset_manifest`

## Claim Traceability Requirements

A publishable claim must include:
- `reportId`
- `workloadId`
- `workloadPath`
- `workloadSha256`
- `configHash`
- `datasetPath`
- `datasetHash`
- `claimBoundary`

Mapping is deterministic from workload-pack bytes and derived artifact payloads.

## Governance Rules

- workload packs are the source of truth for behavior-changing operator policy
- run-root artifacts must preserve workload, dataset, and surface traceability
- browser surfaces must fail closed for unsupported training operator commands
- claim publication requires deterministic traceability fields and reproducible artifacts

## Release Cycle Requirements

Required artifacts per release cycle:
- contract-gate pass
- workload-registry verification
- report-id publication artifact
- compare and quality-gate artifacts for claimable LoRA or distill outputs

## Publication Bundle

Each claim publication must include:
1. workload-pack ID, path, and hash
2. report ID
3. claim-boundary statement
4. surface and runtime metadata
5. compare report and quality-gate report when the claim is about a trained output rather than a raw harness lane

## Rejection Conditions

- missing report ID or workload hash
- workload pack not present in the workload registry
- claimable LoRA or distill output without a corresponding quality-gate artifact
- claimable checkpoint or export without matching eval artifacts
- contract-gate failures in the release window

## Operating Rhythm

1. Validate contract gates.
2. Validate workload registry integrity.
3. Publish the report-id artifact.
4. Build subsets when required by a distill workload.
5. Execute `distill` or `lora` run / eval / watch lanes.
6. Generate compare and quality-gate artifacts before claim publication.
7. Publish only claimable artifacts with deterministic traceability fields.

## Rollout Readiness

Gate readiness:
- `npm run ci:training:contract` passes all lanes
- no lane filtering is used for release checks
- contract-delta artifact is generated for each release cycle

Operator surface readiness:
- `lora` and `distill` are present in `src/tooling/command-api.js`
- CLI and API docs describe the operator commands
- browser surfaces fail closed for unsupported operator actions
- workload packs are validated through the training-workload registry
- run roots write `run_contract.json` and `workload.lock.json`
- finalized checkpoints write `checkpoint.complete.json`
- eval, compare, scoreboard, and quality-gate artifacts are emitted for candidate runs

Traceability readiness:
- workload registry hashes match workload-pack files
- baseline report IDs are present for all workload packs
- report-id publication artifacts are generated and stored
- claimable artifacts carry workload and dataset traceability fields

## Incident Response

1. Freeze publication for affected workload IDs and report IDs.
2. Rerun contract gates and isolate the failing operator lane or artifact class.
3. Regenerate the run-root artifacts and report-id index.
4. Publish a corrective note with the affected workload IDs, report IDs, and claim boundary.

## Related Policy Docs

- [training-artifact-policy.md](training-artifact-policy.md)
- [rlvr-training-contract.md](rlvr-training-contract.md)
- [training-migrations.md](training-migrations.md)
