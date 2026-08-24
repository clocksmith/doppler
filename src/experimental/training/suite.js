import { initDevice, getKernelCapabilities, getDevice } from '../../gpu/device.js';
import { setPlatformsBaseUrl } from '../../config/platforms/loader.js';
import { setRegistryUrl } from '../../config/kernels/registry.js';
import { createTrainingConfig } from '../../config/training-defaults.js';
import {
  runMatmul,
  runResidualAdd,
} from '../../gpu/kernels/index.js';
import { createTensor } from '../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../memory/buffer-pool.js';
import { OpType } from './autograd.js';
import { AdamOptimizer } from './optimizer.js';
import { TrainingRunner } from './runner.js';
import { trainStep } from './trainer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { exportLoRAAdapter } from './export.js';
import { sha256Hex } from '../../utils/sha256.js';
import { computeSampleStats } from '../../debug/stats.js';
import { parseJsonl } from './datasets/jsonl.js';
import {
  buildDistillCandidatePrompt,
  buildDistillPrompt,
  encodeDistillRow,
  normalizeDistillDatasetPath,
  normalizeOptionalString,
  resolveDistillDataScope,
  summarizeDirectionCounts,
} from './distillation/suite-data.js';
import { createDistillStudentRuntimeModelFixture } from './distillation/student-fixture.js';
import { initializeInference } from '../../inference/test-harness.js';
import { createPipeline } from '../../inference/pipelines/text.js';
import { parseManifest } from '../../formats/rdrr/index.js';
import { openModelStore, loadManifestFromStore } from '../../storage/shard-manager.js';
import { DISTILL_ADAPTER_TOP_K, UL_STAGE_SET, buildDistillTrainingOverrides, clampDistillTopK, createDistillRuntimeContext, createToyModelFixture, loadDistillDatasetFromJsonl, normalizeTrainingConfigOverride, normalizeTrainingStage } from './suite/plan.js';
import { assertTrainingSchemaVersion, buildSuiteSummary, buildUlTrainingOverrides, isFiniteNumber, normalizeAdapterActivationConfig, resolveBenchProgressSummary, resolveDistillDatasetPath, resolveDistillShardProgressContext, trainingHarness, tryActivateAdapterPayload } from './suite/execution.js';
export { runTrainingSuite, trainingHarness } from './suite/execution.js';
export { buildDistillTrainingOverrides, createDistillRuntimeContext, createToyModelFixture, loadDistillDatasetFromJsonl, loadDistillModelHandle, normalizeDistillStudentGraphMode } from './suite/plan.js';

function makeTensorFromF16Bits(values, shape, label) {
  const data = values instanceof Uint16Array ? values : new Uint16Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label || 'train_tensor_f16');
  uploadData(buffer, data);
  return createTensor(buffer, 'f16', shape, label || 'train_tensor_f16');
}

export { buildDistillPrompt, resolveDistillDataScope };

export { createDistillStudentRuntimeModelFixture };

function normalizeLoRAExportConfig(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null;
  }
  const tensors = Array.isArray(value.tensors) ? value.tensors : [];
  if (tensors.length === 0) {
    return null;
  }
  const normalizedTensors = tensors.map((entry, index) => {
    const name = normalizeOptionalString(entry?.name);
    const paramIndex = Number.isFinite(entry?.paramIndex)
      ? Math.floor(entry.paramIndex)
      : -1;
    if (!name) {
      throw new Error(`adapterActivation.export.tensors[${index}].name is required.`);
    }
    if (!Number.isInteger(paramIndex) || paramIndex < 0) {
      throw new Error(`adapterActivation.export.tensors[${index}].paramIndex must be a non-negative integer.`);
    }
    return { name, paramIndex };
  });
  const targetModules = Array.isArray(value.targetModules)
    ? value.targetModules.map((moduleName) => String(moduleName || '').trim()).filter(Boolean)
    : [];
  if (targetModules.length === 0) {
    throw new Error('adapterActivation.export.targetModules must contain at least one module.');
  }
  const id = normalizeOptionalString(value.id);
  const name = normalizeOptionalString(value.name);
  const baseModel = normalizeOptionalString(value.baseModel);
  const rank = Number(value.rank);
  const alpha = Number(value.alpha);
  if (!id || !name || !baseModel) {
    throw new Error('adapterActivation.export requires id, name, and baseModel.');
  }
  if (!Number.isFinite(rank) || rank <= 0 || !Number.isInteger(rank)) {
    throw new Error('adapterActivation.export.rank must be a positive integer.');
  }
  if (!Number.isFinite(alpha) || alpha <= 0) {
    throw new Error('adapterActivation.export.alpha must be a positive number.');
  }
  return {
    id,
    name,
    baseModel,
    rank,
    alpha,
    targetModules,
    tensors: normalizedTensors,
    format: value.format === 'array' ? 'array' : 'base64',
    pretty: value.pretty === true,
  };
}

async function exportLoRAAdapterFromModel(model, exportConfig, runIndex = null) {
  const normalizedConfig = normalizeLoRAExportConfig(exportConfig);
  if (!normalizedConfig) return null;
  if (!model || typeof model.loraParams !== 'function') {
    throw new Error('adapterActivation.export requires model.loraParams() support.');
  }
  const params = model.loraParams();
  if (!Array.isArray(params) || params.length === 0) {
    throw new Error('adapterActivation.export requires non-empty model.loraParams().');
  }
  const tensors = normalizedConfig.tensors.map((entry) => {
    const tensor = params[entry.paramIndex];
    if (!tensor) {
      throw new Error(`adapterActivation.export tensor paramIndex ${entry.paramIndex} is out of range.`);
    }
    return {
      name: entry.name,
      tensor,
    };
  });
  const exported = await exportLoRAAdapter({
    id: normalizedConfig.id,
    name: normalizedConfig.name,
    baseModel: normalizedConfig.baseModel,
    rank: normalizedConfig.rank,
    alpha: normalizedConfig.alpha,
    targetModules: normalizedConfig.targetModules,
    tensors,
    format: normalizedConfig.format,
    pretty: normalizedConfig.pretty,
  });
  return {
    runIndex,
    manifest: exported.manifest,
    json: exported.json,
    hash: sha256Hex(exported.json),
  };
}

function toPositiveInteger(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const floored = Math.floor(parsed);
  return floored > 0 ? floored : fallback;
}

function appendTimelineEvent(timeline, type, details = {}) {
  timeline.push({
    index: timeline.length + 1,
    timestamp: new Date().toISOString(),
    type,
    ...details,
  });
}

function resolveBenchRunSettings(options = {}) {
  const benchRun = options.benchRun && typeof options.benchRun === 'object'
    ? options.benchRun
    : {};
  return {
    warmupRuns: Math.max(0, Math.floor(Number(benchRun.warmupRuns) || 0)),
    timedRuns: toPositiveInteger(benchRun.timedRuns, 1),
    stepsPerRun: toPositiveInteger(
      options.trainingBenchSteps ?? benchRun.steps ?? options.trainingSteps,
      2
    ),
  };
}

function resolveTrainingOverrides(options = {}) {
  const distillTraining = buildDistillTrainingOverrides(options);
  if (distillTraining?.distill?.enabled) {
    return distillTraining;
  }
  const ulTraining = buildUlTrainingOverrides(options);
  if (ulTraining) {
    return ulTraining;
  }
  return normalizeTrainingConfigOverride(options.trainingConfig) || undefined;
}

export async function runTrainingBenchSuite(options = {}) {
  const trainingSchemaVersion = assertTrainingSchemaVersion(options.trainingSchemaVersion);
  const startTime = performance.now();
  await trainingHarness.getGPU();

  const benchSettings = resolveBenchRunSettings(options);
  const totalRuns = benchSettings.warmupRuns + benchSettings.timedRuns;
  const trainingOverrides = resolveTrainingOverrides(options);
  const adapterActivation = normalizeAdapterActivationConfig(options);
  const distillEnabled = trainingOverrides?.distill?.enabled === true;
  const distillDatasetPath = resolveDistillDatasetPath(options, trainingOverrides);
  const distillDataScope = resolveDistillDataScope(options, trainingOverrides);
  const distillDatasetReport = distillEnabled
    ? await loadDistillDatasetFromJsonl(distillDatasetPath, distillDataScope)
    : null;
  const resolvedResumeFrom = options.resumeFrom || trainingOverrides?.distill?.resumeFrom || null;
  const resolvedStage1Artifact = options.stage1Artifact || trainingOverrides?.ul?.stage1Artifact || null;
  const resolvedStage1ArtifactHash = options.stage1ArtifactHash || trainingOverrides?.ul?.stage1ArtifactHash || null;
  const resolvedStageAArtifact = options.stageAArtifact || trainingOverrides?.distill?.stageAArtifact || null;
  const resolvedStageAArtifactHash = options.stageAArtifactHash || trainingOverrides?.distill?.stageAArtifactHash || null;
  let distillRuntime = null;
  if (distillEnabled) {
    if (!distillDatasetPath) {
      throw new Error('Distill benchmark requires --distill-dataset-path (training.distill.datasetPath).');
    }
    distillRuntime = await createDistillRuntimeContext(options, trainingOverrides);
  }

  const timedRunDurationsMs = [];
  const timedRunStepsPerSec = [];
  const timedStepDurationsMs = [];
  const timedRunUlArtifacts = [];
  const timedRunDistillArtifacts = [];
  const timedRunAdapterExports = [];
  const trainingMetricsReport = [];
  const distillShardProgress = resolveDistillShardProgressContext(
    options,
    trainingOverrides,
    benchSettings.stepsPerRun,
    distillDatasetReport?.shardCount ?? null
  );
  const checkpointResumeTimeline = [];
  appendTimelineEvent(checkpointResumeTimeline, 'benchmark_started', {
    workloadType: 'training',
    trainingStage: (
      options.trainingStage
      || trainingOverrides?.distill?.stage
      || trainingOverrides?.ul?.stage
      || null
    ),
    forceResume: options.forceResume === true,
    forceResumeReason: options.forceResume === true
      ? (options.forceResumeReason || null)
      : null,
    shardIndex: distillShardProgress.shardIndex,
    shardCount: distillShardProgress.shardCount,
    stepsPerShard: distillShardProgress.stepsPerShard,
  });
  if (resolvedResumeFrom) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_requested', {
      resumeFrom: String(resolvedResumeFrom),
    });
  }
  if (resolvedStage1Artifact) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_declared', {
      dependencyType: 'ul_stage1',
      stage1Artifact: String(resolvedStage1Artifact),
      stage1ArtifactHash: resolvedStage1ArtifactHash,
    });
  }
  if (resolvedStageAArtifact) {
    appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_declared', {
      dependencyType: 'distill_stage_a',
      stageAArtifact: String(resolvedStageAArtifact),
      stageAArtifactHash: resolvedStageAArtifactHash,
    });
  }
  let completedTimedRuns = 0;
  let latestExportedAdapter = null;

  try {
    for (let runIndex = 0; runIndex < totalRuns; runIndex += 1) {
      const fixture = distillEnabled
        ? await createDistillStudentRuntimeModelFixture({
          training: trainingOverrides,
        }, {
          outputDim: distillRuntime?.topK ?? DISTILL_ADAPTER_TOP_K,
          distillRuntime,
        })
        : createToyModelFixture({
          training: trainingOverrides,
        });
      try {
        const runner = new TrainingRunner(fixture.config, {
          optimizer: new AdamOptimizer(fixture.config),
          crossEntropyLoss,
          clipGradients,
        });
        const dataset = distillEnabled
          ? distillDatasetReport.createDataset({
            batchSize: 1,
            shuffle: false,
            seed: 1337 + runIndex,
            distillRuntime,
          })
          : {
            async *batches() {
              for (let i = 0; i < benchSettings.stepsPerRun; i += 1) {
                yield fixture.batch;
              }
            },
          };

        const runStart = performance.now();
        const isTimedRun = runIndex >= benchSettings.warmupRuns;
        appendTimelineEvent(checkpointResumeTimeline, 'run_started', {
          runIndex: runIndex + 1,
          phase: isTimedRun ? 'timed' : 'warmup',
        });
        const runMetrics = await runner.run(fixture.model, dataset, {
          epochs: 1,
          batchSize: 1,
          shuffle: false,
          maxSteps: benchSettings.stepsPerRun,
          modelId: options.modelId || distillRuntime?.studentModelId || 'training',
          modelUrl: options.modelUrl || distillRuntime?.studentModelUrl || null,
          runtimeProfile: options.runtimeProfile || null,
          trainingStage: (
            options.trainingStage
            || trainingOverrides?.distill?.stage
            || trainingOverrides?.ul?.stage
            || null
          ),
          command: options.command || null,
          surface: options.surface || null,
          forceResume: options.forceResume === true,
          forceResumeReason: options.forceResumeReason || null,
          forceResumeSource: options.forceResumeSource || null,
          checkpointOperator: options.checkpointOperator || null,
          checkpointEvery: options.checkpointEvery ?? null,
          gpuAdapterInfo: getKernelCapabilities(),
          timestamp: options.timestamp || null,
          ulArtifactDir: options.ulArtifactDir || null,
          distillArtifactDir: options.distillArtifactDir || null,
          stageAArtifact: resolvedStageAArtifact,
          stageAArtifactHash: resolvedStageAArtifactHash,
          teacherModelId: distillRuntime?.teacherModelId || options.teacherModelId || null,
          studentModelId: distillRuntime?.studentModelId || options.studentModelId || null,
          distillDatasetId: options.distillDatasetId || null,
          distillDatasetPath: distillDatasetReport?.absolutePath || null,
          distillLanguagePair: options.distillLanguagePair || null,
          distillSourceLangs: distillDataScope.sourceLangs || null,
          distillTargetLangs: distillDataScope.targetLangs || null,
          distillPairAllowlist: distillDataScope.pairAllowlist || null,
          strictPairContract: distillDataScope.strictPairContract === true,
          distillShardIndex: distillShardProgress.shardIndex,
          distillShardCount: distillShardProgress.shardCount,
          resumeFrom: resolvedResumeFrom,
        });
        const runDurationMs = Math.max(0, performance.now() - runStart);
        if (runner.resumeState && typeof runner.resumeState === 'object') {
          appendTimelineEvent(checkpointResumeTimeline, 'run_resumed', {
            runIndex: runIndex + 1,
            phase: isTimedRun ? 'timed' : 'warmup',
            resumedStep: runner.resumeState.step ?? null,
            resumedEpoch: runner.resumeState.epoch ?? null,
            resumedBatch: runner.resumeState.batch ?? null,
            resumedCheckpointHash: runner.resumeState.checkpointHash ?? null,
            previousCheckpointHash: runner.resumeState.previousCheckpointHash ?? null,
            resumeAuditCount: Number.isInteger(runner.resumeState.resumeAuditCount)
              ? runner.resumeState.resumeAuditCount
              : 0,
            checkpointKey: runner.resumeState.checkpointKey ?? null,
          });
          if (Number.isInteger(runner.resumeState.resumeAuditCount) && runner.resumeState.resumeAuditCount > 0) {
            appendTimelineEvent(checkpointResumeTimeline, 'resume_override_applied', {
              runIndex: runIndex + 1,
              phase: isTimedRun ? 'timed' : 'warmup',
              resumeAudits: Array.isArray(runner.resumeState.resumeAudits)
                ? runner.resumeState.resumeAudits
                : [],
            });
          }
        }
        appendTimelineEvent(checkpointResumeTimeline, 'run_completed', {
          runIndex: runIndex + 1,
          phase: isTimedRun ? 'timed' : 'warmup',
          durationMs: runDurationMs,
          stepCount: Array.isArray(runMetrics) ? runMetrics.length : 0,
        });
        if (isTimedRun) {
          completedTimedRuns += 1;
          timedRunDurationsMs.push(runDurationMs);
          const runStepCount = Array.isArray(runMetrics) ? runMetrics.length : 0;
          if (runDurationMs > 0 && runStepCount > 0) {
            timedRunStepsPerSec.push((runStepCount * 1000) / runDurationMs);
          }
          for (const stepEntry of runMetrics) {
            const stepWithRun = {
              ...stepEntry,
              bench_run_index: completedTimedRuns,
              bench_run_global_index: runIndex + 1,
            };
            if (isFiniteNumber(stepWithRun?.step_time_ms)) {
              timedStepDurationsMs.push(stepWithRun.step_time_ms);
            }
            trainingMetricsReport.push(stepWithRun);
          }
          if (runner.lastCheckpoint && typeof runner.lastCheckpoint === 'object') {
            appendTimelineEvent(checkpointResumeTimeline, 'checkpoint_state_written', {
              runIndex: runIndex + 1,
              timedRunIndex: completedTimedRuns,
              checkpointKey: runner.lastCheckpoint.key || null,
              checkpointStep: runner.lastCheckpoint.step ?? null,
              checkpointEpoch: runner.lastCheckpoint.epoch ?? null,
              checkpointBatch: runner.lastCheckpoint.batch ?? null,
            });
          }
          if (runner.lastArtifact && typeof runner.lastArtifact === 'object') {
            const artifactEntry = {
              runIndex: completedTimedRuns,
              ...runner.lastArtifact,
              resumeAudits: Array.isArray(runner.resumeState?.resumeAudits)
                ? runner.resumeState.resumeAudits
                : [],
            };
            appendTimelineEvent(checkpointResumeTimeline, 'checkpoint_written', {
              runIndex: runIndex + 1,
              timedRunIndex: completedTimedRuns,
              artifactKind: artifactEntry.kind || null,
              stage: artifactEntry.stage || null,
              manifestPath: artifactEntry.manifestPath || null,
              manifestHash: artifactEntry.manifestHash || null,
              manifestFileHash: artifactEntry.manifestFileHash || null,
            });
            if (artifactEntry.stageADependency) {
              appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_resolved', {
                dependencyType: 'distill_stage_a',
                runIndex: runIndex + 1,
                stageADependency: artifactEntry.stageADependency,
              });
            }
            if (artifactEntry.stage1Dependency) {
              appendTimelineEvent(checkpointResumeTimeline, 'resume_dependency_resolved', {
                dependencyType: 'ul_stage1',
                runIndex: runIndex + 1,
                stage1Dependency: artifactEntry.stage1Dependency,
              });
            }
            if (runner.lastArtifact.kind === 'distill') {
              timedRunDistillArtifacts.push(artifactEntry);
            } else {
              timedRunUlArtifacts.push(artifactEntry);
            }
          }
          if (adapterActivation.enabled && adapterActivation.exportConfig) {
            const exportedAdapter = await exportLoRAAdapterFromModel(
              fixture.model,
              adapterActivation.exportConfig,
              completedTimedRuns
            );
            if (exportedAdapter) {
              latestExportedAdapter = exportedAdapter;
              timedRunAdapterExports.push({
                runIndex: completedTimedRuns,
                id: exportedAdapter.manifest?.id || null,
                name: exportedAdapter.manifest?.name || null,
                hash: exportedAdapter.hash,
              });
            }
          }
        }
      } finally {
        fixture.cleanup();
      }
    }
  } finally {
    if (distillRuntime && typeof distillRuntime.cleanup === 'function') {
      await distillRuntime.cleanup();
    }
  }

  const runMsStats = computeSampleStats(timedRunDurationsMs);
  const stepMsStats = computeSampleStats(timedStepDurationsMs);
  const stepsPerSecStats = computeSampleStats(timedRunStepsPerSec);
  const progress = resolveBenchProgressSummary(trainingMetricsReport, distillShardProgress, startTime);
  const activationPayload = adapterActivation.adapterPayload
    ? adapterActivation.adapterPayload
    : (latestExportedAdapter
      ? {
        adapterManifest: latestExportedAdapter.manifest,
        adapterManifestJson: latestExportedAdapter.json,
      }
      : null);
  const adapterActivationResult = (
    adapterActivation.enabled
    && adapterActivation.autoActivate
  )
    ? await tryActivateAdapterPayload(activationPayload)
    : null;
  appendTimelineEvent(checkpointResumeTimeline, 'benchmark_completed', {
    completedTimedRuns,
    metricEntryCount: trainingMetricsReport.length,
    percentComplete: progress.percentComplete,
    etaMs: progress.etaMs,
  });

  const results = [
    {
      name: 'training-benchmark',
      passed: completedTimedRuns > 0 && trainingMetricsReport.length > 0,
      duration: Math.max(0, performance.now() - startTime),
      error: completedTimedRuns > 0 && trainingMetricsReport.length > 0
        ? undefined
        : 'No timed training benchmark runs completed.',
    },
  ];

  const summary = buildSuiteSummary('bench', results, startTime);
  return {
    ...summary,
    modelId: options.modelId || distillRuntime?.studentModelId || options.modelUrl || 'training',
    metrics: {
      workloadType: 'training',
      warmupRuns: benchSettings.warmupRuns,
      timedRuns: benchSettings.timedRuns,
      completedTimedRuns,
      stepsPerRun: benchSettings.stepsPerRun,
      trainingSchemaVersion,
      trainingMetricsReport,
      progress,
      ulArtifacts: timedRunUlArtifacts,
      distillArtifacts: timedRunDistillArtifacts,
      adapterExports: timedRunAdapterExports,
      adapterActivation: adapterActivationResult,
      checkpointResumeTimeline,
      distillDataset: distillDatasetReport
        ? {
          path: distillDatasetReport.absolutePath,
          rowCount: distillDatasetReport.rowCount,
          sampleCount: distillDatasetReport.sampleCount,
          shardCount: distillDatasetReport.shardCount ?? 1,
          directionCounts: distillDatasetReport.directionCounts,
          dataScope: distillDatasetReport.dataScope || null,
        }
        : null,
      latency: {
        runMs: runMsStats,
        stepMs: stepMsStats,
      },
      throughput: {
        stepsPerSec: stepsPerSecStats,
      },
    },
    deviceInfo: getKernelCapabilities(),
  };
}
