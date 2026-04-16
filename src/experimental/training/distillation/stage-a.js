import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

import { AdamOptimizer } from '../optimizer.js';
import { crossEntropyLoss } from '../loss.js';
import { clipGradients } from '../clip.js';
import { TrainingRunner } from '../runner.js';
import {
  createDistillRuntimeContext,
  createDistillStudentRuntimeModelFixture,
  loadDistillDatasetFromJsonl,
  resolveDistillDataScope,
} from '../suite.js';
import { loadCanonicalTranslationDataset } from './dataset.js';
import { evaluateDistillationModel } from './eval.js';
import {
  buildDistillArtifactBase,
  writeDistillCheckpointComplete,
  writeDistillCheckpointMetadata,
  writeDistillStageManifest,
} from './artifacts.js';
import { appendDistillationScoreboardRow } from './scoreboard.js';
import { buildDistillationTrainingConfigFromWorkload } from './runtime.js';

function padStep(step) {
  return String(step).padStart(6, '0');
}

function resolveComparableMetric(report, metric) {
  if (!report || typeof report !== 'object') return null;
  const direct = report[metric];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }
  const metrics = report.metrics && typeof report.metrics === 'object' ? report.metrics : null;
  const nested = metrics?.[metric];
  if (typeof nested === 'number' && Number.isFinite(nested)) {
    return nested;
  }
  if (metrics?.[metric]?.score != null && Number.isFinite(metrics[metric].score)) {
    return metrics[metric].score;
  }
  return null;
}

function selectBestReport(reports, metric, goal) {
  const normalizedGoal = String(goal || 'max').trim();
  let best = null;
  let bestValue = null;
  for (const report of reports) {
    const value = resolveComparableMetric(report, metric);
    if (!Number.isFinite(value)) continue;
    if (best === null) {
      best = report;
      bestValue = value;
      continue;
    }
    const better = normalizedGoal === 'min'
      ? value < bestValue
      : value > bestValue;
    if (better) {
      best = report;
      bestValue = value;
    }
  }
  return best;
}

function shouldEvalOnCheckpoint(stageEntry) {
  const schedule = String(stageEntry?.evalSchedule || 'on_checkpoint').trim();
  return schedule === 'on_checkpoint';
}

function shouldEvalAtEnd(stageEntry) {
  const schedule = String(stageEntry?.evalSchedule || 'on_checkpoint').trim();
  return schedule === 'final';
}

export async function runDistillationStage(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const stageEntry = options.stageEntry;
  const datasetPath = options.datasetPath || workload.datasetPath;
  const legacyArtifactDir = options.legacyArtifactDir || resolve(options.layout.runRoot, 'legacy-stage-artifacts');
  if (workload.training.batchSize !== 1) {
    throw new Error('Distillation stage currently requires training.batchSize=1.');
  }
  if (workload.training.accumSteps !== 1) {
    throw new Error('Distillation stage currently requires training.accumSteps=1.');
  }
  const configBundle = buildDistillationTrainingConfigFromWorkload(loadedWorkload, stageEntry, {
    datasetPath,
    artifactDir: legacyArtifactDir,
    stageAArtifact: options.stageAArtifact || null,
    stageAArtifactHash: options.stageAArtifactHash || null,
  });
  const distillDataScope = resolveDistillDataScope({
    distillSourceLangs: workload.pipeline.sourceLangs,
    distillTargetLangs: workload.pipeline.targetLangs,
    distillPairAllowlist: workload.pipeline.pairAllowlist,
    strictPairContract: workload.pipeline.strictPairContract === true,
  }, configBundle.trainingConfig.training);
  const distillDatasetReport = await loadDistillDatasetFromJsonl(datasetPath, distillDataScope);
  if (!distillDatasetReport) {
    throw new Error(`Unable to resolve distillation dataset "${datasetPath}".`);
  }
  const canonicalDataset = await loadCanonicalTranslationDataset(datasetPath, {
    strictPairContract: workload.pipeline.strictPairContract === true,
    sourceLangs: workload.pipeline.sourceLangs,
    targetLangs: workload.pipeline.targetLangs,
    pairAllowlist: workload.pipeline.pairAllowlist,
  });
  const distillRuntime = await createDistillRuntimeContext({
    teacherModelId: workload.teacherModelId,
    studentModelId: workload.studentModelId,
    trainingStage: configBundle.internalStage,
    studentGraphMode: workload.pipeline.studentGraphMode,
  }, configBundle.trainingConfig.training);
  let fixture = null;
  try {
    fixture = await createDistillStudentRuntimeModelFixture({
      training: configBundle.trainingConfig.training,
    }, {
      distillRuntime,
      studentGraphMode: workload.pipeline.studentGraphMode,
    });
    const stageId = stageEntry.id;
    const stageCheckpointsDir = resolve(options.layout.checkpoints, stageId);
    await mkdir(stageCheckpointsDir, { recursive: true });
    const dataset = distillDatasetReport.createDataset({
      batchSize: workload.training.batchSize,
      shuffle: false,
      seed: workload.seed,
      distillRuntime,
    });
    const checkpointArtifacts = [];
    const evalReports = [];
    const runner = new TrainingRunner(fixture.config, {
      optimizer: new AdamOptimizer(fixture.config),
      crossEntropyLoss,
      clipGradients,
      resolveCheckpointKey({ step }) {
        const checkpointId = `checkpoint-${padStep(step)}`;
        return resolve(stageCheckpointsDir, checkpointId, 'state.json');
      },
      onCheckpoint: async (checkpoint) => {
        const checkpointId = `checkpoint-${padStep(checkpoint.step)}`;
        const checkpointBase = buildDistillArtifactBase(loadedWorkload, {
          prefix: 'dst_ckpt',
          artifactType: 'training_checkpoint',
          datasetPath: distillDatasetReport.absolutePath,
          datasetHash: canonicalDataset.canonicalHash,
          stage: stageId,
          checkpointStep: checkpoint.step,
          configHash: configBundle.trainingConfigHash,
          parentArtifacts: options.parentArtifacts || [],
        });
        const metadataPayload = {
          ...checkpointBase,
          checkpointId,
          checkpointPath: checkpoint.path,
          step: checkpoint.step,
          epoch: checkpoint.epoch,
          batch: checkpoint.batch,
          optimizerStatePresent: true,
          schedulerStatePresent: workload.training.optimizer.scheduler.enabled === true,
          stageArtifact: options.stageAArtifact || null,
          resumeLineage: checkpoint.metadata?.lineage || null,
        };
        const metadataFile = await writeDistillCheckpointMetadata(
          options.layout,
          stageId,
          checkpointId,
          metadataPayload
        );
        const completeFile = await writeDistillCheckpointComplete(
          options.layout,
          stageId,
          checkpointId,
          {
            ...metadataPayload,
            metadataPath: metadataFile.path,
            finalized: true,
          }
        );
        checkpointArtifacts.push({
          checkpointId,
          checkpointPath: checkpoint.path,
          metadataPath: metadataFile.path,
          completePath: completeFile.path,
          step: checkpoint.step,
        });
        if (!shouldEvalOnCheckpoint(stageEntry)) {
          return;
        }
        const reports = await evaluateDistillationModel({
          loadedWorkload,
          layout: options.layout,
          stageId,
          checkpointId,
          checkpointStep: checkpoint.step,
          checkpointPath: checkpoint.path,
          distillRuntime,
          model: fixture.model,
          configHash: configBundle.trainingConfigHash,
          parentArtifacts: [{
            artifactType: 'training_checkpoint',
            path: metadataFile.path,
            checkpointId,
          }],
        });
        for (const report of reports) {
          evalReports.push(report);
          await appendDistillationScoreboardRow(options.layout, stageId, {
            artifactType: 'training_scoreboard',
            schemaVersion: 1,
            generatedAt: new Date().toISOString(),
            stage: stageId,
            checkpointId,
            checkpointStep: checkpoint.step,
            evalDatasetId: report.evalDatasetId,
            selectionMetric: stageEntry.selectionMetric,
            selectionGoal: stageEntry.selectionGoal,
            primaryMetric: report.primaryMetric,
            primaryScore: report.primaryScore,
            bleu: report.bleu,
            chrf: report.chrf,
            reportPath: report.reportPath || null,
            metrics: {
              bleu: report.bleu,
              chrf: report.chrf,
              primaryScore: report.primaryScore,
            },
          }, {
            selectionMetric: stageEntry.selectionMetric,
            selectionGoal: stageEntry.selectionGoal,
          });
        }
      },
    });
    const metrics = await runner.run(fixture.model, dataset, {
      epochs: 1,
      batchSize: workload.training.batchSize,
      shuffle: false,
      maxSteps: stageEntry.steps,
      checkpointEvery: stageEntry.checkpointEvery,
      modelId: workload.studentModelId,
      trainingStage: configBundle.internalStage,
      runtimeProfile: null,
      command: 'distill',
      surface: 'node',
      timestamp: options.timestamp || null,
      distillArtifactDir: legacyArtifactDir,
      stageAArtifact: options.stageAArtifact || null,
      stageAArtifactHash: options.stageAArtifactHash || null,
      teacherModelId: workload.teacherModelId,
      studentModelId: workload.studentModelId,
      distillDatasetId: workload.datasetId,
      distillDatasetPath: distillDatasetReport.absolutePath,
      distillSourceLangs: workload.pipeline.sourceLangs,
      distillTargetLangs: workload.pipeline.targetLangs,
      distillPairAllowlist: workload.pipeline.pairAllowlist,
      strictPairContract: workload.pipeline.strictPairContract === true,
    });
    if (shouldEvalAtEnd(stageEntry) && runner.lastCheckpoint) {
      const finalCheckpointId = `checkpoint-${padStep(runner.lastCheckpoint.step)}`;
      const reports = await evaluateDistillationModel({
        loadedWorkload,
        layout: options.layout,
        stageId,
        checkpointId: finalCheckpointId,
        checkpointStep: runner.lastCheckpoint.step,
        checkpointPath: runner.lastCheckpoint.path || null,
        distillRuntime,
        model: fixture.model,
        configHash: configBundle.trainingConfigHash,
      });
      evalReports.push(...reports);
    }
    const bestReport = selectBestReport(
      evalReports,
      stageEntry.selectionMetric,
      stageEntry.selectionGoal
    );
    const stageManifestPayload = {
      ...buildDistillArtifactBase(loadedWorkload, {
        prefix: 'dst_stage',
        artifactType: 'distill_stage_manifest',
        datasetPath: distillDatasetReport.absolutePath,
        datasetHash: canonicalDataset.canonicalHash,
        stage: stageId,
        checkpointStep: runner.lastCheckpoint?.step ?? null,
        configHash: configBundle.trainingConfigHash,
        parentArtifacts: options.parentArtifacts || [],
      }),
      stageId,
      trainingStage: configBundle.internalStage,
      objective: stageEntry.objective,
      stagePlanEntry: stageEntry,
      stepCount: Array.isArray(metrics) ? metrics.length : 0,
      checkpointCount: checkpointArtifacts.length,
      bestCheckpointId: bestReport?.checkpointId || null,
      selectionMetric: stageEntry.selectionMetric,
      selectionGoal: stageEntry.selectionGoal,
      checkpointArtifacts,
      evalReports: evalReports.map((report) => ({
        checkpointId: report.checkpointId,
        evalDatasetId: report.evalDatasetId,
        reportPath: report.reportPath || null,
        primaryMetric: report.primaryMetric,
        primaryScore: report.primaryScore,
        bleu: report.bleu,
        chrf: report.chrf,
      })),
      legacyArtifact: runner.lastArtifact || null,
      lastCheckpoint: runner.lastCheckpoint || null,
    };
    const stageManifest = await writeDistillStageManifest(options.layout, stageManifestPayload);
    return {
      stageId,
      trainingStage: configBundle.internalStage,
      metrics,
      checkpointArtifacts,
      evalReports,
      bestReport,
      stageManifestPath: stageManifest.path,
      legacyArtifact: runner.lastArtifact || null,
      lastCheckpoint: runner.lastCheckpoint || null,
    };
  } finally {
    fixture?.cleanup?.();
    await distillRuntime.cleanup();
  }
}

export async function runDistillationStageA(options) {
  return runDistillationStage(options);
}
