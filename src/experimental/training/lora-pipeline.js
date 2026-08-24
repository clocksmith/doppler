import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { loadBackwardRegistry } from '../../config/backward-registry-loader.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../memory/buffer-pool.js';
import { runMatmul } from '../../gpu/kernels/index.js';
import { runResidualAdd } from '../../gpu/kernels/residual.js';
import { parseJsonl } from './datasets/jsonl.js';
import { loadTextPairsDataset, tokenizeTextPairs } from './datasets/text-pairs.js';
import { LoraAdapter } from './lora.js';
import { TrainingRunner, restoreTrainingCheckpointState } from './runner.js';
import { AdamOptimizer } from './optimizer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { OpType, AutogradTape } from './autograd.js';
import { loadCheckpoint } from './checkpoint.js';
import { exportLoRAAdapter } from './export.js';
import { computeEvalMetrics } from './operator-eval.js';
import { summarizeAgentEvalReportRequirements } from './operator-agent-eval.js';
import { appendScoreboardRow } from './operator-scoreboard.js';
import {
  buildArtifactBase,
  createTrainingRunLayout,
  hashArtifactPayload,
  writeJsonArtifact,
  writeRunContract,
  writeWorkloadLock,
} from './operator-artifacts.js';
import { watchFinalizedCheckpoints } from './checkpoint-watch.js';
import { loadLoRAFromManifest } from '../adapters/lora-loader.js';
import { createUploadedTensor } from './tensor-factory.js';
import { stableSortObject } from '../../formats/stable-sort-object.js';
import { LORA_MODULE_ALIASES } from '../../inference/pipelines/text/lora.js';
import { loadDistillModelHandle } from './suite.js';
import { createDistillStudentRuntimeModelFixture } from './distillation/student-fixture.js';
import { f16ToF32Array } from '../../inference/kv-cache/types.js';
import { sha256BytesHex, sha256Hex } from '../../formats/sha256.js';
import { LORA_RUNNER_BASE_MODEL_REGISTRY, finiteMetric, getLoraRunnerCompatibility, getPipelineConfig, isCausalLmLoraWorkload, isObjectRecord, normalizeLoraTargetModules, normalizeProviderEvalReport, preflightCausalLmLoraWorkload } from './lora/plan.js';
import { buildArtifact, createCausalLmDatasetBatches, evaluateCausalLmLoraModel, evaluateToyLoraModel, exportCausalLmLoraModel, exportProviderCausalLmLoraModel, exportToyLoraModel, loadCausalLmTextPairSamples, loadToyLoraDataset, makeTensorFromFloat32, makeTensorFromUint32, releaseTensor, resolveDatasetPathForLoadedWorkload } from './lora/checkpoint.js';
import { runInternalCausalLmLoraPipeline } from './lora/execution.js';
import { buildRunContract, createLoraRunLayout, runProviderCausalLmLoraPipeline } from './lora/recovery.js';
export { LORA_RUNNER_BASE_MODEL_REGISTRY, LORA_RUNNER_DATASET_FORMAT_REGISTRY, LORA_RUNNER_SUPPORT_CONTRACT, getLoraRunnerCompatibility, preflightCausalLmLoraWorkload } from './lora/plan.js';

export function assertLoraRunnerCompatibility(workload) {
  const compatibility = getLoraRunnerCompatibility(workload);
  if (compatibility.supported) return compatibility;
  throw new Error([
    'LoRA run is not supported by the current runner contract.',
    `supported baseModelId="${compatibility.runnerContract.supportedBaseModelId}"`,
    `supported datasetFormat="${compatibility.runnerContract.supportedDatasetFormat}"`,
    `registered baseModelIds="${compatibility.runnerContract.registeredBaseModelIds.join(',')}"`,
    `registered datasetFormats="${compatibility.runnerContract.registeredDatasetFormats.join(',')}"`,
    `observed baseModelId="${compatibility.observed.baseModelId}"`,
    `observed datasetFormat="${compatibility.observed.datasetFormat}"`,
    `observed taskType="${compatibility.observed.taskType}"`,
    `blockedReasons=${compatibility.blockedReasons.join(',')}`,
  ].join(' '));
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function createToyLoraModel(workload) {
  const targetModule = workload.pipeline.adapter.targetModules[0];
  if (!targetModule) {
    throw new Error('LoRA workload requires at least one adapter target module.');
  }
  const baseWeight = makeTensorFromFloat32(
    [0.08, -0.12, 0.16, 0.22, -0.03, 0.09],
    [3, 2],
    'lora_toy_base_weight'
  );
  const adapter = new LoraAdapter({
    inDim: 3,
    outDim: 2,
    rank: workload.pipeline.adapter.rank,
    alpha: workload.pipeline.adapter.alpha,
    dtype: 'f32',
  });
  const model = {
    adapter,
    baseWeight,
    targetModule,
    async forward(inputTensor, tape) {
      const batchSize = Number.isInteger(inputTensor?.shape?.[0]) ? inputTensor.shape[0] : 1;
      const baseLogits = await tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, batchSize, 2, 3, { transposeB: false, outputDtype: 'f32' }),
        [inputTensor, baseWeight],
        { M: batchSize, N: 2, K: 3, transposeB: false }
      );
      const delta = await adapter.forward(inputTensor, tape);
      return tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, batchSize * 2),
        [baseLogits, delta],
        { size: batchSize * 2 }
      );
    },
    loraParams() {
      return [adapter.A, adapter.B];
    },
    paramGroups() {
      return {
        encoder: [],
        prior: [],
        decoder: [],
        base: [baseWeight],
        lora: [adapter.A, adapter.B],
      };
    },
  };
  return {
    model,
    cleanup() {
      adapter.dispose();
      releaseTensor(baseWeight);
    },
  };
}

function createToyDatasetBatches(rows, batchSize) {
  return {
    async *batches() {
      let inputTensor = null;
      let targetTensor = null;
      let tensorBatchSize = 0;
      try {
        for (let offset = 0; offset < rows.length; offset += batchSize) {
          const batchRows = rows.slice(offset, offset + batchSize);
          const inputData = new Float32Array(batchRows.length * 3);
          const targetData = new Uint32Array(batchRows.length);
          for (let rowIndex = 0; rowIndex < batchRows.length; rowIndex += 1) {
            inputData.set(batchRows[rowIndex].input, rowIndex * 3);
            targetData[rowIndex] = batchRows[rowIndex].target;
          }
          if (!inputTensor || !targetTensor || tensorBatchSize !== batchRows.length) {
            releaseTensor(inputTensor);
            releaseTensor(targetTensor);
            inputTensor = makeTensorFromFloat32(inputData, [batchRows.length, 3], 'lora_toy_input');
            targetTensor = makeTensorFromUint32(targetData, [batchRows.length], 'lora_toy_target');
            tensorBatchSize = batchRows.length;
          } else {
            uploadData(inputTensor.buffer, inputData);
            uploadData(targetTensor.buffer, targetData);
          }
          yield {
            input: inputTensor,
            targets: targetTensor,
          };
        }
      } finally {
        releaseTensor(inputTensor);
        releaseTensor(targetTensor);
      }
    },
  };
}

function hasExternalCausalLmTrainer(loadedWorkload, options = {}) {
  return typeof options.causalLmTrainer === 'function'
    || Boolean(getPipelineConfig(loadedWorkload.workload).trainer);
}

async function runCausalLmLoraPipeline(options, compatibility) {
  if (!options.loadedWorkload.workload.datasetPath) {
    throw new Error('preflightCausalLmLoraWorkload requires workload.datasetPath.');
  }
  if (hasExternalCausalLmTrainer(options.loadedWorkload, options)) {
    return runProviderCausalLmLoraPipeline(options, compatibility);
  }
  const layout = await createLoraRunLayout(options, options.loadedWorkload.workload);
  await writeRunContract(layout, buildRunContract(options.loadedWorkload));
  await writeWorkloadLock(layout, options.loadedWorkload);
  return runInternalCausalLmLoraPipeline(options, layout, compatibility);
}

async function selectLatestCheckpoint(runRoot) {
  const checkpointsDir = join(runRoot, 'checkpoints');
  const entries = await readdir(checkpointsDir, { withFileTypes: true });
  const dirs = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((left, right) => left.localeCompare(right));
  const latest = dirs[dirs.length - 1];
  if (!latest) {
    throw new Error(`No checkpoints found in ${checkpointsDir}.`);
  }
  return {
    checkpointId: latest,
    checkpointPath: join(checkpointsDir, latest, 'state.json'),
    markerPath: join(checkpointsDir, latest, 'checkpoint.complete.json'),
  };
}

export async function runLoraPipeline(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  if (workload.kind !== 'lora') {
    throw new Error('runLoraPipeline requires a lora workload.');
  }
  const compatibility = getLoraRunnerCompatibility(workload);
  if (!compatibility.supported) {
    assertLoraRunnerCompatibility(workload);
  }
  if (isCausalLmLoraWorkload(workload, compatibility)) {
    return runCausalLmLoraPipeline(options, compatibility);
  }
  const layout = await createLoraRunLayout(options, workload);
  await writeRunContract(layout, buildRunContract(loadedWorkload));
  await writeWorkloadLock(layout, loadedWorkload);
  const dataset = await loadToyLoraDataset(workload.datasetPath);
  const fixture = createToyLoraModel(workload);
  try {
    const evalReports = [];
    const checkpointArtifacts = [];
    const exports = [];
    const finalizedCheckpointIds = new Set();
    const runner = new TrainingRunner({
      training: {
        enabled: true,
        optimizer: {
          type: workload.training.optimizer.type,
          lr: workload.training.optimizer.lr,
          beta1: workload.training.optimizer.beta1,
          beta2: workload.training.optimizer.beta2,
          eps: workload.training.optimizer.eps,
          weightDecay: workload.training.optimizer.weightDecay,
          scheduler: workload.training.optimizer.scheduler,
        },
        gradient: {
          maxNorm: workload.training.gradientClipping.maxNorm,
        },
        precision: workload.training.precision,
        lossScaling: { enabled: false },
        distill: {
          enabled: false,
          stage: 'stage_a',
          teacherModelId: null,
          studentModelId: null,
          datasetId: null,
          datasetPath: null,
          languagePair: null,
          sourceLangs: null,
          targetLangs: null,
          pairAllowlist: null,
          strictPairContract: false,
          shardIndex: null,
          shardCount: null,
          resumeFrom: null,
          artifactDir: null,
          stageAArtifact: null,
          stageAArtifactHash: null,
          temperature: 1,
          alphaKd: 1,
          alphaCe: 0,
          allowHintFallback: false,
          tripletMargin: 0.2,
          studentGraphMode: 'projection_head',
          freeze: { encoder: false, prior: false, decoder: false, base: true, lora: false },
        },
        ul: {
          enabled: false,
          stage: 'stage1_joint',
          stage1Artifact: null,
          stage1ArtifactHash: null,
          artifactDir: null,
          lambda0: 5,
          seed: workload.seed,
          noiseSchedule: { name: 'linear', minSigma: 0.1, maxSigma: 1, steps: 1 },
          priorAlignment: { enabled: false, weight: 1 },
          decoderSigmoidWeight: { enabled: false, maxWeight: 1 },
          lossWeights: { prior: 1, decoder: 1, recon: 1 },
          freeze: null,
        },
      },
    }, {
      optimizer: new AdamOptimizer({
        training: {
          optimizer: {
            type: workload.training.optimizer.type,
            lr: workload.training.optimizer.lr,
            beta1: workload.training.optimizer.beta1,
            beta2: workload.training.optimizer.beta2,
            eps: workload.training.optimizer.eps,
            weightDecay: workload.training.optimizer.weightDecay,
            scheduler: workload.training.optimizer.scheduler,
          },
          gradient: {
            maxNorm: workload.training.gradientClipping.maxNorm,
          },
          precision: workload.training.precision,
        },
      }),
      crossEntropyLoss,
      clipGradients,
      resolveCheckpointKey({ step }) {
        return join(layout.checkpoints, `checkpoint-${String(step).padStart(6, '0')}`, 'state.json');
      },
      onCheckpoint: async (checkpoint) => {
        const checkpointId = `checkpoint-${String(checkpoint.step).padStart(6, '0')}`;
        if (finalizedCheckpointIds.has(checkpointId)) return;
        finalizedCheckpointIds.add(checkpointId);
        const checkpointPayload = {
          ...buildArtifact(loadedWorkload, {
            prefix: 'lora_ckpt',
            id: checkpointId,
            artifactType: 'training_checkpoint',
            datasetHash: dataset.datasetHash,
            checkpointStep: checkpoint.step,
          }),
          checkpointId,
          checkpointPath: checkpoint.path,
          optimizerStatePresent: true,
          schedulerStatePresent: workload.training.optimizer.scheduler.enabled === true,
          resumeLineage: checkpoint.metadata?.lineage || null,
        };
        await writeJsonArtifact(
          join(layout.checkpoints, checkpointId, 'checkpoint.json'),
          checkpointPayload
        );
        const checkpointArtifact = await writeJsonArtifact(
          join(layout.checkpoints, checkpointId, 'checkpoint.complete.json'),
          checkpointPayload
        );
        checkpointArtifacts.push({
          checkpointId,
          checkpointPath: checkpoint.path,
          markerPath: checkpointArtifact.path,
          checkpointStep: checkpoint.step,
        });
        if (workload.pipeline.export?.enabled === true && workload.pipeline.export.atCheckpoints === true) {
          exports.push(await exportToyLoraModel(
            loadedWorkload,
            layout,
            fixture.model,
            checkpointId,
            checkpoint.step,
            dataset.datasetHash
          ));
        }
        const reports = await evaluateToyLoraModel(workload, fixture.model, dataset, layout, {
          checkpointId,
          checkpointStep: checkpoint.step,
          configHash: workload.configHash,
          workloadPath: loadedWorkload.absolutePath,
          workloadSha256: loadedWorkload.workloadSha256,
        });
        for (const report of reports) {
          evalReports.push(report);
          await appendScoreboardRow(layout.scoreboard, {
            artifactType: 'training_scoreboard',
            schemaVersion: 1,
            generatedAt: new Date().toISOString(),
            checkpointId,
            checkpointStep: checkpoint.step,
            evalDatasetId: report.evalDatasetId,
            primaryMetric: report.primaryMetric,
            primaryScore: report.primaryScore,
            accuracy: report.accuracy,
            metrics: {
              accuracy: report.accuracy,
              primaryScore: report.primaryScore,
            },
          }, {
            selectionMetric: workload.selectionMetric,
            selectionGoal: workload.selectionGoal,
          });
        }
      },
    });
    const metrics = await runner.run(
      fixture.model,
      createToyDatasetBatches(dataset.rows, workload.training.batchSize),
      {
        epochs: 1,
        batchSize: workload.training.batchSize,
        shuffle: false,
        maxSteps: workload.training.steps,
        checkpointEvery: workload.checkpointEvery,
        modelId: workload.baseModelId,
      }
    );
    const finalCheckpointId = runner.lastCheckpoint
      ? `checkpoint-${String(runner.lastCheckpoint.step).padStart(6, '0')}`
      : null;
    if (workload.pipeline.export?.enabled === true && finalCheckpointId && exports.every((entry) => entry.checkpointId !== finalCheckpointId)) {
      exports.push(await exportToyLoraModel(
        loadedWorkload,
        layout,
        fixture.model,
        finalCheckpointId,
        runner.lastCheckpoint.step,
        dataset.datasetHash
      ));
    }
    return {
      ok: true,
      kind: 'lora',
      action: 'run',
      workloadId: workload.id,
      runRoot: layout.runRoot,
      checkpointArtifacts,
      evalReports,
      exports,
      metrics,
      lastCheckpoint: runner.lastCheckpoint,
    };
  } finally {
    fixture.cleanup();
  }
}

export async function evaluateLoraCheckpoint(options) {
  const loadedWorkload = options.loadedWorkload;
  const checkpointPath = resolve(String(options.checkpointPath));
  const workload = loadedWorkload.workload;
  const dataset = await loadToyLoraDataset(workload.datasetPath);
  const checkpointRecord = await loadCheckpoint(checkpointPath);
  if (!checkpointRecord) {
    throw new Error(`Checkpoint not found: ${checkpointPath}`);
  }
  const fixture = createToyLoraModel(workload);
  try {
    await restoreTrainingCheckpointState(fixture.model, { getState: () => null }, checkpointRecord, {
      training: {
        distill: { freeze: { encoder: false, prior: false, decoder: false, base: true, lora: false } },
        ul: { freeze: null },
      },
    });
    return evaluateToyLoraModel(workload, fixture.model, dataset, options.layout || null, {
      checkpointId: options.checkpointId || 'checkpoint',
      checkpointStep: options.checkpointStep ?? null,
      configHash: workload.configHash,
      workloadPath: loadedWorkload.absolutePath,
      workloadSha256: loadedWorkload.workloadSha256,
    });
  } finally {
    fixture.cleanup();
  }
}

export async function exportLoraCheckpoint(options) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const layout = options.layout || {
    exports: resolve(options.exportsDir || 'reports/training/lora/exports'),
  };
  const checkpointPath = resolve(String(options.checkpointPath));
  const checkpointRecord = await loadCheckpoint(checkpointPath);
  if (!checkpointRecord) {
    throw new Error(`Checkpoint not found: ${checkpointPath}`);
  }
  const fixture = createToyLoraModel(workload);
  try {
    await restoreTrainingCheckpointState(fixture.model, { getState: () => null }, checkpointRecord, {
      training: {
        distill: { freeze: { encoder: false, prior: false, decoder: false, base: true, lora: false } },
        ul: { freeze: null },
      },
    });
    const checkpointId = options.checkpointId || 'checkpoint';
    return exportToyLoraModel(
      loadedWorkload,
      { ...layout, exports: layout.exports || resolve(options.exportsDir || 'reports/training/lora/exports') },
      fixture.model,
      checkpointId,
      options.checkpointStep ?? null,
      options.datasetHash || null
    );
  } finally {
    fixture.cleanup();
  }
}

export async function watchLoraCheckpoints(options) {
  const latestCheckpoint = await selectLatestCheckpoint(options.runRoot);
  return watchFinalizedCheckpoints({
    checkpointsDir: join(options.runRoot, 'checkpoints'),
    manifestPath: join(options.runRoot, 'scoreboard', 'watch-manifest.json'),
    pollIntervalMs: options.pollIntervalMs || 2000,
    stopWhenIdle: options.stopWhenIdle === true,
    signal: options.signal ?? null,
    onCheckpoint: async (markerPath) => {
      const raw = await readFile(markerPath, 'utf8');
      const marker = JSON.parse(raw);
      await evaluateLoraCheckpoint({
        loadedWorkload: options.loadedWorkload,
        checkpointPath: marker.checkpointPath || latestCheckpoint.checkpointPath,
        checkpointId: marker.checkpointId || latestCheckpoint.checkpointId,
        checkpointStep: marker.checkpointStep ?? null,
        layout: {
          eval: join(options.runRoot, 'eval'),
        },
      });
    },
  });
}

async function listJsonFilesRecursive(rootDir) {
  const results = [];
  let entries = [];
  try {
    entries = await readdir(rootDir, { withFileTypes: true });
  } catch (error) {
    if (error?.code === 'ENOENT') return results;
    throw error;
  }
  for (const entry of entries) {
    const absolutePath = join(rootDir, entry.name);
    if (entry.isDirectory()) {
      results.push(...await listJsonFilesRecursive(absolutePath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith('.json')) {
      results.push(absolutePath);
    }
  }
  return results.sort((left, right) => left.localeCompare(right));
}

function resolveComparableReportMetric(report, metric) {
  if (!report || typeof report !== 'object') return null;
  const direct = report[metric];
  if (typeof direct === 'number' && Number.isFinite(direct)) {
    return direct;
  }
  const nested = report.metrics?.[metric];
  if (typeof nested === 'number' && Number.isFinite(nested)) {
    return nested;
  }
  if (nested && typeof nested === 'object' && typeof nested.score === 'number' && Number.isFinite(nested.score)) {
    return nested.score;
  }
  return null;
}

async function loadRunWorkloadForComparison(runRoot) {
  try {
    const raw = await readFile(join(runRoot, 'workload.lock.json'), 'utf8');
    const lock = JSON.parse(raw);
    return lock.workload || null;
  } catch (error) {
    if (error?.code === 'ENOENT') return null;
    throw error;
  }
}

function sortTrainingReports(reports, workload = null) {
  const metric = String(workload?.selectionMetric || reports[0]?.primaryMetric || 'primaryScore').trim();
  const goal = String(workload?.selectionGoal || (metric === 'loss' ? 'min' : 'max')).trim();
  return reports
    .slice()
    .sort((left, right) => {
      const missingScore = goal === 'min' ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY;
      const leftScore = resolveComparableReportMetric(left, metric) ?? missingScore;
      const rightScore = resolveComparableReportMetric(right, metric) ?? missingScore;
      if (goal === 'min') return leftScore - rightScore;
      return rightScore - leftScore;
    });
}

async function loadTrainingEvalReports(rootDir) {
  const files = await listJsonFilesRecursive(rootDir);
  const reports = [];
  for (const filePath of files) {
    const raw = await readFile(filePath, 'utf8');
    const report = JSON.parse(raw);
    if (report?.artifactType === 'training_eval_report') {
      reports.push({
        ...report,
        reportPath: report.reportPath || filePath,
      });
    }
  }
  return reports;
}

export async function compareLoraRun(options) {
  const runRoot = resolve(String(options.runRoot));
  const workload = await loadRunWorkloadForComparison(runRoot);
  const sorted = sortTrainingReports(await loadTrainingEvalReports(join(runRoot, 'eval')), workload);
  const payload = {
    artifactType: 'training_compare_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    selectionMetric: workload?.selectionMetric || sorted[0]?.primaryMetric || null,
    selectionGoal: workload?.selectionGoal || null,
    count: sorted.length,
    best: sorted[0] || null,
    reports: sorted.map((report) => ({
      stage: report.stage || null,
      checkpointId: report.checkpointId || null,
      evalDatasetId: report.evalDatasetId || null,
      primaryMetric: report.primaryMetric || null,
      primaryScore: report.primaryScore ?? null,
      loss: report.loss ?? null,
      baseline: report.baseline || null,
      qualityClaim: report.qualityClaim || null,
      accuracy: report.accuracy ?? null,
      agentEval: report.agentEval || report.heldoutGate || null,
      reportPath: report.reportPath || null,
    })),
  };
  const artifact = await writeJsonArtifact(join(runRoot, 'compare', 'compare.json'), payload);
  return {
    ...payload,
    comparePath: artifact.path,
  };
}

function collectRequiredImprovementEvalIds(workload) {
  const evalDatasets = Array.isArray(workload?.evalDatasets) ? workload.evalDatasets : [];
  return evalDatasets
    .filter((entry) => entry?.quality?.requireImprovement === true)
    .map((entry) => entry.id);
}

function summarizeQualityClaims(reports) {
  const claims = reports
    .map((report) => report.qualityClaim)
    .filter((claim) => claim && typeof claim === 'object');
  return {
    count: claims.length,
    improvedCount: claims.filter((claim) => claim.improved === true).length,
    requiredCount: claims.filter((claim) => claim.requireImprovement === true).length,
    failedRequiredCount: claims.filter((claim) => claim.requireImprovement === true && claim.improved !== true).length,
  };
}

export async function qualityGateLoraRun(options) {
  const runRoot = resolve(String(options.runRoot));
  const requiredPaths = [
    join(runRoot, 'run_contract.json'),
    join(runRoot, 'workload.lock.json'),
  ];
  const checks = [];
  for (const filePath of requiredPaths) {
    try {
      await readFile(filePath, 'utf8');
      checks.push({ path: filePath, ok: true });
    } catch (error) {
      checks.push({ path: filePath, ok: false, error: error?.message || String(error) });
    }
  }
  const workload = await loadRunWorkloadForComparison(runRoot);
  const evalReports = await loadTrainingEvalReports(join(runRoot, 'eval'));
  const qualitySummary = summarizeQualityClaims(evalReports);
  const agentEvalSummary = summarizeAgentEvalReportRequirements(workload, evalReports);
  const requiredImprovementEvalIds = collectRequiredImprovementEvalIds(workload);
  if (evalReports.length > 0) {
    checks.push({
      name: 'eval_reports',
      path: join(runRoot, 'eval'),
      ok: true,
      count: evalReports.length,
    });
  }
  if (qualitySummary.count > 0) {
    checks.push({
      name: 'baseline_quality_claims',
      path: join(runRoot, 'eval'),
      ok: qualitySummary.failedRequiredCount === 0,
      ...qualitySummary,
    });
  }
  if (requiredImprovementEvalIds.length > 0 && qualitySummary.count === 0) {
    checks.push({
      name: 'required_improvement_claims',
      path: join(runRoot, 'eval'),
      ok: false,
      requiredEvalDatasetIds: requiredImprovementEvalIds,
      error: 'No baseline quality claims were written for eval datasets that require improvement.',
    });
  }
  if (agentEvalSummary.requiredCount > 0) {
    checks.push({
      name: 'agent_heldout_eval',
      path: join(runRoot, 'eval'),
      ok: agentEvalSummary.failedCount === 0,
      ...agentEvalSummary,
      error: agentEvalSummary.failedCount === 0
        ? null
        : 'One or more required agent held-out eval gates are missing or failing.',
    });
  }
  const passed = checks.every((entry) => entry.ok === true);
  const payload = {
    artifactType: 'training_quality_gate',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    passed,
    qualitySummary,
    agentEvalSummary,
    checks,
  };
  const artifact = await writeJsonArtifact(join(runRoot, 'quality-gate', 'quality-gate.json'), payload);
  return {
    ...payload,
    reportPath: artifact.path,
  };
}
