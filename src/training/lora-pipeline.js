import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import { loadBackwardRegistry } from '../config/backward-registry-loader.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../memory/buffer-pool.js';
import { createTensor } from '../gpu/tensor.js';
import { runMatmul } from '../gpu/kernels/index.js';
import { runResidualAdd } from '../gpu/kernels/residual.js';
import { parseJsonl } from './datasets/jsonl.js';
import { LoraAdapter } from './lora.js';
import { TrainingRunner, restoreTrainingCheckpointState } from './runner.js';
import { AdamOptimizer } from './optimizer.js';
import { crossEntropyLoss } from './loss.js';
import { clipGradients } from './clip.js';
import { OpType, AutogradTape } from './autograd.js';
import { loadCheckpoint } from './checkpoint.js';
import { exportLoRAAdapter } from './export.js';
import { computeEvalMetrics } from './operator-eval.js';
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

function stableSortObject(value) {
  if (Array.isArray(value)) {
    return value.map((entry) => stableSortObject(entry));
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const sorted = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = stableSortObject(value[key]);
  }
  return sorted;
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'f32', [...shape], label);
}

function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  const buffer = acquireBuffer(data.byteLength, undefined, label);
  uploadData(buffer, data);
  return createTensor(buffer, 'u32', [...shape], label);
}

function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
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
  });
  const model = {
    adapter,
    baseWeight,
    targetModule,
    async forward(inputTensor, tape) {
      const batchSize = Number.isInteger(inputTensor?.shape?.[0]) ? inputTensor.shape[0] : 1;
      const baseLogits = await tape.record(
        OpType.MATMUL,
        (a, b) => runMatmul(a, b, batchSize, 2, 3, { transposeB: false }),
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

function normalizeToyRow(record, index) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new Error(`LoRA toy dataset row ${index + 1} must be an object.`);
  }
  const values = Array.isArray(record.input)
    ? record.input
    : (Array.isArray(record.features) ? record.features : null);
  if (!Array.isArray(values) || values.length !== 3) {
    throw new Error(`LoRA toy dataset row ${index + 1} requires input[3].`);
  }
  const input = values.map((value, valueIndex) => {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
      throw new Error(`LoRA toy dataset row ${index + 1} input[${valueIndex}] must be finite.`);
    }
    return parsed;
  });
  const target = Number(record.target ?? record.label);
  if (!Number.isInteger(target) || target < 0 || target > 1) {
    throw new Error(`LoRA toy dataset row ${index + 1} requires integer target 0 or 1.`);
  }
  return {
    id: String(record.id || `row-${index + 1}`),
    input,
    target,
  };
}

async function loadToyLoraDataset(datasetPath) {
  const absolutePath = resolve(String(datasetPath));
  const raw = await readFile(absolutePath, 'utf8');
  const rows = absolutePath.endsWith('.json')
    ? JSON.parse(raw)
    : parseJsonl(raw);
  if (!Array.isArray(rows)) {
    throw new Error(`LoRA dataset "${absolutePath}" must be a JSON array or JSONL file.`);
  }
  const normalizedRows = rows.map((row, index) => normalizeToyRow(row, index));
  return {
    absolutePath,
    raw,
    rows: normalizedRows,
    datasetHash: hashArtifactPayload({ rows: normalizedRows }),
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

function collectProtectedBuffers(model) {
  const protectedBuffers = new Set();
  const groups = model.paramGroups();
  for (const params of Object.values(groups)) {
    for (const tensor of params) {
      if (tensor?.buffer) {
        protectedBuffers.add(tensor.buffer);
      }
    }
  }
  return protectedBuffers;
}

function disposeTapeOutputs(tape, protectedBuffers = new Set()) {
  if (!Array.isArray(tape?.records)) return;
  const released = new Set();
  for (const record of tape.records) {
    const output = record?.output;
    if (output?.buffer && !protectedBuffers.has(output.buffer) && !released.has(output.buffer)) {
      released.add(output.buffer);
      releaseBuffer(output.buffer);
    }
  }
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = Number.isFinite(values[index]) ? values[index] : Number.NEGATIVE_INFINITY;
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

async function evaluateToyLoraModel(workload, model, dataset, layout = null, checkpointMeta = {}) {
  const protectedBuffers = collectProtectedBuffers(model);
  const evalReports = [];
  const evalDatasets = Array.isArray(workload.evalDatasets) ? workload.evalDatasets : [];
  for (const evalDataset of evalDatasets) {
    if (evalDataset.evalKind !== 'classification' && evalDataset.evalKind !== 'text_generation') {
      throw new Error(`LoRA eval currently supports classification/text_generation only, got "${evalDataset.evalKind}".`);
    }
    const evalDatasetMaterialized = evalDataset.datasetPath === dataset.absolutePath
      ? dataset
      : await loadToyLoraDataset(evalDataset.datasetPath);
    const rows = evalDatasetMaterialized.rows;
    const predictions = [];
    const labels = [];
    for (const row of rows) {
      const tape = new AutogradTape(loadBackwardRegistry());
      const inputTensor = makeTensorFromFloat32(row.input, [1, 3], 'lora_eval_input');
      let logits = null;
      try {
        logits = await model.forward(inputTensor, tape);
        const logitsData = new Float32Array(await readBuffer(logits.buffer));
        predictions.push(String(argmax(logitsData)));
        labels.push(String(row.target));
      } finally {
        releaseTensor(inputTensor);
        if (logits?.buffer && !protectedBuffers.has(logits.buffer)) {
          releaseBuffer(logits.buffer);
        }
        disposeTapeOutputs(tape, protectedBuffers);
      }
    }
    const metrics = computeEvalMetrics('classification', predictions, labels, {});
    const reportPayload = {
      artifactType: 'training_eval_report',
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      workloadId: workload.id,
      workloadPath: checkpointMeta.workloadPath || null,
      workloadSha256: checkpointMeta.workloadSha256 || null,
      configHash: checkpointMeta.configHash || workload.configHash,
      datasetPath: evalDataset.datasetPath,
      datasetHash: evalDatasetMaterialized.datasetHash,
      baseModelId: workload.baseModelId,
      stage: 'lora',
      checkpointStep: checkpointMeta.checkpointStep ?? null,
      evalDatasetId: evalDataset.id,
      metrics,
      primaryMetric: metrics.primaryMetric,
      primaryScore: metrics.primaryScore,
      accuracy: metrics.accuracy?.score ?? null,
    };
    const reportFile = layout
      ? await writeJsonArtifact(
        join(layout.eval, `${checkpointMeta.checkpointId || 'checkpoint'}__${evalDataset.id}.json`),
        reportPayload
      )
      : null;
    evalReports.push({
      ...reportPayload,
      reportPath: reportFile?.path || null,
    });
  }
  return evalReports;
}

function buildRunContract(loadedWorkload) {
  return {
    artifactType: 'training_run_contract',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    workloadId: loadedWorkload.workload.id,
    workloadPath: loadedWorkload.absolutePath,
    workloadSha256: loadedWorkload.workloadSha256,
    configHash: loadedWorkload.workload.configHash,
    claimBoundary: loadedWorkload.workload.claimBoundary,
    kind: loadedWorkload.workload.kind,
    evalDatasets: loadedWorkload.workload.evalDatasets,
  };
}

function buildArtifact(loadedWorkload, options) {
  const workload = loadedWorkload.workload;
  const payload = buildArtifactBase({
    artifactType: options.artifactType,
    reportId: `${options.prefix}_${workload.id}_${options.id}`,
    workload,
    workloadPath: loadedWorkload.absolutePath,
    workloadSha256: loadedWorkload.workloadSha256,
    datasetPath: options.datasetPath || workload.datasetPath,
    datasetHash: options.datasetHash || null,
    baseModelId: workload.baseModelId,
    stage: options.stage || 'lora',
    checkpointStep: options.checkpointStep ?? null,
    parentArtifacts: options.parentArtifacts || [],
    runtime: 'node',
    surface: 'node',
    claimBoundary: workload.claimBoundary,
    configHash: options.configHash || workload.configHash,
  });
  return {
    ...payload,
    artifactHash: hashArtifactPayload(payload),
  };
}

async function exportToyLoraModel(loadedWorkload, layout, model, checkpointId, checkpointStep, datasetHash) {
  const workload = loadedWorkload.workload;
  const targetModule = model.targetModule || workload.pipeline.adapter.targetModules[0];
  const exported = await exportLoRAAdapter({
    id: workload.pipeline.export?.id || `${workload.id}-${checkpointId}`,
    name: workload.pipeline.export?.name || `${workload.id}-${checkpointId}`,
    baseModel: workload.baseModelId,
    rank: workload.pipeline.adapter.rank,
    alpha: workload.pipeline.adapter.alpha,
    targetModules: [targetModule],
    tensors: [
      { name: `layers.0.${targetModule}.lora_a`, tensor: model.adapter.A },
      { name: `layers.0.${targetModule}.lora_b`, tensor: model.adapter.B },
    ],
  });
  const manifestPath = join(layout.exports, `${checkpointId}.adapter.manifest.json`);
  await writeFile(manifestPath, exported.json, 'utf8');
  await loadLoRAFromManifest(exported.manifest, {});
  const artifactPayload = {
    ...buildArtifact(loadedWorkload, {
      prefix: 'lora_export',
      id: checkpointId,
      artifactType: 'lora_adapter_manifest',
      checkpointStep,
      datasetHash,
    }),
    checkpointId,
    manifestPath,
    manifest: exported.manifest,
  };
  const artifactFile = await writeJsonArtifact(
    join(layout.exports, `${checkpointId}.export.json`),
    artifactPayload
  );
  return {
    checkpointId,
    manifestPath,
    exportPath: artifactFile.path,
    manifest: exported.manifest,
  };
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
  if (workload.baseModelId !== 'training-toy') {
    throw new Error('LoRA run currently supports baseModelId="training-toy" only.');
  }
  if (workload.pipeline.datasetFormat !== 'toy_linear_classification_jsonl') {
    throw new Error('LoRA run currently supports datasetFormat="toy_linear_classification_jsonl" only.');
  }
  const layout = options.runRoot
    ? {
      runRoot: resolve(String(options.runRoot)),
      logs: join(resolve(String(options.runRoot)), 'logs'),
      checkpoints: join(resolve(String(options.runRoot)), 'checkpoints'),
      eval: join(resolve(String(options.runRoot)), 'eval'),
      scoreboard: join(resolve(String(options.runRoot)), 'scoreboard'),
      exports: join(resolve(String(options.runRoot)), 'exports'),
      compare: join(resolve(String(options.runRoot)), 'compare'),
      qualityGate: join(resolve(String(options.runRoot)), 'quality-gate'),
    }
    : await createTrainingRunLayout({
      kind: 'lora',
      workloadId: workload.id,
      timestamp: options.timestamp || null,
    });
  await Promise.all(Object.values(layout).map((dirPath) => mkdir(dirPath, { recursive: true })));
  await writeRunContract(layout, buildRunContract(loadedWorkload));
  await writeWorkloadLock(layout, loadedWorkload);
  const dataset = await loadToyLoraDataset(workload.datasetPath);
  const fixture = createToyLoraModel(workload);
  try {
    const evalReports = [];
    const checkpointArtifacts = [];
    const exports = [];
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

export async function compareLoraRun(options) {
  const evalDir = join(options.runRoot, 'eval');
  const entries = await readdir(evalDir, { withFileTypes: true });
  const reports = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith('.json')) continue;
    const raw = await readFile(join(evalDir, entry.name), 'utf8');
    reports.push(JSON.parse(raw));
  }
  const sorted = reports
    .slice()
    .sort((left, right) => {
      const leftScore = Number(left?.primaryScore ?? Number.NEGATIVE_INFINITY);
      const rightScore = Number(right?.primaryScore ?? Number.NEGATIVE_INFINITY);
      return rightScore - leftScore;
    });
  const payload = {
    artifactType: 'training_compare_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot: options.runRoot,
    count: sorted.length,
    best: sorted[0] || null,
    reports: sorted.map((report) => ({
      checkpointId: report.checkpointId || null,
      evalDatasetId: report.evalDatasetId || null,
      primaryMetric: report.primaryMetric || null,
      primaryScore: report.primaryScore ?? null,
      accuracy: report.accuracy ?? null,
      reportPath: report.reportPath || null,
    })),
  };
  const artifact = await writeJsonArtifact(join(options.runRoot, 'compare', 'compare.json'), payload);
  return {
    ...payload,
    comparePath: artifact.path,
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
  const passed = checks.every((entry) => entry.ok === true);
  const payload = {
    artifactType: 'training_quality_gate',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    runRoot,
    passed,
    checks,
  };
  const artifact = await writeJsonArtifact(join(runRoot, 'quality-gate', 'quality-gate.json'), payload);
  return {
    ...payload,
    reportPath: artifact.path,
  };
}
