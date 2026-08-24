import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadBackwardRegistry } from '../../../config/backward-registry-loader.js';
import { acquireBuffer, readBuffer, releaseBuffer, uploadData } from '../../../memory/buffer-pool.js';
import { runMatmul } from '../../../gpu/kernels/index.js';
import { runResidualAdd } from '../../../gpu/kernels/residual.js';
import { parseJsonl } from '../datasets/jsonl.js';
import { loadTextPairsDataset, tokenizeTextPairs } from '../datasets/text-pairs.js';
import { LoraAdapter } from '../lora.js';
import { TrainingRunner, restoreTrainingCheckpointState } from '../runner.js';
import { AdamOptimizer } from '../optimizer.js';
import { crossEntropyLoss } from '../loss.js';
import { clipGradients } from '../clip.js';
import { OpType, AutogradTape } from '../autograd.js';
import { loadCheckpoint } from '../checkpoint.js';
import { exportLoRAAdapter } from '../export.js';
import { computeEvalMetrics } from '../operator-eval.js';
import { summarizeAgentEvalReportRequirements } from '../operator-agent-eval.js';
import { appendScoreboardRow } from '../operator-scoreboard.js';
import {
  buildArtifactBase,
  createTrainingRunLayout,
  hashArtifactPayload,
  writeJsonArtifact,
  writeRunContract,
  writeWorkloadLock,
} from '../operator-artifacts.js';
import { watchFinalizedCheckpoints } from '../checkpoint-watch.js';
import { loadLoRAFromManifest } from '../../adapters/lora-loader.js';
import { createUploadedTensor } from '../tensor-factory.js';
import { stableSortObject } from '../../../utils/stable-sort-object.js';
import { LORA_MODULE_ALIASES } from '../../../inference/pipelines/text/lora.js';
import { loadDistillModelHandle } from '../suite.js';
import { createDistillStudentRuntimeModelFixture } from '../distillation/student-fixture.js';
import { f16ToF32Array } from '../../../inference/kv-cache/types.js';
import { sha256BytesHex, sha256Hex } from '../../../utils/sha256.js';
import { LORA_RUNNER_BASE_MODEL_REGISTRY, finiteMetric, getLoraRunnerCompatibility, getPipelineConfig, isCausalLmLoraWorkload, isObjectRecord, normalizeLoraTargetModules, normalizeProviderEvalReport, preflightCausalLmLoraWorkload } from './plan.js';
import { buildArtifact, createCausalLmDatasetBatches, evaluateCausalLmLoraModel, evaluateToyLoraModel, exportCausalLmLoraModel, exportProviderCausalLmLoraModel, exportToyLoraModel, loadCausalLmTextPairSamples, loadToyLoraDataset, makeTensorFromFloat32, makeTensorFromUint32, releaseTensor, resolveDatasetPathForLoadedWorkload } from './checkpoint.js';
import { runInternalCausalLmLoraPipeline } from './execution.js';

export function buildRunContract(loadedWorkload) {
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

export async function createLoraRunLayout(options, workload) {
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
  return layout;
}

export function hashTextPairsDataset(dataset) {
  return hashArtifactPayload({ rows: dataset.rows });
}

export function parseCausalLmLoraTensorName(name) {
  const text = String(name || '');
  const match = text.match(/(?:^|\.)layers?\.?(\d+)\.(?:[A-Za-z0-9_]+\.)*([A-Za-z0-9_]+)\.lora_([ab])(?:\.[A-Za-z0-9_]+)?$/i);
  if (!match) return null;
  const layer = Number.parseInt(match[1], 10);
  const rawModule = match[2].toLowerCase();
  const module = LORA_MODULE_ALIASES[rawModule];
  if (!module) return null;
  return {
    layer,
    module,
    kind: match[3].toLowerCase() === 'a' ? 'a' : 'b',
  };
}

export function normalizeTensorShape(value, label) {
  const shape = Array.isArray(value) ? value.map((entry) => Number(entry)) : [];
  if (shape.length !== 2 || shape.some((entry) => !Number.isInteger(entry) || entry < 1)) {
    throw new Error(`${label} requires shape [rows, cols].`);
  }
  return shape;
}

export function normalizeTrainerTensor(entry, index) {
  if (!isObjectRecord(entry)) {
    throw new Error(`Causal-LM trainer tensor ${index + 1} must be an object.`);
  }
  const name = String(entry.name || '').trim();
  if (!name) {
    throw new Error(`Causal-LM trainer tensor ${index + 1} requires name.`);
  }
  const tensor = entry.tensor ?? entry.data ?? entry.values;
  if (tensor === undefined || tensor === null) {
    throw new Error(`Causal-LM trainer tensor "${name}" requires tensor data.`);
  }
  const shape = normalizeTensorShape(entry.shape ?? tensor?.shape, `Causal-LM trainer tensor "${name}"`);
  let normalizedTensor = tensor;
  if (Array.isArray(tensor)) {
    normalizedTensor = new Float32Array(tensor.map((value) => Number(value)));
  }
  if (normalizedTensor instanceof Float32Array && normalizedTensor.length !== shape[0] * shape[1]) {
    throw new Error(
      `Causal-LM trainer tensor "${name}" shape mismatch: expected ${shape[0] * shape[1]}, got ${normalizedTensor.length}.`
    );
  }
  return {
    name,
    shape,
    dtype: entry.dtype || 'f32',
    tensor: normalizedTensor,
  };
}

export function assertCausalLmTensorCoverage(tensors, adapter) {
  const targetModules = normalizeLoraTargetModules(adapter);
  if (!targetModules.length) {
    throw new Error('Causal-LM LoRA export requires at least one target module.');
  }
  const targetSet = new Set(targetModules);
  const layerModules = new Map();
  const seenModules = new Set();
  for (const tensor of tensors) {
    const parsed = parseCausalLmLoraTensorName(tensor.name);
    if (!parsed) {
      throw new Error(`Unrecognized Causal-LM LoRA tensor name: ${tensor.name}`);
    }
    if (!targetSet.has(parsed.module)) {
      throw new Error(
        `Causal-LM LoRA tensor "${tensor.name}" targets module "${parsed.module}" outside workload targetModules.`
      );
    }
    seenModules.add(parsed.module);
    if (!layerModules.has(parsed.layer)) {
      layerModules.set(parsed.layer, new Map());
    }
    const modules = layerModules.get(parsed.layer);
    if (!modules.has(parsed.module)) {
      modules.set(parsed.module, new Set());
    }
    modules.get(parsed.module).add(parsed.kind);
  }
  if (layerModules.size === 0) {
    throw new Error('Causal-LM trainer returned no LoRA layer tensors.');
  }
  for (const moduleName of targetModules) {
    if (!seenModules.has(moduleName)) {
      throw new Error(`Causal-LM trainer returned no tensors for target module "${moduleName}".`);
    }
  }
  for (const [layerIndex, modules] of layerModules.entries()) {
    for (const [moduleName, kinds] of modules.entries()) {
      if (!kinds?.has('a') || !kinds?.has('b')) {
        throw new Error(
          `Causal-LM trainer layer ${layerIndex} module ${moduleName} must include both lora_a and lora_b tensors.`
        );
      }
    }
  }
}

export function normalizeCausalLmTrainerOutput(output, workload) {
  if (!isObjectRecord(output)) {
    throw new Error('Causal-LM trainer must return an object.');
  }
  const rawTensors = Array.isArray(output.tensors)
    ? output.tensors
    : (Array.isArray(output.weights) ? output.weights : null);
  if (!rawTensors || rawTensors.length === 0) {
    throw new Error('Causal-LM trainer must return non-empty tensors.');
  }
  const tensors = rawTensors.map((entry, index) => normalizeTrainerTensor(entry, index));
  assertCausalLmTensorCoverage(tensors, getPipelineConfig(workload).adapter);
  const checkpointStep = Number.isInteger(Number(output.checkpointStep))
    ? Number(output.checkpointStep)
    : Number(workload.training?.steps || 0);
  const checkpointId = String(output.checkpointId || `checkpoint-${String(checkpointStep).padStart(6, '0')}`).trim();
  if (!/^[A-Za-z0-9_-]+$/.test(checkpointId)) {
    throw new Error(`Causal-LM trainer checkpointId "${checkpointId}" must use alphanumeric, underscore, or hyphen characters.`);
  }
  return {
    checkpointId,
    checkpointStep,
    adapterId: String(output.adapterId || '').trim() || null,
    adapterName: String(output.adapterName || '').trim() || null,
    trainerId: String(output.trainerId || '').trim() || null,
    runnerId: String(output.runnerId || '').trim() || null,
    metrics: isObjectRecord(output.metrics) ? output.metrics : {},
    receipts: Array.isArray(output.receipts) ? output.receipts.slice() : [],
    evalReports: Array.isArray(output.evalReports) ? output.evalReports.slice() : [],
    tensors,
  };
}

export async function writeProviderEvalReports(context) {
  const reports = [];
  for (let index = 0; index < context.trainerOutput.evalReports.length; index += 1) {
    const report = normalizeProviderEvalReport(context.trainerOutput.evalReports[index], index, context);
    const reportFile = await writeJsonArtifact(
      join(context.layout.eval, `${context.checkpointId}__${report.evalDatasetId}.json`),
      report
    );
    const materialized = {
      ...report,
      reportPath: reportFile.path,
    };
    reports.push(materialized);
    await appendScoreboardRow(context.layout.scoreboard, {
      artifactType: 'training_scoreboard',
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      checkpointId: materialized.checkpointId,
      checkpointStep: materialized.checkpointStep,
      stage: materialized.stage,
      evalDatasetId: materialized.evalDatasetId,
      primaryMetric: materialized.primaryMetric,
      primaryScore: materialized.primaryScore,
      loss: materialized.loss,
      qualityClaim: materialized.qualityClaim || null,
      agentEval: materialized.agentEval || null,
      metrics: {
        [materialized.primaryMetric]: materialized.primaryScore,
        primaryScore: materialized.primaryScore,
        agent_heldout_gate: materialized.agentEval?.passRate ?? null,
      },
    }, {
      selectionMetric: context.workload.selectionMetric,
      selectionGoal: context.workload.selectionGoal,
    });
  }
  return reports;
}

export async function resolveCausalLmTrainer(loadedWorkload, options = {}) {
  if (typeof options.causalLmTrainer === 'function') {
    return {
      train: options.causalLmTrainer,
      runnerId: options.causalLmTrainer.runnerId || 'injected_causal_lm_lora_trainer',
      source: 'runLoraPipeline.options.causalLmTrainer',
      exportName: null,
    };
  }
  const trainerConfig = getPipelineConfig(loadedWorkload.workload).trainer;
  if (!trainerConfig) {
    throw new Error(
      'causal_lm_trainer_not_configured: provide runLoraPipeline({ causalLmTrainer }) or lora.trainer.modulePath in the workload.'
    );
  }
  const modulePath = String(trainerConfig.modulePath || trainerConfig.path || '').trim();
  if (!modulePath) {
    throw new Error('causal_lm_trainer_not_configured: lora.trainer.modulePath is required.');
  }
  const exportName = String(trainerConfig.exportName || 'trainCausalLmLora').trim();
  if (!exportName) {
    throw new Error('causal_lm_trainer_not_configured: lora.trainer.exportName is required.');
  }
  const workloadDir = loadedWorkload.absolutePath
    ? dirname(resolve(String(loadedWorkload.absolutePath)))
    : process.cwd();
  const absoluteModulePath = isAbsolute(modulePath)
    ? modulePath
    : resolve(workloadDir, modulePath);
  const trainerModule = await import(pathToFileURL(absoluteModulePath).href);
  const train = trainerModule[exportName];
  if (typeof train !== 'function') {
    throw new Error(`causal_lm_trainer_not_configured: ${absoluteModulePath} does not export ${exportName}().`);
  }
  return {
    train,
    runnerId: String(trainerConfig.runnerId || exportName).trim(),
    source: absoluteModulePath,
    exportName,
  };
}

export async function runProviderCausalLmLoraPipeline(options, compatibility) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const pipeline = getPipelineConfig(workload);
  if (!workload.datasetPath) {
    throw new Error('preflightCausalLmLoraWorkload requires workload.datasetPath.');
  }
  const layout = await createLoraRunLayout(options, workload);
  await writeRunContract(layout, buildRunContract(loadedWorkload));
  await writeWorkloadLock(layout, loadedWorkload);
  const datasetPath = await resolveDatasetPathForLoadedWorkload(workload.datasetPath, loadedWorkload);
  const dataset = await loadTextPairsDataset(datasetPath, {
    fetch: options.fetch,
    readFile: options.readFile,
  });
  if (dataset.rowCount < 1) {
    throw new Error(`Causal-LM LoRA dataset ${dataset.absolutePath} has no rows.`);
  }
  const datasetHash = hashTextPairsDataset(dataset);
  const preflight = await preflightCausalLmLoraWorkload(workload, {
    datasetPath: dataset.absolutePath,
    fetch: options.fetch,
    readFile: options.readFile,
  });
  const trainerInfo = await resolveCausalLmTrainer(loadedWorkload, options);
  const trainerResult = await trainerInfo.train({
    schemaVersion: 1,
    runnerKind: 'causal_lm_lora',
    workload,
    loadedWorkload,
    compatibility,
    preflight,
    dataset: {
      absolutePath: dataset.absolutePath,
      rowCount: dataset.rowCount,
      rows: dataset.rows,
      datasetHash,
    },
    adapter: pipeline.adapter,
    training: workload.training,
    export: pipeline.export,
    layout,
  });
  const trainerOutput = normalizeCausalLmTrainerOutput(trainerResult, workload);
  const checkpointPayload = {
    ...buildArtifact(loadedWorkload, {
      prefix: 'lora_ckpt',
      id: trainerOutput.checkpointId,
      artifactType: 'training_checkpoint',
      datasetPath: dataset.absolutePath,
      datasetHash,
      checkpointStep: trainerOutput.checkpointStep,
    }),
    checkpointId: trainerOutput.checkpointId,
    checkpointPath: join(layout.checkpoints, trainerOutput.checkpointId, 'trainer-output.json'),
    optimizerStatePresent: false,
    schedulerStatePresent: false,
    runnerKey: preflight.runnerKey,
    trainer: {
      runnerId: trainerOutput.runnerId || trainerInfo.runnerId,
      trainerId: trainerOutput.trainerId,
      source: trainerInfo.source,
      exportName: trainerInfo.exportName,
    },
    tensorNames: trainerOutput.tensors.map((entry) => entry.name),
    metrics: trainerOutput.metrics,
    receipts: trainerOutput.receipts,
  };
  await writeJsonArtifact(
    join(layout.checkpoints, trainerOutput.checkpointId, 'trainer-output.json'),
    checkpointPayload
  );
  const checkpointArtifact = await writeJsonArtifact(
    join(layout.checkpoints, trainerOutput.checkpointId, 'checkpoint.complete.json'),
    checkpointPayload
  );
  const exported = pipeline.export?.enabled === false
    ? null
    : await exportProviderCausalLmLoraModel(
      loadedWorkload,
      layout,
      trainerOutput,
      trainerOutput.checkpointId,
      trainerOutput.checkpointStep,
      datasetHash,
      trainerInfo,
      preflight
    );
  const evalReports = await writeProviderEvalReports({
    loadedWorkload,
    workload,
    layout,
    dataset,
    datasetHash,
    preflight,
    trainerOutput,
    checkpointId: trainerOutput.checkpointId,
    checkpointStep: trainerOutput.checkpointStep,
  });
  return {
    ok: true,
    kind: 'lora',
    action: 'run',
    runnerKind: 'causal_lm_lora',
    workloadId: workload.id,
    runRoot: layout.runRoot,
    preflight,
    checkpointArtifacts: [{
      checkpointId: trainerOutput.checkpointId,
      checkpointPath: checkpointPayload.checkpointPath,
      markerPath: checkpointArtifact.path,
      checkpointStep: trainerOutput.checkpointStep,
    }],
    evalReports,
    exports: exported ? [exported] : [],
    metrics: trainerOutput.metrics,
    lastCheckpoint: {
      id: trainerOutput.checkpointId,
      step: trainerOutput.checkpointStep,
      path: checkpointPayload.checkpointPath,
    },
  };
}
