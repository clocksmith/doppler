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
import { stableSortObject } from '../../../formats/stable-sort-object.js';
import { LORA_MODULE_ALIASES } from '../../../inference/pipelines/text/lora.js';
import { loadDistillModelHandle } from '../suite.js';
import { createDistillStudentRuntimeModelFixture } from '../distillation/student-fixture.js';
import { f16ToF32Array } from '../../../inference/kv-cache/types.js';
import { sha256BytesHex, sha256Hex } from '../../../formats/sha256.js';
import { LORA_RUNNER_BASE_MODEL_REGISTRY, finiteMetric, getLoraRunnerCompatibility, getPipelineConfig, isCausalLmLoraWorkload, isObjectRecord, normalizeLoraTargetModules, normalizeProviderEvalReport, preflightCausalLmLoraWorkload } from './plan.js';

export async function pathIsReadable(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function resolveDatasetPathForLoadedWorkload(datasetPath, loadedWorkload) {
  const source = String(datasetPath || '');
  if (!source || isAbsolute(source) || /^https?:\/\//i.test(source)) {
    return source;
  }
  const candidates = [];
  const pushCandidate = (candidate) => {
    if (candidate && !candidates.includes(candidate)) {
      candidates.push(candidate);
    }
  };
  pushCandidate(resolve(source));
  let cursor = loadedWorkload?.absolutePath ? dirname(resolve(String(loadedWorkload.absolutePath))) : null;
  while (cursor) {
    pushCandidate(join(cursor, source));
    const parent = dirname(cursor);
    if (parent === cursor) break;
    cursor = parent;
  }
  for (const candidate of candidates) {
    if (await pathIsReadable(candidate)) {
      return candidate;
    }
  }
  return resolve(source);
}

export function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  return createUploadedTensor(data, 'f32', shape, label);
}

export function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  return createUploadedTensor(data, 'u32', shape, label);
}

export function releaseTensor(tensor) {
  if (!tensor?.buffer) return;
  releaseBuffer(tensor.buffer);
}

export function normalizeToyRow(record, index) {
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

export async function loadToyLoraDataset(datasetPath) {
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

export function createCausalLmDatasetBatches(samples) {
  return {
    async *batches() {
      for (const sample of samples) {
        const inputTensor = makeTensorFromUint32(
          sample.inputIds,
          [sample.inputIds.length],
          `lora_causal_lm_input_${sample.id}`
        );
        const targetTensor = makeTensorFromUint32(
          sample.targetIds,
          [sample.targetIds.length],
          `lora_causal_lm_target_${sample.id}`
        );
        try {
          yield {
            id: sample.id,
            input: inputTensor,
            targets: targetTensor,
            prompt: sample.prompt,
            completion: sample.completion,
            supervisedTokenCount: sample.supervisedTokenCount,
            ignoredTargetCount: sample.ignoredTargetCount,
          };
        } finally {
          releaseTensor(inputTensor);
          releaseTensor(targetTensor);
        }
      }
    },
  };
}

export async function loadCausalLmTextPairSamples(workload, datasetPath, tokenizer) {
  const dataset = await loadTextPairsDataset(datasetPath);
  const pipeline = getPipelineConfig(workload);
  const maxLength = Math.floor(Number(
    pipeline.maxLength
    ?? pipeline.sequenceLength
    ?? workload.training?.maxLength
  ));
  if (!Number.isInteger(maxLength) || maxLength < 2) {
    throw new Error('Causal-LM LoRA workload requires lora.maxLength or lora.sequenceLength >= 2.');
  }
  if (typeof pipeline.joinWith !== 'string') {
    throw new Error('Causal-LM LoRA workload requires lora.joinWith.');
  }
  const joinWith = pipeline.joinWith;
  const samples = await tokenizeTextPairs(tokenizer, dataset.rows, {
    maxLength,
    joinWith,
  });
  if (samples.length < 1) {
    throw new Error(`Causal-LM LoRA dataset ${dataset.absolutePath} produced no tokenized samples.`);
  }
  const tokenizationContract = {
    version: 2,
    objective: 'completion_only_causal_lm',
    truncation: 'preserve_completion_head_tail_prompt',
    maxLength,
    joinWith,
  };
  const supervisedTokenCount = samples.reduce(
    (sum, sample) => sum + sample.supervisedTokenCount,
    0
  );
  const ignoredTargetCount = samples.reduce(
    (sum, sample) => sum + sample.ignoredTargetCount,
    0
  );
  const truncatedPromptTokenCount = samples.reduce(
    (sum, sample) => sum + sample.truncatedPromptTokenCount,
    0
  );
  return {
    ...dataset,
    samples,
    datasetHash: hashArtifactPayload({
      rows: dataset.rows,
      tokenization: tokenizationContract,
    }),
    tokenization: {
      ...tokenizationContract,
      sampleCount: samples.length,
      supervisedTokenCount,
      ignoredTargetCount,
      truncatedPromptTokenCount,
      truncatedSampleCount: samples.filter(
        (sample) => sample.truncatedPromptTokenCount > 0
      ).length,
    },
  };
}

export function collectProtectedBuffers(model) {
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

export function disposeTapeOutputs(tape, protectedBuffers = new Set()) {
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

export function argmax(values) {
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

export async function evaluateToyLoraModel(workload, model, dataset, layout = null, checkpointMeta = {}) {
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

export function buildArtifact(loadedWorkload, options) {
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

export async function exportToyLoraModel(loadedWorkload, layout, model, checkpointId, checkpointStep, datasetHash) {
  const workload = loadedWorkload.workload;
  const targetModule = model.targetModule || workload.pipeline.adapter.targetModules[0];
  const weightsFilename = `${checkpointId}.adapters.safetensors`;
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
    weightsFormat: 'safetensors',
    weightsPath: weightsFilename,
  });
  const manifestPath = join(layout.exports, `${checkpointId}.adapter.manifest.json`);
  const weightsPath = join(layout.exports, weightsFilename);
  if (!exported.weights) {
    throw new Error('LoRA safetensors export did not return weights bytes.');
  }
  await writeFile(weightsPath, new Uint8Array(exported.weights));
  await writeFile(manifestPath, exported.json, 'utf8');
  await loadLoRAFromManifest(exported.manifest, {
    readFile: async (filePath) => readFile(join(layout.exports, filePath)),
  });
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
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    manifest: exported.manifest,
  };
  const artifactFile = await writeJsonArtifact(
    join(layout.exports, `${checkpointId}.export.json`),
    artifactPayload
  );
  return {
    checkpointId,
    manifestPath,
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    exportPath: artifactFile.path,
    manifest: exported.manifest,
  };
}

export async function exportCausalLmLoraModel(loadedWorkload, layout, fixture, checkpointId, checkpointStep, datasetHash) {
  const workload = loadedWorkload.workload;
  const pipeline = getPipelineConfig(workload);
  const adapter = pipeline.adapter || {};
  const weightsFilename = `${checkpointId}.adapters.safetensors`;
  const tensors = typeof fixture.model.loraTensorEntries === 'function'
    ? fixture.model.loraTensorEntries()
    : [];
  if (!tensors.length) {
    throw new Error('Causal-LM LoRA export requires trained adapter tensors.');
  }
  const exported = await exportLoRAAdapter({
    id: pipeline.export?.id || `${workload.id}-${checkpointId}`,
    name: pipeline.export?.name || `${workload.id}-${checkpointId}`,
    baseModel: workload.baseModelId,
    rank: adapter.rank,
    alpha: adapter.alpha,
    targetModules: normalizeLoraTargetModules(adapter),
    tensors,
    weightsFormat: 'safetensors',
    weightsPath: weightsFilename,
    metadata: {
      baseModelRef: fixture.baseModelRef,
      baseModelUrl: fixture.baseModelUrl,
      datasetFormat: pipeline.datasetFormat,
      taskType: pipeline.taskType,
    },
  });
  const manifestPath = join(layout.exports, `${checkpointId}.adapter.manifest.json`);
  const runtimeManifestPath = join(layout.exports, 'runtime-adapter-manifest.json');
  const weightsPath = join(layout.exports, weightsFilename);
  if (!exported.weights) {
    throw new Error('Causal-LM LoRA safetensors export did not return weights bytes.');
  }
  await writeFile(weightsPath, new Uint8Array(exported.weights));
  await writeFile(manifestPath, exported.json, 'utf8');
  await writeFile(runtimeManifestPath, exported.json, 'utf8');
  await loadLoRAFromManifest(exported.manifest, {
    readFile: async (filePath) => readFile(join(layout.exports, filePath)),
  });
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
    runtimeManifestPath,
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    manifest: exported.manifest,
  };
  const artifactFile = await writeJsonArtifact(
    join(layout.exports, `${checkpointId}.export.json`),
    artifactPayload
  );
  return {
    checkpointId,
    manifestPath,
    runtimeManifestPath,
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    exportPath: artifactFile.path,
    manifest: exported.manifest,
  };
}

export async function readSupervisedLossStats(loss, targets, vocabSize) {
  const lossElementCount = loss.shape.reduce((product, value) => product * value, 1);
  const targetElementCount = targets.shape.reduce((product, value) => product * value, 1);
  const [rawLoss, rawTargets] = await Promise.all([
    readBuffer(loss.buffer, lossElementCount * (loss.dtype === 'f16' ? 2 : 4)),
    readBuffer(targets.buffer, targetElementCount * 4),
  ]);
  const data = loss.dtype === 'f16'
    ? f16ToF32Array(new Uint16Array(rawLoss))
    : new Float32Array(rawLoss);
  const targetIds = new Uint32Array(rawTargets);
  if (data.length !== targetIds.length) {
    throw new Error(
      `Causal-LM loss/target length mismatch: ${data.length} losses for ${targetIds.length} targets.`
    );
  }
  let sum = 0;
  let supervisedTokenCount = 0;
  for (let index = 0; index < data.length; index += 1) {
    if (targetIds[index] >= vocabSize) continue;
    const value = data[index];
    if (!Number.isFinite(value)) {
      throw new Error(`Causal-LM loss is non-finite at supervised target ${index}.`);
    }
    sum += value;
    supervisedTokenCount += 1;
  }
  if (supervisedTokenCount < 1) {
    throw new Error('Causal-LM evaluation batch contains no supervised targets.');
  }
  return {
    lossSum: sum,
    meanLoss: sum / supervisedTokenCount,
    supervisedTokenCount,
    ignoredTargetCount: data.length - supervisedTokenCount,
  };
}

export function buildLossQualityClaim(evalDataset, meanLoss, baselineReport) {
  const baselineLoss = finiteMetric(baselineReport?.loss ?? baselineReport?.primaryScore);
  const adapterLoss = finiteMetric(meanLoss);
  if (baselineLoss === null || adapterLoss === null) {
    return null;
  }
  const quality = evalDataset.quality || null;
  const absoluteImprovement = baselineLoss - adapterLoss;
  const relativeImprovement = baselineLoss === 0 ? 0 : absoluteImprovement / Math.abs(baselineLoss);
  const minAbsoluteImprovement = finiteMetric(quality?.minAbsoluteImprovement) ?? 0;
  const minRelativeImprovement = finiteMetric(quality?.minRelativeImprovement) ?? 0;
  const improved = absoluteImprovement >= minAbsoluteImprovement
    && relativeImprovement >= minRelativeImprovement;
  return {
    baseline: quality?.baseline || 'base_model',
    metric: 'loss',
    selectionGoal: 'min',
    adapterScore: adapterLoss,
    baselineScore: baselineLoss,
    delta: adapterLoss - baselineLoss,
    absoluteImprovement,
    relativeImprovement,
    minAbsoluteImprovement,
    minRelativeImprovement,
    improved,
    requireImprovement: quality?.requireImprovement === true,
  };
}

export async function computeCausalLmTextPairLosses(workload, fixture, samples, protectedBuffers) {
  const sampleLosses = [];
  let lossSum = 0;
  let supervisedTokenCount = 0;
  let ignoredTargetCount = 0;
  for await (const resolvedBatch of createCausalLmDatasetBatches(samples).batches()) {
    const tape = new AutogradTape(loadBackwardRegistry());
    let loss = null;
    try {
      const { logits } = await fixture.model.forwardCausalLm(resolvedBatch, tape);
      loss = await crossEntropyLoss(logits, resolvedBatch.targets, { training: { precision: workload.training.precision } }, tape);
      const stats = await readSupervisedLossStats(
        loss,
        resolvedBatch.targets,
        logits.shape.at(-1)
      );
      sampleLosses.push(stats.meanLoss);
      lossSum += stats.lossSum;
      supervisedTokenCount += stats.supervisedTokenCount;
      ignoredTargetCount += stats.ignoredTargetCount;
    } finally {
      disposeTapeOutputs(tape, protectedBuffers);
    }
  }
  return {
    sampleLosses,
    sampleCount: sampleLosses.length,
    lossSum,
    supervisedTokenCount,
    ignoredTargetCount,
    meanLoss: supervisedTokenCount > 0 ? lossSum / supervisedTokenCount : null,
  };
}

export async function evaluateCausalLmLoraModel(workload, fixture, dataset, layout = null, checkpointMeta = {}) {
  const protectedBuffers = collectProtectedBuffers(fixture.model);
  const evalReports = [];
  const evalDatasets = Array.isArray(workload.evalDatasets) ? workload.evalDatasets : [];
  for (const evalDataset of evalDatasets) {
    if (evalDataset.evalKind !== 'text_generation') {
      throw new Error(`Causal-LM LoRA eval supports text_generation only, got "${evalDataset.evalKind}".`);
    }
    const evalDatasetPath = await resolveDatasetPathForLoadedWorkload(evalDataset.datasetPath, {
      absolutePath: checkpointMeta.workloadPath || null,
    });
    const evalDatasetMaterialized = evalDataset.datasetPath === dataset.absolutePath
      ? dataset
      : await loadCausalLmTextPairSamples(workload, evalDatasetPath, fixture.tokenizer);
    const lossStats = await computeCausalLmTextPairLosses(
      workload,
      fixture,
      evalDatasetMaterialized.samples,
      protectedBuffers
    );
    const meanLoss = lossStats.meanLoss;
    const baselineReports = checkpointMeta.baselineReportsByEvalDatasetId || {};
    const baselineReport = baselineReports[evalDataset.id] || null;
    const qualityClaim = buildLossQualityClaim(evalDataset, meanLoss, baselineReport);
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
      baseModelRef: fixture.baseModelRef,
      stage: checkpointMeta.stage || 'lora',
      checkpointStep: checkpointMeta.checkpointStep ?? null,
      checkpointId: checkpointMeta.checkpointId || null,
      evalDatasetId: evalDataset.id,
      metrics: {
        loss: {
          score: meanLoss,
          samples: lossStats.sampleCount,
          supervisedTokens: lossStats.supervisedTokenCount,
          ignoredTargets: lossStats.ignoredTargetCount,
        },
      },
      primaryMetric: 'loss',
      primaryScore: meanLoss,
      loss: meanLoss,
      baseline: baselineReport
        ? {
          checkpointId: baselineReport.checkpointId || null,
          stage: baselineReport.stage || null,
          primaryMetric: baselineReport.primaryMetric || null,
          primaryScore: baselineReport.primaryScore ?? null,
          loss: baselineReport.loss ?? null,
          reportPath: baselineReport.reportPath || null,
        }
        : null,
      qualityClaim,
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

export async function exportProviderCausalLmLoraModel(
  loadedWorkload,
  layout,
  trainerOutput,
  checkpointId,
  checkpointStep,
  datasetHash,
  trainerInfo,
  preflight
) {
  const workload = loadedWorkload.workload;
  const pipeline = getPipelineConfig(workload);
  const weightsFilename = `${checkpointId}.adapters.safetensors`;
  const exported = await exportLoRAAdapter({
    id: pipeline.export?.id || trainerOutput.adapterId || `${workload.id}-${checkpointId}`,
    name: pipeline.export?.name || trainerOutput.adapterName || `${workload.id}-${checkpointId}`,
    baseModel: workload.baseModelId,
    rank: pipeline.adapter.rank,
    alpha: pipeline.adapter.alpha,
    targetModules: normalizeLoraTargetModules(pipeline.adapter),
    tensors: trainerOutput.tensors,
    weightsFormat: 'safetensors',
    weightsPath: weightsFilename,
    metadata: {
      runnerKind: 'causal_lm_text_generation',
      runnerKey: preflight.runnerKey,
      runnerId: trainerOutput.runnerId || trainerInfo.runnerId,
      trainerId: trainerOutput.trainerId,
      trainerSource: trainerInfo.source,
      datasetHash,
      workloadSha256: loadedWorkload.workloadSha256,
      metrics: trainerOutput.metrics,
      receipts: trainerOutput.receipts,
    },
  });
  if (!exported.weights) {
    throw new Error('Causal-LM LoRA safetensors export did not return weights bytes.');
  }
  const manifestPath = join(layout.exports, `${checkpointId}.adapter.manifest.json`);
  const runtimeManifestPath = join(layout.exports, 'runtime-adapter-manifest.json');
  const weightsPath = join(layout.exports, weightsFilename);
  await writeFile(weightsPath, new Uint8Array(exported.weights));
  await writeFile(manifestPath, exported.json, 'utf8');
  await writeFile(runtimeManifestPath, exported.json, 'utf8');
  await loadLoRAFromManifest(exported.manifest, {
    readFile: async (filePath) => readFile(join(layout.exports, filePath)),
  });
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
    runtimeManifestPath,
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    runnerKey: preflight.runnerKey,
    trainer: {
      runnerId: trainerOutput.runnerId || trainerInfo.runnerId,
      trainerId: trainerOutput.trainerId,
      source: trainerInfo.source,
      exportName: trainerInfo.exportName,
    },
    metrics: trainerOutput.metrics,
    manifest: exported.manifest,
  };
  const artifactFile = await writeJsonArtifact(
    join(layout.exports, `${checkpointId}.export.json`),
    artifactPayload
  );
  return {
    checkpointId,
    manifestPath,
    runtimeManifestPath,
    weightsPath,
    weightsSha256: exported.weightsSha256 || null,
    exportPath: artifactFile.path,
    manifest: exported.manifest,
  };
}
