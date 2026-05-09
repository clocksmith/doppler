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
import { stableSortObject } from '../../utils/stable-sort-object.js';
import { LORA_MODULE_ALIASES } from '../../inference/pipelines/text/lora.js';
import { loadDistillModelHandle } from './suite.js';
import { createDistillStudentRuntimeModelFixture } from './distillation/student-fixture.js';
import { f16ToF32Array } from '../../inference/kv-cache/types.js';

const CAUSAL_LM_TEXT_PAIR_RUNNER_KEYS = Object.freeze([
  'gemma-3-270m-it-q4k-ehf16-af32::text-pairs::text_generation',
  'gemma4-e2b-it::text-pairs::text_generation',
  'gemma-4-e2b-it-q4k-ehf16-af32::text-pairs::text_generation',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple::text-pairs::text_generation',
  'qwen-3-5-0-8b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-5-2b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-6-27b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-6-27b-q4k-eaf16::text-pairs::text_generation',
]);

export const LORA_RUNNER_SUPPORT_CONTRACT = Object.freeze({
  supportedBaseModelId: 'training-toy',
  supportedDatasetFormat: 'toy_linear_classification_jsonl',
  registeredBaseModelIds: Object.freeze([
    'training-toy',
    'gemma-3-270m-it-q4k-ehf16-af32',
    'gemma4-e2b-it',
    'gemma-4-e2b-it-q4k-ehf16-af32',
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    'qwen-3-5-0-8b-q4k-ehaf16',
    'qwen-3-5-2b-q4k-ehaf16',
    'qwen-3-6-27b-q4k-ehaf16',
    'qwen-3-6-27b-q4k-eaf16',
  ]),
  registeredDatasetFormats: Object.freeze([
    'toy_linear_classification_jsonl',
    'text-pairs',
  ]),
  implementedRunnerKeys: Object.freeze([
    'training-toy::toy_linear_classification_jsonl::classification',
    ...CAUSAL_LM_TEXT_PAIR_RUNNER_KEYS,
  ]),
});

export const LORA_RUNNER_BASE_MODEL_REGISTRY = Object.freeze({
  'training-toy': Object.freeze({
    baseModelId: 'training-toy',
    family: 'training_fixture',
    runnerKind: 'toy_linear_classification',
  }),
  'gemma-3-270m-it-q4k-ehf16-af32': Object.freeze({
    baseModelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modelRef: 'gemma-3-270m-it-q4k-ehf16-af32',
    family: 'gemma3',
    runnerKind: 'causal_lm_text_generation',
  }),
  'gemma4-e2b-it': Object.freeze({
    baseModelId: 'gemma4-e2b-it',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
  }),
  'gemma-4-e2b-it-q4k-ehf16-af32': Object.freeze({
    baseModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
  }),
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple': Object.freeze({
    baseModelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
  }),
  'qwen-3-5-0-8b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    modelRef: 'qwen-3-5-0-8b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
  }),
  'qwen-3-5-2b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-5-2b-q4k-ehaf16',
    modelRef: 'qwen-3-5-2b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
  }),
  'qwen-3-6-27b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-6-27b-q4k-ehaf16',
    modelRef: 'qwen-3-6-27b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
  }),
  'qwen-3-6-27b-q4k-eaf16': Object.freeze({
    baseModelId: 'qwen-3-6-27b-q4k-eaf16',
    modelRef: 'qwen-3-6-27b-q4k-eaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
  }),
});

export const LORA_RUNNER_DATASET_FORMAT_REGISTRY = Object.freeze({
  toy_linear_classification_jsonl: Object.freeze({
    datasetFormat: 'toy_linear_classification_jsonl',
    datasetKind: 'toy_linear_classification',
  }),
  'text-pairs': Object.freeze({
    datasetFormat: 'text-pairs',
    datasetKind: 'causal_lm_text_pairs',
  }),
});

function getPipelineConfig(workload) {
  return workload?.pipeline || workload?.lora || {};
}

function normalizeLoraTargetModules(adapter) {
  const rawModules = Array.isArray(adapter?.targetModules) ? adapter.targetModules : [];
  const modules = [];
  for (const rawModule of rawModules) {
    const normalized = String(rawModule || '').trim();
    if (!normalized) continue;
    const moduleName = LORA_MODULE_ALIASES[normalized] || normalized;
    if (!modules.includes(moduleName)) {
      modules.push(moduleName);
    }
  }
  if (modules.length === 0) {
    throw new Error('Causal-LM LoRA workload requires adapter.targetModules.');
  }
  return modules;
}

function getCausalLmBaseModelRef(workload) {
  const pipeline = getPipelineConfig(workload);
  const baseModel = LORA_RUNNER_BASE_MODEL_REGISTRY[String(workload?.baseModelId || '')] || null;
  return String(
    pipeline.baseModelRef
    || workload.baseModelRef
    || workload.studentModelId
    || baseModel?.modelRef
    || workload.baseModelId
    || ''
  );
}

function getRunnerKey(baseModelId, datasetFormat, taskType) {
  return `${baseModelId}::${datasetFormat}::${taskType}`;
}

function isCausalLmLoraWorkload(workload, compatibility = getLoraRunnerCompatibility(workload)) {
  return compatibility.observed.baseModelRunnerKind === 'causal_lm_text_generation'
    || compatibility.observed.datasetKind === 'causal_lm_text_pairs'
    || compatibility.observed.taskType === 'text_generation';
}

export function getLoraRunnerCompatibility(workload) {
  const baseModelId = String(workload?.baseModelId || '');
  const pipeline = getPipelineConfig(workload);
  const datasetFormat = String(pipeline?.datasetFormat || '');
  const taskType = String(pipeline?.taskType || '');
  const baseModel = LORA_RUNNER_BASE_MODEL_REGISTRY[baseModelId] || null;
  const dataset = LORA_RUNNER_DATASET_FORMAT_REGISTRY[datasetFormat] || null;
  const runnerKey = getRunnerKey(baseModelId, datasetFormat, taskType);
  const blockedReasons = [];
  if (!baseModel) {
    blockedReasons.push('base_model_not_registered_for_current_lora_runner');
  }
  if (!dataset) {
    blockedReasons.push('dataset_format_not_supported_by_current_lora_runner');
  }
  if (baseModel && dataset && !LORA_RUNNER_SUPPORT_CONTRACT.implementedRunnerKeys.includes(runnerKey)) {
    blockedReasons.push('runner_combination_not_supported_by_current_lora_runner');
  }
  return {
    schemaVersion: 1,
    supported: blockedReasons.length === 0,
    runnerContract: LORA_RUNNER_SUPPORT_CONTRACT,
    observed: {
      baseModelId,
      datasetFormat,
      taskType,
      runnerKey,
      baseModelFamily: baseModel?.family || null,
      baseModelRunnerKind: baseModel?.runnerKind || null,
      datasetKind: dataset?.datasetKind || null,
      registeredBaseModel: Boolean(baseModel),
      registeredDatasetFormat: Boolean(dataset),
    },
    blockedReasons,
  };
}

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

function summarizeTextPairLengths(rows) {
  let minPromptChars = Number.POSITIVE_INFINITY;
  let maxPromptChars = 0;
  let minCompletionChars = Number.POSITIVE_INFINITY;
  let maxCompletionChars = 0;
  for (const row of rows) {
    const promptChars = row.prompt.length;
    const completionChars = row.completion.length;
    minPromptChars = Math.min(minPromptChars, promptChars);
    maxPromptChars = Math.max(maxPromptChars, promptChars);
    minCompletionChars = Math.min(minCompletionChars, completionChars);
    maxCompletionChars = Math.max(maxCompletionChars, completionChars);
  }
  if (!rows.length) {
    minPromptChars = 0;
    minCompletionChars = 0;
  }
  return {
    minPromptChars,
    maxPromptChars,
    minCompletionChars,
    maxCompletionChars,
  };
}

async function pathIsReadable(filePath) {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function resolveDatasetPathForLoadedWorkload(datasetPath, loadedWorkload) {
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

export async function preflightCausalLmLoraWorkload(workload, options = {}) {
  const compatibility = getLoraRunnerCompatibility(workload);
  if (!isCausalLmLoraWorkload(workload, compatibility)) {
    throw new Error('preflightCausalLmLoraWorkload requires a causal-LM LoRA workload.');
  }
  if (!workload?.datasetPath) {
    throw new Error('preflightCausalLmLoraWorkload requires workload.datasetPath.');
  }
  const datasetPath = options.datasetPath || workload.datasetPath;
  const dataset = await loadTextPairsDataset(datasetPath, {
    fetch: options.fetch,
    readFile: options.readFile,
  });
  if (dataset.rowCount < 1) {
    throw new Error(`Causal-LM LoRA dataset ${workload.datasetPath} has no rows.`);
  }
  const pipeline = getPipelineConfig(workload);
  const adapter = pipeline.adapter || {};
  return {
    schemaVersion: 1,
    supported: compatibility.supported,
    runnerKey: compatibility.observed.runnerKey,
    baseModelId: compatibility.observed.baseModelId,
    baseModelFamily: compatibility.observed.baseModelFamily,
    datasetPath: dataset.absolutePath,
    datasetFormat: compatibility.observed.datasetFormat,
    taskType: compatibility.observed.taskType,
    rowCount: dataset.rowCount,
    firstRowId: dataset.rows[0]?.id || null,
    lastRowId: dataset.rows[dataset.rows.length - 1]?.id || null,
    textPairFields: {
      prompt: dataset.rows[0]?.promptField || null,
      completion: dataset.rows[0]?.completionField || null,
    },
    textPairLengths: summarizeTextPairLengths(dataset.rows),
    adapter: {
      rank: Number(adapter.rank ?? 0),
      alpha: Number(adapter.alpha ?? 0),
      targetModules: normalizeLoraTargetModules(adapter),
    },
    blockedReasons: compatibility.blockedReasons.slice(),
  };
}

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function makeTensorFromFloat32(values, shape, label) {
  const data = values instanceof Float32Array ? values : new Float32Array(values);
  return createUploadedTensor(data, 'f32', shape, label);
}

function makeTensorFromUint32(values, shape, label) {
  const data = values instanceof Uint32Array ? values : new Uint32Array(values);
  return createUploadedTensor(data, 'u32', shape, label);
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

function createLossGradient(loss, lossScale) {
  const lossElements = loss.shape.reduce((acc, value) => acc * value, 1);
  const gradData = new Float32Array(lossElements);
  gradData.fill(lossScale);
  return createUploadedTensor(gradData, 'f32', loss.shape, 'lora_causal_lm_loss_grad');
}

function createCausalLmTrainingObjective() {
  return {
    name: 'causal_lm_cross_entropy',
    async forward({ model, batch, tape }) {
      if (typeof model.forwardCausalLm === 'function') {
        return model.forwardCausalLm(batch, tape);
      }
      const logits = await model.forward(batch.input, tape);
      return { logits };
    },
    async computeLoss({ batch, config, tape, forwardState }) {
      const loss = await crossEntropyLoss(forwardState.logits, batch.targets, config, tape);
      return { loss };
    },
    backwardTargets({ loss, lossScale }) {
      return createLossGradient(loss, lossScale);
    },
  };
}

function createCausalLmDatasetBatches(samples) {
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
          };
        } finally {
          releaseTensor(inputTensor);
          releaseTensor(targetTensor);
        }
      }
    },
  };
}

async function loadCausalLmTextPairSamples(workload, datasetPath, tokenizer) {
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
  return {
    ...dataset,
    samples,
    datasetHash: hashArtifactPayload({
      rows: dataset.rows,
      tokenization: {
        maxLength,
        joinWith,
      },
    }),
    tokenization: {
      maxLength,
      joinWith,
      sampleCount: samples.length,
    },
  };
}

async function createCausalLmLoraFixture(workload) {
  const pipeline = getPipelineConfig(workload);
  const adapter = pipeline.adapter || {};
  const modelRef = getCausalLmBaseModelRef(workload);
  const handle = await loadDistillModelHandle(modelRef, 'lora base', {
    runtime: {
      shared: {
        debug: {
          logLevel: {
            defaultLogLevel: 'debug',
          },
        },
      },
      inference: {
        compute: {
          activationDtype: 'f32',
          keepF32Weights: true,
        },
      },
    },
  });
  let fixture = null;
  try {
    fixture = await createDistillStudentRuntimeModelFixture({
      training: {
        precision: workload.training?.precision || {},
      },
    }, {
      distillRuntime: {
        studentPipeline: handle.pipeline,
        studentGraphMode: 'transformer_full',
      },
      studentGraphMode: 'transformer_full',
      loraAdapter: {
        rank: adapter.rank,
        alpha: adapter.alpha,
        targetModules: normalizeLoraTargetModules(adapter),
      },
    });
  } catch (error) {
    if (handle.pipeline && typeof handle.pipeline.unload === 'function') {
      await handle.pipeline.unload();
    }
    throw error;
  }
  return {
    model: fixture.model,
    baseModelRef: modelRef,
    baseModelUrl: handle.modelUrl || null,
    baseManifest: handle.manifest || null,
    tokenizer: handle.pipeline.tokenizer,
    cleanup() {
      fixture.cleanup();
      if (handle.pipeline && typeof handle.pipeline.unload === 'function') {
        return handle.pipeline.unload();
      }
      return undefined;
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

async function exportCausalLmLoraModel(loadedWorkload, layout, fixture, checkpointId, checkpointStep, datasetHash) {
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

async function readLossMean(loss) {
  const raw = await readBuffer(loss.buffer);
  const data = loss.dtype === 'f16'
    ? f16ToF32Array(new Uint16Array(raw))
    : new Float32Array(raw);
  if (!data.length) return 0;
  let sum = 0;
  for (const value of data) {
    sum += Number.isFinite(value) ? value : 0;
  }
  return sum / data.length;
}

async function evaluateCausalLmLoraModel(workload, fixture, dataset, layout = null, checkpointMeta = {}) {
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
    const losses = [];
    for await (const resolvedBatch of createCausalLmDatasetBatches(evalDatasetMaterialized.samples).batches()) {
      const tape = new AutogradTape(loadBackwardRegistry());
      let loss = null;
      try {
        const { logits } = await fixture.model.forwardCausalLm(resolvedBatch, tape);
        loss = await crossEntropyLoss(logits, resolvedBatch.targets, { training: { precision: workload.training.precision } }, tape);
        losses.push(await readLossMean(loss));
      } finally {
        disposeTapeOutputs(tape, protectedBuffers);
      }
    }
    const meanLoss = losses.length
      ? losses.reduce((sum, value) => sum + value, 0) / losses.length
      : null;
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
      stage: 'lora',
      checkpointStep: checkpointMeta.checkpointStep ?? null,
      evalDatasetId: evalDataset.id,
      metrics: {
        loss: {
          score: meanLoss,
          samples: losses.length,
        },
      },
      primaryMetric: 'loss',
      primaryScore: meanLoss,
      loss: meanLoss,
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

function getCausalLmFreezeConfig(workload) {
  const pipeline = getPipelineConfig(workload);
  const freeze = pipeline.freeze;
  if (!freeze || typeof freeze !== 'object' || Array.isArray(freeze)) {
    throw new Error('Causal-LM LoRA workload requires lora.freeze.');
  }
  return {
    encoder: freeze.encoder === true,
    prior: freeze.prior === true,
    decoder: freeze.decoder === true,
    base: freeze.base === true,
    lora: freeze.lora === true,
  };
}

function createLoraRunnerTrainingConfig(workload, freeze) {
  return {
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
        studentGraphMode: 'transformer_full',
        freeze,
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
  };
}

function createLoraOptimizer(workload) {
  return new AdamOptimizer({
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
  });
}

async function createLoraRunLayout(options, workload) {
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

function isObjectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function hashTextPairsDataset(dataset) {
  return hashArtifactPayload({ rows: dataset.rows });
}

function parseCausalLmLoraTensorName(name) {
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

function normalizeTensorShape(value, label) {
  const shape = Array.isArray(value) ? value.map((entry) => Number(entry)) : [];
  if (shape.length !== 2 || shape.some((entry) => !Number.isInteger(entry) || entry < 1)) {
    throw new Error(`${label} requires shape [rows, cols].`);
  }
  return shape;
}

function normalizeTrainerTensor(entry, index) {
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

function assertCausalLmTensorCoverage(tensors, adapter) {
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
    for (const moduleName of targetModules) {
      const kinds = modules.get(moduleName);
      if (!kinds?.has('a') || !kinds?.has('b')) {
        throw new Error(
          `Causal-LM trainer layer ${layerIndex} module ${moduleName} must include both lora_a and lora_b tensors.`
        );
      }
    }
  }
}

function normalizeCausalLmTrainerOutput(output, workload) {
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
    tensors,
  };
}

async function resolveCausalLmTrainer(loadedWorkload, options = {}) {
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

async function exportProviderCausalLmLoraModel(
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

function hasExternalCausalLmTrainer(loadedWorkload, options = {}) {
  return typeof options.causalLmTrainer === 'function'
    || Boolean(getPipelineConfig(loadedWorkload.workload).trainer);
}

async function runInternalCausalLmLoraPipeline(options, layout, compatibility) {
  const loadedWorkload = options.loadedWorkload;
  const workload = loadedWorkload.workload;
  const pipeline = getPipelineConfig(workload);
  const datasetPath = await resolveDatasetPathForLoadedWorkload(workload.datasetPath, loadedWorkload);
  const preflight = await preflightCausalLmLoraWorkload(workload, {
    datasetPath,
    fetch: options.fetch,
    readFile: options.readFile,
  });
  if (!preflight.supported || preflight.blockedReasons.length > 0) {
    throw new Error(`Causal-LM LoRA workload is blocked: ${preflight.blockedReasons.join(',')}`);
  }
  if (Math.floor(Number(workload.training.batchSize)) !== 1) {
    throw new Error('Causal-LM LoRA workload requires training.batchSize=1.');
  }
  const freeze = getCausalLmFreezeConfig(workload);
  const fixture = await createCausalLmLoraFixture(workload);
  try {
    const dataset = await loadCausalLmTextPairSamples(workload, datasetPath, fixture.tokenizer);
    const evalReports = [];
    const checkpointArtifacts = [];
    const exports = [];
    const runner = new TrainingRunner(createLoraRunnerTrainingConfig(workload, freeze), {
      optimizer: createLoraOptimizer(workload),
      crossEntropyLoss,
      clipGradients,
      trainingObjective: createCausalLmTrainingObjective(),
      onCheckpoint: async (checkpoint) => {
        const checkpointId = `checkpoint-${String(checkpoint.step).padStart(6, '0')}`;
        const checkpointPayload = {
          ...buildArtifact(loadedWorkload, {
            prefix: 'lora_ckpt',
            id: checkpointId,
            artifactType: 'training_checkpoint',
            datasetPath: dataset.absolutePath,
            datasetHash: dataset.datasetHash,
            checkpointStep: checkpoint.step,
          }),
          checkpointId,
          checkpointPath: checkpoint.path,
          optimizerStatePresent: true,
          schedulerStatePresent: workload.training.optimizer.scheduler.enabled === true,
          runnerKey: compatibility.observed.runnerKey,
          baseModelRef: fixture.baseModelRef,
          tokenization: dataset.tokenization,
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
        if (pipeline.export?.enabled === true && pipeline.export.atCheckpoints === true) {
          exports.push(await exportCausalLmLoraModel(
            loadedWorkload,
            layout,
            fixture,
            checkpointId,
            checkpoint.step,
            dataset.datasetHash
          ));
        }
        const reports = await evaluateCausalLmLoraModel(workload, fixture, dataset, layout, {
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
            loss: report.loss,
            metrics: {
              loss: report.loss,
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
      createCausalLmDatasetBatches(dataset.samples),
      {
        epochs: 1,
        batchSize: 1,
        shuffle: false,
        maxSteps: workload.training.steps,
        checkpointEvery: workload.checkpointEvery,
        checkpointKey: join(layout.checkpoints, 'latest.state.json'),
        modelId: workload.baseModelId,
        modelUrl: fixture.baseModelUrl,
        tokenizerHash: fixture.baseManifest?.tokenizerHash || null,
      }
    );
    const finalCheckpointId = runner.lastCheckpoint
      ? `checkpoint-${String(runner.lastCheckpoint.step).padStart(6, '0')}`
      : null;
    if (
      pipeline.export?.enabled !== false
      && finalCheckpointId
      && exports.every((entry) => entry.checkpointId !== finalCheckpointId)
    ) {
      exports.push(await exportCausalLmLoraModel(
        loadedWorkload,
        layout,
        fixture,
        finalCheckpointId,
        runner.lastCheckpoint.step,
        dataset.datasetHash
      ));
    }
    return {
      ok: true,
      kind: 'lora',
      action: 'run',
      runnerKind: 'causal_lm_lora',
      workloadId: workload.id,
      runRoot: layout.runRoot,
      preflight,
      checkpointArtifacts,
      evalReports,
      exports,
      metrics,
      lastCheckpoint: runner.lastCheckpoint,
      dataset: {
        path: dataset.absolutePath,
        rowCount: dataset.rowCount,
        sampleCount: dataset.samples.length,
        datasetHash: dataset.datasetHash,
        tokenization: dataset.tokenization,
      },
      baseModel: {
        id: workload.baseModelId,
        ref: fixture.baseModelRef,
        url: fixture.baseModelUrl,
      },
    };
  } finally {
    await fixture.cleanup();
  }
}

async function runProviderCausalLmLoraPipeline(options, compatibility) {
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
    evalReports: [],
    exports: exported ? [exported] : [],
    metrics: trainerOutput.metrics,
    lastCheckpoint: {
      id: trainerOutput.checkpointId,
      step: trainerOutput.checkpointStep,
      path: checkpointPayload.checkpointPath,
    },
  };
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
