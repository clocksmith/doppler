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

export const CAUSAL_LM_TEXT_PAIR_RUNNER_KEYS = Object.freeze([
  'gemma-3-270m-it-f16-af32::text-pairs::text_generation',
  'gemma-3-270m-it-q4k-ehf16-af32::text-pairs::text_generation',
  'gemma4-e2b-it::text-pairs::text_generation',
  'gemma-4-e2b-it-q4k-ehf16-af32::text-pairs::text_generation',
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple::text-pairs::text_generation',
  'qwen-3-5-0-8b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-5-2b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-5-9b-hf-bf16::text-pairs::text_generation',
  'qwen-3-6-27b-q4k-ehaf16::text-pairs::text_generation',
  'qwen-3-6-27b-q4k-eaf16::text-pairs::text_generation',
]);

export const LORA_RUNNER_SUPPORT_CONTRACT = Object.freeze({
  supportedBaseModelId: 'training-toy',
  supportedDatasetFormat: 'toy_linear_classification_jsonl',
  registeredBaseModelIds: Object.freeze([
    'training-toy',
    'gemma-3-270m-it-f16-af32',
    'gemma-3-270m-it-q4k-ehf16-af32',
    'gemma4-e2b-it',
    'gemma-4-e2b-it-q4k-ehf16-af32',
    'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    'qwen-3-5-0-8b-q4k-ehaf16',
    'qwen-3-5-2b-q4k-ehaf16',
    'qwen-3-5-9b-hf-bf16',
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
  'gemma-3-270m-it-f16-af32': Object.freeze({
    baseModelId: 'gemma-3-270m-it-f16-af32',
    modelRef: 'gemma-3-270m-it-f16-af32',
    family: 'gemma3',
    runnerKind: 'causal_lm_text_generation',
  }),
  'gemma-3-270m-it-q4k-ehf16-af32': Object.freeze({
    baseModelId: 'gemma-3-270m-it-q4k-ehf16-af32',
    modelRef: 'gemma-3-270m-it-q4k-ehf16-af32',
    family: 'gemma3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'gemma4-e2b-it': Object.freeze({
    baseModelId: 'gemma4-e2b-it',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'gemma-4-e2b-it-q4k-ehf16-af32': Object.freeze({
    baseModelId: 'gemma-4-e2b-it-q4k-ehf16-af32',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'gemma-4-e2b-it-q4k-ehf16-af32-int4ple': Object.freeze({
    baseModelId: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    modelRef: 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple',
    family: 'gemma4',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'qwen-3-5-0-8b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    modelRef: 'qwen-3-5-0-8b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
    nativeLoraTargets: Object.freeze([
      Object.freeze({ module: 'down_proj', layer: 'last' }),
    ]),
  }),
  'qwen-3-5-2b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-5-2b-q4k-ehaf16',
    modelRef: 'qwen-3-5-2b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'qwen-3-5-9b-hf-bf16': Object.freeze({
    baseModelId: 'qwen-3-5-9b-hf-bf16',
    modelRef: 'Qwen/Qwen3.5-9B',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'qwen-3-6-27b-q4k-ehaf16': Object.freeze({
    baseModelId: 'qwen-3-6-27b-q4k-ehaf16',
    modelRef: 'qwen-3-6-27b-q4k-ehaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
  }),
  'qwen-3-6-27b-q4k-eaf16': Object.freeze({
    baseModelId: 'qwen-3-6-27b-q4k-eaf16',
    modelRef: 'qwen-3-6-27b-q4k-eaf16',
    family: 'qwen3',
    runnerKind: 'causal_lm_text_generation',
    requiresExternalTrainer: true,
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

export function getPipelineConfig(workload) {
  return workload?.pipeline || workload?.lora || {};
}

export function normalizeLoraTargetModules(adapter) {
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

export function getRunnerKey(baseModelId, datasetFormat, taskType) {
  return `${baseModelId}::${datasetFormat}::${taskType}`;
}

export function isCausalLmLoraWorkload(workload, compatibility = getLoraRunnerCompatibility(workload)) {
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
      requiresExternalTrainer: baseModel?.requiresExternalTrainer === true,
      nativeLoraTargets: baseModel?.nativeLoraTargets || Object.freeze([]),
      datasetKind: dataset?.datasetKind || null,
      registeredBaseModel: Boolean(baseModel),
      registeredDatasetFormat: Boolean(dataset),
    },
    blockedReasons,
  };
}

export function summarizeTextPairLengths(rows) {
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

export function finiteMetric(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function isObjectRecord(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

export function normalizeProviderEvalReport(entry, index, context) {
  if (!isObjectRecord(entry)) {
    throw new Error(`Causal-LM trainer evalReports[${index}] must be an object.`);
  }
  const evalDatasetId = String(entry.evalDatasetId || entry.id || '').trim();
  if (!evalDatasetId) {
    throw new Error(`Causal-LM trainer evalReports[${index}].evalDatasetId is required.`);
  }
  const primaryMetric = String(entry.primaryMetric || 'loss').trim();
  if (!primaryMetric) {
    throw new Error(`Causal-LM trainer evalReports[${index}].primaryMetric is required.`);
  }
  const primaryScore = finiteMetric(entry.primaryScore ?? entry.score ?? entry.loss);
  if (primaryScore === null) {
    throw new Error(`Causal-LM trainer evalReports[${index}].primaryScore must be finite.`);
  }
  const loss = finiteMetric(entry.loss ?? (primaryMetric === 'loss' ? primaryScore : null));
  const baselineScore = finiteMetric(entry.baseline?.primaryScore ?? entry.baseline?.loss);
  const qualityClaim = isObjectRecord(entry.qualityClaim)
    ? entry.qualityClaim
    : (baselineScore === null || loss === null
      ? null
      : {
        baseline: entry.baseline?.stage || 'base_model',
        metric: 'loss',
        selectionGoal: 'min',
        adapterScore: loss,
        baselineScore,
        delta: loss - baselineScore,
        absoluteImprovement: baselineScore - loss,
        relativeImprovement: baselineScore === 0 ? 0 : (baselineScore - loss) / Math.abs(baselineScore),
        minAbsoluteImprovement: 0,
        minRelativeImprovement: 0,
        improved: loss <= baselineScore,
        requireImprovement: false,
      });
  const agentEval = isObjectRecord(entry.agentEval)
    ? entry.agentEval
    : (isObjectRecord(entry.heldoutGate) ? entry.heldoutGate : null);
  return {
    artifactType: 'training_eval_report',
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    workloadId: context.workload.id,
    workloadPath: context.loadedWorkload.absolutePath || null,
    workloadSha256: context.loadedWorkload.workloadSha256 || null,
    configHash: context.workload.configHash,
    datasetPath: entry.datasetPath || context.dataset.absolutePath,
    datasetHash: entry.datasetHash || context.datasetHash,
    baseModelId: context.workload.baseModelId,
    baseModelRef: context.preflight.baseModelId,
    stage: entry.stage || 'lora',
    checkpointId: context.checkpointId,
    checkpointStep: context.checkpointStep,
    evalDatasetId,
    metrics: isObjectRecord(entry.metrics)
      ? entry.metrics
      : {
        [primaryMetric]: {
          score: primaryScore,
          samples: finiteMetric(entry.samples) ?? context.dataset.rowCount,
        },
      },
    primaryMetric,
    primaryScore,
    loss,
    baseline: isObjectRecord(entry.baseline) ? entry.baseline : null,
    qualityClaim,
    agentEval,
  };
}
