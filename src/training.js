import {
  LORA_RUNNER_BASE_MODEL_REGISTRY,
  LORA_RUNNER_DATASET_FORMAT_REGISTRY,
  LORA_RUNNER_SUPPORT_CONTRACT,
  compareLoraRun,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  getLoraRunnerCompatibility,
  qualityGateLoraRun,
  runLoraPipeline,
  watchLoraCheckpoints,
} from './experimental/training/lora-pipeline.js';
import {
  loadTrainingWorkloadPack,
  normalizeTrainingWorkloadPack,
  serializeTrainingWorkloadLock,
} from './experimental/training/workloads.js';

export { AutogradTape, OpType } from './experimental/training/autograd.js';
export { buildAttentionSoftmaxCache } from './experimental/training/attention-backward.js';
export { recordAttentionForward } from './experimental/training/attention-forward.js';
export { LoraAdapter } from './experimental/training/lora.js';
export { AdamOptimizer } from './experimental/training/optimizer.js';
export { trainStep } from './experimental/training/trainer.js';
export { crossEntropyLoss } from './experimental/training/loss.js';
export { clipGradients } from './experimental/training/clip.js';
export { exportLoRAAdapter, serializeLoRASafetensors } from './experimental/training/export.js';
export { DynamicLossScaler, detectOverflow } from './experimental/training/loss-scaling.js';
export { TrainingRunner, runTraining } from './experimental/training/runner.js';
export { DataLoader } from './experimental/training/dataloader.js';
export { saveCheckpoint, loadCheckpoint } from './experimental/training/checkpoint.js';

export {
  LORA_RUNNER_BASE_MODEL_REGISTRY,
  LORA_RUNNER_DATASET_FORMAT_REGISTRY,
  LORA_RUNNER_SUPPORT_CONTRACT,
  compareLoraRun,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  getLoraRunnerCompatibility,
  loadTrainingWorkloadPack,
  normalizeTrainingWorkloadPack,
  qualityGateLoraRun,
  serializeTrainingWorkloadLock,
  watchLoraCheckpoints,
};

export const TRAINING_BACKENDS = Object.freeze(['webgpu_native', 'external']);

function getWorkloadPipeline(workload) {
  return workload?.pipeline || workload?.lora || {};
}

function buildBackendCapability(id, supported, blockedReasons) {
  return Object.freeze({
    id,
    supported,
    blockedReasons: Object.freeze(supported ? [] : [...new Set(blockedReasons)]),
  });
}

export function getTrainingCapabilities(workload) {
  const compatibility = getLoraRunnerCompatibility(workload);
  const pipeline = getWorkloadPipeline(workload);
  const blockedReasons = [...compatibility.blockedReasons];
  if (workload?.kind !== 'lora') {
    blockedReasons.push('workload_kind_must_be_lora');
  }
  if (pipeline.datasetFormat !== 'text-pairs') {
    blockedReasons.push('dataset_format_must_be_text_pairs');
  }
  if (pipeline.taskType !== 'text_generation') {
    blockedReasons.push('task_type_must_be_text_generation');
  }

  const sftLoraSupported = blockedReasons.length === 0;
  const nativeBlockedReasons = [...blockedReasons];
  if (compatibility.observed.requiresExternalTrainer) {
    nativeBlockedReasons.push('native_full_graph_runner_unavailable_for_base_model');
  }
  const webgpuNativeSupported = nativeBlockedReasons.length === 0;

  return Object.freeze({
    schemaVersion: 1,
    scope: 'completion_masked_sft_lora',
    supported: sftLoraSupported,
    operatorSurface: 'node',
    runnerKey: compatibility.observed.runnerKey,
    baseModelId: compatibility.observed.baseModelId,
    baseModelFamily: compatibility.observed.baseModelFamily,
    datasetFormat: compatibility.observed.datasetFormat,
    taskType: compatibility.observed.taskType,
    backends: Object.freeze({
      webgpuNative: buildBackendCapability(
        'webgpu_native',
        webgpuNativeSupported,
        nativeBlockedReasons
      ),
      external: buildBackendCapability('external', sftLoraSupported, blockedReasons),
    }),
    adapterExport: Object.freeze({
      format: 'safetensors',
      manifest: 'rdrr_lora_adapter',
      runtimeLoadable: true,
    }),
    compatibility,
    blockedReasons: Object.freeze([...new Set(blockedReasons)]),
  });
}

export function assertTrainingBackend(workload, backend) {
  if (!TRAINING_BACKENDS.includes(backend)) {
    throw new Error(`training_backend_invalid: expected one of ${TRAINING_BACKENDS.join(', ')}, got ${String(backend)}.`);
  }
  const capabilities = getTrainingCapabilities(workload);
  const selected = backend === 'webgpu_native'
    ? capabilities.backends.webgpuNative
    : capabilities.backends.external;
  if (!selected.supported) {
    throw new Error(
      `training_backend_not_supported: backend=${backend} runnerKey=${capabilities.runnerKey} `
      + `blockedReasons=${selected.blockedReasons.join(',')}.`
    );
  }
  return capabilities;
}

export async function trainSftLoRA(options) {
  if (!options || typeof options !== 'object') {
    throw new Error('trainSftLoRA requires an options object.');
  }
  const loadedWorkload = options.loadedWorkload;
  if (!loadedWorkload?.workload) {
    throw new Error('trainSftLoRA requires options.loadedWorkload.');
  }
  const backend = options.backend;
  assertTrainingBackend(loadedWorkload.workload, backend);

  const pipeline = getWorkloadPipeline(loadedWorkload.workload);
  const externalTrainerConfigured = typeof options.causalLmTrainer === 'function'
    || Boolean(pipeline.trainer);
  if (backend === 'external' && !externalTrainerConfigured) {
    throw new Error(
      'external_training_backend_not_configured: provide options.causalLmTrainer or workload.pipeline.trainer.'
    );
  }
  if (backend === 'webgpu_native' && externalTrainerConfigured) {
    throw new Error(
      'training_backend_mismatch: webgpu_native cannot be combined with an external trainer configuration.'
    );
  }

  const { backend: _backend, ...pipelineOptions } = options;
  return runLoraPipeline(pipelineOptions);
}
