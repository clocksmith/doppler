import { canonicalizeLegacyEnumValue, canonicalizeLegacyEnumValues } from './config/legacy-enum-values.js';
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
  loadTrainingWorkloadCapsule,
  normalizeTrainingWorkloadCapsule,
  serializeTrainingWorkloadLock,
} from './experimental/training/workloads.js';
import { trainNativeQwenSftLoRA } from './experimental/training/native-qwen-lora.js';

export { AutogradTape, OpType } from './experimental/training/autograd.js';
export { buildAttentionSoftmaxCache } from './experimental/training/attention-backward.js';
export { recordAttentionForward } from './experimental/training/attention-forward.js';
export { LoraAdapter } from './experimental/training/lora.js';
export { AdamOptimizer } from './experimental/training/optimizer.js';
export { trainStep } from './experimental/training/trainer.js';
export { crossEntropyLoss } from './experimental/training/loss.js';
export { clipGradients } from './experimental/training/clip.js';
export { exportLoRAAdapter, serializeLoRASafetensors } from './experimental/training/export.js';
export { loadLoRAFromManifest } from './experimental/adapters/lora-loader.js';
export { DynamicLossScaler, detectOverflow } from './experimental/training/loss-scaling.js';
export { TrainingRunner, runTraining } from './experimental/training/runner.js';
export { DataLoader } from './experimental/training/dataloader.js';
export { saveCheckpoint, loadCheckpoint } from './experimental/training/checkpoint.js';
export {
  NativeQwenLoRATrainer,
  createNativeQwenLoRATrainer,
  loadNativeQwenTrainingPipeline,
  trainNativeQwenSftLoRA,
} from './experimental/training/native-qwen-lora.js';

export {
  LORA_RUNNER_BASE_MODEL_REGISTRY,
  LORA_RUNNER_DATASET_FORMAT_REGISTRY,
  LORA_RUNNER_SUPPORT_CONTRACT,
  compareLoraRun,
  evaluateLoraCheckpoint,
  exportLoraCheckpoint,
  getLoraRunnerCompatibility,
  loadTrainingWorkloadCapsule,
  normalizeTrainingWorkloadCapsule,
  qualityGateLoraRun,
  serializeTrainingWorkloadLock,
  watchLoraCheckpoints,
};

export const TRAINING_BACKENDS = Object.freeze(['webgpu-native', 'external']);

export async function bootstrapNativeTrainingHost() {
  const [{ installNodeFileFetchShim }, { bootstrapNodeWebGPU }] = await Promise.all([
    import('./tooling/node-file-fetch.js'),
    import('./tooling/node-webgpu.js'),
  ]);
  installNodeFileFetchShim();
  const result = await bootstrapNodeWebGPU();
  if (!result.ok) {
    throw new Error(`native-training-webgpu-bootstrap-failed: ${result.detail || 'no WebGPU provider'}.`);
  }
  return result;
}

export async function releaseNativeTrainingHost() {
  const [{ destroyDevice, resetDeviceState }, { releaseNodeWebGPU }] = await Promise.all([
    import('./gpu/device.js'),
    import('./tooling/node-webgpu.js'),
  ]);
  destroyDevice();
  resetDeviceState();
  return releaseNodeWebGPU();
}

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

function resolveNativeLoraTarget(pipeline, compatibility) {
  const requested = pipeline.nativeTarget;
  const modules = Array.isArray(pipeline.adapter?.targetModules)
    ? pipeline.adapter.targetModules.map((value) => String(value))
    : [];
  if (!requested || modules.length !== 1 || modules[0] !== requested.module) return null;
  const supported = compatibility.observed.nativeLoraTargets.find((target) => (
    target.module === requested.module && target.layer === requested.layer
  ));
  return supported ? Object.freeze({ ...supported }) : null;
}

export function getTrainingCapabilities(input) {
  // Workloads written before 0.2.0 (taskType text_generation, ...) are read throughout 0.x.
  const workload = canonicalizeLegacyEnumValues(input);
  const compatibility = getLoraRunnerCompatibility(workload);
  const pipeline = getWorkloadPipeline(workload);
  const blockedReasons = [...compatibility.blockedReasons];
  if (workload?.kind !== 'lora') {
    blockedReasons.push('workload-kind-must-be-lora');
  }
  if (pipeline.datasetFormat !== 'text-pairs') {
    blockedReasons.push('dataset-format-must-be-text-pairs');
  }
  if (pipeline.taskType !== 'text-generation') {
    blockedReasons.push('task-type-must-be-text-generation');
  }

  const sftLoraSupported = blockedReasons.length === 0;
  const nativeBlockedReasons = [...blockedReasons];
  const nativeTarget = resolveNativeLoraTarget(pipeline, compatibility);
  if (compatibility.observed.requiresExternalTrainer && !nativeTarget) {
    nativeBlockedReasons.push('native-full-graph-runner-unavailable-for-base-model');
  }
  const webgpuNativeSupported = nativeBlockedReasons.length === 0;

  return Object.freeze({
    schemaVersion: 1,
    scope: 'completion-masked-sft-lora',
    supported: sftLoraSupported,
    operatorSurfaces: Object.freeze(['browser', 'node', 'bun']),
    runnerKey: compatibility.observed.runnerKey,
    baseModelId: compatibility.observed.baseModelId,
    baseModelFamily: compatibility.observed.baseModelFamily,
    datasetFormat: compatibility.observed.datasetFormat,
    taskType: compatibility.observed.taskType,
    backends: Object.freeze({
      webgpuNative: buildBackendCapability(
        'webgpu-native',
        webgpuNativeSupported,
        nativeBlockedReasons
      ),
      external: buildBackendCapability('external', sftLoraSupported, blockedReasons),
    }),
    adapterExport: Object.freeze({
      format: 'safetensors',
      manifest: 'rdrr-lora-adapter',
      runtimeLoadable: true,
    }),
    compatibility,
    nativeTarget,
    blockedReasons: Object.freeze([...new Set(blockedReasons)]),
  });
}

export function assertTrainingBackend(workload, requestedBackend) {
  // The legacy spelling webgpu_native is read throughout 0.x.
  const backend = canonicalizeLegacyEnumValue(requestedBackend);
  if (!TRAINING_BACKENDS.includes(backend)) {
    throw new Error(`training-backend-invalid: expected one of ${TRAINING_BACKENDS.join(', ')}, got ${String(backend)}.`);
  }
  const capabilities = getTrainingCapabilities(workload);
  const selected = backend === 'webgpu-native'
    ? capabilities.backends.webgpuNative
    : capabilities.backends.external;
  if (!selected.supported) {
    throw new Error(
      `training-backend-not-supported: backend=${backend} runnerKey=${capabilities.runnerKey} `
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
  const backend = canonicalizeLegacyEnumValue(options.backend);
  const capabilities = assertTrainingBackend(loadedWorkload.workload, backend);

  const pipeline = getWorkloadPipeline(loadedWorkload.workload);
  const externalTrainerConfigured = typeof options.causalLmTrainer === 'function'
    || Boolean(pipeline.trainer);
  if (backend === 'external' && !externalTrainerConfigured) {
    throw new Error(
      'external-training-backend-not-configured: provide options.causalLmTrainer or workload.pipeline.trainer.'
    );
  }
  if (backend === 'webgpu-native' && externalTrainerConfigured) {
    throw new Error(
      'training-backend-mismatch: webgpu-native cannot be combined with an external trainer configuration.'
    );
  }

  if (backend === 'webgpu-native' && capabilities.nativeTarget) {
    if (!options.pipeline) {
      throw new Error('native-qwen-lora-pipeline-required: provide the loaded Doppler pipeline.');
    }
    return trainNativeQwenSftLoRA({
      pipeline: options.pipeline,
      baseModelId: capabilities.baseModelId,
      layerIdx: options.pipeline.modelConfig.numLayers - 1,
      module: capabilities.nativeTarget.module,
      rank: pipeline.adapter.rank,
      alpha: pipeline.adapter.alpha,
      optimizer: loadedWorkload.workload.training.optimizer,
      gradient: loadedWorkload.workload.training.gradientClipping,
      precision: loadedWorkload.workload.training.precision,
      samples: options.samples,
      maxSteps: loadedWorkload.workload.training.steps,
      export: options.export || null,
    });
  }

  const { backend: _backend, ...pipelineOptions } = options;
  return runLoraPipeline(pipelineOptions);
}
