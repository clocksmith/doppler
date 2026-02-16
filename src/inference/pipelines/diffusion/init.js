import { DEFAULT_DIFFUSION_CONFIG } from '../../../config/schema/index.js';

function mergeSection(base, override) {
  if (!override) return { ...base };
  return { ...base, ...override };
}

function mergeDecodeConfig(base, override) {
  if (!override) return { ...base, tiling: { ...base.tiling } };
  return {
    ...base,
    ...override,
    tiling: mergeSection(base.tiling || {}, override.tiling || {}),
  };
}

function mergeBackendConfig(base, override) {
  if (!override) return { ...base, scaffold: { ...base.scaffold } };
  return {
    ...base,
    ...override,
    scaffold: mergeSection(base.scaffold || {}, override.scaffold || {}),
  };
}

export function mergeDiffusionConfig(baseConfig, overrideConfig) {
  const base = baseConfig || DEFAULT_DIFFUSION_CONFIG;
  const override = overrideConfig || {};
  return {
    scheduler: mergeSection(base.scheduler, override.scheduler),
    latent: mergeSection(base.latent, override.latent),
    textEncoder: mergeSection(base.textEncoder, override.textEncoder),
    decode: mergeDecodeConfig(base.decode, override.decode),
    swapper: mergeSection(base.swapper, override.swapper),
    quantization: mergeSection(base.quantization, override.quantization),
    backend: mergeBackendConfig(base.backend, override.backend),
  };
}

function resolveSchedulerType(modelScheduler, runtimeScheduler) {
  const modelClass = modelScheduler?._class_name;
  if (modelClass === 'FlowMatchEulerDiscreteScheduler') {
    return 'flowmatch_euler';
  }
  if (modelClass === 'EulerDiscreteScheduler') {
    return 'euler';
  }
  if (modelClass === 'EulerAncestralDiscreteScheduler') {
    return 'euler_a';
  }
  if (modelClass === 'DPMSolverMultistepScheduler') {
    return 'dpmpp_2m';
  }
  return runtimeScheduler?.type || DEFAULT_DIFFUSION_CONFIG.scheduler.type;
}

function mergeSchedulerConfig(modelConfig, runtimeScheduler) {
  const modelScheduler = modelConfig?.components?.scheduler?.config || {};
  const type = resolveSchedulerType(modelScheduler, runtimeScheduler);
  return {
    ...runtimeScheduler,
    type,
    numTrainTimesteps: modelScheduler.num_train_timesteps ?? runtimeScheduler.numTrainTimesteps,
    shift: modelScheduler.shift ?? runtimeScheduler.shift,
  };
}

function resolveLatentScale(modelConfig, runtimeConfig) {
  const transformerSize = modelConfig?.components?.transformer?.config?.sample_size;
  const vaeSize = modelConfig?.components?.vae?.config?.sample_size;
  if (Number.isFinite(transformerSize) && Number.isFinite(vaeSize) && transformerSize > 0) {
    const ratio = vaeSize / transformerSize;
    if (Number.isFinite(ratio) && ratio > 0) {
      return ratio;
    }
  }
  const runtimeScale = runtimeConfig?.latent?.scale;
  if (Number.isFinite(runtimeScale) && runtimeScale > 0) return runtimeScale;
  return DEFAULT_DIFFUSION_CONFIG.latent.scale;
}

function resolveLatentChannels(modelConfig, runtimeConfig) {
  const vaeChannels = modelConfig?.components?.vae?.config?.latent_channels;
  if (Number.isFinite(vaeChannels) && vaeChannels > 0) return vaeChannels;
  const runtimeChannels = runtimeConfig?.latent?.channels;
  if (Number.isFinite(runtimeChannels) && runtimeChannels > 0) return runtimeChannels;
  return DEFAULT_DIFFUSION_CONFIG.latent.channels;
}

export function initializeDiffusion(manifest, runtimeConfig) {
  const modelConfig = manifest?.config?.diffusion;
  if (!modelConfig) {
    throw new Error('Diffusion manifest missing config.diffusion.');
  }

  const runtimeBase = mergeDiffusionConfig(DEFAULT_DIFFUSION_CONFIG, runtimeConfig?.inference?.diffusion);
  const runtime = {
    ...runtimeBase,
    scheduler: mergeSchedulerConfig(modelConfig, runtimeBase.scheduler),
  };
  if (modelConfig?.components?.transformer && runtime.backend?.pipeline === 'cpu') {
    runtime.backend = { ...runtime.backend, pipeline: 'gpu' };
  }
  const latentScale = resolveLatentScale(modelConfig, runtime);
  const latentChannels = resolveLatentChannels(modelConfig, runtime);

  return {
    modelConfig,
    runtime,
    latentScale,
    latentChannels,
  };
}
