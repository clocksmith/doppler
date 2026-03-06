import { DEFAULT_DIFFUSION_CONFIG } from '../../../config/schema/index.js';

const SUPPORTED_DIFFUSION_RUNTIME_LAYOUTS = new Set(['sd3', 'flux', 'sana']);

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
  if (!override) return { ...base };
  return { ...base, ...override };
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
  if (modelClass === 'SCMScheduler') {
    return 'scm';
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
  return runtimeScheduler?.type;
}

function mergeSchedulerConfig(modelConfig, runtimeScheduler) {
  const modelScheduler = modelConfig?.components?.scheduler?.config || {};
  const type = resolveSchedulerType(modelScheduler, runtimeScheduler);
  return {
    ...runtimeScheduler,
    type,
    numTrainTimesteps: modelScheduler.num_train_timesteps ?? runtimeScheduler.numTrainTimesteps,
    shift: modelScheduler.shift ?? runtimeScheduler.shift,
    predictionType: modelScheduler.prediction_type ?? runtimeScheduler.predictionType,
    sigmaData: modelScheduler.sigma_data ?? runtimeScheduler.sigmaData,
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
  throw new Error('Diffusion runtime requires a valid latent scale.');
}

function resolveLatentChannels(modelConfig, runtimeConfig) {
  const vaeChannels = modelConfig?.components?.vae?.config?.latent_channels;
  if (Number.isFinite(vaeChannels) && vaeChannels > 0) return vaeChannels;
  const runtimeChannels = runtimeConfig?.latent?.channels;
  if (Number.isFinite(runtimeChannels) && runtimeChannels > 0) return runtimeChannels;
  throw new Error('Diffusion runtime requires valid latent channels.');
}

export function initializeDiffusion(manifest, runtimeConfig) {
  const modelConfig = manifest?.config?.diffusion;
  if (!modelConfig) {
    const hasInferenceDiffusion = manifest?.inference?.diffusion && typeof manifest.inference.diffusion === 'object';
    if (hasInferenceDiffusion) {
      throw new Error(
        'Diffusion manifest provides inference.diffusion, but runtime expects config.diffusion model contract. ' +
        'Re-convert the model with config.diffusion populated.'
      );
    }
    throw new Error('Diffusion manifest missing config.diffusion model contract.');
  }
  const layout = modelConfig.layout;
  if (layout && !SUPPORTED_DIFFUSION_RUNTIME_LAYOUTS.has(layout)) {
    throw new Error(
      `Diffusion layout "${layout}" is recognized in the manifest, but the GPU runtime is not implemented yet. ` +
      'Supported runtime layouts: sd3, flux, sana.'
    );
  }

  const runtimeBase = mergeDiffusionConfig(DEFAULT_DIFFUSION_CONFIG, runtimeConfig?.inference?.diffusion);
  const runtime = {
    ...runtimeBase,
    scheduler: mergeSchedulerConfig(modelConfig, runtimeBase.scheduler),
  };
  if (runtime.backend?.pipeline !== 'gpu') {
    throw new Error(
      `Diffusion runtime backend.pipeline must be "gpu"; got "${runtime.backend?.pipeline}".`
    );
  }
  const transformerConfig = modelConfig?.components?.transformer?.config || null;
  if (!transformerConfig) {
    throw new Error(
      'Diffusion runtime requires manifest.config.diffusion.components.transformer.config for GPU execution.'
    );
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
