import { DEFAULT_DIFFUSION_CONFIG } from '../../config/schema/index.js';

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

  const runtime = mergeDiffusionConfig(runtimeConfig?.inference?.diffusion, null);
  const latentScale = resolveLatentScale(modelConfig, runtime);
  const latentChannels = resolveLatentChannels(modelConfig, runtime);

  return {
    modelConfig,
    runtime,
    latentScale,
    latentChannels,
  };
}
