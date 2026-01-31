export const DEFAULT_DIFFUSION_SCHEDULER_CONFIG = {
  type: 'ddim',
  numSteps: 20,
  guidanceScale: 7.5,
  eta: 0.0,
};

export const DEFAULT_DIFFUSION_LATENT_CONFIG = {
  width: 512,
  height: 512,
  scale: 8,
  channels: 4,
  dtype: 'f16',
};

export const DEFAULT_DIFFUSION_TEXT_ENCODER_CONFIG = {
  maxLength: 77,
};

export const DEFAULT_DIFFUSION_DECODE_CONFIG = {
  outputDtype: 'f16',
};

export const DEFAULT_DIFFUSION_TILING_CONFIG = {
  enabled: false,
  tileSize: 64,
  overlap: 8,
};

export const DEFAULT_DIFFUSION_SWAPPER_CONFIG = {
  enabled: false,
  strategy: 'sequential',
  evictTextEncoder: true,
  evictUnet: true,
};

export const DEFAULT_DIFFUSION_QUANTIZATION_CONFIG = {
  weightDtype: 'none',
  dequantize: 'shader',
};

export const DEFAULT_DIFFUSION_CONFIG = {
  scheduler: DEFAULT_DIFFUSION_SCHEDULER_CONFIG,
  latent: DEFAULT_DIFFUSION_LATENT_CONFIG,
  textEncoder: DEFAULT_DIFFUSION_TEXT_ENCODER_CONFIG,
  decode: {
    ...DEFAULT_DIFFUSION_DECODE_CONFIG,
    tiling: DEFAULT_DIFFUSION_TILING_CONFIG,
  },
  swapper: DEFAULT_DIFFUSION_SWAPPER_CONFIG,
  quantization: DEFAULT_DIFFUSION_QUANTIZATION_CONFIG,
};
