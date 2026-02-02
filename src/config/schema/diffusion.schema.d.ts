/**
 * Diffusion Pipeline Config Schema
 *
 * Runtime tunables for diffusion pipelines.
 *
 * @module config/schema/diffusion
 */

export type DiffusionSchedulerType = 'ddim' | 'euler' | 'euler_a' | 'dpmpp_2m';

export type DiffusionDtype = 'f16' | 'f32';

export type DiffusionQuantDtype = 'none' | 'int8';

export type DiffusionSwapperStrategy = 'sequential';

export type DiffusionBackendPipeline = 'cpu' | 'gpu_scaffold' | 'gpu';

export interface DiffusionSchedulerConfigSchema {
  type: DiffusionSchedulerType;
  numSteps: number;
  guidanceScale: number;
  eta: number;
}

export interface DiffusionLatentConfigSchema {
  width: number;
  height: number;
  scale: number;
  channels: number;
  dtype: DiffusionDtype;
}

export interface DiffusionTextEncoderConfigSchema {
  maxLength: number;
}

export interface DiffusionDecodeConfigSchema {
  outputDtype: DiffusionDtype;
  groupNormEps: number;
  tiling: DiffusionTilingConfigSchema;
}

export interface DiffusionTilingConfigSchema {
  enabled: boolean;
  tileSize: number;
  overlap: number;
}

export interface DiffusionSwapperConfigSchema {
  enabled: boolean;
  strategy: DiffusionSwapperStrategy;
  evictTextEncoder: boolean;
  evictUnet: boolean;
}

export interface DiffusionQuantizationConfigSchema {
  weightDtype: DiffusionQuantDtype;
  dequantize: 'shader' | 'cpu';
}

export interface DiffusionScaffoldConfigSchema {
  tokens: number;
  hiddenSize: number;
  numHeads: number;
  seed?: number;
  layerNormEps: number;
}

export interface DiffusionBackendConfigSchema {
  pipeline: DiffusionBackendPipeline;
  scaffold: DiffusionScaffoldConfigSchema;
}

export interface DiffusionConfigSchema {
  scheduler: DiffusionSchedulerConfigSchema;
  latent: DiffusionLatentConfigSchema;
  textEncoder: DiffusionTextEncoderConfigSchema;
  decode: DiffusionDecodeConfigSchema;
  swapper: DiffusionSwapperConfigSchema;
  quantization: DiffusionQuantizationConfigSchema;
  backend: DiffusionBackendConfigSchema;
}

export declare const DEFAULT_DIFFUSION_SCHEDULER_CONFIG: DiffusionSchedulerConfigSchema;
export declare const DEFAULT_DIFFUSION_LATENT_CONFIG: DiffusionLatentConfigSchema;
export declare const DEFAULT_DIFFUSION_TEXT_ENCODER_CONFIG: DiffusionTextEncoderConfigSchema;
export declare const DEFAULT_DIFFUSION_DECODE_CONFIG: DiffusionDecodeConfigSchema;
export declare const DEFAULT_DIFFUSION_TILING_CONFIG: DiffusionTilingConfigSchema;
export declare const DEFAULT_DIFFUSION_SWAPPER_CONFIG: DiffusionSwapperConfigSchema;
export declare const DEFAULT_DIFFUSION_QUANTIZATION_CONFIG: DiffusionQuantizationConfigSchema;
export declare const DEFAULT_DIFFUSION_BACKEND_CONFIG: DiffusionBackendConfigSchema;
export declare const DEFAULT_DIFFUSION_CONFIG: DiffusionConfigSchema;
