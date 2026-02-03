/**
 * Diffusion Pipeline Types (scaffold)
 *
 * @module inference/diffusion/types
 */

export interface DiffusionRequest {
  prompt: string;
  negativePrompt?: string;
  seed?: number;
  steps?: number;
  guidanceScale?: number;
  width?: number;
  height?: number;
}

export interface DiffusionResult {
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
}

export interface DiffusionSchedulerConfig {
  type: string;
  numSteps: number;
  guidanceScale: number;
  eta: number;
  numTrainTimesteps: number;
  shift: number;
}

export interface DiffusionLatentConfig {
  width: number;
  height: number;
  scale: number;
  channels: number;
  dtype: string;
}

export interface DiffusionTextEncoderConfig {
  maxLength: number;
}

export interface DiffusionDecodeConfig {
  outputDtype: string;
  groupNormEps: number;
  tiling: {
    enabled: boolean;
    tileSize: number;
    overlap: number;
  };
}

export interface DiffusionSwapperConfig {
  enabled: boolean;
  strategy: string;
  evictTextEncoder: boolean;
  evictUnet: boolean;
}

export interface DiffusionQuantizationConfig {
  weightDtype: string;
  dequantize: string;
}

export interface DiffusionBackendConfig {
  pipeline: string;
  scaffold: {
    tokens: number;
    hiddenSize: number;
    numHeads: number;
    seed?: number;
    layerNormEps: number;
  };
}

export interface DiffusionRuntimeConfig {
  scheduler: DiffusionSchedulerConfig;
  latent: DiffusionLatentConfig;
  textEncoder: DiffusionTextEncoderConfig;
  decode: DiffusionDecodeConfig;
  swapper: DiffusionSwapperConfig;
  quantization: DiffusionQuantizationConfig;
  backend: DiffusionBackendConfig;
}

export interface DiffusionTokenizerConfig {
  type: string;
  vocabFile?: string;
  mergesFile?: string;
  tokenizerFile?: string;
  spieceFile?: string;
  configFile?: string;
  specialTokensFile?: string;
}

export interface DiffusionModelConfig {
  modelIndex?: Record<string, unknown>;
  components?: Record<string, { config?: Record<string, unknown> }>;
  tokenizers?: Record<string, DiffusionTokenizerConfig>;
}

export interface DiffusionStats {
  totalTimeMs?: number;
  prefillTimeMs?: number;
  prefillTokens?: number;
  decodeTimeMs?: number;
  decodeTokens?: number;
}
