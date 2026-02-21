import { SHARD_SIZE } from './manifest.schema.js';
import { DEFAULT_QUANTIZATION_DEFAULTS } from './quantization-defaults.schema.js';
import { MB } from './units.schema.js';

// =============================================================================
// Converter Config
// =============================================================================

export const DEFAULT_CONVERTER_QUANTIZATION_CONFIG = {
  weights: null,
  embeddings: null,
  lmHead: null,
  vision: DEFAULT_QUANTIZATION_DEFAULTS.visionDtype,
  audio: DEFAULT_QUANTIZATION_DEFAULTS.audioDtype,
  projector: DEFAULT_QUANTIZATION_DEFAULTS.projectorDtype,
  // Q4K layout: 'row' (fused kernel compatible, fast) or 'col' (dequant fallback)
  q4kLayout: 'row',
  computePrecision: 'f16',
};

export const DEFAULT_CONVERTER_SHARDING_CONFIG = {
  shardSizeBytes: SHARD_SIZE,
};

export const DEFAULT_CONVERTER_STREAMING_CONFIG = {
  chunkSizeBytes: 64 * MB,
};

export const DEFAULT_CONVERTER_HTTP_CONFIG = {
  allowDownloadFallback: true,
  maxDownloadBytes: null,
};

export const DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG = {
  transposeWeights: false,
  fuseGateUp: false,
};

export const DEFAULT_CONVERTER_MANIFEST_CONFIG = {
  hashAlgorithm: 'blake3',
  optimizations: null,
  conversion: null,
};

export const DEFAULT_CONVERTER_OUTPUT_CONFIG = {
  modelId: null,
  dir: null,
  baseDir: null,
  textOnly: false,
  fast: false,
};

export const DEFAULT_CONVERTER_PRESET_CONFIG = {
  model: null,
};

export const DEFAULT_CONVERTER_CONFIG = {
  quantization: DEFAULT_CONVERTER_QUANTIZATION_CONFIG,
  sharding: DEFAULT_CONVERTER_SHARDING_CONFIG,
  streaming: DEFAULT_CONVERTER_STREAMING_CONFIG,
  http: DEFAULT_CONVERTER_HTTP_CONFIG,
  weightLayout: DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG,
  manifest: DEFAULT_CONVERTER_MANIFEST_CONFIG,
  output: DEFAULT_CONVERTER_OUTPUT_CONFIG,
  presets: DEFAULT_CONVERTER_PRESET_CONFIG,
};

export function createConverterConfig(overrides) {
  if (!overrides) {
    return {
      quantization: { ...DEFAULT_CONVERTER_QUANTIZATION_CONFIG },
      sharding: { ...DEFAULT_CONVERTER_SHARDING_CONFIG },
      streaming: { ...DEFAULT_CONVERTER_STREAMING_CONFIG },
      http: { ...DEFAULT_CONVERTER_HTTP_CONFIG },
      weightLayout: { ...DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG },
      manifest: { ...DEFAULT_CONVERTER_MANIFEST_CONFIG },
      output: { ...DEFAULT_CONVERTER_OUTPUT_CONFIG },
      presets: { ...DEFAULT_CONVERTER_PRESET_CONFIG },
    };
  }

  return {
    quantization: overrides.quantization
      ? { ...DEFAULT_CONVERTER_QUANTIZATION_CONFIG, ...overrides.quantization }
      : { ...DEFAULT_CONVERTER_QUANTIZATION_CONFIG },
    sharding: overrides.sharding
      ? { ...DEFAULT_CONVERTER_SHARDING_CONFIG, ...overrides.sharding }
      : { ...DEFAULT_CONVERTER_SHARDING_CONFIG },
    streaming: overrides.streaming
      ? { ...DEFAULT_CONVERTER_STREAMING_CONFIG, ...overrides.streaming }
      : { ...DEFAULT_CONVERTER_STREAMING_CONFIG },
    http: overrides.http
      ? { ...DEFAULT_CONVERTER_HTTP_CONFIG, ...overrides.http }
      : { ...DEFAULT_CONVERTER_HTTP_CONFIG },
    weightLayout: overrides.weightLayout
      ? { ...DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG, ...overrides.weightLayout }
      : { ...DEFAULT_CONVERTER_WEIGHT_LAYOUT_CONFIG },
    manifest: overrides.manifest
      ? { ...DEFAULT_CONVERTER_MANIFEST_CONFIG, ...overrides.manifest }
      : { ...DEFAULT_CONVERTER_MANIFEST_CONFIG },
    output: overrides.output
      ? { ...DEFAULT_CONVERTER_OUTPUT_CONFIG, ...overrides.output }
      : { ...DEFAULT_CONVERTER_OUTPUT_CONFIG },
    presets: overrides.presets
      ? { ...DEFAULT_CONVERTER_PRESET_CONFIG, ...overrides.presets }
      : { ...DEFAULT_CONVERTER_PRESET_CONFIG },
  };
}
