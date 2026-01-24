

// Core conversion logic
export {
  SHARD_SIZE,
  RDRR_VERSION,
  ConvertStage,
  sanitizeModelId,
  formatBytes,
  shouldQuantize,
  extractArchitecture,
  buildTensorMap,
  createManifest,
  convertModel,
  generateShardFilename,
} from './core.js';

// RDRR Writer

// Quantization
export {
  float32ToFloat16,
  float16ToFloat32,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  transposeF32,
  getQ4KSize,
  dequantizeQ4KM,
  calculateQuantizationError,
  quantizeF16ToQ4KM,
  shouldQuantize as shouldQuantizeTensor,
  getQuantizedSize,
  QK_K,
  QK4_K_BLOCK_SIZE,
} from './quantizer.js';

// Shard Packer
export {
  ShardPacker,
  sortTensorsByGroup,
  estimateShardCount,
} from './shard-packer.js';
