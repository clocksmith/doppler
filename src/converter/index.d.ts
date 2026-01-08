/**
 * Converter Module - Public API
 *
 * This module provides model conversion utilities for transforming
 * GGUF and SafeTensors models into the RDRR format used by DOPPLER.
 *
 * @module converter
 */

export {
  type TensorInfo,
  type ParsedModel,
  type ModelConfig,
  type ConvertStageType,
  type ConvertProgress,
  type ConvertOptions,
  type ShardInfo,
  type TensorSpan,
  type TensorLocation,
  type TensorLocationSingle,
  type TensorLocationMulti,
  type ArchitectureConfig,
  type TokenizerInfo,
  type RDRRManifest,
  type ConvertResult,
  type ConvertIO,
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

export {
  RDRRWriter,
  type WriterOptions,
  type WriteResult,
  type TensorInfo as WriterTensorInfo,
  type MoEConfig,
  type ComponentGroup,
  type TensorMap,
  type ConversionInfo,
  type RuntimeOptimizations,
  type TensorMetadata,
  type TensorLocation as WriterTensorLocation,
  type ShardData,
  type ShardRecord,
  type TokenizerConfig,
  type HuggingFaceTokenizer,
  type ModelInfo,
  type ProgressEvent,
  type WriteRDRROptions,
  writeRDRR,
  createTestModel,
  DEFAULT_SHARD_SIZE,
  ALIGNMENT,
  computeHash,
} from './writer.js';

export {
  type QuantizeResult,
  type QuantizationError,
  type QuantizeOptions,
  type Q4KLayout,
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

export {
  ShardPacker,
  type ShardIO,
  type TensorSpan as PackerTensorSpan,
  type TensorLocationSingle as PackerTensorLocationSingle,
  type TensorLocationMulti as PackerTensorLocationMulti,
  type TensorLocation as PackerTensorLocation,
  type ShardPackerOptions,
  type ShardPackerResult,
  type PackerTensorInput,
  sortTensorsByGroup,
  estimateShardCount,
} from './shard-packer.js';

export { NodeShardIO } from './io/node.js';
