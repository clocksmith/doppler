/**
 * Model Conversion Tools - Barrel export for all conversion utilities.
 */

// GGUF Parser
export {
  parseGGUF,
  parseGGUFFile,
  getTensor,
  getTensors,
  groupTensorsByLayer,
  identifyMoETensors,
  GGMLType,
  GGMLTypeName,
  GGML_BLOCK_SIZE,
  GGML_TYPE_SIZE,
} from './gguf-parser.js';

export type {
  GGUFTensor,
  GGUFParseResult,
  GGUFConfig,
} from './gguf-parser.js';

// Safetensors Parser
export {
  parseSafetensorsHeader,
  parseSafetensorsFile,
  parseSafetensorsIndex,
  parseSafetensors,
  detectModelFormat,
  loadModelConfig,
  loadTokenizerConfig,
  readTensorData,
  groupTensorsByLayer as groupSafetensorsByLayer,
  calculateTotalSize,
  DTYPE_SIZE,
  DTYPE_MAP,
} from './safetensors-parser.js';

export type {
  SafetensorsTensor,
  SafetensorsHeader,
  SafetensorsHeaderInfo,
  ParsedHeader,
  ParsedSafetensorsFile,
  ParsedSafetensorsIndex,
  ShardInfo,
  ModelFormatInfo,
  SafetensorsDType,
} from './safetensors-parser.js';

// Quantizer
export {
  quantizeToQ4KM,
  dequantizeQ4KM,
  quantizeF16ToQ4KM,
  calculateQuantizationError,
  shouldQuantize,
  getQuantizedSize,
  float32ToFloat16,
  float16ToFloat32,
  QK_K,
  QK4_K_BLOCK_SIZE,
} from './quantizer.js';

export type {
  QuantizeResult,
  QuantizationError,
} from './quantizer.js';

// RDRR Writer
export {
  RDRRWriter,
  writeRDRR,
  createTestModel,
  DEFAULT_SHARD_SIZE,
  ALIGNMENT,
  computeHash,
} from './rdrr-writer.js';

export type {
  WriterOptions,
  TensorMetadata,
  WriteResult,
  ProgressEvent,
} from './rdrr-writer.js';
