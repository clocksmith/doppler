/**
 * Model formats module - unified exports for all format parsers and types.
 *
 * Subdirectories:
 * - gguf/     - GGUF format (llama.cpp models)
 * - safetensors/ - SafeTensors format (HuggingFace models)
 * - tflite/   - TFLite / LiteRT flatbuffer format
 * - litert/   - LiteRT package containers (.task / .litertlm)
 * - rdrr/     - RDRR format (DOPPLER native format)
 * - tokenizer/ - Tokenizer config parsing utilities
 */

// GGUF format
export * as gguf from './gguf/index.js';

// SafeTensors format
export * as safetensors from './safetensors/index.js';

// TFLite / LiteRT flatbuffer format
export * as tflite from './tflite/index.js';

// LiteRT package containers (.task / .litertlm)
export * as litert from './litert/index.js';

// RDRR format
export * as rdrr from './rdrr/index.js';

// Tokenizer utilities
export * as tokenizer from './tokenizer/index.js';

// Direct re-exports for common types (backward compatibility)
export type {
  GGUFParseResult,
  GGUFTensor,
  GGUFConfig,
  GGUFTokenizer,
  ParsedGGUF,
} from './gguf/index.js';

export type {
  SafetensorsTensor,
  SafetensorsHeader,
  SafetensorsHeaderInfo,
  ParsedSafetensorsHeader,
  SafetensorsDtype,
  SafetensorsDType,
} from './safetensors/index.js';

export type {
  ParsedTFLite,
  TFLiteTensor,
  TFLiteSource,
  TFLiteTensorTypeId,
} from './tflite/index.js';

export type {
  LiteRTSource,
  LiteRTTaskEntry,
  ParsedLiteRTTask,
  LiteRTLMSectionItem,
  LiteRTLMSection,
  ParsedLiteRTLM,
} from './litert/index.js';

export type {
  RDRRManifest,
  TensorLocation,
  TensorMap,
  ShardInfo,
  LayerConfig,
  ComponentGroup,
  MoEConfig,
  ConversionInfo,
  ValidationResult,
} from './rdrr/index.js';
