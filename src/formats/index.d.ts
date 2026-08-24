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
export * as gguf from './gguf/types.js';

// SafeTensors format
export * as safetensors from './safetensors/types.js';

// TFLite / LiteRT flatbuffer format
export * as tflite from './tflite/types.js';

// LiteRT package containers (.task / .litertlm)
export * as litert from './litert/types.js';

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
} from './gguf/types.js';

export type {
  SafetensorsTensor,
  SafetensorsHeader,
  SafetensorsHeaderInfo,
  ParsedSafetensorsHeader,
  SafetensorsDtype,
  SafetensorsDType,
} from './safetensors/types.js';

export type {
  ParsedTFLite,
  TFLiteTensor,
  TFLiteSource,
  TFLiteTensorTypeId,
} from './tflite/types.js';

export type {
  LiteRTSource,
  LiteRTTaskEntry,
  ParsedLiteRTTask,
  LiteRTLMSectionItem,
  LiteRTLMSection,
  ParsedLiteRTLM,
} from './litert/types.js';

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
