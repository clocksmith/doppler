#!/usr/bin/env node
/**
 * Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.
 *
 * @module converter/node-converter
 */

export type {
  ConvertOptions,
  TensorInfo,
  ModelInfo,
  InputFormat,
  TensorDataGetter,
} from './types.js';

export { parseArgs, printHelp } from './cli.js';
export { detectInputFormat, detectModelTypeFromPreset } from './detection.js';
export { convertSafetensors, convertGGUF } from './converter.js';
export {
  normalizeQuantTag,
  validateQuantType,
  resolveManifestQuantization,
  buildVariantTag,
  buildQuantizationInfo,
  resolveModelId,
  toWebGPUDtype,
} from './quantization.js';
export {
  createVerboseLogger,
  logConversionStart,
  logConversionComplete,
  logTestFixtureComplete,
  logConversionError,
} from './progress.js';

/**
 * Main entry point for the CLI.
 */
export declare function main(): Promise<void>;
