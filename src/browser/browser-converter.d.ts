/**
 * browser-converter.ts - Browser Model Converter
 *
 * Converts GGUF and safetensors models to RDRR format in the browser.
 * Uses ShardPacker for platform-agnostic shard packing logic.
 *
 * Supports:
 * - GGUF files
 * - Single safetensors files
 * - Sharded HuggingFace models (multiple safetensors + config)
 *
 * Output is written to OPFS when available, with IndexedDB fallback.
 *
 * @module browser/browser-converter
 */

import type { TensorLocation, ShardPackerResult } from '../converter/shard-packer.js';
import type {
  ConvertStageType,
  ConvertProgress,
  ConvertOptions as CoreConvertOptions,
  ShardInfo,
  RDRRManifest,
} from '../converter/core.js';
import type { TensorSource } from './tensor-source-file.js';
import type { HttpTensorSourceOptions } from './tensor-source-http.js';

export { ConvertStage } from '../converter/core.js';
export type {
  ConvertStageType,
  ConvertProgress,
  ShardInfo,
  TensorLocation,
  RDRRManifest,
};

/** Check if browser conversion is supported (OPFS or IndexedDB). */
export declare function isConversionSupported(): boolean;

/**
 * Create TensorSource entries for remote model files.
 */
export declare function createRemoteModelSources(
  urls: string[],
  options?: HttpTensorSourceOptions
): Promise<TensorSource[]>;

/**
 * Browser conversion options (extends core with File[] input)
 */
export interface ConvertOptions extends CoreConvertOptions {
  // Browser-specific: no additional options needed
}

/**
 * Convert model files to RDRR format
 *
 * @param files - Selected model files
 * @param options - Conversion options
 * @returns Model ID
 */
export declare function convertModel(files: Array<File | TensorSource>, options?: ConvertOptions): Promise<string>;

/**
 * Pick model files using File System Access API
 * @returns Selected files
 */
export declare function pickModelFiles(): Promise<File[]>;

export default convertModel;
