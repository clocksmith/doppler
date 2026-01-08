/**
 * RDRR Writer Module
 *
 * @module converter/writer
 */

import type {
  ModelInfo,
  WriteRDRROptions,
  WriteResultSchema,
  TensorInfoSchema,
} from './types.js';

export * from './types.js';

export { computeHash, alignOffset, createPadding, getBytesPerElement, transpose2D } from './utils.js';

export { DEFAULT_SHARD_SIZE, ALIGNMENT } from './types.js';

export { RDRRWriter } from './writer.js';
export { ShardWriter } from './shard-writer.js';
export { ManifestWriter, type ManifestData } from './manifest-writer.js';
export { TokenizerWriter, type TokenizerManifestEntry } from './tokenizer-writer.js';

export { createTestModel } from '../test-model.js';

/**
 * High-level function to write a model in RDRR format.
 */
export declare function writeRDRR(
  outputDir: string,
  modelInfo: ModelInfo,
  getTensorData: (tensor: TensorInfoSchema) => Promise<ArrayBuffer>,
  options?: WriteRDRROptions
): Promise<WriteResultSchema>;
