/**
 * RDRR Writer - Main Orchestration Class
 *
 * @module converter/writer/writer
 */

import type {
  TensorMetadata,
  TensorLocation,
  TokenizerConfig,
  HuggingFaceTokenizer,
  MoEConfigSchema,
  ConversionInfoSchema,
  RuntimeOptimizationsSchema,
  ManifestInferenceSchema,
  WriterOptionsSchema,
  WriteResultSchema,
} from './types.js';

/**
 * Main RDRR writer class.
 * Orchestrates tensor writing, manifest generation, and file output.
 */
export declare class RDRRWriter {
  constructor(outputDir: string, options?: WriterOptionsSchema);

  init(): Promise<void>;

  writeTensor(name: string, data: Uint8Array, metadata: TensorMetadata): Promise<TensorLocation>;

  setConfig(config: Record<string, unknown>): void;
  setTokenizer(tokenizer: Record<string, unknown>): void;
  setMoEConfig(moeConfig: MoEConfigSchema): void;
  setConversion(conversion: ConversionInfoSchema): void;
  setOptimizations(optimizations: RuntimeOptimizationsSchema): void;
  setInference(inference: ManifestInferenceSchema): void;
  setMetadata(meta: Record<string, unknown>): void;

  writeTokenizer(tokenizer: TokenizerConfig): Promise<void>;
  writeHuggingFaceTokenizer(tokenizerJson: HuggingFaceTokenizer): Promise<void>;

  finalize(): Promise<WriteResultSchema>;
  cleanup(): Promise<void>;
}
