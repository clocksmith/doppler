/**
 * Tokenizer Wrapper
 *
 * Provides a unified interface for tokenization across different backends.
 *
 * @module inference/tokenizer
 */

export type { TokenizerConfig, ModelManifest, SpecialTokens } from './tokenizers/types.js';

import type { TokenizerConfig, ModelManifest, SpecialTokens } from './tokenizers/types.js';

/**
 * Tokenizer wrapper that auto-detects backend from model manifest
 * This is a thin wrapper over the backend implementations
 */
export declare class Tokenizer {
  private backend;
  private config;

  /**
   * Initialize from model manifest
   */
  initialize(manifest: ModelManifest, options?: { baseUrl?: string }): Promise<void>;

  /**
   * Infer HuggingFace model ID from manifest architecture
   */
  private _inferHuggingFaceModel(manifest: ModelManifest): string | null;

  /**
   * Encode text to token IDs
   */
  encode(text: string): number[];

  /**
   * Decode token IDs to text
   * @param skipSpecialTokens - Whether to skip special tokens in output
   * @param trim - Whether to trim whitespace (default true, set false for streaming)
   */
  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;

  /**
   * Get special tokens
   */
  getSpecialTokens(): SpecialTokens;

  /**
   * Get vocabulary size
   */
  getVocabSize(): number;
}

export default Tokenizer;
