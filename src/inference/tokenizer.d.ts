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
 * Options for tokenizer initialization
 */
export interface TokenizerInitOptions {
  /** Base URL for loading tokenizer files */
  baseUrl?: string;
  /** Preset tokenizer config as fallback hints (manifest takes precedence) */
  presetTokenizer?: {
    bosToken?: string;
    eosTokens?: string[];
    padToken?: string;
    addBosToken?: boolean;
    addEosToken?: boolean;
    hfModel?: string;
    allowArchFallback?: boolean;
  };
}

/**
 * Tokenizer wrapper that auto-detects backend from model manifest
 * This is a thin wrapper over the backend implementations
 */
export declare class Tokenizer {
  private backend;
  private config;

  /**
   * Initialize from model manifest.
   * Preset tokenizer provides fallback hints when manifest tokenizer is missing fields.
   */
  initialize(manifest: ModelManifest, options?: TokenizerInitOptions): Promise<void>;

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
