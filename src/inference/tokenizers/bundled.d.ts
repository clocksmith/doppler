/**
 * Bundled Tokenizer
 *
 * Support for .rdrr bundled tokenizers and Transformers.js fallback.
 *
 * @module inference/tokenizers/bundled
 */

import { BaseTokenizer } from './base.js';
import type {
  TokenizerConfig,
  TransformersTokenizerType,
  HuggingFaceTokenizerJson,
  BundledTokenizerJson
} from './types.js';

/**
 * Wrapper for Transformers.js tokenizer
 */
export declare class TransformersTokenizer extends BaseTokenizer {
  constructor(config?: TokenizerConfig);

  /**
   * Initialize with a Transformers.js tokenizer instance
   */
  setTokenizer(tokenizer: TransformersTokenizerType): void;

  /**
   * Load tokenizer from HuggingFace model
   * @deprecated Use BundledTokenizer instead - no external dependencies
   */
  load(_modelId: string): Promise<void>;

  encode(text: string): number[];

  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;

  /**
   * Batch encode multiple texts
   */
  batchEncode(texts: string[]): number[][];
}

/**
 * Bundled tokenizer for .rdrr format with embedded vocab.
 * Eliminates runtime dependency on transformers.js CDN.
 * Supports both BPE and Unigram (SentencePiece) algorithms.
 */
export declare class BundledTokenizer extends BaseTokenizer {
  constructor(config?: TokenizerConfig);

  isSpecialToken(tokenId: number): boolean;

  /**
   * Load from tokenizer.json content
   * Auto-detects HuggingFace format vs bundled format
   */
  load(tokenizerJson: HuggingFaceTokenizerJson | BundledTokenizerJson): void;

  encode(text: string): number[];

  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;
}
