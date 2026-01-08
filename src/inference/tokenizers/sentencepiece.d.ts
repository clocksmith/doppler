/**
 * SentencePiece Tokenizer
 *
 * Pure JavaScript implementation of SentencePiece (protobuf-based model).
 * Supports Unigram and BPE algorithms.
 *
 * @module inference/tokenizers/sentencepiece
 */

import { BaseTokenizer } from './base.js';
import type { TokenizerConfig } from './types.js';

/**
 * SentencePiece tokenizer using pure JavaScript implementation
 * For models that provide .model files (protobuf format)
 */
export declare class SentencePieceTokenizer extends BaseTokenizer {
  constructor(config?: TokenizerConfig);

  /**
   * Load SentencePiece model from ArrayBuffer
   */
  load(modelData: ArrayBuffer): Promise<void>;

  encode(text: string): number[];

  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;
}
