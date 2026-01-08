/**
 * Abstract Base Tokenizer
 *
 * @module inference/tokenizers/base
 */

import type { TokenizerConfig, SpecialTokens, TokenizerBackend } from './types.js';

/**
 * Abstract base tokenizer interface
 */
export declare abstract class BaseTokenizer implements TokenizerBackend {
  vocabSize: number;
  specialTokens: SpecialTokens;
  addBosToken: boolean;
  addEosToken: boolean;

  constructor(config?: TokenizerConfig);

  /**
   * Encode text to token IDs
   */
  abstract encode(text: string): number[];

  /**
   * Decode token IDs to text
   */
  abstract decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;

  /**
   * Get vocabulary size
   */
  getVocabSize(): number;

  /**
   * Check if token is special
   */
  isSpecialToken(tokenId: number): boolean;
}
