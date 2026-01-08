/**
 * Abstract Base Tokenizer
 *
 * @module inference/tokenizers/base
 */

import type { TokenizerConfig, SpecialTokens, TokenizerBackend } from './types.js';
import { getRuntimeConfig } from '../../config/runtime.js';

/**
 * Abstract base tokenizer interface
 */
export abstract class BaseTokenizer implements TokenizerBackend {
  vocabSize: number;
  specialTokens: SpecialTokens;
  addBosToken: boolean;
  addEosToken: boolean;

  constructor(config: TokenizerConfig = {}) {
    const runtimeDefaults = getRuntimeConfig().inference.tokenizer;
    this.vocabSize = config.vocabSize || 32000;
    this.specialTokens = {
      pad: config.padToken ?? 0,
      bos: config.bosToken ?? 1,
      eos: config.eosToken ?? 2,
      unk: config.unkToken ?? 0,
      ...config.specialTokens
    };
    this.addBosToken = config.addBosToken ?? runtimeDefaults.addBosToken;
    this.addEosToken = config.addEosToken ?? runtimeDefaults.addEosToken;
  }

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
  getVocabSize(): number {
    return this.vocabSize;
  }

  /**
   * Check if token is special
   */
  isSpecialToken(tokenId: number): boolean {
    return Object.values(this.specialTokens).includes(tokenId);
  }
}
