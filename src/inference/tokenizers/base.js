/**
 * Abstract Base Tokenizer
 *
 * @module inference/tokenizers/base
 */

import { getRuntimeConfig } from '../../config/runtime.js';

/**
 * Abstract base tokenizer interface
 * @implements {import('./types.js').TokenizerBackend}
 */
export class BaseTokenizer {
  /** @type {number} */
  vocabSize;
  /** @type {import('./types.js').SpecialTokens} */
  specialTokens;
  /** @type {boolean} */
  addBosToken;
  /** @type {boolean} */
  addEosToken;

  /**
   * @param {import('./types.js').TokenizerConfig} [config={}]
   */
  constructor(config = {}) {
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
   * @param {string} text
   * @returns {number[]}
   * @abstract
   */
  encode(text) {
    throw new Error('Abstract method not implemented');
  }

  /**
   * Decode token IDs to text
   * @param {number[]} ids
   * @param {boolean} [skipSpecialTokens]
   * @param {boolean} [trim]
   * @returns {string}
   * @abstract
   */
  decode(ids, skipSpecialTokens, trim) {
    throw new Error('Abstract method not implemented');
  }

  /**
   * Get vocabulary size
   * @returns {number}
   */
  getVocabSize() {
    return this.vocabSize;
  }

  /**
   * Check if token is special
   * @param {number} tokenId
   * @returns {boolean}
   */
  isSpecialToken(tokenId) {
    return Object.values(this.specialTokens).includes(tokenId);
  }
}
