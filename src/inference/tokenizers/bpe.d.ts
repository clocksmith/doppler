/**
 * Simple BPE Tokenizer
 *
 * For models with vocab.json + merges.txt
 *
 * @module inference/tokenizers/bpe
 */

import { BaseTokenizer } from './base.js';
import type { TokenizerConfig } from './types.js';

/**
 * Simple BPE tokenizer
 * For models with vocab.json + merges.txt
 */
export declare class BPETokenizer extends BaseTokenizer {
  constructor(config?: TokenizerConfig);

  /**
   * Load vocabulary and merges
   */
  load(vocab: Record<string, number>, merges: string[]): void;

  encode(text: string): number[];

  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;
}
