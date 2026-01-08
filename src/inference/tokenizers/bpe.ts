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
export class BPETokenizer extends BaseTokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: string[] = [];
  private mergeRanks: Map<string, number> = new Map();

  constructor(config: TokenizerConfig = {}) {
    super(config);
  }

  /**
   * Load vocabulary and merges
   */
  load(vocab: Record<string, number>, merges: string[]): void {
    // Build vocab maps
    for (const [token, id] of Object.entries(vocab)) {
      this.vocab.set(token, id);
      this.reverseVocab.set(id, token);
    }

    this.vocabSize = this.vocab.size;

    // Build merge ranks
    this.merges = merges;
    for (let i = 0; i < merges.length; i++) {
      this.mergeRanks.set(merges[i], i);
    }
  }

  /**
   * Get pairs of adjacent symbols in word
   */
  private _getPairs(word: string[]): string[] {
    const pairs: string[] = [];
    for (let i = 0; i < word.length - 1; i++) {
      pairs.push(`${word[i]} ${word[i + 1]}`);
    }
    return pairs;
  }

  /**
   * Apply BPE to a single word
   */
  private _bpe(word: string): string[] {
    let tokens = word.split('');

    while (tokens.length > 1) {
      // Find the pair with lowest rank
      const pairs = this._getPairs(tokens);
      let minPair: string | null = null;
      let minRank = Infinity;

      for (const pair of pairs) {
        const rank = this.mergeRanks.get(pair);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (minPair === null) break;

      // Merge the pair
      const [first, second] = minPair.split(' ');
      const newTokens: string[] = [];
      let i = 0;

      while (i < tokens.length) {
        if (i < tokens.length - 1 &&
            tokens[i] === first &&
            tokens[i + 1] === second) {
          newTokens.push(first + second);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i += 1;
        }
      }

      tokens = newTokens;
    }

    return tokens;
  }

  encode(text: string): number[] {
    const ids: number[] = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Simple word-level tokenization then BPE
    // In production, would use proper pre-tokenization
    const words = text.split(/(\s+)/);

    for (const word of words) {
      if (word.trim() === '') {
        // Handle whitespace
        const wsToken = this.vocab.get(word);
        if (wsToken !== undefined) {
          ids.push(wsToken);
        }
        continue;
      }

      // Apply BPE
      const tokens = this._bpe(word);

      for (const token of tokens) {
        const id = this.vocab.get(token);
        if (id !== undefined) {
          ids.push(id);
        } else {
          // Unknown token
          ids.push(this.specialTokens.unk ?? 0);
        }
      }
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos ?? 2);
    }

    return ids;
  }

  decode(ids: number[], skipSpecialTokens: boolean = true, trim: boolean = true): string {
    const tokens: string[] = [];

    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.reverseVocab.get(id);
      if (token !== undefined) {
        tokens.push(token);
      }
    }

    // Join tokens (handle special whitespace markers like Ġ)
    const result = tokens.join('')
      .replace(/Ġ/g, ' ')
      .replace(/Ċ/g, '\n');
    return trim ? result.trim() : result;
  }
}
