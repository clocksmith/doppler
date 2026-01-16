

import { BaseTokenizer } from './base.js';


export class BPETokenizer extends BaseTokenizer {
  
  #vocab = new Map();
  
  #reverseVocab = new Map();
  
  #merges = [];
  
  #mergeRanks = new Map();

  
  constructor(config = {}) {
    super(config);
  }

  
  load(vocab, merges) {
    // Build vocab maps
    for (const [token, id] of Object.entries(vocab)) {
      this.#vocab.set(token, id);
      this.#reverseVocab.set(id, token);
    }

    this.vocabSize = this.#vocab.size;

    // Build merge ranks
    this.#merges = merges;
    for (let i = 0; i < merges.length; i++) {
      this.#mergeRanks.set(merges[i], i);
    }
  }

  
  #getPairs(word) {
    
    const pairs = [];
    for (let i = 0; i < word.length - 1; i++) {
      pairs.push(`${word[i]} ${word[i + 1]}`);
    }
    return pairs;
  }

  
  #bpe(word) {
    let tokens = word.split('');

    while (tokens.length > 1) {
      // Find the pair with lowest rank
      const pairs = this.#getPairs(tokens);
      
      let minPair = null;
      let minRank = Infinity;

      for (const pair of pairs) {
        const rank = this.#mergeRanks.get(pair);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (minPair === null) break;

      // Merge the pair
      const [first, second] = minPair.split(' ');
      
      const newTokens = [];
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

  
  encode(text) {
    
    const ids = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Simple word-level tokenization then BPE
    // In production, would use proper pre-tokenization
    const words = text.split(/(\s+)/);

    for (const word of words) {
      if (word.trim() === '') {
        // Handle whitespace
        const wsToken = this.#vocab.get(word);
        if (wsToken !== undefined) {
          ids.push(wsToken);
        }
        continue;
      }

      // Apply BPE
      const tokens = this.#bpe(word);

      for (const token of tokens) {
        const id = this.#vocab.get(token);
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

  
  decode(ids, skipSpecialTokens = true, trim = true) {
    
    const tokens = [];

    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.#reverseVocab.get(id);
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
