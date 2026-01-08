/**
 * SentencePiece Tokenizer
 *
 * Pure JavaScript implementation of SentencePiece (protobuf-based model).
 * Supports Unigram and BPE algorithms.
 *
 * @module inference/tokenizers/sentencepiece
 */

import { BaseTokenizer } from './base.js';
import { log } from '../../debug/index.js';

/**
 * @typedef {Object} VarintResult
 * @property {number} value
 * @property {number} newOffset
 */

/**
 * @typedef {Object} PieceData
 * @property {number} id
 * @property {number} score
 * @property {number} type
 */

/**
 * SentencePiece tokenizer using pure JavaScript implementation
 * For models that provide .model files (protobuf format)
 */
export class SentencePieceTokenizer extends BaseTokenizer {
  /** @type {ArrayBuffer | null} */
  #modelData = null;
  /** @type {Map<string, PieceData>} */
  #pieces = new Map();
  /** @type {Map<number, string>} */
  #reverseVocab = new Map();
  /** @type {'unigram' | 'bpe'} */
  #algorithm = 'unigram';
  /** @type {Map<number, number>} */
  #byteTokens = new Map();
  /** @type {number} */
  #unkId = 0;

  /**
   * @param {import('./types.js').TokenizerConfig} [config={}]
   */
  constructor(config = {}) {
    super(config);
  }

  /**
   * Load SentencePiece model from ArrayBuffer
   * @param {ArrayBuffer} modelData
   * @returns {Promise<void>}
   */
  async load(modelData) {
    this.#modelData = modelData;

    try {
      // Parse the SentencePiece model protobuf
      await this.#parseModelProto(modelData);
      log.info('Tokenizer', `Loaded ${this.#pieces.size} pieces (${this.#algorithm})`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      log.warn('Tokenizer', `Failed to parse model, using byte fallback: ${message}`);
      this.#initByteFallback();
    }
  }

  /**
   * Parse SentencePiece model protobuf format
   * SentencePiece uses a simple protobuf schema we can parse manually
   * @param {ArrayBuffer} buffer
   * @returns {Promise<void>}
   */
  async #parseModelProto(buffer) {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // SentencePiece model is a protobuf with:
    // - Field 1: trainer_spec
    // - Field 2: normalizer_spec
    // - Field 3: repeated SentencePiece pieces

    while (offset < bytes.length) {
      // Read varint tag
      const { value: tag, newOffset: tagOffset } = this.#readVarint(bytes, offset);
      offset = tagOffset;

      const fieldNumber = tag >> 3;
      const wireType = tag & 0x7;

      if (fieldNumber === 1 || fieldNumber === 2) {
        // Skip trainer_spec and normalizer_spec (wire type 2 = length-delimited)
        if (wireType === 2) {
          const { value: length, newOffset } = this.#readVarint(bytes, offset);
          offset = newOffset + length;
        }
      } else if (fieldNumber === 3 && wireType === 2) {
        // SentencePiece entry
        const { value: length, newOffset } = this.#readVarint(bytes, offset);
        offset = newOffset;

        const pieceData = bytes.slice(offset, offset + length);
        this.#parsePiece(pieceData);
        offset += length;
      } else {
        // Skip unknown field
        if (wireType === 0) {
          const { newOffset } = this.#readVarint(bytes, offset);
          offset = newOffset;
        } else if (wireType === 2) {
          const { value: length, newOffset } = this.#readVarint(bytes, offset);
          offset = newOffset + length;
        } else if (wireType === 5) {
          offset += 4;
        } else if (wireType === 1) {
          offset += 8;
        } else {
          break; // Unknown wire type, stop parsing
        }
      }
    }

    // Set up special tokens
    this.#unkId = this.specialTokens.unk ?? 0;

    // Determine algorithm from model characteristics
    // (Unigram has scores, BPE typically doesn't)
    const hasScores = [...this.#pieces.values()].some(p => p.score !== 0);
    this.#algorithm = hasScores ? 'unigram' : 'bpe';
  }

  /**
   * Parse a single SentencePiece entry
   * @param {Uint8Array} bytes
   */
  #parsePiece(bytes) {
    let offset = 0;
    let piece = '';
    let score = 0;
    let type = 1; // NORMAL by default

    while (offset < bytes.length) {
      const { value: tag, newOffset: tagOffset } = this.#readVarint(bytes, offset);
      offset = tagOffset;

      const fieldNumber = tag >> 3;
      const wireType = tag & 0x7;

      if (fieldNumber === 1 && wireType === 2) {
        // piece string
        const { value: length, newOffset } = this.#readVarint(bytes, offset);
        offset = newOffset;
        piece = new TextDecoder().decode(bytes.slice(offset, offset + length));
        offset += length;
      } else if (fieldNumber === 2 && wireType === 5) {
        // score (float32)
        const view = new DataView(bytes.buffer, bytes.byteOffset + offset, 4);
        score = view.getFloat32(0, true);
        offset += 4;
      } else if (fieldNumber === 3 && wireType === 0) {
        // type (varint enum)
        const { value, newOffset } = this.#readVarint(bytes, offset);
        type = value;
        offset = newOffset;
      } else {
        // Skip unknown
        if (wireType === 0) {
          const { newOffset } = this.#readVarint(bytes, offset);
          offset = newOffset;
        } else if (wireType === 2) {
          const { value: length, newOffset } = this.#readVarint(bytes, offset);
          offset = newOffset + length;
        } else {
          break;
        }
      }
    }

    if (piece) {
      const id = this.#pieces.size;
      this.#pieces.set(piece, { id, score, type });
      this.#reverseVocab.set(id, piece);
      this.vocabSize = this.#pieces.size;

      // Track byte tokens (▁ prefix tokens and <0xXX> byte tokens)
      if (piece.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
        const byteVal = parseInt(piece.slice(3, 5), 16);
        this.#byteTokens.set(byteVal, id);
      }
    }
  }

  /**
   * Read a protobuf varint
   * @param {Uint8Array} bytes
   * @param {number} offset
   * @returns {VarintResult}
   */
  #readVarint(bytes, offset) {
    let value = 0;
    let shift = 0;
    /** @type {number} */
    let byte;

    do {
      if (offset >= bytes.length) {
        throw new Error('Unexpected end of buffer');
      }
      byte = bytes[offset++];
      value |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);

    return { value, newOffset: offset };
  }

  /**
   * Initialize byte-level fallback vocabulary
   */
  #initByteFallback() {
    // Create a basic byte-level vocabulary
    // Special tokens
    this.#pieces.set('<unk>', { id: 0, score: 0, type: 2 });
    this.#pieces.set('<s>', { id: 1, score: 0, type: 3 });
    this.#pieces.set('</s>', { id: 2, score: 0, type: 3 });
    this.#reverseVocab.set(0, '<unk>');
    this.#reverseVocab.set(1, '<s>');
    this.#reverseVocab.set(2, '</s>');

    // Byte tokens (3-258)
    for (let i = 0; i < 256; i++) {
      const token = `<0x${i.toString(16).padStart(2, '0').toUpperCase()}>`;
      const id = i + 3;
      this.#pieces.set(token, { id, score: 0, type: 6 }); // BYTE type
      this.#reverseVocab.set(id, token);
      this.#byteTokens.set(i, id);
    }

    this.vocabSize = this.#pieces.size;
  }

  /**
   * Encode text using Unigram or BPE algorithm
   * @param {string} text
   * @returns {number[]}
   */
  encode(text) {
    if (!this.#modelData && this.#pieces.size === 0) {
      throw new Error('SentencePiece model not loaded');
    }

    /** @type {number[]} */
    const ids = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Normalize: add sentence piece prefix (▁ for word start)
    const normalized = text.replace(/ /g, '▁');
    const prefixed = (text.startsWith(' ') ? '' : '▁') + normalized;

    if (this.#algorithm === 'unigram') {
      ids.push(...this.#encodeUnigram(prefixed));
    } else {
      ids.push(...this.#encodeBPE(prefixed));
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos ?? 2);
    }

    return ids;
  }

  /**
   * Unigram encoding using Viterbi algorithm
   * @param {string} text
   * @returns {number[]}
   */
  #encodeUnigram(text) {
    const n = text.length;
    if (n === 0) return [];

    // Viterbi: best[i] = {score, prev, tokenLen} for position i
    /** @type {Array<import('./types.js').ViterbiState | null>} */
    const best = new Array(n + 1).fill(null);
    best[0] = { score: 0, prev: -1, tokenLen: 0 };

    for (let i = 0; i < n; i++) {
      if (best[i] === null) continue;

      // Try all possible tokens starting at position i
      for (let len = 1; len <= Math.min(n - i, 32); len++) {
        const substr = text.slice(i, i + len);
        const piece = this.#pieces.get(substr);

        if (piece) {
          const newScore = best[i].score + piece.score;
          if (best[i + len] === null || newScore > best[i + len].score) {
            best[i + len] = { score: newScore, prev: i, tokenLen: len };
          }
        }
      }

      // Byte fallback for single character
      if (best[i + 1] === null) {
        const charCode = text.charCodeAt(i);
        const bytes = new TextEncoder().encode(text[i]);
        // Use byte tokens with a penalty score
        const byteScore = best[i].score - 10 * bytes.length;
        best[i + 1] = { score: byteScore, prev: i, tokenLen: 1, isBytes: true, bytes };
      }
    }

    // Backtrack to get tokens
    /** @type {number[]} */
    const tokens = [];
    let pos = n;
    while (pos > 0) {
      const state = best[pos];
      if (state.isBytes && state.bytes) {
        // Use byte tokens
        for (let j = state.bytes.length - 1; j >= 0; j--) {
          const byteId = this.#byteTokens.get(state.bytes[j]);
          tokens.push(byteId ?? this.#unkId);
        }
      } else {
        const substr = text.slice(state.prev, pos);
        const piece = this.#pieces.get(substr);
        tokens.push(piece?.id ?? this.#unkId);
      }
      pos = state.prev;
    }

    return tokens.reverse();
  }

  /**
   * BPE encoding
   * @param {string} text
   * @returns {number[]}
   */
  #encodeBPE(text) {
    // Start with character-level tokens
    /** @type {string[]} */
    let tokens = [];
    for (const char of text) {
      const piece = this.#pieces.get(char);
      if (piece) {
        tokens.push(char);
      } else {
        // Byte fallback
        const bytes = new TextEncoder().encode(char);
        for (const b of bytes) {
          const byteToken = `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
          tokens.push(byteToken);
        }
      }
    }

    // Iteratively merge pairs with highest score
    while (tokens.length > 1) {
      /** @type {string | null} */
      let bestPair = null;
      let bestScore = -Infinity;
      let bestIndex = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = tokens[i] + tokens[i + 1];
        const piece = this.#pieces.get(merged);
        if (piece && piece.score > bestScore) {
          bestScore = piece.score;
          bestPair = merged;
          bestIndex = i;
        }
      }

      if (bestPair === null) break;

      // Apply merge
      tokens = [
        ...tokens.slice(0, bestIndex),
        bestPair,
        ...tokens.slice(bestIndex + 2)
      ];
    }

    // Convert to IDs
    return tokens.map(t => {
      const piece = this.#pieces.get(t);
      return piece?.id ?? this.#unkId;
    });
  }

  /**
   * @param {number[]} ids
   * @param {boolean} [skipSpecialTokens=true]
   * @param {boolean} [trim=true]
   * @returns {string}
   */
  decode(ids, skipSpecialTokens = true, trim = true) {
    if (this.#pieces.size === 0) {
      throw new Error('SentencePiece model not loaded');
    }

    /** @type {string[]} */
    const tokens = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.#reverseVocab.get(id);
      if (token) {
        // Handle byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          tokens.push(String.fromCharCode(byteVal));
        } else {
          tokens.push(token);
        }
      }
    }

    // Join and convert ▁ back to spaces
    const result = tokens.join('').replace(/▁/g, ' ');
    return trim ? result.trim() : result;
  }
}
