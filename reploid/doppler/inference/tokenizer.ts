/**
 * tokenizer.ts - Tokenizer Wrapper
 *
 * Provides a unified interface for tokenization across different backends:
 * - Transformers.js tokenizers (from HuggingFace)
 * - SentencePiece WASM (for .model files)
 * - Simple BPE (fallback for basic models)
 *
 * @module inference/tokenizer
 */

import type { SpecialTokens } from '../types/inference.js';

/** Tokenizer Configuration */
export interface TokenizerConfig {
  /** Tokenizer backend type */
  type?: 'transformers' | 'sentencepiece' | 'bpe' | 'bundled' | 'huggingface';
  /** Path to tokenizer model/vocab */
  modelPath?: string;
  /** Special token mappings */
  specialTokens?: SpecialTokens;
  /** Vocabulary size */
  vocabSize?: number;
  /** Padding token ID */
  padToken?: number;
  /** Beginning of sequence token ID */
  bosToken?: number;
  /** End of sequence token ID */
  eosToken?: number;
  /** Unknown token ID */
  unkToken?: number;
  /** Whether to add BOS token */
  addBosToken?: boolean;
  /** Whether to add EOS token */
  addEosToken?: boolean;
  /** HuggingFace model ID */
  modelId?: string;
  hfModel?: string;
  /** SentencePiece model data or path */
  sentencepieceModel?: ArrayBuffer | string;
  /** BPE vocabulary */
  vocab?: Record<string, number>;
  /** BPE merge rules */
  merges?: string[];
  /** File path for bundled tokenizer */
  file?: string;
  /** Shard loader function */
  loadShard?: (index: number | string) => Promise<ArrayBuffer>;
}

/** Tokenizer Backend Interface (from MIGRATION-B.md) */
export interface TokenizerBackend {
  /** Encode text to token IDs */
  encode(text: string, addSpecial?: boolean): number[];
  /** Decode token IDs to text */
  decode(tokens: number[], skipSpecial?: boolean, trim?: boolean): string;
  /** Get vocabulary size */
  vocabSize: number;
  /** Get special token IDs */
  specialTokens: SpecialTokens;
}

/** Protobuf varint reading result */
interface VarintResult {
  value: number;
  newOffset: number;
}

/** SentencePiece piece data */
interface PieceData {
  id: number;
  score: number;
  type: number;
}

/** Viterbi state for Unigram encoding */
interface ViterbiState {
  score: number;
  prev: number;
  tokenLen: number;
  isBytes?: boolean;
  bytes?: Uint8Array;
}

/** Special token pattern for matching */
interface SpecialTokenPattern {
  content: string;
  id: number;
}

/** Text segment (plain text or special token) */
interface TextSegment {
  text?: string;
  id?: number;
  isSpecial: boolean;
}

/** Transformers.js tokenizer type (from external library) */
interface TransformersTokenizerType {
  model?: {
    vocab?: Record<string, number>;
  };
  special_tokens_map?: {
    pad_token?: string;
    bos_token?: string;
    eos_token?: string;
    unk_token?: string;
  };
  pad_token_id?: number;
  bos_token_id?: number;
  eos_token_id?: number;
  unk_token_id?: number;
  encode(text: string, options?: { add_special_tokens?: boolean }): ArrayLike<number>;
  decode(ids: number[], options?: { skip_special_tokens?: boolean }): string;
}

/** HuggingFace tokenizer.json format */
interface HuggingFaceTokenizerJson {
  model?: {
    type?: string;
    vocab?: Record<string, number | string> | Array<[string, number]>;
    merges?: string[];
    pad_id?: number;
    unk_id?: number;
    add_prefix_space?: boolean;
    add_dummy_prefix?: boolean;
  };
  added_tokens?: Array<{
    id: number | string;
    content: string;
    special: boolean;
  }>;
  add_bos_token?: boolean;
  add_eos_token?: boolean;
}

/** Bundled tokenizer.json format */
interface BundledTokenizerJson {
  type?: string;
  vocab: Record<string, number | string>;
  merges?: string[];
  scores?: number[];
  tokenTypes?: number[];
  specialTokens?: {
    pad?: number;
    bos?: number;
    eos?: number;
    unk?: number;
  };
  addBosToken?: boolean;
  addEosToken?: boolean;
  addSpacePrefix?: boolean;
}

/** Model manifest for tokenizer initialization */
export interface ModelManifest {
  tokenizer?: TokenizerConfig;
  architecture?: string;
  config?: {
    architectures?: string[];
    vocab_size?: number;
    model_type?: string;
    text_config?: {
      vocab_size?: number;
      model_type?: string;
    };
  };
}

/**
 * Abstract base tokenizer interface
 */
abstract class BaseTokenizer {
  vocabSize: number;
  specialTokens: SpecialTokens;
  addBosToken: boolean;
  addEosToken: boolean;

  constructor(config: TokenizerConfig = {}) {
    this.vocabSize = config.vocabSize || 32000;
    this.specialTokens = {
      pad: config.padToken ?? 0,
      bos: config.bosToken ?? 1,
      eos: config.eosToken ?? 2,
      unk: config.unkToken ?? 0,
      ...config.specialTokens
    };
    this.addBosToken = config.addBosToken !== false;
    this.addEosToken = config.addEosToken || false;
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

/**
 * Wrapper for Transformers.js tokenizer
 */
export class TransformersTokenizer extends BaseTokenizer {
  private tokenizer: TransformersTokenizerType | null = null;
  private modelId?: string;

  constructor(config: TokenizerConfig = {}) {
    super(config);
    this.modelId = config.modelId;
  }

  /**
   * Initialize with a Transformers.js tokenizer instance
   */
  setTokenizer(tokenizer: TransformersTokenizerType): void {
    this.tokenizer = tokenizer;
    if (tokenizer.model?.vocab) {
      this.vocabSize = Object.keys(tokenizer.model.vocab).length;
    }
  }

  /**
   * Load tokenizer from HuggingFace model
   * @deprecated Use BundledTokenizer instead - no external dependencies
   */
  async load(_modelId: string): Promise<void> {
    // DOPPLER uses bundled tokenizers only - no external CDN dependencies
    throw new Error(
      '[Tokenizer] TransformersTokenizer is deprecated. ' +
      'Use bundled tokenizer (type: "bundled" or "huggingface" with file). ' +
      'DOPPLER requires no external runtime dependencies.'
    );
  }

  encode(text: string): number[] {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized');
    }

    const result = this.tokenizer.encode(text, {
      add_special_tokens: this.addBosToken
    });

    return Array.from(result);
  }

  decode(ids: number[], skipSpecialTokens: boolean = true, trim: boolean = true): string {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not initialized');
    }

    const result = this.tokenizer.decode(ids, { skip_special_tokens: skipSpecialTokens });
    return trim ? result.trim() : result;
  }

  /**
   * Batch encode multiple texts
   */
  batchEncode(texts: string[]): number[][] {
    return texts.map(t => this.encode(t));
  }
}

/**
 * SentencePiece tokenizer using pure JavaScript implementation
 * For models that provide .model files (protobuf format)
 *
 * Supports Unigram and BPE algorithms commonly used in LLMs like LLaMA.
 */
export class SentencePieceTokenizer extends BaseTokenizer {
  private modelData: ArrayBuffer | null = null;
  private pieces: Map<string, PieceData> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private algorithm: 'unigram' | 'bpe' = 'unigram';
  private byteTokens: Map<number, number> = new Map();
  private unkId: number = 0;

  constructor(config: TokenizerConfig = {}) {
    super(config);
  }

  /**
   * Load SentencePiece model from ArrayBuffer
   */
  async load(modelData: ArrayBuffer): Promise<void> {
    this.modelData = modelData;

    try {
      // Parse the SentencePiece model protobuf
      await this._parseModelProto(modelData);
      console.log(`[SentencePiece] Loaded ${this.pieces.size} pieces (${this.algorithm})`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.warn('[SentencePiece] Failed to parse model, using byte fallback:', message);
      this._initByteFallback();
    }
  }

  /**
   * Parse SentencePiece model protobuf format
   * SentencePiece uses a simple protobuf schema we can parse manually
   */
  private async _parseModelProto(buffer: ArrayBuffer): Promise<void> {
    const view = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let offset = 0;

    // SentencePiece model is a protobuf with:
    // - Field 1: trainer_spec
    // - Field 2: normalizer_spec
    // - Field 3: repeated SentencePiece pieces

    while (offset < bytes.length) {
      // Read varint tag
      const { value: tag, newOffset: tagOffset } = this._readVarint(bytes, offset);
      offset = tagOffset;

      const fieldNumber = tag >> 3;
      const wireType = tag & 0x7;

      if (fieldNumber === 1 || fieldNumber === 2) {
        // Skip trainer_spec and normalizer_spec (wire type 2 = length-delimited)
        if (wireType === 2) {
          const { value: length, newOffset } = this._readVarint(bytes, offset);
          offset = newOffset + length;
        }
      } else if (fieldNumber === 3 && wireType === 2) {
        // SentencePiece entry
        const { value: length, newOffset } = this._readVarint(bytes, offset);
        offset = newOffset;

        const pieceData = bytes.slice(offset, offset + length);
        this._parsePiece(pieceData);
        offset += length;
      } else {
        // Skip unknown field
        if (wireType === 0) {
          const { newOffset } = this._readVarint(bytes, offset);
          offset = newOffset;
        } else if (wireType === 2) {
          const { value: length, newOffset } = this._readVarint(bytes, offset);
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
    this.unkId = this.specialTokens.unk ?? 0;

    // Determine algorithm from model characteristics
    // (Unigram has scores, BPE typically doesn't)
    const hasScores = [...this.pieces.values()].some(p => p.score !== 0);
    this.algorithm = hasScores ? 'unigram' : 'bpe';
  }

  /**
   * Parse a single SentencePiece entry
   */
  private _parsePiece(bytes: Uint8Array): void {
    let offset = 0;
    let piece = '';
    let score = 0;
    let type = 1; // NORMAL by default

    while (offset < bytes.length) {
      const { value: tag, newOffset: tagOffset } = this._readVarint(bytes, offset);
      offset = tagOffset;

      const fieldNumber = tag >> 3;
      const wireType = tag & 0x7;

      if (fieldNumber === 1 && wireType === 2) {
        // piece string
        const { value: length, newOffset } = this._readVarint(bytes, offset);
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
        const { value, newOffset } = this._readVarint(bytes, offset);
        type = value;
        offset = newOffset;
      } else {
        // Skip unknown
        if (wireType === 0) {
          const { newOffset } = this._readVarint(bytes, offset);
          offset = newOffset;
        } else if (wireType === 2) {
          const { value: length, newOffset } = this._readVarint(bytes, offset);
          offset = newOffset + length;
        } else {
          break;
        }
      }
    }

    if (piece) {
      const id = this.pieces.size;
      this.pieces.set(piece, { id, score, type });
      this.reverseVocab.set(id, piece);
      this.vocabSize = this.pieces.size;

      // Track byte tokens (▁ prefix tokens and <0xXX> byte tokens)
      if (piece.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
        const byteVal = parseInt(piece.slice(3, 5), 16);
        this.byteTokens.set(byteVal, id);
      }
    }
  }

  /**
   * Read a protobuf varint
   */
  private _readVarint(bytes: Uint8Array, offset: number): VarintResult {
    let value = 0;
    let shift = 0;
    let byte: number;

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
  private _initByteFallback(): void {
    // Create a basic byte-level vocabulary
    // Special tokens
    this.pieces.set('<unk>', { id: 0, score: 0, type: 2 });
    this.pieces.set('<s>', { id: 1, score: 0, type: 3 });
    this.pieces.set('</s>', { id: 2, score: 0, type: 3 });
    this.reverseVocab.set(0, '<unk>');
    this.reverseVocab.set(1, '<s>');
    this.reverseVocab.set(2, '</s>');

    // Byte tokens (3-258)
    for (let i = 0; i < 256; i++) {
      const token = `<0x${i.toString(16).padStart(2, '0').toUpperCase()}>`;
      const id = i + 3;
      this.pieces.set(token, { id, score: 0, type: 6 }); // BYTE type
      this.reverseVocab.set(id, token);
      this.byteTokens.set(i, id);
    }

    this.vocabSize = this.pieces.size;
  }

  /**
   * Encode text using Unigram or BPE algorithm
   */
  encode(text: string): number[] {
    if (!this.modelData && this.pieces.size === 0) {
      throw new Error('SentencePiece model not loaded');
    }

    const ids: number[] = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Normalize: add sentence piece prefix (▁ for word start)
    const normalized = text.replace(/ /g, '▁');
    const prefixed = (text.startsWith(' ') ? '' : '▁') + normalized;

    if (this.algorithm === 'unigram') {
      ids.push(...this._encodeUnigram(prefixed));
    } else {
      ids.push(...this._encodeBPE(prefixed));
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos ?? 2);
    }

    return ids;
  }

  /**
   * Unigram encoding using Viterbi algorithm
   */
  private _encodeUnigram(text: string): number[] {
    const n = text.length;
    if (n === 0) return [];

    // Viterbi: best[i] = {score, prev, tokenLen} for position i
    const best: Array<ViterbiState | null> = new Array(n + 1).fill(null);
    best[0] = { score: 0, prev: -1, tokenLen: 0 };

    for (let i = 0; i < n; i++) {
      if (best[i] === null) continue;

      // Try all possible tokens starting at position i
      for (let len = 1; len <= Math.min(n - i, 32); len++) {
        const substr = text.slice(i, i + len);
        const piece = this.pieces.get(substr);

        if (piece) {
          const newScore = best[i]!.score + piece.score;
          if (best[i + len] === null || newScore > best[i + len]!.score) {
            best[i + len] = { score: newScore, prev: i, tokenLen: len };
          }
        }
      }

      // Byte fallback for single character
      if (best[i + 1] === null) {
        const charCode = text.charCodeAt(i);
        const bytes = new TextEncoder().encode(text[i]);
        // Use byte tokens with a penalty score
        const byteScore = best[i]!.score - 10 * bytes.length;
        best[i + 1] = { score: byteScore, prev: i, tokenLen: 1, isBytes: true, bytes };
      }
    }

    // Backtrack to get tokens
    const tokens: number[] = [];
    let pos = n;
    while (pos > 0) {
      const state = best[pos]!;
      if (state.isBytes && state.bytes) {
        // Use byte tokens
        for (let j = state.bytes.length - 1; j >= 0; j--) {
          const byteId = this.byteTokens.get(state.bytes[j]);
          tokens.push(byteId ?? this.unkId);
        }
      } else {
        const substr = text.slice(state.prev, pos);
        const piece = this.pieces.get(substr);
        tokens.push(piece?.id ?? this.unkId);
      }
      pos = state.prev;
    }

    return tokens.reverse();
  }

  /**
   * BPE encoding
   */
  private _encodeBPE(text: string): number[] {
    // Start with character-level tokens
    let tokens: string[] = [];
    for (const char of text) {
      const piece = this.pieces.get(char);
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
      let bestPair: string | null = null;
      let bestScore = -Infinity;
      let bestIndex = -1;

      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = tokens[i] + tokens[i + 1];
        const piece = this.pieces.get(merged);
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
      const piece = this.pieces.get(t);
      return piece?.id ?? this.unkId;
    });
  }

  decode(ids: number[], skipSpecialTokens: boolean = true, trim: boolean = true): string {
    if (this.pieces.size === 0) {
      throw new Error('SentencePiece model not loaded');
    }

    const tokens: string[] = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.reverseVocab.get(id);
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

/**
 * Bundled tokenizer for .rdrr format with embedded vocab.
 * Eliminates runtime dependency on transformers.js CDN.
 * Supports both BPE and Unigram (SentencePiece) algorithms.
 */
export class BundledTokenizer extends BaseTokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: string[] = [];
  private mergeRanks: Map<string, number> = new Map();
  private scores: number[] = [];
  private tokenTypes: number[] = [];
  private type: 'bpe' | 'unigram' = 'bpe';
  private byteTokens: Map<number, number> = new Map();
  private specialTokenPatterns: SpecialTokenPattern[] = [];
  private addSpacePrefix: boolean = true;
  /** Space prefix character: 'Ġ' for GPT-style (Llama 3), '▁' for SentencePiece-style (Gemma) */
  private spacePrefixChar: string = '▁';

  constructor(config: TokenizerConfig = {}) {
    super(config);
  }

  /**
   * Load from tokenizer.json content
   * Auto-detects HuggingFace format vs bundled format
   */
  load(tokenizerJson: HuggingFaceTokenizerJson | BundledTokenizerJson): void {
    // Detect format: HuggingFace has model.vocab, bundled has top-level vocab
    const isHuggingFace = 'model' in tokenizerJson && tokenizerJson.model?.vocab !== undefined;

    if (isHuggingFace) {
      this._loadHuggingFaceFormat(tokenizerJson as HuggingFaceTokenizerJson);
    } else {
      this._loadBundledFormat(tokenizerJson as BundledTokenizerJson);
    }
  }

  /**
   * Load HuggingFace tokenizer.json format
   */
  private _loadHuggingFaceFormat(hf: HuggingFaceTokenizerJson): void {
    const model = hf.model!;
    this.type = (model.type?.toLowerCase() as 'bpe' | 'unigram') || 'bpe';
    console.log(`[BundledTokenizer] HuggingFace model.type="${model.type}", using type="${this.type}"`);
    let maxId = -1;

    // Handle vocab based on type
    if (this.type === 'unigram' && Array.isArray(model.vocab)) {
      // Unigram format: [[token, score], ...]
      for (let i = 0; i < model.vocab.length; i++) {
        const [token, score] = model.vocab[i];
        this.vocab.set(token, i);
        this.reverseVocab.set(i, token);
        this.scores.push(score);
        if (i > maxId) maxId = i;

        // Track byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          this.byteTokens.set(byteVal, i);
        }
      }
    } else {
      // BPE format: { token: id }
      for (const [token, id] of Object.entries(model.vocab || {})) {
        const numId = typeof id === 'number' ? id : parseInt(id as string, 10);
        this.vocab.set(token, numId);
        this.reverseVocab.set(numId, token);
        if (Number.isFinite(numId) && numId > maxId) maxId = numId;

        // Track byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          this.byteTokens.set(byteVal, numId);
        }
      }
    }

    this.vocabSize = this.vocab.size;

    // Load merges from model.merges
    if (model.merges && model.merges.length > 0) {
      this.merges = model.merges;
      for (let i = 0; i < this.merges.length; i++) {
        this.mergeRanks.set(this.merges[i], i);
      }
    }

    // Extract special tokens from added_tokens
    this.specialTokens = {
      pad: model.pad_id ?? 0,
      bos: 1,
      eos: 2,
      unk: model.unk_id ?? 0,
    };

    for (const token of hf.added_tokens || []) {
      if (token.special) {
        const content = token.content;
        const id = typeof token.id === 'number' ? token.id : parseInt(token.id as string, 10);
        if (Number.isFinite(id) && id > maxId) maxId = id;
        // Add to vocab if not already there
        if (!this.vocab.has(content)) {
          this.vocab.set(content, id);
          this.reverseVocab.set(id, content);
        }
        // Store for pattern matching during encode (skip single-char tokens)
        if (content.length > 1) {
          this.specialTokenPatterns.push({ content, id });
        }
        // Identify special token types
        if (content === '<bos>' || content === '<s>' || content.includes('bos')) {
          this.specialTokens.bos = id;
        } else if (content === '<eos>' || content === '</s>' || content.includes('eos')) {
          this.specialTokens.eos = id;
        } else if (content === '<pad>' || content.includes('pad')) {
          this.specialTokens.pad = id;
        } else if (content === '<unk>' || content.includes('unk')) {
          this.specialTokens.unk = id;
        }
      }
    }
    // Sort special tokens by length (longest first) for greedy matching
    this.specialTokenPatterns.sort((a, b) => b.content.length - a.content.length);
    // Debug: log special tokens
    console.log('[BundledTokenizer] Special token patterns:', this.specialTokenPatterns.map(t => `${t.id}:"${t.content}"`).join(', '));

    // Some models add special tokens with IDs above the base vocab range.
    // Keep vocabSize aligned to the maximum ID + 1 to match embedding/LM-head shapes.
    if (maxId >= 0) {
      this.vocabSize = Math.max(this.vocabSize, maxId + 1);
    }

    // Handle behavior flags
    this.addBosToken = hf.add_bos_token !== false;
    this.addEosToken = hf.add_eos_token || false;
    // NOTE: Default to FALSE for add_prefix_space - HuggingFace tokenizers typically
    // don't add a space prefix to the first token. Models like Gemma, Llama use
    // the normalizer/pre_tokenizer to handle spaces within text, not at the start.
    this.addSpacePrefix = model.add_prefix_space ?? model.add_dummy_prefix ?? false;

    // Detect space prefix style by checking which WORD tokens exist in vocab
    // GPT-style uses 'Ġ' (U+0120), SentencePiece uses '▁' (U+2581)
    // IMPORTANT: Only check for actual word tokens like 'Ġthe', not single char 'Ġ'
    // because some models (Gemma) have 'Ġ' as a token but use '▁' for actual word prefixes
    const hasGptStyle = this.vocab.has('Ġthe') || this.vocab.has('Ġa') || this.vocab.has('Ġis');
    const hasSentencePieceStyle = this.vocab.has('▁the') || this.vocab.has('▁a') || this.vocab.has('▁is');
    if (hasGptStyle && !hasSentencePieceStyle) {
      this.spacePrefixChar = 'Ġ';
      console.log('[BundledTokenizer] Detected GPT-style space prefix (Ġ)');
    } else if (hasSentencePieceStyle && !hasGptStyle) {
      this.spacePrefixChar = '▁';
      console.log('[BundledTokenizer] Detected SentencePiece-style space prefix (▁)');
    } else if (hasGptStyle && hasSentencePieceStyle) {
      // Both styles exist - prefer GPT style for Llama-family models
      this.spacePrefixChar = 'Ġ';
      console.log('[BundledTokenizer] Both space styles found, defaulting to GPT-style (Ġ)');
    } else {
      // Neither style found - might be byte-level or no space prefix needed
      this.addSpacePrefix = false;
      console.log('[BundledTokenizer] No space prefix tokens found, disabling space prefix');
    }

    console.log(`[BundledTokenizer] Loaded HuggingFace ${this.vocabSize} tokens (${this.type}), ${this.specialTokenPatterns.length} special patterns, ${this.merges.length} merges, spacePrefixChar='${this.spacePrefixChar}'`);
    // Debug: show sample vocab entries (look for common words)
    const commonWords = ['the', '▁the', 'Ġthe', 'a', '▁a', 'is', '▁is', 'user', '▁user', 'u', 's', 'e', 'r'];
    const foundTokens = commonWords.map(w => {
      const id = this.vocab.get(w);
      return id !== undefined ? `"${w}"=${id}` : null;
    }).filter(Boolean);
    console.log('[BundledTokenizer] Common tokens in vocab:', foundTokens.join(', ') || 'NONE FOUND');
    // Show first few merges (escape whitespace)
    if (this.merges.length > 0) {
      const escapedMerges = this.merges.slice(0, 5).map(m =>
        String(m).replace(/\n/g, '\\n').replace(/\r/g, '\\r').replace(/ /g, '␣')
      );
      console.log('[BundledTokenizer] First 5 merges:', escapedMerges.join(' | '));
    }
  }

  /**
   * Load bundled (GGUF-extracted) tokenizer.json format
   */
  private _loadBundledFormat(tokenizerJson: BundledTokenizerJson): void {
    this.type = (tokenizerJson.type as 'bpe' | 'unigram') || 'bpe';

    // Build vocab maps
    for (const [token, id] of Object.entries(tokenizerJson.vocab)) {
      const numId = typeof id === 'number' ? id : parseInt(id as string, 10);
      this.vocab.set(token, numId);
      this.reverseVocab.set(numId, token);

      // Track byte tokens for fallback
      if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
        const byteVal = parseInt(token.slice(3, 5), 16);
        this.byteTokens.set(byteVal, numId);
      }
    }

    this.vocabSize = this.vocab.size;

    // Load merges for BPE
    if (tokenizerJson.merges && tokenizerJson.merges.length > 0) {
      this.merges = tokenizerJson.merges;
      for (let i = 0; i < this.merges.length; i++) {
        this.mergeRanks.set(this.merges[i], i);
      }
    }

    // Load scores for Unigram
    if (tokenizerJson.scores && tokenizerJson.scores.length > 0) {
      this.scores = tokenizerJson.scores;
    }

    // Load token types if available
    if (tokenizerJson.tokenTypes) {
      this.tokenTypes = tokenizerJson.tokenTypes;
    }

    // Set special tokens - support both camelCase and snake_case formats
    const specialTokensRaw = (tokenizerJson.specialTokens || (tokenizerJson as unknown as Record<string, unknown>).special_tokens) as Record<string, number | undefined> | undefined;
    if (specialTokensRaw) {
      this.specialTokens = {
        pad: specialTokensRaw.pad ?? specialTokensRaw.pad_token_id ?? 0,
        bos: specialTokensRaw.bos ?? specialTokensRaw.bos_token_id ?? 1,
        eos: specialTokensRaw.eos ?? specialTokensRaw.eos_token_id ?? 2,
        unk: specialTokensRaw.unk ?? specialTokensRaw.unk_token_id ?? 0,
      };
      console.log(`[BundledTokenizer] Special tokens: BOS=${this.specialTokens.bos}, EOS=${this.specialTokens.eos}`);
    }

    this.addBosToken = tokenizerJson.addBosToken !== false;
    this.addEosToken = tokenizerJson.addEosToken || false;
    // NOTE: Default to FALSE - first word shouldn't get space prefix
    // Space prefixes are only for words that follow a space in original text
    this.addSpacePrefix = tokenizerJson.addSpacePrefix === true;

    // Detect space prefix style based on vocab tokens
    // GPT-style uses 'Ġ' (U+0120), SentencePiece uses '▁' (U+2581)
    // IMPORTANT: Only check for actual word tokens like 'Ġthe', not single char 'Ġ'
    // because some models (Gemma) have 'Ġ' as a token but use '▁' for actual word prefixes
    const hasGptStyle = this.vocab.has('Ġthe') || this.vocab.has('Ġa') || this.vocab.has('Ġis');
    const hasSentencePieceStyle = this.vocab.has('▁the') || this.vocab.has('▁a') || this.vocab.has('▁is');

    if (hasGptStyle && !hasSentencePieceStyle) {
      this.spacePrefixChar = 'Ġ';
      console.log('[BundledTokenizer] Detected GPT-style space prefix (Ġ)');
    } else if (hasSentencePieceStyle && !hasGptStyle) {
      this.spacePrefixChar = '▁';
      console.log('[BundledTokenizer] Detected SentencePiece-style space prefix (▁)');
    } else if (hasGptStyle && hasSentencePieceStyle) {
      // Both exist - prefer GPT-style for Llama 3 compatibility
      this.spacePrefixChar = 'Ġ';
      console.log('[BundledTokenizer] Both space prefix styles found, using GPT-style (Ġ)');
    } else {
      // Default to SentencePiece style
      this.spacePrefixChar = '▁';
      console.log('[BundledTokenizer] No space prefix tokens found, defaulting to SentencePiece-style (▁)');
    }

    console.log(`[BundledTokenizer] Loaded ${this.vocabSize} tokens (${this.type})`);
  }

  encode(text: string): number[] {
    if (this.vocab.size === 0) {
      throw new Error('BundledTokenizer not loaded');
    }

    const ids: number[] = [];

    if (this.addBosToken) {
      ids.push(this.specialTokens.bos ?? 1);
    }

    // Split text around special tokens and tokenize each segment
    const segments = this._splitOnSpecialTokens(text);
    for (const seg of segments) {
      if (seg.isSpecial && seg.id !== undefined) {
        ids.push(seg.id);
      } else if (seg.text && seg.text.length > 0) {
        if (this.type === 'unigram') {
          ids.push(...this._encodeUnigram(seg.text));
        } else {
          ids.push(...this._encodeBPE(seg.text));
        }
      }
    }

    if (this.addEosToken) {
      ids.push(this.specialTokens.eos ?? 2);
    }

    return ids;
  }

  /**
   * Split text around special tokens for proper encoding
   */
  private _splitOnSpecialTokens(text: string): TextSegment[] {
    if (this.specialTokenPatterns.length === 0) {
      return [{ text, isSpecial: false }];
    }

    const segments: TextSegment[] = [];
    let remaining = text;

    while (remaining.length > 0) {
      // Find the EARLIEST special token match
      let earliestIdx = Infinity;
      let earliestToken: SpecialTokenPattern | null = null;

      for (const { content, id } of this.specialTokenPatterns) {
        const idx = remaining.indexOf(content);
        if (idx !== -1 && idx < earliestIdx) {
          earliestIdx = idx;
          earliestToken = { content, id };
        }
      }

      if (earliestToken === null) {
        // No special tokens found, rest is plain text
        segments.push({ text: remaining, isSpecial: false });
        break;
      }

      if (earliestIdx === 0) {
        // Special token at start
        segments.push({ id: earliestToken.id, isSpecial: true });
        remaining = remaining.slice(earliestToken.content.length);
      } else {
        // Text before special token
        segments.push({ text: remaining.slice(0, earliestIdx), isSpecial: false });
        segments.push({ id: earliestToken.id, isSpecial: true });
        remaining = remaining.slice(earliestIdx + earliestToken.content.length);
      }
    }

    return segments;
  }

  /**
   * Unigram encoding using Viterbi algorithm
   */
  private _encodeUnigram(text: string): number[] {
    // Normalize: convert spaces to the model's space prefix character
    // This turns "The color of" into "The▁color▁of"
    // Note: we do NOT add an extra prefix at the start - the first word has no prefix
    const sp = this.spacePrefixChar;
    const prefixed = text.replace(/ /g, sp);

    const n = prefixed.length;
    if (n === 0) return [];

    // Viterbi: best[i] = {score, prev, tokenLen} for position i
    const best: Array<ViterbiState | null> = new Array(n + 1).fill(null);
    best[0] = { score: 0, prev: -1, tokenLen: 0 };

    for (let i = 0; i < n; i++) {
      if (best[i] === null) continue;

      // Try all possible tokens starting at position i
      for (let len = 1; len <= Math.min(n - i, 32); len++) {
        const substr = prefixed.slice(i, i + len);
        const tokenId = this.vocab.get(substr);

        if (tokenId !== undefined) {
          const score = this.scores[tokenId] || 0;
          const newScore = best[i]!.score + score;
          if (best[i + len] === null || newScore > best[i + len]!.score) {
            best[i + len] = { score: newScore, prev: i, tokenLen: len };
          }
        }
      }

      // Byte fallback for single character
      if (best[i + 1] === null) {
        const bytes = new TextEncoder().encode(prefixed[i]);
        const byteScore = best[i]!.score - 10 * bytes.length;
        best[i + 1] = { score: byteScore, prev: i, tokenLen: 1, isBytes: true, bytes };
      }
    }

    // Backtrack to get tokens
    const tokens: number[] = [];
    let pos = n;
    while (pos > 0) {
      const state = best[pos]!;
      if (state.isBytes && state.bytes) {
        for (let j = state.bytes.length - 1; j >= 0; j--) {
          const byteId = this.byteTokens.get(state.bytes[j]);
          tokens.push(byteId ?? (this.specialTokens.unk ?? 0));
        }
      } else {
        const substr = prefixed.slice(state.prev, pos);
        const tokenId = this.vocab.get(substr);
        tokens.push(tokenId ?? (this.specialTokens.unk ?? 0));
      }
      pos = state.prev;
    }

    return tokens.reverse();
  }

  /**
   * BPE encoding
   */
  private _encodeBPE(text: string): number[] {
    // Normalize text: ALWAYS convert spaces to the model's space prefix character
    // Use detected space prefix character (Ġ for GPT-style, ▁ for SentencePiece)
    const sp = this.spacePrefixChar;
    // Replace all spaces with the prefix character
    // This turns "The color of" into "The▁color▁of" or "TheĠcolorĠof"
    // Note: we do NOT add an extra prefix at the start - the first word has no prefix
    const prefixed = text.replace(/ /g, sp);

    // Use greedy longest-match tokenization
    // This directly finds tokens like "user", "Ġthe" instead of character-by-character
    const ids: number[] = [];
    let pos = 0;

    while (pos < prefixed.length) {
      let bestLen = 0;
      let bestId = this.specialTokens.unk ?? 0;

      // Try to find the longest matching token starting at pos
      // Limit max token length to 32 chars for efficiency
      const maxLen = Math.min(32, prefixed.length - pos);
      for (let len = maxLen; len >= 1; len--) {
        const substr = prefixed.slice(pos, pos + len);
        const id = this.vocab.get(substr);
        if (id !== undefined) {
          bestLen = len;
          bestId = id;
          break; // Found longest match
        }
      }

      if (bestLen === 0) {
        // No match found - use byte fallback for single character
        const char = prefixed[pos];
        const bytes = new TextEncoder().encode(char);
        for (const b of bytes) {
          const byteToken = `<0x${b.toString(16).padStart(2, '0').toUpperCase()}>`;
          const byteId = this.vocab.get(byteToken);
          ids.push(byteId ?? (this.specialTokens.unk ?? 0));
        }
        pos += 1;
      } else {
        ids.push(bestId);
        pos += bestLen;
      }
    }

    return ids;
  }

  decode(ids: number[], skipSpecialTokens: boolean = true, trim: boolean = true): string {
    if (this.vocab.size === 0) {
      throw new Error('BundledTokenizer not loaded');
    }

    const tokens: string[] = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.isSpecialToken(id)) {
        continue;
      }

      const token = this.reverseVocab.get(id);
      if (token !== undefined) {
        // Handle byte tokens
        if (token.match(/^<0x[0-9A-Fa-f]{2}>$/)) {
          const byteVal = parseInt(token.slice(3, 5), 16);
          tokens.push(String.fromCharCode(byteVal));
        } else {
          tokens.push(token);
        }
      }
    }

    // Join and convert ▁ back to spaces, handle GPT-style markers
    let result = tokens.join('')
      .replace(/▁/g, ' ')
      .replace(/Ġ/g, ' ')
      .replace(/Ċ/g, '\n');

    // Only trim when requested (not during streaming where spaces matter)
    return trim ? result.trim() : result;
  }
}

/**
 * Tokenizer wrapper that auto-detects backend from model manifest
 * This is a thin wrapper over the backend implementations
 */
export class Tokenizer {
  private backend: BaseTokenizer | null = null;
  private config: TokenizerConfig | null = null;

  /**
   * Initialize from model manifest
   */
  async initialize(manifest: ModelManifest, options: { baseUrl?: string } = {}): Promise<void> {
    const tokenizerConfig = manifest.tokenizer || {};

    // Check for bundled or HuggingFace tokenizer first (eliminates transformers.js dependency)
    const isBundled = tokenizerConfig.type === 'bundled' || tokenizerConfig.type === 'huggingface';
    if (isBundled && tokenizerConfig.file) {
      console.log(`[Tokenizer] Loading ${tokenizerConfig.type} tokenizer from ${tokenizerConfig.file}`);
      this.backend = new BundledTokenizer(tokenizerConfig);

      const baseUrl = options.baseUrl;
      let tokenizerJson: HuggingFaceTokenizerJson | BundledTokenizerJson | null = null;

      // Try to load tokenizer.json
      if (baseUrl) {
        // Load from remote URL
        const tokenizerUrl = `${baseUrl}/${tokenizerConfig.file}`;
        try {
          const response = await fetch(tokenizerUrl);
          if (!response.ok) {
            throw new Error(`Failed to fetch tokenizer: ${response.status}`);
          }
          tokenizerJson = await response.json();
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.warn(`[Tokenizer] Failed to fetch bundled tokenizer from URL: ${message}`);
        }
      } else {
        // Try to load from OPFS (for cached models)
        try {
          const { loadTokenizerFromOPFS } = await import('../storage/shard-manager.js');
          const tokenizerStr = await loadTokenizerFromOPFS();
          if (tokenizerStr) {
            tokenizerJson = JSON.parse(tokenizerStr);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          console.warn(`[Tokenizer] Failed to load bundled tokenizer from OPFS: ${message}`);
        }
      }

      if (tokenizerJson) {
        (this.backend as BundledTokenizer).load(tokenizerJson);
        this.config = tokenizerConfig;
        return;
      }

      // No external fallback - bundled tokenizer is required
      throw new Error(
        '[Tokenizer] Bundled tokenizer not found. ' +
        'Ensure tokenizer.json is in OPFS or model directory. ' +
        'Clear browser storage and re-download the model.'
      );
    }

    // Try to infer HuggingFace model ID from manifest if not explicitly set
    let hfModel = tokenizerConfig.hfModel;
    if (!hfModel && !tokenizerConfig.sentencepieceModel && !tokenizerConfig.vocab) {
      hfModel = this._inferHuggingFaceModel(manifest);
    }

    if (hfModel) {
      // Use Transformers.js for HuggingFace models (fallback)
      console.log(`[Tokenizer] Loading from HuggingFace: ${hfModel}`);
      this.backend = new TransformersTokenizer({
        modelId: hfModel,
        ...tokenizerConfig
      });
      await (this.backend as TransformersTokenizer).load(hfModel);
    } else if (tokenizerConfig.sentencepieceModel) {
      // Load SentencePiece model
      this.backend = new SentencePieceTokenizer(tokenizerConfig);

      // Load the model data from the provided source
      let modelData: ArrayBuffer | undefined;
      if (tokenizerConfig.sentencepieceModel instanceof ArrayBuffer) {
        modelData = tokenizerConfig.sentencepieceModel;
      } else if (tokenizerConfig.loadShard) {
        // Use provided shard loader
        modelData = await tokenizerConfig.loadShard(tokenizerConfig.sentencepieceModel);
      } else if (typeof tokenizerConfig.sentencepieceModel === 'string') {
        // Try to fetch as URL
        const response = await fetch(tokenizerConfig.sentencepieceModel);
        modelData = await response.arrayBuffer();
      }

      if (modelData) {
        await (this.backend as SentencePieceTokenizer).load(modelData);
      } else {
        throw new Error('Could not load SentencePiece model data');
      }
    } else if (tokenizerConfig.vocab && tokenizerConfig.merges) {
      // BPE with vocab + merges
      this.backend = new BPETokenizer(tokenizerConfig);
      (this.backend as BPETokenizer).load(tokenizerConfig.vocab, tokenizerConfig.merges);
    } else {
      throw new Error('No valid tokenizer configuration in manifest');
    }

    this.config = tokenizerConfig;
  }

  /**
   * Infer HuggingFace model ID from manifest architecture
   */
  private _inferHuggingFaceModel(manifest: ModelManifest): string | null {
    const arch = manifest.architecture || manifest.config?.architectures?.[0] || '';
    const archLower = arch.toLowerCase();

    // Map architecture names to public HuggingFace tokenizer repos
    // Using Xenova's repos where possible as they are optimized for Transformers.js
    const archToHF: Record<string, string> = {
      'gemma3': 'google/gemma-3-4b-it',    // Gemma 3 (fallback only - bundled tokenizer preferred)
      'gemma2': 'Xenova/gemma-tokenizer',
      'gemma': 'Xenova/gemma-tokenizer',
      'llama3': 'Xenova/llama3-tokenizer-new',
      'llama2': 'Xenova/llama2-tokenizer',
      'llama': 'Xenova/llama2-tokenizer',
      // Mistral v0.1/v0.2 has 32000 vocab, v0.3 has 32768 vocab with different tokenization
      'mistral': 'Xenova/mistral-tokenizer-v1',
      'mixtral': 'Xenova/mistral-tokenizer-v1',
      'qwen2': 'Xenova/qwen2.5-0.5b-instruct',
      'qwen': 'Xenova/qwen1.5-0.5b',
      'phi3': 'Xenova/phi-3-mini-4k-instruct',
      'phi': 'Xenova/phi-2',
      'smollm': 'HuggingFaceTB/SmolLM-360M-Instruct',
      'tinyllama': 'Xenova/TinyLlama-1.1B-Chat-v1.0',
    };

    // Check vocab size for Mistral - v0.3 has 32768 vocab with different tokenization
    const vocabSize = manifest.config?.vocab_size ||
                      manifest.config?.text_config?.vocab_size ||
                      manifest.tokenizer?.vocabSize;
    if ((archLower.includes('mistral') || archLower.includes('mixtral')) && vocabSize === 32768) {
      // Mistral v0.3+ with extended vocabulary needs the official tokenizer
      console.log(`[Tokenizer] Detected Mistral v0.3+ (vocab_size=32768), using official tokenizer`);
      return 'mistralai/Mistral-7B-Instruct-v0.3';
    }

    for (const [key, hfModel] of Object.entries(archToHF)) {
      if (archLower.includes(key)) {
        console.log(`[Tokenizer] Inferred HuggingFace model from architecture "${arch}": ${hfModel}`);
        return hfModel;
      }
    }

    // Check model type in config
    const modelType = manifest.config?.model_type || manifest.config?.text_config?.model_type || '';
    if (modelType) {
      for (const [key, hfModel] of Object.entries(archToHF)) {
        if (modelType.toLowerCase().includes(key)) {
          console.log(`[Tokenizer] Inferred HuggingFace model from model_type "${modelType}": ${hfModel}`);
          return hfModel;
        }
      }
    }

    return null;
  }

  /**
   * Encode text to token IDs
   */
  encode(text: string): number[] {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.encode(text);
  }

  /**
   * Decode token IDs to text
   * @param skipSpecialTokens - Whether to skip special tokens in output
   * @param trim - Whether to trim whitespace (default true, set false for streaming)
   */
  decode(ids: number[], skipSpecialTokens: boolean = true, trim: boolean = true): string {
    if (!this.backend) {
      throw new Error('Tokenizer not initialized');
    }
    return this.backend.decode(ids, skipSpecialTokens, trim);
  }

  /**
   * Get special tokens
   */
  getSpecialTokens(): SpecialTokens {
    return this.backend?.specialTokens || {};
  }

  /**
   * Get vocabulary size
   */
  getVocabSize(): number {
    return this.backend?.getVocabSize() || 0;
  }
}

export default Tokenizer;
