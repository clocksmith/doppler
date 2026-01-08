/**
 * Tokenizer Types and Interfaces
 *
 * @module inference/tokenizers/types
 */

import type { SpecialTokens } from '../../types/inference.js';

export type { SpecialTokens };

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
  /** Padding token ID (alternate field) */
  padTokenId?: number;
  /** Beginning of sequence token ID */
  bosToken?: number;
  /** Beginning of sequence token ID (alternate field) */
  bosTokenId?: number;
  /** End of sequence token ID */
  eosToken?: number;
  /** End of sequence token IDs (string or numeric) */
  eosTokens?: Array<string | number>;
  /** End of sequence token ID (alternate field) */
  eosTokenId?: number;
  /** Unknown token ID */
  unkToken?: number;
  /** Unknown token ID (alternate field) */
  unkTokenId?: number;
  /** Whether to add BOS token */
  addBosToken?: boolean;
  /** Whether to add EOS token */
  addEosToken?: boolean;
  /** Defer special token validation until tokenizer data is loaded */
  deferSpecialTokens?: boolean;
  /** HuggingFace model ID */
  modelId?: string;
  hfModel?: string;
  /** Allow architecture-based HuggingFace fallback */
  allowArchFallback?: boolean;
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

/** Tokenizer Backend Interface */
export interface TokenizerBackend {
  /** Encode text to token IDs */
  encode(text: string, addSpecial?: boolean): number[];
  /** Decode token IDs to text */
  decode(tokens: number[], skipSpecial?: boolean, trim?: boolean): string;
  /** Get vocabulary size */
  getVocabSize(): number;
  /** Get special token IDs */
  specialTokens: SpecialTokens;
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

/** Transformers.js tokenizer type (from external library) */
export interface TransformersTokenizerType {
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
export interface HuggingFaceTokenizerJson {
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
export interface BundledTokenizerJson {
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

/** Viterbi state for Unigram encoding */
export interface ViterbiState {
  score: number;
  prev: number;
  tokenLen: number;
  isBytes?: boolean;
  bytes?: Uint8Array;
}

/** Special token pattern for matching */
export interface SpecialTokenPattern {
  content: string;
  id: number;
}

/** Text segment (plain text or special token) */
export interface TextSegment {
  text?: string;
  id?: number;
  isSpecial: boolean;
}
