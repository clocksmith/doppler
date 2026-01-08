/**
 * RDRR Tokenizer Writer
 *
 * @module converter/writer/tokenizer-writer
 */

import type { TokenizerConfig, HuggingFaceTokenizer } from './types.js';

export interface TokenizerManifestEntry {
  type?: string;
  file?: string;
  vocabSize?: number;
  tokenizerType?: string;
  model?: string;
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
}

/**
 * Writes tokenizer configuration files.
 */
export declare class TokenizerWriter {
  constructor(outputDir: string);

  writeTokenizer(tokenizer: TokenizerConfig): Promise<TokenizerManifestEntry>;
  writeHuggingFaceTokenizer(tokenizerJson: HuggingFaceTokenizer): Promise<TokenizerManifestEntry | null>;
}
