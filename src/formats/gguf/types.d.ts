/**
 * Shared GGUF parser (browser + tools).
 */

export interface GGUFTensor {
  name: string;
  shape: number[];
  dtype: string;
  dtypeId: number;
  offset: number;
  size: number;
}

export interface GGUFTokenizer {
  model?: string;
  tokens?: string[];
  scores?: number[];
  tokenTypes?: number[];
  merges?: string[];
  bosTokenId?: number;
  eosTokenId?: number;
  padTokenId?: number;
  unkTokenId?: number;
  sepTokenId?: number;
  clsTokenId?: number;
  maskTokenId?: number;
  addBosToken?: boolean;
  addEosToken?: boolean;
  addSpacePrefix?: boolean;
}

export interface GGUFConfig {
  architecture: string;
  vocabSize?: number;
  contextLength?: number;
  embeddingLength?: number;
  blockCount?: number;
  feedForwardLength?: number;
  attentionHeadCount?: number;
  attentionHeadCountKV?: number;
  attentionLayerNormEpsilon?: number;
  attentionLayerNormRMSEpsilon?: number;
  ropeFreqBase?: number;
  ropeScalingType?: string;
  ropeScalingFactor?: number;
  expertCount?: number;
  expertUsedCount?: number;
  tokenizer: GGUFTokenizer;
  /** Allow additional unknown fields from GGUF header */
  [key: string]: unknown;
}

export interface GGUFParseResult {
  version: number;
  architecture: string;
  modelName: string;
  metadata: Record<string, unknown>;
  config: GGUFConfig;
  tensors: GGUFTensor[];
  quantization: string;
  tensorDataOffset: number;
  totalTensorSize: number;
  headerSize: number;
  fileSize?: number;
  filePath?: string;
}

export type ParsedGGUF = GGUFParseResult;

export declare function parseGGUF(buffer: ArrayBuffer): GGUFParseResult;

export declare function parseGGUFHeader(buffer: ArrayBuffer): GGUFParseResult;

export declare function groupTensorsByLayer(parsed: GGUFParseResult): Map<number, GGUFTensor[]>;
