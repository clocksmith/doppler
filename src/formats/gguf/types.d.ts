/**
 * Shared GGUF parser (browser + tools).
 */

export declare const GGUFValueType: {
  readonly UINT8: 0;
  readonly INT8: 1;
  readonly UINT16: 2;
  readonly INT16: 3;
  readonly UINT32: 4;
  readonly INT32: 5;
  readonly FLOAT32: 6;
  readonly BOOL: 7;
  readonly STRING: 8;
  readonly ARRAY: 9;
  readonly UINT64: 10;
  readonly INT64: 11;
  readonly FLOAT64: 12;
};

export type GGUFValueTypeId = (typeof GGUFValueType)[keyof typeof GGUFValueType];

export declare const GGMLType: {
  readonly F32: 0;
  readonly F16: 1;
  readonly Q4_0: 2;
  readonly Q4_1: 3;
  readonly Q5_0: 6;
  readonly Q5_1: 7;
  readonly Q8_0: 8;
  readonly Q8_1: 9;
  readonly Q2_K: 10;
  readonly Q3_K: 11;
  readonly Q4_K: 12;
  readonly Q5_K: 13;
  readonly Q6_K: 14;
  readonly Q8_K: 15;
  readonly IQ2_XXS: 16;
  readonly IQ2_XS: 17;
  readonly IQ3_XXS: 18;
  readonly IQ1_S: 19;
  readonly IQ4_NL: 20;
  readonly IQ3_S: 21;
  readonly IQ2_S: 22;
  readonly IQ4_XS: 23;
  readonly I8: 24;
  readonly I16: 25;
  readonly I32: 26;
  readonly I64: 27;
  readonly F64: 28;
  readonly BF16: 29;
};

export type GGMLTypeId = (typeof GGMLType)[keyof typeof GGMLType];

export declare const GGMLTypeName: Record<number, string>;

export declare const GGML_BLOCK_SIZE: Record<number, number>;

export declare const GGML_TYPE_SIZE: Record<number, number>;

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

export declare function getTensor(parsed: GGUFParseResult, name: string): GGUFTensor | null;

export declare function getTensors(parsed: GGUFParseResult, pattern: RegExp): GGUFTensor[];

export declare function groupTensorsByLayer(parsed: GGUFParseResult): Map<number, GGUFTensor[]>;

export declare function identifyMoETensors(parsed: GGUFParseResult): Map<number, Map<number, GGUFTensor[]>>;
