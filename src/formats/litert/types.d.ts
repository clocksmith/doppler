export declare const LITERTLM_MAGIC: 'LITERTLM';
export declare const LITERT_TASK_DEFAULT_TFLITE_ENTRY: 'TF_LITE_PREFILL_DECODE';
export declare const LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY: 'TOKENIZER_MODEL';
export declare const LITERT_TASK_DEFAULT_METADATA_ENTRY: 'METADATA';

export declare const LITERTLM_SECTION_TYPE: {
  readonly NONE: 0;
  readonly GenericBinaryData: 1;
  readonly Deprecated: 2;
  readonly TFLiteModel: 3;
  readonly SP_Tokenizer: 4;
  readonly LlmMetadataProto: 5;
  readonly HF_Tokenizer_Zlib: 6;
  readonly TFLiteWeights: 7;
};

export declare const LITERTLM_SECTION_TYPE_NAME: Record<number, string>;

export interface LiteRTSource {
  name?: string | null;
  size: number;
  readRange: (offset: number, length: number) => Promise<ArrayBuffer | Uint8Array>;
}

export interface LiteRTTaskEntry {
  name: string;
  compressionMethod: number;
  offset: number;
  size: number;
  compressedSize: number;
  localHeaderOffset: number;
}

export interface ParsedLiteRTTask {
  entries: LiteRTTaskEntry[];
  entryMap: Map<string, LiteRTTaskEntry>;
}

export interface LiteRTLMSectionItem {
  key: string | null;
  valueType: number | null;
  value: string | number | boolean | null;
}

export interface LiteRTLMSection {
  beginOffset: number;
  endOffset: number;
  size: number;
  dataType: number;
  dataTypeName: string;
  items: LiteRTLMSectionItem[];
}

export interface ParsedLiteRTLM {
  majorVersion: number;
  minorVersion: number;
  patchVersion: number;
  headerEndOffset: number;
  sections: LiteRTLMSection[];
}

export declare function parseLiteRTTaskFromSource(source: LiteRTSource): Promise<ParsedLiteRTTask>;

export declare function parseLiteRTLMFromSource(source: LiteRTSource): Promise<ParsedLiteRTLM>;

export declare function findLiteRTLMSectionByType(
  parsed: ParsedLiteRTLM | null | undefined,
  dataTypeName: string
): LiteRTLMSection[];

export declare function findLiteRTLMTFLiteModelSection(
  parsed: ParsedLiteRTLM | null | undefined,
  modelType?: string
): LiteRTLMSection | null;

export declare function findLiteRTLMTFLiteWeightsSection(
  parsed: ParsedLiteRTLM | null | undefined,
  modelType?: string
): LiteRTLMSection | null;

export declare function findLiteRTLMSentencePieceTokenizerSection(
  parsed: ParsedLiteRTLM | null | undefined
): LiteRTLMSection | null;

export declare function findLiteRTLMMetadataSection(
  parsed: ParsedLiteRTLM | null | undefined
): LiteRTLMSection | null;
