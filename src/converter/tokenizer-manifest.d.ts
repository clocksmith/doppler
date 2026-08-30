export declare function resolveTokenizerId(value: unknown): number | null;
export declare function resolveTokenizerIds(value: unknown): number[] | null;
export declare function resolveTokenizerField(
  tokenizerConfig: Record<string, unknown> | null | undefined,
  ...keys: string[]
): unknown;
export declare function resolveConfigBoolean(
  rawConfig: Record<string, unknown> | null | undefined,
  ...keys: string[]
): unknown;
export declare function resolveTokenizerVocabSize(
  tokenizerConfig: Record<string, unknown> | null | undefined,
  rawConfig: Record<string, unknown> | null | undefined,
  architecture: Record<string, unknown> | null | undefined
): number | null;
export declare function resolveConfigTokenId(
  rawConfig: Record<string, unknown> | null | undefined,
  key: string
): number | null;
export declare function resolveConfigTokenIds(
  rawConfig: Record<string, unknown> | null | undefined,
  key: string
): number[] | null;
export declare function buildSentencepieceTokenizer(
  tokenizerConfig: Record<string, unknown> | null,
  rawConfig: Record<string, unknown>,
  architecture: Record<string, unknown>,
  modelTokenizerModel: string | { file?: string } | null
): Record<string, unknown> | null;
export declare function resolveBundledTokenizerVocabSize(tokenizerJson: unknown): number;
export declare function buildBundledTokenizer(
  tokenizerJson: Record<string, unknown>,
  tokenizerConfig: Record<string, unknown> | null,
  rawConfig: Record<string, unknown>,
  generationConfig?: Record<string, unknown> | null
): Record<string, unknown>;
