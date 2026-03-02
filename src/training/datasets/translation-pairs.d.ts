export interface TranslationPair {
  source: string;
  target: string;
}

export interface TranslationTokenizedSample {
  inputIds: number[];
  targetIds: number[];
  source: string;
  target: string;
  text: string;
}

export interface MapTranslationPairsOptions {
  sourceKey?: string;
  targetKey?: string;
}

export interface TokenizeTranslationPairsOptions {
  maxLength?: number | null;
  promptPrefix?: string;
  separator?: string;
}

export declare function mapTranslationPairs(
  records: unknown[],
  options?: MapTranslationPairsOptions
): TranslationPair[];

export declare function tokenizeTranslationPairs(
  tokenizer: { encode: (text: string) => number[] },
  pairs: TranslationPair[],
  options?: TokenizeTranslationPairsOptions
): Promise<TranslationTokenizedSample[]>;
