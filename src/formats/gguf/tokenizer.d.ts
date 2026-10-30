export interface GGUFTokenizerAssets {
  tokenizerJson: Record<string, unknown>;
  tokenizerConfig: Record<string, unknown>;
}
export declare function buildGGUFTokenizer(metadata: Record<string, unknown>): GGUFTokenizerAssets | null;
