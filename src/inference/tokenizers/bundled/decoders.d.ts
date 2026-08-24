export interface ByteLevelCodec {
  decoder: Map<string, number>;
  encoder: Map<number, string>;
}

export interface ByteLevelPretokenizerConfig {
  useByteLevel: boolean;
  addPrefixSpace: boolean | null;
}

export function resolveByteLevelPretokenizerConfig(
  preTokenizer: unknown
): ByteLevelPretokenizerConfig;

export function parseByteTokenValue(token: unknown): number | null;

export function decodeByteFallbackTokens(tokens: string[]): string;

export function createByteLevelCodec(): ByteLevelCodec;

export function encodeByteLevelText(
  text: string,
  encoder: Map<number, string> | null
): string;

export function decodeByteLevelTokens(
  tokens: string[],
  decoder: Map<string, number>
): string;
