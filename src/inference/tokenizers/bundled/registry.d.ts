export interface TokenRegistrySpecialTokens {
  pad: number | null;
  bos: number | null;
  eos: number | null;
  unk: number | null;
}

export interface AddedToken {
  id: number | string;
  content: string;
  special: boolean;
}

export interface AddedTokenPattern {
  content: string;
  id: number;
}

export function resolveSpecialTokens(
  specialTokensRaw: Record<string, unknown> | null | undefined,
  fallbackTokens: Partial<Record<keyof TokenRegistrySpecialTokens, number | string | null>> | null,
  vocab: Map<string, number>,
  options?: { allowMissingEos?: boolean }
): TokenRegistrySpecialTokens;

export function appendVocabEntry(
  token: string,
  id: number | string,
  vocab: Map<string, number>,
  reverseVocab: Map<number, string>,
  byteTokens: Map<number, number>
): number;

export function loadObjectVocab(
  vocabObject: Record<string, number | string>,
  vocab: Map<string, number>,
  reverseVocab: Map<number, string>,
  byteTokens: Map<number, number>
): number;

export function buildMergeRanks(
  merges: Array<string | string[]>,
  mergeRanks: Map<string, number>
): string[];

export function rankFallbackBpeHotTokenIds(
  reverseVocab: Map<number, string>,
  limit: number,
  isSpecialToken?: (tokenId: number) => boolean
): number[];

export function registerAddedTokens(
  addedTokens: AddedToken[],
  vocab: Map<string, number>,
  reverseVocab: Map<number, string>,
  patterns: AddedTokenPattern[],
  specialTokenIds: Set<number>,
  derivedSpecialTokens?: TokenRegistrySpecialTokens | null
): number;
