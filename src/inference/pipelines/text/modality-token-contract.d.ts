export declare function resolveSingleSpecialTokenId(tokenizer: any, tokenText: string, label: string): number;
export declare function expandImagePlaceholderTokenIds(
  tokenIds: ArrayLike<number>,
  imageTokenId: number,
  numImageTokens: number,
  options?: { boiTokenId?: number | null; eoiTokenId?: number | null }
): { inputIds: Int32Array; imageStartOffset: number };
export declare function buildConservativeMultimodalGenerationOptions<T extends object>(options?: T): T & {
  disableCommandBatching: true;
  disableMultiTokenDecode: true;
  stopCheckMode: 'per-token';
};
export declare function resolveMultimodalMaxTokens(runtimeConfig: any, requestedMaxTokens?: number): number;
