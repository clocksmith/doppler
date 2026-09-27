import type { ResolvedGenerationOptions, GenerationStoppingReason } from '../config/generation-contract.js';
export interface GenerationTokenContract {
  padTokenId?: number | null;
  eosTokenId?: number | null;
  stopTokenIds?: readonly number[];
}
export function sampleCapsuleLogits(sourceLogits: ArrayLike<number>, contextTokens: number[],
  options: ResolvedGenerationOptions, tokenContract?: GenerationTokenContract): number;
export function stoppingReason(tokenId: number, generatedCount: number, options: ResolvedGenerationOptions,
  tokenContract: GenerationTokenContract, getText: () => string): GenerationStoppingReason | null;
