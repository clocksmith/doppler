import type { ResolvedGenerationOptions, GenerationStoppingReason } from '../config/generation-contract.js';
import type { CanonicalGenerationPolicy } from '../config/generation-contract.js';
export function sampleCanonicalGenerationStep(sourceLogits: Iterable<number> & ArrayLike<number>, contextTokens: readonly number[],
  policy: CanonicalGenerationPolicy, acceptedGeneratedTokenCount: number): number;
export interface GenerationTokenContract {
  reasoningOutput?: import('../config/reasoning-output.js').ReasoningOutputContract;
  padTokenId?: number | null;
  eosTokenId?: number | null;
  stopTokenIds?: readonly number[];
}
export function sampleCapsuleLogits(sourceLogits: ArrayLike<number>, contextTokens: number[],
  options: ResolvedGenerationOptions, tokenContract?: GenerationTokenContract): number;
export function stoppingReason(tokenId: number, generatedCount: number, options: ResolvedGenerationOptions,
  tokenContract: GenerationTokenContract, getText: () => string): GenerationStoppingReason | null;
