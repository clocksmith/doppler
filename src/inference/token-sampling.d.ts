import type { ResolvedGenerationOptions } from '../config/generation-contract.js';
import type { CanonicalGenerationPolicy } from '../config/generation-contract.js';
export interface SamplingCandidate { token: number; logit: number; prob: number; }
/** Derive one existing scalar draw from an immutable seed and generated-token index. */
export function samplingSeedForToken(seed: number | undefined, tokenIndex: number): number | undefined;
export type TokenSamplingOptions = Pick<ResolvedGenerationOptions, 'temperature' | 'topP' | 'topK'> & {
  seed?: number;
  padTokenId?: number | null;
  suppressTokenIds?: number[];
  onCandidates?: (candidates: SamplingCandidate[]) => void;
};
export function applyRepetitionPenalty(logits: Float32Array, tokens: number[], penalty: number, windowSize: number): void;
export function applyPresencePenalty(logits: Float32Array, tokens: number[], penalty: number, windowSize: number): void;
export function softmax(logits: Float32Array): Float32Array;
export function sample(logits: Float32Array, options: TokenSamplingOptions): number;
export function sampleCanonicalLogits(logits: Iterable<number> & ArrayLike<number>, history: readonly number[],
  policy: CanonicalGenerationPolicy, tokenIndex: number, purpose: 0 | 1): number;
export function selectCanonicalCandidate(candidates: readonly SamplingCandidate[], uniform: number): number;
