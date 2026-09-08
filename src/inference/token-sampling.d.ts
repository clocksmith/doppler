import type { ResolvedGenerationOptions } from '../config/generation-contract.js';
export interface SamplingCandidate { token: number; logit: number; prob: number; }
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
