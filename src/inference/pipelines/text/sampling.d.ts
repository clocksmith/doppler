/**
 * Token sampling from logits with temperature, top-k, top-p, and repetition penalty.
 */

export interface SamplingOptions {
  temperature: number;
  topP: number;
  topK: number;
  decode?: (tokens: number[]) => string;
  debug?: boolean;
  padTokenId?: number;
}

export interface TokenCandidate {
  token: number;
  prob: number;
}

export interface TopKResult {
  token: number;
  logit: number;
  prob: number;
  text: string;
}

export interface LogitStats {
  min: number;
  max: number;
  nanCount: number;
  infCount: number;
  top5: TopKResult[];
}

export function applyRepetitionPenalty(
  logits: Float32Array,
  previousTokens: number[],
  penalty: number
): void;

export function softmax(logits: Float32Array): Float32Array;

export function sample(logits: Float32Array, opts: SamplingOptions): number;

export function getTopK(
  logits: Float32Array,
  k?: number,
  decode?: (tokens: number[]) => string
): TopKResult[];

export function logitsSanity(
  logits: Float32Array,
  label: string,
  decode?: (tokens: number[]) => string
): LogitStats;
