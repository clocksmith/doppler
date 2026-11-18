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
  penalty: number,
  windowSize?: number
): void;

export function applyPresencePenalty(
  logits: Float32Array,
  previousTokens: number[],
  penalty: number,
  windowSize?: number
): void;

/**
 * Applies the caller's optional pre-sample logit mask (`opts.logitMaskFn`) to post-penalty
 * logits, in place. A no-op without a mask. The hook must be synchronous and sees the tokens
 * generated so far, excluding the prompt; `opts.promptTokenCount` must mark that boundary.
 */
export function applyLogitMask(
  logits: Float32Array,
  currentIds: number[],
  opts: {
    promptTokenCount?: number;
    logitMaskFn?: ((logits: Float32Array, context: { generatedIds: number[]; tokenizer: unknown; vocabSize: number }) => void) | null;
  },
  state: { tokenizer?: unknown; modelConfig?: { vocabSize?: number } | null } | null
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
