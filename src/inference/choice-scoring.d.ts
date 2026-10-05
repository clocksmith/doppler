import type { ChoiceScoringRequest, ChoiceScoringResult } from '../config/choice-scoring.js';
export interface ChoiceScoringPipeline {
  tokenizer: { encode(text: string): number[] | Uint32Array };
  prefillWithTokenLogits(prompt: string, tokenIds: number[], options: {
    inputIds: number[]; useChatTemplate: false; maxSeqLen: number; signal?: AbortSignal;
  }): Promise<{ logits: ArrayLike<number>; tokens: ArrayLike<number> }>;
  resetGenerationState(): void;
}
export function scoreModelChoices(pipeline: ChoiceScoringPipeline, input: ChoiceScoringRequest,
  control?: { signal?: AbortSignal }): Promise<ChoiceScoringResult>;
