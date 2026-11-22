import type { TargetSequenceScoringRequest, TargetSequenceScoreResult } from '../config/target-sequence-scoring.js';
export interface TargetSequenceScoringPipeline {
  tokenizer: { encode(text: string): number[] | Uint32Array };
  /** Read for `messages` requests; formatting uses this template in non-thinking mode. */
  modelConfig?: { chatTemplateType?: string | null } | null;
  prefillWithTargetTokenLogProbs(prompt: string, targetTokenIds: number[], options: {
    inputIds: number[]; useChatTemplate: false; maxSeqLen: number; signal?: AbortSignal;
  }): Promise<{ tokens: ArrayLike<number>; targetTokenIds: ArrayLike<number>; logProbs: ArrayLike<number> }>;
  resetGenerationState(): void;
}
export function resolveTargetSequencePrompt(pipeline: TargetSequenceScoringPipeline,
  request: Readonly<TargetSequenceScoringRequest>): string;
export function tokenizeTargetContinuation(pipeline: TargetSequenceScoringPipeline, prompt: string, target: string,
  maxSeqLen: number): { promptTokens: number[]; targetTokens: number[] };
export function scoreTargetSequence(pipeline: TargetSequenceScoringPipeline, input: TargetSequenceScoringRequest,
  control?: { signal?: AbortSignal }): Promise<TargetSequenceScoreResult>;
