export interface TargetSequenceScoringMessage {
  role: 'system' | 'user';
  content: string;
}
export type TargetSequenceScoringRequest = {
  /** Model-ready text ending where the target begins; no chat template is applied. */
  prompt: string;
  messages?: never;
  /** Exact continuation to score; never trimmed or normalized. */
  target: string;
  maxSeqLen: number;
} | {
  /** Formatted with the model's chat template in non-thinking mode, ending at the assistant turn. */
  messages: readonly TargetSequenceScoringMessage[];
  prompt?: never;
  target: string;
  maxSeqLen: number;
};
export interface TargetSequenceScoreResult {
  schema: 'doppler.target-sequence-score/v1';
  interpretation: 'teacher-forced-target-log-probability';
  /** Number of target tokens (the canonical tokenization of the target in the prompt's context). */
  tokenCount: number;
  promptTokenCount: number;
  targetTokenIds: number[];
  /** Natural-log log-softmax of each target token over the complete vocabulary row. */
  tokenLogProbs: number[];
  /** Index-order float64 sum of tokenLogProbs. */
  totalLogProb: number;
  /** totalLogProb / tokenCount. */
  meanLogProb: number;
  /** totalLogProb rounded to 0.01 (half toward +Infinity); the only value intended for comparison. */
  roundedTotalLogProb: number;
}
export const TARGET_SEQUENCE_SCORING_CONTRACT: Readonly<{
  schema: 'doppler.target-sequence-scoring-contract/v1';
  resultSchema: 'doppler.target-sequence-score/v1';
  interpretation: 'teacher-forced-target-log-probability';
  logBase: 'e';
  chatTemplateThinking: false;
  messageRoles: readonly ('system' | 'user')[];
  maxTargetTokens: number;
  roundingStep: number;
}>;
export function sumTargetSequenceLogProbs(tokenLogProbs: readonly number[]): number;
export function roundTargetSequenceLogProb(totalLogProb: number): number;
export function snapshotTargetSequenceScoringRequest(request: unknown): Readonly<TargetSequenceScoringRequest>;
export function validateTargetSequenceScoreResult(input: TargetSequenceScoringRequest, result: unknown): TargetSequenceScoreResult;
