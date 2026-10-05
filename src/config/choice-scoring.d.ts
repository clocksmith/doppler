export interface ChoiceScoringRequest {
  /** Exact model prompt ending at the answer position; no chat template is applied. */
  prompt: string;
  choices: readonly { id: string; label: string }[];
  maxSeqLen: number;
}
export interface ChoiceScoringResult {
  schema: 'doppler.choice-scores/v1';
  interpretation: 'next-token-logits';
  /** Raw logits are not calibrated correctness probabilities. */
  calibration: null;
  choices: { id: string; label: string; tokenId: number; logit: number }[];
  selectedId: string;
  promptTokenCount: number;
}
export const CHOICE_SCORING_CONTRACT: Readonly<{
  schema: 'doppler.choice-scoring-contract/v1'; resultSchema: 'doppler.choice-scores/v1';
  minChoices: number; maxChoices: number; interpretation: 'next-token-logits'; calibration: null;
}>;
export function snapshotChoiceScoringRequest(request: unknown): Readonly<ChoiceScoringRequest>;
export function validateChoiceScoringResult(input: ChoiceScoringRequest, result: unknown): ChoiceScoringResult;
