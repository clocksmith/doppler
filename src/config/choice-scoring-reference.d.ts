import type { ChoiceScoringRequest, ChoiceScoringResult } from './choice-scoring.js';
export interface ChoiceScoringReferenceRow {
  id: string; input: ChoiceScoringRequest; output: ChoiceScoringResult;
  promptTokenIds: number[]; expectedId: string;
}
export interface ChoiceScoringReferenceTranscript {
  schema: 'doppler.choice-scoring-reference-transcript/v1'; operation: 'scoreChoices';
  modelId: string; surface: string; executionGraphHash: string; manifestHash: string;
  source: { kind: string; path: string; hash: string };
  referenceDigest: string;
  reference: {
    schema: 'doppler.choice-scoring-source-reference/v1';
    engine: string; engineVersions: Record<string, string>;
    contractHash: string; manifestHash: string;
    maximumAbsoluteLogitError: number; minimumCorrectChoices: number;
    cases: ChoiceScoringReferenceRow[];
  };
  observation: { cases: ChoiceScoringReferenceRow[] };
}
export const CHOICE_SCORING_REFERENCE_SCHEMA_ID: 'doppler.choice-scoring-reference-transcript/v1';
export function assertChoiceScoringReferenceTranscript(value: unknown): ChoiceScoringReferenceTranscript;
