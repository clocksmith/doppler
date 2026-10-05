import type { ChoiceScoringReferenceTranscript } from '../../config/choice-scoring-reference.js';
export interface ChoiceScoringQualificationReport {
  schema: 'doppler.choiceScoringModelQualification.v1'; passed: boolean;
  model: { modelId: string; manifestHash: string };
  runtime: { executionGraphHash: string; surface: string; adapterInfo: unknown };
  reference: ChoiceScoringReferenceTranscript['reference'];
  referenceDigest: string; observation: ChoiceScoringReferenceTranscript['observation'];
}
export function buildChoiceScoringReferenceTranscript(report: unknown,
  artifact: { path: string; hash: string }, executionGraphHash: string): {
  artifact: { path: string; hash: string }; transcript: ChoiceScoringReferenceTranscript;
  adapter: { source: string; surface: string; deviceInfo: unknown };
};
