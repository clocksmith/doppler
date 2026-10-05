import { CHOICE_SCORING_REFERENCE_SCHEMA_ID, assertChoiceScoringReferenceTranscript } from '../../config/choice-scoring-reference.js';

/** @param {unknown} input
 * @param {{ path: string; hash: string }} artifact
 * @param {string} executionGraphHash */
export function buildChoiceScoringReferenceTranscript(input, artifact, executionGraphHash) {
  const report = /** @type {import('./choice-scoring-reference.js').ChoiceScoringQualificationReport} */ (input);
  if (report?.schema !== 'doppler.choiceScoringModelQualification.v1' || report.passed !== true
    || report.runtime?.executionGraphHash !== executionGraphHash) {
    throw new Error('Program Bundle requires passed choice scoring qualification for its exact execution graph.');
  }
  const transcript = assertChoiceScoringReferenceTranscript({
    schema: CHOICE_SCORING_REFERENCE_SCHEMA_ID, operation: 'scoreChoices',
    modelId: report.model.modelId, surface: report.runtime.surface, executionGraphHash,
    manifestHash: report.model.manifestHash,
    source: { kind: 'choice-scoring-qualification', path: artifact.path, hash: artifact.hash },
    reference: structuredClone(report.reference), referenceDigest: report.referenceDigest,
    observation: structuredClone(report.observation),
  });
  return { artifact, transcript, adapter: { source: 'reference-report',
    surface: report.runtime.surface, deviceInfo: report.runtime.adapterInfo } };
}
