import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { snapshotChoiceScoringRequest, validateChoiceScoringResult } from './choice-scoring.js';

export const CHOICE_SCORING_REFERENCE_SCHEMA_ID = 'doppler.choice-scoring-reference-transcript/v1';
const digest = /^sha256:[0-9a-f]{64}$/u;
/** @param {unknown} condition @param {string} message */
function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid choice scoring reference: ${message}`);
}

// Compare qualification observations only. This never changes runtime scores.
/** @param {unknown} value
 * @returns {import('./choice-scoring-reference.js').ChoiceScoringReferenceTranscript} */
export function assertChoiceScoringReferenceTranscript(value) {
  const transcript = /** @type {import('./choice-scoring-reference.js').ChoiceScoringReferenceTranscript} */ (value);
  requireValue(transcript?.schema === CHOICE_SCORING_REFERENCE_SCHEMA_ID
    && transcript.operation === 'scoreChoices', 'schema or operation differs.');
  requireValue(typeof transcript.modelId === 'string' && transcript.modelId.trim()
    && typeof transcript.surface === 'string' && transcript.surface.endsWith('-webgpu')
    && !transcript.surface.startsWith('unknown'), 'exact model and physical surface required.');
  for (const hash of [transcript.manifestHash, transcript.executionGraphHash,
    transcript.source?.hash, transcript.referenceDigest]) requireValue(digest.test(hash ?? ''), 'identity digest required.');
  requireValue(typeof transcript.source?.path === 'string' && transcript.source.path.trim(), 'source path required.');
  const reference = transcript.reference;
  requireValue(reference?.schema === 'doppler.choice-scoring-source-reference/v1'
    && typeof reference.engine === 'string' && reference.engine.trim()
    && reference.engineVersions && Object.keys(reference.engineVersions).length > 0
    && Object.values(reference.engineVersions).every(version => typeof version === 'string' && version.trim()), 'independent engine identity required.');
  requireValue(digest.test(reference.contractHash ?? '') && reference.manifestHash === transcript.manifestHash
    && computeCanonicalSha256(reference) === transcript.referenceDigest, 'reference contract or model identity changed.');
  requireValue(Number.isFinite(reference.maximumAbsoluteLogitError) && reference.maximumAbsoluteLogitError >= 0
    && Number.isSafeInteger(reference.minimumCorrectChoices) && reference.minimumCorrectChoices > 0
    && Array.isArray(reference.cases) && reference.cases.length >= reference.minimumCorrectChoices
    && Array.isArray(transcript.observation?.cases) && transcript.observation.cases.length === reference.cases.length,
  'frozen tolerance and complete observations required.');
  const ids = new Set();
  let referenceCorrect = 0, observedCorrect = 0;
  for (const [index, expected] of reference.cases.entries()) {
    const actual = transcript.observation.cases[index];
    requireValue(typeof expected?.id === 'string' && expected.id.trim() && !ids.has(expected.id)
      && actual?.id === expected.id && actual.expectedId === expected.expectedId, 'case identity or task expectation changed.');
    ids.add(expected.id);
    const request = snapshotChoiceScoringRequest(expected.input);
    requireValue(computeCanonicalSha256(actual.input) === computeCanonicalSha256(request), 'observed request changed.');
    const referenceOutput = validateChoiceScoringResult(request, expected.output);
    const output = validateChoiceScoringResult(request, actual.output);
    requireValue(request.choices.some(choice => choice.id === expected.expectedId), 'expected choice is outside allowed answers.');
    requireValue(Array.isArray(expected.promptTokenIds) && expected.promptTokenIds.length === referenceOutput.promptTokenCount
      && expected.promptTokenIds.every(token => Number.isSafeInteger(token) && token >= 0)
      && computeCanonicalSha256(actual.promptTokenIds) === computeCanonicalSha256(expected.promptTokenIds)
      && output.promptTokenCount === referenceOutput.promptTokenCount, 'prompt token identity differs.');
    requireValue(output.selectedId === referenceOutput.selectedId, 'selected answer differs from independent reference.');
    for (const [choiceIndex, choice] of output.choices.entries()) {
      const baseline = referenceOutput.choices[choiceIndex];
      requireValue(choice.tokenId === baseline.tokenId
        && Math.abs(choice.logit - baseline.logit) <= reference.maximumAbsoluteLogitError, 'answer token or logit exceeds tolerance.');
    }
    referenceCorrect += Number(referenceOutput.selectedId === expected.expectedId);
    observedCorrect += Number(output.selectedId === expected.expectedId);
  }
  requireValue(referenceCorrect >= reference.minimumCorrectChoices && observedCorrect >= reference.minimumCorrectChoices,
    'declared task quality gate failed.');
  requireValue(!Object.hasOwn(transcript, 'tokens') && !Object.hasOwn(transcript, 'generationConfig'),
    'generation evidence cannot qualify choice scoring.');
  return transcript;
}
