import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

export const EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID = 'doppler.embedding-reference-transcript/v1';
const DIGEST = /^sha256:[0-9a-f]{64}$/u;
const text = value => typeof value === 'string' && value.trim().length > 0;

function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid embedding reference: ${message}`);
}

function validateRows(rows, inputs, contract, label) {
  requireValue(Array.isArray(rows) && rows.length === inputs.length, `${label} must cover every text.`);
  rows.forEach((row, index) => {
    requireValue(row?.text === inputs[index], `${label}[${index}] input binding differs.`);
    requireValue(Array.isArray(row.tokenIds) && row.tokenIds.length > 0
      && row.tokenIds.every(id => Number.isSafeInteger(id) && id >= 0), `${label}[${index}] requires exact token IDs.`);
    requireValue(Array.isArray(row.embedding) && row.embedding.length === contract.dimension
      && row.embedding.every(Number.isFinite), `${label}[${index}] requires a complete finite vector with declared dimensions.`);
  });
}

export function assertEmbeddingReference(reference) {
  requireValue(reference?.schema === 'doppler.embedding-source-reference/v1', 'unsupported source schema.');
  for (const field of ['checkpointId', 'repository', 'revision', 'engine']) {
    requireValue(text(reference.source?.[field]), `source.${field} is required.`);
  }
  requireValue(/^[0-9a-f]{40}$/u.test(reference.source.revision), 'source revision must be immutable.');
  requireValue(Array.isArray(reference.source.files) && reference.source.files.length > 0
    && reference.source.files.every(file => text(file.path) && DIGEST.test(file.hash)), 'source files require digests.');
  requireValue(Array.isArray(reference.input?.texts) && reference.input.texts.length > 0
    && reference.input.texts.every(text), 'input texts are required.');
  const contract = reference.embeddingContract;
  requireValue(Number.isSafeInteger(contract?.dimension) && contract.dimension > 0
    && ['mean', 'last'].includes(contract.postprocessor?.poolingMode)
    && contract.postprocessor.includePrompt === true && Array.isArray(contract.postprocessor.projections)
    && [null, 'l2'].includes(contract.postprocessor.normalize), 'explicit embedding contract is required.');
  requireValue(Number.isFinite(reference.tolerances?.embeddingMaxAbs)
    && reference.tolerances.embeddingMaxAbs >= 0 && reference.tolerances.tokenIds === 'exact', 'explicit vector tolerance and exact token policy are required.');
  validateRows(reference.outputs, reference.input.texts, contract, 'reference.outputs');
  return reference;
}

export function assertEmbeddingSourceIdentity(identity, reference) {
  for (const [field, expected] of Object.entries({ sourceCheckpointId: reference.source.checkpointId,
    sourceRepo: reference.source.repository, sourceRevision: reference.source.revision })) {
    requireValue(identity?.[field] === expected, `artifactIdentity.${field} must bind the pinned source reference.`);
  }
}

// Qualification observation only: compare all frozen output components; never
// produce a runtime embedding or feed a modified tensor back into execution.
export function evaluateEmbeddingReference(reference, observation) {
  assertEmbeddingReference(reference);
  requireValue(computeCanonicalSha256(observation?.input) === computeCanonicalSha256(reference.input), 'observed inputs differ.');
  requireValue(computeCanonicalSha256(observation.embeddingContract) === computeCanonicalSha256(reference.embeddingContract), 'observed embedding contract differs.');
  validateRows(observation.outputs, reference.input.texts, reference.embeddingContract, 'observation.outputs');
  const checks = [];
  for (const [index, expected] of reference.outputs.entries()) {
    const actual = observation.outputs[index];
    checks.push({ id: `text.${index}.tokens`, passed: computeCanonicalSha256(actual.tokenIds) === computeCanonicalSha256(expected.tokenIds) });
    let maxAbsoluteError = 0;
    for (let component = 0; component < expected.embedding.length; component += 1) {
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual.embedding[component] - expected.embedding[component]));
    }
    const tolerance = reference.tolerances.embeddingMaxAbs;
    checks.push({ id: `text.${index}.embedding`, passed: maxAbsoluteError <= tolerance,
      valueCount: expected.embedding.length, maxAbsoluteError, tolerance });
  }
  return { passed: checks.every(check => check.passed), checks };
}

export function assertEmbeddingReferenceTranscript(transcript) {
  requireValue(transcript?.schema === EMBEDDING_REFERENCE_TRANSCRIPT_SCHEMA_ID
    && transcript.operation === 'embed', 'unsupported transcript schema or operation.');
  for (const field of ['modelId', 'surface']) requireValue(text(transcript[field]), `${field} is required.`);
  for (const digest of [transcript.manifestHash, transcript.executionGraphHash, transcript.source?.hash, transcript.referenceDigest]) {
    requireValue(DIGEST.test(digest ?? ''), 'transcript identity requires SHA-256 digests.');
  }
  requireValue(text(transcript.source?.path), 'source path is required.');
  requireValue(transcript.referenceDigest === computeCanonicalSha256(transcript.reference), 'source reference digest differs.');
  requireValue(transcript.tokens === undefined && transcript.generationConfig === undefined, 'generation evidence cannot qualify embeddings.');
  const result = evaluateEmbeddingReference(transcript.reference, transcript.observation);
  requireValue(result.passed, `source comparison failed: ${result.checks.filter(check => !check.passed).map(check => check.id).join(', ')}.`);
  return transcript;
}
