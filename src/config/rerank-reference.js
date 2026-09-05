import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

export const RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID = 'doppler.rerank-reference-transcript/v1';
const DIGEST = /^sha256:[0-9a-f]{64}$/u;

function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid rerank reference: ${message}`);
}

function text(value) { return typeof value === 'string' && value.trim().length > 0; }

function validateRows(rows, documents, label) {
  requireValue(Array.isArray(rows) && rows.length === documents.length, `${label} must cover every document.`);
  rows.forEach((row, index) => {
    requireValue(row.index === index && row.document === documents[index], `${label}[${index}] input binding differs.`);
    requireValue(Array.isArray(row.tokenIds) && row.tokenIds.length > 0
      && row.tokenIds.every((id) => Number.isSafeInteger(id) && id >= 0), `${label}[${index}] requires exact token IDs.`);
    for (const field of ['trueLogit', 'falseLogit', 'score', 'probability']) {
      requireValue(Number.isFinite(row[field]), `${label}[${index}].${field} must be finite.`);
    }
    requireValue(row.probability >= 0 && row.probability <= 1, `${label}[${index}] probability is outside [0, 1].`);
  });
}

export function assertRerankReference(reference) {
  requireValue(reference?.schema === 'doppler.rerank-source-reference/v1', 'unsupported source schema.');
  for (const field of ['checkpointId', 'repository', 'revision', 'engine']) {
    requireValue(text(reference.source?.[field]), `source.${field} is required.`);
  }
  requireValue(/^[0-9a-f]{40}$/u.test(reference.source.revision), 'source revision must be immutable.');
  requireValue(Array.isArray(reference.source.files) && reference.source.files.length > 0
    && reference.source.files.every((file) => text(file.path) && DIGEST.test(file.hash)), 'source files require digests.');
  requireValue(text(reference.input?.query), 'query is required.');
  const documents = reference.input?.documents;
  requireValue(Array.isArray(documents) && documents.length > 0 && documents.every(text), 'documents are required.');
  requireValue(reference.scoringConfig && typeof reference.scoringConfig === 'object', 'scoringConfig is required.');
  for (const field of ['logitMaxAbs', 'scoreMaxAbs', 'probabilityMaxAbs']) {
    requireValue(Number.isFinite(reference.tolerances?.[field]) && reference.tolerances[field] >= 0, `tolerances.${field} is required.`);
  }
  requireValue(reference.tolerances.ranking === 'exact', 'ranking policy must be explicit and exact.');
  validateRows(reference.outputs, documents, 'reference.outputs');
  return reference;
}

export function assertRerankSourceIdentity(identity, reference) {
  for (const [field, expected] of Object.entries({ sourceCheckpointId: reference.source.checkpointId,
    sourceRepo: reference.source.repository, sourceRevision: reference.source.revision })) {
    requireValue(identity?.[field] === expected, `artifactIdentity.${field} must bind the pinned source reference.`);
  }
}

export function evaluateRerankReference(reference, observation) {
  assertRerankReference(reference);
  requireValue(computeCanonicalSha256(observation?.input) === computeCanonicalSha256(reference.input), 'observed inputs differ.');
  requireValue(computeCanonicalSha256(observation.scoringConfig) === computeCanonicalSha256(reference.scoringConfig), 'observed scoring contract differs.');
  validateRows(observation.outputs, reference.input.documents, 'observation.outputs');
  const checks = [];
  for (const [index, expected] of reference.outputs.entries()) {
    const actual = observation.outputs[index];
    checks.push({ id: `document.${index}.tokens`, passed: computeCanonicalSha256(actual.tokenIds) === computeCanonicalSha256(expected.tokenIds) });
    for (const [field, tolerance] of Object.entries({
      trueLogit: reference.tolerances.logitMaxAbs, falseLogit: reference.tolerances.logitMaxAbs,
      score: reference.tolerances.scoreMaxAbs, probability: reference.tolerances.probabilityMaxAbs,
    })) {
      const absoluteError = Math.abs(actual[field] - expected[field]);
      checks.push({ id: `document.${index}.${field}`, passed: absoluteError <= tolerance,
        expected: expected[field], actual: actual[field], absoluteError, tolerance });
    }
  }
  const order = (rows) => [...rows].sort((a, b) => b.score - a.score || a.index - b.index).map((row) => row.index);
  const expected = order(reference.outputs);
  const actual = order(observation.outputs);
  checks.push({ id: 'ranking.exact', passed: computeCanonicalSha256(actual) === computeCanonicalSha256(expected), expected, actual });
  return { passed: checks.every((check) => check.passed), checks };
}

export function assertRerankReferenceTranscript(transcript) {
  requireValue(transcript?.schema === RERANK_REFERENCE_TRANSCRIPT_SCHEMA_ID && transcript.operation === 'rerank', 'unsupported transcript schema or operation.');
  for (const field of ['modelId', 'surface']) requireValue(text(transcript[field]), `${field} is required.`);
  for (const digest of [transcript.manifestHash, transcript.executionGraphHash, transcript.source?.hash, transcript.referenceDigest]) {
    requireValue(DIGEST.test(digest ?? ''), 'transcript identity requires SHA-256 digests.');
  }
  requireValue(text(transcript.source?.path), 'source path is required.');
  requireValue(transcript.referenceDigest === computeCanonicalSha256(transcript.reference), 'source reference digest differs.');
  requireValue(transcript.tokens === undefined && transcript.generationConfig === undefined, 'generation evidence cannot qualify reranking.');
  const result = evaluateRerankReference(transcript.reference, transcript.observation);
  requireValue(result.passed, `source comparison failed: ${result.checks.filter((check) => !check.passed).map((check) => check.id).join(', ')}.`);
  return transcript;
}
