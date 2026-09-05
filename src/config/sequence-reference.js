export const SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID = 'doppler.sequence-reference-transcript/v1';

const DIGEST = /^sha256:[0-9a-f]{64}$/u;

export function validateSequenceReferenceTranscript(transcript) {
  const errors = [];
  const value = transcript ?? {};
  if (value.schema !== SEQUENCE_REFERENCE_TRANSCRIPT_SCHEMA_ID) errors.push('Invalid sequence transcript schema.');
  if (value.operation !== 'encodeSequence') errors.push('Sequence transcript operation must be encodeSequence.');
  for (const [label, digest] of Object.entries({
    executionGraphHash: value.executionGraphHash,
    manifestHash: value.manifestHash,
    'source.hash': value.source?.hash,
    'reference.digest': value.reference?.digest,
    'output.pooledEmbedding': value.output?.digests?.pooledEmbedding,
    'output.tokenEmbeddings': value.output?.digests?.tokenEmbeddings,
  })) {
    if (!DIGEST.test(digest ?? '')) errors.push(`${label} must be a SHA-256 digest.`);
  }
  for (const [label, text] of Object.entries({
    surface: value.surface,
    modelId: value.modelId,
    'source.path': value.source?.path,
    'reference.source.checkpointId': value.reference?.source?.checkpointId,
    'reference.source.repository': value.reference?.source?.repository,
    'reference.source.revision': value.reference?.source?.revision,
    'reference.input.sequence': value.reference?.input?.sequence,
    'reference.input.alphabet': value.reference?.input?.alphabet,
  })) {
    if (typeof text !== 'string' || !text.trim()) errors.push(`${label} must be a non-empty string.`);
  }
  const tokens = value.reference?.input?.tokenIds;
  if (!Array.isArray(tokens) || tokens.length === 0
    || tokens.some((token) => !Number.isInteger(token) || token < 0)) {
    errors.push('Sequence transcript requires exact input token IDs.');
  }
  if (!Number.isInteger(value.output?.embeddingDim) || value.output.embeddingDim < 1
    || value.output?.tokenCount !== tokens?.length) {
    errors.push('Sequence transcript output geometry must match its tokenized input.');
  }
  if (value.options?.includeTokenEmbeddings !== true || value.options?.includeLogits !== false
    || value.output?.digests?.logits !== null) {
    errors.push('Sequence transcript currently qualifies token and pooled embeddings with logits disabled.');
  }
  const checks = value.checks;
  if (!Array.isArray(checks) || checks.some((check) => check?.passed !== true)
    || new Set(checks?.map((check) => check?.id)).size !== checks?.length) {
    errors.push('Sequence transcript requires unique passed checks.');
  }
  const byId = new Map(Array.isArray(checks) ? checks.map((check) => [check?.id, check]) : []);
  for (const id of ['model.identity', 'sequence.contract', 'tokenizer.ids',
    'pooledEmbedding.finite', 'tokenEmbeddings.finite', 'logits.not-requested',
    'pooledEmbedding.parity', 'tokenEmbeddings.parity']) {
    if (byId.get(id)?.passed !== true) errors.push(`Sequence transcript requires passed ${id}.`);
  }
  const tokenCheck = byId.get('tokenizer.ids');
  if (tokenCheck?.expectedCount !== tokens?.length || tokenCheck?.actualCount !== tokens?.length
    || tokenCheck?.mismatchCount !== 0) errors.push('Sequence transcript tokenizer check does not prove exact parity.');
  const geometry = {
    pooledEmbedding: value.output?.embeddingDim,
    tokenEmbeddings: value.output?.embeddingDim * value.output?.tokenCount,
  };
  for (const [name, count] of Object.entries(geometry)) {
    const finite = byId.get(`${name}.finite`);
    const parity = byId.get(`${name}.parity`);
    const toleranceName = name === 'pooledEmbedding' ? 'pooledEmbeddingMaxAbs' : 'tokenEmbeddingMaxAbs';
    const tolerance = value.reference?.tolerances?.[toleranceName];
    if (finite?.valueCount !== count || finite?.nonFiniteCount !== 0) errors.push(`${name} finite check has wrong geometry.`);
    if (!Number.isFinite(tolerance) || tolerance < 0 || parity?.tolerance !== tolerance
      || !Number.isInteger(parity?.sampleCount) || parity.sampleCount < 1
      || !Number.isFinite(parity?.maxAbsoluteError) || parity.maxAbsoluteError < 0
      || parity.maxAbsoluteError > tolerance || parity?.failures?.length !== 0) {
      errors.push(`${name} parity must satisfy its explicit reference tolerance.`);
    }
  }
  if (value.tokens !== undefined || value.generationConfig !== undefined) errors.push('Encoder evidence cannot contain generation evidence.');
  return { ok: errors.length === 0, errors };
}

export function assertSequenceReferenceTranscript(transcript) {
  const validation = validateSequenceReferenceTranscript(transcript);
  if (!validation.ok) throw new Error(`Invalid sequence reference transcript: ${validation.errors.join('; ')}`);
  return transcript;
}
