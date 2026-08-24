import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

export const SOURCE_BOUNDARY_PACK_SCHEMA = 'doppler.source-boundary-pack/v1';
export const RUNTIME_BOUNDARY_CAPTURE_SCHEMA = 'doppler.runtime-boundary-capture/v1';
export const BOUNDARY_COMPARISON_RECEIPT_SCHEMA =
  'doppler.boundary-comparison-receipt/v1';
export const DETERMINISTIC_TOKEN_EVIDENCE_SCHEMA =
  'doppler.deterministic-token-evidence/v1';
export const BOUNDARY_PROVIDER_CAPTURE_SCHEMA =
  'doppler.boundary-provider-capture/v1';

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function assertEvidencePolicy(policy) {
  if (policy?.schema !== 'doppler.boundary-evidence-policy/v1') {
    throw new Error('boundary evidence: doppler.boundary-evidence-policy/v1 is required');
  }
}

function assertArtifactDigest(artifact, label) {
  const { digest, ...core } = artifact;
  if (!/^sha256:[0-9a-f]{64}$/.test(digest ?? '')) {
    throw new Error(`boundary evidence: ${label}.digest is required`);
  }
  if (computeCanonicalSha256(core) !== digest) {
    throw new Error(`boundary evidence: ${label}.digest does not match its payload`);
  }
}

function validateSourceIdentity(identity) {
  if (!isObject(identity)) throw new Error('boundary evidence: source-pack identity is required');
  for (const field of ['sourceRevision', 'dtype']) {
    if (typeof identity[field] !== 'string' || !identity[field]) {
      throw new Error(`boundary evidence: source-pack identity.${field} is required`);
    }
  }
  for (const field of ['promptDigest', 'modelConfigDigest', 'referenceScriptDigest']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(identity[field] ?? '')) {
      throw new Error(`boundary evidence: source-pack identity.${field} is required`);
    }
  }
}

export function buildDeterministicTokenEvidenceFromReferenceTranscript(transcript) {
  if (transcript?.schema !== 'doppler.reference-transcript/v1') {
    throw new Error(
      'boundary evidence: doppler.reference-transcript/v1 is required for token evidence'
    );
  }
  const tokenIds = transcript.tokens?.ids;
  const tokenCount = transcript.output?.tokensGenerated;
  const completeTokenIds = Array.isArray(tokenIds)
    && Number.isInteger(tokenCount)
    && tokenIds.length === tokenCount
    && tokenIds.every(Number.isInteger)
    && transcript.tokens?.coverage?.mode === 'full-token-ids'
    && transcript.tokens?.coverage?.omitted === 0;
  for (const field of ['generatedTokenIdsHash', 'generatedTextHash']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(transcript.tokens?.[field] ?? '')) {
      throw new Error(`boundary evidence: reference transcript tokens.${field} is required`);
    }
  }
  const core = {
    schema: DETERMINISTIC_TOKEN_EVIDENCE_SCHEMA,
    source: {
      kind: 'reference-transcript',
      transcriptDigest: computeCanonicalSha256(transcript),
    },
    exact: completeTokenIds,
    tokenCount: Number.isInteger(tokenCount) ? tokenCount : 0,
    generatedTokenIdsHash: transcript.tokens.generatedTokenIdsHash,
    generatedTextHash: transcript.tokens.generatedTextHash,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}

function resolveTimeline(report) {
  const candidates = [
    report?.result?.metrics?.operatorDiagnostics?.timeline,
    report?.metrics?.operatorDiagnostics?.timeline,
    report?.operatorDiagnostics?.timeline,
    report?.timeline,
  ];
  return candidates.find(Array.isArray) ?? null;
}

function resolveTemplate(policy, stageName) {
  for (const boundary of policy.boundaries) {
    const priority = boundary.stages.indexOf(stageName);
    if (priority >= 0) return { boundary, priority };
  }
  return null;
}

function normalizeSamples(capture) {
  const values = Array.isArray(capture.sample) ? capture.sample : [];
  const coordinates = Array.isArray(capture.sampleCoordinates)
    ? capture.sampleCoordinates
    : values.map((_, index) => [index]);
  return values.map((value, index) => ({
    coordinate: coordinates[index] ?? [index],
    flatIndex: index,
    value,
  }));
}

function boundaryFromRecord(record, idTemplate, tolerancePolicyId) {
  const layer = Number.isInteger(record.layerIndex) ? record.layerIndex : null;
  const boundaryId = idTemplate.replace('{L}', String(layer));
  const capture = record.capture;
  if (!isObject(capture) || capture.error) {
    throw new Error(`boundary evidence: ${boundaryId} has no successful tensor capture`);
  }
  if (!capture.fullTensorDigest) {
    throw new Error(
      `boundary evidence: ${boundaryId} capture has no fullTensorDigest; rerun with current probes`
    );
  }
  return {
    boundaryId,
    phase: record.phase ?? null,
    tokenIndex: record.tokenIndex ?? null,
    layerIndex: layer,
    shape: Array.isArray(capture.shape) ? capture.shape : record.shapeSignature,
    dtype: capture.dtype ?? record.dtype ?? null,
    samples: normalizeSamples(capture),
    fullTensorDigest: capture.fullTensorDigest,
    statistics: capture.stats,
    tolerancePolicyId,
    execution: {
      opId: record.opId ?? null,
      stageName: record.stageName,
      kernelDigest: record.kernelDigest ?? null,
      pipelineHash: record.pipelineHash ?? null,
    },
  };
}

function addOccurrences(boundaries) {
  const counts = new Map();
  return boundaries.map((boundary) => {
    const key = [
      boundary.phase ?? '',
      boundary.tokenIndex ?? '',
      boundary.boundaryId,
    ].join(':');
    const occurrence = counts.get(key) ?? 0;
    counts.set(key, occurrence + 1);
    return { ...boundary, occurrence };
  });
}

export function buildRuntimeBoundaryCapture({
  report,
  policy,
  tolerancePolicyId = 'doppler.boundary-tolerance/source-f16-v1',
  identity = {},
}) {
  assertEvidencePolicy(policy);
  if (!policy.tolerancePolicies?.[tolerancePolicyId]) {
    throw new Error(`boundary evidence: unknown tolerance policy "${tolerancePolicyId}"`);
  }
  const timeline = resolveTimeline(report);
  if (!timeline) {
    throw new Error('boundary evidence: report has no operator diagnostics timeline');
  }
  const selected = new Map();
  for (let recordIndex = 0; recordIndex < timeline.length; recordIndex += 1) {
    const record = timeline[recordIndex];
    const resolved = resolveTemplate(policy, record.stageName);
    if (!resolved) continue;
    const layerKey = Number.isInteger(record.layerIndex) ? record.layerIndex : 'model';
    const key = [
      record.phase ?? '',
      record.tokenIndex ?? '',
      layerKey,
      resolved.boundary.idTemplate,
    ].join(':');
    const previous = selected.get(key);
    if (!previous || resolved.priority < previous.priority) {
      selected.set(key, { ...resolved, record, recordIndex });
    }
  }
  const boundaries = addOccurrences(
    Array.from(selected.values())
      .sort((left, right) => left.recordIndex - right.recordIndex)
      .map(({ record, boundary }) => (
        boundaryFromRecord(record, boundary.idTemplate, tolerancePolicyId)
      ))
  );
  const core = {
    schema: RUNTIME_BOUNDARY_CAPTURE_SCHEMA,
    identity: {
      modelDigest: identity.modelDigest ?? timeline[0]?.modelHash ?? null,
      runtimeConfigDigest: identity.runtimeConfigDigest ?? timeline[0]?.runtimeConfigHash ?? null,
      executionGraphDigest: identity.executionGraphDigest ?? timeline[0]?.executionPlanHash ?? null,
      sourceReportDigest: identity.sourceReportDigest ?? computeCanonicalSha256(report),
      promptDigest: identity.promptDigest ?? null,
    },
    boundaries,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}

export function buildSourceBoundaryPack({
  identity,
  boundaries,
}) {
  validateSourceIdentity(identity);
  if (!Array.isArray(boundaries) || boundaries.length === 0) {
    throw new Error('boundary evidence: source pack requires boundaries');
  }
  for (const boundary of boundaries) {
    validateBoundary(boundary, 'source pack');
  }
  const core = {
    schema: SOURCE_BOUNDARY_PACK_SCHEMA,
    identity,
    boundaries,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}

export function buildSourceBoundaryPackFromProviderCapture(capture) {
  if (capture?.schema !== BOUNDARY_PROVIDER_CAPTURE_SCHEMA) {
    throw new Error(
      `boundary evidence: expected ${BOUNDARY_PROVIDER_CAPTURE_SCHEMA}`
    );
  }
  if (typeof capture.provider !== 'string' || !capture.provider) {
    throw new Error('boundary evidence: provider capture requires provider');
  }
  return buildSourceBoundaryPack({
    identity: {
      ...capture.identity,
      provider: capture.provider,
      providerRuntime: capture.runtime ?? null,
    },
    boundaries: capture.boundaries,
  });
}

function validateBoundary(boundary, label) {
  if (!isObject(boundary) || typeof boundary.boundaryId !== 'string') {
    throw new Error(`boundary evidence: ${label} contains an invalid boundary`);
  }
  if (!Array.isArray(boundary.shape) || !boundary.shape.every(Number.isInteger)) {
    throw new Error(`boundary evidence: ${boundary.boundaryId} shape is required`);
  }
  if (typeof boundary.dtype !== 'string' || !boundary.dtype) {
    throw new Error(`boundary evidence: ${boundary.boundaryId} dtype is required`);
  }
  if (!Array.isArray(boundary.samples) || boundary.samples.length === 0) {
    throw new Error(`boundary evidence: ${boundary.boundaryId} samples are required`);
  }
  if (!/^sha256:[0-9a-f]{64}$/.test(boundary.fullTensorDigest ?? '')) {
    throw new Error(`boundary evidence: ${boundary.boundaryId} fullTensorDigest is required`);
  }
  if (!isObject(boundary.statistics)) {
    throw new Error(`boundary evidence: ${boundary.boundaryId} statistics are required`);
  }
  if (typeof boundary.tolerancePolicyId !== 'string') {
    throw new Error(`boundary evidence: ${boundary.boundaryId} tolerancePolicyId is required`);
  }
}

function boundaryKey(boundary) {
  return [
    boundary.phase ?? '',
    boundary.tokenIndex ?? '',
    boundary.boundaryId,
    boundary.occurrence ?? 0,
  ].join(':');
}

function compareSamples(expected, actual, tolerance) {
  const actualByCoordinate = new Map(
    actual.samples.map((sample) => [JSON.stringify(sample.coordinate), sample])
  );
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  const failures = [];
  for (const expectedSample of expected.samples) {
    const coordinateKey = JSON.stringify(expectedSample.coordinate);
    const actualSample = actualByCoordinate.get(coordinateKey);
    if (!actualSample) {
      failures.push({ coordinate: expectedSample.coordinate, reason: 'missing_sample' });
      continue;
    }
    const absoluteError = Math.abs(actualSample.value - expectedSample.value);
    const scale = Math.max(Math.abs(expectedSample.value), Number.MIN_VALUE);
    const relativeError = absoluteError / scale;
    maxAbsoluteError = Math.max(maxAbsoluteError, absoluteError);
    maxRelativeError = Math.max(maxRelativeError, relativeError);
    if (absoluteError > tolerance.absolute && relativeError > tolerance.relative) {
      failures.push({
        coordinate: expectedSample.coordinate,
        expected: expectedSample.value,
        actual: actualSample.value,
        absoluteError,
        relativeError,
      });
    }
  }
  return { maxAbsoluteError, maxRelativeError, failures };
}

function compareBoundary(expected, actual, tolerance) {
  const reasons = [];
  if (JSON.stringify(expected.shape) !== JSON.stringify(actual.shape)) reasons.push('shape_mismatch');
  if (tolerance.requireDtypeMatch && expected.dtype !== actual.dtype) reasons.push('dtype_mismatch');
  const samples = compareSamples(expected, actual, tolerance);
  if (samples.failures.length > 0) reasons.push('sample_tolerance_exceeded');
  if (
    tolerance.absolute === 0
    && tolerance.relative === 0
    && expected.fullTensorDigest !== actual.fullTensorDigest
  ) {
    reasons.push('full_tensor_digest_mismatch');
  }
  return {
    boundaryId: expected.boundaryId,
    key: boundaryKey(expected),
    passed: reasons.length === 0,
    reasons,
    expectedDigest: expected.fullTensorDigest,
    actualDigest: actual.fullTensorDigest,
    sampleComparison: samples,
  };
}

export function compareBoundaryEvidence({
  sourcePack,
  runtimeCapture,
  policy,
  artifactPrecision = 'source',
  sourcePrecisionControlReceipt = null,
  deterministicTokenEvidence = null,
}) {
  assertEvidencePolicy(policy);
  if (artifactPrecision !== 'source' && artifactPrecision !== 'quantized') {
    throw new Error('boundary evidence: artifactPrecision must be source or quantized');
  }
  if (sourcePack?.schema !== SOURCE_BOUNDARY_PACK_SCHEMA) {
    throw new Error(`boundary evidence: expected ${SOURCE_BOUNDARY_PACK_SCHEMA}`);
  }
  if (runtimeCapture?.schema !== RUNTIME_BOUNDARY_CAPTURE_SCHEMA) {
    throw new Error(`boundary evidence: expected ${RUNTIME_BOUNDARY_CAPTURE_SCHEMA}`);
  }
  validateSourceIdentity(sourcePack.identity);
  assertArtifactDigest(sourcePack, 'sourcePack');
  assertArtifactDigest(runtimeCapture, 'runtimeCapture');
  const actualByKey = new Map(
    runtimeCapture.boundaries.map((boundary) => [boundaryKey(boundary), boundary])
  );
  const comparisons = [];
  for (const expected of sourcePack.boundaries) {
    validateBoundary(expected, 'source pack');
    const actual = actualByKey.get(boundaryKey(expected));
    if (!actual) {
      comparisons.push({
        boundaryId: expected.boundaryId,
        key: boundaryKey(expected),
        passed: false,
        reasons: ['missing_runtime_boundary'],
      });
      break;
    }
    validateBoundary(actual, 'runtime capture');
    const tolerance = policy.tolerancePolicies[expected.tolerancePolicyId];
    if (!tolerance) {
      throw new Error(
        `boundary evidence: unknown tolerance policy "${expected.tolerancePolicyId}"`
      );
    }
    const comparison = compareBoundary(expected, actual, tolerance);
    comparisons.push(comparison);
    if (!comparison.passed) break;
  }
  const boundaryCompatible = comparisons.length === sourcePack.boundaries.length
    && comparisons.every((comparison) => comparison.passed);
  const sourceControlPassed = artifactPrecision !== 'quantized'
    || (
      sourcePrecisionControlReceipt?.schema === BOUNDARY_COMPARISON_RECEIPT_SCHEMA
      && sourcePrecisionControlReceipt?.promotionGate?.passed === true
    );
  const tokenParityPassed = deterministicTokenEvidence?.exact === true
    && deterministicTokenEvidence?.tokenCount >= 128
    && deterministicTokenEvidence?.schema === DETERMINISTIC_TOKEN_EVIDENCE_SCHEMA;
  const promotionGate = {
    boundaryCompatible,
    sourcePrecisionControlPassed: sourceControlPassed,
    deterministicTokenParityPassed: tokenParityPassed,
    passed: boundaryCompatible && sourceControlPassed && tokenParityPassed,
  };
  const core = {
    schema: BOUNDARY_COMPARISON_RECEIPT_SCHEMA,
    sourcePackDigest: sourcePack.digest ?? computeCanonicalSha256(sourcePack),
    runtimeCaptureDigest: runtimeCapture.digest ?? computeCanonicalSha256(runtimeCapture),
    artifactPrecision,
    comparisons,
    firstDivergence: comparisons.find((comparison) => !comparison.passed) ?? null,
    promotionGate,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}
