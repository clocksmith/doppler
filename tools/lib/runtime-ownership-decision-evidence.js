import { computeCanonicalJsonSha256 } from './canonical-json.js';

const DIMENSION_SCHEMA = 'doppler.runtime-ownership-dimension-evidence/v1';
const HYPOTHESIS_SCHEMA = 'doppler.runtime-ownership-hypothesis-evidence/v1';
const PROMOTION_SCHEMA = 'doppler.runtime-ownership-promotion-evidence/v1';
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const REVISION_PATTERN = /^[0-9a-f]{40}$/;

export const RUNTIME_OWNERSHIP_DIMENSION_CLASSES = Object.freeze([
  'correctness',
  'taskQuality',
  'usability',
  'memory',
  'endToEndPerformance',
  'diagnosticDepth',
  'distributionCost',
  'integrationBurden',
  'providerRisk',
]);

function normalizeText(value) {
  return typeof value === 'string' ? value.trim() : '';
}

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, fields, label, errors) {
  if (!isPlainObject(value)) {
    errors.push(`${label} must be an object`);
    return false;
  }
  const expected = new Set(fields);
  for (const field of Object.keys(value)) {
    if (!expected.has(field)) errors.push(`${label}.${field} is not supported`);
  }
  for (const field of fields) {
    if (!Object.hasOwn(value, field)) errors.push(`${label}.${field} is required`);
  }
  return true;
}

function text(value, label, errors) {
  const normalized = normalizeText(value);
  if (!normalized) errors.push(`${label} must be a non-empty string`);
  return normalized || null;
}

function sha256(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !SHA256_PATTERN.test(normalized)) {
    errors.push(`${label} must be a SHA-256 identity`);
  }
  return normalized;
}

function revision(value, label, errors) {
  const normalized = text(value, label, errors)?.toLowerCase() ?? null;
  if (normalized && !REVISION_PATTERN.test(normalized)) {
    errors.push(`${label} must be a lowercase 40-hex revision`);
  }
  return normalized;
}

function finiteNumber(value, label, errors, minimum = null) {
  if (!Number.isFinite(value)) {
    errors.push(`${label} must be finite`);
    return null;
  }
  if (minimum !== null && value < minimum) errors.push(`${label} must be at least ${minimum}`);
  return value;
}

function integer(value, label, errors, minimum = 0) {
  if (!Number.isInteger(value) || value < minimum) {
    errors.push(`${label} must be an integer of at least ${minimum}`);
    return null;
  }
  return value;
}

function boolean(value, label, errors) {
  if (typeof value !== 'boolean') errors.push(`${label} must be boolean`);
  return value === true;
}

function instant(value, label, errors) {
  const normalized = text(value, label, errors);
  const parsed = normalized ? new Date(normalized) : null;
  if (!parsed || !Number.isFinite(parsed.getTime()) || parsed.toISOString() !== normalized) {
    errors.push(`${label} must be an ISO instant`);
    return null;
  }
  return parsed;
}

function match(actual, expected, label, errors) {
  if (expected != null && actual != null && actual !== expected) {
    errors.push(`${label} does not match the decision`);
  }
}

function validateBinding(receipt, expected, label, errors) {
  const workload = text(receipt.workload, `${label}.workload`, errors);
  const logicalModelId = text(receipt.logicalModelId, `${label}.logicalModelId`, errors);
  const sourceExecutionId = sha256(
    receipt.sourceExecutionId,
    `${label}.sourceExecutionId`,
    errors
  );
  const incumbentExecutionId = sha256(
    receipt.incumbentExecutionId,
    `${label}.incumbentExecutionId`,
    errors
  );
  const resolvedArtifactVariantId = sha256(
    receipt.resolvedArtifactVariantId,
    `${label}.resolvedArtifactVariantId`,
    errors
  );
  const resolvedExecutionId = sha256(
    receipt.resolvedExecutionId,
    `${label}.resolvedExecutionId`,
    errors
  );
  const harnessRevision = revision(
    receipt.harnessRevision,
    `${label}.harnessRevision`,
    errors
  );
  const environmentFingerprint = sha256(
    receipt.environmentFingerprint,
    `${label}.environmentFingerprint`,
    errors
  );
  match(workload, expected.workload, `${label}.workload`, errors);
  match(logicalModelId, expected.logicalModelId, `${label}.logicalModelId`, errors);
  match(sourceExecutionId, expected.sourceExecutionId, `${label}.sourceExecutionId`, errors);
  match(
    incumbentExecutionId,
    expected.incumbentExecutionId,
    `${label}.incumbentExecutionId`,
    errors
  );
  match(
    resolvedArtifactVariantId,
    expected.resolvedArtifactVariantId,
    `${label}.resolvedArtifactVariantId`,
    errors
  );
  match(
    resolvedExecutionId,
    expected.resolvedExecutionId,
    `${label}.resolvedExecutionId`,
    errors
  );
  match(harnessRevision, expected.harnessRevision, `${label}.harnessRevision`, errors);
  match(
    environmentFingerprint,
    expected.environmentFingerprint,
    `${label}.environmentFingerprint`,
    errors
  );
  return { harnessRevision, environmentFingerprint };
}

function correctnessPassed(value, label, errors) {
  const fields = [
    'referenceValid',
    'workloadEquivalent',
    'incumbentAcceptable',
    'dopplerAcceptable',
  ];
  exactKeys(value, fields, label, errors);
  boolean(value?.incumbentAcceptable, `${label}.incumbentAcceptable`, errors);
  return boolean(value?.referenceValid, `${label}.referenceValid`, errors)
    && boolean(value?.workloadEquivalent, `${label}.workloadEquivalent`, errors)
    && boolean(value?.dopplerAcceptable, `${label}.dopplerAcceptable`, errors);
}

function taskQualityPassed(value, label, errors) {
  const fields = [
    'heldOutSetDigest',
    'sourceScore',
    'incumbentScore',
    'dopplerScore',
    'minimumAcceptedScore',
    'higherIsBetter',
  ];
  exactKeys(value, fields, label, errors);
  sha256(value?.heldOutSetDigest, `${label}.heldOutSetDigest`, errors);
  finiteNumber(value?.sourceScore, `${label}.sourceScore`, errors);
  finiteNumber(value?.incumbentScore, `${label}.incumbentScore`, errors);
  const dopplerScore = finiteNumber(value?.dopplerScore, `${label}.dopplerScore`, errors);
  const threshold = finiteNumber(
    value?.minimumAcceptedScore,
    `${label}.minimumAcceptedScore`,
    errors
  );
  const higherIsBetter = boolean(value?.higherIsBetter, `${label}.higherIsBetter`, errors);
  if (dopplerScore === null || threshold === null) return false;
  return higherIsBetter ? dopplerScore >= threshold : dopplerScore <= threshold;
}

function usabilityPassed(value, label, errors) {
  const fields = ['installSucceeded', 'loadSucceeded', 'invokeSucceeded', 'fallbackExplicit'];
  exactKeys(value, fields, label, errors);
  return fields.map((field) => boolean(value?.[field], `${label}.${field}`, errors)).every(Boolean);
}

function memoryPassed(value, label, errors) {
  const fields = [
    'sourcePeakBytes',
    'incumbentPeakBytes',
    'dopplerPeakBytes',
    'maximumDopplerBytes',
    'measurementScopeMatched',
  ];
  exactKeys(value, fields, label, errors);
  finiteNumber(value?.sourcePeakBytes, `${label}.sourcePeakBytes`, errors, 0);
  finiteNumber(value?.incumbentPeakBytes, `${label}.incumbentPeakBytes`, errors, 0);
  const observed = finiteNumber(value?.dopplerPeakBytes, `${label}.dopplerPeakBytes`, errors, 0);
  const maximum = finiteNumber(
    value?.maximumDopplerBytes,
    `${label}.maximumDopplerBytes`,
    errors,
    0
  );
  return boolean(value?.measurementScopeMatched, `${label}.measurementScopeMatched`, errors)
    && observed !== null
    && maximum !== null
    && observed <= maximum;
}

function endToEndPerformancePassed(value, label, errors) {
  const fields = [
    'sourceValue',
    'incumbentValue',
    'dopplerValue',
    'unit',
    'sampleCount',
    'minimumSampleCount',
    'timingScopeMatched',
    'workloadEquivalent',
  ];
  exactKeys(value, fields, label, errors);
  finiteNumber(value?.sourceValue, `${label}.sourceValue`, errors, 0);
  finiteNumber(value?.incumbentValue, `${label}.incumbentValue`, errors, 0);
  finiteNumber(value?.dopplerValue, `${label}.dopplerValue`, errors, 0);
  text(value?.unit, `${label}.unit`, errors);
  const samples = integer(value?.sampleCount, `${label}.sampleCount`, errors, 1);
  const minimum = integer(value?.minimumSampleCount, `${label}.minimumSampleCount`, errors, 1);
  return boolean(value?.timingScopeMatched, `${label}.timingScopeMatched`, errors)
    && boolean(value?.workloadEquivalent, `${label}.workloadEquivalent`, errors)
    && samples !== null
    && minimum !== null
    && samples >= minimum;
}

function diagnosticDepthPassed(value, label, errors) {
  const fields = [
    'firstDivergenceLocalized',
    'semanticBoundaryRecorded',
    'correctionPathActionable',
    'fallbackVisible',
  ];
  exactKeys(value, fields, label, errors);
  return fields.map((field) => boolean(value?.[field], `${label}.${field}`, errors)).every(Boolean);
}

function distributionCostPassed(value, label, errors) {
  const fields = [
    'sourceArtifactBytes',
    'incumbentArtifactBytes',
    'dopplerArtifactBytes',
    'maximumDopplerBytes',
    'dependencyBytesCounted',
  ];
  exactKeys(value, fields, label, errors);
  finiteNumber(value?.sourceArtifactBytes, `${label}.sourceArtifactBytes`, errors, 0);
  finiteNumber(value?.incumbentArtifactBytes, `${label}.incumbentArtifactBytes`, errors, 0);
  const observed = finiteNumber(value?.dopplerArtifactBytes, `${label}.dopplerArtifactBytes`, errors, 0);
  const maximum = finiteNumber(
    value?.maximumDopplerBytes,
    `${label}.maximumDopplerBytes`,
    errors,
    0
  );
  return boolean(value?.dependencyBytesCounted, `${label}.dependencyBytesCounted`, errors)
    && observed !== null
    && maximum !== null
    && observed <= maximum;
}

function integrationBurdenPassed(value, label, errors) {
  const fields = [
    'sourceSteps',
    'incumbentSteps',
    'dopplerSteps',
    'maximumDopplerSteps',
    'cleanInstallPassed',
    'apiInvocationPassed',
  ];
  exactKeys(value, fields, label, errors);
  integer(value?.sourceSteps, `${label}.sourceSteps`, errors);
  integer(value?.incumbentSteps, `${label}.incumbentSteps`, errors);
  const observed = integer(value?.dopplerSteps, `${label}.dopplerSteps`, errors);
  const maximum = integer(value?.maximumDopplerSteps, `${label}.maximumDopplerSteps`, errors);
  return boolean(value?.cleanInstallPassed, `${label}.cleanInstallPassed`, errors)
    && boolean(value?.apiInvocationPassed, `${label}.apiInvocationPassed`, errors)
    && observed !== null
    && maximum !== null
    && observed <= maximum;
}

function providerRiskPassed(value, label, errors) {
  const fields = [
    'standardWebGpuPassed',
    'selectedNodeProviderPassed',
    'fallbackVisible',
    'doeRequired',
    'undeclaredProviderRequired',
  ];
  exactKeys(value, fields, label, errors);
  const standard = boolean(value?.standardWebGpuPassed, `${label}.standardWebGpuPassed`, errors);
  const node = boolean(
    value?.selectedNodeProviderPassed,
    `${label}.selectedNodeProviderPassed`,
    errors
  );
  const fallback = boolean(value?.fallbackVisible, `${label}.fallbackVisible`, errors);
  if (typeof value?.doeRequired !== 'boolean') errors.push(`${label}.doeRequired must be boolean`);
  if (typeof value?.undeclaredProviderRequired !== 'boolean') {
    errors.push(`${label}.undeclaredProviderRequired must be boolean`);
  }
  return standard && node && fallback
    && value?.doeRequired === false
    && value?.undeclaredProviderRequired === false;
}

const DIMENSION_VALIDATORS = Object.freeze({
  correctness: correctnessPassed,
  taskQuality: taskQualityPassed,
  usability: usabilityPassed,
  memory: memoryPassed,
  endToEndPerformance: endToEndPerformancePassed,
  diagnosticDepth: diagnosticDepthPassed,
  distributionCost: distributionCostPassed,
  integrationBurden: integrationBurdenPassed,
  providerRisk: providerRiskPassed,
});

export function computeRuntimeOwnershipDecisionEvidenceDigest(receipt) {
  return computeCanonicalJsonSha256(receipt);
}

export function validateRuntimeOwnershipDimensionEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'decisionId',
    'evidenceClass',
    'workload',
    'logicalModelId',
    'sourceExecutionId',
    'incumbentExecutionId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'harnessRevision',
    'environmentFingerprint',
    'capturedAtUtc',
    'result',
  ];
  if (!exactKeys(receipt, fields, 'dimension evidence', errors)) {
    return {
      errors,
      reasons: ['dimension-evidence-invalid'],
      passed: null,
      capturedAt: null,
      observations: null,
    };
  }
  if (receipt.schema !== DIMENSION_SCHEMA) {
    errors.push(`dimension evidence.schema must be ${DIMENSION_SCHEMA}`);
  }
  const decisionId = text(receipt.decisionId, 'dimension evidence.decisionId', errors);
  const evidenceClass = text(receipt.evidenceClass, 'dimension evidence.evidenceClass', errors);
  match(decisionId, expected.decisionId, 'dimension evidence.decisionId', errors);
  match(evidenceClass, expected.evidenceClass, 'dimension evidence.evidenceClass', errors);
  if (evidenceClass && !RUNTIME_OWNERSHIP_DIMENSION_CLASSES.includes(evidenceClass)) {
    errors.push('dimension evidence.evidenceClass is not recognized');
  }
  const binding = validateBinding(receipt, expected, 'dimension evidence', errors);
  const capturedAt = instant(receipt.capturedAtUtc, 'dimension evidence.capturedAtUtc', errors);
  let passed = null;
  if (exactKeys(receipt.result, ['passed', 'observations'], 'dimension evidence.result', errors)) {
    if (typeof receipt.result.passed !== 'boolean') {
      errors.push('dimension evidence.result.passed must be boolean');
    }
    const validator = DIMENSION_VALIDATORS[evidenceClass];
    const derivedPassed = validator
      ? validator(receipt.result.observations, 'dimension evidence.result.observations', errors)
      : false;
    if (typeof receipt.result.passed === 'boolean' && receipt.result.passed !== derivedPassed) {
      errors.push('dimension evidence.result.passed does not match its observations');
    }
    passed = errors.length === 0 ? derivedPassed : null;
  }
  if (passed !== true) reasons.push(`${evidenceClass || 'dimension'}-evidence-gate-failed`);
  return {
    errors,
    reasons,
    passed,
    capturedAt,
    observations: isPlainObject(receipt.result?.observations)
      ? receipt.result.observations
      : null,
    harnessRevision: binding.harnessRevision,
    environmentFingerprint: binding.environmentFingerprint,
  };
}

function thresholdMatches(operator, thresholdValue, observedValue) {
  if (operator === 'greater-than-or-equal') return observedValue >= thresholdValue;
  if (operator === 'less-than-or-equal') return observedValue <= thresholdValue;
  return null;
}

export function validateRuntimeOwnershipHypothesisEvidence(receipt, expected = {}) {
  const errors = [];
  const reasons = [];
  const fields = [
    'schema',
    'decisionId',
    'axis',
    'metric',
    'controlMetric',
    'workload',
    'logicalModelId',
    'sourceExecutionId',
    'incumbentExecutionId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'harnessRevision',
    'environmentFingerprint',
    'evaluatedAtUtc',
    'observedValue',
    'qualitativePassed',
    'controlPassed',
    'endToEndAcceptancePassed',
  ];
  if (!exactKeys(receipt, fields, 'hypothesis evidence', errors)) {
    return {
      errors,
      reasons: ['hypothesis-evidence-invalid'],
      passed: null,
      observedValue: null,
      evaluatedAt: null,
    };
  }
  if (receipt.schema !== HYPOTHESIS_SCHEMA) {
    errors.push(`hypothesis evidence.schema must be ${HYPOTHESIS_SCHEMA}`);
  }
  const decisionId = text(receipt.decisionId, 'hypothesis evidence.decisionId', errors);
  const axis = text(receipt.axis, 'hypothesis evidence.axis', errors);
  const metric = text(receipt.metric, 'hypothesis evidence.metric', errors);
  const controlMetric = text(receipt.controlMetric, 'hypothesis evidence.controlMetric', errors);
  match(decisionId, expected.decisionId, 'hypothesis evidence.decisionId', errors);
  match(axis, expected.axis, 'hypothesis evidence.axis', errors);
  match(metric, expected.metric, 'hypothesis evidence.metric', errors);
  match(controlMetric, expected.controlMetric, 'hypothesis evidence.controlMetric', errors);
  const binding = validateBinding(receipt, expected, 'hypothesis evidence', errors);
  const evaluatedAt = instant(receipt.evaluatedAtUtc, 'hypothesis evidence.evaluatedAtUtc', errors);
  const numeric = expected.operator !== 'pass';
  let observedValue = receipt.observedValue;
  if (numeric) {
    observedValue = finiteNumber(
      observedValue,
      'hypothesis evidence.observedValue',
      errors
    );
    if (receipt.qualitativePassed !== null) {
      errors.push('numeric hypothesis evidence.qualitativePassed must be null');
    }
  } else {
    if (!Number.isFinite(observedValue) && !normalizeText(observedValue)) {
      errors.push('qualitative hypothesis evidence.observedValue must be finite or non-empty text');
    }
    if (typeof receipt.qualitativePassed !== 'boolean') {
      errors.push('qualitative hypothesis evidence.qualitativePassed must be boolean');
    }
  }
  const controlPassed = boolean(
    receipt.controlPassed,
    'hypothesis evidence.controlPassed',
    errors
  );
  const endToEndAcceptancePassed = boolean(
    receipt.endToEndAcceptancePassed,
    'hypothesis evidence.endToEndAcceptancePassed',
    errors
  );
  const thresholdPassed = numeric
    ? observedValue !== null
      && thresholdMatches(expected.operator, expected.thresholdValue, observedValue)
    : receipt.qualitativePassed === true;
  const passed = errors.length === 0
    ? thresholdPassed && controlPassed && endToEndAcceptancePassed
    : null;
  if (!thresholdPassed) reasons.push('hypothesis-threshold-not-passed');
  if (!controlPassed) reasons.push('hypothesis-control-not-passed');
  if (!endToEndAcceptancePassed) reasons.push('hypothesis-end-to-end-acceptance-not-passed');
  return {
    errors,
    reasons,
    passed,
    observedValue,
    evaluatedAt,
    harnessRevision: binding.harnessRevision,
    environmentFingerprint: binding.environmentFingerprint,
  };
}

export function computeRuntimeOwnershipEvidenceSetDigest(evidenceReferences) {
  return computeCanonicalJsonSha256(evidenceReferences);
}

export function computeRuntimeOwnershipHypothesisSetDigest(hypotheses) {
  return computeCanonicalJsonSha256(hypotheses);
}

export function validateRuntimeOwnershipPromotionEvidence(receipt, expected = {}) {
  const errors = [];
  const fields = [
    'schema',
    'decisionId',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceProviderId',
    'sourceArtifactId',
    'sourceExecutionId',
    'incumbentProviderId',
    'incumbentArtifactId',
    'incumbentExecutionId',
    'correctnessClass',
    'harnessRevision',
    'environmentFingerprint',
    'disposition',
    'decisionRationale',
    'evidenceSetDigest',
    'hypothesisSetDigest',
    'decision',
    'authority',
    'reviewer',
    'reviewerRevision',
    'rationale',
    'promotedAtUtc',
    'qualifiedAtUtc',
    'expiresAtUtc',
  ];
  if (!exactKeys(receipt, fields, 'runtime ownership promotion evidence', errors)) {
    return { errors, promotedAt: null };
  }
  if (receipt.schema !== PROMOTION_SCHEMA) {
    errors.push(`runtime ownership promotion evidence.schema must be ${PROMOTION_SCHEMA}`);
  }
  const textFields = [
    'decisionId',
    'workload',
    'logicalModelId',
    'manifestVariantId',
    'sourceProviderId',
    'sourceArtifactId',
    'incumbentProviderId',
    'incumbentArtifactId',
    'correctnessClass',
    'disposition',
    'decisionRationale',
  ];
  for (const field of textFields) {
    const actual = text(
      receipt[field],
      `runtime ownership promotion evidence.${field}`,
      errors
    );
    match(actual, expected[field], `runtime ownership promotion evidence.${field}`, errors);
  }
  for (const field of [
    'resolvedArtifactVariantId',
    'resolvedExecutionId',
    'sourceExecutionId',
    'incumbentExecutionId',
    'environmentFingerprint',
    'evidenceSetDigest',
    'hypothesisSetDigest',
  ]) {
    const actual = sha256(
      receipt[field],
      `runtime ownership promotion evidence.${field}`,
      errors
    );
    match(actual, expected[field], `runtime ownership promotion evidence.${field}`, errors);
  }
  const harnessRevision = revision(
    receipt.harnessRevision,
    'runtime ownership promotion evidence.harnessRevision',
    errors
  );
  match(
    harnessRevision,
    expected.harnessRevision,
    'runtime ownership promotion evidence.harnessRevision',
    errors
  );
  if (receipt.decision !== 'promote-disposition') {
    errors.push('runtime ownership promotion evidence.decision must be promote-disposition');
  }
  if (receipt.authority !== 'human') {
    errors.push('runtime ownership promotion evidence.authority must be human');
  }
  text(receipt.reviewer, 'runtime ownership promotion evidence.reviewer', errors);
  revision(
    receipt.reviewerRevision,
    'runtime ownership promotion evidence.reviewerRevision',
    errors
  );
  text(receipt.rationale, 'runtime ownership promotion evidence.rationale', errors);
  const promotedAt = instant(
    receipt.promotedAtUtc,
    'runtime ownership promotion evidence.promotedAtUtc',
    errors
  );
  const qualifiedAt = instant(
    receipt.qualifiedAtUtc,
    'runtime ownership promotion evidence.qualifiedAtUtc',
    errors
  );
  const expiresAt = instant(
    receipt.expiresAtUtc,
    'runtime ownership promotion evidence.expiresAtUtc',
    errors
  );
  match(
    receipt.qualifiedAtUtc,
    expected.qualifiedAtUtc,
    'runtime ownership promotion evidence.qualifiedAtUtc',
    errors
  );
  match(
    receipt.expiresAtUtc,
    expected.expiresAtUtc,
    'runtime ownership promotion evidence.expiresAtUtc',
    errors
  );
  if (promotedAt && qualifiedAt && promotedAt.getTime() < qualifiedAt.getTime()) {
    errors.push(
      'runtime ownership promotion evidence.promotedAtUtc must not predate qualifiedAtUtc'
    );
  }
  if (qualifiedAt && expiresAt && expiresAt.getTime() <= qualifiedAt.getTime()) {
    errors.push('runtime ownership promotion evidence.expiresAtUtc must follow qualifiedAtUtc');
  }
  if (promotedAt && expiresAt && promotedAt.getTime() >= expiresAt.getTime()) {
    errors.push('runtime ownership promotion evidence.promotedAtUtc must predate expiresAtUtc');
  }
  return { errors, promotedAt };
}
