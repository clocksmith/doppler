import { computeCanonicalSha256 } from '../utils/canonical-hash.js';

export const REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA =
  'doppler.registered-variant-calibration-plan/v1';
export const REGISTERED_VARIANT_CALIBRATION_RECEIPT_SCHEMA =
  'doppler.registered-variant-calibration-receipt/v1';

const PHASES = new Set(['prefill', 'decode']);
const TAIL_CLASSES = new Set(['full-block', 'tail']);

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function assertDigest(value, label) {
  if (!/^sha256:[0-9a-f]{64}$/.test(value ?? '')) {
    throw new Error(`registered calibration: ${label} must be sha256:<64 hex chars>`);
  }
}

function validateShape(shape, index) {
  const label = `shapeSuite[${index}]`;
  if (!isObject(shape) || typeof shape.shapeId !== 'string' || !shape.shapeId) {
    throw new Error(`registered calibration: ${label}.shapeId is required`);
  }
  if (!PHASES.has(shape.phase)) {
    throw new Error(`registered calibration: ${label}.phase must be prefill or decode`);
  }
  for (const field of ['sequenceLength', 'batch']) {
    if (!Number.isInteger(shape[field]) || shape[field] <= 0) {
      throw new Error(`registered calibration: ${label}.${field} must be a positive integer`);
    }
  }
  for (const field of ['query', 'kv', 'dim']) {
    if (!Number.isInteger(shape.heads?.[field]) || shape.heads[field] <= 0) {
      throw new Error(`registered calibration: ${label}.heads.${field} is required`);
    }
  }
  if (!TAIL_CLASSES.has(shape.tailClass)) {
    throw new Error(`registered calibration: ${label}.tailClass is invalid`);
  }
  for (const field of ['input', 'weight', 'output', 'kv']) {
    if (typeof shape.layouts?.[field] !== 'string' || !shape.layouts[field]) {
      throw new Error(`registered calibration: ${label}.layouts.${field} is required`);
    }
  }
  for (const field of ['storage', 'materialization', 'accumulation']) {
    if (typeof shape.dtypes?.[field] !== 'string' || !shape.dtypes[field]) {
      throw new Error(`registered calibration: ${label}.dtypes.${field} is required`);
    }
  }
  for (const field of ['fusionRole', 'quantizationFormat']) {
    if (typeof shape[field] !== 'string' || !shape[field]) {
      throw new Error(`registered calibration: ${label}.${field} is required`);
    }
  }
}

export function digestRegisteredVariantDescriptor(operation, variantId, descriptor) {
  return computeCanonicalSha256({ operation, variantId, descriptor });
}

function resolveVariant(registry, reference, capabilities) {
  const descriptor = registry?.operations?.[reference.operation]?.variants?.[reference.variantId];
  if (!descriptor) {
    throw new Error(
      `registered calibration: unregistered variant ${reference.operation}/${reference.variantId}`
    );
  }
  const descriptorDigest = digestRegisteredVariantDescriptor(
    reference.operation,
    reference.variantId,
    descriptor
  );
  assertDigest(reference.descriptorDigest, `${reference.operation}/${reference.variantId} descriptorDigest`);
  assertDigest(reference.kernelDigest, `${reference.operation}/${reference.variantId} kernelDigest`);
  if (reference.descriptorDigest !== descriptorDigest) {
    throw new Error(
      `registered calibration: descriptor digest mismatch for ` +
      `${reference.operation}/${reference.variantId}`
    );
  }
  const available = new Set(capabilities);
  const missingCapabilities = (descriptor.requires ?? []).filter(
    (capability) => !available.has(capability)
  );
  return {
    reference: { ...reference },
    descriptor,
    descriptorDigest,
    compatible: missingCapabilities.length === 0,
    missingCapabilities,
  };
}

function buildSelectionPolicy(candidate) {
  const requiredCapabilities = [...(candidate.descriptor.requires ?? [])].sort();
  const precision = candidate.descriptor.precision ?? {};
  const usesF16 = requiredCapabilities.includes('shader-f16')
    || candidate.descriptor.inputDtype === 'f16'
    || candidate.descriptor.outputDtype === 'f16'
    || precision.inputDtype === 'f16'
    || precision.outputDtype === 'f16'
    || precision.activationDtype === 'f16';
  return {
    precisionPreference: usesF16 ? 'f16' : 'best-proven',
    afterPromotion: usesF16
      ? 'required-on-compatible-hardware'
      : 'selected-for-matching-evidence-scope',
    requiredCapabilities,
    fallback:
      'baseline-only-when-capability-incompatible-or-promoted-evidence-is-revoked',
  };
}

export function validateRegisteredVariantCalibrationPlan(plan, registry) {
  if (plan?.schema !== REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA) {
    throw new Error(
      `registered calibration: expected ${REGISTERED_VARIANT_CALIBRATION_PLAN_SCHEMA}`
    );
  }
  for (const field of [
    'artifactDigest',
    'manifestDigest',
    'executionGraphDigest',
    'executionEngineDigest',
    'browserDigest',
    'adapterDigest',
    'wrapperDigest',
  ]) {
    assertDigest(plan.identity?.[field], `identity.${field}`);
  }
  if (!Array.isArray(plan.identity?.capabilities)) {
    throw new Error('registered calibration: identity.capabilities is required');
  }
  if (!Array.isArray(plan.shapeSuite) || plan.shapeSuite.length === 0) {
    throw new Error('registered calibration: shapeSuite is required');
  }
  plan.shapeSuite.forEach(validateShape);
  if (!Array.isArray(plan.candidates) || plan.candidates.length === 0) {
    throw new Error('registered calibration: candidates are required');
  }
  const resolvedCandidates = plan.candidates.map((candidate) => (
    resolveVariant(registry, candidate, plan.identity.capabilities)
  ));
  const baseline = resolveVariant(registry, plan.baseline, plan.identity.capabilities);
  if (!baseline.compatible) {
    throw new Error('registered calibration: baseline is incompatible with capability fingerprint');
  }
  const baselineKey = `${baseline.reference.operation}/${baseline.reference.variantId}`;
  const candidateKeys = new Set();
  for (const candidate of resolvedCandidates) {
    const key = `${candidate.reference.operation}/${candidate.reference.variantId}`;
    if (key === baselineKey) {
      throw new Error('registered calibration: candidate must differ from baseline');
    }
    if (candidateKeys.has(key)) {
      throw new Error(`registered calibration: duplicate candidate ${key}`);
    }
    candidateKeys.add(key);
  }
  return {
    ...plan,
    identity: {
      ...plan.identity,
      capabilities: [...plan.identity.capabilities].sort(),
    },
    baseline,
    resolvedCandidates,
  };
}

function gatePassed(result, mode, candidate = null) {
  if (mode === 'operator-reference') {
    return result?.passed === true
      && result?.kernelDigest === candidate?.reference?.kernelDigest;
  }
  if (mode === 'boundary-pack') {
    return result?.schema === 'doppler.boundary-comparison-receipt/v1'
      && result?.promotionGate?.boundaryCompatible === true
      && result?.promotionGate?.sourcePrecisionControlPassed === true;
  }
  return result?.exact === true && result?.tokenCount >= 128;
}

async function evaluateCorrectnessGates(plan, candidate, runCorrectness) {
  const operatorReference = [];
  for (const shape of plan.shapeSuite) {
    const result = await runCorrectness({
      mode: 'operator-reference',
      identity: plan.identity,
      baseline: plan.baseline,
      candidate,
      shape,
    });
    operatorReference.push({ shapeId: shape.shapeId, result });
    if (!gatePassed(result, 'operator-reference', candidate)) {
      return {
        passed: false,
        failedGate: 'operator-reference',
        operatorReference,
      };
    }
  }
  const boundaryPack = await runCorrectness({
    mode: 'boundary-pack',
    identity: plan.identity,
    baseline: plan.baseline,
    candidate,
    shapeSuite: plan.shapeSuite,
  });
  if (!gatePassed(boundaryPack, 'boundary-pack')) {
    return { passed: false, failedGate: 'boundary-pack', operatorReference, boundaryPack };
  }
  const tokenParity = await runCorrectness({
    mode: 'token-parity',
    identity: plan.identity,
    baseline: plan.baseline,
    candidate,
    minimumTokenCount: 128,
  });
  return {
    passed: gatePassed(tokenParity, 'token-parity'),
    failedGate: gatePassed(tokenParity, 'token-parity') ? null : 'token-parity',
    operatorReference,
    boundaryPack,
    tokenParity,
  };
}

export async function calibrateRegisteredVariants(planInput, options) {
  if (typeof options?.runCorrectness !== 'function') {
    throw new Error('registered calibration: runCorrectness callback is required');
  }
  if (typeof options?.evaluatePerformance !== 'function') {
    throw new Error('registered calibration: evaluatePerformance callback is required');
  }
  const plan = validateRegisteredVariantCalibrationPlan(planInput, options.registry);
  const results = [];
  for (const candidate of plan.resolvedCandidates) {
    if (!candidate.compatible) {
      results.push({
        candidate: candidate.reference,
        decision: 'incompatible',
        missingCapabilities: candidate.missingCapabilities,
      });
      continue;
    }
    const correctness = await evaluateCorrectnessGates(
      plan,
      candidate,
      options.runCorrectness
    );
    if (!correctness.passed) {
      results.push({
        candidate: candidate.reference,
        correctness,
        decision: 'rejected',
        reason: `${correctness.failedGate}_failed`,
      });
      continue;
    }
    const typedCandidate = {
      kind: 'registered-kernel-variant',
      reference: {
        operation: candidate.reference.operation,
        variantId: candidate.reference.variantId,
        descriptorDigest: candidate.descriptorDigest,
        kernelDigest: candidate.reference.kernelDigest,
      },
      scope: {
        artifactDigest: plan.identity.artifactDigest,
        executionGraphDigest: plan.identity.executionGraphDigest,
        executionEngineDigest: plan.identity.executionEngineDigest,
        browserDigest: plan.identity.browserDigest,
        adapterDigest: plan.identity.adapterDigest,
        shapeSignatures: plan.shapeSuite,
      },
    };
    const performance = await options.evaluatePerformance({
      typedCandidate,
      baseline: plan.baseline,
      candidate,
      shapeSuite: plan.shapeSuite,
      identity: plan.identity,
    });
    if (performance?.schema !== 'doppler.runtime-optimization-receipt/v1') {
      throw new Error(
        'registered calibration: evaluatePerformance must return ' +
        'doppler.runtime-optimization-receipt/v1'
      );
    }
    const accepted = performance?.decision?.accepted === true;
    results.push({
      candidate: candidate.reference,
      correctness,
      performance,
      decision: accepted ? 'proposed' : 'rejected',
      proposal: accepted
        ? {
          kind: plan.outputKind ?? 'registered-execution-graph-patch',
          activation: 'manual-promotion-required',
          candidate: typedCandidate,
          selectionPolicy: buildSelectionPolicy(candidate),
        }
        : null,
      reason: accepted ? null : 'performance_evidence_failed',
    });
  }
  const core = {
    schema: REGISTERED_VARIANT_CALIBRATION_RECEIPT_SCHEMA,
    planDigest: computeCanonicalSha256(planInput),
    identity: plan.identity,
    results,
    proposedSelections: results
      .filter((result) => result.decision === 'proposed')
      .map((result) => result.proposal),
    precisionSelectionPolicy: {
      policy: 'prefer-proven-f16',
      rule:
        'On shader-f16 hardware, a promoted F16 candidate is required for its ' +
        'bound artifact, execution graph, adapter, execution engine, browser when present, ' +
        'phase, and shape scope.',
      gates: [
        'compatible-hardware',
        'operator-reference',
        'source-boundary-pack',
        'exact-128-token-parity',
        'positive-paired-performance',
        'neighboring-workload-guards',
      ],
    },
    runtimeMutationApplied: false,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}
