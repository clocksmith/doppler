import { isPlainObject } from '../utils/plain-object.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const WGSL_REPAIR_SEMANTIC_READINESS_SCHEMA_ID =
  'doppler.wgsl-repair-semantic-readiness/v1';

function stableJson(value) {
  return JSON.stringify(stableSortObject(value));
}

function hashStableJson(value) {
  return sha256Hex(stableJson(value));
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function nonFiniteMatches(expected, actual) {
  if (Number.isNaN(expected)) return Number.isNaN(actual);
  if (expected === Number.POSITIVE_INFINITY) return actual === Number.POSITIVE_INFINITY;
  if (expected === Number.NEGATIVE_INFINITY) return actual === Number.NEGATIVE_INFINITY;
  return false;
}

export function evaluateNumericAgreement(expectedValues, actualValues, tolerance) {
  const expected = Array.from(expectedValues || []);
  const actual = Array.from(actualValues || []);
  const mode = tolerance?.mode;
  const absTolerance = Number(tolerance?.absTolerance);
  const relTolerance = Number(tolerance?.relTolerance);
  const mismatches = [];
  let maxAbsError = 0;
  let maxRelError = 0;
  if (expected.length !== actual.length) {
    return {
      pass: false,
      expectedElements: expected.length,
      actualElements: actual.length,
      mismatchCount: Math.max(expected.length, actual.length),
      maxAbsError: null,
      maxRelError: null,
      mismatches: [{ index: -1, reason: 'length_mismatch' }],
    };
  }
  if (!['exact', 'numeric'].includes(mode)) {
    throw new Error('WGSL semantic verifier: tolerance.mode must be exact or numeric.');
  }
  if (!finiteNumber(absTolerance) || absTolerance < 0 || !finiteNumber(relTolerance) || relTolerance < 0) {
    throw new Error('WGSL semantic verifier: tolerances must be finite non-negative numbers.');
  }
  for (let index = 0; index < expected.length; index += 1) {
    const reference = expected[index];
    const observed = actual[index];
    if (!finiteNumber(reference) || !finiteNumber(observed)) {
      if (!nonFiniteMatches(reference, observed)) {
        mismatches.push({ index, reason: 'nonfinite_mismatch', expected: reference, actual: observed });
      }
      continue;
    }
    const absError = Math.abs(observed - reference);
    const relError = absError / Math.max(Math.abs(reference), Number.MIN_VALUE);
    maxAbsError = Math.max(maxAbsError, absError);
    maxRelError = Math.max(maxRelError, relError);
    const pass = mode === 'exact'
      ? Object.is(observed, reference) || observed === reference
      : absError <= absTolerance + relTolerance * Math.abs(reference);
    if (!pass) {
      mismatches.push({
        index,
        reason: 'numeric_mismatch',
        expected: reference,
        actual: observed,
        absError,
        relError,
      });
    }
  }
  return {
    pass: mismatches.length === 0,
    expectedElements: expected.length,
    actualElements: actual.length,
    mismatchCount: mismatches.length,
    maxAbsError,
    maxRelError,
    mismatches: mismatches.slice(0, 16),
  };
}

function allTrue(value, keys) {
  return isPlainObject(value) && keys.every((key) => value[key] === true);
}

export function evaluateWgslSemanticTaskEvidence(policy, evidence) {
  if (!isPlainObject(policy) || !isPlainObject(evidence)) {
    throw new Error('WGSL semantic verifier: policy and evidence must be objects.');
  }
  const blockers = new Set();
  if (evidence.responseContractPass !== true) blockers.add('response_contract_violation');
  if (evidence.compilation?.status !== 'pass') blockers.add('compilation_failure');
  const variants = Array.isArray(evidence.variants) ? evidence.variants : [];
  const shapeClasses = new Set(variants.map((variant) => variant.shapeClass));
  const shapeIds = new Set(variants.map((variant) => variant.shapeId));
  const workgroupIds = new Set(variants.map((variant) => variant.workgroupId));
  const shapeContract = policy.taskContract.shapeVariation;
  if (shapeIds.size < shapeContract.minimumDistinctShapesPerTask) {
    blockers.add('shape_variant_failure');
  }
  for (const required of shapeContract.requiredClasses) {
    if (!shapeClasses.has(required)) blockers.add('shape_variant_failure');
  }
  if (workgroupIds.size < policy.taskContract.workgroupVariation.minimumSemanticallyValidVariantsPerTask) {
    blockers.add('workgroup_variant_failure');
  }
  const allowedMetamorphic = new Set(policy.taskContract.metamorphic.allowedRelations);
  const passedMetamorphic = new Set();
  const variantResults = [];
  for (const variant of variants) {
    if (variant.dispatch?.status !== 'pass' || variant.dispatch?.backend !== 'chromium_webgpu') {
      blockers.add('dispatch_failure');
    }
    const numeric = evaluateNumericAgreement(
      variant.oracle?.expected,
      variant.oracle?.actual,
      variant.oracle?.tolerance
    );
    if (!numeric.pass) {
      const hasNonFinite = numeric.mismatches.some((entry) => entry.reason === 'nonfinite_mismatch');
      blockers.add(hasNonFinite ? 'nonfinite_mismatch' : 'cpu_oracle_mismatch');
    }
    const boundsPass = allTrue(variant.bufferBounds, [
      'prefixCanaryIntact',
      'suffixCanaryIntact',
      'readOnlyBuffersUnchanged',
      'outputPaddingUnchanged',
      'validationErrorsAbsent',
    ]);
    if (!boundsPass) {
      if (variant.bufferBounds?.prefixCanaryIntact !== true
        || variant.bufferBounds?.suffixCanaryIntact !== true) {
        blockers.add('buffer_canary_corruption');
      }
      if (variant.bufferBounds?.readOnlyBuffersUnchanged !== true) {
        blockers.add('read_only_buffer_mutation');
      }
      if (variant.bufferBounds?.outputPaddingUnchanged !== true) {
        blockers.add('output_padding_mutation');
      }
      if (variant.bufferBounds?.validationErrorsAbsent !== true) blockers.add('dispatch_failure');
    }
    for (const relation of variant.metamorphic || []) {
      if (allowedMetamorphic.has(relation.id) && relation.status === 'pass') {
        passedMetamorphic.add(relation.id);
      } else {
        blockers.add('metamorphic_failure');
      }
    }
    variantResults.push({
      shapeId: variant.shapeId,
      shapeClass: variant.shapeClass,
      workgroupId: variant.workgroupId,
      dispatchPass: variant.dispatch?.status === 'pass',
      numeric,
      boundsPass,
    });
  }
  if (passedMetamorphic.size < policy.taskContract.metamorphic.minimumApplicableRelationsPerTask) {
    blockers.add('metamorphic_failure');
  }
  if (evidence.historicalRegressionsPass !== true) blockers.add('historical_regression');
  const resultCore = {
    taskId: evidence.taskId || null,
    pass: blockers.size === 0,
    variants: variantResults,
    distinctShapeCount: shapeIds.size,
    distinctWorkgroupCount: workgroupIds.size,
    passedMetamorphicRelations: [...passedMetamorphic].sort(),
    blockers: [...blockers].sort(),
  };
  return { ...resultCore, resultHash: hashStableJson(resultCore) };
}

export function evaluateWgslSemanticReadiness(options = {}) {
  const policy = options.policy;
  if (!isPlainObject(policy) || policy.policyId !== 'doppler-wgsl-repair-v13-semantic-evaluation') {
    throw new Error('WGSL semantic readiness: unsupported policy.');
  }
  const blockers = new Set(Array.isArray(policy.blockers) ? policy.blockers : []);
  if (options.predecessorVerified !== true) blockers.add('v12_predecessor_identity_verification_failed');
  if (options.preservationReceipt?.decision !== 'complete') {
    blockers.add('v12_adapter_external_preservation_incomplete');
  }
  const populationBlockerIds = {
    calibration: 'calibration',
    checkpointSelection: 'checkpoint_selection',
    seedConfirmation: 'seed_confirmation',
    promotion: 'promotion',
  };
  for (const [role, population] of Object.entries(policy.populations)) {
    if (role === 'overlapPolicy') continue;
    if (population.status !== 'frozen' || !population.manifestPath || !population.populationHash) {
      blockers.add(`semantic_${populationBlockerIds[role] || role}_population_unmaterialized`);
    }
  }
  if (policy.candidate.seedSelectionStatus !== 'selected'
    || !policy.candidate.adapterSha256
    || !policy.candidate.adapterPath) {
    blockers.add('external20_seed_checkpoint_not_selected');
  }
  if (!policy.candidate.trainerToDopplerParityReceipt) {
    blockers.add('trainer_to_doppler_adapter_parity_absent');
  }
  if (!policy.taskContract.cpuOracle.implementationRevision) {
    blockers.add('cpu_oracle_implementation_revision_absent');
  }
  if (!policy.taskContract.historicalRegressions.manifestPath
    || !policy.taskContract.historicalRegressions.manifestSha256) {
    blockers.add('historical_regression_manifest_absent');
  }
  const taskEvidence = Array.isArray(options.taskEvidence)
    ? options.taskEvidence.map((entry) => evaluateWgslSemanticTaskEvidence(policy, entry))
    : [];
  if (taskEvidence.length === 0) blockers.add('semantic_dispatch_evidence_absent');
  if (taskEvidence.some((entry) => !entry.pass)) blockers.add('semantic_task_failure');

  const executionPrerequisiteBlockers = [...blockers].filter(
    (blocker) => !['semantic_dispatch_evidence_absent', 'semantic_task_failure'].includes(blocker)
  );
  const semanticEvaluationAllowed = executionPrerequisiteBlockers.length === 0
    && policy.status !== 'frozen_requirements_populations_unmaterialized';
  const semanticClaimAllowed = semanticEvaluationAllowed
    && taskEvidence.length > 0
    && taskEvidence.every((entry) => entry.pass)
    && blockers.size === 0;
  const wgslDoctorAllowed = semanticClaimAllowed && policy.productization.wgslDoctorAllowed === true;
  const core = {
    schema: WGSL_REPAIR_SEMANTIC_READINESS_SCHEMA_ID,
    policyId: policy.policyId,
    policyStatus: policy.status,
    predecessorVerified: options.predecessorVerified === true,
    preservationDecision: options.preservationReceipt?.decision ?? null,
    candidate: policy.candidate,
    taskEvidence,
    admission: {
      semanticEvaluationAllowed,
      semanticClaimAllowed,
      wgslDoctorAllowed,
      autonomousShaderAuthorAllowed: false,
    },
    decision: wgslDoctorAllowed
      ? 'wgsl_doctor_allowed'
      : semanticClaimAllowed
        ? 'semantic_claim_allowed'
        : semanticEvaluationAllowed
          ? 'semantic_evaluation_allowed'
          : 'blocked',
    blockers: [...blockers].sort(),
  };
  return { ...core, receiptHash: hashStableJson(core) };
}
