import { ERROR_CODES } from '../../../errors/doppler-error.js';

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function validateNonEmptyString(errors, value, path) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    errors.push(`Missing or invalid ${path}`);
  }
}

function validateNullableHash(errors, value, path) {
  if (value === null) {
    return;
  }
  validateNonEmptyString(errors, value, path);
}

function compareExpectedHash(errors, actual, expected, path, code) {
  if (expected == null) {
    return null;
  }
  if (actual !== expected) {
    errors.push(`${path} mismatch: expected ${expected}, got ${actual}`);
    return code;
  }
  return null;
}

export function validateDistributedPlan(plan, options = {}) {
  const errors = [];
  let code = null;

  if (!isPlainObject(plan)) {
    return {
      valid: false,
      errors: ['Distributed plan must be an object'],
      code: ERROR_CODES.DISTRIBUTED_PLAN_INVALID,
    };
  }
  if (plan.rdrd !== 1) {
    return {
      valid: false,
      errors: [`Unsupported distributed plan version ${plan.rdrd}`],
      code: ERROR_CODES.DISTRIBUTED_PLAN_UNSUPPORTED,
    };
  }
  if (!isPlainObject(plan.compatibility)) {
    return {
      valid: false,
      errors: ['Missing or invalid compatibility block'],
      code: ERROR_CODES.DISTRIBUTED_PLAN_INVALID,
    };
  }

  const compatibility = plan.compatibility;
  validateNonEmptyString(errors, compatibility.artifactIdentityHash, 'compatibility.artifactIdentityHash');
  validateNonEmptyString(errors, compatibility.manifestHash, 'compatibility.manifestHash');
  validateNonEmptyString(errors, compatibility.executionGraphDigest, 'compatibility.executionGraphDigest');
  validateNullableHash(errors, compatibility.integrityExtensionsHash, 'compatibility.integrityExtensionsHash');
  validateNonEmptyString(errors, compatibility.synthesizerVersion, 'compatibility.synthesizerVersion');
  validateNonEmptyString(errors, compatibility.synthesizedAt, 'compatibility.synthesizedAt');

  if (!Array.isArray(plan.plans) || plan.plans.length === 0) {
    errors.push('Distributed plan requires a non-empty plans array');
  } else {
    for (const [index, entry] of plan.plans.entries()) {
      if (!isPlainObject(entry)) {
        errors.push(`plans[${index}] must be an object`);
        continue;
      }
      validateNonEmptyString(errors, entry.id, `plans[${index}].id`);
      validateNonEmptyString(errors, entry.topologyHash, `plans[${index}].topologyHash`);
    }
  }

  const expectedCompatibility = options.expectedCompatibility ?? null;
  if (expectedCompatibility != null && isPlainObject(expectedCompatibility)) {
    code = code ?? compareExpectedHash(
      errors,
      compatibility.artifactIdentityHash,
      expectedCompatibility.artifactIdentityHash,
      'compatibility.artifactIdentityHash',
      ERROR_CODES.DISTRIBUTED_PLAN_ARTIFACT_MISMATCH
    );
    code = code ?? compareExpectedHash(
      errors,
      compatibility.manifestHash,
      expectedCompatibility.manifestHash,
      'compatibility.manifestHash',
      ERROR_CODES.DISTRIBUTED_PLAN_STALE
    );
    code = code ?? compareExpectedHash(
      errors,
      compatibility.executionGraphDigest,
      expectedCompatibility.executionGraphDigest,
      'compatibility.executionGraphDigest',
      ERROR_CODES.DISTRIBUTED_PLAN_STALE
    );
    code = code ?? compareExpectedHash(
      errors,
      compatibility.integrityExtensionsHash,
      expectedCompatibility.integrityExtensionsHash,
      'compatibility.integrityExtensionsHash',
      ERROR_CODES.DISTRIBUTED_PLAN_STALE
    );
  }

  if (options.expectedPlanId != null) {
    const selectedPlan = Array.isArray(plan.plans)
      ? plan.plans.find((entry) => entry?.id === options.expectedPlanId)
      : null;
    if (!selectedPlan) {
      errors.push(`No plan with id "${options.expectedPlanId}" found`);
      code = code ?? ERROR_CODES.DISTRIBUTED_PLAN_INVALID;
    } else if (options.expectedTopologyHash != null && selectedPlan.topologyHash !== options.expectedTopologyHash) {
      errors.push(
        `plans.${options.expectedPlanId}.topologyHash mismatch: expected ${options.expectedTopologyHash}, got ${selectedPlan.topologyHash}`
      );
      code = code ?? ERROR_CODES.DISTRIBUTED_PLAN_TOPOLOGY_UNFIT;
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    code: errors.length === 0 ? null : (code ?? ERROR_CODES.DISTRIBUTED_PLAN_INVALID),
  };
}

