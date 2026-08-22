/**
 * Doppler TargetPlan: Concrete Target Implementation and Specialization Plan
 *
 * @module config/target-plan
 */

import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const TARGET_PLAN_SCHEMA_ID = 'doppler.target-plan/v1';
export const TARGET_PLAN_SCHEMA_VERSION = 1;

/**
 * Validates that an object conforms to the TargetPlan contract.
 *
 * @param {object} plan
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateTargetPlan(plan) {
  const errors = [];
  if (!plan || typeof plan !== 'object' || Array.isArray(plan)) {
    return { ok: false, errors: ['TargetPlan must be a non-null object.'] };
  }
  if (plan.schema !== TARGET_PLAN_SCHEMA_ID) {
    errors.push(`schema must be "${TARGET_PLAN_SCHEMA_ID}", received "${plan.schema}".`);
  }
  if (plan.schemaVersion !== TARGET_PLAN_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${TARGET_PLAN_SCHEMA_VERSION}, received ${plan.schemaVersion}.`);
  }
  if (typeof plan.targetId !== 'string' || !plan.targetId.trim()) {
    errors.push('targetId must be a non-empty string.');
  }
  if (typeof plan.modelId !== 'string' || !plan.modelId.trim()) {
    errors.push('modelId must be a non-empty string.');
  }
  if (!plan.capabilityPredicate || typeof plan.capabilityPredicate !== 'object') {
    errors.push('capabilityPredicate must be an object.');
  }
  if (!plan.dtypes || typeof plan.dtypes !== 'object') {
    errors.push('dtypes must be an object.');
  }
  if (!Array.isArray(plan.kernelClosure) || plan.kernelClosure.length === 0) {
    errors.push('kernelClosure must be a non-empty array of kernel descriptors.');
  }
  if (!plan.memoryLayout || typeof plan.memoryLayout !== 'object') {
    errors.push('memoryLayout must be an object.');
  }
  if (!plan.phases || typeof plan.phases !== 'object') {
    errors.push('phases must be an object.');
  }

  return {
    ok: errors.length === 0,
    errors,
  };
}

/**
 * Canonical cryptographic digest of a TargetPlan object.
 *
 * @param {object} plan
 * @returns {`sha256:${string}`}
 */
export function hashTargetPlan(plan) {
  const validation = validateTargetPlan(plan);
  if (!validation.ok) {
    throw new Error(`Cannot hash invalid TargetPlan: ${validation.errors.join('; ')}`);
  }
  const canonicalJson = JSON.stringify(stableSortObject(plan));
  return `sha256:${sha256Hex(canonicalJson)}`;
}

/**
 * Evaluates whether a device's hardware profile satisfies a TargetPlan's capability predicate.
 *
 * @param {object} targetPlan
 * @param {object} deviceProfile
 * @returns {boolean}
 */
export function matchesDeviceCapability(targetPlan, deviceProfile) {
  if (!targetPlan?.capabilityPredicate || !deviceProfile) {
    return false;
  }
  const pred = targetPlan.capabilityPredicate;
  if (pred.requiresF16 && !deviceProfile.hasF16) {
    return false;
  }
  if (pred.requiresSubgroups && !deviceProfile.hasSubgroups) {
    return false;
  }
  if (typeof pred.minBufferSize === 'number' && (deviceProfile.maxBufferSize || 0) < pred.minBufferSize) {
    return false;
  }
  if (Array.isArray(pred.supportedVendors) && pred.supportedVendors.length > 0) {
    const vendor = (deviceProfile.adapter?.vendor || '').toLowerCase();
    const matches = pred.supportedVendors.some((v) => vendor.includes(v.toLowerCase()));
    if (!matches) return false;
  }
  return true;
}

/**
 * Creates and validates a TargetPlan structure.
 *
 * @param {object} params
 * @returns {object} TargetPlan
 */
export function createTargetPlan(params) {
  const plan = {
    schema: TARGET_PLAN_SCHEMA_ID,
    schemaVersion: TARGET_PLAN_SCHEMA_VERSION,
    targetId: String(params.targetId || '').trim(),
    modelId: String(params.modelId || '').trim(),
    capabilityPredicate: params.capabilityPredicate || { requiresF16: false, requiresSubgroups: false, minBufferSize: 0 },
    dtypes: params.dtypes || { activation: 'f32', kv: 'f32', weight: 'f32' },
    fusions: Array.isArray(params.fusions) ? params.fusions : [],
    kernelClosure: Array.isArray(params.kernelClosure) ? params.kernelClosure : [],
    memoryLayout: params.memoryLayout || { kvCacheLayout: 'contiguous', estimatedPeakBytes: 0, bufferSlots: [] },
    phases: params.phases || { prefill: [], decode: [] },
  };

  const validation = validateTargetPlan(plan);
  if (!validation.ok) {
    throw new Error(`Failed to create valid TargetPlan: ${validation.errors.join('; ')}`);
  }
  return plan;
}
