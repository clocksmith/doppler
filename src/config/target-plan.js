
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';
import { validateInitialExecutionIdentity } from './initial-execution-identity.js';

export const TARGET_PLAN_SCHEMA_ID = 'doppler.target-plan/v1';
export const TARGET_PLAN_SCHEMA_VERSION = 1;
export const TARGET_PLAN_V2_SCHEMA_ID = 'doppler.target-plan/v2';
export const TARGET_PLAN_V2_SCHEMA_VERSION = 2;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const SLOT_SCOPES = new Set(['static', 'layer-recycled', 'transient', 'session']);
const SLOT_OWNERS = new Set(['runtime', 'program']);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function requireDigest(value, label, errors) {
  if (!SHA256_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}

function validateMemoryExpression(expression, label, errors) {
  if (!isObject(expression)) {
    errors.push(`${label} must be an object.`);
    return;
  }
  if (expression.op === 'constant') {
    if (!Number.isInteger(expression.bytes) || expression.bytes < 1) {
      errors.push(`${label}.bytes must be a positive integer.`);
    }
    return;
  }
  if (expression.op === 'affine') {
    if (!Number.isInteger(expression.constantBytes) || expression.constantBytes < 0) {
      errors.push(`${label}.constantBytes must be a non-negative integer.`);
    }
    if (!isObject(expression.terms) || Object.keys(expression.terms).length === 0) {
      errors.push(`${label}.terms must be a non-empty object.`);
    } else if (Object.values(expression.terms).some((coefficient) => !Number.isInteger(coefficient) || coefficient < 0)) {
      errors.push(`${label}.terms coefficients must be non-negative integers.`);
    }
    if (!Number.isInteger(expression.alignment) || expression.alignment < 1) {
      errors.push(`${label}.alignment must be a positive integer.`);
    }
    if (!Number.isInteger(expression.minimumBytes) || expression.minimumBytes < 1) {
      errors.push(`${label}.minimumBytes must be a positive integer.`);
    }
    return;
  }
  errors.push(`${label}.op must be "constant" or "affine".`);
}

function validateCommand(command, phase, label, errors) {
  if (!isObject(command)) {
    errors.push(`${label} must be an object.`);
    return;
  }
  if (command.kind === 'program-phase') {
    if (command.phase !== phase) errors.push(`${label}.phase must be "${phase}".`);
    requireDigest(command.executionGraphHash, `${label}.executionGraphHash`, errors);
    if (!Array.isArray(command.declaredStepIds) || command.declaredStepIds.length === 0
      || command.declaredStepIds.some((stepId) => typeof stepId !== 'string' || !stepId.trim())) {
      errors.push(`${label}.declaredStepIds must be a non-empty string array.`);
    }
    return;
  }
  if (command.kind === 'dispatch') {
    requireString(command.moduleId, `${label}.moduleId`, errors);
    requireString(command.entry, `${label}.entry`, errors);
    if (!Array.isArray(command.workgroups) || command.workgroups.length < 1 || command.workgroups.length > 3
      || command.workgroups.some((value) => !Number.isInteger(value) || value < 1)) {
      errors.push(`${label}.workgroups must contain one to three positive integers.`);
    }
    if (!Array.isArray(command.bindings) || command.bindings.some((binding) => (
      !isObject(binding) || !Number.isInteger(binding.binding) || binding.binding < 0
      || typeof binding.slotId !== 'string' || !binding.slotId.trim()
    ))) {
      errors.push(`${label}.bindings must bind non-negative binding indices to slot IDs.`);
    }
    return;
  }
  errors.push(`${label}.kind must be "program-phase" or "dispatch".`);
}

export function validateTargetPlan(plan) {
  const errors = [];
  if (!isObject(plan)) {
    return { ok: false, errors: ['TargetPlan must be a non-null object.'] };
  }
  const isV1 = plan.schema === TARGET_PLAN_SCHEMA_ID && plan.schemaVersion === TARGET_PLAN_SCHEMA_VERSION;
  const isV2 = plan.schema === TARGET_PLAN_V2_SCHEMA_ID && plan.schemaVersion === TARGET_PLAN_V2_SCHEMA_VERSION;
  if (!isV1 && !isV2) {
    errors.push(
      `schema/version must be "${TARGET_PLAN_SCHEMA_ID}"/${TARGET_PLAN_SCHEMA_VERSION} ` +
      `or "${TARGET_PLAN_V2_SCHEMA_ID}"/${TARGET_PLAN_V2_SCHEMA_VERSION}.`
    );
  }
  requireString(plan.targetId, 'targetId', errors);
  requireString(plan.modelId, 'modelId', errors);
  requireDigest(plan.modelIRHash, 'modelIRHash', errors);
  requireDigest(plan.executionGraphHash, 'executionGraphHash', errors);
  requireDigest(plan.programBundleHash, 'programBundleHash', errors);

  if (!isObject(plan.capabilityPredicate)) {
    errors.push('capabilityPredicate must be an object.');
  } else {
    for (const field of ['requiresF16', 'requiresSubgroups']) {
      if (typeof plan.capabilityPredicate[field] !== 'boolean') {
        errors.push(`capabilityPredicate.${field} must be boolean.`);
      }
    }
    if (!Number.isInteger(plan.capabilityPredicate.minBufferSize) || plan.capabilityPredicate.minBufferSize < 0) {
      errors.push('capabilityPredicate.minBufferSize must be a non-negative integer.');
    }
  }
  if (!isObject(plan.dtypes)) {
    errors.push('dtypes must be an object.');
  } else {
    for (const field of ['activation', 'kv', 'weight']) requireString(plan.dtypes[field], `dtypes.${field}`, errors);
  }

  if (!Array.isArray(plan.kernelClosure) || plan.kernelClosure.length === 0) {
    errors.push('kernelClosure must be a non-empty array.');
  } else {
    const moduleIds = new Set();
    for (const [index, module] of plan.kernelClosure.entries()) {
      if (!isObject(module)) {
        errors.push(`kernelClosure[${index}] must be an object.`);
        continue;
      }
      requireString(module.moduleId, `kernelClosure[${index}].moduleId`, errors);
      requireDigest(module.digest, `kernelClosure[${index}].digest`, errors);
      requireDigest(module.sourceHash, `kernelClosure[${index}].sourceHash`, errors);
      if (moduleIds.has(module.moduleId)) errors.push(`kernelClosure contains duplicate moduleId "${module.moduleId}".`);
      moduleIds.add(module.moduleId);
    }
  }

  if (!isObject(plan.memoryLayout)) {
    errors.push('memoryLayout must be an object.');
  } else {
    requireString(plan.memoryLayout.kvCacheLayout, 'memoryLayout.kvCacheLayout', errors);
    if (!Array.isArray(plan.memoryLayout.bufferSlots) || plan.memoryLayout.bufferSlots.length === 0) {
      errors.push('memoryLayout.bufferSlots must be a non-empty array.');
    } else {
      const slots = new Set();
      for (const [index, slot] of plan.memoryLayout.bufferSlots.entries()) {
        if (!isObject(slot)) {
          errors.push(`memoryLayout.bufferSlots[${index}] must be an object.`);
          continue;
        }
        requireString(slot.slotId, `memoryLayout.bufferSlots[${index}].slotId`, errors);
        requireString(slot.role, `memoryLayout.bufferSlots[${index}].role`, errors);
        if (!SLOT_SCOPES.has(slot.scope)) errors.push(`memoryLayout.bufferSlots[${index}].scope is invalid.`);
        if (!SLOT_OWNERS.has(slot.owner)) errors.push(`memoryLayout.bufferSlots[${index}].owner is invalid.`);
        if ((!Array.isArray(slot.usage) || slot.usage.length === 0)
          && (!Number.isInteger(slot.usageBits) || slot.usageBits < 1)) {
          errors.push(`memoryLayout.bufferSlots[${index}] must declare usage or usageBits.`);
        }
        validateMemoryExpression(slot.size, `memoryLayout.bufferSlots[${index}].size`, errors);
        if (slots.has(slot.slotId)) errors.push(`memoryLayout contains duplicate slotId "${slot.slotId}".`);
        slots.add(slot.slotId);
      }
    }
  }

  if (!isObject(plan.phases)) {
    errors.push('phases must be an object.');
  } else {
    for (const phase of ['prefill', 'decode']) {
      if (!Array.isArray(plan.phases[phase]) || plan.phases[phase].length === 0) {
        errors.push(`phases.${phase} must be a non-empty command array.`);
      } else {
        plan.phases[phase].forEach((command, index) => validateCommand(command, phase, `phases.${phase}[${index}]`, errors));
      }
    }
  }

  if (!Array.isArray(plan.qualification) || plan.qualification.length === 0) {
    errors.push('qualification must contain at least one evidence record.');
  } else {
    for (const [index, record] of plan.qualification.entries()) {
      if (!isObject(record)) {
        errors.push(`qualification[${index}] must be an object.`);
        continue;
      }
      requireString(record.surface, `qualification[${index}].surface`, errors);
      if (record.status !== 'passed') errors.push(`qualification[${index}].status must be "passed".`);
      requireString(record.evidenceArtifactId, `qualification[${index}].evidenceArtifactId`, errors);
      requireDigest(record.evidenceHash, `qualification[${index}].evidenceHash`, errors);
      if (record.transcriptHash !== undefined) requireDigest(record.transcriptHash, `qualification[${index}].transcriptHash`, errors);
      if (record.operation === 'rerank') {
        if (!Number.isInteger(record.rerankedDocuments) || record.rerankedDocuments < 1
          || record.generatedTokens !== undefined || record.encodedSequences !== undefined
          || !SHA256_PATTERN.test(record.transcriptHash ?? '')) {
          errors.push(`qualification[${index}] requires rerankedDocuments and transcriptHash without other operation counts.`);
        }
      } else if (record.operation === 'encodeSequence') {
        if (!Number.isInteger(record.encodedSequences) || record.encodedSequences < 1
          || record.generatedTokens !== undefined || !SHA256_PATTERN.test(record.transcriptHash ?? '')) {
          errors.push(`qualification[${index}] requires encodedSequences and transcriptHash without generatedTokens.`);
        }
      } else if (record.operation !== undefined && record.operation !== 'generate') {
        errors.push(`qualification[${index}].operation is unsupported.`);
      } else if (!Number.isInteger(record.generatedTokens) || record.generatedTokens < 1) {
        errors.push(`qualification[${index}].generatedTokens must be a positive integer.`);
      }
    }
  }

  if (isV2) {
    const identityValidation = validateInitialExecutionIdentity(plan.initialExecutionIdentity);
    if (!identityValidation.ok) {
      errors.push(...identityValidation.errors.map((error) => `initialExecutionIdentity.${error}`));
    } else if (plan.initialExecutionIdentity.executionGraphHash !== plan.executionGraphHash) {
      errors.push('initialExecutionIdentity.executionGraphHash must equal executionGraphHash.');
    }
  }

  return { ok: errors.length === 0, errors };
}

export function hashTargetPlan(plan) {
  const validation = validateTargetPlan(plan);
  if (!validation.ok) {
    throw new Error(`Cannot hash invalid TargetPlan: ${validation.errors.join('; ')}`);
  }
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(plan)))}`;
}

export function matchesDeviceCapability(targetPlan, deviceProfile) {
  if (!targetPlan?.capabilityPredicate || !deviceProfile) return false;
  const predicate = targetPlan.capabilityPredicate;
  if (predicate.requiresF16 && !deviceProfile.hasF16) return false;
  if (predicate.requiresSubgroups && !deviceProfile.hasSubgroups) return false;
  if ((deviceProfile.maxBufferSize || 0) < predicate.minBufferSize) return false;
  if (Array.isArray(predicate.supportedVendors) && predicate.supportedVendors.length > 0) {
    const vendor = String(deviceProfile.adapter?.vendor || '').toLowerCase();
    if (!predicate.supportedVendors.some((candidate) => vendor.includes(candidate.toLowerCase()))) return false;
  }
  return true;
}

export function selectQualifiedTargetPlan(targetPlans, deviceProfile) {
  if (!Array.isArray(targetPlans) || targetPlans.length === 0) {
    throw new Error('TargetSelector: Pack contains no target plans.');
  }
  if (!deviceProfile) {
    throw new Error('TargetSelector: deviceProfile is required for target selection.');
  }
  if (typeof deviceProfile.surface !== 'string' || !deviceProfile.surface.trim()) {
    throw new Error('TargetSelector: deviceProfile.surface is required for qualification selection.');
  }

  for (const plan of targetPlans) {
    const qualifiedForSurface = plan.qualification?.some((record) => (
      record.status === 'passed' && record.surface === deviceProfile.surface
    ));
    if (qualifiedForSurface && matchesDeviceCapability(plan, deviceProfile)) return plan;
  }

  const available = targetPlans.map((plan) => plan.targetId || 'unknown').join(', ');
  throw new Error(
    `TargetSelector: Device does not satisfy capability predicates and surface qualification for any prequalified target plan in Pack. Available targets: [${available}]. (surface: ${deviceProfile.surface}, hasF16: ${Boolean(deviceProfile.hasF16)}, hasSubgroups: ${Boolean(deviceProfile.hasSubgroups)})`
  );
}

export function createTargetPlan(params) {
  if (!isObject(params)) throw new Error('createTargetPlan requires an object.');
  const plan = {
    schema: TARGET_PLAN_SCHEMA_ID,
    schemaVersion: TARGET_PLAN_SCHEMA_VERSION,
    targetId: params.targetId,
    modelId: params.modelId,
    modelIRHash: params.modelIRHash,
    executionGraphHash: params.executionGraphHash,
    programBundleHash: params.programBundleHash,
    capabilityPredicate: params.capabilityPredicate,
    dtypes: params.dtypes,
    fusions: Array.isArray(params.fusions) ? params.fusions : [],
    kernelClosure: params.kernelClosure,
    memoryLayout: params.memoryLayout,
    phases: params.phases,
    qualification: params.qualification,
  };
  const validation = validateTargetPlan(plan);
  if (!validation.ok) {
    throw new Error(`Failed to create valid TargetPlan: ${validation.errors.join('; ')}`);
  }
  return plan;
}

export function createTargetPlanV2(params) {
  if (!isObject(params)) throw new Error('createTargetPlanV2 requires an object.');
  const plan = {
    schema: TARGET_PLAN_V2_SCHEMA_ID,
    schemaVersion: TARGET_PLAN_V2_SCHEMA_VERSION,
    targetId: params.targetId,
    modelId: params.modelId,
    modelIRHash: params.modelIRHash,
    executionGraphHash: params.executionGraphHash,
    programBundleHash: params.programBundleHash,
    capabilityPredicate: params.capabilityPredicate,
    dtypes: params.dtypes,
    fusions: Array.isArray(params.fusions) ? params.fusions : [],
    kernelClosure: params.kernelClosure,
    memoryLayout: params.memoryLayout,
    phases: params.phases,
    qualification: params.qualification,
    initialExecutionIdentity: params.initialExecutionIdentity,
  };
  const validation = validateTargetPlan(plan);
  if (!validation.ok) {
    throw new Error(`Failed to create valid TargetPlan v2: ${validation.errors.join('; ')}`);
  }
  return plan;
}
