import {
  buildExecutionV0KernelProfileKey,
  indexExecutionV0KernelProfiles,
  normalizeExecutionV0Dtype,
  resolveExecutionV0KernelProfile,
  resolveExecutionV0KVIO,
  resolveExecutionV0Precision,
} from '../../../config/execution-v0-contract-check.js';
import {
  EXECUTION_V0_SCHEMA_ID,
  isExecutionV0Digest,
  isExecutionV0Semver,
} from '../../../config/schema/execution-v0.schema.js';
import { KERNEL_CONFIGS } from '../../../gpu/kernels/kernel-configs.js';
import { buildKernelRefFromKernelEntry, isKernelRefBoundToKernel } from '../../../config/kernels/kernel-ref.js';

const PATCH_SET_MUTABLE_FIELDS = new Set(['precision', 'kvIO', 'constants', 'entry']);
const EXECUTION_V0_RUNTIME_KEYS = new Set(['session', 'executionPatch']);

const KERNEL_OUTPUT_CAPABILITIES = (() => {
  const byKernelEntry = new Map();
  for (const variants of Object.values(KERNEL_CONFIGS)) {
    for (const config of Object.values(variants)) {
      const kernel = config?.shaderFile;
      const entry = config?.entryPoint ?? 'main';
      if (typeof kernel !== 'string' || kernel.length === 0) continue;
      const key = `${kernel}#${entry}`;
      if (!byKernelEntry.has(key)) {
        byKernelEntry.set(key, new Set());
      }
      const outputDtype = config?.outputDtype;
      if (typeof outputDtype === 'string' && outputDtype.length > 0) {
        byKernelEntry.get(key).add(String(outputDtype).toLowerCase());
      }
    }
  }
  return byKernelEntry;
})();

const normalizeDtype = normalizeExecutionV0Dtype;
const resolvePrecision = resolveExecutionV0Precision;
const resolveKVIO = resolveExecutionV0KVIO;
const indexKernelProfiles = indexExecutionV0KernelProfiles;
const buildKernelProfileKey = buildExecutionV0KernelProfileKey;

function getKernelOutputCapabilities(step) {
  const kernel = String(step?.kernel ?? '').trim();
  const entry = String(step?.entry ?? 'main').trim() || 'main';
  if (!kernel) {
    return null;
  }
  return KERNEL_OUTPUT_CAPABILITIES.get(`${kernel}#${entry}`) ?? null;
}

export function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function normalizeStopCheckMode(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'batch' && normalized !== 'per-token') {
    throw new Error(`[ExecutionV0] ${label} must be "batch" or "per-token".`);
  }
  return normalized;
}

function normalizeKVLayout(value, label) {
  if (value == null) {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (!normalized) {
    return null;
  }
  return normalized;
}

function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`[ExecutionV0] ${label} must be an object.`);
  }
  return value;
}

function requireOwnProperty(root, key, label) {
  if (!Object.prototype.hasOwnProperty.call(root, key)) {
    throw new Error(`[ExecutionV0] ${label} is required.`);
  }
  return root[key];
}

function requireNullableObject(root, key, label) {
  const value = requireOwnProperty(root, key, label);
  if (value === null) {
    return null;
  }
  return requirePlainObject(value, label);
}

function requireArrayProperty(root, key, label) {
  const value = requireOwnProperty(root, key, label);
  if (!Array.isArray(value)) {
    throw new Error(`[ExecutionV0] ${label} must be an array.`);
  }
  return value;
}

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`[ExecutionV0] ${label} must be a positive integer.`);
  }
  return value;
}

function requireOptionalBoolean(value, label) {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== 'boolean') {
    throw new Error(`[ExecutionV0] ${label} must be a boolean when provided.`);
  }
  return value;
}

function requireDtypeProperty(root, key, label) {
  const value = requireOwnProperty(root, key, label);
  if (value == null) {
    throw new Error(`[ExecutionV0] ${label} is required.`);
  }
  return normalizeDtype(value, label);
}

function validateDecodeLoopContract(sessionDefaults) {
  const decodeLoop = requireNullableObject(sessionDefaults, 'decodeLoop', 'sessionDefaults.decodeLoop');
  if (decodeLoop === null) {
    return;
  }
  requirePositiveInteger(
    decodeLoop.batchSize,
    'sessionDefaults.decodeLoop.batchSize'
  );
  requirePositiveInteger(
    decodeLoop.readbackInterval,
    'sessionDefaults.decodeLoop.readbackInterval'
  );
  normalizeStopCheckMode(
    decodeLoop.stopCheckMode,
    'sessionDefaults.decodeLoop.stopCheckMode'
  );
  if (decodeLoop.ringTokens !== undefined) {
    requirePositiveInteger(
      decodeLoop.ringTokens,
      'sessionDefaults.decodeLoop.ringTokens'
    );
  }
  if (decodeLoop.ringStop !== undefined) {
    requirePositiveInteger(
      decodeLoop.ringStop,
      'sessionDefaults.decodeLoop.ringStop'
    );
  }
  if (decodeLoop.ringStaging !== undefined) {
    requirePositiveInteger(
      decodeLoop.ringStaging,
      'sessionDefaults.decodeLoop.ringStaging'
    );
  }
  requireOptionalBoolean(
    decodeLoop.disableCommandBatching,
    'sessionDefaults.decodeLoop.disableCommandBatching'
  );
}

export function validateManifestSessionDefaultsContract(manifestInference) {
  const sessionDefaults = requirePlainObject(
    manifestInference?.sessionDefaults,
    'manifest.inference.sessionDefaults'
  );
  const compute = requirePlainObject(
    requireOwnProperty(sessionDefaults, 'compute', 'sessionDefaults.compute'),
    'sessionDefaults.compute'
  );
  const computeDefaults = requirePlainObject(
    requireOwnProperty(compute, 'defaults', 'sessionDefaults.compute.defaults'),
    'sessionDefaults.compute.defaults'
  );
  requireDtypeProperty(
    computeDefaults,
    'activationDtype',
    'sessionDefaults.compute.defaults.activationDtype'
  );
  requireDtypeProperty(
    computeDefaults,
    'mathDtype',
    'sessionDefaults.compute.defaults.mathDtype'
  );
  requireDtypeProperty(
    computeDefaults,
    'accumDtype',
    'sessionDefaults.compute.defaults.accumDtype'
  );
  requireDtypeProperty(
    computeDefaults,
    'outputDtype',
    'sessionDefaults.compute.defaults.outputDtype'
  );
  requireArrayProperty(
    compute,
    'kernelProfiles',
    'sessionDefaults.compute.kernelProfiles'
  );
  const kvcache = requireNullableObject(sessionDefaults, 'kvcache', 'sessionDefaults.kvcache');
  if (kvcache !== null) {
    requireDtypeProperty(
      kvcache,
      'kvDtype',
      'sessionDefaults.kvcache.kvDtype'
    );
  }
  validateDecodeLoopContract(sessionDefaults);
}

function assertKernelRef(kernelRef, label) {
  if (!kernelRef) return;
  if (typeof kernelRef.id !== 'string' || kernelRef.id.trim().length === 0) {
    throw new Error(`[ExecutionV0] ${label}.id is required`);
  }
  if (!isExecutionV0Semver(kernelRef.version)) {
    throw new Error(`[ExecutionV0] ${label}.version must be semver; got "${kernelRef.version}"`);
  }
  if (!isExecutionV0Digest(kernelRef.digest)) {
    throw new Error(`[ExecutionV0] ${label}.digest must match sha256:<64-hex>`);
  }
}

export function isPhaseMatch(phase, targetPhase) {
  return phase === 'both' || phase === targetPhase;
}

export function stepHasLayer(step, layerIdx) {
  if (step.layers === 'all') return true;
  if (!Array.isArray(step.layers)) return false;
  return step.layers.includes(layerIdx);
}

export function normalizePhase(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'prefill' && normalized !== 'decode' && normalized !== 'both') {
    throw new Error(`[ExecutionV0] ${label} must be prefill|decode|both; got "${value}"`);
  }
  return normalized;
}

export function normalizeSection(value, label) {
  const normalized = String(value ?? '').trim();
  if (!['preLayer', 'layer', 'postLayer', 'sampling'].includes(normalized)) {
    throw new Error(`[ExecutionV0] ${label} must be preLayer|layer|postLayer|sampling; got "${value}"`);
  }
  return normalized;
}

export function normalizeSlot(value, label) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`[ExecutionV0] ${label} must be a non-empty string`);
  }
  return value.trim();
}

function assertKernelPrecisionCapability(step, resolvedPrecision, policies) {
  if (step.op === 'cast') {
    return;
  }
  if (policies.unsupportedPrecision !== 'error') {
    return;
  }
  const kernel = String(step.kernel ?? '').trim();
  const entry = String(step.entry ?? 'main').trim() || 'main';
  const supportedOutputDtypes = getKernelOutputCapabilities(step);
  if (!supportedOutputDtypes) {
    throw new Error(
      `[ExecutionV0] step "${step.id}" kernel "${kernel}#${entry}" ` +
      'is not present in kernel registry; cannot validate precision capability.'
    );
  }
  if (supportedOutputDtypes.size === 0) {
    return;
  }
  const outputDtype = normalizeDtype(resolvedPrecision.outputDtype, `${step.id}.precision.outputDtype`);
  if (!supportedOutputDtypes.has(outputDtype)) {
    throw new Error(
      `[ExecutionV0] step "${step.id}" outputDtype=${outputDtype} is unsupported by ` +
      `kernel "${kernel}#${entry}" (supported: ${[...supportedOutputDtypes].join(', ') || 'none'}).`
    );
  }
}

export function createSourceTrace() {
  return {
    session: {},
    steps: {},
  };
}

export function setSourceTrace(trace, path, source) {
  if (!trace || typeof path !== 'string' || path.length === 0) return;
  trace[path] = { source };
}

function setStepSourceTrace(trace, stepId, path, source) {
  if (!trace || !stepId || !path) return;
  if (!trace.steps[stepId]) {
    trace.steps[stepId] = {};
  }
  trace.steps[stepId][path] = { source };
}

function isExecutionV0PlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

export function collectLeafPaths(value, prefix = [], out = []) {
  if (Array.isArray(value)) {
    if (prefix.length > 0) {
      out.push(prefix);
    }
    return out;
  }
  if (!isExecutionV0PlainObject(value)) {
    if (prefix.length > 0) {
      out.push(prefix);
    }
    return out;
  }
  for (const [key, child] of Object.entries(value)) {
    collectLeafPaths(child, [...prefix, key], out);
  }
  return out;
}

export function hasDefinedPath(root, pathSegments) {
  let current = root;
  for (const segment of pathSegments) {
    if (!isExecutionV0PlainObject(current) || !Object.prototype.hasOwnProperty.call(current, segment)) {
      return false;
    }
    current = current[segment];
  }
  return current !== undefined;
}

function resolveProfile(profileIndex, step) {
  return resolveExecutionV0KernelProfile(profileIndex, step);
}

export function validateStepShape(step, index) {
  if (!step || typeof step !== 'object') {
    throw new Error(`[ExecutionV0] execution.steps[${index}] must be an object`);
  }
  if (typeof step.id !== 'string' || step.id.trim().length === 0) {
    throw new Error(`[ExecutionV0] execution.steps[${index}].id is required`);
  }
  if (typeof step.op !== 'string' || step.op.trim().length === 0) {
    throw new Error(`[ExecutionV0] execution.steps[${index}].op is required`);
  }
  normalizePhase(step.phase, `execution.steps[${index}].phase`);
  normalizeSection(step.section, `execution.steps[${index}].section`);
  normalizeSlot(step.src, `execution.steps[${index}].src`);
  normalizeSlot(step.dst, `execution.steps[${index}].dst`);
  if (step.layers !== 'all' && !Array.isArray(step.layers)) {
    throw new Error(`[ExecutionV0] execution.steps[${index}].layers must be "all" or number[]`);
  }
  if (step.layers !== 'all') {
    for (const layer of step.layers) {
      if (!Number.isInteger(layer) || layer < 0) {
        throw new Error(`[ExecutionV0] execution.steps[${index}].layers must contain non-negative integers`);
      }
    }
  }
  if (step.op === 'cast') {
    normalizeDtype(step.toDtype, `execution.steps[${index}].toDtype`);
    if (step.fromDtype != null) {
      normalizeDtype(step.fromDtype, `execution.steps[${index}].fromDtype`);
    }
  } else {
    if (typeof step.kernel !== 'string' || step.kernel.trim().length === 0) {
      throw new Error(
        `[ExecutionV0] execution.steps[${index}] "${step.id}" requires kernel (non-cast op)`
      );
    }
    if (!step.kernelRef || typeof step.kernelRef !== 'object' || Array.isArray(step.kernelRef)) {
      throw new Error(
        `[ExecutionV0] execution.steps[${index}] "${step.id}" requires kernelRef {id, version, digest} (non-cast op)`
      );
    }
    assertKernelRef(step.kernelRef, `execution.steps[${index}].kernelRef`);
    const entry = String(step.entry ?? 'main').trim() || 'main';
    let expectedKernelRef;
    try {
      expectedKernelRef = buildKernelRefFromKernelEntry(step.kernel, entry);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `[ExecutionV0] execution.steps[${index}] "${step.id}" kernel "${step.kernel}#${entry}" ` +
        `cannot be content-pinned: ${message}`
      );
    }
    if (!isKernelRefBoundToKernel(step.kernelRef, step.kernel, entry)) {
      throw new Error(
        `[ExecutionV0] execution.steps[${index}] "${step.id}" kernelRef does not match kernel binding ` +
        `("${step.kernel}#${entry}"). Expected ${expectedKernelRef.id}@${expectedKernelRef.version} ${expectedKernelRef.digest}.`
      );
    }
  }
}

export function assertExecutionRuntimeOverlay(runtimeInference) {
  if (!runtimeInference || typeof runtimeInference !== 'object') {
    return;
  }
  const unknownKeys = Object.keys(runtimeInference).filter((key) => !EXECUTION_V0_RUNTIME_KEYS.has(key));
  if (unknownKeys.length > 0) {
    throw new Error(
      `[ExecutionV0] runtime.inference overlay supports only ${[...EXECUTION_V0_RUNTIME_KEYS].join(', ')}; ` +
      `got unsupported keys: ${unknownKeys.join(', ')}.`
    );
  }
}

export function validateUniqueStepIds(steps) {
  const ids = new Set();
  for (const step of steps) {
    if (ids.has(step.id)) {
      throw new Error(`[ExecutionV0] duplicate step id "${step.id}"`);
    }
    ids.add(step.id);
  }
}

export function hasExecutionV0(manifestInference) {
  return !!manifestInference?.execution && Array.isArray(manifestInference.execution.steps);
}

export function assertExecutionV0Schema(manifestInference) {
  if (!hasExecutionV0(manifestInference)) return;
  const discriminator = manifestInference?.schema ?? null;
  if (discriminator !== EXECUTION_V0_SCHEMA_ID) {
    throw new Error(
      `[ExecutionV0] manifest.inference.schema must be "${EXECUTION_V0_SCHEMA_ID}" ` +
      `when execution is present; got "${discriminator}".`
    );
  }
}

export function applyExecutionPatchAtomic(baseSteps, patch) {
  if (!patch) {
    return baseSteps;
  }
  const steps = cloneJson(baseSteps);
  const byId = new Map(steps.map((step, index) => [step.id, index]));

  for (const entry of patch.set ?? []) {
    if (!entry || typeof entry !== 'object' || typeof entry.id !== 'string') {
      throw new Error('[ExecutionV0] executionPatch.set entries require id');
    }
    if (!byId.has(entry.id)) {
      throw new Error(`[ExecutionV0] executionPatch.set target "${entry.id}" does not exist`);
    }
    for (const key of Object.keys(entry)) {
      if (key === 'id') continue;
      if (!PATCH_SET_MUTABLE_FIELDS.has(key)) {
        throw new Error(`[ExecutionV0] executionPatch.set "${entry.id}" cannot mutate "${key}"`);
      }
    }
  }

  for (const entry of patch.remove ?? []) {
    if (!entry || typeof entry !== 'object' || typeof entry.id !== 'string') {
      throw new Error('[ExecutionV0] executionPatch.remove entries require id');
    }
    if (!byId.has(entry.id)) {
      throw new Error(`[ExecutionV0] executionPatch.remove target "${entry.id}" does not exist`);
    }
  }

  for (const entry of patch.set ?? []) {
    const index = byId.get(entry.id);
    const target = steps[index];
    if (entry.precision !== undefined) target.precision = cloneJson(entry.precision);
    if (entry.kvIO !== undefined) target.kvIO = cloneJson(entry.kvIO);
    if (entry.constants !== undefined) target.constants = cloneJson(entry.constants);
    if (entry.entry !== undefined) target.entry = entry.entry;
  }

  const removeIds = new Set((patch.remove ?? []).map((entry) => entry.id));
  const removedSteps = steps.filter((step) => !removeIds.has(step.id));

  let current = removedSteps;
  const insertedAfterAnchors = new Map();
  for (const entry of patch.add ?? []) {
    if (!entry?.step || typeof entry.step !== 'object') {
      throw new Error('[ExecutionV0] executionPatch.add requires a step payload');
    }
    const hasBefore = typeof entry.insertBefore === 'string' && entry.insertBefore.length > 0;
    const hasAfter = typeof entry.insertAfter === 'string' && entry.insertAfter.length > 0;
    if (hasBefore === hasAfter) {
      throw new Error('[ExecutionV0] executionPatch.add requires exactly one of insertBefore or insertAfter');
    }
    if (current.some((step) => step.id === entry.step.id)) {
      throw new Error(`[ExecutionV0] executionPatch.add step id "${entry.step.id}" already exists`);
    }
    const anchorId = hasBefore ? entry.insertBefore : entry.insertAfter;
    const anchorIndex = current.findIndex((step) => step.id === anchorId);
    if (anchorIndex < 0) {
      throw new Error(`[ExecutionV0] executionPatch.add anchor "${anchorId}" not found`);
    }
    let insertIndex = hasBefore ? anchorIndex : anchorIndex + 1;
    if (!hasBefore) {
      const insertedIds = insertedAfterAnchors.get(anchorId) ?? [];
      while (insertIndex < current.length && insertedIds.includes(current[insertIndex].id)) {
        insertIndex += 1;
      }
    }
    current = [
      ...current.slice(0, insertIndex),
      cloneJson(entry.step),
      ...current.slice(insertIndex),
    ];
    if (!hasBefore) {
      const insertedIds = insertedAfterAnchors.get(anchorId) ?? [];
      insertedIds.push(entry.step.id);
      insertedAfterAnchors.set(anchorId, insertedIds);
    }
  }

  validateUniqueStepIds(current);
  return current;
}

export function indexRuntimePatchMeta(patch) {
  const meta = {
    addedSteps: new Set(),
    precisionFieldsByStep: new Map(),
    kvIOFieldsByStep: new Set(),
  };
  if (!patch || typeof patch !== 'object') {
    return meta;
  }

  for (const add of patch.add ?? []) {
    const stepId = add?.step?.id;
    if (typeof stepId === 'string' && stepId.length > 0) {
      meta.addedSteps.add(stepId);
    }
  }

  for (const set of patch.set ?? []) {
    const stepId = set?.id;
    if (typeof stepId !== 'string' || stepId.length === 0) continue;
    if (set.precision && typeof set.precision === 'object') {
      meta.precisionFieldsByStep.set(stepId, new Set(Object.keys(set.precision)));
    }
    if (set.kvIO && typeof set.kvIO === 'object') {
      meta.kvIOFieldsByStep.add(stepId);
    }
  }
  return meta;
}

export function requireSessionActivationDtype(
  sessionDefaults,
  label = 'sessionDefaults.compute.defaults.activationDtype'
) {
  const activationDtype = sessionDefaults?.compute?.defaults?.activationDtype;
  if (activationDtype == null) {
    throw new Error(`[ExecutionV0] ${label} is required.`);
  }
  return normalizeDtype(activationDtype, label);
}

export function createInitialSlotDtypes(sessionDefaults) {
  const activationDefault = requireSessionActivationDtype(
    sessionDefaults,
    'sessionDefaults.compute.defaults.activationDtype'
  );
  return new Map([['state', activationDefault]]);
}

function ensureCompatibleKV(step, kvIO, sessionDefaults) {
  if (step.op !== 'attention' || !kvIO) {
    return;
  }
  const runtimeKvDtypeRaw = sessionDefaults?.kvcache?.kvDtype;
  if (runtimeKvDtypeRaw == null) {
    return;
  }
  const runtimeKvDtype = normalizeDtype(runtimeKvDtypeRaw, 'sessionDefaults.kvcache.kvDtype');
  if (kvIO.readDtype !== runtimeKvDtype || kvIO.writeDtype !== runtimeKvDtype) {
    throw new Error(
      `[ExecutionV0] step "${step.id}" kvIO read/write (${kvIO.readDtype}/${kvIO.writeDtype}) ` +
      `must match sessionDefaults.kvcache.kvDtype (${runtimeKvDtype}).`
    );
  }
}

export function resolvePhaseSteps(phase, steps, sessionDefaults, profileIndex, policies, options = {}) {
  const slotDtypes = options.initialSlotDtypes
    ? new Map(options.initialSlotDtypes)
    : createInitialSlotDtypes(sessionDefaults);
  const resolved = [];
  const sourceTrace = options.sourceTrace ?? null;
  const sessionDefaultSources = options.sessionDefaultSources ?? {};
  const runtimePatchMeta = options.runtimePatchMeta ?? {
    addedSteps: new Set(),
    precisionFieldsByStep: new Map(),
    kvIOFieldsByStep: new Set(),
  };

  for (const step of steps) {
    const stepPhase = normalizePhase(step.phase, `${step.id}.phase`);
    if (!isPhaseMatch(stepPhase, phase)) continue;
    const profile = resolveProfile(profileIndex, step);
    if (
      step.kernelRef
      && !profile
      && policies.unresolvedKernel === 'error'
    ) {
      throw new Error(
        `[ExecutionV0] step "${step.id}" references kernel profile ` +
        `${step.kernelRef.id}@${step.kernelRef.version} (${step.kernelRef.digest}) ` +
        'but no matching sessionDefaults.compute.kernelProfiles entry was found.'
      );
    }
    const { precision, sources: precisionSources } = resolvePrecision(step, profile, sessionDefaults);
    const src = normalizeSlot(step.src, `${step.id}.src`);
    const dst = normalizeSlot(step.dst, `${step.id}.dst`);
    if (!slotDtypes.has(src)) {
      throw new Error(
        `[ExecutionV0] step "${step.id}" reads slot "${src}" before it is produced. ` +
        'Add an explicit producer step or cast/load bridge.'
      );
    }
    const derivedInput = slotDtypes.get(src);
    const inputDtype = normalizeDtype(precision.inputDtype ?? derivedInput, `${step.id}.precision.inputDtype`);

    if (
      policies.dtypeTransition === 'require_cast_step'
      && step.op !== 'cast'
      && inputDtype !== derivedInput
    ) {
      throw new Error(
        `[ExecutionV0] step "${step.id}" requires inputDtype=${inputDtype} ` +
        `but slot "${src}" currently holds ${derivedInput}. Insert explicit cast step.`
      );
    }

    let outputDtype = normalizeDtype(precision.outputDtype, `${step.id}.precision.outputDtype`);
    let outputDtypeSource = precisionSources.outputDtype;
    if (step.op !== 'cast' && outputDtypeSource === 'sessionDefault') {
      const declaredOutputDtypes = getKernelOutputCapabilities(step);
      if (declaredOutputDtypes && declaredOutputDtypes.size === 1) {
        outputDtype = [...declaredOutputDtypes][0];
        outputDtypeSource = 'derived';
      }
    }
    if (step.op === 'cast') {
      outputDtype = normalizeDtype(step.toDtype, `${step.id}.toDtype`);
      outputDtypeSource = 'manifest';
      const fromDtype = step.fromDtype
        ? normalizeDtype(step.fromDtype, `${step.id}.fromDtype`)
        : derivedInput;
      if (fromDtype !== derivedInput) {
        throw new Error(
          `[ExecutionV0] cast step "${step.id}" fromDtype=${fromDtype} does not match slot "${src}" dtype=${derivedInput}`
        );
      }
    }

    const resolvedPrecision = {
      inputDtype,
      mathDtype: normalizeDtype(precision.mathDtype, `${step.id}.precision.mathDtype`),
      accumDtype: normalizeDtype(precision.accumDtype, `${step.id}.precision.accumDtype`),
      outputDtype,
    };
    assertKernelPrecisionCapability(step, resolvedPrecision, policies);
    slotDtypes.set(dst, outputDtype);

    const kvIOResolved = step.op === 'attention'
      ? resolveKVIO(step, profile, sessionDefaults)
      : null;
    const kvIO = kvIOResolved?.value ?? null;
    ensureCompatibleKV(step, kvIO, sessionDefaults);

    if (sourceTrace) {
      const precisionFieldsPatched = runtimePatchMeta.precisionFieldsByStep.get(step.id) ?? new Set();
      const isAddedStep = runtimePatchMeta.addedSteps.has(step.id);
      const inputPatched = isAddedStep
        ? step.precision?.inputDtype != null
        : precisionFieldsPatched.has('inputDtype');
      const mathPatched = isAddedStep
        ? step.precision?.mathDtype != null
        : precisionFieldsPatched.has('mathDtype');
      const accumPatched = isAddedStep
        ? step.precision?.accumDtype != null
        : precisionFieldsPatched.has('accumDtype');
      const outputPatched = isAddedStep
        ? step.precision?.outputDtype != null
        : precisionFieldsPatched.has('outputDtype');
      const mathSource = precisionSources.mathDtype === 'sessionDefault'
        ? sessionDefaultSources.mathDtype ?? 'derived'
        : precisionSources.mathDtype;
      const accumSource = precisionSources.accumDtype === 'sessionDefault'
        ? sessionDefaultSources.accumDtype ?? 'derived'
        : precisionSources.accumDtype;
      const outputSource = precisionSources.outputDtype === 'sessionDefault'
        ? outputDtypeSource === 'sessionDefault'
          ? (sessionDefaultSources.outputDtype ?? 'derived')
          : outputDtypeSource
        : outputDtypeSource;
      setStepSourceTrace(sourceTrace, step.id, 'precision.inputDtype',
        inputPatched
          ? 'runtime.patch'
          : precision.inputDtype != null
            ? precisionSources.inputDtype
            : 'derived');
      setStepSourceTrace(sourceTrace, step.id, 'precision.mathDtype', mathPatched ? 'runtime.patch' : mathSource);
      setStepSourceTrace(sourceTrace, step.id, 'precision.accumDtype', accumPatched ? 'runtime.patch' : accumSource);
      setStepSourceTrace(sourceTrace, step.id, 'precision.outputDtype', outputPatched ? 'runtime.patch' : outputSource);
      if (step.op === 'attention') {
        const kvPatched = runtimePatchMeta.kvIOFieldsByStep.has(step.id)
          || (isAddedStep && !!step.kvIO);
        const kvSource = kvIOResolved?.source === 'sessionDefault'
          ? sessionDefaultSources.kvDtype ?? 'derived'
          : kvIOResolved?.source ?? 'derived';
        const resolvedKvSource = kvPatched ? 'runtime.patch' : kvSource;
        setStepSourceTrace(sourceTrace, step.id, 'kvIO.readDtype', resolvedKvSource);
        setStepSourceTrace(sourceTrace, step.id, 'kvIO.writeDtype', resolvedKvSource);
      }
    }

    resolved.push({
      ...step,
      src,
      dst,
      phase: stepPhase,
      section: normalizeSection(step.section, `${step.id}.section`),
      precision: resolvedPrecision,
      kvIO,
    });
  }

  return {
    steps: resolved,
    finalSlotDtypes: slotDtypes,
  };
}

function stripSchemaDefaultComputeDefaults(compute, manifestComputeDefaults, defaultComputeDefaults) {
  if (!compute?.defaults || !manifestComputeDefaults || !defaultComputeDefaults) {
    return compute;
  }
  const dtypeKeys = ['activationDtype', 'mathDtype', 'accumDtype', 'outputDtype'];
  const hasManifestDtype = dtypeKeys.some(
    (key) => manifestComputeDefaults[key] !== undefined && manifestComputeDefaults[key] !== null
  );
  if (!hasManifestDtype) {
    return compute;
  }
  const nextDefaults = { ...compute.defaults };
  let changed = false;
  for (const key of dtypeKeys) {
    if (
      manifestComputeDefaults[key] !== undefined
      && manifestComputeDefaults[key] !== null
      && nextDefaults[key] === defaultComputeDefaults[key]
    ) {
      delete nextDefaults[key];
      changed = true;
    }
  }
  if (!changed) {
    return compute;
  }
  if (Object.keys(nextDefaults).length === 0) {
    const nextCompute = { ...compute };
    delete nextCompute.defaults;
    return Object.keys(nextCompute).length === 0 ? null : nextCompute;
  }
  return { ...compute, defaults: nextDefaults };
}

export function normalizeRuntimeSessionForExecutionV0(
  runtimeSession,
  manifestInference,
  defaultComputeDefaults
) {
  const manifestSessionDefaults = manifestInference?.sessionDefaults ?? null;
  const manifestProfiles = manifestSessionDefaults?.compute?.kernelProfiles;
  const hasManifestProfiles = Array.isArray(manifestProfiles) && manifestProfiles.length > 0;
  const manifestComputeDefaults = manifestSessionDefaults?.compute?.defaults ?? null;
  const hasManifestKVCache = manifestSessionDefaults?.kvcache != null;
  const hasManifestDecodeLoop = manifestSessionDefaults?.decodeLoop != null;

  if (!runtimeSession || typeof runtimeSession !== 'object') {
    return runtimeSession;
  }

  let compute = runtimeSession.compute ?? null;
  let kvcache = Object.prototype.hasOwnProperty.call(runtimeSession, 'kvcache')
    ? runtimeSession.kvcache
    : undefined;
  let decodeLoop = Object.prototype.hasOwnProperty.call(runtimeSession, 'decodeLoop')
    ? runtimeSession.decodeLoop
    : undefined;
  let changed = false;

  if (manifestComputeDefaults) {
    const stripped = stripSchemaDefaultComputeDefaults(
      compute,
      manifestComputeDefaults,
      defaultComputeDefaults
    );
    if (stripped !== compute) {
      compute = stripped;
      changed = true;
    }
  }

  if (compute && Object.prototype.hasOwnProperty.call(compute, 'kernelProfiles')) {
    const kernelProfiles = compute.kernelProfiles;
    if (Array.isArray(kernelProfiles) && kernelProfiles.length === 0 && hasManifestProfiles) {
      const nextCompute = { ...compute };
      delete nextCompute.kernelProfiles;
      compute = Object.keys(nextCompute).length === 0 ? null : nextCompute;
      changed = true;
    }
  }

  if (kvcache === null && hasManifestKVCache) {
    kvcache = undefined;
    changed = true;
  }

  if (decodeLoop === null && hasManifestDecodeLoop) {
    decodeLoop = undefined;
    changed = true;
  }

  if (!changed) {
    return runtimeSession;
  }

  const nextRuntimeSession = { ...runtimeSession };
  if (!compute) {
    delete nextRuntimeSession.compute;
  } else {
    nextRuntimeSession.compute = compute;
  }
  if (kvcache === undefined) {
    delete nextRuntimeSession.kvcache;
  } else {
    nextRuntimeSession.kvcache = kvcache;
  }
  if (decodeLoop === undefined) {
    delete nextRuntimeSession.decodeLoop;
  } else {
    nextRuntimeSession.decodeLoop = decodeLoop;
  }

  return Object.keys(nextRuntimeSession).length === 0 ? {} : nextRuntimeSession;
}

export function validatePhaseBoundaryCompatibility(options) {
  const {
    steps,
    prefillFinalSlotDtypes,
    decodeInitialSlotDtypes,
    sessionDefaults,
    profileIndex,
    policies,
  } = options;
  const decodeSteps = steps.filter((step) => isPhaseMatch(normalizePhase(step.phase, `${step.id}.phase`), 'decode'));
  if (decodeSteps.length === 0) {
    return;
  }
  const writtenSlots = new Set();
  for (const step of decodeSteps) {
    const src = normalizeSlot(step.src, `${step.id}.src`);
    const dst = normalizeSlot(step.dst, `${step.id}.dst`);
    const readsCarriedSlot = !writtenSlots.has(src) && prefillFinalSlotDtypes.has(src);
    if (readsCarriedSlot && step.op !== 'cast') {
      const profile = resolveProfile(profileIndex, step);
      const { precision } = resolvePrecision(step, profile, sessionDefaults);
      const carriedDtype = prefillFinalSlotDtypes.get(src);
      const decodeInput = normalizeDtype(
        precision.inputDtype
          ?? carriedDtype
          ?? decodeInitialSlotDtypes.get(src),
        `${step.id}.precision.inputDtype`
      );
      if (decodeInput !== carriedDtype) {
        throw new Error(
          `[ExecutionV0] decode step "${step.id}" reads carried slot "${src}" as ${decodeInput} ` +
          `but prefill left ${carriedDtype}. Add explicit cast at phase boundary.`
        );
      }
    }
    writtenSlots.add(dst);
  }
}

export function assertKVLayoutExecutionCompatibility(steps, sessionDefaults) {
  const kvLayout = normalizeKVLayout(sessionDefaults?.kvcache?.layout, 'sessionDefaults.kvcache.layout');
  if (kvLayout !== 'bdpa') {
    return;
  }
  const incompatibleStep = steps.find((step) => (
    step?.op === 'attention'
    && isPhaseMatch(normalizePhase(step.phase, `${step.id}.phase`), 'prefill')
  ));
  if (!incompatibleStep) {
    return;
  }
  throw new Error(
    `[ExecutionV0] sessionDefaults.kvcache.layout="bdpa" is decode-only, ` +
    `but step "${incompatibleStep.id}" declares prefill attention. ` +
    'Use a non-BDPA KV layout for prefill-capable models or remove prefill attention from the execution contract.'
  );
}

export {
  buildKernelProfileKey,
  indexKernelProfiles,
  normalizeDtype,
};
