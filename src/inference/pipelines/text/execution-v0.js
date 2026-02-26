import { mergeRuntimeValues } from '../../../config/runtime-merge.js';
import {
  EXECUTION_V0_SCHEMA_ID,
  DEFAULT_EXECUTION_V0_POLICIES,
  DEFAULT_EXECUTION_V0_SESSION_DEFAULTS,
  isExecutionV0Digest,
  isExecutionV0Semver,
} from '../../../config/schema/execution-v0.schema.js';
import { KERNEL_CONFIGS } from '../../../gpu/kernels/kernel-configs.js';
import { buildKernelRefFromKernelEntry, isKernelRefBoundToKernel } from '../../../config/kernels/kernel-ref.js';

const PATCH_SET_MUTABLE_FIELDS = new Set(['precision', 'kvIO', 'constants', 'entry']);
const EXECUTION_V0_RUNTIME_KEYS = new Set(['session', 'executionPatch']);
const PIPELINE_COMPATIBLE_OPS = new Set([
  'save',
  'load',
  'attention',
  'rmsnorm',
  'ffn',
  'residual_add',
  'cast',
  'noop',
]);

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

function getKernelOutputCapabilities(step) {
  const kernel = String(step?.kernel ?? '').trim();
  const entry = String(step?.entry ?? 'main').trim() || 'main';
  if (!kernel) {
    return null;
  }
  return KERNEL_OUTPUT_CAPABILITIES.get(`${kernel}#${entry}`) ?? null;
}

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function normalizeDtype(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`[ExecutionV0] ${label} must be "f16" or "f32"; got "${value}"`);
  }
  return normalized;
}

function normalizePhase(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'prefill' && normalized !== 'decode' && normalized !== 'both') {
    throw new Error(`[ExecutionV0] ${label} must be prefill|decode|both; got "${value}"`);
  }
  return normalized;
}

function normalizeSection(value, label) {
  const normalized = String(value ?? '').trim();
  if (!['preLayer', 'layer', 'postLayer', 'sampling'].includes(normalized)) {
    throw new Error(`[ExecutionV0] ${label} must be preLayer|layer|postLayer|sampling; got "${value}"`);
  }
  return normalized;
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

function isPhaseMatch(phase, targetPhase) {
  return phase === 'both' || phase === targetPhase;
}

function stepHasLayer(step, layerIdx) {
  if (step.layers === 'all') return true;
  if (!Array.isArray(step.layers)) return false;
  return step.layers.includes(layerIdx);
}

function buildKernelProfileKey(kernelRef) {
  if (!kernelRef) return null;
  return `${kernelRef.id}|${kernelRef.version}|${kernelRef.digest}`;
}

function normalizeSlot(value, label) {
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
    // Some kernels do not declare output dtype metadata yet; treat as unknown.
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

function createSourceTrace() {
  return {
    session: {},
    steps: {},
  };
}

function setSourceTrace(trace, path, source) {
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

function isPlainObject(value) {
  return value != null && typeof value === 'object' && !Array.isArray(value);
}

function collectLeafPaths(value, prefix = [], out = []) {
  if (Array.isArray(value)) {
    if (prefix.length > 0) {
      out.push(prefix);
    }
    return out;
  }
  if (!isPlainObject(value)) {
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

function hasDefinedPath(root, pathSegments) {
  let current = root;
  for (const segment of pathSegments) {
    if (!isPlainObject(current) || !Object.prototype.hasOwnProperty.call(current, segment)) {
      return false;
    }
    current = current[segment];
  }
  return current !== undefined;
}

function indexKernelProfiles(sessionDefaults) {
  const byKey = new Map();
  const profiles = sessionDefaults?.compute?.kernelProfiles ?? [];
  for (const profile of profiles) {
    assertKernelRef(profile.kernelRef, 'sessionDefaults.compute.kernelProfiles[].kernelRef');
    byKey.set(buildKernelProfileKey(profile.kernelRef), profile);
  }
  return byKey;
}

function resolveProfile(profileIndex, step) {
  const key = buildKernelProfileKey(step.kernelRef);
  if (!key) return null;
  return profileIndex.get(key) ?? null;
}

function resolvePrecision(step, profile, sessionDefaults) {
  const defaults = sessionDefaults.compute.defaults;
  const precision = {
    inputDtype: step.precision?.inputDtype
      ?? profile?.precision?.inputDtype
      ?? null,
    mathDtype: step.precision?.mathDtype
      ?? profile?.precision?.mathDtype
      ?? defaults.mathDtype,
    accumDtype: step.precision?.accumDtype
      ?? profile?.precision?.accumDtype
      ?? defaults.accumDtype,
    outputDtype: step.precision?.outputDtype
      ?? profile?.precision?.outputDtype
      ?? defaults.outputDtype,
  };
  const sources = {
    inputDtype: step.precision?.inputDtype != null
      ? 'manifest'
      : profile?.precision?.inputDtype != null
        ? 'kernelProfile'
        : 'derived',
    mathDtype: step.precision?.mathDtype != null
      ? 'manifest'
      : profile?.precision?.mathDtype != null
        ? 'kernelProfile'
        : 'sessionDefault',
    accumDtype: step.precision?.accumDtype != null
      ? 'manifest'
      : profile?.precision?.accumDtype != null
        ? 'kernelProfile'
        : 'sessionDefault',
    outputDtype: step.precision?.outputDtype != null
      ? 'manifest'
      : profile?.precision?.outputDtype != null
        ? 'kernelProfile'
        : 'sessionDefault',
  };
  return { precision, sources };
}

function resolveKVIO(step, profile, sessionDefaults) {
  if (step.kvIO) {
    return {
      value: {
        readDtype: normalizeDtype(step.kvIO.readDtype, `${step.id}.kvIO.readDtype`),
        writeDtype: normalizeDtype(step.kvIO.writeDtype, `${step.id}.kvIO.writeDtype`),
      },
      source: 'manifest',
    };
  }
  if (profile?.kvIO) {
    return {
      value: {
        readDtype: normalizeDtype(profile.kvIO.readDtype, `${step.id}.profile.kvIO.readDtype`),
        writeDtype: normalizeDtype(profile.kvIO.writeDtype, `${step.id}.profile.kvIO.writeDtype`),
      },
      source: 'kernelProfile',
    };
  }
  const kvDtype = normalizeDtype(
    sessionDefaults?.kvcache?.kvDtype ?? sessionDefaults.compute.defaults.activationDtype,
    `${step.id}.sessionDefaults.kvcache.kvDtype`
  );
  return {
    value: { readDtype: kvDtype, writeDtype: kvDtype },
    source: 'sessionDefault',
  };
}

function validateStepShape(step, index) {
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

function assertExecutionRuntimeOverlay(runtimeInference) {
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

function validateUniqueStepIds(steps) {
  const ids = new Set();
  for (const step of steps) {
    if (ids.has(step.id)) {
      throw new Error(`[ExecutionV0] duplicate step id "${step.id}"`);
    }
    ids.add(step.id);
  }
}

function assertExecutionV0Schema(manifestInference) {
  if (!hasExecutionV0(manifestInference)) return;
  const discriminator = manifestInference?.schema ?? null;
  if (discriminator !== EXECUTION_V0_SCHEMA_ID) {
    throw new Error(
      `[ExecutionV0] manifest.inference.schema must be "${EXECUTION_V0_SCHEMA_ID}" ` +
      `when execution is present; got "${discriminator}".`
    );
  }
}

function applyExecutionPatchAtomic(baseSteps, patch) {
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

function indexRuntimePatchMeta(patch) {
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

function createInitialSlotDtypes(sessionDefaults) {
  const activationDefault = normalizeDtype(
    sessionDefaults?.compute?.defaults?.activationDtype ?? 'f16',
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

function resolvePhaseSteps(phase, steps, sessionDefaults, profileIndex, policies, options = {}) {
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

function normalizeRuntimeSessionForExecutionV0(runtimeSession, manifestInference) {
  const manifestProfiles = manifestInference?.sessionDefaults?.compute?.kernelProfiles;
  const hasManifestProfiles = Array.isArray(manifestProfiles) && manifestProfiles.length > 0;

  if (!runtimeSession || !runtimeSession.compute) {
    return runtimeSession;
  }

  const compute = runtimeSession.compute;
  if (!Object.prototype.hasOwnProperty.call(compute, 'kernelProfiles')) {
    return runtimeSession;
  }

  const kernelProfiles = compute.kernelProfiles;
  if (!Array.isArray(kernelProfiles) || kernelProfiles.length > 0) {
    return runtimeSession;
  }

  if (!hasManifestProfiles) {
    return runtimeSession;
  }

  const nextCompute = { ...compute };
  delete nextCompute.kernelProfiles;
  return {
    ...runtimeSession,
    compute: nextCompute,
  };
}

function validatePhaseBoundaryCompatibility(options) {
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

function toKernelPathStep(step) {
  if (step.op === 'cast') return null;
  if (!step.kernel) return null;
  return {
    op: step.op,
    kernel: step.kernel,
    entry: step.entry ?? 'main',
    ...(step.weights ? { weights: step.weights } : {}),
    ...(step.constants ? { constants: step.constants } : {}),
  };
}

function getSectionSteps(steps, section, phase = null) {
  return steps
    .filter((step) => step.section === section)
    .filter((step) => (phase ? isPhaseMatch(step.phase, phase) : true))
    .map(toKernelPathStep)
    .filter((step) => step != null);
}

function buildLayerPhaseSteps(steps, phase, layerIdx) {
  return steps
    .filter((step) => step.section === 'layer' && isPhaseMatch(step.phase, phase))
    .filter((step) => stepHasLayer(step, layerIdx))
    .map(toKernelPathStep)
    .filter((step) => step != null);
}

function buildInlineKernelPath(steps, sessionDefaults, modelId, numLayers) {
  const activationDtype = normalizeDtype(
    sessionDefaults?.compute?.defaults?.activationDtype ?? 'f16',
    'sessionDefaults.compute.defaults.activationDtype'
  );
  const kvDtype = normalizeDtype(
    sessionDefaults?.kvcache?.kvDtype ?? activationDtype,
    'sessionDefaults.kvcache.kvDtype'
  );
  const decodeSteps = buildLayerPhaseSteps(steps, 'decode', 0);
  const prefillSteps = buildLayerPhaseSteps(steps, 'prefill', 0);
  if (decodeSteps.length === 0 && prefillSteps.length === 0) {
    return null;
  }

  const path = {
    id: `${modelId || 'model'}-execution-v0`,
    name: 'Execution v0 inline kernel path',
    description: 'Generated from manifest.inference.execution.steps',
    activationDtype,
    kvDtype,
    decode: {
      steps: decodeSteps.length > 0 ? decodeSteps : prefillSteps,
    },
    prefill: {
      steps: prefillSteps.length > 0 ? prefillSteps : decodeSteps,
    },
  };

  if (numLayers > 0) {
    const overrides = [];
    for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
      const decodeLayerSteps = buildLayerPhaseSteps(steps, 'decode', layerIdx);
      const prefillLayerSteps = buildLayerPhaseSteps(steps, 'prefill', layerIdx);
      const hasCustomDecode = JSON.stringify(decodeLayerSteps) !== JSON.stringify(path.decode.steps);
      const hasCustomPrefill = JSON.stringify(prefillLayerSteps) !== JSON.stringify(path.prefill.steps);
      if (!hasCustomDecode && !hasCustomPrefill) continue;
      // Kernel path layerOverrides are single-step lists per layer.
      const mergedLayerSteps = decodeLayerSteps.length > 0
        ? decodeLayerSteps
        : prefillLayerSteps;
      if (mergedLayerSteps.length > 0) {
        overrides.push({
          layers: [layerIdx],
          steps: mergedLayerSteps,
        });
      }
    }
    if (overrides.length > 0) {
      path.layerOverrides = overrides;
    }
  }

  const preLayer = getSectionSteps(steps, 'preLayer');
  if (preLayer.length > 0) {
    path.preLayer = preLayer;
  }
  const postLayer = getSectionSteps(steps, 'postLayer');
  if (postLayer.length > 0) {
    path.postLayer = postLayer;
  }
  const sampling = getSectionSteps(steps, 'sampling', 'decode');
  if (sampling.length > 0) {
    path.sampling = sampling;
  }

  return path;
}

function buildLayerPipelineFromExecution(steps) {
  const layerSectionSteps = steps.filter((step) => step.section === 'layer');
  if (layerSectionSteps.length === 0) {
    return null;
  }
  if (layerSectionSteps.some((step) => !PIPELINE_COMPATIBLE_OPS.has(step.op))) {
    return null;
  }

  const layerSteps = layerSectionSteps
    .map((step) => ({
      op: step.op,
      phase: step.phase,
      src: step.src ?? 'state',
      dst: step.dst ?? 'state',
      ...(step.residual !== undefined ? { residual: step.residual } : {}),
      ...(step.a !== undefined ? { a: step.a } : {}),
      ...(step.b !== undefined ? { b: step.b } : {}),
      ...(step.variant !== undefined ? { variant: step.variant } : {}),
      ...(step.skipInputNorm !== undefined ? { skipInputNorm: step.skipInputNorm } : {}),
      ...(step.precision?.inputDtype ? { inputDtype: step.precision.inputDtype } : {}),
      ...(step.precision?.outputDtype ? { outputDtype: step.precision.outputDtype } : {}),
      ...(step.fromDtype ? { fromDtype: step.fromDtype } : {}),
      ...(step.toDtype ? { toDtype: step.toDtype } : {}),
      ...(step.probeStage ? { probeStage: step.probeStage } : {}),
      ...(step.name ? { name: step.name } : {}),
      ...(step.weight ? { weight: step.weight } : {}),
    }));

  return {
    steps: layerSteps,
    overrides: [],
  };
}

function buildSessionRuntimePatch(sessionDefaults) {
  const patch = {};
  const computeDefaults = sessionDefaults?.compute?.defaults ?? null;
  const computePatch = {};
  const activationDtype = computeDefaults?.activationDtype;
  if (activationDtype) {
    computePatch.activationDtype = activationDtype;
  }
  if (computeDefaults && (computeDefaults.mathDtype || computeDefaults.accumDtype || computeDefaults.outputDtype)) {
    computePatch.defaults = {
      ...(computeDefaults.mathDtype ? { mathDtype: computeDefaults.mathDtype } : {}),
      ...(computeDefaults.accumDtype ? { accumDtype: computeDefaults.accumDtype } : {}),
      ...(computeDefaults.outputDtype ? { outputDtype: computeDefaults.outputDtype } : {}),
    };
  }
  if (Object.keys(computePatch).length > 0) {
    patch.compute = computePatch;
  }
  if (sessionDefaults?.kvcache) {
    patch.kvcache = sessionDefaults.kvcache;
  }
  if (sessionDefaults?.decodeLoop) {
    patch.batching = {
      batchSize: sessionDefaults.decodeLoop.batchSize,
      stopCheckMode: sessionDefaults.decodeLoop.stopCheckMode,
      readbackInterval: sessionDefaults.decodeLoop.readbackInterval,
      ringTokens: sessionDefaults.decodeLoop.ringTokens,
      ringStop: sessionDefaults.decodeLoop.ringStop,
      ringStaging: sessionDefaults.decodeLoop.ringStaging,
    };
  }
  return patch;
}

function buildModelRuntimeOverrides(manifestInference) {
  const model = manifestInference?.model;
  if (!model || typeof model !== 'object') {
    return null;
  }
  return cloneJson(model);
}

export function hasExecutionV0(manifestInference) {
  return !!manifestInference?.execution && Array.isArray(manifestInference.execution.steps);
}

export function compileExecutionV0(options = {}) {
  const manifestInference = options.manifestInference ?? null;
  if (!hasExecutionV0(manifestInference)) {
    return null;
  }
  assertExecutionV0Schema(manifestInference);

  const modelId = options.modelId ?? 'model';
  const numLayers = Number.isInteger(options.numLayers) ? options.numLayers : 0;
  const runtimeInference = options.runtimeInference ?? {};
  assertExecutionRuntimeOverlay(runtimeInference);
  const policies = {
    ...DEFAULT_EXECUTION_V0_POLICIES,
    ...(manifestInference.execution?.policies ?? {}),
  };
  const normalizedRuntimeSession = normalizeRuntimeSessionForExecutionV0(
    runtimeInference.session ?? {},
    manifestInference
  );
  const sessionDefaults = mergeRuntimeValues(
    DEFAULT_EXECUTION_V0_SESSION_DEFAULTS,
    manifestInference.sessionDefaults ?? {}
  );
  const resolvedSession = mergeRuntimeValues(
    sessionDefaults,
    normalizedRuntimeSession ?? {}
  );

  const baseSteps = cloneJson(manifestInference.execution.steps ?? []);
  baseSteps.forEach(validateStepShape);
  validateUniqueStepIds(baseSteps);

  const patchedSteps = applyExecutionPatchAtomic(baseSteps, runtimeInference.executionPatch ?? null);
  patchedSteps.forEach(validateStepShape);
  validateUniqueStepIds(patchedSteps);
  const runtimePatchMeta = indexRuntimePatchMeta(runtimeInference.executionPatch ?? null);

  const manifestSessionDefaults = manifestInference.sessionDefaults ?? {};
  const sessionSourceByPath = new Map();
  for (const pathSegments of collectLeafPaths(resolvedSession)) {
    const path = pathSegments.join('.');
    const source = hasDefinedPath(runtimeInference.session ?? {}, pathSegments)
      ? 'runtime.session'
      : hasDefinedPath(manifestSessionDefaults, pathSegments)
        ? 'manifest'
        : 'derived';
    sessionSourceByPath.set(path, source);
  }

  const profileIndex = indexKernelProfiles(resolvedSession);
  const sourceTrace = createSourceTrace();
  const sessionDefaultSources = {
    mathDtype: sessionSourceByPath.get('compute.defaults.mathDtype') ?? 'derived',
    accumDtype: sessionSourceByPath.get('compute.defaults.accumDtype') ?? 'derived',
    outputDtype: sessionSourceByPath.get('compute.defaults.outputDtype') ?? 'derived',
    kvDtype: sessionSourceByPath.get('kvcache.kvDtype') ?? 'derived',
  };
  const resolvedPrefill = resolvePhaseSteps(
    'prefill',
    patchedSteps,
    resolvedSession,
    profileIndex,
    policies,
    {
      initialSlotDtypes: createInitialSlotDtypes(resolvedSession),
      sourceTrace,
      sessionDefaultSources,
      runtimePatchMeta,
    }
  );
  const decodeInitialSlotDtypes = createInitialSlotDtypes(resolvedSession);
  validatePhaseBoundaryCompatibility({
    steps: patchedSteps,
    prefillFinalSlotDtypes: resolvedPrefill.finalSlotDtypes,
    decodeInitialSlotDtypes,
    sessionDefaults: resolvedSession,
    profileIndex,
    policies,
  });
  const resolvedDecode = resolvePhaseSteps(
    'decode',
    patchedSteps,
    resolvedSession,
    profileIndex,
    policies,
    {
      initialSlotDtypes: decodeInitialSlotDtypes,
      sourceTrace,
      sessionDefaultSources,
      runtimePatchMeta,
    }
  );
  const resolvedPrefillSteps = resolvedPrefill.steps;
  const resolvedDecodeSteps = resolvedDecode.steps;
  const resolvedSteps = [
    ...resolvedPrefillSteps,
    ...resolvedDecodeSteps.filter((step) => step.phase === 'decode'),
  ];

  const kernelPath = buildInlineKernelPath(patchedSteps, resolvedSession, modelId, numLayers);
  const layerPipeline = buildLayerPipelineFromExecution(resolvedSteps);
  const sessionPatch = buildSessionRuntimePatch(resolvedSession);
  const modelOverrides = buildModelRuntimeOverrides(manifestInference);
  for (const [path, source] of sessionSourceByPath.entries()) {
    setSourceTrace(sourceTrace.session, `sessionDefaults.${path}`, source);
  }

  return {
    sessionDefaults: resolvedSession,
    policies,
    resolvedSteps: {
      prefill: resolvedPrefillSteps,
      decode: resolvedDecodeSteps,
      all: resolvedSteps,
    },
    runtimeInferencePatch: {
      ...sessionPatch,
      ...(kernelPath ? { kernelPath, kernelPathSource: 'execution-v0' } : {}),
      ...(layerPipeline ? { pipeline: layerPipeline } : {}),
      ...(modelOverrides ? { modelOverrides } : {}),
    },
    resolvedSources: sourceTrace,
  };
}

export function applyExecutionV0RuntimeConfig(options = {}) {
  const runtimeConfig = options.runtimeConfig ?? null;
  const manifest = options.manifest ?? null;
  if (!runtimeConfig || !manifest?.inference) {
    return { runtimeConfig, executionV0State: null };
  }
  if (!hasExecutionV0(manifest.inference)) {
    return { runtimeConfig, executionV0State: null };
  }

  const runtimeInference = runtimeConfig.inference ?? {};
  const runtimeExecutionOverlay = {
    ...(runtimeInference.session ? { session: runtimeInference.session } : {}),
    ...(runtimeInference.executionPatch ? { executionPatch: runtimeInference.executionPatch } : {}),
  };

  const executionV0State = compileExecutionV0({
    manifestInference: manifest.inference,
    runtimeInference: runtimeExecutionOverlay,
    modelId: options.modelId ?? manifest.modelId ?? 'model',
    numLayers: Number.isInteger(options.numLayers)
      ? options.numLayers
      : Number(manifest.architecture?.numLayers ?? 0),
  });
  if (!executionV0State) {
    return { runtimeConfig, executionV0State: null };
  }

  const runtimeInferencePatch = { ...executionV0State.runtimeInferencePatch };
  if (runtimeInferencePatch.modelOverrides) {
    runtimeInferencePatch.modelOverrides = mergeRuntimeValues(
      runtimeInferencePatch.modelOverrides,
      runtimeInference.modelOverrides ?? {}
    );
  }

  return {
    runtimeConfig: {
      ...runtimeConfig,
      inference: mergeRuntimeValues(runtimeConfig.inference, runtimeInferencePatch),
    },
    executionV0State,
  };
}
