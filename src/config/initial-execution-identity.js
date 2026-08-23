import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const INITIAL_EXECUTION_IDENTITY_SCHEMA_ID = 'doppler.initial-execution-identity/v1';

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hashValue(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function collectKernelIds(entries, result) {
  if (!Array.isArray(entries)) return;
  for (const entry of entries) {
    if (Array.isArray(entry) && typeof entry[1] === 'string') result.add(entry[1]);
    if (isObject(entry) && Array.isArray(entry.steps)) collectKernelIds(entry.steps, result);
  }
}

function buildObservedKernelClosure(execution) {
  const ids = new Set();
  for (const section of ['preLayer', 'decode', 'prefill', 'postLayer']) {
    collectKernelIds(execution?.[section], ids);
  }
  for (const moduleId of execution?.mechanismKernels ?? []) {
    if (typeof moduleId !== 'string' || !moduleId.trim()) {
      throw new Error('Observed execution contains an invalid mechanism kernel reference.');
    }
    ids.add(moduleId);
  }
  return [...ids].sort().map((moduleId) => {
    const declaration = execution?.kernels?.[moduleId];
    if (!isObject(declaration) || !SHA256_PATTERN.test(declaration.digest || '')) {
      throw new Error(`Observed execution references unresolved kernel "${moduleId}".`);
    }
    return {
      moduleId,
      file: declaration.kernel,
      entry: declaration.entry,
      digest: declaration.digest,
    };
  });
}

function coreIdentity(identity) {
  const { digest: ignoredDigest, ...core } = identity;
  void ignoredDigest;
  return core;
}

export function validateInitialExecutionIdentity(identity) {
  const errors = [];
  if (!isObject(identity)) return { ok: false, errors: ['Initial execution identity must be an object.'] };
  if (identity.schema !== INITIAL_EXECUTION_IDENTITY_SCHEMA_ID) {
    errors.push(`schema must be "${INITIAL_EXECUTION_IDENTITY_SCHEMA_ID}".`);
  }
  for (const field of [
    'executionGraphHash', 'resolvedGraphHash', 'kernelClosureHash', 'fusionSetHash',
    'kvLayoutHash', 'memoryPolicyHash', 'executionPlanDigest', 'runtimeEngineDigest', 'digest',
  ]) {
    if (!SHA256_PATTERN.test(identity[field] || '')) errors.push(`${field} must be a SHA-256 digest.`);
  }
  if (!Array.isArray(identity.kernelClosure) || identity.kernelClosure.length === 0) {
    errors.push('kernelClosure must be a non-empty array.');
  }
  if (!isObject(identity.dtypeLane)) errors.push('dtypeLane must be an object.');
  if (!Array.isArray(identity.fusionSet)) errors.push('fusionSet must be an array.');
  if (!isObject(identity.kvLayout)) errors.push('kvLayout must be an object.');
  if (!isObject(identity.memoryPolicy)) errors.push('memoryPolicy must be an object.');
  if (!isObject(identity.runtimeEngine)) errors.push('runtimeEngine must be an object.');
  if (errors.length === 0) {
    if (identity.kernelClosureHash !== hashValue(identity.kernelClosure)) errors.push('kernelClosureHash mismatch.');
    if (identity.fusionSetHash !== hashValue(identity.fusionSet)) errors.push('fusionSetHash mismatch.');
    if (identity.kvLayoutHash !== hashValue(identity.kvLayout)) errors.push('kvLayoutHash mismatch.');
    if (identity.memoryPolicyHash !== hashValue(identity.memoryPolicy)) errors.push('memoryPolicyHash mismatch.');
    if (identity.runtimeEngineDigest !== hashValue(identity.runtimeEngine)) errors.push('runtimeEngineDigest mismatch.');
    if (identity.digest !== hashValue(coreIdentity(identity))) errors.push('digest mismatch.');
  }
  return { ok: errors.length === 0, errors };
}

export function createInitialExecutionIdentity(fields) {
  if (!isObject(fields)) throw new Error('createInitialExecutionIdentity requires an object.');
  const core = {
    schema: INITIAL_EXECUTION_IDENTITY_SCHEMA_ID,
    executionGraphHash: fields.executionGraphHash,
    resolvedGraphHash: fields.resolvedGraphHash,
    kernelClosure: fields.kernelClosure,
    kernelClosureHash: hashValue(fields.kernelClosure),
    dtypeLane: fields.dtypeLane,
    fusionSet: fields.fusionSet,
    fusionSetHash: hashValue(fields.fusionSet),
    kvLayout: fields.kvLayout,
    kvLayoutHash: hashValue(fields.kvLayout),
    memoryPolicy: fields.memoryPolicy,
    memoryPolicyHash: hashValue(fields.memoryPolicy),
    executionPlanDigest: fields.executionPlanDigest,
    runtimeEngine: fields.runtimeEngine,
    runtimeEngineDigest: hashValue(fields.runtimeEngine),
  };
  const identity = { ...core, digest: hashValue(core) };
  const validation = validateInitialExecutionIdentity(identity);
  if (!validation.ok) throw new Error(`Invalid initial execution identity: ${validation.errors.join('; ')}`);
  return identity;
}

export function observeInitialExecutionIdentity(resolved) {
  if (!isObject(resolved)) throw new Error('Loaded program did not expose a resolved runtime session.');
  if (!SHA256_PATTERN.test(resolved.id || '')) {
    throw new Error('Resolved runtime session is missing its canonical identity digest.');
  }
  const execution = resolved.manifestInference?.execution;
  if (!isObject(execution)) throw new Error('Resolved runtime session is missing manifest execution graph.');
  if (!Array.isArray(resolved.execution?.resolvedSteps)) {
    throw new Error('Resolved runtime session is missing compiled execution steps.');
  }
  const kernelClosure = buildObservedKernelClosure(execution);
  if (kernelClosure.length === 0) throw new Error('Resolved runtime session has no reachable kernel closure.');
  const session = resolved.runtime?.session;
  const kvLayout = session?.kvcache;
  if (!isObject(kvLayout)) throw new Error('Resolved runtime session is missing KV layout policy.');
  const fusionSet = Array.isArray(resolved.execution.appliedTransforms)
    ? resolved.execution.appliedTransforms
    : [];
  const runtimeEngine = {
    resolvedRuntimeSessionId: resolved.id,
    resolvedRuntimeSchema: resolved.schema,
    kernelPath: resolved.kernelPath,
    capabilityPolicy: resolved.capabilityPolicy,
    laneIntegrity: resolved.laneIntegrity,
  };
  const memoryPolicy = {
    kvcache: session.kvcache,
    perLayerInputs: session.perLayerInputs ?? null,
    largeWeights: session.largeWeights ?? null,
  };
  return createInitialExecutionIdentity({
    executionGraphHash: hashValue(execution),
    resolvedGraphHash: hashValue({
      steps: resolved.execution.resolvedSteps,
      mechanismKernels: execution.mechanismKernels ?? [],
      layerPattern: resolved.manifestInference?.layerPattern ?? null,
    }),
    kernelClosure,
    dtypeLane: resolved.dtypes,
    fusionSet,
    kvLayout,
    memoryPolicy,
    executionPlanDigest: hashValue({
      primary: resolved.execution.primary,
      resolvedStepsHash: resolved.execution.resolvedStepsHash,
    }),
    runtimeEngine,
  });
}

export function assertInitialExecutionIdentity(expected, observed) {
  const expectedValidation = validateInitialExecutionIdentity(expected);
  if (!expectedValidation.ok) {
    throw new Error(`TargetPlan initial execution identity is invalid: ${expectedValidation.errors.join('; ')}`);
  }
  const observedValidation = validateInitialExecutionIdentity(observed);
  if (!observedValidation.ok) {
    throw new Error(`Observed initial execution identity is invalid: ${observedValidation.errors.join('; ')}`);
  }
  if (expected.digest === observed.digest) return true;
  const fields = Object.keys(coreIdentity(expected)).filter((field) => (
    !valuesEqual(expected[field], observed[field])
  ));
  throw new Error(`Loaded execution identity does not match TargetPlan: ${fields.join(', ')}.`);
}

function valuesEqual(left, right) {
  return JSON.stringify(stableSortObject(left)) === JSON.stringify(stableSortObject(right));
}
