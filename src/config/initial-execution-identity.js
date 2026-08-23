import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const INITIAL_EXECUTION_IDENTITY_SCHEMA_ID = 'doppler.initial-execution-identity/v1';
export const INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID = 'doppler.initial-execution-identity/v2';
export const PROGRAM_LOAD_POLICY_V1_SCHEMA_ID = 'doppler.pack-program-load-policy/v1';
export const PROGRAM_LOAD_POLICY_SCHEMA_ID = 'doppler.pack-program-load-policy/v2';

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
  const isV1 = identity.schema === INITIAL_EXECUTION_IDENTITY_SCHEMA_ID;
  const isV2 = identity.schema === INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID;
  if (!isV1 && !isV2) {
    errors.push(
      `schema must be "${INITIAL_EXECUTION_IDENTITY_SCHEMA_ID}" or `
      + `"${INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID}".`
    );
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
  if (isV2) {
    if (!SHA256_PATTERN.test(identity.programLoadPolicyHash || '')) {
      errors.push('programLoadPolicyHash must be a SHA-256 digest.');
    }
    if (!isObject(identity.programLoadPolicy)) {
      errors.push('programLoadPolicy must be an object.');
    } else {
      const policySchema = identity.programLoadPolicy.schema;
      const isPolicyV1 = policySchema === PROGRAM_LOAD_POLICY_V1_SCHEMA_ID;
      const isPolicyV2 = policySchema === PROGRAM_LOAD_POLICY_SCHEMA_ID;
      if (!isPolicyV1 && !isPolicyV2) {
        errors.push(
          `programLoadPolicy.schema must be "${PROGRAM_LOAD_POLICY_V1_SCHEMA_ID}" or `
          + `"${PROGRAM_LOAD_POLICY_SCHEMA_ID}".`
        );
      }
      const runtimeConfig = identity.programLoadPolicy.runtimeConfig;
      if (!isObject(runtimeConfig) || !isObject(runtimeConfig.inference)) {
        errors.push('programLoadPolicy.runtimeConfig.inference must be an object.');
      } else {
        if (!isObject(runtimeConfig.inference.session)) {
          errors.push('programLoadPolicy.runtimeConfig.inference.session must be an object.');
        }
        if (!isObject(runtimeConfig.inference.compute)) {
          errors.push('programLoadPolicy.runtimeConfig.inference.compute must be an object.');
        }
        if (isPolicyV2) {
          const generation = runtimeConfig.inference.generation;
          if (!isObject(generation)) {
            errors.push('programLoadPolicy.runtimeConfig.inference.generation must be an object.');
          } else {
            if (typeof generation.disableMultiTokenDecode !== 'boolean') {
              errors.push(
                'programLoadPolicy.runtimeConfig.inference.generation.disableMultiTokenDecode '
                + 'must be a boolean.'
              );
            }
            const extraGenerationFields = Object.keys(generation)
              .filter((field) => field !== 'disableMultiTokenDecode');
            if (extraGenerationFields.length > 0) {
              errors.push(
                'programLoadPolicy.runtimeConfig.inference.generation may contain only '
                + 'disableMultiTokenDecode.'
              );
            }
          }
        }
        const allowedInferenceFields = isPolicyV2
          ? ['session', 'compute', 'generation']
          : ['session', 'compute'];
        const extraInferenceFields = Object.keys(runtimeConfig.inference)
          .filter((field) => !allowedInferenceFields.includes(field));
        if (extraInferenceFields.length > 0) {
          errors.push(
            isPolicyV2
              ? 'programLoadPolicy.runtimeConfig.inference may contain only session, compute, '
                + 'and generation.'
              : 'programLoadPolicy.runtimeConfig.inference may contain only session and compute.'
          );
        }
      }
      const extraRuntimeFields = isObject(runtimeConfig)
        ? Object.keys(runtimeConfig).filter((field) => field !== 'inference')
        : [];
      if (extraRuntimeFields.length > 0) {
        errors.push('programLoadPolicy.runtimeConfig may contain only inference.');
      }
    }
  }
  if (errors.length === 0) {
    if (identity.kernelClosureHash !== hashValue(identity.kernelClosure)) errors.push('kernelClosureHash mismatch.');
    if (identity.fusionSetHash !== hashValue(identity.fusionSet)) errors.push('fusionSetHash mismatch.');
    if (identity.kvLayoutHash !== hashValue(identity.kvLayout)) errors.push('kvLayoutHash mismatch.');
    if (identity.memoryPolicyHash !== hashValue(identity.memoryPolicy)) errors.push('memoryPolicyHash mismatch.');
    if (identity.runtimeEngineDigest !== hashValue(identity.runtimeEngine)) errors.push('runtimeEngineDigest mismatch.');
    if (isV2 && identity.programLoadPolicyHash !== hashValue(identity.programLoadPolicy)) {
      errors.push('programLoadPolicyHash mismatch.');
    }
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

export function createInitialExecutionIdentityV2(fields) {
  if (!isObject(fields)) throw new Error('createInitialExecutionIdentityV2 requires an object.');
  const programLoadPolicy = fields.programLoadPolicy;
  const core = {
    schema: INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID,
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
    programLoadPolicy,
    programLoadPolicyHash: hashValue(programLoadPolicy),
  };
  const identity = { ...core, digest: hashValue(core) };
  const validation = validateInitialExecutionIdentity(identity);
  if (!validation.ok) throw new Error(`Invalid initial execution identity v2: ${validation.errors.join('; ')}`);
  return identity;
}

export function observeInitialExecutionIdentity(resolved) {
  if (!isObject(resolved)) throw new Error('Loaded program did not expose a resolved runtime session.');
  if (!SHA256_PATTERN.test(resolved.id || '')) {
    throw new Error('Resolved runtime session is missing its canonical identity digest.');
  }
  const execution = resolved.manifestInference?.execution;
  if (!isObject(execution)) throw new Error('Resolved runtime session is missing manifest execution graph.');
  const resolvedSteps = resolved.execution?.resolvedSteps;
  if (!isObject(resolvedSteps) || !Array.isArray(resolvedSteps.all)) {
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
  return createInitialExecutionIdentityV2({
    executionGraphHash: hashValue(execution),
    resolvedGraphHash: hashValue({
      steps: resolvedSteps,
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
    programLoadPolicy: {
      schema: PROGRAM_LOAD_POLICY_SCHEMA_ID,
      runtimeConfig: {
        inference: {
          session: resolved.runtime.session,
          compute: resolved.runtime.compute,
          generation: {
            disableMultiTokenDecode:
              resolved.execution.primary?.defaultDisableMultiTokenDecode === true,
          },
        },
      },
    },
  });
}

function projectV2IdentityToV1(identity) {
  return createInitialExecutionIdentity({
    executionGraphHash: identity.executionGraphHash,
    resolvedGraphHash: identity.resolvedGraphHash,
    kernelClosure: identity.kernelClosure,
    dtypeLane: identity.dtypeLane,
    fusionSet: identity.fusionSet,
    kvLayout: identity.kvLayout,
    memoryPolicy: identity.memoryPolicy,
    executionPlanDigest: identity.executionPlanDigest,
    runtimeEngine: identity.runtimeEngine,
  });
}

export function resolveProgramLoadRuntimeConfig(identity) {
  const validation = validateInitialExecutionIdentity(identity);
  if (!validation.ok) {
    throw new Error(`Initial execution identity is invalid: ${validation.errors.join('; ')}`);
  }
  if (identity.schema === INITIAL_EXECUTION_IDENTITY_SCHEMA_ID) return null;
  return structuredClone(identity.programLoadPolicy.runtimeConfig);
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
  const comparableObserved = expected.schema === INITIAL_EXECUTION_IDENTITY_SCHEMA_ID
    && observed.schema === INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID
    ? projectV2IdentityToV1(observed)
    : observed;
  if (expected.digest === comparableObserved.digest) return true;
  const fields = [...new Set([
    ...Object.keys(coreIdentity(expected)),
    ...Object.keys(coreIdentity(comparableObserved)),
  ])].filter((field) => (
    !valuesEqual(expected[field], comparableObserved[field])
  ));
  throw new Error(`Loaded execution identity does not match TargetPlan: ${fields.join(', ')}.`);
}

function valuesEqual(left, right) {
  return JSON.stringify(stableSortObject(left)) === JSON.stringify(stableSortObject(right));
}
