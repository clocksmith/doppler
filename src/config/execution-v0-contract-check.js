import { isExecutionV0Digest, isExecutionV0Semver } from './schema/execution-v0.schema.js';

function normalizeDtype(value, label) {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(`[ExecutionV0Contract] ${label} must be "f16" or "f32"; got "${value}"`);
  }
  return normalized;
}

function buildKernelProfileKey(kernelRef) {
  if (!kernelRef) return null;
  return `${kernelRef.id}|${kernelRef.version}|${kernelRef.digest}`;
}

function assertExecutionV0KernelRef(kernelRef, label) {
  if (!kernelRef || typeof kernelRef !== 'object' || Array.isArray(kernelRef)) {
    throw new Error(`[ExecutionV0Contract] ${label} is required.`);
  }
  if (typeof kernelRef.id !== 'string' || kernelRef.id.trim().length === 0) {
    throw new Error(`[ExecutionV0Contract] ${label}.id is required.`);
  }
  if (!isExecutionV0Semver(kernelRef.version)) {
    throw new Error(`[ExecutionV0Contract] ${label}.version must be semver.`);
  }
  if (!isExecutionV0Digest(kernelRef.digest)) {
    throw new Error(`[ExecutionV0Contract] ${label}.digest must match sha256:<64-hex>.`);
  }
}

function requirePlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`[ExecutionV0Contract] ${label} must be an object.`);
  }
  return value;
}

function requireOwnProperty(root, key, label) {
  if (!Object.prototype.hasOwnProperty.call(root, key)) {
    throw new Error(`[ExecutionV0Contract] ${label} is required.`);
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
    throw new Error(`[ExecutionV0Contract] ${label} must be an array.`);
  }
  return value;
}

function requireDtypeProperty(root, key, label) {
  const value = requireOwnProperty(root, key, label);
  if (value == null) {
    throw new Error(`[ExecutionV0Contract] ${label} is required.`);
  }
  return normalizeDtype(value, label);
}

function validateExecutionV0SessionDefaults(sessionDefaults = {}) {
  const normalizedSessionDefaults = requirePlainObject(
    sessionDefaults,
    'manifest.inference.sessionDefaults'
  );
  const compute = requirePlainObject(
    requireOwnProperty(normalizedSessionDefaults, 'compute', 'sessionDefaults.compute'),
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

  const kvcache = requireNullableObject(
    normalizedSessionDefaults,
    'kvcache',
    'sessionDefaults.kvcache'
  );
  if (kvcache !== null) {
    requireDtypeProperty(
      kvcache,
      'kvDtype',
      'sessionDefaults.kvcache.kvDtype'
    );
  }

  requireNullableObject(
    normalizedSessionDefaults,
    'decodeLoop',
    'sessionDefaults.decodeLoop'
  );

  return normalizedSessionDefaults;
}

function createPrecisionSources(step, profile) {
  return {
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
}

export function normalizeExecutionV0Dtype(value, label) {
  return normalizeDtype(value, label);
}

export function buildExecutionV0KernelProfileKey(kernelRef) {
  return buildKernelProfileKey(kernelRef);
}

export function indexExecutionV0KernelProfiles(sessionDefaults = {}) {
  const byKey = new Map();
  const profiles = sessionDefaults?.compute?.kernelProfiles ?? [];
  for (const profile of profiles) {
    assertExecutionV0KernelRef(profile?.kernelRef, 'sessionDefaults.compute.kernelProfiles[].kernelRef');
    const key = buildKernelProfileKey(profile?.kernelRef);
    if (byKey.has(key)) {
      throw new Error(
        `[ExecutionV0Contract] duplicate kernel profile for ${profile.kernelRef.id}@${profile.kernelRef.version} ` +
        `(${profile.kernelRef.digest}). Expected exactly one pinned profile per kernelRef.`
      );
    }
    byKey.set(key, profile);
  }
  return byKey;
}

export function resolveExecutionV0KernelProfile(profileIndex, step) {
  const key = buildKernelProfileKey(step?.kernelRef);
  if (!key) return null;
  return profileIndex.get(key) ?? null;
}

export function resolveExecutionV0Precision(step, profile, sessionDefaults = {}) {
  const defaults = requirePlainObject(
    sessionDefaults?.compute?.defaults,
    'sessionDefaults.compute.defaults'
  );
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
  return {
    precision,
    sources: createPrecisionSources(step, profile),
  };
}

export function resolveExecutionV0KVIO(step, profile, sessionDefaults = {}) {
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
    requireOwnProperty(
      requireNullableObject(sessionDefaults, 'kvcache', 'sessionDefaults.kvcache') ?? {},
      'kvDtype',
      'sessionDefaults.kvcache.kvDtype'
    ),
    `${step.id}.sessionDefaults.kvcache.kvDtype`
  );
  return {
    value: { readDtype: kvDtype, writeDtype: kvDtype },
    source: 'sessionDefault',
  };
}

export function buildExecutionV0ContractArtifact(manifestInference, options = {}) {
  if (!manifestInference?.execution || !Array.isArray(manifestInference.execution.steps)) {
    return null;
  }

  const modelId = String(options.modelId ?? 'model');
  const checks = [];
  const errors = [];
  const perStep = {};
  let sessionDefaults = manifestInference.sessionDefaults ?? {};
  let profileIndex;

  try {
    sessionDefaults = validateExecutionV0SessionDefaults(sessionDefaults);
    profileIndex = indexExecutionV0KernelProfiles(sessionDefaults);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
    checks.push({ id: `${modelId}.kernelProfilePinning`, ok: false });
    checks.push({ id: `${modelId}.precisionPrecedence`, ok: false });
    checks.push({ id: `${modelId}.kvIOPrecedence`, ok: false });
    return {
      schemaVersion: 1,
      source: 'doppler',
      ok: false,
      checks,
      errors,
      stats: {
        kernelProfiles: sessionDefaults?.compute?.kernelProfiles?.length ?? 0,
        pinnedSteps: 0,
      },
      perStep,
    };
  }

  let pinningOk = true;
  let precisionOk = true;
  let kvOk = true;
  let pinnedSteps = 0;

  for (const step of manifestInference.execution.steps) {
    if (!step || typeof step !== 'object') continue;
    if (step.op === 'cast') continue;
    if (!step.kernelRef) {
      pinningOk = false;
      precisionOk = false;
      errors.push(`[ExecutionV0Contract] step "${step.id ?? 'unknown'}" requires kernelRef.`);
      continue;
    }
    const profile = resolveExecutionV0KernelProfile(profileIndex, step);
    if (!profile) {
      pinningOk = false;
      precisionOk = false;
      errors.push(
        `[ExecutionV0Contract] step "${step.id ?? 'unknown'}" kernelRef ` +
        `${step.kernelRef.id}@${step.kernelRef.version} (${step.kernelRef.digest}) is unresolved.`
      );
      continue;
    }
    pinnedSteps += 1;
    const { precision, sources } = resolveExecutionV0Precision(step, profile, sessionDefaults);
    perStep[step.id] = {
      precision,
      precisionSources: sources,
    };
    try {
      perStep[step.id].resolvedPrecision = {
        inputDtype: precision.inputDtype == null ? null : normalizeDtype(precision.inputDtype, `${step.id}.precision.inputDtype`),
        mathDtype: normalizeDtype(precision.mathDtype, `${step.id}.precision.mathDtype`),
        accumDtype: normalizeDtype(precision.accumDtype, `${step.id}.precision.accumDtype`),
        outputDtype: normalizeDtype(precision.outputDtype, `${step.id}.precision.outputDtype`),
      };
    } catch (error) {
      precisionOk = false;
      errors.push(error instanceof Error ? error.message : String(error));
    }
    if (step.op === 'attention') {
      try {
        const kvIO = resolveExecutionV0KVIO(step, profile, sessionDefaults);
        perStep[step.id].kvIO = kvIO.value;
        perStep[step.id].kvIOSource = kvIO.source;
      } catch (error) {
        kvOk = false;
        errors.push(error instanceof Error ? error.message : String(error));
      }
    }
  }

  checks.push({ id: `${modelId}.kernelProfilePinning`, ok: pinningOk });
  checks.push({ id: `${modelId}.precisionPrecedence`, ok: precisionOk });
  checks.push({ id: `${modelId}.kvIOPrecedence`, ok: kvOk });

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      kernelProfiles: sessionDefaults?.compute?.kernelProfiles?.length ?? 0,
      pinnedSteps,
    },
    perStep,
  };
}
