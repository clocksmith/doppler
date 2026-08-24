import { cloneJsonValue } from '../../formats/clone-json.js';
import { computeCanonicalSha256, canonicalizeJson } from '../../formats/canonical-hash.js';
import { isPlainObject } from '../../formats/plain-object.js';
import { runBrowserCommand } from '../browser-command-runner.js';
import {
  finalizeRuntimeOptimizationReceipt,
  validateRuntimeOptimizationCampaign,
} from '../runtime-optimization-campaign.js';

export const RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA = 'doppler.runtime-optimization-contract/v1';

export const RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA = 'doppler.runtime-optimization-candidate/v1';

export const RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA =
  'doppler.runtime-optimization-candidate-registry/v1';

export const WORKLOADS = new Set(['inference', 'embedding', 'rerank']);

export const DIRECTIONS = new Set(['maximize', 'minimize']);

export const CANDIDATE_KINDS = new Set([
  'runtime-profile',
  'registered-kernel-variant',
  'registered-execution-graph-patch',
]);

export const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

export const SAFE_MUTATION_PREFIXES = Object.freeze([
  '/loading/shardCache',
  '/loading/memoryManagement',
  '/loading/prefetch',
  '/loading/expertCache',
  '/inference/batching',
  '/inference/generation',
  '/inference/session',
  '/shared/bufferPool',
  '/shared/gpuCache',
  '/shared/kernelWarmup',
  '/shared/memory',
]);

export const FORBIDDEN_MUTATION_PREFIXES = Object.freeze([
  '/inference/executionPatch',
  '/inference/kernelPath',
  '/inference/kernelPathPolicy',
  '/shared/benchmark',
  '/shared/debug',
  '/shared/harness',
  '/shared/kernelRegistry',
  '/shared/platform',
  '/shared/tooling',
]);

export const SAFE_COMPARISON_PATHS = new Set([
  'result.output',
  'result.metrics.referenceTranscript.tokens.generatedTokenIdsHash',
  'result.metrics.referenceTranscript.output.textHash',
]);

export const SAFE_METRIC_PATHS = new Set([
  'result.metrics.decodeTokensPerSec',
  'result.metrics.embeddingMs',
  'result.metrics.rerankMs',
  'result.timing.decodeTokensPerSec',
  'result.timing.totalRunMs',
]);

export function assertObject(value, label) {
  if (!isPlainObject(value)) {
    throw new Error(`runtime optimization: ${label} must be an object.`);
  }
  return value;
}

export function assertExactKeys(value, keys, label) {
  const allowed = new Set(keys);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`runtime optimization: ${label}.${key} is not supported.`);
    }
  }
}

export function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`runtime optimization: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

export function assertDigestOrNull(value, label) {
  if (value === null) return null;
  if (typeof value !== 'string' || !DIGEST_PATTERN.test(value)) {
    throw new Error(`runtime optimization: ${label} must be sha256:<64 lowercase hex> or null.`);
  }
  return value;
}

export function assertIntegerRange(value, label, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(
      `runtime optimization: ${label} must be an integer in [${minimum}, ${maximum}].`
    );
  }
  return value;
}

export function assertFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`runtime optimization: ${label} must be a finite number.`);
  }
  return value;
}

export function assertJsonValue(value, label) {
  try {
    canonicalizeJson(value);
  } catch (error) {
    throw new Error(`runtime optimization: ${label} must be canonical JSON: ${error.message}`);
  }
  return value;
}

export function normalizeCandidateKind(value) {
  const normalized = value === 'runtime_profile' ? 'runtime-profile' : value;
  if (!CANDIDATE_KINDS.has(normalized)) {
    throw new Error(
      'runtime optimization: contract.kind must be runtime-profile, ' +
      'registered-kernel-variant, or registered-execution-graph-patch.'
    );
  }
  return normalized;
}

export function validateRegisteredReference(reference, label) {
  assertObject(reference, label);
  assertExactKeys(reference, ['registryId', 'digest'], label);
  assertString(reference.registryId, `${label}.registryId`);
  assertDigestOrNull(reference.digest, `${label}.digest`);
  if (reference.digest === null) {
    throw new Error(`runtime optimization: ${label}.digest must not be null.`);
  }
  return reference;
}

export function pointerMatchesPrefix(path, prefix) {
  return path === prefix || path.startsWith(`${prefix}/`);
}

export function assertSafeMutationPath(path, label) {
  const normalized = assertString(path, label);
  if (!normalized.startsWith('/') || normalized.endsWith('/')) {
    throw new Error(`runtime optimization: ${label} must be a canonical JSON pointer.`);
  }
  if (FORBIDDEN_MUTATION_PREFIXES.some((prefix) => pointerMatchesPrefix(normalized, prefix))) {
    throw new Error(`runtime optimization: ${label} targets evaluator or manifest-owned policy: ${normalized}.`);
  }
  if (!SAFE_MUTATION_PREFIXES.some((prefix) => pointerMatchesPrefix(normalized, prefix))) {
    throw new Error(`runtime optimization: ${label} is outside the runtime-owned allowlist: ${normalized}.`);
  }
  decodeJsonPointer(normalized);
  return normalized;
}

export function decodeJsonPointer(path) {
  if (path === '') return [];
  if (!path.startsWith('/')) {
    throw new Error(`runtime optimization: invalid JSON pointer "${path}".`);
  }
  return path.slice(1).split('/').map((segment) => {
    if (/~(?:[^01]|$)/.test(segment)) {
      throw new Error(`runtime optimization: invalid JSON pointer escape in "${path}".`);
    }
    const decoded = segment.replace(/~1/g, '/').replace(/~0/g, '~');
    if (!decoded || /^\d+$/.test(decoded)) {
      throw new Error(`runtime optimization: JSON pointer segments must name object fields: "${path}".`);
    }
    return decoded;
  });
}

export function validateWorkloadRequest(value, workload) {
  const request = assertObject(value, 'workload.request');
  assertExactKeys(request, ['inferenceInput', 'cacheMode', 'loadMode'], 'workload.request');
  if (request.inferenceInput !== undefined && request.inferenceInput !== null) {
    if (workload !== 'inference') {
      throw new Error('runtime optimization: workload.request.inferenceInput requires workload.type="inference".');
    }
    assertJsonValue(assertObject(request.inferenceInput, 'workload.request.inferenceInput'), 'workload.request.inferenceInput');
  }
  if (![undefined, null, 'cold', 'warm'].includes(request.cacheMode)) {
    throw new Error('runtime optimization: workload.request.cacheMode must be "cold", "warm", or null.');
  }
  if (![undefined, null, 'opfs', 'http', 'memory', 'file'].includes(request.loadMode)) {
    throw new Error('runtime optimization: workload.request.loadMode is unsupported.');
  }
  return request;
}

export function validateDimension(dimension, index, seenPaths) {
  assertObject(dimension, `mutationPolicy.dimensions[${index}]`);
  assertExactKeys(dimension, ['path', 'values'], `mutationPolicy.dimensions[${index}]`);
  const path = assertSafeMutationPath(
    dimension.path,
    `mutationPolicy.dimensions[${index}].path`
  );
  if (seenPaths.has(path)) {
    throw new Error(`runtime optimization: duplicate mutation dimension "${path}".`);
  }
  seenPaths.add(path);
  if (!Array.isArray(dimension.values) || dimension.values.length === 0) {
    throw new Error(`runtime optimization: mutationPolicy.dimensions[${index}].values must be non-empty.`);
  }
  const valueKeys = new Set();
  for (let valueIndex = 0; valueIndex < dimension.values.length; valueIndex += 1) {
    const value = assertJsonValue(
      dimension.values[valueIndex],
      `mutationPolicy.dimensions[${index}].values[${valueIndex}]`
    );
    const key = canonicalizeJson(value);
    if (valueKeys.has(key)) {
      throw new Error(`runtime optimization: mutation dimension "${path}" contains duplicate values.`);
    }
    valueKeys.add(key);
  }
}

export function validateRuntimeOptimizationContract(input) {
  const contract = cloneJsonValue(assertObject(input, 'contract'));
  assertExactKeys(contract, [
    'schema', 'contractId', 'kind', 'campaign', 'model', 'baseline', 'workload',
    'mutationPolicy', 'verification', 'measurement', 'neighboringWorkloads',
  ], 'contract');
  if (contract.schema !== RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA) {
    throw new Error(`runtime optimization: contract.schema must be "${RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA}".`);
  }
  assertString(contract.contractId, 'contract.contractId');
  const candidateKind = normalizeCandidateKind(contract.kind);

  assertObject(contract.model, 'contract.model');
  assertExactKeys(contract.model, ['modelId', 'modelUrl', 'expectedExecutionContractHash'], 'contract.model');
  assertString(contract.model.modelId, 'contract.model.modelId');
  if (contract.model.modelUrl !== null) {
    assertString(contract.model.modelUrl, 'contract.model.modelUrl');
  }
  assertDigestOrNull(
    contract.model.expectedExecutionContractHash,
    'contract.model.expectedExecutionContractHash'
  );

  assertObject(contract.baseline, 'contract.baseline');
  assertExactKeys(contract.baseline, ['runtimeProfile', 'runtimeConfig'], 'contract.baseline');
  if (contract.baseline.runtimeProfile !== null) {
    throw new Error(
      'runtime optimization: contract.baseline.runtimeProfile must be null in v1; provide an explicit runtimeConfig overlay.'
    );
  }
  assertJsonValue(
    assertObject(contract.baseline.runtimeConfig, 'contract.baseline.runtimeConfig'),
    'contract.baseline.runtimeConfig'
  );

  assertObject(contract.workload, 'contract.workload');
  assertExactKeys(contract.workload, ['type', 'request'], 'contract.workload');
  if (!WORKLOADS.has(contract.workload.type)) {
    throw new Error(`runtime optimization: unsupported workload "${contract.workload.type}".`);
  }
  validateWorkloadRequest(contract.workload.request, contract.workload.type);

  assertObject(contract.mutationPolicy, 'contract.mutationPolicy');
  if (candidateKind === 'runtime-profile') {
    assertExactKeys(contract.mutationPolicy, ['dimensions', 'maxCandidates'], 'contract.mutationPolicy');
    if (!Array.isArray(contract.mutationPolicy.dimensions) || contract.mutationPolicy.dimensions.length === 0) {
      throw new Error('runtime optimization: mutationPolicy.dimensions must be non-empty.');
    }
    const seenPaths = new Set();
    contract.mutationPolicy.dimensions.forEach((dimension, index) => (
      validateDimension(dimension, index, seenPaths)
    ));
    const maxCandidates = assertIntegerRange(
      contract.mutationPolicy.maxCandidates,
      'contract.mutationPolicy.maxCandidates',
      1,
      256
    );
    const candidateCount = contract.mutationPolicy.dimensions.reduce(
      (count, dimension) => count * dimension.values.length,
      1
    );
    if (candidateCount > maxCandidates) {
      throw new Error(
        `runtime optimization: search grid has ${candidateCount} candidates, exceeding maxCandidates=${maxCandidates}.`
      );
    }
  } else {
    assertExactKeys(contract.mutationPolicy, ['references', 'maxCandidates'], 'contract.mutationPolicy');
    if (!Array.isArray(contract.mutationPolicy.references)
      || contract.mutationPolicy.references.length === 0) {
      throw new Error('runtime optimization: mutationPolicy.references must be non-empty.');
    }
    contract.mutationPolicy.references.forEach((reference, index) => (
      validateRegisteredReference(reference, `mutationPolicy.references[${index}]`)
    ));
    const maxCandidates = assertIntegerRange(
      contract.mutationPolicy.maxCandidates,
      'contract.mutationPolicy.maxCandidates',
      1,
      256
    );
    if (contract.mutationPolicy.references.length > maxCandidates) {
      throw new Error('runtime optimization: registered references exceed maxCandidates.');
    }
  }

  assertObject(contract.verification, 'contract.verification');
  assertExactKeys(contract.verification, ['comparisons'], 'contract.verification');
  if (!Array.isArray(contract.verification.comparisons) || contract.verification.comparisons.length === 0) {
    throw new Error('runtime optimization: verification.comparisons must be non-empty.');
  }
  const comparisonPaths = new Set();
  contract.verification.comparisons.forEach((comparison, index) => {
    assertObject(comparison, `verification.comparisons[${index}]`);
    assertExactKeys(comparison, ['path', 'mode'], `verification.comparisons[${index}]`);
    if (!SAFE_COMPARISON_PATHS.has(comparison.path)) {
      throw new Error(`runtime optimization: unsupported comparison path "${comparison.path}".`);
    }
    if (comparison.mode !== 'canonical_exact') {
      throw new Error('runtime optimization: comparison mode must be "canonical_exact".');
    }
    if (comparisonPaths.has(comparison.path)) {
      throw new Error(`runtime optimization: duplicate comparison path "${comparison.path}".`);
    }
    comparisonPaths.add(comparison.path);
  });

  assertObject(contract.measurement, 'contract.measurement');
  assertExactKeys(contract.measurement, [
    'metricPath', 'direction', 'pairCount', 'minValidPairs',
    'minImprovementPercent', 'requirePositiveConfidence', 'maxRelativeStdDevPercent',
    'orderPolicy', 'sequentialDecision',
  ], 'contract.measurement');
  if (!SAFE_METRIC_PATHS.has(contract.measurement.metricPath)) {
    throw new Error(`runtime optimization: unsupported metric path "${contract.measurement.metricPath}".`);
  }
  if (!DIRECTIONS.has(contract.measurement.direction)) {
    throw new Error('runtime optimization: measurement.direction must be "maximize" or "minimize".');
  }
  const pairCount = assertIntegerRange(contract.measurement.pairCount, 'measurement.pairCount', 1, 64);
  const minValidPairs = assertIntegerRange(
    contract.measurement.minValidPairs,
    'measurement.minValidPairs',
    1,
    64
  );
  if (minValidPairs > pairCount) {
    throw new Error('runtime optimization: measurement.minValidPairs must not exceed pairCount.');
  }
  assertFiniteNumber(contract.measurement.minImprovementPercent, 'measurement.minImprovementPercent');
  if (typeof contract.measurement.requirePositiveConfidence !== 'boolean') {
    throw new Error('runtime optimization: measurement.requirePositiveConfidence must be boolean.');
  }
  if (contract.measurement.maxRelativeStdDevPercent !== null) {
    const maxStdDev = assertFiniteNumber(
      contract.measurement.maxRelativeStdDevPercent,
      'measurement.maxRelativeStdDevPercent'
    );
    if (maxStdDev < 0) {
      throw new Error('runtime optimization: measurement.maxRelativeStdDevPercent must be non-negative or null.');
    }
  }
  if (contract.measurement.orderPolicy !== undefined) {
    const order = assertObject(contract.measurement.orderPolicy, 'measurement.orderPolicy');
    assertExactKeys(order, ['kind', 'seed', 'blockSize'], 'measurement.orderPolicy');
    if (order.kind !== 'randomized-blocks') {
      throw new Error('runtime optimization: measurement.orderPolicy.kind must be "randomized-blocks".');
    }
    assertIntegerRange(order.seed, 'measurement.orderPolicy.seed', 0, 0xffffffff);
    if (order.blockSize !== 2) {
      throw new Error('runtime optimization: measurement.orderPolicy.blockSize must be 2.');
    }
  }
  if (contract.measurement.sequentialDecision !== undefined) {
    const sequential = assertObject(
      contract.measurement.sequentialDecision,
      'measurement.sequentialDecision'
    );
    assertExactKeys(
      sequential,
      ['kind', 'lookEveryPairs', 'minimumPairs', 'maximumLooks', 'alpha'],
      'measurement.sequentialDecision'
    );
    if (sequential.kind !== 'bonferroni-fixed-looks') {
      throw new Error(
        'runtime optimization: measurement.sequentialDecision.kind must be "bonferroni-fixed-looks".'
      );
    }
    const lookEveryPairs = assertIntegerRange(
      sequential.lookEveryPairs,
      'measurement.sequentialDecision.lookEveryPairs',
      1,
      pairCount
    );
    assertIntegerRange(
      sequential.minimumPairs,
      'measurement.sequentialDecision.minimumPairs',
      minValidPairs,
      pairCount
    );
    const expectedLooks = Math.ceil(pairCount / lookEveryPairs);
    if (sequential.maximumLooks !== expectedLooks) {
      throw new Error(
        `runtime optimization: measurement.sequentialDecision.maximumLooks must be ${expectedLooks}.`
      );
    }
    if (!Number.isFinite(sequential.alpha) || sequential.alpha <= 0 || sequential.alpha >= 0.5) {
      throw new Error('runtime optimization: measurement.sequentialDecision.alpha must be in (0, 0.5).');
    }
  }
  if (contract.neighboringWorkloads !== undefined) {
    if (!Array.isArray(contract.neighboringWorkloads)) {
      throw new Error('runtime optimization: neighboringWorkloads must be an array.');
    }
    const ids = new Set();
    contract.neighboringWorkloads.forEach((guard, index) => {
      const label = `neighboringWorkloads[${index}]`;
      assertObject(guard, label);
      assertExactKeys(
        guard,
        ['guardId', 'workload', 'metricPath', 'direction', 'maxRegressionPercent', 'pairCount'],
        label
      );
      const guardId = assertString(guard.guardId, `${label}.guardId`);
      if (ids.has(guardId)) throw new Error(`runtime optimization: duplicate guardId "${guardId}".`);
      ids.add(guardId);
      assertObject(guard.workload, `${label}.workload`);
      assertExactKeys(guard.workload, ['type', 'request'], `${label}.workload`);
      if (!WORKLOADS.has(guard.workload.type)) {
        throw new Error(`runtime optimization: unsupported neighboring workload "${guard.workload.type}".`);
      }
      validateWorkloadRequest(guard.workload.request, guard.workload.type);
      if (!SAFE_METRIC_PATHS.has(guard.metricPath)) {
        throw new Error(`runtime optimization: unsupported neighboring metric "${guard.metricPath}".`);
      }
      if (!DIRECTIONS.has(guard.direction)) {
        throw new Error(`runtime optimization: ${label}.direction is invalid.`);
      }
      const maxRegression = assertFiniteNumber(
        guard.maxRegressionPercent,
        `${label}.maxRegressionPercent`
      );
      if (maxRegression < 0) {
        throw new Error(`runtime optimization: ${label}.maxRegressionPercent must be non-negative.`);
      }
      assertIntegerRange(guard.pairCount, `${label}.pairCount`, 1, 16);
    });
  }
  validateRuntimeOptimizationCampaign(contract.campaign, contract);
  return contract;
}

export function buildParentHash(contract) {
  return computeCanonicalSha256({
    runtimeProfile: contract.baseline.runtimeProfile,
    runtimeConfig: contract.baseline.runtimeConfig,
  });
}

export function findDimension(contract, path) {
  return contract.mutationPolicy.dimensions.find((dimension) => dimension.path === path) ?? null;
}

export function validateRuntimeOptimizationCandidate(candidateInput, contractInput) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = cloneJsonValue(assertObject(candidateInput, 'candidate'));
  assertExactKeys(candidate, [
    'schema', 'candidateId', 'contractHash', 'parentHash', 'kind', 'patch',
    'registeredReference',
  ], 'candidate');
  if (candidate.schema !== RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA) {
    throw new Error(`runtime optimization: candidate.schema must be "${RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA}".`);
  }
  assertString(candidate.candidateId, 'candidate.candidateId');
  const expectedContractHash = computeCanonicalSha256(contract);
  if (candidate.contractHash !== expectedContractHash) {
    throw new Error('runtime optimization: candidate.contractHash does not match the frozen contract.');
  }
  const expectedParentHash = buildParentHash(contract);
  if (candidate.parentHash !== expectedParentHash) {
    throw new Error('runtime optimization: candidate.parentHash does not match the baseline runtime inputs.');
  }
  const contractKind = normalizeCandidateKind(contract.kind);
  const candidateKind = candidate.kind === undefined
    ? 'runtime-profile'
    : normalizeCandidateKind(candidate.kind);
  if (candidateKind !== contractKind) {
    throw new Error('runtime optimization: candidate.kind does not match contract.kind.');
  }
  if (contractKind !== 'runtime-profile') {
    if (!Array.isArray(candidate.patch) || candidate.patch.length !== 0) {
      throw new Error('runtime optimization: registered candidates must not contain inline patches.');
    }
    validateRegisteredReference(candidate.registeredReference, 'candidate.registeredReference');
    const referenceMatch = contract.mutationPolicy.references.some((reference) => (
      reference.registryId === candidate.registeredReference.registryId
      && reference.digest === candidate.registeredReference.digest
    ));
    if (!referenceMatch) {
      throw new Error('runtime optimization: candidate reference is outside the frozen registry domain.');
    }
    candidate.kind = candidateKind;
    return candidate;
  }
  if (!Array.isArray(candidate.patch) || candidate.patch.length !== contract.mutationPolicy.dimensions.length) {
    throw new Error('runtime optimization: candidate.patch must set every frozen mutation dimension exactly once.');
  }
  const paths = new Set();
  candidate.patch.forEach((operation, index) => {
    assertObject(operation, `candidate.patch[${index}]`);
    assertExactKeys(operation, ['op', 'path', 'value'], `candidate.patch[${index}]`);
    if (operation.op !== 'set') {
      throw new Error('runtime optimization: candidate patch operations must use op="set".');
    }
    const path = assertSafeMutationPath(operation.path, `candidate.patch[${index}].path`);
    if (paths.has(path)) {
      throw new Error(`runtime optimization: candidate.patch contains duplicate path "${path}".`);
    }
    paths.add(path);
    const dimension = findDimension(contract, path);
    if (!dimension) {
      throw new Error(`runtime optimization: candidate path "${path}" is not in the frozen grid.`);
    }
    const candidateValue = canonicalizeJson(assertJsonValue(operation.value, `candidate.patch[${index}].value`));
    if (!dimension.values.some((value) => canonicalizeJson(value) === candidateValue)) {
      throw new Error(`runtime optimization: candidate value for "${path}" is outside the frozen domain.`);
    }
  });
  candidate.kind = candidateKind;
  return candidate;
}

export function setPointerValue(target, path, value) {
  const segments = decodeJsonPointer(path);
  let cursor = target;
  for (let index = 0; index < segments.length - 1; index += 1) {
    const segment = segments[index];
    const current = cursor[segment];
    if (current === undefined) {
      cursor[segment] = {};
    } else if (!isPlainObject(current)) {
      throw new Error(`runtime optimization: candidate path "${path}" crosses a non-object field.`);
    }
    cursor = cursor[segment];
  }
  cursor[segments.at(-1)] = cloneJsonValue(value);
}

export function validateRuntimeOptimizationCandidateRegistry(registryInput) {
  const registry = cloneJsonValue(assertObject(registryInput, 'candidate registry'));
  assertExactKeys(registry, ['$schema', 'schema', 'entries'], 'candidate registry');
  if (
    registry.$schema !== undefined
    && registry.$schema !== RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA
  ) {
    throw new Error(
      `runtime optimization: candidate registry $schema must be ` +
      `"${RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA}".`
    );
  }
  if (registry.schema !== RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA) {
    throw new Error(
      `runtime optimization: candidate registry schema must be ` +
      `"${RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA}".`
    );
  }
  assertObject(registry.entries, 'candidate registry.entries');
  for (const [registryId, entry] of Object.entries(registry.entries)) {
    const label = `candidate registry.entries.${registryId}`;
    assertObject(entry, label);
    assertExactKeys(
      entry,
      ['registryId', 'kind', 'digest', 'runtimeInputs', 'evidenceScope', 'checkedInPath'],
      label
    );
    if (entry.registryId !== registryId) {
      throw new Error(`runtime optimization: ${label}.registryId must match its registry key.`);
    }
    const kind = normalizeCandidateKind(entry.kind);
    if (kind === 'runtime-profile') {
      throw new Error(`runtime optimization: ${label}.kind must be a registered candidate kind.`);
    }
    assertObject(entry.runtimeInputs, `${label}.runtimeInputs`);
    assertExactKeys(entry.runtimeInputs, ['runtimeProfile', 'runtimeConfig'], `${label}.runtimeInputs`);
    if (entry.runtimeInputs.runtimeProfile !== null) {
      throw new Error(`runtime optimization: ${label}.runtimeInputs.runtimeProfile must be null.`);
    }
    assertJsonValue(
      assertObject(entry.runtimeInputs.runtimeConfig, `${label}.runtimeInputs.runtimeConfig`),
      `${label}.runtimeInputs.runtimeConfig`
    );
    assertJsonValue(
      assertObject(entry.evidenceScope, `${label}.evidenceScope`),
      `${label}.evidenceScope`
    );
    const checkedInPath = assertString(entry.checkedInPath, `${label}.checkedInPath`);
    if (
      !checkedInPath.startsWith('src/config/runtime/optimization-candidates/')
      || checkedInPath.includes('..')
    ) {
      throw new Error(
        `runtime optimization: ${label}.checkedInPath must be under ` +
        'src/config/runtime/optimization-candidates/.'
      );
    }
    assertDigestOrNull(entry.digest, `${label}.digest`);
    const expectedDigest = computeCanonicalSha256({
      registryId,
      kind,
      runtimeInputs: entry.runtimeInputs,
      evidenceScope: entry.evidenceScope,
      checkedInPath,
    });
    if (entry.digest !== expectedDigest) {
      throw new Error(`runtime optimization: ${label}.digest does not match its frozen payload.`);
    }
    entry.kind = kind;
  }
  return registry;
}

export function materializeRegisteredCandidate(contract, candidate, registryInput) {
  if (!registryInput) {
    throw new Error('runtime optimization: registered candidates require candidateRegistry.');
  }
  const registry = validateRuntimeOptimizationCandidateRegistry(registryInput);
  const entry = registry.entries[candidate.registeredReference.registryId];
  if (!entry || entry.digest !== candidate.registeredReference.digest) {
    throw new Error('runtime optimization: registered candidate is missing or has a digest mismatch.');
  }
  if (entry.kind !== normalizeCandidateKind(contract.kind)) {
    throw new Error('runtime optimization: registered candidate kind does not match the contract.');
  }
  return cloneJsonValue(entry.runtimeInputs);
}

export function materializeRuntimeOptimizationCandidate(
  contractInput,
  candidateInput,
  options = {}
) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = validateRuntimeOptimizationCandidate(candidateInput, contract);
  if (normalizeCandidateKind(contract.kind) !== 'runtime-profile') {
    return materializeRegisteredCandidate(contract, candidate, options.candidateRegistry);
  }
  const runtimeConfig = cloneJsonValue(contract.baseline.runtimeConfig);
  for (const operation of candidate.patch) {
    setPointerValue(runtimeConfig, operation.path, operation.value);
  }
  return {
    runtimeProfile: null,
    runtimeConfig,
  };
}
