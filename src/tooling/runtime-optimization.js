import { cloneJsonValue } from '../utils/clone-json.js';
import { computeCanonicalSha256, canonicalizeJson } from '../utils/canonical-hash.js';
import { isPlainObject } from '../utils/plain-object.js';
import { runBrowserCommand } from './browser-command-runner.js';

export const RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA = 'doppler.runtime-optimization-contract/v1';
export const RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA = 'doppler.runtime-optimization-candidate/v1';
export const RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA = 'doppler.runtime-optimization-receipt/v1';

const WORKLOADS = new Set(['inference', 'embedding', 'rerank']);
const DIRECTIONS = new Set(['maximize', 'minimize']);
const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;
const SAFE_MUTATION_PREFIXES = Object.freeze([
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
const FORBIDDEN_MUTATION_PREFIXES = Object.freeze([
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
const SAFE_COMPARISON_PATHS = new Set([
  'result.output',
  'result.metrics.referenceTranscript.tokens.generatedTokenIdsHash',
  'result.metrics.referenceTranscript.output.textHash',
]);
const SAFE_METRIC_PATHS = new Set([
  'result.metrics.decodeTokensPerSec',
  'result.metrics.embeddingMs',
  'result.metrics.rerankMs',
  'result.timing.decodeTokensPerSec',
  'result.timing.totalRunMs',
]);
const STUDENT_T_95 = Object.freeze({
  1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
  21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
  26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
});

function assertObject(value, label) {
  if (!isPlainObject(value)) {
    throw new Error(`runtime optimization: ${label} must be an object.`);
  }
  return value;
}

function assertExactKeys(value, keys, label) {
  const allowed = new Set(keys);
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      throw new Error(`runtime optimization: ${label}.${key} is not supported.`);
    }
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`runtime optimization: ${label} must be a non-empty string.`);
  }
  return value.trim();
}

function assertDigestOrNull(value, label) {
  if (value === null) return null;
  if (typeof value !== 'string' || !DIGEST_PATTERN.test(value)) {
    throw new Error(`runtime optimization: ${label} must be sha256:<64 lowercase hex> or null.`);
  }
  return value;
}

function assertIntegerRange(value, label, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(
      `runtime optimization: ${label} must be an integer in [${minimum}, ${maximum}].`
    );
  }
  return value;
}

function assertFiniteNumber(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`runtime optimization: ${label} must be a finite number.`);
  }
  return value;
}

function assertJsonValue(value, label) {
  try {
    canonicalizeJson(value);
  } catch (error) {
    throw new Error(`runtime optimization: ${label} must be canonical JSON: ${error.message}`);
  }
  return value;
}

function pointerMatchesPrefix(path, prefix) {
  return path === prefix || path.startsWith(`${prefix}/`);
}

function assertSafeMutationPath(path, label) {
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

function decodeJsonPointer(path) {
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

function validateWorkloadRequest(value, workload) {
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

function validateDimension(dimension, index, seenPaths) {
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
    'schema', 'contractId', 'kind', 'model', 'baseline', 'workload',
    'mutationPolicy', 'verification', 'measurement',
  ], 'contract');
  if (contract.schema !== RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA) {
    throw new Error(`runtime optimization: contract.schema must be "${RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA}".`);
  }
  assertString(contract.contractId, 'contract.contractId');
  if (contract.kind !== 'runtime_profile') {
    throw new Error('runtime optimization: contract.kind must be "runtime_profile".');
  }

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
  return contract;
}

export function hashRuntimeOptimizationContract(input) {
  return computeCanonicalSha256(validateRuntimeOptimizationContract(input));
}

function buildParentHash(contract) {
  return computeCanonicalSha256({
    runtimeProfile: contract.baseline.runtimeProfile,
    runtimeConfig: contract.baseline.runtimeConfig,
  });
}

function buildCandidate(contract, patch) {
  const contractHash = computeCanonicalSha256(contract);
  const parentHash = buildParentHash(contract);
  const identity = computeCanonicalSha256({
    schema: RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA,
    contractHash,
    parentHash,
    patch,
  });
  return {
    schema: RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA,
    candidateId: `candidate-${identity.slice('sha256:'.length, 'sha256:'.length + 12)}`,
    contractHash,
    parentHash,
    patch,
  };
}

export function enumerateRuntimeOptimizationCandidates(input) {
  const contract = validateRuntimeOptimizationContract(input);
  let patches = [[]];
  for (const dimension of contract.mutationPolicy.dimensions) {
    const next = [];
    for (const patch of patches) {
      for (const value of dimension.values) {
        next.push([
          ...patch,
          { op: 'set', path: dimension.path, value: cloneJsonValue(value) },
        ]);
      }
    }
    patches = next;
  }
  return patches.map((patch) => buildCandidate(contract, patch));
}

function findDimension(contract, path) {
  return contract.mutationPolicy.dimensions.find((dimension) => dimension.path === path) ?? null;
}

export function validateRuntimeOptimizationCandidate(candidateInput, contractInput) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = cloneJsonValue(assertObject(candidateInput, 'candidate'));
  assertExactKeys(candidate, ['schema', 'candidateId', 'contractHash', 'parentHash', 'patch'], 'candidate');
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
  return candidate;
}

function setPointerValue(target, path, value) {
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

export function materializeRuntimeOptimizationCandidate(contractInput, candidateInput) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = validateRuntimeOptimizationCandidate(candidateInput, contract);
  const runtimeConfig = cloneJsonValue(contract.baseline.runtimeConfig);
  for (const operation of candidate.patch) {
    setPointerValue(runtimeConfig, operation.path, operation.value);
  }
  return {
    runtimeProfile: null,
    runtimeConfig,
  };
}

function valueAtPath(value, path) {
  const segments = String(path).split('.');
  let cursor = value;
  for (const segment of segments) {
    if (cursor == null || !Object.prototype.hasOwnProperty.call(cursor, segment)) {
      return undefined;
    }
    cursor = cursor[segment];
  }
  return cursor;
}

function summarizeError(error) {
  return {
    name: typeof error?.name === 'string' ? error.name : 'Error',
    message: typeof error?.message === 'string' ? error.message : String(error),
    code: typeof error?.code === 'string' ? error.code : null,
    retryable: typeof error?.retryable === 'boolean' ? error.retryable : null,
  };
}

function assertRunEnvelope(envelope, label, modelId) {
  if (!isPlainObject(envelope) || envelope.ok !== true || !isPlainObject(envelope.result)) {
    throw new Error(`${label} did not return a Doppler success envelope.`);
  }
  if (envelope.result.modelId !== modelId) {
    throw new Error(`${label} modelId mismatch: expected "${modelId}", got "${envelope.result.modelId}".`);
  }
  if (!Number.isInteger(envelope.result.passed) || envelope.result.passed < 1) {
    throw new Error(`${label} did not report a passing suite result.`);
  }
  if (!Number.isInteger(envelope.result.failed) || envelope.result.failed !== 0) {
    throw new Error(`${label} reported one or more failed suite results.`);
  }
  return envelope;
}

function summarizeRun(envelope, metricPath = null) {
  const executionContract = envelope.result?.metrics?.executionContractArtifact ?? null;
  const metric = metricPath ? valueAtPath(envelope, metricPath) : null;
  return {
    envelopeHash: computeCanonicalSha256(envelope),
    modelId: envelope.result?.modelId ?? null,
    suite: envelope.result?.suite ?? null,
    passed: envelope.result?.passed ?? null,
    failed: envelope.result?.failed ?? null,
    outputHash: computeCanonicalSha256(envelope.result?.output ?? null),
    executionContractHash: executionContract == null
      ? null
      : computeCanonicalSha256(executionContract),
    metric: metric == null ? null : metric,
    deviceInfo: cloneJsonValue(envelope.result?.deviceInfo ?? null),
  };
}

function buildCommandRequest(contract, runtimeInputs, command) {
  const request = contract.workload.request;
  return {
    command,
    workload: contract.workload.type,
    modelId: contract.model.modelId,
    ...(contract.model.modelUrl === null ? {} : { modelUrl: contract.model.modelUrl }),
    ...(request.inferenceInput == null ? {} : { inferenceInput: cloneJsonValue(request.inferenceInput) }),
    ...(request.cacheMode == null ? {} : { cacheMode: request.cacheMode }),
    ...(request.loadMode == null ? {} : { loadMode: request.loadMode }),
    runtimeProfile: null,
    runtimeConfig: { runtime: cloneJsonValue(runtimeInputs.runtimeConfig) },
    captureOutput: true,
    keepPipeline: false,
  };
}

function compareVerificationRuns(contract, baselineEnvelope, candidateEnvelope) {
  const comparisons = contract.verification.comparisons.map((comparison) => {
    const baselineValue = valueAtPath(baselineEnvelope, comparison.path);
    const candidateValue = valueAtPath(candidateEnvelope, comparison.path);
    const baselineHash = computeCanonicalSha256(baselineValue);
    const candidateHash = computeCanonicalSha256(candidateValue);
    return {
      path: comparison.path,
      mode: comparison.mode,
      passed: baselineValue !== undefined
        && candidateValue !== undefined
        && baselineHash === candidateHash,
      baselineHash,
      candidateHash,
    };
  });
  const baselineSummary = summarizeRun(baselineEnvelope);
  const candidateSummary = summarizeRun(candidateEnvelope);
  const artifactMatches = baselineSummary.executionContractHash === candidateSummary.executionContractHash;
  const expectedArtifact = contract.model.expectedExecutionContractHash;
  const expectedArtifactMatches = expectedArtifact === null
    || (
      baselineSummary.executionContractHash === expectedArtifact
      && candidateSummary.executionContractHash === expectedArtifact
    );
  return {
    passed: comparisons.every((comparison) => comparison.passed)
      && artifactMatches
      && expectedArtifactMatches,
    comparisons,
    artifactMatches,
    expectedArtifactMatches,
    baseline: baselineSummary,
    candidate: candidateSummary,
  };
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[midpoint - 1] + sorted[midpoint]) / 2
    : sorted[midpoint];
}

function sampleStats(values) {
  if (values.length === 0) {
    return {
      count: 0, min: null, max: null, mean: null, median: null,
      stdDev: null, relativeStdDevPercent: null, confidence95: null,
    };
  }
  const count = values.length;
  const mean = values.reduce((sum, value) => sum + value, 0) / count;
  const variance = count > 1
    ? values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / (count - 1)
    : 0;
  const stdDev = Math.sqrt(variance);
  const relativeStdDevPercent = mean === 0 ? null : Math.abs((stdDev / mean) * 100);
  let confidence95 = null;
  if (count > 1) {
    const degrees = count - 1;
    const critical = STUDENT_T_95[degrees] ?? 1.96;
    const halfWidth = critical * stdDev / Math.sqrt(count);
    confidence95 = { low: mean - halfWidth, high: mean + halfWidth };
  }
  return {
    count,
    min: Math.min(...values),
    max: Math.max(...values),
    mean,
    median: median(values),
    stdDev,
    relativeStdDevPercent,
    confidence95,
  };
}

function computeImprovementPercent(baseline, candidate, direction) {
  if (!Number.isFinite(baseline) || !Number.isFinite(candidate) || baseline <= 0 || candidate <= 0) {
    return null;
  }
  return direction === 'maximize'
    ? ((candidate - baseline) / baseline) * 100
    : ((baseline - candidate) / baseline) * 100;
}

function assertNotAborted(signal) {
  if (signal?.aborted) {
    const error = new Error('runtime optimization: evaluation aborted.');
    error.name = 'AbortError';
    throw error;
  }
}

async function runCommandSafely(runCommand, request, options, label) {
  assertNotAborted(options.signal);
  options.onEvent?.({ type: 'command:start', label, request: cloneJsonValue(request) });
  try {
    const envelope = await runCommand(request, {
      ...(options.commandOptions ?? {}),
      runtimeLoadOptions: {
        ...(options.commandOptions?.runtimeLoadOptions ?? {}),
        ...(options.signal ? { signal: options.signal } : {}),
      },
    });
    options.onEvent?.({ type: 'command:complete', label, ok: true });
    return { ok: true, envelope };
  } catch (error) {
    const summarized = summarizeError(error);
    options.onEvent?.({ type: 'command:complete', label, ok: false, error: summarized });
    return { ok: false, error: summarized };
  }
}

function baseReceipt(contract, candidate, runtimeInputs) {
  return {
    schema: RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA,
    contractId: contract.contractId,
    contractHash: candidate.contractHash,
    candidateId: candidate.candidateId,
    candidateHash: computeCanonicalSha256(candidate),
    parentHash: candidate.parentHash,
    model: cloneJsonValue(contract.model),
    runtimeInputs: {
      baseline: cloneJsonValue(contract.baseline),
      candidate: cloneJsonValue(runtimeInputs),
      candidateRuntimeConfigHash: computeCanonicalSha256(runtimeInputs.runtimeConfig),
    },
  };
}

function finalizeReceipt(receipt) {
  return {
    ...receipt,
    receiptHash: computeCanonicalSha256(receipt),
  };
}

function rejectedReceipt(base, verification, measurement, reasons, status = 'rejected') {
  return finalizeReceipt({
    ...base,
    verification,
    measurement,
    decision: {
      accepted: false,
      status,
      reasons,
    },
  });
}

export async function evaluateBrowserRuntimeOptimizationCandidate(
  contractInput,
  candidateInput,
  options = {}
) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = validateRuntimeOptimizationCandidate(candidateInput, contract);
  const runtimeInputs = materializeRuntimeOptimizationCandidate(contract, candidate);
  const baselineInputs = cloneJsonValue(contract.baseline);
  const runCommand = options.runCommand ?? runBrowserCommand;
  if (typeof runCommand !== 'function') {
    throw new Error('runtime optimization: options.runCommand must be a function.');
  }
  const base = baseReceipt(contract, candidate, runtimeInputs);
  options.onEvent?.({
    type: 'candidate:start',
    contractHash: candidate.contractHash,
    candidateId: candidate.candidateId,
    candidateHash: base.candidateHash,
  });

  const baselineVerifyRequest = buildCommandRequest(contract, baselineInputs, 'verify');
  const candidateVerifyRequest = buildCommandRequest(contract, runtimeInputs, 'verify');
  const baselineVerifyRun = await runCommandSafely(
    runCommand,
    baselineVerifyRequest,
    options,
    'verification:baseline'
  );
  if (!baselineVerifyRun.ok) {
    return rejectedReceipt(base, {
      passed: false,
      baselineError: baselineVerifyRun.error,
      candidateError: null,
    }, { completedPairs: 0, pairs: [] }, ['baseline_verification_failed'], 'invalid');
  }
  try {
    assertRunEnvelope(baselineVerifyRun.envelope, 'baseline verification', contract.model.modelId);
  } catch (error) {
    return rejectedReceipt(base, {
      passed: false,
      baselineError: summarizeError(error),
      candidateError: null,
    }, { completedPairs: 0, pairs: [] }, ['baseline_verification_failed'], 'invalid');
  }

  const candidateVerifyRun = await runCommandSafely(
    runCommand,
    candidateVerifyRequest,
    options,
    'verification:candidate'
  );
  if (!candidateVerifyRun.ok) {
    return rejectedReceipt(base, {
      passed: false,
      baseline: summarizeRun(baselineVerifyRun.envelope),
      candidateError: candidateVerifyRun.error,
    }, { completedPairs: 0, pairs: [] }, ['candidate_verification_failed']);
  }
  try {
    assertRunEnvelope(candidateVerifyRun.envelope, 'candidate verification', contract.model.modelId);
  } catch (error) {
    return rejectedReceipt(base, {
      passed: false,
      baseline: summarizeRun(baselineVerifyRun.envelope),
      candidateError: summarizeError(error),
    }, { completedPairs: 0, pairs: [] }, ['candidate_verification_failed']);
  }

  const verification = compareVerificationRuns(
    contract,
    baselineVerifyRun.envelope,
    candidateVerifyRun.envelope
  );
  if (!verification.passed) {
    return rejectedReceipt(
      base,
      verification,
      { completedPairs: 0, pairs: [] },
      ['candidate_parity_failed']
    );
  }

  const baselineBenchRequest = buildCommandRequest(contract, baselineInputs, 'bench');
  const candidateBenchRequest = buildCommandRequest(contract, runtimeInputs, 'bench');
  const pairs = [];
  const baselineValues = [];
  const candidateValues = [];
  const improvements = [];
  for (let pairIndex = 0; pairIndex < contract.measurement.pairCount; pairIndex += 1) {
    const order = pairIndex % 2 === 0
      ? ['baseline', 'candidate']
      : ['candidate', 'baseline'];
    const pairRuns = {};
    for (const role of order) {
      const request = role === 'baseline' ? baselineBenchRequest : candidateBenchRequest;
      pairRuns[role] = await runCommandSafely(
        runCommand,
        request,
        options,
        `measurement:${pairIndex}:${role}`
      );
    }
    const pair = { index: pairIndex, order, valid: false, baseline: null, candidate: null, improvementPercent: null };
    try {
      if (!pairRuns.baseline.ok || !pairRuns.candidate.ok) {
        throw new Error('one or more paired commands failed');
      }
      assertRunEnvelope(pairRuns.baseline.envelope, 'baseline benchmark', contract.model.modelId);
      assertRunEnvelope(pairRuns.candidate.envelope, 'candidate benchmark', contract.model.modelId);
      const baselineValue = valueAtPath(pairRuns.baseline.envelope, contract.measurement.metricPath);
      const candidateValue = valueAtPath(pairRuns.candidate.envelope, contract.measurement.metricPath);
      const improvementPercent = computeImprovementPercent(
        baselineValue,
        candidateValue,
        contract.measurement.direction
      );
      if (improvementPercent === null) {
        throw new Error('paired metric values must be finite and greater than zero');
      }
      pair.valid = true;
      pair.baseline = summarizeRun(pairRuns.baseline.envelope, contract.measurement.metricPath);
      pair.candidate = summarizeRun(pairRuns.candidate.envelope, contract.measurement.metricPath);
      pair.improvementPercent = improvementPercent;
      baselineValues.push(baselineValue);
      candidateValues.push(candidateValue);
      improvements.push(improvementPercent);
    } catch (error) {
      pair.error = summarizeError(error);
      if (pairRuns.baseline.ok) {
        pair.baseline = summarizeRun(pairRuns.baseline.envelope, contract.measurement.metricPath);
      } else {
        pair.baselineError = pairRuns.baseline.error;
      }
      if (pairRuns.candidate.ok) {
        pair.candidate = summarizeRun(pairRuns.candidate.envelope, contract.measurement.metricPath);
      } else {
        pair.candidateError = pairRuns.candidate.error;
      }
    }
    pairs.push(pair);
    options.onEvent?.({ type: 'measurement:pair', candidateId: candidate.candidateId, pair });
  }

  const baselineStats = sampleStats(baselineValues);
  const candidateStats = sampleStats(candidateValues);
  const improvementStats = sampleStats(improvements);
  const measurement = {
    metricPath: contract.measurement.metricPath,
    direction: contract.measurement.direction,
    requestedPairs: contract.measurement.pairCount,
    completedPairs: improvements.length,
    pairs,
    baseline: baselineStats,
    candidate: candidateStats,
    improvementPercent: improvementStats,
  };
  const reasons = [];
  if (improvements.length < contract.measurement.minValidPairs) {
    reasons.push('insufficient_valid_pairs');
  }
  if (
    improvementStats.median === null
    || improvementStats.median < contract.measurement.minImprovementPercent
  ) {
    reasons.push('improvement_below_threshold');
  }
  if (contract.measurement.requirePositiveConfidence) {
    if (
      improvementStats.confidence95 === null
      || improvementStats.confidence95.low < contract.measurement.minImprovementPercent
    ) {
      reasons.push('confidence_interval_below_threshold');
    }
  }
  if (
    contract.measurement.maxRelativeStdDevPercent !== null
    && (
      candidateStats.relativeStdDevPercent === null
      || candidateStats.relativeStdDevPercent > contract.measurement.maxRelativeStdDevPercent
    )
  ) {
    reasons.push('candidate_variance_above_threshold');
  }
  const receipt = finalizeReceipt({
    ...base,
    verification,
    measurement,
    decision: {
      accepted: reasons.length === 0,
      status: reasons.length === 0 ? 'accepted' : 'rejected',
      reasons,
    },
  });
  options.onEvent?.({
    type: 'candidate:complete',
    candidateId: candidate.candidateId,
    candidateHash: receipt.candidateHash,
    decision: cloneJsonValue(receipt.decision),
  });
  return receipt;
}
