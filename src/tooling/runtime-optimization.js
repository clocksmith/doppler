import { cloneJsonValue } from '../utils/clone-json.js';
import { computeCanonicalSha256, canonicalizeJson } from '../utils/canonical-hash.js';
import { isPlainObject } from '../utils/plain-object.js';
import { runBrowserCommand } from './browser-command-runner.js';
import {
  finalizeRuntimeOptimizationReceipt,
  validateRuntimeOptimizationCampaign,
} from './runtime-optimization-campaign.js';
import { RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA, buildParentHash, materializeRuntimeOptimizationCandidate, normalizeCandidateKind, validateRuntimeOptimizationCandidate, validateRuntimeOptimizationContract } from './runtime-optimization/candidates.js';
export { RUNTIME_OPTIMIZATION_CANDIDATE_REGISTRY_SCHEMA, RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA, RUNTIME_OPTIMIZATION_CONTRACT_SCHEMA, materializeRuntimeOptimizationCandidate, validateRuntimeOptimizationCandidate, validateRuntimeOptimizationCandidateRegistry, validateRuntimeOptimizationContract } from './runtime-optimization/candidates.js';

export const RUNTIME_OPTIMIZATION_RECEIPT_SCHEMA = 'doppler.runtime-optimization-receipt/v1';

const STUDENT_T_95 = Object.freeze({
  1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
  6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
  11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
  16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
  21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
  26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
});

export function hashRuntimeOptimizationContract(input) {
  return computeCanonicalSha256(validateRuntimeOptimizationContract(input));
}

function buildCandidate(contract, patch, registeredReference = null) {
  const contractHash = computeCanonicalSha256(contract);
  const parentHash = buildParentHash(contract);
  const kind = normalizeCandidateKind(contract.kind);
  const identity = computeCanonicalSha256({
    schema: RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA,
    contractHash,
    parentHash,
    kind,
    patch,
    registeredReference,
  });
  return {
    schema: RUNTIME_OPTIMIZATION_CANDIDATE_SCHEMA,
    candidateId: `candidate-${identity.slice('sha256:'.length, 'sha256:'.length + 12)}`,
    contractHash,
    parentHash,
    kind,
    patch,
    ...(registeredReference ? { registeredReference } : {}),
  };
}

export function enumerateRuntimeOptimizationCandidates(input) {
  const contract = validateRuntimeOptimizationContract(input);
  if (normalizeCandidateKind(contract.kind) !== 'runtime-profile') {
    return contract.mutationPolicy.references.map((reference) => (
      buildCandidate(contract, [], cloneJsonValue(reference))
    ));
  }
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

function buildCommandRequest(contract, runtimeInputs, command, workload = contract.workload) {
  const request = workload.request;
  return {
    command,
    workload: workload.type,
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

function inverseNormalCdf(probability) {
  const a = [-39.6968302866538, 220.946098424521, -275.928510446969,
    138.357751867269, -30.6647980661472, 2.50662827745924];
  const b = [-54.4760987982241, 161.585836858041, -155.698979859887,
    66.8013118877197, -13.2806815528857];
  const c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
    -2.54973253934373, 4.37466414146497, 2.93816398269878];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143,
    3.75440866190742];
  const low = 0.02425;
  const high = 1 - low;
  if (probability < low) {
    const q = Math.sqrt(-2 * Math.log(probability));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (probability > high) {
    const q = Math.sqrt(-2 * Math.log(1 - probability));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
      / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = probability - 0.5;
  const r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function studentTCritical(confidenceLevel, degrees) {
  if (confidenceLevel === 0.95) return STUDENT_T_95[degrees] ?? 1.96;
  const z = inverseNormalCdf(0.5 + confidenceLevel / 2);
  const z2 = z * z;
  const first = (z ** 3 + z) / (4 * degrees);
  const second = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * degrees ** 2);
  const third = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z)
    / (384 * degrees ** 3);
  return z + first + second + third;
}

function sampleStats(values, confidenceLevel = 0.95) {
  if (values.length === 0) {
    return {
      count: 0, min: null, max: null, mean: null, median: null,
      stdDev: null, relativeStdDevPercent: null, confidence95: null,
      decisionConfidence: null,
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
  let decisionConfidence = null;
  if (count > 1) {
    const degrees = count - 1;
    const critical = studentTCritical(0.95, degrees);
    const halfWidth = critical * stdDev / Math.sqrt(count);
    confidence95 = { low: mean - halfWidth, high: mean + halfWidth };
    const decisionCritical = studentTCritical(confidenceLevel, degrees);
    const decisionHalfWidth = decisionCritical * stdDev / Math.sqrt(count);
    decisionConfidence = {
      level: confidenceLevel,
      low: mean - decisionHalfWidth,
      high: mean + decisionHalfWidth,
    };
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
    decisionConfidence,
  };
}

function randomizedBlockOrder(pairIndex, seed, candidateHash) {
  const blockIndex = Math.floor(pairIndex / 2);
  const seedHash = computeCanonicalSha256({ seed, candidateHash, blockIndex });
  const firstBaseline = Number.parseInt(seedHash.slice(7, 15), 16) % 2 === 0;
  const baselineFirst = pairIndex % 2 === 0 ? firstBaseline : !firstBaseline;
  return baselineFirst
    ? ['baseline', 'candidate']
    : ['candidate', 'baseline'];
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
    candidateKind: candidate.kind,
    registeredReference: cloneJsonValue(candidate.registeredReference ?? null),
    parentHash: candidate.parentHash,
    campaign: cloneJsonValue(contract.campaign),
    model: cloneJsonValue(contract.model),
    runtimeInputs: {
      baseline: cloneJsonValue(contract.baseline),
      candidate: cloneJsonValue(runtimeInputs),
      candidateRuntimeConfigHash: computeCanonicalSha256(runtimeInputs.runtimeConfig),
    },
    neighboringWorkloadGuards: null,
  };
}

function rejectedReceipt(base, verification, measurement, reasons, status = 'rejected') {
  return finalizeRuntimeOptimizationReceipt({
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

async function evaluateNeighboringWorkloadGuards({
  contract,
  baselineInputs,
  runtimeInputs,
  runCommand,
  options,
}) {
  const results = [];
  for (const guard of contract.neighboringWorkloads ?? []) {
    const baselineVerify = await runCommandSafely(
      runCommand,
      buildCommandRequest(contract, baselineInputs, 'verify', guard.workload),
      options,
      `neighbor:${guard.guardId}:verification:baseline`
    );
    const candidateVerify = await runCommandSafely(
      runCommand,
      buildCommandRequest(contract, runtimeInputs, 'verify', guard.workload),
      options,
      `neighbor:${guard.guardId}:verification:candidate`
    );
    let verification = null;
    try {
      if (!baselineVerify.ok || !candidateVerify.ok) {
        throw new Error('neighbor verification command failed');
      }
      assertRunEnvelope(baselineVerify.envelope, 'neighbor baseline verification', contract.model.modelId);
      assertRunEnvelope(candidateVerify.envelope, 'neighbor candidate verification', contract.model.modelId);
      verification = compareVerificationRuns(
        contract,
        baselineVerify.envelope,
        candidateVerify.envelope
      );
    } catch (error) {
      results.push({
        guardId: guard.guardId,
        passed: false,
        verification: { passed: false, error: summarizeError(error) },
        pairs: [],
        reason: 'neighbor_verification_failed',
      });
      continue;
    }
    if (!verification.passed) {
      results.push({
        guardId: guard.guardId,
        passed: false,
        verification,
        pairs: [],
        reason: 'neighbor_parity_failed',
      });
      continue;
    }
    const improvements = [];
    const pairs = [];
    for (let pairIndex = 0; pairIndex < guard.pairCount; pairIndex += 1) {
      const order = pairIndex % 2 === 0
        ? ['baseline', 'candidate']
        : ['candidate', 'baseline'];
      const runs = {};
      for (const role of order) {
        runs[role] = await runCommandSafely(
          runCommand,
          buildCommandRequest(
            contract,
            role === 'baseline' ? baselineInputs : runtimeInputs,
            'bench',
            guard.workload
          ),
          options,
          `neighbor:${guard.guardId}:measurement:${pairIndex}:${role}`
        );
      }
      const pair = { index: pairIndex, order, valid: false };
      try {
        if (!runs.baseline.ok || !runs.candidate.ok) {
          throw new Error('neighbor paired command failed');
        }
        assertRunEnvelope(runs.baseline.envelope, 'neighbor baseline benchmark', contract.model.modelId);
        assertRunEnvelope(runs.candidate.envelope, 'neighbor candidate benchmark', contract.model.modelId);
        const baselineValue = valueAtPath(runs.baseline.envelope, guard.metricPath);
        const candidateValue = valueAtPath(runs.candidate.envelope, guard.metricPath);
        const improvement = computeImprovementPercent(
          baselineValue,
          candidateValue,
          guard.direction
        );
        if (improvement === null) throw new Error('neighbor metric is invalid');
        pair.valid = true;
        pair.improvementPercent = improvement;
        improvements.push(improvement);
      } catch (error) {
        pair.error = summarizeError(error);
      }
      pairs.push(pair);
    }
    const stats = sampleStats(improvements);
    const passed = improvements.length === guard.pairCount
      && stats.median >= -guard.maxRegressionPercent;
    results.push({
      guardId: guard.guardId,
      passed,
      verification,
      pairs,
      improvementPercent: stats,
      maxRegressionPercent: guard.maxRegressionPercent,
      reason: passed ? null : 'neighbor_regression_exceeded',
    });
  }
  return {
    passed: results.every((result) => result.passed),
    results,
  };
}

export async function evaluateBrowserRuntimeOptimizationCandidate(
  contractInput,
  candidateInput,
  options = {}
) {
  const contract = validateRuntimeOptimizationContract(contractInput);
  const candidate = validateRuntimeOptimizationCandidate(candidateInput, contract);
  const runtimeInputs = materializeRuntimeOptimizationCandidate(contract, candidate, {
    candidateRegistry: options.candidateRegistry,
  });
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
  const sequentialLooks = [];
  let sequentialStop = null;
  for (let pairIndex = 0; pairIndex < contract.measurement.pairCount; pairIndex += 1) {
    const order = contract.measurement.orderPolicy
      ? randomizedBlockOrder(
        pairIndex,
        contract.measurement.orderPolicy.seed,
        base.candidateHash
      )
      : (
        pairIndex % 2 === 0
          ? ['baseline', 'candidate']
          : ['candidate', 'baseline']
      );
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
    const sequential = contract.measurement.sequentialDecision;
    const isScheduledLook = sequential
      && (
        (pairIndex + 1) % sequential.lookEveryPairs === 0
        || pairIndex + 1 === contract.measurement.pairCount
      );
    if (isScheduledLook && improvements.length >= sequential.minimumPairs) {
      const adjustedAlpha = sequential.alpha / sequential.maximumLooks;
      const lookStats = sampleStats(improvements, 1 - adjustedAlpha);
      const look = {
        look: sequentialLooks.length + 1,
        completedPairs: pairIndex + 1,
        validPairs: improvements.length,
        adjustedAlpha,
        confidence: lookStats.decisionConfidence,
        decision: 'continue',
      };
      if (
        lookStats.decisionConfidence
        && lookStats.decisionConfidence.low >= contract.measurement.minImprovementPercent
      ) {
        look.decision = 'accept';
        sequentialStop = 'accept';
      } else if (
        lookStats.decisionConfidence
        && lookStats.decisionConfidence.high < contract.measurement.minImprovementPercent
      ) {
        look.decision = 'reject';
        sequentialStop = 'reject';
      }
      sequentialLooks.push(look);
      options.onEvent?.({
        type: 'measurement:sequential-look',
        candidateId: candidate.candidateId,
        look,
      });
      if (sequentialStop) break;
    }
  }

  const baselineStats = sampleStats(baselineValues);
  const candidateStats = sampleStats(candidateValues);
  const sequential = contract.measurement.sequentialDecision;
  const decisionConfidenceLevel = sequential
    ? 1 - (sequential.alpha / sequential.maximumLooks)
    : 0.95;
  const improvementStats = sampleStats(improvements, decisionConfidenceLevel);
  const measurement = {
    metricPath: contract.measurement.metricPath,
    direction: contract.measurement.direction,
    requestedPairs: contract.measurement.pairCount,
    completedPairs: improvements.length,
    pairs,
    baseline: baselineStats,
    candidate: candidateStats,
    improvementPercent: improvementStats,
    orderPolicy: contract.measurement.orderPolicy ?? { kind: 'alternating' },
    sequentialDecision: sequential
      ? {
        policy: cloneJsonValue(sequential),
        looks: sequentialLooks,
        stoppedEarly: pairs.length < contract.measurement.pairCount,
        stopDecision: sequentialStop,
      }
      : null,
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
      improvementStats.decisionConfidence === null
      || improvementStats.decisionConfidence.low < contract.measurement.minImprovementPercent
    ) {
      reasons.push('confidence_interval_below_threshold');
    }
  }
  if (sequentialStop === 'reject') {
    reasons.push('sequential_evidence_below_threshold');
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
  const neighboringWorkloadGuards = reasons.length === 0
    ? await evaluateNeighboringWorkloadGuards({
      contract,
      baselineInputs,
      runtimeInputs,
      runCommand,
      options,
    })
    : { passed: true, results: [], skipped: 'primary_candidate_rejected' };
  if (!neighboringWorkloadGuards.passed) {
    reasons.push('neighboring_workload_guard_failed');
  }
  const receipt = finalizeRuntimeOptimizationReceipt({
    ...base,
    verification,
    measurement,
    neighboringWorkloadGuards,
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
