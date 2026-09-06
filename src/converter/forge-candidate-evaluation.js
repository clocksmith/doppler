import { computeCanonicalSha256 } from '../formats/canonical-hash.js';
import { isPlainObject } from '../formats/plain-object.js';

export const FORGE_EVALUATION_SCHEMA = 'doppler.forge-candidate-evaluation/v1';
const DIGEST = /^sha256:[0-9a-f]{64}$/u;

function requireValue(condition, message) {
  if (!condition) throw new Error(`Forge evaluation: ${message}`);
}

function fields(value, keys, label) {
  requireValue(isPlainObject(value), `${label} must be an object.`);
  requireValue(Object.keys(value).length === keys.length
    && keys.every(key => Object.hasOwn(value, key)), `${label} requires exactly ${keys.join(', ')}.`);
}

function nonempty(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function unique(values, label) {
  requireValue(Array.isArray(values) && values.length > 0, `${label} must not be empty.`);
  requireValue(new Set(values).size === values.length, `${label} contains duplicates.`);
}

function assertJson(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number') {
    requireValue(Number.isFinite(value), 'nonfinite JSON number.');
    return;
  }
  requireValue(Array.isArray(value) || isPlainObject(value), 'only JSON data may enter evaluation.');
  for (const child of Object.values(value)) assertJson(child);
}

export function validateForgeEvaluationContract(contract, reference) {
  assertJson(contract);
  assertJson(reference);
  fields(contract, ['schema', 'evaluationId', 'modelIRHash', 'candidateHashes', 'referenceHash',
    'scope', 'sampling', 'checks', 'metrics', 'selection'], 'contract');
  requireValue(contract.schema === FORGE_EVALUATION_SCHEMA, 'unsupported contract schema.');
  requireValue(nonempty(contract.evaluationId), 'evaluationId is required.');
  requireValue(DIGEST.test(contract.modelIRHash) && DIGEST.test(contract.referenceHash), 'invalid identity digest.');
  unique(contract.candidateHashes, 'candidateHashes');
  requireValue(contract.candidateHashes.every(hash => DIGEST.test(hash)), 'invalid candidate digest.');
  fields(contract.scope, ['surface', 'runtimeHash', 'environmentHash', 'workloadHash', 'cacheMode', 'loadMode'], 'scope');
  for (const key of ['surface', 'cacheMode', 'loadMode']) {
    requireValue(nonempty(contract.scope[key]), `scope.${key} is required.`);
  }
  for (const key of ['runtimeHash', 'environmentHash', 'workloadHash']) {
    requireValue(DIGEST.test(contract.scope[key]), `scope.${key} must be a digest.`);
  }
  fields(contract.sampling, ['warmupRuns', 'timedRuns', 'order', 'seed'], 'sampling');
  for (const key of ['warmupRuns', 'timedRuns', 'seed']) {
    requireValue(Number.isSafeInteger(contract.sampling[key]) && contract.sampling[key] >= 0,
      `sampling.${key} must be a nonnegative safe integer.`);
  }
  requireValue(contract.sampling.timedRuns > 0 && contract.sampling.order === 'balanced-rotation', 'invalid sampling policy.');
  requireValue(contract.selection === 'observed-range-pareto', 'unsupported selection policy.');
  requireValue(Array.isArray(contract.checks) && Array.isArray(contract.metrics), 'checks and metrics must be arrays.');
  unique(contract.checks.map(check => check?.id), 'check IDs');
  unique(contract.metrics.map(metric => metric?.id), 'metric IDs');
  for (const check of contract.checks) {
    fields(check, ['id', 'mode', 'maxAbsoluteError'], 'check');
    requireValue(nonempty(check.id), 'check.id is required.');
    requireValue(check.mode === 'canonical-exact' ? check.maxAbsoluteError === null
      : check.mode === 'absolute-array' && Number.isFinite(check.maxAbsoluteError)
        && check.maxAbsoluteError >= 0, 'invalid oracle comparison policy.');
  }
  for (const metric of contract.metrics) {
    fields(metric, ['id', 'unit', 'direction', 'limit'], 'metric');
    requireValue(nonempty(metric.id) && nonempty(metric.unit), 'metric ID and unit are required.');
    requireValue(['minimize', 'maximize'].includes(metric.direction), 'invalid metric direction.');
    requireValue(metric.limit === null || Number.isFinite(metric.limit) && metric.limit >= 0,
      'metric.limit must be nonnegative or explicitly null.');
  }
  fields(reference, ['schema', 'sourceHash', 'oracleHash', 'cases'], 'reference');
  requireValue(reference.schema === 'doppler.forge-source-reference/v1'
    && DIGEST.test(reference.sourceHash) && DIGEST.test(reference.oracleHash), 'source and independent oracle digests are required.');
  requireValue(computeCanonicalSha256(reference) === contract.referenceHash, 'reference digest differs.');
  requireValue(Array.isArray(reference.cases), 'reference cases must be an array.');
  unique(reference.cases.map(row => row?.id), 'reference case IDs');
  for (const row of reference.cases) {
    fields(row, ['id', 'input', 'expected'], 'reference case');
    requireValue(nonempty(row.id) && row.input !== undefined, 'reference case identity and input are required.');
    fields(row.expected, contract.checks.map(check => check.id), 'reference outputs');
    for (const check of contract.checks) {
      if (check.mode === 'absolute-array') {
        const values = row.expected[check.id];
        requireValue(Array.isArray(values) && values.length > 0 && values.every(Number.isFinite),
          `reference.${row.id}.${check.id} requires a complete finite numeric array.`);
      }
    }
  }
  const workload = reference.cases.map(({ id, input }) => ({ id, input }));
  requireValue(computeCanonicalSha256(workload) === contract.scope.workloadHash, 'workload digest differs.');
  computeCanonicalSha256(contract);
  return contract;
}

export function createForgeEvaluationSchedule(contract, reference) {
  validateForgeEvaluationContract(contract, reference);
  const contractHash = computeCanonicalSha256(contract);
  const hashes = [...contract.candidateHashes].sort();
  const schedule = [];
  for (const [phase, runs] of [['warmup', contract.sampling.warmupRuns], ['timed', contract.sampling.timedRuns]]) {
    for (let run = 0; run < runs; run += 1) {
      for (const [caseIndex, row] of reference.cases.entries()) {
        const offset = ((contract.sampling.seed % hashes.length) + run + caseIndex) % hashes.length;
        const order = [...hashes.slice(offset), ...hashes.slice(0, offset)];
        for (const candidateHash of order) {
          schedule.push({ attemptId: computeCanonicalSha256({ contractHash, phase, run, caseId: row.id, candidateHash }),
            candidateHash, phase, run, caseId: row.id, inputHash: computeCanonicalSha256(row.input) });
        }
      }
    }
  }
  return schedule;
}

// Observation only. Every frozen output component is checked; these values
// never feed a model tensor or supply an alternative runtime implementation.
function compareOutputs(contract, expected, output) {
  fields(output, contract.checks.map(check => check.id), 'observed outputs');
  return contract.checks.map(check => {
    const actual = output[check.id];
    const target = expected[check.id];
    if (check.mode === 'canonical-exact') {
      return { id: check.id, passed: computeCanonicalSha256(actual) === computeCanonicalSha256(target) };
    }
    if (!Array.isArray(actual) || actual.length !== target.length || !actual.every(Number.isFinite)) {
      return { id: check.id, passed: false, reason: 'incomplete_or_nonfinite_output' };
    }
    let maxAbsoluteError = 0;
    for (let index = 0; index < target.length; index += 1) {
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(actual[index] - target[index]));
    }
    return { id: check.id, passed: maxAbsoluteError <= check.maxAbsoluteError, maxAbsoluteError };
  });
}

function evaluateAttempt(contract, reference, expected, observation) {
  if (!observation) return { ...expected, passed: false, reason: 'missing_attempt', checks: [] };
  try {
    fields(observation, ['attemptId', 'candidateHash', 'contractHash', 'scopeHash', 'inputHash',
      'status', 'output', 'metrics', 'error'], 'observation');
    requireValue(observation.contractHash === computeCanonicalSha256(contract), 'attempt contract differs.');
    requireValue(observation.scopeHash === computeCanonicalSha256(contract.scope), 'attempt scope differs.');
    requireValue(observation.candidateHash === expected.candidateHash && observation.inputHash === expected.inputHash,
      'attempt candidate or input differs.');
    requireValue(['completed', 'failed', 'cancelled'].includes(observation.status), 'invalid attempt status.');
    if (observation.status !== 'completed') {
      requireValue(observation.output === null && observation.metrics === null && nonempty(observation.error),
        'failed attempts require an error, not successful outputs.');
      return { ...expected, passed: false, reason: observation.status, error: observation.error, checks: [] };
    }
    requireValue(observation.error === null, 'completed attempt carries an error.');
    const row = reference.cases.find(item => item.id === expected.caseId);
    const checks = compareOutputs(contract, row.expected, observation.output);
    if (expected.phase === 'warmup') {
      requireValue(observation.metrics === null, 'warmup must not enter timed measurements.');
    } else {
      fields(observation.metrics, contract.metrics.map(metric => metric.id), 'observed metrics');
      for (const metric of contract.metrics) {
        const value = observation.metrics[metric.id];
        requireValue(Number.isFinite(value) && value >= 0, `metric ${metric.id} must be finite and nonnegative.`);
        if (metric.limit !== null) {
          checks.push({ id: `metric:${metric.id}`, passed: metric.direction === 'minimize'
            ? value <= metric.limit : value >= metric.limit });
        }
      }
    }
    return { ...expected, passed: checks.every(check => check.passed), checks, reason: null };
  } catch (error) {
    return { ...expected, passed: false, reason: error.message, checks: [] };
  }
}

function dominates(left, right, metrics, cases) {
  let strictlyBetter = false;
  for (const { id: caseId } of cases) {
    for (const metric of metrics) {
      const a = left.ranges[caseId][metric.id];
      const b = right.ranges[caseId][metric.id];
      const worst = metric.direction === 'minimize' ? a.max : a.min;
      const best = metric.direction === 'minimize' ? b.min : b.max;
      if (metric.direction === 'minimize' ? worst > best : worst < best) return false;
      if (worst !== best) strictlyBetter = true;
    }
  }
  return strictlyBetter;
}

export function evaluateForgeCandidates({ contract, reference, observations }) {
  const schedule = createForgeEvaluationSchedule(contract, reference);
  requireValue(Array.isArray(observations), 'observations must be an ordered array.');
  assertJson(observations);
  const expectedById = new Map(schedule.map((attempt, index) => [attempt.attemptId, index]));
  const observedById = new Map();
  let previousIndex = -1;
  for (const observation of observations) {
    const index = expectedById.get(observation?.attemptId);
    requireValue(index !== undefined && index > previousIndex, 'unknown, duplicate, or reordered attempt.');
    observedById.set(observation.attemptId, observation);
    previousIndex = index;
  }
  const attempts = schedule.map(attempt => evaluateAttempt(contract, reference, attempt, observedById.get(attempt.attemptId)));
  const candidates = [...contract.candidateHashes].sort().map(candidateHash => {
    const ownAttempts = attempts.filter(attempt => attempt.candidateHash === candidateHash);
    const failures = ownAttempts.filter(attempt => !attempt.passed);
    const ranges = Object.create(null);
    if (failures.length === 0) {
      for (const row of reference.cases) {
        const timed = ownAttempts.filter(attempt => attempt.caseId === row.id && attempt.phase === 'timed');
        ranges[row.id] = Object.fromEntries(contract.metrics.map(metric => {
          const values = timed.map(attempt => observedById.get(attempt.attemptId).metrics[metric.id]);
          return [metric.id, { min: Math.min(...values), max: Math.max(...values), count: values.length }];
        }));
      }
    }
    return { candidateHash, eligible: failures.length === 0, failedAttemptIds: failures.map(attempt => attempt.attemptId), ranges };
  });
  const eligible = candidates.filter(candidate => candidate.eligible);
  const results = candidates.map(candidate => {
    const dominatedBy = candidate.eligible ? eligible.filter(other => other !== candidate
      && dominates(other, candidate, contract.metrics, reference.cases)).map(other => other.candidateHash) : [];
    return { ...candidate, dominatedBy, retained: candidate.eligible && dominatedBy.length === 0 };
  });
  return {
    schema: 'doppler.forge-candidate-evaluation-receipt/v1',
    contractHash: computeCanonicalSha256(contract), referenceHash: contract.referenceHash,
    observationsHash: computeCanonicalSha256(observations), modelIRHash: contract.modelIRHash,
    selection: contract.selection, scope: structuredClone(contract.scope), attempts, candidates: results,
    selectedCandidateHashes: results.filter(candidate => candidate.retained).map(candidate => candidate.candidateHash),
    claimAllowed: false, promotionAllowed: false,
  };
}

// The host adapter owns execution and cleanup, not selection. It must report
// observed identities; the runner does not stamp successful scope on its behalf.
export async function runForgeCandidateEvaluation({ contract, reference, runAttempt, signal, onObservation }) {
  const frozenContract = structuredClone(contract);
  const frozenReference = structuredClone(reference);
  const schedule = createForgeEvaluationSchedule(frozenContract, frozenReference);
  requireValue(typeof runAttempt === 'function', 'runAttempt adapter is required.');
  const contractHash = computeCanonicalSha256(frozenContract);
  const scopeHash = computeCanonicalSha256(frozenContract.scope);
  const observations = [];
  for (const attempt of schedule) {
    if (signal?.aborted) break;
    let observation;
    try {
      observation = await runAttempt({ attempt: structuredClone(attempt),
        input: structuredClone(frozenReference.cases.find(row => row.id === attempt.caseId).input),
        contractHash, scopeHash, signal });
      requireValue(!signal?.aborted, 'attempt cancelled before its result was admitted.');
      assertJson(observation);
      requireValue(observation.attemptId === attempt.attemptId, 'adapter returned another attempt.');
      observation = structuredClone(observation);
    } catch (error) {
      observation = { attemptId: attempt.attemptId, candidateHash: attempt.candidateHash,
        contractHash, scopeHash, inputHash: attempt.inputHash,
        status: signal?.aborted ? 'cancelled' : 'failed', output: null, metrics: null,
        error: String(error?.message ?? error) || 'Execution failed without an error message.' };
    }
    observations.push(observation);
    if (onObservation) await onObservation(structuredClone(observation));
  }
  return { contract: frozenContract, reference: frozenReference, observations,
    receipt: evaluateForgeCandidates({ contract: frozenContract, reference: frozenReference, observations }) };
}
