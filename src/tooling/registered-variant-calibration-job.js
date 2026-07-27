import { computeCanonicalSha256 } from '../utils/canonical-hash.js';
import {
  calibrateRegisteredVariants,
} from './registered-variant-calibration.js';
import {
  enumerateRuntimeOptimizationCandidates,
  evaluateBrowserRuntimeOptimizationCandidate,
  validateRuntimeOptimizationContract,
} from './runtime-optimization.js';

export const REGISTERED_VARIANT_CALIBRATION_JOB_SCHEMA =
  'doppler.registered-variant-calibration-job/v1';

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function candidateKey(candidate) {
  return `${candidate.reference.operation}/${candidate.reference.variantId}`;
}

function assertCurrentKernelBindings(job, registry, kernelDigests) {
  if (!isObject(kernelDigests)) {
    throw new Error('registered calibration job: kernelDigests is required');
  }
  for (const reference of [job.plan?.baseline, ...(job.plan?.candidates ?? [])]) {
    const descriptor = registry?.operations?.[reference?.operation]?.variants?.[reference?.variantId];
    if (!descriptor) continue;
    const key = `${descriptor.wgsl}#${descriptor.entryPoint}`;
    const digest = kernelDigests[key];
    const expected = typeof digest === 'string'
      ? `sha256:${digest.replace(/^sha256:/, '')}`
      : null;
    if (expected === null) {
      throw new Error(`registered calibration job: current kernel digest is missing for ${key}`);
    }
    if (reference.kernelDigest !== expected) {
      throw new Error(
        `registered calibration job: ${reference.operation}/${reference.variantId} ` +
        `kernelDigest does not match current ${key}`
      );
    }
  }
}

function assertEvidenceBinding(evidence, input) {
  const label = candidateKey(input.candidate);
  if (!isObject(evidence)) {
    throw new Error(`registered calibration job: correctnessEvidence.${label} is required`);
  }
  const binding = evidence.binding;
  for (const [field, expected] of Object.entries({
    artifactDigest: input.identity.artifactDigest,
    executionGraphDigest: input.identity.executionGraphDigest,
    descriptorDigest: input.candidate.descriptorDigest,
    kernelDigest: input.candidate.reference.kernelDigest,
    executionEngineDigest: input.identity.executionEngineDigest,
    browserDigest: input.identity.browserDigest,
    adapterDigest: input.identity.adapterDigest,
  })) {
    if (binding?.[field] !== expected) {
      throw new Error(
        `registered calibration job: correctnessEvidence.${label}.binding.${field} mismatch`
      );
    }
  }
}

function resolveCorrectnessEvidence(job, input) {
  const key = candidateKey(input.candidate);
  const evidence = job.correctnessEvidence?.[key];
  assertEvidenceBinding(evidence, input);
  if (input.mode === 'operator-reference') {
    const result = evidence.operatorReference?.[input.shape.shapeId];
    if (!result) {
      throw new Error(
        `registered calibration job: ${key} lacks operator reference for ${input.shape.shapeId}`
      );
    }
    return result;
  }
  if (input.mode === 'boundary-pack') return evidence.boundaryPack;
  return evidence.tokenParity;
}

function assertPerformanceBinding(job, input) {
  const key = candidateKey(input.candidate);
  const performance = job.performance?.[key];
  if (!isObject(performance)) {
    throw new Error(`registered calibration job: performance.${key} is required`);
  }
  const contract = validateRuntimeOptimizationContract(performance.contract);
  const references = contract.mutationPolicy.references ?? [];
  if (references.length !== 1 || references[0].registryId !== performance.registryId) {
    throw new Error(
      `registered calibration job: performance.${key} must select exactly registryId ` +
      `"${String(performance.registryId)}"`
    );
  }
  const registryEntry = job.candidateRegistry.entries?.[performance.registryId];
  const scope = registryEntry?.evidenceScope;
  for (const [field, expected] of Object.entries({
    artifactDigest: input.identity.artifactDigest,
    executionGraphDigest: input.identity.executionGraphDigest,
    descriptorDigest: input.candidate.descriptorDigest,
    kernelDigest: input.candidate.reference.kernelDigest,
    executionEngineDigest: input.identity.executionEngineDigest,
    browserDigest: input.identity.browserDigest,
    adapterDigest: input.identity.adapterDigest,
  })) {
    if (scope?.[field] !== expected) {
      throw new Error(
        `registered calibration job: candidate registry ${performance.registryId} ` +
        `evidenceScope.${field} mismatch`
      );
    }
  }
  const candidates = enumerateRuntimeOptimizationCandidates(contract);
  if (candidates.length !== 1) {
    throw new Error(`registered calibration job: performance.${key} must enumerate one candidate`);
  }
  return { contract, candidate: candidates[0] };
}

function assertExecutionIdentity(job, executionEngine) {
  const actual = computeCanonicalSha256({
    surface: job.surface,
    executionEngine,
  });
  if (job.plan?.identity?.executionEngineDigest !== actual) {
    throw new Error(
      'registered calibration job: plan.identity.executionEngineDigest does not match ' +
      'the active execution engine'
    );
  }
}

function assertBenchIdentity(job, envelope) {
  const identity = envelope?.result?.metrics?.tokenCostLedger?.identity;
  if (!isObject(identity)) {
    throw new Error(
      'registered calibration job: benchmark result must include a token cost ledger identity'
    );
  }
  for (const field of [
    'artifactDigest',
    'executionGraphDigest',
    'browserDigest',
    'adapterDigest',
  ]) {
    if (identity[field] !== job.plan.identity[field]) {
      throw new Error(
        `registered calibration job: benchmark token cost ledger ${field} mismatch`
      );
    }
  }
}

export async function runRegisteredVariantCalibrationJob(job, options) {
  if (job?.schema !== REGISTERED_VARIANT_CALIBRATION_JOB_SCHEMA) {
    throw new Error(
      `registered calibration job: expected ${REGISTERED_VARIANT_CALIBRATION_JOB_SCHEMA}`
    );
  }
  if (!['node', 'browser'].includes(job.surface)) {
    throw new Error('registered calibration job: surface must be node or browser');
  }
  if (!isObject(job.candidateRegistry)) {
    throw new Error('registered calibration job: candidateRegistry is required');
  }
  if (typeof options?.runCommand !== 'function') {
    throw new Error('registered calibration job: runCommand is required');
  }
  if (typeof options.executionEngine !== 'string' || !options.executionEngine) {
    throw new Error('registered calibration job: executionEngine is required');
  }
  assertExecutionIdentity(job, options.executionEngine);
  assertCurrentKernelBindings(job, options.registry, options.kernelDigests);
  const receipt = await calibrateRegisteredVariants(job.plan, {
    registry: options.registry,
    runCorrectness: async (input) => resolveCorrectnessEvidence(job, input),
    evaluatePerformance: async (input) => {
      const performance = assertPerformanceBinding(job, input);
      return evaluateBrowserRuntimeOptimizationCandidate(
        performance.contract,
        performance.candidate,
        {
          runCommand: async (request, commandOptions) => {
            const envelope = await options.runCommand(request, commandOptions);
            if (request.command === 'bench') {
              assertBenchIdentity(job, envelope);
            }
            return envelope;
          },
          commandOptions: job.commandOptions ?? {},
          candidateRegistry: job.candidateRegistry,
          onEvent: options.onEvent,
        }
      );
    },
  });
  const core = {
    ...receipt,
    jobDigest: computeCanonicalSha256(job),
    executionSurface: job.surface,
    executionEngine: options.executionEngine,
  };
  const { digest: _oldDigest, ...withoutDigest } = core;
  return { ...withoutDigest, digest: computeCanonicalSha256(withoutDigest) };
}
