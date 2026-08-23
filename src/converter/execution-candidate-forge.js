import { hashModelIR, validateModelIR } from '../config/model-ir.js';
import { createTargetPlanV2 } from '../config/target-plan.js';
import {
  INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID,
  PROGRAM_LOAD_POLICY_SCHEMA_ID,
} from '../config/initial-execution-identity.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const EXECUTION_CANDIDATE_FORGE_SCHEMA_ID = 'doppler.execution-candidate-forge/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hashValue(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function requireAttribution(proposal) {
  if (!isObject(proposal.author) || !['human', 'ai', 'tool'].includes(proposal.author.kind)
    || typeof proposal.author.actor !== 'string' || !proposal.author.actor.trim()) {
    throw new Error(`Proposal "${proposal.id}" has no attributable author.`);
  }
}

function resolveEntryPoint(modelIR, entryPointId) {
  const entryPoint = modelIR.entryPoints.find((candidate) => candidate.id === entryPointId);
  if (!entryPoint) throw new Error(`ModelIR has no entry point "${entryPointId}".`);
  if (entryPoint.status !== 'lowered') throw new Error(`Entry point "${entryPointId}" is not lowered.`);
  return entryPoint;
}

function reachableSchedule(modelIR, componentId) {
  const schedules = modelIR.blockSchedules.filter((schedule) => schedule.componentId === componentId);
  if (schedules.length !== 1) throw new Error(`Component "${componentId}" must have exactly one block schedule.`);
  return schedules[0];
}

function resolveKernelClosure(kernelIds, kernels) {
  return [...new Set(kernelIds)].sort().map((moduleId) => {
    const kernel = kernels[moduleId];
    if (!isObject(kernel)) throw new Error(`Kernel "${moduleId}" is absent from the lowering vocabulary.`);
    if (!DIGEST_PATTERN.test(kernel.digest || '') || !DIGEST_PATTERN.test(kernel.sourceHash || '')) {
      throw new Error(`Kernel "${moduleId}" is not content-addressed.`);
    }
    return { moduleId, digest: kernel.digest, sourceHash: kernel.sourceHash };
  });
}

function compileProposal(modelIR, entryPoint, schedule, vocabulary, proposal) {
  requireAttribution(proposal);
  if (!Number.isFinite(proposal.score)) throw new Error(`Proposal "${proposal.id}" requires a finite score.`);
  const classById = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  const loweringsById = new Map(vocabulary.blockLowerings.map((lowering) => [lowering.id, lowering]));
  const classPrograms = {};
  const kernelIds = [...(vocabulary.entryPointKernels?.[entryPoint.kind] || [])];
  for (const blockClassId of new Set(schedule.blocks.map((block) => block.blockClassId))) {
    const blockClass = classById.get(blockClassId);
    const loweringId = proposal.selections?.[blockClassId];
    const lowering = loweringsById.get(loweringId);
    if (!lowering) throw new Error(`Proposal "${proposal.id}" does not lower block class "${blockClassId}".`);
    if (!Array.isArray(lowering.blockKinds) || !lowering.blockKinds.includes(blockClass.kind)) {
      throw new Error(`Lowering "${lowering.id}" cannot implement block kind "${blockClass.kind}".`);
    }
    for (const phase of entryPoint.phases) {
      const steps = lowering.phases?.[phase];
      if (!Array.isArray(steps) || steps.length === 0) {
        throw new Error(`Lowering "${lowering.id}" has no ${phase} program.`);
      }
      kernelIds.push(...steps.map((step) => step.kernelId));
    }
    classPrograms[blockClassId] = {
      loweringId: lowering.id,
      phases: Object.fromEntries(entryPoint.phases.map((phase) => [phase, lowering.phases[phase]])),
    };
  }
  const stateKinds = new Set(modelIR.stateSpaces.map((state) => state.kind));
  const supportedStateKinds = new Set(proposal.supportedStateKinds || []);
  for (const stateKind of stateKinds) {
    if (!supportedStateKinds.has(stateKind)) {
      throw new Error(`Proposal "${proposal.id}" does not implement ${stateKind} state.`);
    }
  }
  const kernelClosure = resolveKernelClosure(kernelIds, vocabulary.kernels);
  const executionGraph = {
    schema: 'doppler.semantic-execution-graph/v2',
    modelIRHash: hashModelIR(modelIR),
    entryPointId: entryPoint.id,
    componentId: entryPoint.componentId,
    phases: entryPoint.phases,
    schedule: schedule.blocks,
    classPrograms,
    entryPointKernels: vocabulary.entryPointKernels?.[entryPoint.kind] || [],
  };
  const executionGraphHash = hashValue(executionGraph);
  const declaredStepIds = Object.fromEntries(entryPoint.phases.map((phase) => [
    phase,
    schedule.blocks.flatMap((block) => classPrograms[block.blockClassId].phases[phase].map((step) => (
      `${phase}.block.${block.index}.${step.id}`
    ))),
  ]));
  const programBundle = {
    schema: 'doppler.generated-program-bundle/v2',
    modelId: modelIR.modelId,
    modelIRHash: hashModelIR(modelIR),
    executionGraphHash,
    executionGraph,
    kernelClosure,
    proposal: { id: proposal.id, author: proposal.author },
  };
  const programBundleHash = hashValue(programBundle);
  return {
    schema: 'doppler.target-plan-candidate/v2',
    proposalId: proposal.id,
    author: proposal.author,
    score: proposal.score,
    targetId: proposal.targetId,
    modelId: modelIR.modelId,
    modelIRHash: hashModelIR(modelIR),
    executionGraphHash,
    programBundleHash,
    capabilityPredicate: proposal.capabilityPredicate,
    dtypes: proposal.dtypes,
    fusions: proposal.fusions || [],
    kernelClosure,
    memoryLayout: proposal.memoryLayout,
    phases: Object.fromEntries(entryPoint.phases.map((phase) => [phase, [{
      kind: 'program-phase',
      phase,
      executionGraphHash,
      declaredStepIds: declaredStepIds[phase],
    }]])),
    executionGraph,
    programBundle,
  };
}

export function searchExecutionCandidates({ modelIR, entryPointId, vocabulary, proposals }) {
  const modelValidation = validateModelIR(modelIR);
  if (!modelValidation.ok || modelIR.schema !== 'doppler.model-ir/v2') {
    throw new Error(`Execution candidate search requires ModelIR v2: ${modelValidation.errors.join('; ')}`);
  }
  if (!isObject(vocabulary) || vocabulary.schema !== EXECUTION_CANDIDATE_FORGE_SCHEMA_ID) {
    throw new Error(`Execution candidate search requires vocabulary "${EXECUTION_CANDIDATE_FORGE_SCHEMA_ID}".`);
  }
  if (!Array.isArray(proposals) || proposals.length === 0) {
    throw new Error('Execution candidate search requires proposals.');
  }
  const entryPoint = resolveEntryPoint(modelIR, entryPointId);
  const schedule = reachableSchedule(modelIR, entryPoint.componentId);
  const valid = [];
  const rejected = [];
  for (const proposal of proposals) {
    try {
      valid.push(compileProposal(modelIR, entryPoint, schedule, vocabulary, proposal));
    } catch (error) {
      rejected.push({ proposalId: proposal?.id ?? null, reason: error.message });
    }
  }
  if (valid.length === 0) throw new Error(`All execution candidates were rejected: ${rejected.map((entry) => entry.reason).join('; ')}`);
  valid.sort((left, right) => left.score - right.score || left.proposalId.localeCompare(right.proposalId));
  const accepted = valid[0];
  rejected.push(...valid.slice(1).map((candidate) => ({
    proposalId: candidate.proposalId,
    reason: `Deterministic score ${candidate.score} did not beat ${accepted.score}.`,
  })));
  return {
    schema: 'doppler.execution-candidate-search-receipt/v1',
    modelId: modelIR.modelId,
    entryPointId,
    generatedCandidates: proposals.length,
    rejectedCandidates: rejected,
    acceptedCandidate: accepted,
    acceptedProposalId: accepted.proposalId,
  };
}

export function promoteExecutionCandidate(candidate, evidence) {
  if (!isObject(candidate) || candidate.schema !== 'doppler.target-plan-candidate/v2') {
    throw new Error('Only a validated TargetPlan candidate can be promoted.');
  }
  if (evidence?.initialExecutionIdentity?.schema !== INITIAL_EXECUTION_IDENTITY_V2_SCHEMA_ID
    || evidence.initialExecutionIdentity?.programLoadPolicy?.schema !== PROGRAM_LOAD_POLICY_SCHEMA_ID) {
    throw new Error(
      'Execution candidate promotion requires initial execution identity v2 with current signed '
      + 'program-load policy.'
    );
  }
  return createTargetPlanV2({
    targetId: candidate.targetId,
    modelId: candidate.modelId,
    modelIRHash: candidate.modelIRHash,
    executionGraphHash: candidate.executionGraphHash,
    programBundleHash: candidate.programBundleHash,
    capabilityPredicate: candidate.capabilityPredicate,
    dtypes: candidate.dtypes,
    fusions: candidate.fusions,
    kernelClosure: candidate.kernelClosure,
    memoryLayout: candidate.memoryLayout,
    phases: candidate.phases,
    qualification: evidence.qualification,
    initialExecutionIdentity: evidence.initialExecutionIdentity,
  });
}
