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

function resolveEntryPoint(modelIR, entryPointId, requireLowered = true) {
  const entryPoint = modelIR.entryPoints.find((candidate) => candidate.id === entryPointId);
  if (!entryPoint) throw new Error(`ModelIR has no entry point "${entryPointId}".`);
  if (requireLowered && entryPoint.status !== 'lowered') {
    throw new Error(`Entry point "${entryPointId}" is not lowered.`);
  }
  return entryPoint;
}

function reachableSchedule(modelIR, componentId) {
  const schedules = modelIR.blockSchedules.filter((schedule) => schedule.componentId === componentId);
  if (schedules.length !== 1) throw new Error(`Component "${componentId}" must have exactly one block schedule.`);
  return schedules[0];
}

function entryPointAuditPhases(entryPoint) {
  if (Array.isArray(entryPoint.phases) && entryPoint.phases.length > 0) return entryPoint.phases;
  if (entryPoint.kind === 'generate') return ['prefill', 'decode'];
  if (entryPoint.kind === 'encode') return ['encode'];
  if (entryPoint.kind === 'speculative-generate') return ['draft', 'verify'];
  throw new Error(`Entry point "${entryPoint.id}" has no declared or known phases.`);
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

function valuesEqual(left, right) {
  return JSON.stringify(stableSortObject(left)) === JSON.stringify(stableSortObject(right));
}

function semanticContractErrors({ semantics, contract, semanticsLabel, contractLabel }) {
  if (!isObject(semantics)) return [`${semanticsLabel} has no declarative semantics.`];
  if (!isObject(contract)) return [`${contractLabel} is not declared.`];
  const dynamicFields = new Set(contract.dynamicFields || []);
  const allowedValues = contract.allowedValues || {};
  if (!Array.isArray(contract.dynamicFields) || !isObject(allowedValues)) {
    return [`${contractLabel} is invalid.`];
  }
  const errors = [];
  for (const [field, value] of Object.entries(semantics)) {
    if (Object.hasOwn(allowedValues, field)) {
      const admitted = allowedValues[field];
      if (!Array.isArray(admitted) || !admitted.some((candidate) => valuesEqual(candidate, value))) {
        errors.push(`${contractLabel} does not admit ${semanticsLabel}.${field}=${JSON.stringify(value)}.`);
      }
      continue;
    }
    if (!dynamicFields.has(field)) {
      errors.push(`${contractLabel} does not bind ${semanticsLabel}.${field}.`);
    }
  }
  return errors;
}

function semanticSectionErrors(blockClass, lowering, sectionName) {
  return semanticContractErrors({
    semantics: blockClass[sectionName],
    contract: lowering.blockContract?.[sectionName],
    semanticsLabel: sectionName,
    contractLabel: `Lowering "${lowering.id}"`,
  });
}

function entryPointSemanticContracts(modelIR, entryPoint, vocabulary) {
  const component = modelIR.components.find((candidate) => candidate.id === entryPoint.componentId);
  if (!component) throw new Error(`Entry point "${entryPoint.id}" references an absent component.`);
  const componentReasons = semanticContractErrors({
    semantics: component.properties,
    contract: vocabulary.componentContracts?.[component.type],
    semanticsLabel: 'properties',
    contractLabel: `Component contract "${component.type}"`,
  });
  const reachableHeads = modelIR.outputHeads.filter((head) => head.componentId === component.id);
  const outputHeadReasons = [];
  if (['generate', 'speculative-generate'].includes(entryPoint.kind) && reachableHeads.length === 0) {
    outputHeadReasons.push(`Entry point "${entryPoint.id}" has no output head.`);
  }
  const outputHeads = reachableHeads.map((head) => {
    const reasons = semanticContractErrors({
      semantics: head.properties,
      contract: vocabulary.outputHeadContracts?.[head.kind],
      semanticsLabel: 'properties',
      contractLabel: `Output-head contract "${head.kind}"`,
    });
    return {
      outputHeadId: head.id,
      outputHeadKind: head.kind,
      compatible: reasons.length === 0,
      reasons,
    };
  });
  return {
    component: {
      componentId: component.id,
      componentType: component.type,
      compatible: componentReasons.length === 0,
      reasons: componentReasons,
    },
    outputHeads,
    outputHeadReasons,
  };
}

function loweringCompatibilityErrors(blockClass, lowering, phases) {
  const errors = [];
  if (!Array.isArray(lowering.blockKinds) || !lowering.blockKinds.includes(blockClass.kind)) {
    errors.push(`Lowering "${lowering.id}" cannot implement block kind "${blockClass.kind}".`);
    return errors;
  }
  for (const sectionName of ['geometry', 'normalization', 'positional', 'feedForward']) {
    errors.push(...semanticSectionErrors(blockClass, lowering, sectionName));
  }
  for (const phase of phases) {
    if (blockClass.phaseBehavior?.[phase] !== true) {
      errors.push(`Block class "${blockClass.id}" does not admit phase "${phase}".`);
    }
    const steps = lowering.phases?.[phase];
    if (!Array.isArray(steps) || steps.length === 0) {
      errors.push(`Lowering "${lowering.id}" has no ${phase} program.`);
    }
  }
  return errors;
}

export function auditEntryPointLowerability({ modelIR, entryPointId, vocabulary }) {
  const modelValidation = validateModelIR(modelIR);
  if (!modelValidation.ok || modelIR.schema !== 'doppler.model-ir/v2') {
    throw new Error(`Lowerability audit requires ModelIR v2: ${modelValidation.errors.join('; ')}`);
  }
  if (!isObject(vocabulary) || vocabulary.schema !== EXECUTION_CANDIDATE_FORGE_SCHEMA_ID) {
    throw new Error(`Lowerability audit requires vocabulary "${EXECUTION_CANDIDATE_FORGE_SCHEMA_ID}".`);
  }
  const entryPoint = resolveEntryPoint(modelIR, entryPointId, false);
  const schedule = reachableSchedule(modelIR, entryPoint.componentId);
  const phases = entryPointAuditPhases(entryPoint);
  const semanticContracts = entryPointSemanticContracts(modelIR, entryPoint, vocabulary);
  const classById = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  const blockClasses = [...new Set(schedule.blocks.map((block) => block.blockClassId))].map((blockClassId) => {
    const blockClass = classById.get(blockClassId);
    const compatibleLoweringIds = [];
    const rejectedLowerings = [];
    for (const lowering of vocabulary.blockLowerings || []) {
      const reasons = loweringCompatibilityErrors(blockClass, lowering, phases);
      if (reasons.length === 0) compatibleLoweringIds.push(lowering.id);
      else rejectedLowerings.push({ loweringId: lowering.id, reasons });
    }
    return {
      blockClassId,
      blockKind: blockClass.kind,
      compatibleLoweringIds,
      rejectedLowerings,
    };
  });
  const supportedStateKinds = new Set(vocabulary.supportedStateKinds || []);
  const unimplementedStateKinds = [...new Set(modelIR.stateSpaces
    .filter((state) => state.persistence === 'session')
    .map((state) => state.kind)
    .filter((kind) => !supportedStateKinds.has(kind)))].sort();
  const lowerable = blockClasses.every((entry) => entry.compatibleLoweringIds.length > 0)
    && unimplementedStateKinds.length === 0
    && semanticContracts.component.compatible
    && semanticContracts.outputHeads.every((head) => head.compatible)
    && semanticContracts.outputHeadReasons.length === 0;
  return {
    schema: 'doppler.entry-point-lowerability-audit/v1',
    modelId: modelIR.modelId,
    modelIRHash: hashModelIR(modelIR),
    entryPointId: entryPoint.id,
    entryPointStatus: entryPoint.status,
    requiredPhases: phases,
    scheduleId: schedule.id,
    lowerable,
    component: semanticContracts.component,
    outputHeads: semanticContracts.outputHeads,
    outputHeadReasons: semanticContracts.outputHeadReasons,
    blockClasses,
    unimplementedStateKinds,
  };
}

function compileProposal(modelIR, entryPoint, schedule, vocabulary, proposal) {
  requireAttribution(proposal);
  if (!Number.isFinite(proposal.score)) throw new Error(`Proposal "${proposal.id}" requires a finite score.`);
  const semanticContracts = entryPointSemanticContracts(modelIR, entryPoint, vocabulary);
  const semanticErrors = [
    ...semanticContracts.component.reasons,
    ...semanticContracts.outputHeadReasons,
    ...semanticContracts.outputHeads.flatMap((head) => head.reasons),
  ];
  if (semanticErrors.length > 0) throw new Error(semanticErrors.join(' '));
  const classById = new Map(modelIR.blockClasses.map((blockClass) => [blockClass.id, blockClass]));
  const loweringsById = new Map(vocabulary.blockLowerings.map((lowering) => [lowering.id, lowering]));
  const classPrograms = {};
  const kernelIds = [...(vocabulary.entryPointKernels?.[entryPoint.kind] || [])];
  for (const blockClassId of new Set(schedule.blocks.map((block) => block.blockClassId))) {
    const blockClass = classById.get(blockClassId);
    const loweringId = proposal.selections?.[blockClassId];
    const lowering = loweringsById.get(loweringId);
    if (!lowering) throw new Error(`Proposal "${proposal.id}" does not lower block class "${blockClassId}".`);
    const compatibilityErrors = loweringCompatibilityErrors(blockClass, lowering, entryPoint.phases);
    if (compatibilityErrors.length > 0) throw new Error(compatibilityErrors.join(' '));
    for (const phase of entryPoint.phases) {
      const steps = lowering.phases?.[phase];
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
