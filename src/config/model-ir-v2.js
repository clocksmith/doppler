export const MODEL_IR_V2_SCHEMA_ID = 'doppler.model-ir/v2';
export const MODEL_IR_V2_SCHEMA_VERSION = 2;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const ELIGIBLE_CONFIDENCE = new Set(['direct', 'derived']);
const AUTHOR_KINDS = new Set(['human', 'ai', 'tool']);

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function requireDigest(value, label, errors) {
  if (!SHA256_PATTERN.test(value || '')) errors.push(`${label} must be a SHA-256 digest.`);
}

function requireArray(value, label, errors) {
  if (!Array.isArray(value) || value.length === 0) {
    errors.push(`${label} must be a non-empty array.`);
    return false;
  }
  return true;
}

function validateEvidence(evidence, label, artifactIds, errors) {
  if (!isObject(evidence)) {
    errors.push(`${label} must be an object.`);
    return;
  }
  if (!['json-pointer', 'tensor-header'].includes(evidence.kind)) {
    errors.push(`${label}.kind must be "json-pointer" or "tensor-header".`);
  }
  requireString(evidence.artifactId, `${label}.artifactId`, errors);
  if (!artifactIds.has(evidence.artifactId)) {
    errors.push(`${label}.artifactId must reference sourceIdentity.artifacts.`);
  }
  requireString(evidence.file, `${label}.file`, errors);
  if (evidence.kind === 'json-pointer') requireString(evidence.pointer, `${label}.pointer`, errors);
  if (evidence.kind === 'tensor-header') {
    requireString(evidence.tensorName, `${label}.tensorName`, errors);
    requireString(evidence.dtype, `${label}.dtype`, errors);
    if (!Array.isArray(evidence.shape)
      || evidence.shape.some((dimension) => !Number.isInteger(dimension) || dimension < 1)) {
      errors.push(`${label}.shape must contain positive integer dimensions.`);
    }
  }
}

function validateFact(fact, label, artifactIds, errors) {
  if (!isObject(fact)) {
    errors.push(`${label} must be an object.`);
    return;
  }
  requireString(fact.id, `${label}.id`, errors);
  requireString(fact.subject, `${label}.subject`, errors);
  requireString(fact.predicate, `${label}.predicate`, errors);
  if (!Object.hasOwn(fact, 'value')) errors.push(`${label}.value is required.`);
  if (!ELIGIBLE_CONFIDENCE.has(fact.confidence)) {
    errors.push(`${label}.confidence must be "direct" or "derived" for signed ModelIR.`);
  }
  if (fact.disposition !== 'accepted') {
    errors.push(`${label}.disposition must be "accepted" for signed ModelIR.`);
  }
  if (!isObject(fact.authorship)) {
    errors.push(`${label}.authorship must be an object.`);
  } else {
    if (!AUTHOR_KINDS.has(fact.authorship.kind)) errors.push(`${label}.authorship.kind is invalid.`);
    requireString(fact.authorship.actor, `${label}.authorship.actor`, errors);
  }
  if (!isObject(fact.validation) || fact.validation.status !== 'passed') {
    errors.push(`${label}.validation.status must be "passed".`);
  } else {
    requireString(fact.validation.validator, `${label}.validation.validator`, errors);
    requireDigest(fact.validation.receipt, `${label}.validation.receipt`, errors);
  }
  if (requireArray(fact.evidence, `${label}.evidence`, errors)) {
    fact.evidence.forEach((entry, index) => validateEvidence(
      entry,
      `${label}.evidence[${index}]`,
      artifactIds,
      errors
    ));
  }
}

function validateFactRefs(value, label, factIds, errors) {
  if (!requireArray(value, label, errors)) return;
  for (const factId of value) {
    if (typeof factId !== 'string' || !factIds.has(factId)) {
      errors.push(`${label} must contain only IDs from provenance.facts.`);
    }
  }
}

function validateNodes(nodes, label, factIds, errors, validateNode) {
  if (!requireArray(nodes, label, errors)) return new Map();
  const byId = new Map();
  nodes.forEach((node, index) => {
    const nodeLabel = `${label}[${index}]`;
    if (!isObject(node)) {
      errors.push(`${nodeLabel} must be an object.`);
      return;
    }
    requireString(node.id, `${nodeLabel}.id`, errors);
    if (byId.has(node.id)) errors.push(`${label} contains duplicate id "${node.id}".`);
    byId.set(node.id, node);
    validateFactRefs(node.factRefs, `${nodeLabel}.factRefs`, factIds, errors);
    validateNode?.(node, nodeLabel);
  });
  return byId;
}

function validateSourceIdentity(sourceIdentity, errors) {
  if (!isObject(sourceIdentity)) {
    errors.push('sourceIdentity must be an object.');
    return new Set();
  }
  for (const field of ['checkpointId', 'repository', 'revision']) {
    requireString(sourceIdentity[field], `sourceIdentity.${field}`, errors);
  }
  const artifactIds = new Set();
  if (requireArray(sourceIdentity.artifacts, 'sourceIdentity.artifacts', errors)) {
    sourceIdentity.artifacts.forEach((artifact, index) => {
      const label = `sourceIdentity.artifacts[${index}]`;
      if (!isObject(artifact)) {
        errors.push(`${label} must be an object.`);
        return;
      }
      for (const field of ['artifactId', 'path', 'role']) {
        requireString(artifact[field], `${label}.${field}`, errors);
      }
      requireDigest(artifact.hash, `${label}.hash`, errors);
      if (artifactIds.has(artifact.artifactId)) {
        errors.push(`sourceIdentity.artifacts contains duplicate artifactId "${artifact.artifactId}".`);
      }
      artifactIds.add(artifact.artifactId);
    });
  }
  return artifactIds;
}

export function validateModelIRV2(ir) {
  const errors = [];
  if (!isObject(ir)) return { ok: false, errors: ['ModelIR must be a non-null object.'] };
  if (ir.schema !== MODEL_IR_V2_SCHEMA_ID) errors.push(`schema must be "${MODEL_IR_V2_SCHEMA_ID}".`);
  if (ir.schemaVersion !== MODEL_IR_V2_SCHEMA_VERSION) {
    errors.push(`schemaVersion must be ${MODEL_IR_V2_SCHEMA_VERSION}.`);
  }
  requireString(ir.modelId, 'modelId', errors);
  const artifactIds = validateSourceIdentity(ir.sourceIdentity, errors);
  if (!isObject(ir.provenance)) {
    errors.push('provenance must be an object.');
    return { ok: false, errors };
  }
  requireString(ir.provenance.forgeVersion, 'provenance.forgeVersion', errors);
  requireDigest(ir.provenance.intakeDigest, 'provenance.intakeDigest', errors);
  const factIds = new Set();
  if (requireArray(ir.provenance.facts, 'provenance.facts', errors)) {
    ir.provenance.facts.forEach((fact, index) => {
      validateFact(fact, `provenance.facts[${index}]`, artifactIds, errors);
      if (factIds.has(fact?.id)) errors.push(`provenance.facts contains duplicate id "${fact.id}".`);
      factIds.add(fact?.id);
    });
  }
  const components = validateNodes(ir.components, 'components', factIds, errors, (component, label) => {
    requireString(component.type, `${label}.type`, errors);
    requireString(component.role, `${label}.role`, errors);
    if (!isObject(component.properties)) errors.push(`${label}.properties must be an object.`);
  });
  const blockClasses = validateNodes(ir.blockClasses, 'blockClasses', factIds, errors, (block, label) => {
    requireString(block.kind, `${label}.kind`, errors);
    for (const field of ['geometry', 'normalization', 'positional', 'feedForward', 'phaseBehavior']) {
      if (!isObject(block[field])) errors.push(`${label}.${field} must be an object.`);
    }
  });
  const schedules = validateNodes(ir.blockSchedules, 'blockSchedules', factIds, errors, (schedule, label) => {
    if (!components.has(schedule.componentId)) errors.push(`${label}.componentId must reference components.`);
    if (requireArray(schedule.blocks, `${label}.blocks`, errors)) {
      const indices = new Set();
      schedule.blocks.forEach((block, index) => {
        const blockLabel = `${label}.blocks[${index}]`;
        if (!isObject(block) || !Number.isInteger(block.index) || block.index < 0) {
          errors.push(`${blockLabel}.index must be a non-negative integer.`);
        } else if (indices.has(block.index)) {
          errors.push(`${label}.blocks contains duplicate index ${block.index}.`);
        }
        indices.add(block?.index);
        if (!blockClasses.has(block?.blockClassId)) {
          errors.push(`${blockLabel}.blockClassId must reference blockClasses.`);
        }
      });
    }
  });
  validateNodes(ir.stateSpaces, 'stateSpaces', factIds, errors, (state, label) => {
    requireString(state.kind, `${label}.kind`, errors);
    requireString(state.persistence, `${label}.persistence`, errors);
    if (!isObject(state.contract)) errors.push(`${label}.contract must be an object.`);
  });
  validateNodes(ir.tensorRoleBindings, 'tensorRoleBindings', factIds, errors, (binding, label) => {
    if (!components.has(binding.componentId)) errors.push(`${label}.componentId must reference components.`);
    requireString(binding.role, `${label}.role`, errors);
    if (!isObject(binding.selector)) errors.push(`${label}.selector must be an object.`);
  });
  const entryPoints = validateNodes(ir.entryPoints, 'entryPoints', factIds, errors, (entryPoint, label) => {
    if (!components.has(entryPoint.componentId)) errors.push(`${label}.componentId must reference components.`);
    requireString(entryPoint.kind, `${label}.kind`, errors);
    if (!['lowered', 'unlowered'].includes(entryPoint.status)) {
      errors.push(`${label}.status must be "lowered" or "unlowered".`);
    }
    if (!Array.isArray(entryPoint.phases)) errors.push(`${label}.phases must be an array.`);
    if (entryPoint.status === 'unlowered') requireString(entryPoint.reason, `${label}.reason`, errors);
  });
  validateNodes(ir.outputHeads, 'outputHeads', factIds, errors, (head, label) => {
    if (!components.has(head.componentId)) errors.push(`${label}.componentId must reference components.`);
    requireString(head.kind, `${label}.kind`, errors);
  });
  if (!isObject(ir.supportScope)) {
    errors.push('supportScope must be an object.');
  } else {
    if (ir.supportScope.sourceTopology !== 'complete') {
      errors.push('supportScope.sourceTopology must be "complete".');
    }
    for (const field of ['loweredEntryPoints', 'qualifiedEntryPoints', 'unloweredEntryPoints']) {
      if (!Array.isArray(ir.supportScope[field])) {
        errors.push(`supportScope.${field} must be an array.`);
      } else if (ir.supportScope[field].some((id) => !entryPoints.has(id))) {
        errors.push(`supportScope.${field} must reference entryPoints.`);
      }
    }
  }
  if (!schedules.size) errors.push('ModelIR v2 requires at least one block schedule.');
  return { ok: errors.length === 0, errors };
}

export function createModelIRV2(params) {
  if (!isObject(params)) throw new Error('createModelIRV2 requires an object.');
  const ir = {
    schema: MODEL_IR_V2_SCHEMA_ID,
    schemaVersion: MODEL_IR_V2_SCHEMA_VERSION,
    modelId: params.modelId,
    sourceIdentity: params.sourceIdentity,
    provenance: params.provenance,
    components: params.components,
    blockClasses: params.blockClasses,
    blockSchedules: params.blockSchedules,
    stateSpaces: params.stateSpaces,
    tensorRoleBindings: params.tensorRoleBindings,
    entryPoints: params.entryPoints,
    outputHeads: params.outputHeads,
    supportScope: params.supportScope,
  };
  const validation = validateModelIRV2(ir);
  if (!validation.ok) throw new Error(`Failed to create valid ModelIR v2: ${validation.errors.join('; ')}`);
  return ir;
}
