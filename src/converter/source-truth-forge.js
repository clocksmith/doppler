import { createModelIRV2, validateModelIRV2 } from '../config/model-ir-v2.js';
import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const SOURCE_TRUTH_FORGE_SCHEMA_ID = 'doppler.source-truth-forge/v2';
export const SOURCE_TRUTH_FORGE_VERSION = '2.0.0';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function canonicalDigest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function resolveJsonPointer(root, pointer) {
  if (pointer === '') return root;
  if (typeof pointer !== 'string' || !pointer.startsWith('/')) {
    throw new Error(`Invalid JSON pointer "${pointer}".`);
  }
  return pointer.slice(1).split('/').reduce((value, token) => {
    const key = token.replaceAll('~1', '/').replaceAll('~0', '~');
    if (!isObject(value) && !Array.isArray(value)) throw new Error(`JSON pointer "${pointer}" is unresolved.`);
    if (!Object.hasOwn(value, key)) throw new Error(`JSON pointer "${pointer}" is unresolved.`);
    return value[key];
  }, root);
}

function valuesEqual(left, right) {
  return JSON.stringify(stableSortObject(left)) === JSON.stringify(stableSortObject(right));
}

function validateDirectEvidence(fact, evidence, sources) {
  const source = sources[evidence.artifactId];
  if (!source) throw new Error(`Fact "${fact.id}" references unavailable artifact "${evidence.artifactId}".`);
  if (evidence.kind === 'json-pointer') {
    const observed = resolveJsonPointer(source, evidence.pointer);
    if (!valuesEqual(observed, fact.value)) {
      throw new Error(`Fact "${fact.id}" does not match ${evidence.file}${evidence.pointer}.`);
    }
    return;
  }
  if (evidence.kind === 'tensor-header') {
    const observed = source.tensors?.[evidence.tensorName];
    if (!observed) throw new Error(`Fact "${fact.id}" tensor "${evidence.tensorName}" is absent.`);
    if (!valuesEqual(observed.shape, evidence.shape) || observed.dtype !== evidence.dtype) {
      throw new Error(`Fact "${fact.id}" tensor-header evidence does not match its source.`);
    }
    if (!valuesEqual(fact.value, { shape: observed.shape, dtype: observed.dtype })) {
      throw new Error(`Fact "${fact.id}" value does not match its tensor header.`);
    }
    return;
  }
  throw new Error(`Fact "${fact.id}" has unsupported evidence kind "${evidence.kind}".`);
}

function resolveDerivedValue(fact, factsById) {
  const derivation = fact.derivation;
  if (!isObject(derivation) || !Array.isArray(derivation.inputs) || derivation.inputs.length === 0) {
    throw new Error(`Derived fact "${fact.id}" requires derivation inputs.`);
  }
  const inputs = derivation.inputs.map((id) => {
    if (!factsById.has(id)) throw new Error(`Derived fact "${fact.id}" references unknown fact "${id}".`);
    return factsById.get(id).value;
  });
  if (derivation.operation === 'length') return inputs[0]?.length;
  if (derivation.operation === 'product') return inputs.reduce((total, value) => total * Number(value), 1);
  if (derivation.operation === 'unique') return [...new Set(inputs[0])];
  throw new Error(`Derived fact "${fact.id}" uses unsupported operation "${derivation.operation}".`);
}

function validateFact(fact, factsById, sources) {
  if (!isObject(fact)) throw new Error('Source-truth facts must be objects.');
  if (!['direct', 'derived'].includes(fact.confidence)) {
    throw new Error(`Fact "${fact.id}" confidence "${fact.confidence}" cannot enter signed ModelIR.`);
  }
  if (fact.disposition !== 'accepted') {
    throw new Error(`Fact "${fact.id}" disposition "${fact.disposition}" cannot enter signed ModelIR.`);
  }
  if (!isObject(fact.authorship) || !['human', 'ai', 'tool'].includes(fact.authorship.kind)
    || typeof fact.authorship.actor !== 'string' || !fact.authorship.actor.trim()) {
    throw new Error(`Fact "${fact.id}" requires attributable authorship.`);
  }
  if (!Array.isArray(fact.evidence) || fact.evidence.length === 0) {
    throw new Error(`Fact "${fact.id}" requires source evidence.`);
  }
  if (fact.confidence === 'direct') {
    fact.evidence.forEach((evidence) => validateDirectEvidence(fact, evidence, sources));
  } else {
    const observed = resolveDerivedValue(fact, factsById);
    if (!valuesEqual(observed, fact.value)) throw new Error(`Derived fact "${fact.id}" failed recomputation.`);
  }
  const validationCore = {
    id: fact.id,
    value: fact.value,
    evidence: fact.evidence,
    derivation: fact.derivation ?? null,
  };
  return {
    ...fact,
    validation: {
      status: 'passed',
      validator: 'doppler.source-truth-forge/v2',
      receipt: canonicalDigest(validationCore),
    },
  };
}

function verifyArtifactDigests(sourceIdentity, sources) {
  for (const artifact of sourceIdentity.artifacts || []) {
    const source = sources[artifact.artifactId];
    if (source === undefined) throw new Error(`Source artifact "${artifact.artifactId}" is unavailable.`);
    const observed = canonicalDigest(source);
    if (artifact.hash !== observed) {
      throw new Error(`Source artifact "${artifact.artifactId}" hash mismatch: expected ${artifact.hash}, got ${observed}.`);
    }
  }
}

export function forgeModelIRV2(packet, sources) {
  if (!isObject(packet) || packet.schema !== SOURCE_TRUTH_FORGE_SCHEMA_ID) {
    throw new Error(`Source-Truth Forge requires schema "${SOURCE_TRUTH_FORGE_SCHEMA_ID}".`);
  }
  if (!isObject(sources)) throw new Error('Source-Truth Forge requires source artifacts.');
  verifyArtifactDigests(packet.sourceIdentity, sources);
  const factsById = new Map((packet.facts || []).map((fact) => [fact.id, fact]));
  if (factsById.size !== packet.facts?.length) throw new Error('Source-truth fact IDs must be unique.');
  const facts = packet.facts.map((fact) => validateFact(fact, factsById, sources));
  const intakeDigest = canonicalDigest({
    sourceIdentity: packet.sourceIdentity,
    facts,
    topology: packet.topology,
  });
  const modelIR = createModelIRV2({
    modelId: packet.modelId,
    sourceIdentity: packet.sourceIdentity,
    provenance: { forgeVersion: SOURCE_TRUTH_FORGE_VERSION, intakeDigest, facts },
    ...packet.topology,
  });
  const validation = validateModelIRV2(modelIR);
  if (!validation.ok) throw new Error(`Forged ModelIR v2 failed validation: ${validation.errors.join('; ')}`);
  return Object.freeze({
    schema: 'doppler.source-truth-forge-receipt/v2',
    modelIR,
    intakeDigest,
    unresolvedFacts: [],
    generatedCandidates: Number(packet.candidateAudit?.generated || 1),
    rejectedCandidates: Number(packet.candidateAudit?.rejected || 0),
    acceptedCandidates: 1,
    acceptedProposalId: packet.candidateAudit?.acceptedProposalId ?? 'source-truth-baseline',
  });
}
