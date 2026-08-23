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

function sourceContent(source) {
  return isObject(source) && Object.hasOwn(source, 'content') ? source.content : source;
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
    const observed = resolveJsonPointer(sourceContent(source), evidence.pointer);
    if (!valuesEqual(observed, fact.value)) {
      throw new Error(`Fact "${fact.id}" does not match ${evidence.file}${evidence.pointer}.`);
    }
    return;
  }
  if (evidence.kind === 'tensor-header') {
    const observed = sourceContent(source)?.tensors?.[evidence.tensorName];
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

function materializeFactValues(value, factsById) {
  if (Array.isArray(value)) return value.map((entry) => materializeFactValues(entry, factsById));
  if (!isObject(value)) return value;
  if (Object.keys(value).length === 1 && typeof value.$fact === 'string') {
    const fact = factsById.get(value.$fact);
    if (!fact) throw new Error(`Topology references unknown fact "${value.$fact}".`);
    return fact.value;
  }
  return Object.fromEntries(Object.entries(value).map(([key, entry]) => [
    key,
    materializeFactValues(entry, factsById),
  ]));
}

function expandBlockSchedules(topology, factsById) {
  const materialized = materializeFactValues(topology, factsById);
  return {
    ...materialized,
    blockSchedules: materialized.blockSchedules.map((schedule) => {
      if (isObject(schedule.blocksFromCount)) {
        const { blocksFromCount, ...base } = schedule;
        const fact = factsById.get(blocksFromCount.factId);
        if (!fact || !Number.isInteger(fact.value) || fact.value < 1) {
          throw new Error(`Block schedule "${schedule.id}" requires a positive count fact.`);
        }
        return {
          ...base,
          blocks: Array.from({ length: fact.value }, (_, index) => ({
            index,
            blockClassId: blocksFromCount.blockClassId,
          })),
        };
      }
      if (!isObject(schedule.blocksFromFact)) return schedule;
      const { blocksFromFact, ...base } = schedule;
      const fact = factsById.get(blocksFromFact.factId);
      if (!fact || !Array.isArray(fact.value)) {
        throw new Error(`Block schedule "${schedule.id}" requires an array fact.`);
      }
      return {
        ...base,
        blocks: fact.value.map((blockType, index) => {
          const blockClassId = blocksFromFact.classByValue?.[blockType];
          if (typeof blockClassId !== 'string' || !blockClassId) {
            throw new Error(`Block schedule "${schedule.id}" has no class for "${blockType}".`);
          }
          return { index, blockClassId };
        }),
      };
    }),
  };
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
    const observed = isObject(source) && typeof source.hash === 'string'
      ? source.hash
      : canonicalDigest(sourceContent(source));
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
  const validatedFactsById = new Map(facts.map((fact) => [fact.id, fact]));
  const topology = expandBlockSchedules(packet.topology, validatedFactsById);
  const intakeDigest = canonicalDigest({
    sourceIdentity: packet.sourceIdentity,
    facts,
    topology,
  });
  const modelIR = createModelIRV2({
    modelId: packet.modelId,
    sourceIdentity: packet.sourceIdentity,
    provenance: { forgeVersion: SOURCE_TRUTH_FORGE_VERSION, intakeDigest, facts },
    ...topology,
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

export function createSourceTruthPacket(spec, sources) {
  if (!isObject(spec) || !Array.isArray(spec.facts)) {
    throw new Error('createSourceTruthPacket requires a fact specification.');
  }
  const facts = spec.facts.map((factSpec) => {
    const sourceSpec = factSpec.source;
    const evidence = sourceSpec?.kind === 'tensor-header'
      ? [{
        kind: 'tensor-header',
        artifactId: sourceSpec.artifactId,
        file: sourceSpec.file,
        tensorName: sourceSpec.tensorName,
        dtype: sourceContent(sources?.[sourceSpec.artifactId])?.tensors?.[sourceSpec.tensorName]?.dtype,
        shape: sourceContent(sources?.[sourceSpec.artifactId])?.tensors?.[sourceSpec.tensorName]?.shape,
      }]
      : sourceSpec
        ? [{
          kind: 'json-pointer',
          artifactId: sourceSpec.artifactId,
          file: sourceSpec.file,
          pointer: sourceSpec.pointer,
        }]
        : factSpec.evidence;
    const fact = {
      ...factSpec,
      confidence: factSpec.confidence ?? 'direct',
      disposition: factSpec.disposition ?? 'accepted',
      evidence,
      authorship: factSpec.authorship ?? spec.defaultAuthorship,
    };
    delete fact.source;
    if (Object.hasOwn(fact, 'value')) return { ...fact };
    const primaryEvidence = fact.evidence?.[0];
    const source = sources?.[primaryEvidence?.artifactId];
    if (source === undefined) throw new Error(`Fact "${fact.id}" source artifact is unavailable.`);
    if (fact.confidence === 'derived') return { ...fact, value: null };
    if (primaryEvidence?.kind === 'json-pointer') {
      return { ...fact, value: resolveJsonPointer(sourceContent(source), primaryEvidence.pointer) };
    }
    if (primaryEvidence?.kind === 'tensor-header') {
      const tensor = sourceContent(source)?.tensors?.[primaryEvidence.tensorName];
      if (!tensor) throw new Error(`Fact "${fact.id}" tensor is unavailable.`);
      return { ...fact, value: { shape: tensor.shape, dtype: tensor.dtype } };
    }
    throw new Error(`Fact "${fact.id}" cannot be derived from its evidence.`);
  });
  const factsById = new Map(facts.map((fact) => [fact.id, fact]));
  for (const fact of facts) {
    if (fact.confidence === 'derived') fact.value = resolveDerivedValue(fact, factsById);
  }
  return {
    schema: SOURCE_TRUTH_FORGE_SCHEMA_ID,
    modelId: spec.modelId,
    sourceIdentity: spec.sourceIdentity,
    facts,
    topology: spec.topology,
    candidateAudit: spec.candidateAudit,
  };
}
