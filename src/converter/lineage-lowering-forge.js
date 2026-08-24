import { validateModelIR } from '../config/model-ir.js';
import { KERNEL_REF_CONTENT_DIGESTS } from '../config/kernels/kernel-ref-digests.js';
import { sanitizeModelId } from './core.js';
import { sha256Hex } from '../formats/sha256.js';
import { stableSortObject } from '../formats/stable-sort-object.js';

export const LINEAGE_LOWERING_FORGE_SCHEMA_ID = 'doppler.lineage-lowering-forge/v1';

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function clone(value) {
  return structuredClone(value);
}

function digest(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function pointerTokens(pointer) {
  if (typeof pointer !== 'string' || !pointer.startsWith('/')) {
    throw new Error(`Invalid JSON pointer "${String(pointer)}".`);
  }
  return pointer.slice(1).split('/').map((token) => token.replaceAll('~1', '/').replaceAll('~0', '~'));
}

function readPointer(root, pointer) {
  return pointerTokens(pointer).reduce((value, token) => {
    if ((!isObject(value) && !Array.isArray(value)) || !Object.hasOwn(value, token)) {
      throw new Error(`JSON pointer "${pointer}" is unresolved.`);
    }
    return value[token];
  }, root);
}

function writePointer(root, pointer, value) {
  const tokens = pointerTokens(pointer);
  const key = tokens.pop();
  const parent = tokens.reduce((target, token) => {
    if (!isObject(target[token]) && !Array.isArray(target[token])) target[token] = {};
    return target[token];
  }, root);
  parent[key] = clone(value);
}

function deletePointer(root, pointer) {
  const tokens = pointerTokens(pointer);
  const key = tokens.pop();
  const parent = tokens.reduce((target, token) => target?.[token], root);
  if (parent != null) delete parent[key];
}

function requireAuthor(value, label) {
  if (!isObject(value) || !['human', 'ai', 'tool'].includes(value.kind)
    || typeof value.actor !== 'string' || !value.actor.trim()) {
    throw new Error(`${label} requires attributable authorship.`);
  }
}

function resolveFact(factsById, factId) {
  const fact = factsById.get(factId);
  if (!fact || !['direct', 'derived'].includes(fact.confidence)
    || fact.disposition !== 'accepted' || fact.validation?.status !== 'passed') {
    throw new Error(`Lineage lowering requires accepted, validated fact "${factId}".`);
  }
  return fact;
}

function resolveFactValue(fact, valuePointer) {
  return valuePointer ? readPointer(fact.value, valuePointer) : fact.value;
}

function bindKernelDigests(value, dispositions) {
  if (!value || typeof value !== 'object') return;
  if (Array.isArray(value)) {
    value.forEach((entry) => bindKernelDigests(entry, dispositions));
    return;
  }
  if (typeof value.kernel === 'string' && typeof value.entry === 'string') {
    const key = `${value.kernel}#${value.entry}`;
    const resolved = KERNEL_REF_CONTENT_DIGESTS[key];
    if (typeof resolved !== 'string') throw new Error(`Kernel reference "${key}" has no canonical digest.`);
    value.digest = `sha256:${resolved}`;
    dispositions.push({
      kind: 'kernel-digest-binding',
      kernelRef: key,
      digest: value.digest,
      disposition: 'accepted',
    });
  }
  Object.values(value).forEach((entry) => bindKernelDigests(entry, dispositions));
}

export function materializeLineageConversionCandidate({ modelIR, template, recipe }) {
  const validation = validateModelIR(modelIR);
  if (!validation.ok || modelIR?.schema !== 'doppler.model-ir/v2') {
    throw new Error(`Lineage lowering requires ModelIR v2: ${validation.errors.join('; ')}`);
  }
  if (!isObject(template)) throw new Error('Lineage lowering requires a conversion template.');
  if (!isObject(recipe) || recipe.schema !== LINEAGE_LOWERING_FORGE_SCHEMA_ID) {
    throw new Error(`Lineage lowering requires recipe "${LINEAGE_LOWERING_FORGE_SCHEMA_ID}".`);
  }
  requireAuthor(recipe.author, 'Lineage recipe');
  const factsById = new Map(modelIR.provenance.facts.map((fact) => [fact.id, fact]));
  const config = clone(template);
  const dispositions = [];

  for (const requirement of recipe.compatibilityRequirements || []) {
    const fact = resolveFact(factsById, requirement.factId);
    const observed = resolveFactValue(fact, requirement.valuePointer);
    if (Object.hasOwn(requirement, 'equals')
      && JSON.stringify(observed) !== JSON.stringify(requirement.equals)) {
      throw new Error(`Lineage compatibility fact "${requirement.factId}" does not equal its required value.`);
    }
    if (Array.isArray(requirement.includes)) {
      const observedValues = Array.isArray(observed) ? new Set(observed) : new Set([observed]);
      for (const expected of requirement.includes) {
        if (!observedValues.has(expected)) {
          throw new Error(`Lineage compatibility fact "${requirement.factId}" does not include "${expected}".`);
        }
      }
    }
    dispositions.push({
      kind: 'compatibility-requirement',
      factId: requirement.factId,
      disposition: 'accepted',
      evidence: fact.evidence,
    });
  }

  for (const assertion of recipe.templateAssertions || []) {
    const fact = resolveFact(factsById, assertion.factId);
    const expected = resolveFactValue(fact, assertion.valuePointer);
    const observed = readPointer(config, assertion.targetPointer);
    if (JSON.stringify(observed) !== JSON.stringify(expected)) {
      throw new Error(
        `Template assertion "${assertion.targetPointer}" does not match fact "${assertion.factId}".`
      );
    }
    dispositions.push({
      kind: 'template-assertion',
      factId: assertion.factId,
      targetPointer: assertion.targetPointer,
      disposition: 'accepted',
      evidence: fact.evidence,
    });
  }

  for (const binding of recipe.factBindings || []) {
    const fact = resolveFact(factsById, binding.factId);
    const value = resolveFactValue(fact, binding.valuePointer);
    writePointer(config, binding.targetPointer, value);
    dispositions.push({
      kind: 'source-fact-binding',
      factId: binding.factId,
      targetPointer: binding.targetPointer,
      disposition: 'accepted',
      evidence: fact.evidence,
    });
  }

  for (const override of recipe.policyOverrides || []) {
    requireAuthor(override.author ?? recipe.author, `Policy override "${override.targetPointer}"`);
    if (!['forge', 'pack-scope', 'qualification'].includes(override.lifecycle)
      || typeof override.rationale !== 'string' || !override.rationale.trim()) {
      throw new Error(`Policy override "${override.targetPointer}" requires lifecycle and rationale.`);
    }
    writePointer(config, override.targetPointer, override.value);
    dispositions.push({
      kind: 'policy-override',
      targetPointer: override.targetPointer,
      lifecycle: override.lifecycle,
      disposition: 'accepted',
      author: override.author ?? recipe.author,
      rationale: override.rationale,
    });
  }

  for (const pointer of recipe.removePointers || []) deletePointer(config, pointer);

  bindKernelDigests(config, dispositions);

  config.output.modelBaseId = recipe.modelId;
  const artifactModelId = sanitizeModelId(recipe.modelId);
  if (!artifactModelId) {
    throw new Error(`Lineage lowering cannot derive an artifact model id from "${String(recipe.modelId)}".`);
  }
  dispositions.push({
    kind: 'artifact-identity-normalization',
    requestedModelId: recipe.modelId,
    modelId: artifactModelId,
    disposition: 'accepted',
  });
  config.manifest.artifactIdentity = {
    sourceCheckpointId: modelIR.sourceIdentity.checkpointId,
    sourceRepo: modelIR.sourceIdentity.repository,
    sourceRevision: modelIR.sourceIdentity.revision,
    artifactCompleteness: 'complete',
  };
  const packModelIR = { ...clone(modelIR), modelId: artifactModelId };
  const packModelIRValidation = validateModelIR(packModelIR);
  if (!packModelIRValidation.ok) {
    throw new Error(`Lineage lowering produced invalid Pack-bound ModelIR: ${packModelIRValidation.errors.join('; ')}`);
  }
  const configDigest = digest(config);
  return Object.freeze({
    schema: 'doppler.lineage-lowering-receipt/v1',
    modelId: artifactModelId,
    requestedModelId: recipe.modelId,
    sourceModelIRHash: digest(modelIR),
    modelIRHash: digest(packModelIR),
    modelIR: packModelIR,
    template: recipe.template,
    author: recipe.author,
    generatedCandidates: Number(recipe.candidateAudit?.generated || 1),
    rejectedCandidates: clone(recipe.candidateAudit?.rejected || []),
    acceptedCandidateId: recipe.candidateAudit?.acceptedCandidateId ?? 'lineage-conservative',
    dispositions,
    unresolvedFacts: [],
    conversionConfigDigest: configDigest,
    conversionConfig: config,
  });
}
