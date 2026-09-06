import { auditEntryPointLowerability } from './execution-candidate-forge.js';
import { computeCanonicalSha256 } from '../formats/canonical-hash.js';

export function assessModelSupport({ modelIR, entryPointIds, vocabulary, unresolvedFacts }) {
  if (!Array.isArray(entryPointIds) || entryPointIds.length === 0
    || new Set(entryPointIds).size !== entryPointIds.length
    || Array.from(entryPointIds).some((id) => typeof id !== 'string' || !id.trim() || id.trim() !== id)) {
    throw new Error('Model support assessment requires unique, explicit entryPointIds.');
  }
  if (!Array.isArray(unresolvedFacts)) throw new Error('Model support assessment requires unresolvedFacts.');
  const audits = entryPointIds.map((entryPointId) => auditEntryPointLowerability({ modelIR, entryPointId, vocabulary }));
  const factsById = new Map(modelIR.provenance.facts.map((fact) => [fact.id, fact]));
  const tasks = [];
  function addTask(kind, node, reasons, entryPointId, comparison) {
    const facts = (node?.factRefs || []).map((id) => factsById.get(id)).filter(Boolean);
    tasks.push({
      kind, entryPointId, nodeId: node?.id ?? null, reasons,
      sourceEvidence: facts.flatMap((fact) => fact.evidence),
      semanticRequirement: node ?? null,
      regression: { entryPointId, modelIRHash: audits[0].modelIRHash, comparison },
    });
  }
  for (const fact of unresolvedFacts) {
    tasks.push({ kind: 'source-evidence', factId: fact.id, reasons: [fact.reason], sourceEvidence: fact.evidence });
  }
  for (const audit of audits) {
    if (!audit.component.compatible) {
      addTask('component-semantics', modelIR.components.find((node) => node.id === audit.component.componentId),
        audit.component.reasons, audit.entryPointId, audit.component);
    }
    for (const head of audit.outputHeads.filter((entry) => !entry.compatible)) {
      addTask('output-semantics', modelIR.outputHeads.find((node) => node.id === head.outputHeadId),
        head.reasons, audit.entryPointId, head);
    }
    if (audit.outputHeadReasons.length) addTask('output-head', null, audit.outputHeadReasons, audit.entryPointId, null);
    for (const block of audit.blockClasses.filter((entry) => entry.compatibleLoweringIds.length === 0)) {
      addTask('block-semantics', modelIR.blockClasses.find((node) => node.id === block.blockClassId),
        block.rejectedLowerings.length ? block.rejectedLowerings.flatMap((entry) => entry.reasons)
          : [`No lowering is declared for block kind "${block.blockKind}".`], audit.entryPointId, block);
    }
    for (const kind of audit.unimplementedStateKinds) {
      for (const state of modelIR.stateSpaces.filter((node) => node.kind === kind && node.persistence === 'session')) {
        addTask('state-lifecycle', state, [`Session state kind "${kind}" is not implemented by this vocabulary.`],
          audit.entryPointId, { supportedStateKinds: vocabulary.supportedStateKinds });
      }
    }
  }
  const report = {
    schema: 'doppler.model-support-assessment/v1',
    sourceIdentity: modelIR.sourceIdentity,
    modelIRHash: audits[0].modelIRHash,
    vocabularyDigest: computeCanonicalSha256(vocabulary),
    implementationClass: tasks.length ? 'missing-behavior-or-evidence' : 'known-operations-require-recipe',
    audits, tasks,
    qualified: false,
    nextRequirement: tasks.length ? 'Resolve the retained semantic/source gaps without relaxing reference acceptance.'
      : 'Materialize an explicit recipe and compare actual inference against independent source references.',
  };
  return { ...report, digest: computeCanonicalSha256(report) };
}
