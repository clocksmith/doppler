import assert from 'node:assert/strict';
import { forgeModelIRV2 } from '../../src/converter/source-truth-forge.js';
import { sha256Hex } from '../../src/utils/sha256.js';
import { stableSortObject } from '../../src/utils/stable-sort-object.js';

const digest = (value) => `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
const config = { model_type: 'fixture', layer_types: ['linear_attention', 'full_attention'], hidden_size: 8 };
const headers = { tensors: { 'embed.weight': { dtype: 'BF16', shape: [16, 8] } } };
const sourceIdentity = {
  checkpointId: 'example/fixture', repository: 'example/fixture', revision: '0123456789abcdef',
  artifacts: [
    { artifactId: 'config', path: 'config.json', role: 'source-config', hash: digest(config) },
    { artifactId: 'headers', path: 'headers.json', role: 'tensor-headers', hash: digest(headers) },
  ],
};
const directFact = (id, subject, predicate, value, pointer) => ({
  id, subject, predicate, value, confidence: 'direct', disposition: 'accepted',
  evidence: [{ kind: 'json-pointer', artifactId: 'config', file: 'config.json', pointer }],
  authorship: { kind: 'ai', actor: 'codex', proposalId: 'fixture-proposal' },
});
const facts = [
  directFact('component.type', 'component.decoder', 'type', 'fixture', '/model_type'),
  directFact('schedule.types', 'schedule.decoder', 'layerTypes', config.layer_types, '/layer_types'),
  directFact('decoder.hidden', 'component.decoder', 'hiddenSize', 8, '/hidden_size'),
  {
    id: 'tensor.embedding', subject: 'tensor.embed.weight', predicate: 'header',
    value: headers.tensors['embed.weight'], confidence: 'direct', disposition: 'accepted',
    evidence: [{
      kind: 'tensor-header', artifactId: 'headers', file: 'headers.json', tensorName: 'embed.weight',
      dtype: 'BF16', shape: [16, 8],
    }],
    authorship: { kind: 'tool', actor: 'safetensors-header-reader' },
  },
  {
    id: 'schedule.count', subject: 'schedule.decoder', predicate: 'blockCount', value: 2,
    confidence: 'derived', disposition: 'accepted', derivation: { operation: 'length', inputs: ['schedule.types'] },
    evidence: [{ kind: 'json-pointer', artifactId: 'config', file: 'config.json', pointer: '/layer_types' }],
    authorship: { kind: 'tool', actor: 'doppler.source-truth-forge/v2' },
  },
];
const refs = facts.map((fact) => fact.id);
const node = (id, extra) => ({ id, factRefs: refs, ...extra });
const packet = {
  schema: 'doppler.source-truth-forge/v2', modelId: 'fixture', sourceIdentity, facts,
  unresolvedFacts: [{
    id: 'attention.scale.application',
    subject: 'block.attention',
    predicate: 'scaleApplication',
    confidence: 'ambiguous',
    disposition: 'unresolved',
    reason: 'The config value does not establish the operational formula.',
    evidence: [{ kind: 'json-pointer', artifactId: 'config', file: 'config.json', pointer: '/hidden_size' }],
    authorship: { kind: 'ai', actor: 'codex', proposalId: 'fixture-proposal' },
  }],
  candidateAudit: { generated: 3, rejected: 2, acceptedProposalId: 'fixture-proposal' },
  topology: {
    components: [node('decoder', { type: 'text-decoder', role: 'primary', properties: { hiddenSize: 8 } })],
    blockClasses: [
      node('linear', {
        kind: 'linear-recurrent-attention', geometry: {}, normalization: {}, positional: {}, feedForward: {},
        phaseBehavior: { prefill: true, decode: true },
      }),
      node('full', {
        kind: 'full-attention', geometry: {}, normalization: {}, positional: {}, feedForward: {},
        phaseBehavior: { prefill: true, decode: true },
      }),
    ],
    blockSchedules: [node('decoder.schedule', {
      componentId: 'decoder', blocks: [{ index: 0, blockClassId: 'linear' }, { index: 1, blockClassId: 'full' }],
    })],
    stateSpaces: [node('decoder.recurrent', { kind: 'recurrent', persistence: 'session', contract: {} })],
    tensorRoleBindings: [node('decoder.embedding', {
      componentId: 'decoder', role: 'token-embedding', selector: { exact: 'embed.weight' },
    })],
    entryPoints: [node('text.generate', {
      componentId: 'decoder', kind: 'generate', status: 'lowered', phases: ['prefill', 'decode'],
    })],
    outputHeads: [node('text.head', { componentId: 'decoder', kind: 'causal-lm' })],
    supportScope: {
      sourceTopology: 'complete', loweredEntryPoints: ['text.generate'],
      qualifiedEntryPoints: [], unloweredEntryPoints: [],
    },
  },
};

const receipt = forgeModelIRV2(packet, { config, headers });
assert.equal(receipt.modelIR.schema, 'doppler.model-ir/v2');
assert.equal(receipt.generatedCandidates, 3);
assert.equal(receipt.rejectedCandidates, 2);
assert.equal(receipt.acceptedCandidates, 1);
assert.ok(receipt.modelIR.provenance.facts.every((fact) => fact.validation.status === 'passed'));
assert.equal(receipt.unresolvedFacts[0].validation.status, 'preserved-unresolved');
assert.equal(receipt.modelIR.provenance.facts.some((fact) => (
  fact.id === 'attention.scale.application'
)), false);

const invented = structuredClone(packet);
invented.facts[2].value = 16;
assert.throws(() => forgeModelIRV2(invented, { config, headers }), /does not match/);
const inferred = structuredClone(packet);
inferred.facts[0].confidence = 'family-inferred';
assert.throws(() => forgeModelIRV2(inferred, { config, headers }), /cannot enter signed ModelIR/);
const unattributed = structuredClone(packet);
delete unattributed.facts[0].authorship;
assert.throws(() => forgeModelIRV2(unattributed, { config, headers }), /attributable authorship/);
const promotedUnresolved = structuredClone(packet);
promotedUnresolved.unresolvedFacts[0].disposition = 'accepted';
assert.throws(() => forgeModelIRV2(promotedUnresolved, { config, headers }), /must be "unresolved"/);
const unsupportedUnresolvedEvidence = structuredClone(packet);
unsupportedUnresolvedEvidence.unresolvedFacts[0].evidence[0].kind = 'family-memory';
assert.throws(
  () => forgeModelIRV2(unsupportedUnresolvedEvidence, { config, headers }),
  /unsupported evidence kind/
);

console.log('✔ source-truth-forge.test.js passed');
