import assert from 'node:assert/strict';
import { createModelIRV2, hashModelIR, validateModelIR } from '../../src/config/model-ir.js';

const digest = `sha256:${'a'.repeat(64)}`;
const fact = {
  id: 'decoder.kind',
  subject: 'component.decoder',
  predicate: 'type',
  value: 'text-decoder',
  confidence: 'direct',
  disposition: 'accepted',
  evidence: [{ kind: 'json-pointer', artifactId: 'config', file: 'config.json', pointer: '/model_type' }],
  authorship: { kind: 'tool', actor: 'doppler.source-truth-forge/v2' },
  validation: { status: 'passed', validator: 'fixture', receipt: digest },
};
const node = (id, extra) => ({ id, factRefs: [fact.id], ...extra });
const modelIR = createModelIRV2({
  modelId: 'heterogeneous-fixture',
  sourceIdentity: {
    checkpointId: 'example/heterogeneous-fixture',
    repository: 'example/heterogeneous-fixture',
    revision: '0123456789abcdef',
    artifacts: [{ artifactId: 'config', path: 'config.json', role: 'source-config', hash: digest }],
  },
  provenance: { forgeVersion: '2.0.0', intakeDigest: digest, facts: [fact] },
  components: [node('decoder', { type: 'text-decoder', role: 'primary', properties: { hiddenSize: 8 } })],
  blockClasses: [
    node('full', {
      kind: 'full-attention', geometry: { heads: 2 }, normalization: { type: 'rmsnorm' },
      positional: { type: 'rope' }, feedForward: { type: 'dense' }, phaseBehavior: { prefill: true, decode: true },
    }),
    node('linear', {
      kind: 'linear-recurrent-attention', geometry: { keyHeads: 2 }, normalization: { type: 'rmsnorm' },
      positional: { type: 'none' }, feedForward: { type: 'dense' }, phaseBehavior: { prefill: true, decode: true },
    }),
  ],
  blockSchedules: [node('decoder.schedule', {
    componentId: 'decoder',
    blocks: [{ index: 0, blockClassId: 'linear' }, { index: 1, blockClassId: 'full' }],
  })],
  stateSpaces: [node('decoder.kv', { kind: 'kv', persistence: 'session', contract: { layout: 'contiguous' } })],
  tensorRoleBindings: [node('decoder.embedding', {
    componentId: 'decoder', role: 'token-embedding', selector: { exact: 'embed.weight' },
  })],
  entryPoints: [node('text.generate', {
    componentId: 'decoder', kind: 'generate', status: 'lowered', phases: ['prefill', 'decode'],
  })],
  outputHeads: [node('text.lm-head', { componentId: 'decoder', kind: 'causal-lm', properties: {} })],
  supportScope: {
    sourceTopology: 'complete',
    loweredEntryPoints: ['text.generate'],
    qualifiedEntryPoints: [],
    unloweredEntryPoints: [],
  },
});

assert.equal(validateModelIR(modelIR).ok, true);
assert.match(hashModelIR(modelIR), /^sha256:[0-9a-f]{64}$/);

for (const confidence of ['family-inferred', 'ambiguous', 'unsupported']) {
  const invalid = structuredClone(modelIR);
  invalid.provenance.facts[0].confidence = confidence;
  assert.equal(validateModelIR(invalid).ok, false, `${confidence} must fail closed`);
}
const missingComponent = structuredClone(modelIR);
missingComponent.blockSchedules[0].componentId = 'absent';
assert.equal(validateModelIR(missingComponent).ok, false);

console.log('✔ model-ir-v2.test.js passed');
