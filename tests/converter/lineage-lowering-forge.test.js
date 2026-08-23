import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { materializeLineageConversionCandidate } from '../../src/converter/lineage-lowering-forge.js';

const modelIR = JSON.parse(await fs.readFile(
  'reports/model-ir-v2/qwen3.8-27b.model-ir-receipt.json',
  'utf8'
)).modelIR;
const template = {
  output: { modelBaseId: 'old' },
  manifest: { artifactIdentity: { sourceRevision: 'untrusted', shardSetHash: 'untrusted' } },
  inference: { attention: { scalar: 0 }, layerPattern: { layerTypes: [] } },
};
const recipe = {
  schema: 'doppler.lineage-lowering-forge/v1',
  modelId: 'candidate',
  template: 'fixture',
  author: { kind: 'ai', actor: 'test-agent' },
  compatibilityRequirements: [{ factId: 'text.layerTypes', includes: ['linear_attention', 'full_attention'] }],
  factBindings: [
    { factId: 'full.headDim', targetPointer: '/inference/attention/scalar' },
    { factId: 'text.layerTypes', targetPointer: '/inference/layerPattern/layerTypes' },
  ],
  policyOverrides: [{
    targetPointer: '/output/textOnly', value: true, lifecycle: 'pack-scope',
    rationale: 'Qualify only the lowered text entry point.',
  }],
  removePointers: ['/manifest/artifactIdentity/shardSetHash'],
  candidateAudit: {
    generated: 2,
    rejected: [{ candidateId: 'incomplete', reason: 'Missing recurrent state.' }],
    acceptedCandidateId: 'conservative',
  },
};
const receipt = materializeLineageConversionCandidate({ modelIR, template, recipe });
assert.equal(receipt.conversionConfig.inference.attention.scalar, 256);
assert.equal(receipt.conversionConfig.inference.layerPattern.layerTypes.length, 64);
assert.equal(receipt.conversionConfig.output.textOnly, true);
assert.equal(receipt.conversionConfig.manifest.artifactIdentity.sourceRevision, modelIR.sourceIdentity.revision);
assert.equal('shardSetHash' in receipt.conversionConfig.manifest.artifactIdentity, false);
assert.equal(receipt.generatedCandidates, 2);
assert.equal(receipt.rejectedCandidates.length, 1);
assert.equal(receipt.unresolvedFacts.length, 0);

assert.throws(() => materializeLineageConversionCandidate({
  modelIR,
  template,
  recipe: {
    ...recipe,
    compatibilityRequirements: [{ factId: 'text.layerTypes', includes: ['moe'] }],
  },
}), /does not include/);

console.log('✔ lineage-lowering-forge.test.js passed');
