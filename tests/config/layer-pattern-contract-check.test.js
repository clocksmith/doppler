import assert from 'node:assert/strict';

import { buildLayerPatternContractArtifact } from '../../src/rules/layer-pattern-contract-check.js';
import { getInferenceLayerPatternContractArtifact } from '../../src/rules/rule-registry.js';

{
  const artifact = getInferenceLayerPatternContractArtifact();
  assert.equal(artifact.ok, true);
  assert.equal(artifact.stats.patternKindContexts, 16);
  assert.equal(artifact.stats.layerTypeContexts, 12);
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.layerPattern.patternKind.semantics' && entry.ok)
  );
  assert.ok(
    artifact.checks.some((entry) => entry.id === 'inference.layerPattern.computeGlobalLayers.every_n_negative_offset' && entry.ok)
  );
}

{
  const artifact = buildLayerPatternContractArtifact({
    patternKind: [
      { match: { patternType: 'alternating', globalPattern: 'odd' }, value: 'alternating_even' },
      { match: { patternType: 'alternating', globalPattern: 'odd' }, value: 'alternating_odd' },
      { match: { patternType: 'every_n' }, value: 'every_n' },
      { match: {}, value: null },
    ],
    layerType: [
      { match: { patternKind: 'alternating_even', isEven: true }, value: 'full_attention' },
      { match: { patternKind: 'alternating_even' }, value: 'sliding_attention' },
      { match: { patternKind: 'alternating_odd', isEven: false }, value: 'full_attention' },
      { match: { patternKind: 'alternating_odd' }, value: 'sliding_attention' },
      { match: { patternKind: 'every_n', isStride: true }, value: 'full_attention' },
      { match: { patternKind: 'every_n' }, value: 'sliding_attention' },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) => message.includes('patternKind rule[0] drifted'))
  );
}

{
  const artifact = buildLayerPatternContractArtifact({
    patternKind: [
      { match: { patternType: 'alternating', globalPattern: 'even' }, value: 'alternating_even' },
      { match: { patternType: 'alternating', globalPattern: 'odd' }, value: 'alternating_odd' },
      { match: { patternType: 'every_n' }, value: 'every_n' },
      { match: {}, value: null },
    ],
    layerType: [
      { match: { patternKind: 'alternating_even', isEven: true }, value: 'sliding_attention' },
      { match: { patternKind: 'alternating_even' }, value: 'sliding_attention' },
      { match: { patternKind: 'alternating_odd', isEven: false }, value: 'full_attention' },
      { match: { patternKind: 'alternating_odd' }, value: 'sliding_attention' },
      { match: { patternKind: 'every_n', isStride: true }, value: 'full_attention' },
      { match: { patternKind: 'every_n' }, value: 'sliding_attention' },
    ],
  });

  assert.equal(artifact.ok, false);
  assert.ok(
    artifact.errors.some((message) => message.includes('layerType rule[0] drifted') || message.includes('layerType mismatched context'))
  );
}

console.log('layer-pattern-contract-check.test: ok');
