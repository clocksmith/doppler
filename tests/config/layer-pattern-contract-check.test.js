import assert from 'node:assert/strict';

import { buildLayerPatternContractArtifact } from '../../src/rules/layer-pattern-contract-check.js';
import { getInferenceLayerPatternContractArtifact } from '../../src/rules/rule-registry.js';
import { computeGlobalLayers } from '../../src/config/schema/inference.schema.js';
import ruleBundle from '../../src/rules/generated/rule-bundle.json' with { type: 'json' };

const checkedIn = ruleBundle.files['inference/layer-pattern.rules.json'];

function assertRejected(ruleFile, fragment) {
  const artifact = buildLayerPatternContractArtifact(ruleFile);
  assert.equal(artifact.ok, false, `expected rejection containing "${fragment}"`);
  assert.ok(
    artifact.errors.some((message) => message.includes(fragment)),
    `expected an error containing "${fragment}"; got ${JSON.stringify(artifact.errors)}`
  );
}

{
  const artifact = getInferenceLayerPatternContractArtifact();
  assert.equal(artifact.ok, true, artifact.errors.join(' | '));
  assert.ok(artifact.checks.every((entry) => entry.ok));
  assert.deepEqual(artifact, buildLayerPatternContractArtifact(checkedIn));
}

assertRejected(
  { ...checkedIn, patternKind: checkedIn.patternKind.slice(0, -1) },
  'patternKind must end with a catch-all rule'
);
assertRejected(
  { ...checkedIn, patternKind: [...checkedIn.patternKind.slice(0, -1), { match: { patternTyp: 'x' }, value: 'x' }, { match: {}, value: null }] },
  'patternKind[3].match.patternTyp is not a context field'
);
assertRejected(
  { ...checkedIn, layerType: [{ match: { patternKind: 'unproduced_kind' }, value: 'full_attention' }, ...checkedIn.layerType] },
  'layerType[0].match.patternKind must be a kind produced by patternKind'
);
{
  const kind = checkedIn.patternKind[0].value;
  assertRejected(
    { ...checkedIn, layerType: checkedIn.layerType.filter((rule) => !(rule.match.patternKind === kind && Object.keys(rule.match).length === 1)) },
    `layerType must end pattern kind "${kind}"`
  );
}
assertRejected(
  { ...checkedIn, layerType: [{ match: { patternKind: checkedIn.patternKind[0].value }, value: 7 }, ...checkedIn.layerType] },
  'layerType[0].value must be a non-empty string'
);

// computeGlobalLayers normalizes layer pattern offsets.
assert.deepEqual(computeGlobalLayers({ type: 'alternating', globalPattern: 'even' }, 6), [0, 2, 4]);
assert.deepEqual(computeGlobalLayers({ type: 'alternating', globalPattern: 'odd' }, 6), [1, 3, 5]);
assert.deepEqual(computeGlobalLayers({ type: 'every_n', period: 6, offset: 5 }, 12), [5, 11]);
assert.deepEqual(computeGlobalLayers({ type: 'every_n', period: 6, offset: -1 }, 12), [5, 11]);

console.log('layer-pattern-contract-check.test: ok');
