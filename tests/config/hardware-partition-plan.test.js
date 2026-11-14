import assert from 'node:assert/strict';
import { test } from 'node:test';
import { proposeHardwarePartitionPlan } from '../../src/config/hardware-partition-plan.js';
import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';
import { validateOrderedLayerPartitionPlan, hashLayerPartitionPlan } from '../../src/config/layer-partition-plan.js';

const manifest = { modelId: 'fixture', architecture: { numLayers: 4, hiddenSize: 8, vocabSize: 16 } };
function request() {
  const workloadIdentity = 'sha256:' + '1'.repeat(64);
  return { workloadIdentity, nowMs: 100, maxProfileAgeMs: 10, maxLayers: 4,
    activation: { activationDtype: 'f32', activationPrecision: 'f32' }, transferMs: [1, 1, 1],
    profiles: ['fast', 'slow'].map((hardwareId, index) => ({ hardwareId, workloadIdentity,
      manifestHash: computeCanonicalSha256(manifest), sourceReceipt: `fixture://${hardwareId}`, measuredAtMs: 95,
      memoryBudgetBytes: 100, fixedBytes: 10, fixedMs: 2,
      layers: Array.from({ length: 4 }, () => ({ residentBytes: 10, scratchBytes: 5, elapsedMs: index ? 10 : 1 })) })) };
}
test('chooses a non-midpoint split and emits unchanged canonical contract', () => {
  const input = request(), before = JSON.stringify(input);
  const result = proposeHardwarePartitionPlan(manifest, input);
  assert.deepEqual(result.plan.splitLayers, [3]);
  assert.equal(result.estimatedMs, 18);
  assert.deepEqual(result.memoryBytes, [45, 25]);
  assert.equal(result.evaluatedSplits, 3);
  assert.equal(result.feasibleSplits, 3);
  assert.deepEqual(validateOrderedLayerPartitionPlan(manifest, result.plan), result.plan);
  assert.equal(hashLayerPartitionPlan(result.plan), hashLayerPartitionPlan(validateOrderedLayerPartitionPlan(manifest, result.plan)));
  assert.equal(JSON.stringify(input), before);
  assert.ok(Object.isFrozen(result) && Object.isFrozen(result.memoryBytes));
});
test('memory caps include fixed reserves and peak scratch rather than weights alone', () => {
  const input = request(); input.profiles[0].memoryBudgetBytes = 35;
  assert.deepEqual(proposeHardwarePartitionPlan(manifest, input).plan.splitLayers, [2]);
  input.profiles[0].layers[1].scratchBytes = 30;
  assert.deepEqual(proposeHardwarePartitionPlan(manifest, input).plan.splitLayers, [1]);
  input.profiles[1].memoryBudgetBytes = 20;
  assert.throws(() => proposeHardwarePartitionPlan(manifest, input), /No feasible/);
});
test('boundary transport costs change the selected cut; exact ties choose earliest', () => {
  const input = request(); input.transferMs[2] = 100;
  assert.deepEqual(proposeHardwarePartitionPlan(manifest, input).plan.splitLayers, [2]);
  input.profiles[1].layers.forEach(layer => { layer.elapsedMs = 1; }); input.transferMs = [1, 1, 1];
  assert.deepEqual(proposeHardwarePartitionPlan(manifest, input).plan.splitLayers, [1]);
});
test('fails closed on missing, stale, foreign or nonfinite profile evidence', () => {
  for (const mutate of [
    x => { x.profiles[0].manifestHash = 'other'; }, x => { x.profiles[0].workloadIdentity = 'other'; },
    x => { x.profiles[0].measuredAtMs = 89; }, x => { x.profiles[0].measuredAtMs = 101; },
    x => { x.profiles[0].sourceReceipt = ''; }, x => { x.profiles[0].layers.pop(); },
    x => { x.profiles[0].layers[0].elapsedMs = NaN; }, x => { x.profiles[0].layers[0].residentBytes = -1; },
    x => { x.profiles[0].layers[0].scratchBytes = 0.5; }, x => { x.transferMs = [1]; },
    x => { x.maxLayers = 3; }, x => { x.profiles[1].hardwareId = 'fast'; },
  ]) { const input = request(); mutate(input); assert.throws(() => proposeHardwarePartitionPlan(manifest, input)); }
});
test('preserves hyperconnection activation geometry and adapter identity', () => {
  const model = { ...manifest, inference: { hyperConnections: { streamCount: 4, streamWidth: 8 } } };
  const input = request(); input.profiles.forEach(profile => { profile.manifestHash = computeCanonicalSha256(model); });
  input.activation.adapterIdentities = ['sha256:' + '2'.repeat(64)];
  const result = proposeHardwarePartitionPlan(model, input);
  assert.equal(result.plan.partitions[0].outputContract.hiddenSize, 32);
  assert.deepEqual(result.plan.adapterIdentities, input.activation.adapterIdentities);
});
test('rejects numeric overflow instead of accepting corrupt scores', () => {
  const input = request(); input.profiles[0].layers.forEach(layer => { layer.elapsedMs = Number.MAX_VALUE; });
  assert.throws(() => proposeHardwarePartitionPlan(manifest, input), /overflows/);
});
test('matches simple exhaustive oracle across heterogeneous memory, timing and transfer profiles', () => {
  for (let trial = 0; trial < 20; trial++) {
    const input = request();
    input.profiles.forEach((profile, device) => {
      profile.memoryBudgetBytes = 35 + (trial % 4) * 10;
      profile.layers.forEach((layer, index) => {
        layer.elapsedMs = 1 + ((trial + index * 3 + device * 7) % 11);
        layer.residentBytes = 3 + ((trial + index + device) % 8);
        layer.scratchBytes = index * 2;
      });
    });
    input.transferMs = [trial % 3, trial % 5, trial % 7];
    const choices = [1, 2, 3].map(split => {
      let ms = input.transferMs[split - 1];
      const memory = input.profiles.map((profile, index) => {
        const layers = index ? profile.layers.slice(split) : profile.layers.slice(0, split);
        ms += profile.fixedMs + layers.reduce((sum, layer) => sum + layer.elapsedMs, 0);
        return profile.fixedBytes + layers.reduce((sum, layer) => sum + layer.residentBytes, 0)
          + Math.max(...layers.map(layer => layer.scratchBytes));
      });
      return { split, ms, memory };
    }).filter(choice => choice.memory.every((value, index) => value <= input.profiles[index].memoryBudgetBytes))
      .sort((a, b) => a.ms - b.ms || a.split - b.split);
    if (!choices.length) { assert.throws(() => proposeHardwarePartitionPlan(manifest, input), /No feasible/); continue; }
    const result = proposeHardwarePartitionPlan(manifest, input);
    assert.equal(result.plan.splitLayers[0], choices[0].split);
    assert.equal(result.estimatedMs, choices[0].ms);
    assert.deepEqual(result.memoryBytes, choices[0].memory);
  }
});
