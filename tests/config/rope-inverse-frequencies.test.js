import assert from 'node:assert/strict';
import { validateRoPEInverseFrequencies } from '../../src/config/rope-frequencies.js';
import { DEFAULT_MANIFEST_INFERENCE } from '../../src/config/schema/manifest.schema.js';
import { mergeConfig } from '../../src/config/merge.js';

const frequencies = [1, 0.464111328125];
assert.deepEqual(validateRoPEInverseFrequencies(frequencies, 4, 'test frequencies'), frequencies);
for (const invalid of [undefined, [], [1], [1, -1], [1, Infinity], [1, 0.1]]) {
  assert.throws(() => validateRoPEInverseFrequencies(invalid, 4, 'test frequencies'), /positive, finite f32/);
}
const manifest = { modelId: 'frequency-contract', inference: structuredClone(DEFAULT_MANIFEST_INFERENCE) };
manifest.inference.rope.ropeInverseFrequencies = frequencies;
const resolved = mergeConfig(manifest);
assert.deepEqual(resolved.inference.rope.ropeInverseFrequencies, frequencies);
assert.equal(resolved._sources.get('inference.rope.ropeInverseFrequencies'), 'manifest');
assert.throws(() => mergeConfig(manifest, { rope: { ropeInverseFrequencies: null } }), /source-owned/);
delete manifest.inference.rope.ropeInverseFrequencies;
assert.equal(mergeConfig(manifest).inference.rope.ropeInverseFrequencies, null);
console.log('rope-inverse-frequencies.test: ok');
