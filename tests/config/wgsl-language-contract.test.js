import assert from 'node:assert/strict';
import { getRequiredWgslFeatures, assertWgslFeaturesSupported } from '../../src/config/wgsl-language-contract.js';

assert.deepEqual(getRequiredWgslFeatures(`
// requires fake;
/* requires fake; /* nested */ still commented */
enable subgroups;
requires subgroup_id, /* comment */ subgroup_uniformity;
requires subgroup_id;
requires subgroup_id,;
`), ['subgroup_id', 'subgroup_uniformity']);
assert.deepEqual(getRequiredWgslFeatures('// requires fake;'), []);
assert.deepEqual(getRequiredWgslFeatures('@builtin(/* gap */num_subgroups) count: u32'), ['subgroup_id']);
assert.deepEqual(getRequiredWgslFeatures('// @builtin(subgroup_id)\n/* @builtin(num_subgroups) */'), []);
assert.throws(() => getRequiredWgslFeatures('/* unfinished'), /unterminated/);
assert.throws(() => getRequiredWgslFeatures('requires invalid-feature;'), /Invalid/);
assert.throws(() => getRequiredWgslFeatures('requires ;'), /must name/);
assert.throws(() => assertWgslFeaturesSupported(['subgroup_id'], ['subgroups'], 'probe'), /subgroup_id/);
assertWgslFeaturesSupported(['subgroup_id'], new Set(['subgroup_id']), 'probe');
console.log('wgsl-language-contract.test: ok');
