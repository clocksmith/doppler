import assert from 'node:assert/strict';

import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { canUseRmsNormWideTileProjectionFusion } from '../../src/inference/pipelines/text/attention/rmsnorm-fusion-gate.js';

function weight(label, withQ4K = false) {
  return createWeightBuffer(
    { label },
    'f16',
    'row',
    [4, 4],
    label,
    withQ4K ? { q4k: { buffer: { label: `${label}.q4k` }, layout: 'row' } } : null
  );
}

assert.equal(canUseRmsNormWideTileProjectionFusion(null, false), false);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qProj: weight('q'),
  kProj: weight('k'),
  vProj: weight('v'),
}, false), false);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qProj: weight('q', true),
  kProj: weight('k'),
  vProj: weight('v'),
}, false), false);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qProj: weight('q', true),
  kProj: weight('k'),
  vProj: weight('v'),
}, true), true);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qProj: weight('q', true),
  kProj: weight('k', true),
  vProj: weight('v', true),
}, false), true);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qProj: weight('q', true),
  kProj: weight('k', true),
}, false), true);
assert.equal(canUseRmsNormWideTileProjectionFusion({
  qkvProj: weight('qkv', true),
}, false), true);

console.log('rmsnorm-fusion-gate.test: ok');
