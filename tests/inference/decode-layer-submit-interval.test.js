import assert from 'node:assert/strict';
import { resolveDecodeLayerSubmitInterval } from '../../src/inference/pipelines/text/generator/decode.js';

const recorder = {};
assert.equal(resolveDecodeLayerSubmitInterval(undefined, recorder), null);
assert.equal(resolveDecodeLayerSubmitInterval({}, recorder), null);
assert.equal(resolveDecodeLayerSubmitInterval({ layerSubmitInterval: null }, null), null);
assert.equal(resolveDecodeLayerSubmitInterval({ layerSubmitInterval: 4 }, recorder), 4);
for (const layerSubmitInterval of [0, -1, 2.5, '4']) {
  assert.throws(() => resolveDecodeLayerSubmitInterval({ layerSubmitInterval }, recorder), /positive integer or null/);
}
assert.throws(() => resolveDecodeLayerSubmitInterval({ layerSubmitInterval: 4 }, null), /requires a recorded decode step/);
console.log('decode-layer-submit-interval.test: ok');
