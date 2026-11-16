import assert from 'node:assert/strict';
import { isMoELayer } from '../../src/inference/pipelines/text/ffn/types.js';
import { isMoELayer as isMoELayerFromLayer } from "../../src/inference/pipelines/text/execution/layer-input-execution.js";

assert.equal(isMoELayerFromLayer, isMoELayer, 'layer execution re-exports the single MoE-layer test');

// layerTypes that name attention mixers (Qwen 3.5 hybrid) do not make MoE layers dense.
const hybrid = { useMoE: true, layerTypes: ['linear_attention', 'linear_attention', 'full_attention'] };
for (let layer = 0; layer < hybrid.layerTypes.length; layer++) {
  assert.equal(isMoELayer(layer, hybrid), true, `hybrid MoE layer ${layer}`);
}

assert.equal(isMoELayer(0, { useMoE: true, layerTypes: null }), true);
assert.equal(isMoELayer(0, { useMoE: false, layerTypes: ['moe'] }), false);

// Explicit 'moe' entries make layerTypes the per-layer FFN declaration.
const declared = { useMoE: true, layerTypes: ['full_attention', 'moe'] };
assert.equal(isMoELayer(0, declared), false);
assert.equal(isMoELayer(1, declared), true);

console.log('moe-layer-selection.test: passed');
