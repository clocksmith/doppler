import assert from 'node:assert/strict';

import {
  buildRefreshRawConfig,
  extractSourceQuantization,
} from '../../tools/refresh-converted-manifest.js';

{
  assert.equal(
    extractSourceQuantization({
      quantizationInfo: {
        weights: 'q4k',
      },
    }),
    'q4k'
  );
  assert.equal(
    extractSourceQuantization({
      quantization: 'F16',
    }),
    'F16'
  );
  assert.equal(extractSourceQuantization({}), '');
}

{
  const rawConfig = buildRefreshRawConfig({
    config: {},
    modelType: 'transformer',
    inference: {
      layerPattern: {
        layerTypes: ['attention', 'ffn'],
      },
    },
  });
  assert.equal(rawConfig.model_type, 'transformer');
  assert.deepEqual(rawConfig.layer_types, ['attention', 'ffn']);
}

console.log('refresh-converted-manifest.test: ok');
