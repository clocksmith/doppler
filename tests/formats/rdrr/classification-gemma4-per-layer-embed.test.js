import assert from 'node:assert/strict';

import { classifyTensorRole, classifyTensor } from '../../../src/formats/rdrr/classification.js';

const name = 'model.language_model.embed_tokens_per_layer.weight';
const projectorName = 'model.embed_vision.embedding_projection.weight';

assert.equal(classifyTensorRole(name), 'embedding');
assert.equal(classifyTensor(name, 'gemma4'), 'per_layer_input');
assert.equal(classifyTensorRole(projectorName), 'projector');
assert.equal(classifyTensor(projectorName, 'gemma4'), 'projector');

console.log('classification-gemma4-per-layer-embed.test: ok');
