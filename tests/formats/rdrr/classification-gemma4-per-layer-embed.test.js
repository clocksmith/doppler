import assert from 'node:assert/strict';

import { classifyTensorRole, classifyTensor } from '../../../src/formats/rdrr/classification.js';

const name = 'model.language_model.embed_tokens_per_layer.weight';
const projectorName = 'model.embed_vision.embedding_projection.weight';
const audioTowerName = 'model.audio_tower.encoder.layers.0.self_attn.q_proj.weight';
const audioModelName = 'model.audio_model.encoder.layers.0.self_attn.q_proj.weight';

assert.equal(classifyTensorRole(name), 'embedding');
assert.equal(classifyTensor(name, 'gemma4'), 'per_layer_input');
assert.equal(classifyTensorRole(projectorName), 'projector');
assert.equal(classifyTensor(projectorName, 'gemma4'), 'projector');
assert.equal(classifyTensorRole(audioTowerName), 'audio');
assert.equal(classifyTensorRole(audioModelName), 'audio');

console.log('classification-gemma4-per-layer-embed.test: ok');
