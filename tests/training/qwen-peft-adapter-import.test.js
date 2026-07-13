import assert from 'node:assert/strict';

import { serializeLoRASafetensors } from '../../src/experimental/training/export.js';
import {
  buildQwenPeftAdapterUploadPlan,
  parseQwenPeftAdapterSafetensors,
} from '../../src/experimental/training/qwen-peft-adapter-import.js';

const rank = 2;
const layerTypes = ['linear_attention', 'full_attention'];
const targetModules = [
  'q_proj',
  'k_proj',
  'v_proj',
  'o_proj',
  'gate_proj',
  'up_proj',
  'down_proj',
];

function values(length, start) {
  return Float32Array.from({ length }, (_, index) => start + index + 1);
}

function peftPair(layerIndex, branch, projection, inputSize, outputSize, start) {
  const prefix = `base_model.model.model.language_model.layers.${layerIndex}.${branch}.${projection}`;
  return [
    {
      name: `${prefix}.lora_A.default.weight`,
      shape: [rank, inputSize],
      data: values(rank * inputSize, start),
    },
    {
      name: `${prefix}.lora_B.default.weight`,
      shape: [outputSize, rank],
      data: values(outputSize * rank, start + 100),
    },
  ];
}

function fixtureTensors() {
  const tensors = [];
  for (let layerIndex = 0; layerIndex < layerTypes.length; layerIndex += 1) {
    tensors.push(
      ...peftPair(layerIndex, 'mlp', 'gate_proj', 3, 4, 1000 * layerIndex),
      ...peftPair(layerIndex, 'mlp', 'up_proj', 3, 4, 1000 * layerIndex + 200),
      ...peftPair(layerIndex, 'mlp', 'down_proj', 4, 3, 1000 * layerIndex + 400)
    );
    if (layerTypes[layerIndex] === 'full_attention') {
      tensors.push(
        ...peftPair(layerIndex, 'self_attn', 'q_proj', 3, 6, 2000),
        ...peftPair(layerIndex, 'self_attn', 'k_proj', 3, 2, 2200),
        ...peftPair(layerIndex, 'self_attn', 'v_proj', 3, 2, 2400),
        ...peftPair(layerIndex, 'self_attn', 'o_proj', 3, 3, 2600)
      );
    }
  }
  return tensors;
}

const tensors = fixtureTensors();
const weights = serializeLoRASafetensors(tensors);
const adapter = parseQwenPeftAdapterSafetensors(weights, {
  r: rank,
  lora_alpha: 4,
  target_modules: targetModules,
  layerTypes,
});

assert.equal(adapter.rank, 2);
assert.equal(adapter.alpha, 4);
assert.equal(adapter.scale, 2);
assert.equal(adapter.tensorCount, 20);
assert.equal(adapter.pairCount, 10);
assert.equal(adapter.elementCount, tensors.reduce((sum, tensor) => sum + tensor.data.length, 0));

const first = adapter.tensors.find((tensor) => (
  tensor.canonicalName === 'layers.0.mlp.gate_proj.lora_a'
));
assert.deepEqual(first.sourceShape, [2, 3]);
assert.deepEqual(first.shape, [3, 2]);
assert.deepEqual(Array.from(first.data), [1, 4, 2, 5, 3, 6]);

function fakePair(aShape, bShape) {
  return {
    A: { buffer: {}, dtype: 'f32', shape: aShape },
    B: { buffer: {}, dtype: 'f32', shape: bShape },
  };
}

function pairFor(layerIndex, projection) {
  const pair = adapter.tensors.filter((tensor) => (
    tensor.layerIndex === layerIndex && tensor.projection === projection
  ));
  return fakePair(
    pair.find((tensor) => tensor.kind === 'a').shape,
    pair.find((tensor) => tensor.kind === 'b').shape
  );
}

const layers = layerTypes.map((type, layerIndex) => ({
  type,
  inputs: {
    ...(type === 'full_attention'
      ? {
          attention: {
            lora: Object.fromEntries(
              ['q', 'k', 'v', 'o'].map((name) => [name, pairFor(layerIndex, `${name}_proj`)])
            ),
          },
        }
      : { attention: {} }),
    mlp: {
      lora: Object.fromEntries(
        ['gate', 'up', 'down'].map((name) => [name, pairFor(layerIndex, `${name}_proj`)])
      ),
    },
  },
}));

const plan = buildQwenPeftAdapterUploadPlan(layers, adapter);
assert.equal(plan.length, 20);
assert.deepEqual(plan.map((entry) => entry.canonicalName), adapter.tensors.map((entry) => entry.canonicalName));

const missingPairWeights = serializeLoRASafetensors(tensors.slice(0, -1));
assert.throws(
  () => parseQwenPeftAdapterSafetensors(missingPairWeights, {
    rank,
    alpha: 4,
    targetModules,
    layerTypes,
  }),
  /missing a complete A\/B pair/
);

const wrongRank = tensors.map((tensor, index) => (
  index === 0 ? { ...tensor, shape: [1, tensor.shape[1]], data: values(tensor.shape[1], 0) } : tensor
));
assert.throws(
  () => parseQwenPeftAdapterSafetensors(serializeLoRASafetensors(wrongRank), {
    rank,
    alpha: 4,
    targetModules,
    layerTypes,
  }),
  /does not match declared rank/
);

const linearAttentionTensor = peftPair(0, 'self_attn', 'q_proj', 3, 6, 4000);
assert.throws(
  () => parseQwenPeftAdapterSafetensors(
    serializeLoRASafetensors([...tensors, ...linearAttentionTensor]),
    { rank, alpha: 4, targetModules, layerTypes }
  ),
  /targets attention on a linear-attention layer/
);

layers[1].inputs.attention.lora.q.A.shape = [99, 2];
assert.throws(
  () => buildQwenPeftAdapterUploadPlan(layers, adapter),
  /upload shape mismatch/
);

console.log('qwen-peft-adapter-import.test: ok');
