import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { MoERouter } from '../../src/inference/moe-router.js';
import { readBuffer } from '../../src/memory/buffer-pool.js';

export async function captureLoadedRouterWeights(pipeline, manifest, outputPrefix) {
  const records = [];
  assert.equal(pipeline.layerRouterWeights.size, manifest.architecture.numLayers);
  for (let layer = 0; layer < manifest.architecture.numLayers; layer++) {
    const name = `model.language_model.layers.${layer}.mlp.gate.weight`;
    const descriptor = manifest.tensors[name];
    assert(descriptor, `Missing source router ${name}`);
    const weight = pipeline.layerRouterWeights.get(layer)?.weight;
    assert.equal(weight?.dtype, 'f32');
    assert.equal(weight.layout, 'row');
    assert.deepEqual(weight.shape, descriptor.shape);
    const bytes = Buffer.from(await readBuffer(weight.buffer, weight.shape.reduce((a, b) => a * b, 1) * 4));
    const file = `${outputPrefix}.router-weight-${layer}.f32`;
    await fs.writeFile(file, bytes);
    records.push({ layer, tensor: name, file, shape: weight.shape,
      sha256: createHash('sha256').update(bytes).digest('hex') });
  }
  return records;
}

// Observation only: keep the original buffers and result, and restore the
// method even after a failed readback. Never feed captured values into a run.
export function observeLearnedRouters(layerIds, outputPrefix) {
  const original = MoERouter.prototype.computeRouterLogitsGPU;
  const records = [], errors = [];
  MoERouter.prototype.computeRouterLogitsGPU = async function(input, numTokens, gpuContext, options) {
    const output = await original.call(this, input, numTokens, gpuContext, options);
    if (!layerIds.includes(options.layerIdx)) return output;
    try {
      assert.equal(options.recorder ?? null, null, 'router capture requires the explicit immediate lane');
      assert.equal(options.inputDtype, 'f32');
      assert.equal(options.outputDtype, 'f32');
      const files = {};
      for (const [name, buffer, width] of [['input', input, this.hiddenSize], ['logits', output, this.numExperts]]) {
        const bytes = Buffer.from(await readBuffer(buffer, numTokens * width * 4));
        const file = `${outputPrefix}.router${options.layerIdx}.${name}.f32`;
        await fs.writeFile(file, bytes);
        files[name] = { file, shape: [numTokens, width], sha256: createHash('sha256').update(bytes).digest('hex') };
      }
      records.push({ layer: options.layerIdx, ...files });
    } catch (error) { errors.push(String(error)); }
    return output;
  };
  return { records, errors, restore() { MoERouter.prototype.computeRouterLogitsGPU = original; } };
}
