import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { MoERouter } from '../../src/inference/moe-router.js';
import { getDevice } from '../../src/gpu/device.js';
import { readBuffer, releaseBuffer } from '../../src/memory/buffer-pool.js';

// Counterfactual TEST PORT, not observation or production execution. Replace
// router logits with an independently produced CPU fixture to isolate the
// continuous stack from expert-choice differences. Always report this control.
export async function installReferenceRouterControl(directory, manifestSha256, inputsSha256) {
  const hash = bytes => createHash('sha256').update(bytes).digest('hex');
  const receiptBytes = await fs.readFile(path.join(directory, 'raw-logits-receipt.json'));
  const receipt = JSON.parse(receiptBytes);
  assert.equal(receipt.manifestSha256, manifestSha256);
  assert.equal(receipt.inputsSha256, inputsSha256);
  assert.equal(receipt.layers, 48);
  const fixtures = await Promise.all(receipt.files.map(async item => {
    const bytes = await fs.readFile(path.join(directory, item.file));
    assert.equal(hash(bytes), item.sha256); assert.equal(bytes.length, item.shape[0] * item.shape[1] * 4);
    assert(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).every(Number.isFinite));
    return { ...item, bytes };
  }));
  assert.equal(fixtures.length, 48);
  assert.deepEqual(fixtures.map(item => item.layer), Array.from({ length: 48 }, (_, i) => i));
  const original = MoERouter.prototype.computeRouterLogitsGPU;
  const control = { kind: 'counterfactual-upstream-CPU-router-logits',
    scope: 'All router probabilities and expert choices come from the CPU fixture; this is not normal learned-model qualification',
    receiptSha256: hash(receiptBytes), replaced: [] };
  MoERouter.prototype.computeRouterLogitsGPU = async function(input, numTokens, gpuContext, options) {
    assert.equal(options.recorder ?? null, null, 'reference routing requires the declared immediate lane');
    assert.equal(options.outputDtype, 'f32');
    const fixture = fixtures[options.layerIdx];
    assert(fixture); assert.equal(fixture.shape[0], numTokens); assert.equal(fixture.shape[1], this.numExperts);
    const output = await original.call(this, input, numTokens, gpuContext, options);
    try {
      const observed = await readBuffer(output, fixture.bytes.length);
      assert(new Float32Array(observed).every(Number.isFinite), 'control must not hide nonfinite GPU routing');
      getDevice().queue.writeBuffer(output, 0, fixture.bytes);
      control.replaced.push({ layer: options.layerIdx, originalSha256: hash(new Uint8Array(observed)), fixtureSha256: fixture.sha256 });
      return output;
    } catch (error) {
      if (output !== options.outputBuffer) {
        try { releaseBuffer(output); }
        catch (cleanupError) {
          throw new AggregateError([error, cleanupError], 'Reference router control cleanup failed', { cause: error });
        }
      }
      throw error;
    }
  };
  return { control, restore() { MoERouter.prototype.computeRouterLogitsGPU = original; } };
}
