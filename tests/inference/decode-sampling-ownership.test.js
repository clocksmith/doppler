import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/generator.js', import.meta.url).href;
const buffer = { size: 1048576 };
globalThis.decodeSamplingOwnership = { released: [] };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../../memory/buffer-pool.js') {
    return { url: `data:text/javascript,${encodeURIComponent(`
      export const releaseBuffer = buffer => globalThis.decodeSamplingOwnership.released.push(buffer);
      export const acquireBuffer = () => {};
      export const readBuffer = () => {};
      export const readBufferSlice = () => {};
      export const uploadData = () => {};
    `)}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { PipelineGenerator } = await import(target);
  const generator = Object.create(PipelineGenerator.prototype);
  const logits = new Float32Array([1, 3]);
  const ids = [4, 5];
  const options = { temperature: 0 };
  generator._decodeStepToLogits = async (actualIds, actualOptions) => {
    assert.equal(actualIds, ids);
    assert.equal(actualOptions, options);
    return { logits, logitsBuffer: buffer };
  };
  for (const fail of [false, true, false]) {
    globalThis.decodeSamplingOwnership.released.length = 0;
    generator._sampleNextTokenFromLogits = (actualLogits, actualIds, actualOptions) => {
      assert.equal(actualLogits, logits);
      assert.equal(actualIds, ids);
      assert.equal(actualOptions, options);
      if (fail) throw new Error('sampling failed');
      return 1;
    };
    const result = generator._decodeNextTokenViaLogits(ids, options);
    if (fail) await assert.rejects(result, /sampling failed/);
    else assert.equal(await result, 1);
    assert.deepEqual(globalThis.decodeSamplingOwnership.released, [buffer]);
  }
  globalThis.decodeSamplingOwnership.released.length = 0;
  generator._decodeStepToLogits = async () => { throw new Error('decode failed'); };
  await assert.rejects(generator._decodeNextTokenViaLogits(ids, options), /decode failed/);
  assert.deepEqual(globalThis.decodeSamplingOwnership.released, []);
  generator._decodeStepToLogits = async () => ({ logits, logitsBuffer: null });
  assert.equal(await generator._decodeNextTokenViaLogits(ids, options), 1);
  assert.deepEqual(globalThis.decodeSamplingOwnership.released, []);
} finally {
  hooks.deregister();
  delete globalThis.decodeSamplingOwnership;
}
console.log('decode-sampling-ownership: passed');
