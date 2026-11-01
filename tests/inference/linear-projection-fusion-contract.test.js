import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { setRuntimeConfig } from '../../src/config/runtime.js';
const target = new URL('../../src/inference/pipelines/text/linear-attention-ab-fusion.js', import.meta.url).href;
const live = new Set();
let failure = '';
const fail = stage => { if (failure === stage) throw Error(`injected ${stage}`); };
globalThis.projectionFusionDevice = {
  createBuffer({ size }) { const b = { size, destroy() { assert(live.delete(b)); } }; live.add(b); return b; },
  createCommandEncoder() { fail('encoder'); return { copyBufferToBuffer() { fail('copy'); }, finish() { fail('finish'); return {}; } }; },
  queue: { submit() { fail('submit'); } },
};
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_DST: 2 };
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === target && specifier === '../../../gpu/device.js') return { url: 'data:text/javascript,export const getDevice=()=>globalThis.projectionFusionDevice;', shortCircuit: true };
  return next(specifier, context);
} });
try {
  const { resolveLinearAttentionABProjection: ab, resolveLinearAttentionQKVZProjection: qkvz } = await import(target);
  const runtimeSession = Object.freeze({ useLinearAttentionABProjectionFusion: true, useLinearAttentionQKVZProjectionFusion: true });
  const options = { phase: 'decode', numTokens: 1, hiddenSize: 256, numVHeads: 2, convDim: 3, valueDim: 2, layerIdx: 0, runtimeSession };
  const weight = (rows, dtype) => ({ buffer: { size: rows * 512 }, dtype, layout: 'row', shape: [rows, 256] });
  for (const [resolve, weights, key] of [[ab, () => ({ linearInProjA: weight(2, 'f16'), linearInProjB: weight(2, 'f16') }), 'linearABProj'],
    [qkvz, () => ({ qkvProj: weight(3, 'q4k'), linearInProjZ: weight(2, 'q4k') }), 'linearQKVZProj']]) {
    assert.throws(() => resolve(weights(), { ...options, runtimeSession: undefined }), /requires resolved/);
    for (failure of ['encoder', 'copy', 'finish', 'submit']) {
      const w = weights();
      assert.throws(() => resolve(w, options), /injected/);
      assert.equal(live.size, 0, `${failure}: no orphan allocation`);
      assert.equal(w[key], undefined);
    }
    failure = '';
    const w = weights();
    const result = resolve(w, options);
    assert(result);
    setRuntimeConfig({ inference: { session: { useLinearAttentionABProjectionFusion: false, useLinearAttentionQKVZProjectionFusion: false } } });
    assert.equal(resolve(w, { ...options, debugProbes: [{}], operatorDiagnostics: { enabled: true } }).weight, result.weight);
    assert.equal(live.size, 1, 'cached projection reused under observation/global change');
    result.weight.buffer.destroy();
  }
} finally { hooks.deregister(); delete globalThis.projectionFusionDevice; delete globalThis.GPUBufferUsage; }
console.log('linear projection fusion: explicit policy, observation parity and allocation failure cleanup passed');
