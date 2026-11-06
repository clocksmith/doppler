import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setDevice } from '../../src/gpu/device.js';
import { createWeightBuffer } from '../../src/gpu/weight-buffer.js';
import { prepareQwenExpertLayer } from '../../src/loader/experts/qwen-paging.js';
import { validateQwen35PagerConfig } from '../../src/config/schema/loading.schema.js';

const policy = { enabled: true, maxResidentLayers: 2 };
function fixture() {
  const context = { manifest: { architecture: { numLayers: 4 }, moeConfig: { expertFormat: 'qwen3_5' } }, experts: new Map(), gpuBuffers: new Set() };
  const consumers = new Map(), destroyed = [];
  const bank = layer => {
    const buffer = { __dopplerFakeGPUBuffer: true, size: 1024, destroy() { destroyed.push(layer); } };
    const packed = { gateUp: createWeightBuffer(buffer, 'q4k', 'row', [1, 1], 'fixture'), down: buffer };
    context.gpuBuffers.add(buffer);
    context.experts.set(`layer_${layer}_qwen35_packed`, packed);
    for (const expert of [0, 7]) {
      const key = `layer_${layer}_expert_${expert}`;
      context.experts.set(key, packed); consumers.set(key, packed);
    }
    return buffer;
  };
  return { context, consumers, bank, destroyed };
}
const device = wait => ({ features: new Set(), limits: {}, createBindGroup() {}, queue: { submit() {}, onSubmittedWorkDone: wait } });

await test('paging waits for GPU settlement, then retires one complete bank and both alias caches', async () => {
  const f = fixture(); f.bank(0); f.bank(1);
  const gate = Promise.withResolvers(); setDevice(device(() => gate.promise));
  const preparing = prepareQwenExpertLayer(f.context, 2, f.consumers, policy, null);
  await Promise.resolve(); assert.deepEqual(f.destroyed, []);
  assert.ok(f.consumers.has('layer_0_expert_7'));
  gate.resolve(); const receipt = await preparing;
  assert.deepEqual(receipt.evictedLayers, [0]);
  assert.deepEqual(f.destroyed, [0], 'aliased buffers are released exactly once');
  assert.equal(f.context.experts.has('layer_0_expert_7'), false);
  assert.equal(f.consumers.has('layer_0_expert_7'), false);
  assert.ok(f.consumers.has('layer_1_expert_7'));
  f.bank(2); assert.equal(f.context.gpuBuffers.size, 2);
});
await test('resident access updates eviction order without reloading or destroying the active bank', async () => {
  const f = fixture(); f.bank(0); f.bank(1); setDevice(device(async () => {}));
  await prepareQwenExpertLayer(f.context, 0, f.consumers, policy, null);
  const receipt = await prepareQwenExpertLayer(f.context, 2, f.consumers, policy, null);
  assert.deepEqual(receipt.evictedLayers, [1]); assert.ok(f.context.experts.has('layer_0_qwen35_packed'));
});
await test('abort and settlement failure preserve bank ownership without early destruction', async () => {
  for (const abort of [true, false]) {
    const f = fixture(); f.bank(0); f.bank(1); const controller = new AbortController();
    setDevice(device(async () => { if (abort) controller.abort(); else throw Error('GPU settle failed'); }));
    await assert.rejects(prepareQwenExpertLayer(f.context, 2, f.consumers, policy, null, controller.signal));
    assert.deepEqual(f.destroyed, []); assert.equal(f.context.gpuBuffers.size, 2);
    assert.ok(f.consumers.has('layer_0_expert_7'));
  }
});
await test('cleanup failure invalidates stale aliases and preserves the failed buffer owner', async () => {
  const f = fixture(); const buffer = f.bank(0); f.bank(1);
  buffer.destroy = () => { throw Error('destroy failed'); }; setDevice(device(async () => {}));
  await assert.rejects(prepareQwenExpertLayer(f.context, 2, f.consumers, policy, null), AggregateError);
  assert.ok(f.context.gpuBuffers.has(buffer)); assert.equal(f.consumers.has('layer_0_expert_0'), false);
  assert.equal(f.context.experts.has('layer_0_qwen35_packed'), false);
});
await test('invalid policy, foreign format and recorded retirement fail before mutation', async () => {
  const f = fixture(); f.bank(0); f.bank(1);
  await assert.rejects(prepareQwenExpertLayer(f.context, 2, f.consumers, policy, {}), /disableCommandBatching/);
  f.context.manifest.moeConfig.expertFormat = 'gpt-oss';
  await assert.rejects(prepareQwenExpertLayer(f.context, 2, f.consumers, policy, null), /qwen3_5/);
  assert.throws(() => validateQwen35PagerConfig({ enabled: true, maxResidentLayers: 0 }));
  assert.deepEqual(f.destroyed, []);
  assert.equal(await prepareQwenExpertLayer(f.context, 2, f.consumers, { ...policy, enabled: false }, null), null);
});
setDevice(null);
