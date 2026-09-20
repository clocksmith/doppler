import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PIPELINE_OPERATIONS, scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { InferencePipeline, EmbeddingPipeline } from '../../src/inference/pipelines/text.js';
import { EnergyPipeline } from '../../src/experimental/energy/pipeline.js';
import { DiffusionPipeline } from '../../src/experimental/diffusion/pipeline.js';
import { DiffusionGemmaPipeline } from '../../src/inference/pipelines/diffusion-gemma/pipeline.js';
import { StructuredJsonHeadPipeline } from '../../src/inference/pipelines/structured/json-head-pipeline.js';
import { EnergyRowHeadPipeline } from '../../src/experimental/orchestration/energy-row-head-pipeline.js';
import { embedBatch } from '../../src/inference/pipelines/text/execution.js';
import { executeEmbeddingBatch } from '../../src/inference/pipelines/text/embedding-batch.js';
import { getRuntimeConfig, snapshotRuntimeConfig } from '../../src/config/runtime.js';

test('every exposed pipeline method declares ownership', () => {
  for (const Class of [InferencePipeline, EmbeddingPipeline, EnergyPipeline, DiffusionPipeline,
    DiffusionGemmaPipeline, StructuredJsonHeadPipeline, EnergyRowHeadPipeline]) {
    const pipeline = new Class();
    const contract = { ...PIPELINE_OPERATIONS, ...pipeline.operationContract };
    for (let prototype = Object.getPrototypeOf(pipeline); prototype !== Object.prototype;
      prototype = Object.getPrototypeOf(prototype)) {
      for (const [name, descriptor] of Object.entries(Object.getOwnPropertyDescriptors(prototype))) {
        if (name === 'constructor' || name.startsWith('_') || typeof descriptor.value !== 'function') continue;
        assert(Object.hasOwn(contract, name), `${Class.name}.${name} requires an ownership declaration`);
      }
    }
  }
});

test('ordinary promise functions and wrapped iterators retain one owner through aliases', async () => {
  const gate = Promise.withResolvers();
  const started = Promise.withResolvers();
  let closes = 0, resets = 0;
  const raw = {
    isLoaded: true,
    work: (...args) => Promise.resolve().then(async () => { started.resolve(); await gate.promise; return args[0]; }),
    iterate: () => (async function* () { try { yield 1; } finally { throw new Error('iterator cleanup'); } })(),
    reset() { resets++; },
    unload: () => Promise.resolve().then(() => { closes++; raw.isLoaded = false; }),
    surprise() {},
  };
  const scoped = scopePipelineShaders(raw, null, { work: 'execution', iterate: 'streaming' });
  const alias = scopePipelineShaders(raw);
  assert.equal(alias, scoped);
  const pending = scoped.work(42);
  await started.promise;
  assert.throws(() => alias.reset(), /in progress/);
  gate.resolve();
  assert.equal(await pending, 42);
  const iterator = alias.iterate();
  assert.equal((await iterator.next()).value, 1);
  assert.throws(() => scoped.reset(), /in progress/);
  await assert.rejects(iterator.return(), /iterator cleanup/);
  scoped.reset();
  assert.equal(resets, 1);
  assert.throws(() => scoped.surprise, /explicit operation contract/);
  await Promise.all([scoped.unload(), alias.unload()]);
  assert.equal(closes, 1);
  await assert.rejects(alias.work(), /closed/);
});

test('embedding batch scheduling releases compatibility scope between explicit execution calls', async () => {
  const original = getRuntimeConfig();
  const first = Promise.withResolvers(), resume = Promise.withResolvers();
  const order = [];
  const a = scopePipelineShaders({
    isLoaded: true, runtimeConfig: snapshotRuntimeConfig({ inference: { sampling: { temperature: 0.25 } } }),
    embedBatch,
    async embed(prompt) {
      assert.equal(getRuntimeConfig().inference.sampling.temperature, 0.25);
      order.push(prompt);
      if (prompt === 'A1') { first.resolve(); await resume.promise; }
      return prompt;
    },
    reset() {}, async unload() {},
  }, null);
  const b = scopePipelineShaders({
    isLoaded: true, runtimeConfig: snapshotRuntimeConfig({ inference: { sampling: { temperature: 0.75 } } }),
    generateTokenIds() {
      assert.equal(getRuntimeConfig().inference.sampling.temperature, 0.75);
      order.push('B'); return Promise.resolve('B');
    },
  }, null);
  const batch = a.embedBatch(['A1', 'A2']);
  await first.promise;
  assert.throws(() => a.reset(), /in progress/);
  const other = b.generateTokenIds();
  resume.resolve();
  assert.deepEqual(await batch, ['A1', 'A2']);
  await other;
  assert.deepEqual(order, ['A1', 'B', 'A2']);
  assert.equal(getRuntimeConfig(), original);
  await a.unload();
});

test('standalone batch scheduler needs only request and execution ports and honors cancellation', async () => {
  const abort = new AbortController();
  const calls = [];
  await assert.rejects(executeEmbeddingBatch(['one', 'two'], { signal: abort.signal }, async prompt => {
    calls.push(prompt); abort.abort(new Error('cancel batch')); return prompt;
  }), /aborted/);
  assert.deepEqual(calls, ['one']);
});
