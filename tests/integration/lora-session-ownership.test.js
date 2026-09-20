import assert from 'node:assert/strict';
import { test } from 'node:test';
import { InferencePipeline } from '../../src/inference/pipelines/text.js';
import { scopePipelineShaders } from '../../src/inference/pipelines/shader-scoped-pipeline.js';
import { createModelHandle } from '../../src/client/model-host/model-session.js';
import { createScopedModelSession } from '../../src/client/runtime/scoped-session.js';
import { snapshotRuntimeConfig } from '../../src/config/runtime.js';
import { authorizeBundledAdapter } from '../../src/config/revocation-policy.js';
import { wrapPipelineAsDreamProvider } from '../../src/client/wrap-pipeline-handle.js';

function adapter(id) {
  return {
    id, name: id, baseModel: 'session-model', layers: new Map([[0, { q_proj: {} }]]),
    identity: { schema: 'doppler.lora-execution-identity/v1', id, name: id,
      tensorCount: 2, digest: `sha256:${id.repeat(64)}` },
  };
}

async function fixture() {
  const initial = await authorizeBundledAdapter(adapter('a'));
  const candidate = await authorizeBundledAdapter(adapter('b'));
  const started = Promise.withResolvers();
  const resume = Promise.withResolvers();
  const events = [];
  const raw = {
    isLoaded: true, isGenerating: false,
    manifest: { modelId: 'session-model' },
    modelConfig: {}, runtimeConfig: snapshotRuntimeConfig(),
    resolvedRuntimeSession: { id: `sha256:${'d'.repeat(64)}` },
    tokenizer: { decode(ids) { return ids.join(' '); } },
    getStats() { return { kernelPathId: 'fixture', kernelPathSource: 'execution-v1' }; },
    getKernelCapabilities() { return { hasF16: false, hasSubgroups: false, adapterInfo: { vendor: 'fixture' } }; },
    weights: new Map([['layer_0', { qProj: {} }]]),
    lora: initial, currentSeqLen: 3,
    setLoRAAdapter: InferencePipeline.prototype.setLoRAAdapter,
    getActiveLoRA() { return this.lora; },
    resetGenerationState() { this.currentSeqLen = 0; },
    resetToSeqLen(length) { this.currentSeqLen = length; },
    async *generate() {
      this.isGenerating = true;
      try { yield this.lora.name; yield this.lora.name; }
      finally { this.isGenerating = false; }
    },
    async generateTokenIds(_prompt, options) {
      const tokenIds = [];
      for await (const _text of this.generate()) {
        await options.onToken?.(1);
        tokenIds.push(1);
      }
      return { tokenIds, stats: this.getStats() };
    },
    dopplerLoader: {
      manifest: { modelId: 'session-model' },
      async init() {},
      async loadLoRAWeights() {
        const previous = this.manifest;
        this.manifest = { adapterType: 'lora' };
        events.push('load-start');
        started.resolve();
        try {
          await resume.promise;
          return { adapter: candidate };
        } finally {
          this.manifest = previous;
          events.push('load-end');
        }
      },
    },
    async unload() {
      events.push('unload');
      this.dopplerLoader.manifest = null;
      this.weights.clear();
      this.isLoaded = false;
      this.lora = null;
    },
  };
  const pipeline = scopePipelineShaders(raw, null);
  const handle = createModelHandle(pipeline, { modelId: 'session-model', manifestHash: `sha256:${'c'.repeat(64)}` });
  return { raw, pipeline, handle, candidate, session: createScopedModelSession(handle), started, resume, events };
}

for (const operation of ['loadLoRA', 'unloadLoRA', 'setLoRAAdapter', 'resetGenerationState', 'resetToSeqLen']) {
  test(`${operation} cannot mutate an active stream`, { timeout: 5000 }, async () => {
    const { raw, pipeline, handle, candidate, session, events, resume } = await fixture();
    const stream = session.stream('prompt');
    try {
      assert.equal((await stream.next()).value.text, 'a');
      const identity = handle.activeLoRAIdentity;
      if (operation === 'loadLoRA') {
        const loading = session.loadLoRA({ adapterType: 'lora' });
        resume.resolve();
        await assert.rejects(loading, /in progress|busy|active/);
        assert.deepEqual(events, [], 'reject before starting adapter acquisition');
      } else if (operation === 'unloadLoRA') {
        await assert.rejects(session.unloadLoRA(), /in progress|busy|active/);
      } else if (operation === 'setLoRAAdapter') {
        assert.throws(() => pipeline.setLoRAAdapter(candidate), /in progress|busy|active/);
      } else if (operation === 'resetGenerationState') {
        assert.throws(() => session.resetGenerationState(), /in progress|busy|active/);
      } else {
        assert.throws(() => handle.advanced.resetToSeqLen(1), /in progress|busy|active/);
      }
      assert.equal(handle.activeLoRAIdentity, identity);
      assert.equal(raw.currentSeqLen, 3);
      assert.equal((await stream.next()).value.text, 'a');
    } finally {
      await stream.return();
      await session.close();
    }
  });
}

test('closing during adapter loading drains preparation without activating the result', { timeout: 5000 }, async () => {
  const { raw, pipeline, candidate, session, started, resume, events } = await fixture();
  const loading = session.loadLoRA({ adapterType: 'lora' });
  const rejected = assert.rejects(loading, /clos(ed|ing)/);
  await started.promise;
  const closing = session.close();
  try {
    assert.throws(() => pipeline.setLoRAAdapter(candidate), /clos(ed|ing)/);
    assert.deepEqual(events, ['load-start'], 'unload must not race the loader restoring its metadata');
  } finally {
    resume.resolve();
    await Promise.all([rejected, closing]);
  }
  assert.deepEqual(events, ['load-start', 'load-end', 'unload']);
  assert.equal(raw.lora, null);
  assert.equal(raw.isLoaded, false);
  assert.equal(raw.dopplerLoader.manifest, null, 'late preparation cannot restore unloaded metadata');
  await assert.rejects(session.loadLoRA({ adapterType: 'lora' }), /closed/);
});

test('direct text adapter activation rejects mutation while generation is active', async () => {
  const prior = adapter('a');
  const raw = { isLoaded: true, isGenerating: true, lora: prior };
  const candidate = await authorizeBundledAdapter(adapter('b'));
  assert.throws(() => InferencePipeline.prototype.setLoRAAdapter.call(raw, candidate), /in progress|busy|active/);
  assert.equal(raw.lora, prior);
});

test('adapter loading excludes other work and duplicate preparation on the same owner', { timeout: 5000 }, async () => {
  const { raw, pipeline, candidate, session, started, resume, events } = await fixture();
  const secondHandle = createModelHandle(pipeline, { modelId: 'session-model' });
  const loading = session.loadLoRA({ adapterType: 'lora' });
  await started.promise;
  try {
    await assert.rejects(session.stream('prompt').next(), /in progress/);
    await assert.rejects(secondHandle.loadLoRA({ adapterType: 'lora' }), /in progress/);
    await assert.rejects(secondHandle.unloadLoRA(), /in progress/);
    assert.throws(() => pipeline.setLoRAAdapter(candidate), /in progress/);
    assert.throws(() => secondHandle.advanced.resetToSeqLen(0), /in progress/);
    assert.equal(raw.lora.name, 'a');
    assert.equal(raw.currentSeqLen, 3);
    assert.deepEqual(events, ['load-start']);
  } finally {
    resume.resolve();
    await loading;
  }
  assert.equal(raw.lora, candidate);
  const stream = session.stream('prompt');
  assert.equal((await stream.next()).value.text, 'b');
  await stream.return();
  await session.unloadLoRA();
  assert.equal(raw.lora, null);
  await session.close();
});

test('failed adapter preparation preserves the old adapter and releases its operation', { timeout: 5000 }, async () => {
  const { raw, session, started, resume } = await fixture();
  const before = raw.lora;
  const failure = new Error('adapter acquisition failed');
  const loading = session.loadLoRA({ adapterType: 'lora' });
  const rejected = assert.rejects(loading, error => error === failure);
  await started.promise;
  resume.reject(failure);
  await rejected;
  assert.equal(raw.lora, before);
  assert.equal(raw.currentSeqLen, 3);
  const stream = session.stream('prompt');
  assert.equal((await stream.next()).value.text, 'a');
  await stream.return();
  await session.close();
});

test('rejected callback mutations leave generation evidence unchanged', { timeout: 5000 }, async () => {
  const { raw, pipeline, candidate, handle, session } = await fixture();
  const baseline = await handle.generateWithEvidence('prompt', { seed: 17 });
  const capabilities = raw.getKernelCapabilities();
  raw.getKernelCapabilities = () => {
    assert.throws(() => pipeline.setLoRAAdapter(candidate), /in progress/,
      'the owner remains reserved between execution and evidence construction');
    return capabilities;
  };
  let callbacks = 0;
  const observed = await handle.generateWithEvidence('prompt', {
    seed: 17,
    async onToken() {
      callbacks += 1;
      await assert.rejects(session.unloadLoRA(), /in progress/);
      assert.throws(() => handle.advanced.resetToSeqLen(0), /in progress/);
    },
  });
  assert.equal(callbacks, 2);
  assert.deepEqual(observed, baseline);
  assert.equal(raw.currentSeqLen, 3);
  await session.close();
});

test('queued execution reserves its adapter before acquiring the compatibility scope', { timeout: 5000 }, async () => {
  const a = await fixture();
  const b = await fixture();
  const streamA = a.session.stream('prompt');
  await streamA.next();
  const request = { seed: 17, presencePenalty: 0.25 };
  const workB = b.handle.generateWithEvidence('prompt', request);
  request.presencePenalty = 0.75;
  try {
    await assert.rejects(b.session.unloadLoRA(), /in progress/);
    assert.throws(() => b.pipeline.setLoRAAdapter(b.candidate), /in progress/);
    assert.equal(b.raw.lora.name, 'a');
  } finally {
    await streamA.return();
  }
  const resultB = await workB;
  assert.equal(resultB.runtimeProfile.model.activeAdapter, 'a');
  assert.equal(resultB.generationConfig.presencePenalty, 0.25, 'queued requests retain their normalized settings');
  await a.session.close();
  assert.equal((await b.handle.generateWithEvidence('prompt')).runtimeProfile.model.activeAdapter, 'a');
  await b.session.close();
});

test('close cancels queued adapter preparation before it touches the loader', { timeout: 5000 }, async () => {
  const a = await fixture();
  const b = await fixture();
  const stream = a.session.stream('prompt');
  await stream.next();
  const rejected = assert.rejects(b.session.loadLoRA({ adapterType: 'lora' }), /clos(ed|ing)/);
  const closing = b.session.close();
  await stream.return();
  await Promise.all([rejected, closing]);
  assert.deepEqual(b.events, ['unload']);
  await a.session.close();
});

test('a loader failure during close is preserved while cleanup still drains it', { timeout: 5000 }, async () => {
  const { raw, session, started, resume, events } = await fixture();
  const failure = new Error('adapter input failed');
  const rejected = assert.rejects(session.loadLoRA({ adapterType: 'lora' }), error => error === failure);
  await started.promise;
  const closing = session.close();
  resume.reject(failure);
  await Promise.all([rejected, closing]);
  assert.deepEqual(events, ['load-start', 'load-end', 'unload']);
  assert.equal(raw.dopplerLoader.manifest, null);
  assert.equal(raw.lora, null);
});

test('loss of the owning device during preparation prevents adapter activation', { timeout: 5000 }, async () => {
  const { raw, session, started, resume } = await fixture();
  const loss = Promise.withResolvers();
  raw.gpuContext = { device: {
    lost: loss.promise, features: new Set(), limits: {},
    createBindGroup(descriptor) { return descriptor; },
  } };
  const prior = raw.lora;
  const rejected = assert.rejects(session.loadLoRA({ adapterType: 'lora' }), /device is lost/);
  await started.promise;
  loss.resolve({ reason: 'destroyed', message: 'test preparation device loss' });
  await loss.promise;
  resume.resolve();
  await rejected;
  assert.equal(raw.lora, prior);
  assert.equal(raw.currentSeqLen, 3);
  await session.close();
});

test('provider adapter restoration cannot hide an execution failure during close', { timeout: 5000 }, async () => {
  const { raw, pipeline, session } = await fixture();
  const started = Promise.withResolvers();
  const resume = Promise.withResolvers();
  const failure = new Error('generation failed');
  raw.generate = async function* () {
    this.isGenerating = true;
    try {
      started.resolve();
      await resume.promise;
      throw failure;
    } finally { this.isGenerating = false; }
  };
  const provider = wrapPipelineAsDreamProvider(pipeline);
  const rejected = assert.rejects(provider.generate('prompt'), error => error === failure);
  await started.promise;
  const closing = session.close();
  resume.resolve();
  await Promise.all([rejected, closing]);
  assert.equal(raw.lora, null);
  assert.equal(raw.isLoaded, false);
});

test('rejected provider mutations do not add or remove registry entries', { timeout: 5000 }, async () => {
  const { raw, pipeline, session } = await fixture();
  const provider = wrapPipelineAsDreamProvider(pipeline);
  const input = {
    adapterId: 'registered', baseModel: 'session-model', rank: 1, alpha: 1,
    targetModules: ['q_proj'], layers: new Map([[0, { q_proj: { a: [1], b: [1] } }]]),
  };
  await provider.attachLoraAdapter(input);
  const current = raw.lora;
  const stream = session.stream('prompt');
  await stream.next();
  try {
    await assert.rejects(provider.detachLoraAdapter('registered'), /in progress/);
    await assert.rejects(provider.attachLoraAdapter({ ...input, adapterId: 'rejected' }), /in progress/);
    assert.equal(raw.lora, current);
  } finally { await stream.return(); }
  await assert.rejects(provider.generate({ prompt: 'prompt', loraAdapterId: 'rejected' }), /unknown LoRA adapter/);
  assert.equal((await provider.generate({ prompt: 'prompt', loraAdapterId: 'registered' })).loraAdapterId, 'registered');
  await session.close();
});
