import assert from 'node:assert/strict';

const { MultiModelLoader } = await import('../../src/loader/multi-model-loader.js');

class FakePipeline {
  constructor(options = {}) {
    this.options = options;
    this.initializeCalls = 0;
    this.loadModelCalls = 0;
    this.unloadCalls = 0;
    this.preloadedWeights = null;
    this.loadedManifest = null;
  }

  async initialize() {
    this.initializeCalls += 1;
    if (this.options.initializeError) {
      throw this.options.initializeError;
    }
  }

  setPreloadedWeights(weights) {
    this.preloadedWeights = weights;
  }

  async loadModel(manifest) {
    this.loadModelCalls += 1;
    this.loadedManifest = manifest;
    if (this.options.loadModelError) {
      throw this.options.loadModelError;
    }
  }

  async unload() {
    this.unloadCalls += 1;
  }
}

class TestMultiModelLoader extends MultiModelLoader {
  constructor() {
    super();
    this.nextWeights = null;
    this.nextLoadError = null;
    this.fakeLoader = {
      unloadCalls: 0,
      async unload() {
        this.unloadCalls += 1;
      },
    };
    this.pipelineQueue = [];
    this.loadBaseCalls = [];
  }

  async _loadBaseWeights(manifest, options, runtimeConfig) {
    this.loadBaseCalls.push({ manifest, options, runtimeConfig });
    if (this.nextLoadError) {
      throw this.nextLoadError;
    }
    return this.nextWeights;
  }

  _createPipeline() {
    const pipeline = this.pipelineQueue.shift();
    if (!pipeline) {
      throw new Error('No fake pipeline queued');
    }
    return pipeline;
  }

  _getBaseLoader() {
    return this.fakeLoader;
  }
}

{
  const loader = new TestMultiModelLoader();
  loader.baseManifest = { modelId: 'old-model' };
  loader.baseWeights = { id: 'old-weights' };
  loader.adapters.set('adapter-a', { name: 'adapter-a' });
  loader.nextWeights = { id: 'new-weights' };

  const stalePipeline = new FakePipeline();
  loader.pipelineQueue.push(stalePipeline);
  const tracked = await loader.createSharedPipeline();

  assert.equal(tracked, stalePipeline);
  assert.equal(stalePipeline.preloadedWeights, loader.baseWeights);

  const manifest = { modelId: 'new-model' };
  const resolved = await loader.loadBase(manifest);

  assert.equal(resolved, loader.nextWeights);
  assert.equal(loader.baseManifest, manifest);
  assert.equal(loader.baseWeights, loader.nextWeights);
  assert.equal(stalePipeline.unloadCalls, 1);
  assert.equal(loader.fakeLoader.unloadCalls, 1);
  assert.equal(loader.adapters.size, 0);
}

{
  const loader = new TestMultiModelLoader();
  loader.baseManifest = { modelId: 'old-model' };
  loader.baseWeights = { id: 'old-weights' };
  loader.adapters.set('adapter-a', { name: 'adapter-a' });
  loader.nextLoadError = new Error('base load failed');

  await assert.rejects(() => loader.loadBase({ modelId: 'broken-model' }), /base load failed/);
  assert.equal(loader.fakeLoader.unloadCalls, 1);
  assert.equal(loader.baseManifest, null);
  assert.equal(loader.baseWeights, null);
  assert.equal(loader.adapters.size, 0);
}

{
  const loader = new TestMultiModelLoader();
  loader.baseManifest = { modelId: 'model-a' };
  loader.baseWeights = { id: 'weights-a' };
  const pipeline = new FakePipeline();
  loader.pipelineQueue.push(pipeline);

  const created = await loader.createSharedPipeline();
  await created.unload();
  await loader.unload();

  assert.equal(created, pipeline);
  assert.equal(pipeline.initializeCalls, 1);
  assert.equal(pipeline.loadModelCalls, 1);
  assert.equal(pipeline.unloadCalls, 1);
  assert.equal(loader.fakeLoader.unloadCalls, 1);
}

{
  const loader = new TestMultiModelLoader();
  loader.baseManifest = { modelId: 'model-a' };
  loader.baseWeights = { id: 'weights-a' };
  const pipeline = new FakePipeline({ loadModelError: new Error('pipeline load failed') });
  loader.pipelineQueue.push(pipeline);

  await assert.rejects(() => loader.createSharedPipeline(), /pipeline load failed/);
  assert.equal(pipeline.initializeCalls, 1);
  assert.equal(pipeline.loadModelCalls, 1);
  assert.equal(pipeline.unloadCalls, 1);
}

console.log('multi-model-loader-lifecycle.test: ok');
