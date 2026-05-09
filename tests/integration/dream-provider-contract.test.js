import assert from 'node:assert/strict';

import { wrapPipelineAsDreamProvider } from '../../src/client/provider.js';

function createPipeline() {
  return {
    isLoaded: true,
    manifest: { modelId: 'gemma4-e2b-it' },
    activeAdapter: null,
    generations: [],
    setLoRAAdapter(adapter) {
      this.activeAdapter = adapter;
    },
    getActiveLoRA() {
      return this.activeAdapter;
    },
    async *generate(prompt, options = {}) {
      this.generations.push({
        prompt,
        options,
        adapterName: this.activeAdapter?.name || null,
      });
      yield this.activeAdapter
        ? `lora:${this.activeAdapter.name}:${prompt}`
        : `base:${prompt}`;
    },
  };
}

{
  const pipeline = createPipeline();
  const provider = wrapPipelineAsDreamProvider(pipeline);
  const attached = await provider.attachLoraAdapter({
    adapterId: 'dream-intent-rewrite-v0',
    rank: 2,
    scale: 4,
    targetModules: ['q_proj', 'v_proj'],
    layers: new Map([
      [0, {
        q_proj: { a: [1, 2], b: [3, 4] },
        v_proj: { a: new Float32Array([5, 6]), b: new Float32Array([7, 8]) },
      }],
    ]),
  });

  assert.equal(attached.adapterId, 'dream-intent-rewrite-v0');
  assert.equal(attached.alpha, 8);
  assert.equal(attached.scale, 4);
  assert.equal(attached.layerCount, 1);
  assert.equal(pipeline.getActiveLoRA().name, 'dream-intent-rewrite-v0');
  assert.equal(pipeline.getActiveLoRA().layers.get(0).q_proj.a instanceof Float32Array, true);

  const base = await provider.generate({
    prompt: 'hello',
    loraAdapterId: null,
    samplingOptions: { maxTokens: 3 },
  });
  assert.deepEqual(base, {
    text: 'base:hello',
    useLora: false,
    baseModelId: 'gemma4-e2b-it',
    loraAdapterId: null,
  });
  assert.equal(pipeline.getActiveLoRA().name, 'dream-intent-rewrite-v0');

  const lora = await provider.generate({
    prompt: 'hello',
    loraAdapterId: 'dream-intent-rewrite-v0',
    samplingOptions: { maxTokens: 3 },
  });
  assert.deepEqual(lora, {
    text: 'lora:dream-intent-rewrite-v0:hello',
    useLora: true,
    baseModelId: 'gemma4-e2b-it',
    loraAdapterId: 'dream-intent-rewrite-v0',
  });
  assert.equal(pipeline.getActiveLoRA().name, 'dream-intent-rewrite-v0');
  assert.deepEqual(pipeline.generations.map((row) => row.adapterName), [
    null,
    'dream-intent-rewrite-v0',
  ]);
}

{
  const provider = wrapPipelineAsDreamProvider(createPipeline());
  await assert.rejects(
    () => provider.generate({ prompt: 'hello', loraAdapterId: 'missing-adapter' }),
    /unknown LoRA adapter "missing-adapter"/
  );
  await assert.rejects(
    () => provider.attachLoraAdapter({
      adapterId: 'bad',
      rank: 2,
      scale: 1,
      targetModules: ['not_a_projection'],
      layers: new Map([[0, { q_proj: { a: [1], b: [1] } }]]),
    }),
    /unsupported LoRA target module "not_a_projection"/
  );
}

console.log('dream-provider-contract.test: ok');
