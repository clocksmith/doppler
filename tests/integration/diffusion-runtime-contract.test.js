import assert from 'node:assert/strict';
import { initializeDiffusion } from '../../src/inference/pipelines/diffusion/init.js';
import { assertClipHiddenActivationSupported } from '../../src/inference/pipelines/diffusion/text-encoder-gpu.js';

function createManifest({ includeTransformer = true } = {}) {
  const components = {
    vae: {
      config: {
        sample_size: 64,
        latent_channels: 4,
        norm_num_groups: 32,
        scaling_factor: 0.18215,
      },
    },
    scheduler: {
      config: {
        _class_name: 'FlowMatchEulerDiscreteScheduler',
        num_train_timesteps: 1000,
        shift: 1.0,
      },
    },
  };
  if (includeTransformer) {
    components.transformer = {
      config: {
        sample_size: 8,
      },
    };
  }
  return {
    config: {
      diffusion: {
        components,
      },
    },
  };
}

{
  const state = initializeDiffusion(
    createManifest({ includeTransformer: true }),
    { inference: { diffusion: { backend: { pipeline: 'gpu' } } } }
  );
  assert.equal(state.runtime.backend.pipeline, 'gpu');
  assert.equal(state.latentScale, 8);
  assert.equal(state.latentChannels, 4);
}

{
  assert.throws(
    () => initializeDiffusion(
      createManifest({ includeTransformer: true }),
      { inference: { diffusion: { backend: { pipeline: 'legacy_gpu' } } } }
    ),
    /backend\.pipeline must be "gpu"/
  );
}

{
  assert.throws(
    () => initializeDiffusion(
      createManifest({ includeTransformer: true }),
      { inference: { diffusion: { backend: { pipeline: 'cpu' } } } }
    ),
    /backend\.pipeline must be "gpu"/
  );
}

{
  assert.throws(
    () => initializeDiffusion(
      createManifest({ includeTransformer: false }),
      { inference: { diffusion: { backend: { pipeline: 'gpu' } } } }
    ),
    /requires manifest\.config\.diffusion\.components\.transformer\.config/
  );
}

{
  assert.doesNotThrow(() => assertClipHiddenActivationSupported({ hidden_act: 'gelu' }));
  assert.doesNotThrow(() => assertClipHiddenActivationSupported({ hidden_act: 'quick_gelu' }));
  assert.throws(
    () => assertClipHiddenActivationSupported({ hidden_act: 'relu' }),
    /Unsupported CLIP hidden_act/
  );
}

console.log('diffusion-runtime-contract.test: ok');
