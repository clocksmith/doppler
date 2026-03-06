import assert from 'node:assert/strict';
import { initializeDiffusion } from '../../src/inference/pipelines/diffusion/init.js';
import { buildScheduler, stepScmScheduler } from '../../src/inference/pipelines/diffusion/scheduler.js';
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
        layout: 'sd3',
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
  const state = initializeDiffusion(
    {
      config: {
        diffusion: {
          layout: 'sd3',
          components: {
            transformer: { config: { sample_size: 8 } },
            vae: { config: { sample_size: 64, latent_channels: 4 } },
            scheduler: {
              config: {
                _class_name: 'SCMScheduler',
                num_train_timesteps: 1000,
                prediction_type: 'trigflow',
                sigma_data: 0.5,
              },
            },
          },
        },
      },
    },
    { inference: { diffusion: { backend: { pipeline: 'gpu' } } } }
  );
  assert.equal(state.runtime.scheduler.type, 'scm');
  assert.equal(state.runtime.scheduler.predictionType, 'trigflow');
  assert.equal(state.runtime.scheduler.sigmaData, 0.5);
}

{
  const scheduler = buildScheduler({
    type: 'scm',
    numSteps: 1,
    numTrainTimesteps: 1000,
    predictionType: 'trigflow',
    sigmaData: 0.5,
  });
  assert.equal(scheduler.type, 'scm');
  assert.equal(scheduler.steps, 1);
  assert.equal(scheduler.sigmas, null);
  assert.equal(scheduler.timesteps.length, 2);
  assert.equal(scheduler.predictionType, 'trigflow');
  assert.equal(scheduler.sigmaData, 0.5);
}

{
  const scheduler = buildScheduler({
    type: 'scm',
    numSteps: 2,
    numTrainTimesteps: 1000,
    predictionType: 'trigflow',
    sigmaData: 0.5,
  });
  assert.equal(scheduler.timesteps.length, 3);

  const modelOutput = new Float32Array([0.2, -0.4]);
  const sample = new Float32Array([1.0, 0.5]);
  const noise = new Float32Array([0.25, -0.75]);
  const step = stepScmScheduler(scheduler, modelOutput, scheduler.timesteps[0], sample, 0, noise);
  assert.equal(step.prevSample.length, 2);
  assert.equal(step.predOriginalSample.length, 2);
  assert.throws(
    () => stepScmScheduler(scheduler, modelOutput, scheduler.timesteps[0], sample, 0),
    /requires a Float32Array noise tensor/
  );
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
  const state = initializeDiffusion(
    {
      config: {
        diffusion: {
          layout: 'sana',
          components: {
            transformer: { config: { sample_size: 8 } },
            vae: { config: { sample_size: 64, latent_channels: 4 } },
            scheduler: {
              config: {
                _class_name: 'SCMScheduler',
                num_train_timesteps: 1000,
              },
            },
          },
        },
      },
    },
    { inference: { diffusion: { backend: { pipeline: 'gpu' } } } }
  );
  assert.equal(state.runtime.scheduler.type, 'scm');
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
