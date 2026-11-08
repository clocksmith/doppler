import assert from 'node:assert/strict';
import { buildScheduler } from '../../src/experimental/diffusion/scheduler.js';
import { initializeDiffusion } from '../../src/inference/pipelines/diffusion/init.js';
import { assertSupportedDiffusionSchedulerType } from '../../src/inference/pipelines/diffusion/scheduler-contract.js';

const close = (actual, expected, tolerance, label) => assert.ok(
  Math.abs(actual - expected) <= tolerance,
  `${label}: expected ${expected} ± ${tolerance}, got ${actual}`
);

// Reference flow-match schedule (diffusers FlowMatchEulerDiscreteScheduler):
// shifted(s) = shift * s / (1 + (shift - 1) * s); sigma_min = shifted(1/T);
// grid = shifted(linspace(1, sigma_min, steps)); the last step integrates to 0.
{
  const scheduler = buildScheduler({ type: 'flowmatch_euler', numSteps: 4, numTrainTimesteps: 1000, shift: 1.0 });
  assert.equal(scheduler.steps, 4);
  close(scheduler.sigmas[0], 1.0, 1e-6, 'sigma_max');
  close(scheduler.sigmas[3], 0.001, 1e-6, 'sigma_min at shift 1');
  assert.ok(scheduler.sigmas[3] > 0, 'the schedule never reaches 0; the final Euler step moves to 0 itself');
  close(scheduler.timesteps[0], 1000, 1e-3, 'first timestep');
  close(scheduler.timesteps[3], 1, 1e-3, 'last timestep');
}

{
  // SD3 ships shift=3. The old formula t / (t + shift (1 - t)) pushed the
  // mid-point to 0.25; the reference pushes it toward 1 (0.751).
  const scheduler = buildScheduler({ type: 'flowmatch_euler', numSteps: 3, numTrainTimesteps: 1000, shift: 3.0 });
  const sigmaMinOnce = (3 * 0.001) / (1 + 2 * 0.001);
  const mid = (1 + sigmaMinOnce) / 2;
  close(scheduler.sigmas[0], 1.0, 1e-6, 'shifted sigma_max');
  close(scheduler.sigmas[1], (3 * mid) / (1 + 2 * mid), 1e-6, 'shifted mid-point');
  assert.ok(scheduler.sigmas[1] > 0.5, 'shift > 1 moves noise levels toward sigma = 1');
  close(scheduler.sigmas[2], (3 * sigmaMinOnce) / (1 + 2 * sigmaMinOnce), 1e-6, 'shifted sigma_min');
  for (let i = 1; i < scheduler.sigmas.length; i += 1) {
    assert.ok(scheduler.sigmas[i] < scheduler.sigmas[i - 1], 'sigmas decrease monotonically');
  }
  // Every step has a non-zero delta, including the last one (to 0).
  const deltas = Array.from(scheduler.sigmas, (sigma, i) => (i + 1 < scheduler.steps ? scheduler.sigmas[i + 1] : 0) - sigma);
  assert.ok(deltas.every((delta) => delta < 0), `all deltas negative: ${deltas.join(', ')}`);
}

assert.throws(() => buildScheduler({ type: 'flowmatch_euler', numSteps: 2, numTrainTimesteps: 1000, shift: 0 }), /shift must be positive/);

// Schedulers the GPU loop does not implement are rejected by the builder and by
// the contract helper, naming what is implemented.
for (const type of ['euler', 'euler_a', 'dpmpp_2m', 'ddim']) {
  assert.throws(
    () => buildScheduler({ type, numSteps: 4, numTrainTimesteps: 1000 }),
    /not implemented by the GPU runtime.*flowmatch_euler \(FlowMatchEulerDiscreteScheduler\), scm \(SCMScheduler\)/
  );
}
assert.equal(assertSupportedDiffusionSchedulerType('scm'), 'scm');
assert.throws(() => assertSupportedDiffusionSchedulerType('euler_a', 'EulerAncestralDiscreteScheduler'), /model scheduler class "EulerAncestralDiscreteScheduler"/);

// Load-time gates: unsupported layouts and scheduler classes fail when the
// pipeline initializes, before any text encoder or transformer work.
function createManifest({ layout = 'sd3', schedulerClass = 'FlowMatchEulerDiscreteScheduler' } = {}) {
  const scheduler = schedulerClass
    ? { config: { _class_name: schedulerClass, num_train_timesteps: 1000, shift: 3.0 } }
    : { config: { num_train_timesteps: 1000 } };
  return {
    config: {
      diffusion: {
        layout,
        components: {
          vae: { config: { sample_size: 64, latent_channels: 16, norm_num_groups: 32, scaling_factor: 1.5305 } },
          transformer: { config: { sample_size: 8 } },
          scheduler,
        },
      },
    },
  };
}
const runtimeConfig = { inference: { diffusion: { backend: { pipeline: 'gpu' } } } };
{
  const state = initializeDiffusion(createManifest(), runtimeConfig);
  assert.equal(state.runtime.scheduler.type, 'flowmatch_euler');
  assert.equal(state.runtime.scheduler.shift, 3.0);
}
assert.throws(() => initializeDiffusion(createManifest({ layout: 'flux' }), runtimeConfig), /layout "flux".*Supported runtime layouts: sd3\./);
assert.throws(() => initializeDiffusion(createManifest({ schedulerClass: 'EulerDiscreteScheduler' }), runtimeConfig), /"euler" from model scheduler class "EulerDiscreteScheduler" is not implemented/);
assert.throws(() => initializeDiffusion(createManifest({ schedulerClass: 'DPMSolverMultistepScheduler' }), runtimeConfig), /"dpmpp_2m"/);
assert.throws(() => initializeDiffusion(createManifest({ schedulerClass: null }), runtimeConfig), /"ddim" from runtime scheduler default is not implemented/);
console.log('diffusion flow-match scheduler and runtime gates ok');
