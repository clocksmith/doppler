import assert from 'node:assert/strict';
import { probeNodeGPU } from '../helpers/gpu-probe.js';
import { getDevice, destroyDevice } from '../../src/gpu/device.js';
import { CommandRecorder } from '../../src/gpu/command-recorder.js';
import { acquireBuffer, releaseBuffer, readBuffer, getBufferPool } from '../../src/memory/buffer-pool.js';
import { processLayerPlanGPU } from '../../src/inference/pipelines/text/layer-plan-gpu.js';
import { resolveLayerPipeline } from '../../src/inference/pipelines/text/layer-plan.js';

const probe = await probeNodeGPU({ installFileFetchShim: true });
if (probe.ready) {
  const device = getDevice(), H = 128, T = 2;
  const input = Float32Array.from({ length: T * H * 2 }, (_, i) => i % 2 ? 2 ** -28 : ((i % 43) - 21) / 32);
  const norm = Float32Array.from({ length: H }, (_, i) => 1 + i / 256);
  const buffer = acquireBuffer(input.byteLength, undefined, 'paired_plan_input');
  device.queue.writeBuffer(buffer, 0, input);
  const normStep = { op: 'rmsnorm', src: 'state', dst: 'state', weight: 'input' };
  const steps = [{ op: 'save', src: 'state', name: 'residual' }, normStep,
    { op: 'save', src: 'state', name: 'alias' },
    { op: 'load', name: 'alias', dst: 'state' },
    { op: 'cast', src: 'state', dst: 'state', fromDtype: 'f32', toDtype: 'f32' },
    { op: 'residual_add', a: 'state', b: 'residual', dst: 'state' }];
  const makeContext = (selected, recorder) => ({
    config: { hiddenSize: H, numHeads: 1, rmsNormEps: 1e-6 },
    weightConfig: { rmsNormWeightOffset: false }, debugFlags: {},
    activationDtype: 'f32', activationPrecision: 'two-f32', activationComponents: 2,
    activationElementStrideBytes: 8, finitenessGuardEnabled: false,
    pipelinePlan: resolveLayerPipeline({ steps: selected }, null, 1), recorder,
  });
  const reference = [];
  for (let t = 0; t < T; t++) {
    const row = Array.from({ length: H }, (_, i) => input[(t * H + i) * 2] + input[(t * H + i) * 2 + 1]);
    const inverse = 1 / Math.sqrt(row.reduce((s, x) => s + x * x, 0) / H + 1e-6);
    reference.push(...row.map((x, i) => x * inverse * norm[i] + x));
  }
  const results = [];
  try {
    const active = getBufferPool().getStats().activeBuffers;
    for (const recorded of [false, true]) {
      const recorder = recorded ? new CommandRecorder(device, 'paired_plan') : null;
      let output;
      try {
        output = await processLayerPlanGPU(0, buffer, T, true, T * H, makeContext(steps, recorder), { inputNorm: norm }, {});
        if (recorder) await recorder.submitAndWait();
        const components = new Float32Array(await readBuffer(output, T * H * 8));
        let maximum = 0;
        for (let i = 0; i < reference.length; i++) maximum = Math.max(maximum, Math.abs(components[2 * i] + components[2 * i + 1] - reference[i]));
        assert(maximum <= 1e-11, `paired layer plan max error ${maximum}`);
        results.push({ recorded, maximum, components: Array.from(components) });
      } finally { if (recorder) await recorder.abort(); if (output) releaseBuffer(output); }
      assert.equal(getBufferPool().getStats().activeBuffers, active, 'slot aliases release intermediates exactly once');
      const rejected = recorded ? new CommandRecorder(device, 'paired_plan_failure') : null;
      try {
        await assert.rejects(processLayerPlanGPU(0, buffer, T, true, T * H,
          makeContext([normStep, { ...steps.at(-1), b: 'state', outputDtype: 'f16' }], rejected), { inputNorm: norm }, {}), /output dtype mismatch/);
      } finally { if (rejected) await rejected.abort(); }
      assert.equal(getBufferPool().getStats().activeBuffers, active, 'rejected output dtype releases the actual output');
    }
    await assert.rejects(processLayerPlanGPU(0, buffer, T, true, T * H,
      makeContext([{ op: 'cast', src: 'state', dst: 'state', toDtype: 'f16' }], null), {}, {}), /preserving activation storage/);
    assert.equal(getBufferPool().getStats().activeBuffers, active);
    console.log(JSON.stringify({ schema: 'doppler.paired-layer-plan-control/v1', passed: true,
      input: Array.from(input), norm: Array.from(norm), reference, results, tolerance: 1e-11, provider: probe.providerReceipt }));
  } finally { releaseBuffer(buffer); }
} else console.log(JSON.stringify({ passed: false, skipped: probe.reason }));
destroyDevice();
