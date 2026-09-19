import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';

function validate(input, gate, weight, options) {
  const rows = Math.floor(Number(options?.rows));
  const width = Math.floor(Number(options?.width));
  const eps = Number(options?.eps);
  if (rows < 1 || width < 1 || !Number.isFinite(eps) || eps <= 0) {
    throw new Error('gated RMSNorm requires rows, width, and positive eps.');
  }
  if (input?.dtype !== 'f32' || gate?.dtype !== 'f32' || weight?.dtype !== 'f32') {
    throw new Error('gated RMSNorm requires f32 input, gate, and weight.');
  }
  return { rows, width, eps };
}

async function execute(recorder, input, gate, weight, options = {}) {
  const dims = validate(input, gate, weight, options);
  const device = recorder?.device || getDevice();
  if (!device) throw new Error('gated RMSNorm requires an active GPU device.');
  const ownsOutput = options.outputBuffer == null;
  const outputBuffer = options.outputBuffer
    || acquireBuffer(dims.rows * dims.width * 4, undefined, 'gated_rmsnorm_output');
  let uniformBuffer = null;
  let completed = false;
  try {
    const pipeline = await createPipeline('gated_rmsnorm', 'default');
    uniformBuffer = createUniformBufferWithView(
      'gated_rmsnorm_uniforms',
      16,
      (view) => {
        view.setUint32(0, dims.rows, true);
        view.setUint32(4, dims.width, true);
        view.setFloat32(8, dims.eps, true);
      },
      recorder || null,
      device
    );
    const bindGroup = device.createBindGroup({
      label: 'gated_rmsnorm_bind_group',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: input.buffer } },
        { binding: 2, resource: { buffer: gate.buffer } },
        { binding: 3, resource: { buffer: weight.buffer } },
        { binding: 4, resource: { buffer: outputBuffer } },
      ],
    });
    if (recorder) {
      recordDispatch(recorder, pipeline, bindGroup, dims.rows, 'gated_rmsnorm');
    } else {
      dispatch(device, pipeline, bindGroup, dims.rows, 'gated_rmsnorm');
    }
    completed = true;
    return createTensor(outputBuffer, 'f32', [dims.rows, dims.width], 'gated_rmsnorm_output');
  } finally {
    if (!recorder && uniformBuffer) releaseUniformBuffer(uniformBuffer);
    if (!completed && ownsOutput) releaseBuffer(outputBuffer);
  }
}

export function runGatedRmsNorm(input, gate, weight, options = {}) {
  return execute(null, input, gate, weight, options);
}

export function recordGatedRmsNorm(recorder, input, gate, weight, options = {}) {
  return execute(recorder, input, gate, weight, options);
}
