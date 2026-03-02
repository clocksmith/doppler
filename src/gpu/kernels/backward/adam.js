import { getDevice } from '../../device.js';
import { createTensor, dtypeBytes } from '../../tensor.js';
import { WORKGROUP_SIZES } from '../constants.js';
import { dispatch, recordDispatch } from '../dispatch.js';
import { createPipeline, createUniformBufferWithView } from '../utils.js';
import { acquireBuffer, releaseBuffer } from '../../../memory/buffer-pool.js';

const MAX_ADAM_ELEMENTS_PER_DISPATCH = 65535 * WORKGROUP_SIZES.DEFAULT;

async function runAdamChunked(device, pipeline, params, grads, moment1, moment2, options, inferredCount) {
  if (params.dtype !== 'f32' || grads.dtype !== 'f32' || moment1.dtype !== 'f32' || moment2.dtype !== 'f32') {
    throw new Error(
      `adam chunked path currently requires f32 tensors; got params=${params.dtype}, ` +
      `grads=${grads.dtype}, moment1=${moment1.dtype}, moment2=${moment2.dtype}.`
    );
  }
  const bytesPerElement = dtypeBytes(params.dtype);

  for (let offset = 0; offset < inferredCount; offset += MAX_ADAM_ELEMENTS_PER_DISPATCH) {
    const chunkCount = Math.min(MAX_ADAM_ELEMENTS_PER_DISPATCH, inferredCount - offset);
    const chunkBytes = chunkCount * bytesPerElement;
    const chunkOffsetBytes = offset * bytesPerElement;

    const paramsChunkBuffer = acquireBuffer(chunkBytes, undefined, 'adam_params_chunk');
    const gradsChunkBuffer = acquireBuffer(chunkBytes, undefined, 'adam_grads_chunk');
    const mChunkBuffer = acquireBuffer(chunkBytes, undefined, 'adam_m_chunk');
    const vChunkBuffer = acquireBuffer(chunkBytes, undefined, 'adam_v_chunk');

    const copyIn = device.createCommandEncoder();
    copyIn.copyBufferToBuffer(params.buffer, chunkOffsetBytes, paramsChunkBuffer, 0, chunkBytes);
    copyIn.copyBufferToBuffer(grads.buffer, chunkOffsetBytes, gradsChunkBuffer, 0, chunkBytes);
    copyIn.copyBufferToBuffer(moment1.buffer, chunkOffsetBytes, mChunkBuffer, 0, chunkBytes);
    copyIn.copyBufferToBuffer(moment2.buffer, chunkOffsetBytes, vChunkBuffer, 0, chunkBytes);
    device.queue.submit([copyIn.finish()]);

    const paramsChunk = createTensor(paramsChunkBuffer, params.dtype, [chunkCount], 'adam_params_chunk');
    const gradsChunk = createTensor(gradsChunkBuffer, grads.dtype, [chunkCount], 'adam_grads_chunk');
    const mChunk = createTensor(mChunkBuffer, moment1.dtype, [chunkCount], 'adam_m_chunk');
    const vChunk = createTensor(vChunkBuffer, moment2.dtype, [chunkCount], 'adam_v_chunk');

    const uniformBuffer = createUniformBufferWithView(
      'adam_uniforms_chunk',
      32,
      (view) => {
        view.setUint32(0, chunkCount, true);
        view.setUint32(4, options.step, true);
        view.setFloat32(8, options.lr, true);
        view.setFloat32(12, options.beta1, true);
        view.setFloat32(16, options.beta2, true);
        view.setFloat32(20, options.eps, true);
      },
      null,
      device
    );

    const bindGroup = device.createBindGroup({
      label: 'adam_bind_group_chunk',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: paramsChunk.buffer } },
        { binding: 2, resource: { buffer: gradsChunk.buffer } },
        { binding: 3, resource: { buffer: mChunk.buffer } },
        { binding: 4, resource: { buffer: vChunk.buffer } },
      ],
    });

    const workgroups = Math.ceil(chunkCount / WORKGROUP_SIZES.DEFAULT);
    dispatch(device, pipeline, bindGroup, workgroups, 'adam_chunk');
    uniformBuffer.destroy();

    const copyOut = device.createCommandEncoder();
    copyOut.copyBufferToBuffer(paramsChunk.buffer, 0, params.buffer, chunkOffsetBytes, chunkBytes);
    copyOut.copyBufferToBuffer(mChunk.buffer, 0, moment1.buffer, chunkOffsetBytes, chunkBytes);
    copyOut.copyBufferToBuffer(vChunk.buffer, 0, moment2.buffer, chunkOffsetBytes, chunkBytes);
    device.queue.submit([copyOut.finish()]);

    releaseBuffer(paramsChunkBuffer);
    releaseBuffer(gradsChunkBuffer);
    releaseBuffer(mChunkBuffer);
    releaseBuffer(vChunkBuffer);
  }
}

export async function runAdam(
  params,
  grads,
  moment1,
  moment2,
  options = {}
) {
  const device = getDevice();
  const { count, step = 1, lr, beta1, beta2, eps } = options;

  const bytesPerElement = dtypeBytes(params.dtype);
  const inferredCount = count ?? Math.floor(params.buffer.size / bytesPerElement);
  const pipeline = await createPipeline('adam', 'default');

  if (inferredCount > MAX_ADAM_ELEMENTS_PER_DISPATCH) {
    await runAdamChunked(
      device,
      pipeline,
      params,
      grads,
      moment1,
      moment2,
      {
        step,
        lr,
        beta1,
        beta2,
        eps,
      },
      inferredCount
    );
    return createTensor(params.buffer, params.dtype, [...params.shape], 'adam_params');
  }

  const uniformBuffer = createUniformBufferWithView(
    'adam_uniforms',
    32,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setUint32(4, step, true);
      view.setFloat32(8, lr, true);
      view.setFloat32(12, beta1, true);
      view.setFloat32(16, beta2, true);
      view.setFloat32(20, eps, true);
    },
    null,
    device
  );

  const bindGroup = device.createBindGroup({
    label: 'adam_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.buffer } },
      { binding: 2, resource: { buffer: grads.buffer } },
      { binding: 3, resource: { buffer: moment1.buffer } },
      { binding: 4, resource: { buffer: moment2.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  dispatch(device, pipeline, bindGroup, workgroups, 'adam');

  uniformBuffer.destroy();

  return createTensor(params.buffer, params.dtype, [...params.shape], 'adam_params');
}

export async function recordAdam(
  recorder,
  params,
  grads,
  moment1,
  moment2,
  options = {}
) {
  const device = recorder.device;
  const { count, step = 1, lr, beta1, beta2, eps } = options;

  const bytesPerElement = dtypeBytes(params.dtype);
  const inferredCount = count ?? Math.floor(params.buffer.size / bytesPerElement);
  const pipeline = await createPipeline('adam', 'default');

  const uniformBuffer = createUniformBufferWithView(
    'adam_uniforms',
    32,
    (view) => {
      view.setUint32(0, inferredCount, true);
      view.setUint32(4, step, true);
      view.setFloat32(8, lr, true);
      view.setFloat32(12, beta1, true);
      view.setFloat32(16, beta2, true);
      view.setFloat32(20, eps, true);
    },
    recorder
  );

  const bindGroup = device.createBindGroup({
    label: 'adam_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: params.buffer } },
      { binding: 2, resource: { buffer: grads.buffer } },
      { binding: 3, resource: { buffer: moment1.buffer } },
      { binding: 4, resource: { buffer: moment2.buffer } },
    ],
  });

  const workgroups = Math.ceil(inferredCount / WORKGROUP_SIZES.DEFAULT);
  recordDispatch(recorder, pipeline, bindGroup, workgroups, 'adam');

  return createTensor(params.buffer, params.dtype, [...params.shape], 'adam_params');
}
