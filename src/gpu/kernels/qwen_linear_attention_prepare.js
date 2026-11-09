import { getDevice } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { dispatch } from './dispatch.js';
import { createPipeline } from './pipeline-cache.js';
import { createUniformBufferWithView } from './uniform-utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { requirePositiveInteger } from './kernel-geometry.js';
import { withKernelOutputs } from './kernel-execution.js';

const WORKGROUP_SIZE = 128;
const CONTEXT = 'Qwen linear-attention prepare';

function requireElements(tensor, expected, label) {
  if (tensor?.dtype !== 'f32') {
    throw new Error(`Qwen linear-attention prepare requires f32 ${label}.`);
  }
  const actual = tensor.shape?.reduce((product, value) => product * value, 1);
  if (actual !== expected) {
    throw new Error(`Qwen linear-attention prepare requires ${label} to have ${expected} elements.`);
  }
}

function resolveDimensions(mixed, a, b, aLog, dtBias, options) {
  const numTokens = requirePositiveInteger(options?.numTokens, 'numTokens', CONTEXT);
  const numKeyHeads = requirePositiveInteger(options?.numKeyHeads, 'numKeyHeads', CONTEXT);
  const numValueHeads = requirePositiveInteger(options?.numValueHeads, 'numValueHeads', CONTEXT);
  const keyDim = requirePositiveInteger(options?.keyDim, 'keyDim', CONTEXT);
  const valueDim = requirePositiveInteger(options?.valueDim, 'valueDim', CONTEXT);
  const eps = Number(options?.eps);
  if (numValueHeads % numKeyHeads !== 0) {
    throw new Error('numValueHeads must be divisible by numKeyHeads.');
  }
  if (!Number.isFinite(eps) || eps <= 0) {
    throw new Error('eps must be finite and positive.');
  }
  const querySize = numKeyHeads * keyDim;
  const keySize = querySize;
  const valueSize = numValueHeads * valueDim;
  const convSize = querySize + keySize + valueSize;
  const scalarElements = numTokens * numValueHeads;
  requireElements(mixed, numTokens * convSize, 'mixed');
  requireElements(a, scalarElements, 'a');
  requireElements(b, scalarElements, 'b');
  requireElements(aLog, numValueHeads, 'aLog');
  requireElements(dtBias, numValueHeads, 'dtBias');
  return {
    numTokens,
    numKeyHeads,
    numValueHeads,
    keyDim,
    valueDim,
    eps,
    querySize,
    keySize,
    valueSize,
    convSize,
    repeatFactor: numValueHeads / numKeyHeads,
    queryElements: numTokens * numValueHeads * keyDim,
    valueElements: numTokens * valueSize,
    scalarElements,
  };
}

export async function runQwenLinearAttentionPrepare(
  mixed,
  a,
  b,
  aLog,
  dtBias,
  options = {}
) {
  const dims = resolveDimensions(mixed, a, b, aLog, dtBias, options);
  const device = getDevice();
  if (!device) throw new Error('Qwen linear-attention prepare requires an active GPU device.');
  const elementBytes = Float32Array.BYTES_PER_ELEMENT;
  const headBytes = dims.numValueHeads * elementBytes;
  const queryBytes = dims.queryElements * elementBytes;
  const valueBytes = dims.valueElements * elementBytes;
  const scalarBytes = dims.scalarElements * elementBytes;
  // The five outputs are one transaction; the packed-parameter scratch buffer
  // is a temporary owned by this call and released after the queue settles.
  return withKernelOutputs(device, {
    query: { bytes: queryBytes, label: 'qwen_linear_prepare_query' },
    key: { bytes: queryBytes, label: 'qwen_linear_prepare_key' },
    value: { bytes: valueBytes, label: 'qwen_linear_prepare_value' },
    logDecay: { bytes: scalarBytes, label: 'qwen_linear_prepare_log_decay' },
    beta: { bytes: scalarBytes, label: 'qwen_linear_prepare_beta' },
  }, async ({ query, key, value, logDecay, beta }) => {
    let packedParameters = null;
    let uniformBuffer = null;
    try {
      packedParameters = acquireBuffer(headBytes * 2, undefined, 'qwen_linear_prepare_parameters');
      const packEncoder = device.createCommandEncoder({ label: 'qwen_linear_prepare_pack' });
      packEncoder.copyBufferToBuffer(aLog.buffer, 0, packedParameters, 0, headBytes);
      packEncoder.copyBufferToBuffer(dtBias.buffer, 0, packedParameters, headBytes, headBytes);
      device.queue.submit([packEncoder.finish()]);

      const pipeline = await createPipeline('qwen_linear_attention_prepare', 'default');
      uniformBuffer = createUniformBufferWithView(
        'qwen_linear_attention_prepare_uniforms',
        48,
        (view) => {
          view.setUint32(0, dims.numTokens, true);
          view.setUint32(4, dims.numKeyHeads, true);
          view.setUint32(8, dims.numValueHeads, true);
          view.setUint32(12, dims.keyDim, true);
          view.setUint32(16, dims.valueDim, true);
          view.setUint32(20, dims.convSize, true);
          view.setUint32(24, dims.querySize, true);
          view.setUint32(28, dims.keySize, true);
          view.setUint32(32, dims.valueSize, true);
          view.setUint32(36, dims.repeatFactor, true);
          view.setFloat32(40, dims.eps, true);
        },
        null,
        device
      );
      const bindGroup = device.createBindGroup({
        label: 'qwen_linear_attention_prepare_bind_group',
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: mixed.buffer } },
          { binding: 2, resource: { buffer: a.buffer } },
          { binding: 3, resource: { buffer: b.buffer } },
          { binding: 4, resource: { buffer: packedParameters } },
          { binding: 5, resource: { buffer: query } },
          { binding: 6, resource: { buffer: key } },
          { binding: 7, resource: { buffer: value } },
          { binding: 8, resource: { buffer: logDecay } },
          { binding: 9, resource: { buffer: beta } },
        ],
      });
      const sourceRows = dims.numTokens * dims.numKeyHeads;
      const linearElements = dims.valueElements + (dims.scalarElements * 2);
      const workgroups = (sourceRows * 2) + Math.ceil(linearElements / WORKGROUP_SIZE);
      dispatch(device, pipeline, bindGroup, workgroups, 'qwen_linear_attention_prepare');
      await device.queue.onSubmittedWorkDone();
      return {
        query: createTensor(query, 'f32', [dims.numTokens, dims.numValueHeads, dims.keyDim], 'qwen_linear_prepare_query'),
        key: createTensor(key, 'f32', [dims.numTokens, dims.numValueHeads, dims.keyDim], 'qwen_linear_prepare_key'),
        value: createTensor(value, 'f32', [dims.numTokens, dims.numValueHeads, dims.valueDim], 'qwen_linear_prepare_value'),
        logDecay: createTensor(logDecay, 'f32', [dims.numTokens, dims.numValueHeads], 'qwen_linear_prepare_log_decay'),
        beta: createTensor(beta, 'f32', [dims.numTokens, dims.numValueHeads], 'qwen_linear_prepare_beta'),
      };
    } finally {
      await device.queue.onSubmittedWorkDone().catch(() => {});
      if (uniformBuffer) releaseUniformBuffer(uniformBuffer);
      if (packedParameters) releaseBuffer(packedParameters);
    }
  });
}
