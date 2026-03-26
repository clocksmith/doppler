
import { getDevice, getKernelCapabilities } from '../device.js';
import { dispatch, recordDispatch } from './dispatch.js';
import { createPipeline, createUniformBufferWithView, getKernelConfig, hasRequiredFeatures } from './utils.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';


function resolveQuantizeVariant(mode) {
  if (mode === 'turboquant_outlier') {
    throw new Error(
      'TurboQuant outlier KV quantize is not supported yet. ' +
      'Outlier-mode high-precision buffers are not wired end to end.'
    );
  }
  return selectKernelRuleValue('kv_quantize', 'variant', { mode });
}

function isTurboQuantMode(mode) {
  return mode === 'turboquant' || mode === 'turboquant_prod';
}


function createQuantizeUniformBuffer(device, recorder, params) {
  return createUniformBufferWithView(
    'kv_quantize_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.numKVHeads, true);
      view.setUint32(4, params.headDim, true);
      view.setUint32(8, params.startPos, true);
      view.setUint32(12, params.numTokens, true);
      view.setUint32(16, params.packedStride, true);
      view.setUint32(20, 0, true);
    },
    recorder,
    device
  );
}


function buildTurboQuantBindEntries(uniformBuffer, keys, values, outputKeys, outputValues, scalesK, scalesV, options) {
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: keys } },
    { binding: 2, resource: { buffer: values } },
    { binding: 3, resource: { buffer: outputKeys } },
    { binding: 4, resource: { buffer: outputValues } },
    { binding: 5, resource: { buffer: scalesK } },
    { binding: 6, resource: { buffer: scalesV } },
    { binding: 7, resource: { buffer: options.rotationMatrixBuffer } },
    { binding: 8, resource: { buffer: options.codebookBoundariesBuffer } },
  ];

  if (options.mode === 'turboquant_prod') {
    // Prod mode: bindings 7-14 (residual + rotation + codebook + QJL)
    entries.length = 7;
    entries.push(
      { binding: 7, resource: { buffer: options.residualKBuffer } },
      { binding: 8, resource: { buffer: options.residualVBuffer } },
      { binding: 9, resource: { buffer: options.residualNormsKBuffer } },
      { binding: 10, resource: { buffer: options.residualNormsVBuffer } },
      { binding: 11, resource: { buffer: options.rotationMatrixBuffer } },
      { binding: 12, resource: { buffer: options.codebookCentroidsBuffer } },
      { binding: 13, resource: { buffer: options.codebookBoundariesBuffer } },
      { binding: 14, resource: { buffer: options.qjlMatrixBuffer } },
    );
  }
  return entries;
}


function dispatchQuantize(device, pipeline, bindGroup, workgroups, uniformBuffer, recorder) {
  if (recorder) {
    recordDispatch(recorder, pipeline, bindGroup, workgroups, 'kv_quantize');
  } else {
    try {
      dispatch(device, pipeline, bindGroup, workgroups, 'kv_quantize');
    } finally {
      uniformBuffer.destroy();
    }
  }
}


async function executeKVQuantize(
  recorder,
  keys,
  values,
  outputKeys,
  outputValues,
  scalesK,
  scalesV,
  options = {}
) {
  const device = recorder?.device || getDevice();
  const {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
    mode = 'int8',
  } = options;

  const variant = resolveQuantizeVariant(mode);
  const config = getKernelConfig('kv_quantize', variant);
  const caps = getKernelCapabilities();
  if (!hasRequiredFeatures(config.requires, caps)) {
    throw new Error(`KV quantize kernel "${variant}" requires unsupported GPU features.`);
  }

  const pipeline = await createPipeline('kv_quantize', variant);
  const uniformBuffer = createQuantizeUniformBuffer(device, recorder, {
    numKVHeads,
    headDim,
    startPos,
    numTokens,
    packedStride,
  });

  let entries;
  if (isTurboQuantMode(mode)) {
    entries = buildTurboQuantBindEntries(
      uniformBuffer, keys, values, outputKeys, outputValues, scalesK, scalesV, options
    );
  } else {
    entries = [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: keys } },
      { binding: 2, resource: { buffer: values } },
      { binding: 3, resource: { buffer: outputKeys } },
      { binding: 4, resource: { buffer: outputValues } },
      { binding: 5, resource: { buffer: scalesK } },
      { binding: 6, resource: { buffer: scalesV } },
    ];
  }

  const bindGroup = device.createBindGroup({
    label: 'kv_quantize_bind_group',
    layout: pipeline.getBindGroupLayout(0),
    entries,
  });

  const workgroups = [numKVHeads, numTokens, 1];
  dispatchQuantize(device, pipeline, bindGroup, workgroups, uniformBuffer, recorder);
}


export async function runKVQuantize(
  keys,
  values,
  outputKeys,
  outputValues,
  scalesK,
  scalesV,
  options = {}
) {
  return executeKVQuantize(null, keys, values, outputKeys, outputValues, scalesK, scalesV, options);
}


export async function recordKVQuantize(
  recorder,
  keys,
  values,
  outputKeys,
  outputValues,
  scalesK,
  scalesV,
  options = {}
) {
  return executeKVQuantize(recorder, keys, values, outputKeys, outputValues, scalesK, scalesV, options);
}
