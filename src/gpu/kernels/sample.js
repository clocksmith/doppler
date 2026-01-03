

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { WORKGROUP_SIZES } from './constants.js';
import { createPipeline, createUniformBufferWithView, getOrCreateBindGroupLayout } from './utils.js';
import { allowReadback } from '../perf-guards.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';
import { getKernelThresholds } from '../../config/schema/index.js';


function getSampleBindGroupLayout(device) {
  return getOrCreateBindGroupLayout(
    'sample_bind_group_layout',
    [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
    device
  );
}


async function createSamplePipeline(device, entryPoint) {
  return createPipeline('sample', entryPoint, getSampleBindGroupLayout(device));
}

function resolveSampleVariants(logitsDtype) {
  const caps = getKernelCapabilities();
  const useF16 = logitsDtype === 'f16';
  if (useF16 && !caps.hasF16) {
    throw new Error('[Sample] F16 logits requested but shader-f16 is unavailable.');
  }
  const suffix = selectKernelRuleValue('sample', 'suffix', { useF16 });
  return {
    argmax: `argmax${suffix}`,
    argmaxReduce: `argmax_reduce${suffix}`,
    phase1: `find_topk_phase1${suffix}`,
    phase2: `find_topk_phase2${suffix}`,
    phase3: `softmax_and_sample${suffix}`,
    singlePass: `single_pass${suffix}`,
  };
}


function resolveLogitsDtype(logitsDtype) {
  return selectSharedRuleValue('shared', 'dtype', 'logitsDtype', { logitsDtype });
}

function assertArgmaxOptions(options, recordMode = false) {
  const suffix = recordMode ? ' (record)' : '';
  if (options.logitsDtype == null) {
    throw new Error(`[Sample] logitsDtype is required for argmax${suffix}.`);
  }
  if (options.outputIndex == null) {
    throw new Error(`[Sample] outputIndex is required for argmax${suffix}.`);
  }
  if (options.logitSoftcap === undefined) {
    throw new Error(`[Sample] logitSoftcap is required for argmax${suffix}.`);
  }
  if (options.padTokenId === undefined) {
    throw new Error(`[Sample] padTokenId is required for argmax${suffix}.`);
  }
}

function assertSampleOptions(options, recordMode = false) {
  const suffix = recordMode ? ' (record)' : '';
  if (options.temperature == null) {
    throw new Error(`[Sample] temperature is required for sampling${suffix}.`);
  }
  if (options.topK == null) {
    throw new Error(`[Sample] topK is required for sampling${suffix}.`);
  }
  if (options.logitsDtype == null) {
    throw new Error(`[Sample] logitsDtype is required for sampling${suffix}.`);
  }
  if (options.outputIndex == null) {
    throw new Error(`[Sample] outputIndex is required for sampling${suffix}.`);
  }
  if (options.logitSoftcap === undefined) {
    throw new Error(`[Sample] logitSoftcap is required for sampling${suffix}.`);
  }
  if (options.padTokenId === undefined) {
    throw new Error(`[Sample] padTokenId is required for sampling${suffix}.`);
  }
  if (options.greedyThreshold == null) {
    throw new Error(`[Sample] greedyThreshold is required for sampling${suffix}.`);
  }
}

async function resolveArgmaxPipelines(device, vocabSize, variants) {
  const argmaxPipeline = await createSamplePipeline(device, variants.argmax);
  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));
  const useSinglePassArgmax = numWorkgroups === 1;
  const argmaxReduceVocabThreshold = getKernelThresholds().sample.argmaxReduceVocabThreshold;
  const reducePipeline = useSinglePassArgmax || vocabSize <= argmaxReduceVocabThreshold
    ? null
    : await createSamplePipeline(device, variants.argmaxReduce);
  const singlePassPipeline = useSinglePassArgmax
    ? await createSamplePipeline(device, variants.singlePass)
    : null;

  return {
    argmaxPipeline,
    reducePipeline,
    singlePassPipeline,
    numWorkgroups,
    useSinglePassArgmax,
  };
}

function createArgmaxUniformBuffer(device, recorder, vocabSize, options) {
  const padTokenValue = options.padTokenId == null ? 0xFFFFFFFF : options.padTokenId;
  return createUniformBufferWithView(
    'argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1.0, true);
      view.setFloat32(12, 0.0, true);
      view.setUint32(16, padTokenValue, true);
      view.setFloat32(20, options.logitSoftcap, true);
      view.setUint32(24, options.outputIndex, true);
    },
    recorder,
    device
  );
}

function createSampleUniformBuffer(device, recorder, vocabSize, topK, temperature, randomValue, padTokenId, logitSoftcap, outputIndex) {
  return createUniformBufferWithView(
    'sample_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId == null ? 0xFFFFFFFF : padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);
      view.setUint32(24, outputIndex, true);
    },
    recorder,
    device
  );
}

function ensureOutputBufferSize(outputBuffer, minBytes, label) {
  if (outputBuffer.size < minBytes) {
    throw new Error(`[Sample] outputBuffer too small for ${label}.`);
  }
}

function readTokenFromOutput(device, outputBuffer, outputIndex, label) {
  const stagingBuffer = device.createBuffer({
    label,
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({ label: `${label}_copy` });
  copyEncoder.copyBufferToBuffer(outputBuffer, outputIndex * 4, stagingBuffer, 0, 4);
  device.queue.submit([copyEncoder.finish()]);

  return stagingBuffer;
}

async function executeArgmaxRun(logits, vocabSize, options) {
  if (!allowReadback('sample.runArgmax')) {
    throw new Error('[Sample] GPU readback disabled for argmax');
  }

  const device = getDevice();
  if (!device) throw new Error('GPU device not initialized');

  assertArgmaxOptions(options, false);

  const logitsDtype = resolveLogitsDtype(options.logitsDtype);
  const variants = resolveSampleVariants(logitsDtype);
  const {
    argmaxPipeline,
    reducePipeline,
    singlePassPipeline,
    numWorkgroups,
    useSinglePassArgmax,
  } = await resolveArgmaxPipelines(device, vocabSize, variants);

  const tempLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_indices');
  const outputIndex = options.outputIndex;
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = options.outputBuffer ?? acquireBuffer(minOutputBytes, undefined, 'argmax_output');
  const ownsOutputBuffer = !options.outputBuffer;
  ensureOutputBufferSize(outputBuffer, minOutputBytes, 'argmax outputIndex');

  const uniformBuffer = createArgmaxUniformBuffer(device, null, vocabSize, options);

  const bindGroupLayout = getSampleBindGroupLayout(device);
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: logits } },
    { binding: 2, resource: { buffer: outputBuffer } },
    { binding: 3, resource: { buffer: tempIndices } },
    { binding: 4, resource: { buffer: tempLogits } },
  ];
  const argmaxBindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries,
  });

  const encoder = device.createCommandEncoder({ label: 'argmax_encoder' });

  const pass1 = encoder.beginComputePass({ label: 'argmax_pass1' });
  pass1.setPipeline(singlePassPipeline ?? argmaxPipeline);
  pass1.setBindGroup(0, argmaxBindGroup);
  pass1.dispatchWorkgroups(useSinglePassArgmax ? 1 : numWorkgroups);
  pass1.end();

  if (reducePipeline) {
    const reduceBindGroup = device.createBindGroup({
      label: 'argmax_reduce_bind_group',
      layout: bindGroupLayout,
      entries,
    });

    const pass2 = encoder.beginComputePass({ label: 'argmax_pass2' });
    pass2.setPipeline(reducePipeline);
    pass2.setBindGroup(0, reduceBindGroup);
    pass2.dispatchWorkgroups(1);
    pass2.end();
  }

  device.queue.submit([encoder.finish()]);

  const stagingBuffer = readTokenFromOutput(device, outputBuffer, outputIndex, 'argmax_staging');
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(tempLogits);
  releaseBuffer(tempIndices);
  if (ownsOutputBuffer) {
    releaseBuffer(outputBuffer);
  }

  return tokenId;
}

async function executeArgmaxRecord(recorder, logits, vocabSize, options) {
  const device = recorder.device;

  assertArgmaxOptions(options, true);

  const logitsDtype = resolveLogitsDtype(options.logitsDtype);
  const variants = resolveSampleVariants(logitsDtype);
  const {
    argmaxPipeline,
    reducePipeline,
    singlePassPipeline,
    numWorkgroups,
    useSinglePassArgmax,
  } = await resolveArgmaxPipelines(device, vocabSize, variants);

  const tempLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_indices');
  const outputIndex = options.outputIndex;
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = options.outputBuffer ?? acquireBuffer(minOutputBytes, undefined, 'argmax_output');
  ensureOutputBufferSize(outputBuffer, minOutputBytes, 'argmax outputIndex');

  const uniformBuffer = createArgmaxUniformBuffer(device, recorder, vocabSize, options);

  const bindGroupLayout = getSampleBindGroupLayout(device);
  const entries = [
    { binding: 0, resource: { buffer: uniformBuffer } },
    { binding: 1, resource: { buffer: logits } },
    { binding: 2, resource: { buffer: outputBuffer } },
    { binding: 3, resource: { buffer: tempIndices } },
    { binding: 4, resource: { buffer: tempLogits } },
  ];
  const bindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries,
  });

  const pass1 = recorder.beginComputePass('argmax_phase1');
  pass1.setPipeline(singlePassPipeline ?? argmaxPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(useSinglePassArgmax ? 1 : numWorkgroups);
  pass1.end();

  if (reducePipeline) {
    const reduceBindGroup = device.createBindGroup({
      label: 'argmax_reduce_bind_group',
      layout: bindGroupLayout,
      entries,
    });

    const pass2 = recorder.beginComputePass('argmax_phase2');
    pass2.setPipeline(reducePipeline);
    pass2.setBindGroup(0, reduceBindGroup);
    pass2.dispatchWorkgroups(1);
    pass2.end();
  }

  recorder.trackTemporaryBuffer(tempLogits);
  recorder.trackTemporaryBuffer(tempIndices);

  return outputBuffer;
}

export async function runArgmax(
  logits,
  vocabSize,
  options = {}
) {
  return executeArgmaxRun(logits, vocabSize, options);
}


export async function runGPUSample(
  logits,
  vocabSize,
  options = {}
) {
  if (!allowReadback('sample.runGPUSample')) {
    throw new Error('[Sample] GPU readback disabled for sampling');
  }

  assertSampleOptions(options, false);

  const {
    temperature,
    topK,
    randomSeed,
    padTokenId,
    logitSoftcap,
    greedyThreshold,
    outputBuffer: outputBufferOverride,
    outputIndex,
  } = options;
  const logitsDtype = resolveLogitsDtype(options.logitsDtype);

  if (temperature < greedyThreshold || topK <= 1) {
    return runArgmax(logits, vocabSize, {
      padTokenId,
      logitSoftcap,
      logitsDtype,
      outputBuffer: outputBufferOverride,
      outputIndex,
    });
  }

  const device = getDevice();
  if (!device) throw new Error('GPU device not initialized');

  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  const variants = resolveSampleVariants(logitsDtype);
  const phase1Pipeline = await createSamplePipeline(device, variants.phase1);
  const phase2Pipeline = await createSamplePipeline(device, variants.phase2);
  const phase3Pipeline = await createSamplePipeline(device, variants.phase3);

  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  const topkLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_indices');
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = outputBufferOverride ?? acquireBuffer(minOutputBytes, undefined, 'sample_output');
  const ownsOutputBuffer = !outputBufferOverride;
  ensureOutputBufferSize(outputBuffer, minOutputBytes, 'sample outputIndex');

  const uniformBuffer = createSampleUniformBuffer(
    device,
    null,
    vocabSize,
    topK,
    temperature,
    randomValue,
    padTokenId,
    logitSoftcap,
    outputIndex
  );

  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'sample_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: topkIndices } },
      { binding: 4, resource: { buffer: topkLogits } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: 'sample_encoder' });

  const pass1 = encoder.beginComputePass({ label: 'sample_phase1' });
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  const pass2 = encoder.beginComputePass({ label: 'sample_phase2' });
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  const pass3 = encoder.beginComputePass({ label: 'sample_phase3' });
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  device.queue.submit([encoder.finish()]);

  const stagingBuffer = readTokenFromOutput(device, outputBuffer, outputIndex, 'sample_staging');
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(topkLogits);
  releaseBuffer(topkIndices);
  if (ownsOutputBuffer) {
    releaseBuffer(outputBuffer);
  }

  return tokenId;
}


export async function recordArgmax(
  recorder,
  logits,
  vocabSize,
  options = {}
) {
  return executeArgmaxRecord(recorder, logits, vocabSize, options);
}


export async function recordGPUSample(
  recorder,
  logits,
  vocabSize,
  options = {}
) {
  assertSampleOptions(options, true);

  const {
    temperature,
    topK,
    randomSeed,
    padTokenId,
    logitSoftcap,
    greedyThreshold,
    outputBuffer: outputBufferOverride,
    outputIndex,
  } = options;
  const logitsDtype = resolveLogitsDtype(options.logitsDtype);

  if (temperature < greedyThreshold || topK <= 1) {
    return recordArgmax(recorder, logits, vocabSize, {
      padTokenId,
      logitSoftcap,
      logitsDtype,
      outputBuffer: outputBufferOverride,
      outputIndex,
    });
  }

  const device = recorder.device;

  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  const variants = resolveSampleVariants(logitsDtype);
  const phase1Pipeline = await createSamplePipeline(device, variants.phase1);
  const phase2Pipeline = await createSamplePipeline(device, variants.phase2);
  const phase3Pipeline = await createSamplePipeline(device, variants.phase3);

  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  const topkLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_indices');
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = outputBufferOverride ?? acquireBuffer(minOutputBytes, undefined, 'sample_output');
  ensureOutputBufferSize(outputBuffer, minOutputBytes, 'sample outputIndex');

  const uniformBuffer = createSampleUniformBuffer(
    device,
    recorder,
    vocabSize,
    topK,
    temperature,
    randomValue,
    padTokenId,
    logitSoftcap,
    outputIndex
  );

  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'sample_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: topkIndices } },
      { binding: 4, resource: { buffer: topkLogits } },
    ],
  });

  const pass1 = recorder.beginComputePass('sample_phase1');
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  const pass2 = recorder.beginComputePass('sample_phase2');
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  const pass3 = recorder.beginComputePass('sample_phase3');
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  recorder.trackTemporaryBuffer(topkLogits);
  recorder.trackTemporaryBuffer(topkIndices);

  return outputBuffer;
}


function seededRandom(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}


export function isGPUSamplingAvailable() {
  return getDevice() !== null;
}
