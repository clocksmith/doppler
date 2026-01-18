

import { getDevice, getKernelCapabilities } from '../device.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { WORKGROUP_SIZES } from './constants.js';
import { createPipeline, createUniformBufferWithView, getOrCreateBindGroupLayout } from './utils.js';
import { allowReadback } from '../perf-guards.js';
import { selectRuleValue as selectKernelRuleValue } from './rule-registry.js';
import { selectRuleValue as selectSharedRuleValue } from '../../rules/rule-registry.js';


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


export async function runArgmax(
  logits,
  vocabSize,
  options = {}
) {
  if (!allowReadback('sample.runArgmax')) {
    throw new Error('[Sample] GPU readback disabled for argmax');
  }

  const device = getDevice();
  if (!device) throw new Error('GPU device not initialized');

  // Pipelines with explicit layout
  if (options.logitsDtype == null) {
    throw new Error('[Sample] logitsDtype is required for argmax.');
  }
  if (options.outputIndex == null) {
    throw new Error('[Sample] outputIndex is required for argmax.');
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[Sample] logitSoftcap is required for argmax.');
  }
  if (options.padTokenId === undefined) {
    throw new Error('[Sample] padTokenId is required for argmax.');
  }
  const logitsDtype = resolveLogitsDtype(options.logitsDtype);
  const variants = resolveSampleVariants(logitsDtype);
  const argmaxPipeline = await createSamplePipeline(device, variants.argmax);
  const reducePipeline = await createSamplePipeline(device, variants.argmaxReduce);

  // Workgroups for first pass
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));

  // Intermediate buffers
  const tempLogits = acquireBuffer(workgroupSize * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(workgroupSize * 4, undefined, 'argmax_temp_indices');
  const outputIndex = options.outputIndex;
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = options.outputBuffer ?? acquireBuffer(minOutputBytes, undefined, 'argmax_output');
  const ownsOutputBuffer = !options.outputBuffer;
  if (outputBuffer.size < minOutputBytes) {
    throw new Error('[Sample] outputBuffer too small for argmax outputIndex.');
  }

  // Uniforms
  const padTokenId = options.padTokenId;
  const padTokenValue = padTokenId == null ? 0xFFFFFFFF : padTokenId;
  const logitSoftcap = options.logitSoftcap;
  const uniformBuffer = createUniformBufferWithView(
    'argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);     // vocabSize
      view.setUint32(4, 1, true);             // topK (unused for argmax)
      view.setFloat32(8, 1.0, true);          // temperature (unused)
      view.setFloat32(12, 0.0, true);         // randomValue (unused)
      view.setUint32(16, padTokenValue, true);   // padTokenId
      view.setFloat32(20, logitSoftcap, true); // logitSoftcap (Gemma 2: 30.0)
      view.setUint32(24, outputIndex, true); // outputIndex
    },
    null,
    device
  );

  // Bind groups with explicit layout (auto-layout fails for multi-entry-point shaders)
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const argmaxBindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  const reduceBindGroup = device.createBindGroup({
    label: 'argmax_reduce_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },  // Shader may not use, but layout requires
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  // Execute
  const encoder = device.createCommandEncoder({ label: 'argmax_encoder' });

  // Pass 1: Find max per workgroup
  const pass1 = encoder.beginComputePass({ label: 'argmax_pass1' });
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, argmaxBindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Pass 2: Reduce workgroup results
  const pass2 = encoder.beginComputePass({ label: 'argmax_pass2' });
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  device.queue.submit([encoder.finish()]);

  // Read result
  const stagingBuffer = device.createBuffer({
    label: 'argmax_staging',
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({ label: 'argmax_copy' });
  copyEncoder.copyBufferToBuffer(outputBuffer, outputIndex * 4, stagingBuffer, 0, 4);
  device.queue.submit([copyEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  // Cleanup
  stagingBuffer.destroy();
  uniformBuffer.destroy();
  releaseBuffer(tempLogits);
  releaseBuffer(tempIndices);
  if (ownsOutputBuffer) {
    releaseBuffer(outputBuffer);
  }

  return tokenId;
}


export async function runGPUSample(
  logits,
  vocabSize,
  options = {}
) {
  if (!allowReadback('sample.runGPUSample')) {
    throw new Error('[Sample] GPU readback disabled for sampling');
  }

  if (options.temperature == null) {
    throw new Error('[Sample] temperature is required for sampling.');
  }
  if (options.topK == null) {
    throw new Error('[Sample] topK is required for sampling.');
  }
  if (options.logitsDtype == null) {
    throw new Error('[Sample] logitsDtype is required for sampling.');
  }
  if (options.outputIndex == null) {
    throw new Error('[Sample] outputIndex is required for sampling.');
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[Sample] logitSoftcap is required for sampling.');
  }
  if (options.padTokenId === undefined) {
    throw new Error('[Sample] padTokenId is required for sampling.');
  }
  if (options.greedyThreshold == null) {
    throw new Error('[Sample] greedyThreshold is required for sampling.');
  }
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

  // For temperature=0 or very low, use greedy argmax
  if (temperature < greedyThreshold) {
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

  // Generate random value for sampling
  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  // Get pipelines with explicit layout
  const variants = resolveSampleVariants(logitsDtype);
  const phase1Pipeline = await createSamplePipeline(device, variants.phase1);
  const phase2Pipeline = await createSamplePipeline(device, variants.phase2);
  const phase3Pipeline = await createSamplePipeline(device, variants.phase3);

  // Workgroups for phase 1
  const workgroupSize = WORKGROUP_SIZES.DEFAULT;
  const numWorkgroups = Math.min(workgroupSize, Math.ceil(vocabSize / workgroupSize));

  // Buffers
  const topkLogits = acquireBuffer(workgroupSize * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(workgroupSize * 4, undefined, 'topk_indices');
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = outputBufferOverride ?? acquireBuffer(minOutputBytes, undefined, 'sample_output');
  const ownsOutputBuffer = !outputBufferOverride;
  if (outputBuffer.size < minOutputBytes) {
    throw new Error('[Sample] outputBuffer too small for sample outputIndex.');
  }

  // Uniforms
  const uniformBuffer = createUniformBufferWithView(
    'sample_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId == null ? 0xFFFFFFFF : padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
      view.setUint32(24, outputIndex, true);
    },
    null,
    device
  );

  // Bind group with explicit layout (auto-layout fails for multi-entry-point shaders)
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

  // Execute all phases
  const encoder = device.createCommandEncoder({ label: 'sample_encoder' });

  // Phase 1: Find per-workgroup top values
  const pass1 = encoder.beginComputePass({ label: 'sample_phase1' });
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Phase 2: Merge and select top-k
  const pass2 = encoder.beginComputePass({ label: 'sample_phase2' });
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Phase 3: Softmax and sample
  const pass3 = encoder.beginComputePass({ label: 'sample_phase3' });
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  device.queue.submit([encoder.finish()]);

  // Read result (just 4 bytes!)
  const stagingBuffer = device.createBuffer({
    label: 'sample_staging',
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const copyEncoder = device.createCommandEncoder({ label: 'sample_copy' });
  copyEncoder.copyBufferToBuffer(outputBuffer, outputIndex * 4, stagingBuffer, 0, 4);
  device.queue.submit([copyEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const tokenId = new Uint32Array(stagingBuffer.getMappedRange())[0];
  stagingBuffer.unmap();

  // Cleanup
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
  const device = recorder.device;

  if (options.logitsDtype == null) {
    throw new Error('[Sample] logitsDtype is required for argmax (record).');
  }
  if (options.outputIndex == null) {
    throw new Error('[Sample] outputIndex is required for argmax (record).');
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[Sample] logitSoftcap is required for argmax (record).');
  }
  if (options.padTokenId === undefined) {
    throw new Error('[Sample] padTokenId is required for argmax (record).');
  }

  // Pipelines with explicit layout
  const logitsDtype = resolveLogitsDtype(options.logitsDtype);
  const variants = resolveSampleVariants(logitsDtype);
  const argmaxPipeline = await createSamplePipeline(device, variants.argmax);
  const reducePipeline = await createSamplePipeline(device, variants.argmaxReduce);

  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  // Buffers
  const tempLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_logits');
  const tempIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'argmax_temp_indices');
  const outputIndex = options.outputIndex;
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = options.outputBuffer ?? acquireBuffer(minOutputBytes, undefined, 'argmax_output');
  if (outputBuffer.size < minOutputBytes) {
    throw new Error('[Sample] outputBuffer too small for argmax outputIndex.');
  }

  // Uniforms
  const padTokenId = options.padTokenId;
  const padTokenValue = padTokenId == null ? 0xFFFFFFFF : padTokenId;
  const logitSoftcap = options.logitSoftcap;
  const uniformBuffer = createUniformBufferWithView(
    'argmax_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, 1, true);
      view.setFloat32(8, 1.0, true);
      view.setFloat32(12, 0.0, true);
      view.setUint32(16, padTokenValue, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
      view.setUint32(24, outputIndex, true);
    },
    recorder
  );

  // Bind groups with explicit layout
  const bindGroupLayout = getSampleBindGroupLayout(device);
  const bindGroup = device.createBindGroup({
    label: 'argmax_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  // Pass 1
  const pass1 = recorder.beginComputePass('argmax_phase1');
  pass1.setPipeline(argmaxPipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Pass 2 (reuse same bind group since layout is the same)
  const reduceBindGroup = device.createBindGroup({
    label: 'argmax_reduce_bind_group',
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: logits } },
      { binding: 2, resource: { buffer: outputBuffer } },
      { binding: 3, resource: { buffer: tempIndices } },
      { binding: 4, resource: { buffer: tempLogits } },
    ],
  });

  const pass2 = recorder.beginComputePass('argmax_phase2');
  pass2.setPipeline(reducePipeline);
  pass2.setBindGroup(0, reduceBindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Schedule cleanup of temp buffers after submit.
  recorder.trackTemporaryBuffer(tempLogits);
  recorder.trackTemporaryBuffer(tempIndices);

  return outputBuffer;
}


export async function recordGPUSample(
  recorder,
  logits,
  vocabSize,
  options = {}
) {
  if (options.temperature == null) {
    throw new Error('[Sample] temperature is required for sampling (record).');
  }
  if (options.topK == null) {
    throw new Error('[Sample] topK is required for sampling (record).');
  }
  if (options.logitsDtype == null) {
    throw new Error('[Sample] logitsDtype is required for sampling (record).');
  }
  if (options.outputIndex == null) {
    throw new Error('[Sample] outputIndex is required for sampling (record).');
  }
  if (options.logitSoftcap === undefined) {
    throw new Error('[Sample] logitSoftcap is required for sampling (record).');
  }
  if (options.padTokenId === undefined) {
    throw new Error('[Sample] padTokenId is required for sampling (record).');
  }
  if (options.greedyThreshold == null) {
    throw new Error('[Sample] greedyThreshold is required for sampling (record).');
  }
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

  // For temperature=0 or very low, use greedy argmax
  if (temperature < greedyThreshold) {
    return recordArgmax(recorder, logits, vocabSize, {
      padTokenId,
      logitSoftcap,
      logitsDtype,
      outputBuffer: outputBufferOverride,
      outputIndex,
    });
  }

  const device = recorder.device;

  // Generate random value for sampling
  const randomValue = randomSeed !== undefined
    ? seededRandom(randomSeed)
    : Math.random();

  // Get pipelines with explicit layout
  const variants = resolveSampleVariants(logitsDtype);
  const phase1Pipeline = await createSamplePipeline(device, variants.phase1);
  const phase2Pipeline = await createSamplePipeline(device, variants.phase2);
  const phase3Pipeline = await createSamplePipeline(device, variants.phase3);

  // Workgroups for phase 1
  const numWorkgroups = Math.min(WORKGROUP_SIZES.DEFAULT, Math.ceil(vocabSize / WORKGROUP_SIZES.DEFAULT));

  // Buffers
  const topkLogits = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_logits');
  const topkIndices = acquireBuffer(WORKGROUP_SIZES.DEFAULT * 4, undefined, 'topk_indices');
  const minOutputBytes = Math.max(4, (outputIndex + 1) * 4);
  const outputBuffer = outputBufferOverride ?? acquireBuffer(minOutputBytes, undefined, 'sample_output');
  if (outputBuffer.size < minOutputBytes) {
    throw new Error('[Sample] outputBuffer too small for sample outputIndex.');
  }

  // Uniforms
  const uniformBuffer = createUniformBufferWithView(
    'sample_uniforms',
    32,
    (view) => {
      view.setUint32(0, vocabSize, true);
      view.setUint32(4, topK, true);
      view.setFloat32(8, temperature, true);
      view.setFloat32(12, randomValue, true);
      view.setUint32(16, padTokenId == null ? 0xFFFFFFFF : padTokenId, true);
      view.setFloat32(20, logitSoftcap, true);  // Gemma 2: 30.0
      view.setUint32(24, outputIndex, true);
    },
    recorder
  );

  // Bind group with explicit layout
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

  // Phase 1: Find per-workgroup top values
  const pass1 = recorder.beginComputePass('sample_phase1');
  pass1.setPipeline(phase1Pipeline);
  pass1.setBindGroup(0, bindGroup);
  pass1.dispatchWorkgroups(numWorkgroups);
  pass1.end();

  // Phase 2: Merge and select top-k
  const pass2 = recorder.beginComputePass('sample_phase2');
  pass2.setPipeline(phase2Pipeline);
  pass2.setBindGroup(0, bindGroup);
  pass2.dispatchWorkgroups(1);
  pass2.end();

  // Phase 3: Softmax and sample
  const pass3 = recorder.beginComputePass('sample_phase3');
  pass3.setPipeline(phase3Pipeline);
  pass3.setBindGroup(0, bindGroup);
  pass3.dispatchWorkgroups(1);
  pass3.end();

  // Track temp buffers for cleanup
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
