/**
 * Token embedding lookup with optional Gemma scaling.
 */

import { getDevice, getKernelCapabilities } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runGather, recordGather } from '../../gpu/kernel-selector.js';
import { trace } from '../../debug/index.js';
import { runProbes } from './probes.js';
import { createTensor } from '../../gpu/tensor.js';
import { castF32ToF16, recordCastF32ToF16 } from '../../gpu/kernels/cast.js';
import { isCpuWeightBuffer } from '../../gpu/weight-buffer.js';

const scaleShaderCode = `
  struct Uniforms { scale: f32, count: u32 }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> input: array<f32>;
  @group(0) @binding(2) var<storage, read_write> output: array<f32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.count) { return; }
    output[gid.x] = input[gid.x] * uniforms.scale;
  }
`;

// F16 scale shader: F16 input -> F16 output (for F16 activation mode)
const scaleShaderCodeF16 = `
  enable f16;

  struct Uniforms { scale: f32, count: u32 }
  @group(0) @binding(0) var<uniform> uniforms: Uniforms;
  @group(0) @binding(1) var<storage, read> input: array<f16>;
  @group(0) @binding(2) var<storage, read_write> output: array<f16>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= uniforms.count) { return; }
    // Read F16, compute in F32 for precision, write F16
    output[gid.x] = f16(f32(input[gid.x]) * uniforms.scale);
  }
`;

/** @type {GPUComputePipeline | null} */
let scalePipeline = null;
/** @type {GPUComputePipeline | null} */
let scalePipelineF16 = null;

/**
 * Record scale operation (batched, no submit)
 * @param {import('../../gpu/command-recorder.js').CommandRecorder} recorder
 * @param {GPUBuffer} inputBuffer
 * @param {number} scale
 * @param {number} count
 * @param {boolean} [useF16]
 * @returns {GPUBuffer}
 */
export function recordScale(recorder, inputBuffer, scale, count, useF16 = false) {
  const device = recorder.device;
  const bytesPerElement = useF16 ? 2 : 4;
  const outputBuffer = acquireBuffer(count * bytesPerElement, undefined, 'scaled_embed');

  const uniformData = new ArrayBuffer(8);
  const uniformView = new DataView(uniformData);
  uniformView.setFloat32(0, scale, true);
  uniformView.setUint32(4, count, true);

  const uniformBuffer = recorder.createUniformBuffer(uniformData, 'scale_uniforms');

  // Select and cache appropriate pipeline
  /** @type {GPUComputePipeline} */
  let pipeline;
  if (useF16) {
    if (!scalePipelineF16) {
      const shaderModule = device.createShaderModule({ code: scaleShaderCodeF16 });
      scalePipelineF16 = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
    }
    pipeline = scalePipelineF16;
  } else {
    if (!scalePipeline) {
      const shaderModule = device.createShaderModule({ code: scaleShaderCode });
      scalePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
    }
    pipeline = scalePipeline;
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const pass = recorder.beginComputePass('scale');
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();

  return outputBuffer;
}

/**
 * Scale GPU buffer (standalone, with submit)
 * @deprecated Use recordScale with CommandRecorder instead
 * @param {GPUBuffer} inputBuffer
 * @param {number} scale
 * @param {number} count
 * @param {boolean} [useF16]
 * @returns {Promise<GPUBuffer>}
 */
export async function scaleGPUBuffer(inputBuffer, scale, count, useF16 = false) {
  const device = getDevice();
  if (!device) throw new Error('GPU device not available');

  const bytesPerElement = useF16 ? 2 : 4;
  const outputBuffer = acquireBuffer(count * bytesPerElement, undefined, 'scaled_embed');

  const uniformData = new ArrayBuffer(8);
  const uniformView = new DataView(uniformData);
  uniformView.setFloat32(0, scale, true);
  uniformView.setUint32(4, count, true);

  const uniformBuffer = device.createBuffer({
    label: 'scale_uniforms',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  // Select and cache appropriate pipeline
  /** @type {GPUComputePipeline} */
  let pipeline;
  if (useF16) {
    if (!scalePipelineF16) {
      const shaderModule = device.createShaderModule({ code: scaleShaderCodeF16 });
      scalePipelineF16 = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
    }
    pipeline = scalePipelineF16;
  } else {
    if (!scalePipeline) {
      const shaderModule = device.createShaderModule({ code: scaleShaderCode });
      scalePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: shaderModule, entryPoint: 'main' },
      });
    }
    pipeline = scalePipeline;
  }

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();

  device.queue.submit([encoder.finish()]);

  // CRITICAL: Wait for GPU to complete scale operation before returning
  // Without this, the caller may read stale data from outputBuffer
  await device.queue.onSubmittedWorkDone();

  uniformBuffer.destroy();

  return outputBuffer;
}

/**
 * @param {number[] | Uint32Array | GPUBuffer} tokenIds
 * @param {GPUBuffer | Float32Array | import('../../gpu/weight-buffer.js').CpuWeightBuffer} embedBuffer
 * @param {import('./embed.js').EmbedConfig} config
 * @returns {Promise<import('../../gpu/tensor.js').Tensor>}
 */
export async function embed(tokenIds, embedBuffer, config) {
  const { hiddenSize, vocabSize, scaleEmbeddings, debug = false, recorder, outputBuffer: preAllocatedOutput, transpose = false, activationDtype = 'f32', embeddingDtype = 'f32' } = config;
  const device = getDevice();
  const tokenBufferInput = tokenIds instanceof GPUBuffer;
  const tokenIdArray = tokenBufferInput ? null : /** @type {number[] | Uint32Array} */ (tokenIds);
  const numTokens = tokenBufferInput
    ? (config.numTokens ?? 0)
    : (tokenIdArray?.length ?? 0);
  const indexOffset = tokenBufferInput ? (config.indexOffset ?? 0) : 0;

  if (!device) throw new Error('GPU device not available');

  // Check if F16 output is requested and supported
  const caps = getKernelCapabilities();
  const useF16 = activationDtype === 'f16' && caps.hasF16;
  /** @type {import('../../gpu/tensor.js').TensorDtype} */
  const dtype = useF16 ? 'f16' : 'f32';

  const cpuEmbeddings = isCpuWeightBuffer(embedBuffer)
    ? embedBuffer.data
    : embedBuffer instanceof Float32Array
      ? embedBuffer
      : null;

  if (debug) {
    trace.embed(`tokens=${numTokens}, hidden=${hiddenSize}, vocab=${vocabSize}, scaleEmbeddings=${scaleEmbeddings}, transpose=${transpose}, indexOffset=${indexOffset}, activationDtype=${activationDtype}, useF16=${useF16}`);
    if (tokenBufferInput) {
      trace.embed('TOKEN_IDS: [gpu-buffer]');
    } else {
      trace.embed(`TOKEN_IDS: [${Array.from(tokenIdArray ?? []).join(', ')}]`);
    }
  }

  if (cpuEmbeddings) {
    if (tokenBufferInput) {
      throw new Error('[Embed] GPU token buffer requires GPU-resident embeddings.');
    }
    if (debug) {
      trace.embed('Using CPU embedding gather (oversized embedding)');
    }

    const output = new Float32Array(numTokens * hiddenSize);
    if (!transpose) {
      for (let t = 0; t < numTokens; t++) {
        const tokenId = /** @type {number[] | Uint32Array} */ (tokenIdArray)[t];
        const srcOffset = tokenId * hiddenSize;
        output.set(cpuEmbeddings.subarray(srcOffset, srcOffset + hiddenSize), t * hiddenSize);
      }
    } else {
      for (let t = 0; t < numTokens; t++) {
        const tokenId = /** @type {number[] | Uint32Array} */ (tokenIdArray)[t];
        const dstOffset = t * hiddenSize;
        for (let h = 0; h < hiddenSize; h++) {
          output[dstOffset + h] = cpuEmbeddings[h * vocabSize + tokenId];
        }
      }
    }

    if (scaleEmbeddings) {
      const scaleFactor = Math.sqrt(hiddenSize);
      for (let i = 0; i < output.length; i++) {
        output[i] *= scaleFactor;
      }
    }

    if (useF16) {
      const f32Buffer = acquireBuffer(output.byteLength, undefined, 'embed_cpu_f32');
      device.queue.writeBuffer(f32Buffer, 0, output);
      const f32Tensor = createTensor(f32Buffer, 'f32', [numTokens, hiddenSize], 'embed_cpu_f32');
      const outputBytes = numTokens * hiddenSize * 2;
      const outputBuffer = preAllocatedOutput && preAllocatedOutput.size >= outputBytes ? preAllocatedOutput : null;
      const f16Tensor = recorder
        ? await recordCastF32ToF16(recorder, f32Tensor, { outputBuffer })
        : await castF32ToF16(f32Tensor, { outputBuffer });
      if (recorder) {
        recorder.trackTemporaryBuffer(f32Buffer);
      } else {
        releaseBuffer(f32Buffer);
      }
      await runProbes('embed_out', f16Tensor.buffer, {
        numTokens,
        hiddenSize,
        probes: config.debugProbes,
        recorder,
      });
      return f16Tensor;
    }

    const outputBytes = output.byteLength;
    const outputBuffer = preAllocatedOutput && preAllocatedOutput.size >= outputBytes
      ? preAllocatedOutput
      : acquireBuffer(outputBytes, undefined, 'embed_cpu_f32_out');
    device.queue.writeBuffer(outputBuffer, 0, output);
    await runProbes('embed_out', outputBuffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes,
      recorder,
    });
    return createTensor(outputBuffer, dtype, [numTokens, hiddenSize], 'embed_output');
  }

  if (tokenBufferInput && numTokens <= 0) {
    throw new Error('[Embed] numTokens must be provided when tokenIds is a GPUBuffer.');
  }
  const tokenIdBuffer = tokenBufferInput
    ? tokenIds
    : acquireBuffer(Math.max(numTokens * 4, 256), undefined, 'embed_tokens');
  if (!tokenBufferInput) {
    device.queue.writeBuffer(tokenIdBuffer, 0, new Uint32Array(/** @type {number[] | Uint32Array} */ (tokenIdArray)));
  }

  // Use pre-allocated output buffer if provided, otherwise acquire from pool
  // Pass outputDtype to enable F16 output when in F16 activation mode
  // Pass embeddingDtype so gather kernel uses correct input format
  const gatherOptions = {
    outputBuffer: preAllocatedOutput,
    transpose,
    outputDtype: useF16 ? /** @type {'f16'} */ ('f16') : /** @type {'f32'} */ ('f32'),
    embeddingDtype,
    indexOffset,
  };
  if (!(embedBuffer instanceof GPUBuffer)) {
    throw new Error('[Embed] GPU embeddings required for gather path.');
  }
  const gatherOutput = recorder
    ? await recordGather(recorder, tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
    : await runGather(tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions);

  // Debug: Verify first token embedding
  if (debug && !recorder && tokenIdArray && tokenIdArray.length > 0) {
    const firstTokenId = tokenIdArray[0];
    const bytesPerElement = useF16 ? 2 : 4;
    const sampleSize = Math.min(32 * bytesPerElement, hiddenSize * bytesPerElement);
    const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gatherOutput.buffer, 0, staging, 0, sampleSize);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();

    // Compute statistics
    let sum = 0, sumSq = 0;
    for (const v of data) { sum += v; sumSq += v * v; }
    const mean = sum / data.length;
    const variance = (sumSq / data.length) - (mean * mean);
    const std = Math.sqrt(variance);
    const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));

    trace.embed(`FIRST_TOKEN[${firstTokenId}]: maxAbs=${maxAbs.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}, first8=[${Array.from(data).slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
  }
  if (!tokenBufferInput) {
    if (recorder) {
      recorder.trackTemporaryBuffer(tokenIdBuffer);
    } else {
      releaseBuffer(tokenIdBuffer);
    }
  }

  if (!scaleEmbeddings) {
    await runProbes('embed_out', gatherOutput.buffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes,
      recorder,
    });
    return gatherOutput;
  }

  // Apply Gemma scaling: sqrt(hiddenSize)
  const scaleFactor = Math.sqrt(hiddenSize);

  // Debug: check raw embedding values before scaling
  if (debug && !recorder) {
    const sample = await readBuffer(gatherOutput.buffer, Math.min(gatherOutput.buffer.size, numTokens * hiddenSize * 4));
    const f32 = new Float32Array(sample);
    const maxAbs = Math.max(...Array.from(f32).map(x => Math.abs(x)));
    trace.embed(`RAW (before scale): maxAbs=${maxAbs.toFixed(4)}, scaleFactor=${scaleFactor.toFixed(4)}`);
  }

  const scaledBuffer = recorder
    ? recordScale(recorder, gatherOutput.buffer, scaleFactor, numTokens * hiddenSize, useF16)
    : await scaleGPUBuffer(gatherOutput.buffer, scaleFactor, numTokens * hiddenSize, useF16);
  if (recorder) {
    // Only track if we created this buffer (not pre-allocated)
    // Pre-allocated buffers are managed by the caller (e.g., DecodeBufferManager)
    if (!preAllocatedOutput) {
      recorder.trackTemporaryBuffer(gatherOutput.buffer);
    }
  } else {
    // For sync path: only release if not pre-allocated
    if (!preAllocatedOutput) {
      releaseBuffer(gatherOutput.buffer);
    }
  }

  if (debug && !recorder) {
    const sample = await readBuffer(scaledBuffer, Math.min(scaledBuffer.size, numTokens * hiddenSize * 4));
    const f32 = new Float32Array(sample);
    const maxAbs = Math.max(...Array.from(f32).map(x => Math.abs(x)));
    trace.embed(`SCALED (after *${scaleFactor.toFixed(2)}): maxAbs=${maxAbs.toFixed(4)}, buffer.label=${scaledBuffer.label}, buffer.size=${scaledBuffer.size}`);
    trace.embed(`RETURNING buffer with first8=[${Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
    if (f32.some(x => !Number.isFinite(x))) {
      throw new Error('[Embed] Scaled embedding contains NaN/Inf');
    }
  }
  await runProbes('embed_out', scaledBuffer, {
    numTokens,
    hiddenSize,
    probes: config.debugProbes,
    recorder,
  });

  return createTensor(scaledBuffer, dtype, [numTokens, hiddenSize], 'embed_output');
}

/**
 * @param {GPUBuffer} buffer
 * @param {string} label
 * @param {number} numTokens
 * @param {number} hiddenSize
 * @returns {Promise<import('./embed.js').ValidationResult | null>}
 */
export async function validateEmbedding(buffer, label, numTokens, hiddenSize) {
  const device = getDevice();
  if (!device) return null;

  const sampleSize = Math.min(1024 * 4, buffer.size);
  const sample = await readBuffer(buffer, sampleSize);
  const f32 = new Float32Array(sample);

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < f32.length; i++) {
    const v = f32[i];
    if (Number.isNaN(v)) {
      nanCount++;
    } else if (!Number.isFinite(v)) {
      infCount++;
    } else {
      if (v === 0) zeros++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
  }

  const mean = sum / f32.length;

  trace.embed(`${label}: tokens=${numTokens}, hidden=${hiddenSize}`);
  trace.embed(`${label}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}`);
  trace.embed(`${label}: zeros=${zeros}/${f32.length}, NaN=${nanCount}, Inf=${infCount}`);

  return { min, max, mean, zeros, nanCount, infCount };
}
