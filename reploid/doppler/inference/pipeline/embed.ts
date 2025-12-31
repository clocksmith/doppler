/**
 * Token embedding lookup with optional Gemma scaling.
 */

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runGather, recordGather } from '../../gpu/kernel-selector.js';
import { log, trace } from '../../debug/index.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';

export interface EmbedConfig {
  hiddenSize: number;
  vocabSize: number;
  scaleEmbeddings: boolean;
  debug?: boolean;
  recorder?: CommandRecorder;
  /** Pre-allocated output buffer (avoids pool allocation) */
  outputBuffer?: GPUBuffer;
}

export interface ValidationResult {
  min: number;
  max: number;
  mean: number;
  zeros: number;
  nanCount: number;
  infCount: number;
}

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

let scalePipeline: GPUComputePipeline | null = null;

/**
 * Record scale operation (batched, no submit)
 */
export function recordScale(
  recorder: CommandRecorder,
  inputBuffer: GPUBuffer,
  scale: number,
  count: number
): GPUBuffer {
  const device = recorder.device;
  const outputBuffer = acquireBuffer(count * 4, undefined, 'scaled_embed');

  const uniformData = new ArrayBuffer(8);
  const uniformView = new DataView(uniformData);
  uniformView.setFloat32(0, scale, true);
  uniformView.setUint32(4, count, true);

  const uniformBuffer = recorder.createUniformBuffer(uniformData, 'scale_uniforms');

  // Cache pipeline
  if (!scalePipeline) {
    const shaderModule = device.createShaderModule({ code: scaleShaderCode });
    scalePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  const bindGroup = device.createBindGroup({
    layout: scalePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const pass = recorder.beginComputePass('scale');
  pass.setPipeline(scalePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / 256));
  pass.end();

  return outputBuffer;
}

/**
 * Scale GPU buffer (standalone, with submit)
 * @deprecated Use recordScale with CommandRecorder instead
 */
export async function scaleGPUBuffer(
  inputBuffer: GPUBuffer,
  scale: number,
  count: number
): Promise<GPUBuffer> {
  const device = getDevice();
  if (!device) throw new Error('GPU device not available');

  const outputBuffer = acquireBuffer(count * 4, undefined, 'scaled_embed');

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

  // Cache pipeline
  if (!scalePipeline) {
    const shaderModule = device.createShaderModule({ code: scaleShaderCode });
    scalePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  const bindGroup = device.createBindGroup({
    layout: scalePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: inputBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(scalePipeline);
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

export async function embed(
  tokenIds: number[],
  embedBuffer: GPUBuffer,
  config: EmbedConfig
): Promise<GPUBuffer> {
  const { hiddenSize, vocabSize, scaleEmbeddings, debug = false, recorder, outputBuffer: preAllocatedOutput } = config;
  const device = getDevice();
  const numTokens = tokenIds.length;

  if (!device) throw new Error('GPU device not available');

  if (debug) {
    trace.embed(`tokens=${numTokens}, hidden=${hiddenSize}, vocab=${vocabSize}, scaleEmbeddings=${scaleEmbeddings}`);
    trace.embed(`TOKEN_IDS: [${tokenIds.join(', ')}]`);
  }

  const tokenIdBuffer = acquireBuffer(Math.max(numTokens * 4, 256), undefined, 'embed_tokens');
  device.queue.writeBuffer(tokenIdBuffer, 0, new Uint32Array(tokenIds));

  // Use pre-allocated output buffer if provided, otherwise acquire from pool
  const outputBuffer = recorder
    ? await recordGather(recorder, tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, { outputBuffer: preAllocatedOutput })
    : await runGather(tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, { outputBuffer: preAllocatedOutput });

  // Debug: Verify first token embedding
  if (debug && !recorder && tokenIds.length > 0) {
    const firstTokenId = tokenIds[0];
    const sampleSize = Math.min(32 * 4, hiddenSize * 4);
    const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(outputBuffer, 0, staging, 0, sampleSize);
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
  if (recorder) {
    recorder.trackTemporaryBuffer(tokenIdBuffer);
  } else {
    releaseBuffer(tokenIdBuffer);
  }

  if (!scaleEmbeddings) return outputBuffer;

  // Apply Gemma scaling: sqrt(hiddenSize)
  const scaleFactor = Math.sqrt(hiddenSize);

  // Debug: check raw embedding values before scaling
  if (debug && !recorder) {
    const sample = await readBuffer(outputBuffer, Math.min(outputBuffer.size, numTokens * hiddenSize * 4));
    const f32 = new Float32Array(sample);
    const maxAbs = Math.max(...Array.from(f32).map(x => Math.abs(x)));
    trace.embed(`RAW (before scale): maxAbs=${maxAbs.toFixed(4)}, scaleFactor=${scaleFactor.toFixed(4)}`);
  }

  const scaledBuffer = recorder
    ? recordScale(recorder, outputBuffer, scaleFactor, numTokens * hiddenSize)
    : await scaleGPUBuffer(outputBuffer, scaleFactor, numTokens * hiddenSize);
  if (recorder) {
    recorder.trackTemporaryBuffer(outputBuffer);
  } else {
    releaseBuffer(outputBuffer);
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

  return scaledBuffer;
}

export async function validateEmbedding(
  buffer: GPUBuffer,
  label: string,
  numTokens: number,
  hiddenSize: number
): Promise<ValidationResult | null> {
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
