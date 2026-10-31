import { getDevice, getDeviceEpoch } from '../../device.js';
import { KernelBase } from '../kernel-base.js';
import { createUniformBufferWithView } from '../uniform-utils.js';
import { releaseUniformBuffer } from '../../uniform-cache.js';

/** @typedef {import('../../command-recorder.js').CommandRecorder} CommandRecorder */
/** @typedef {import('./executors-components.js').AttentionWorkgroups} AttentionWorkgroups */

// Attention execution components shared by the attention executors: device-epoch
// fallback bindings, the kernel classes, uniform construction and the immediate or
// recorded dispatch and uniform release of one execution.

/** @type {GPUBuffer | null} */
export let kvLenFallbackBuffer = null;
export let kvLenFallbackBufferEpoch = -1;
export const U32_BYTES = Uint32Array.BYTES_PER_ELEMENT;
export const F32_BYTES = Float32Array.BYTES_PER_ELEMENT;
/** @type {typeof import('./executors-components.js').getKvLenFallbackBuffer} */
export function getKvLenFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!kvLenFallbackBuffer || kvLenFallbackBufferEpoch !== epoch) {
    const buffer = device.createBuffer({
      label: 'attention_kv_len_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, new Uint32Array([0]));
    kvLenFallbackBuffer = buffer;
    kvLenFallbackBufferEpoch = epoch;
    return buffer;
  }
  return kvLenFallbackBuffer;
}

/** @type {GPUBuffer | null} */
export let pageTableFallbackBuffer = null;
export let pageTableFallbackBufferEpoch = -1;
/** @type {typeof import('./executors-components.js').getPageTableFallbackBuffer} */
export function getPageTableFallbackBuffer(device) {
  const epoch = getDeviceEpoch();
  if (!pageTableFallbackBuffer || pageTableFallbackBufferEpoch !== epoch) {
    const buffer = device.createBuffer({
      label: 'attention_page_table_fallback',
      size: U32_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, new Uint32Array([0]));
    pageTableFallbackBuffer = buffer;
    pageTableFallbackBufferEpoch = epoch;
    return buffer;
  }
  return pageTableFallbackBuffer;
}

export class AttentionKernel extends KernelBase {
  /**
   * @param {string} variant
   * @param {Record<string, number | boolean> | null} [constants]
   */
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('attention', variant, null, constants);
  }

  /**
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {AttentionWorkgroups} workgroups
   */
  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention');
  }

  /**
   * @param {CommandRecorder} recorder
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {AttentionWorkgroups} workgroups
   */
  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention');
  }
}

export class AttentionBDPAKernel extends KernelBase {
  /** @param {string} variant */
  async getPipeline(variant) {
    return this.getPipelineFor('attention_bdpa', variant);
  }

  /**
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {AttentionWorkgroups} workgroups
   */
  dispatch(
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'attention_bdpa');
  }

  /**
   * @param {CommandRecorder} recorder
   * @param {GPUComputePipeline} pipeline
   * @param {GPUBindGroup} bindGroup
   * @param {AttentionWorkgroups} workgroups
   */
  record(
    recorder,
    pipeline,
    bindGroup,
    workgroups
  ) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'attention_bdpa');
  }
}

/** @type {typeof import('./executors-components.js').createAttentionUniformBuffer} */
export function createAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_uniforms',
    80,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true); // Gemma 2: 50.0, 0 = disabled
      view.setUint32(36, params.slidingWindow, true); // Sliding window size, 0 = disabled
      view.setUint32(40, params.kvLenSource, true); // 0 = uniform kvLen, 1 = buffer
      view.setUint32(44, params.kvStart ?? 0, true);
      view.setUint32(48, params.pageSize ?? 0, true);
      view.setUint32(52, params.kvLayout ?? 0, true);
      view.setUint32(56, params.bidirectionalSpanStart ?? 0, true);
      view.setUint32(60, params.bidirectionalSpanLength ?? 0, true);
      view.setUint32(64, 0, true);
      view.setUint32(68, 0, true);
      view.setUint32(72, 0, true);
    },
    recorder,
    device
  );
}

/** @type {typeof import('./executors-components.js').createBDPAAttentionUniformBuffer} */
export function createBDPAAttentionUniformBuffer(
  device,
  recorder,
  params
) {
  return createUniformBufferWithView(
    'attention_bdpa_uniforms',
    64,
    (view) => {
      view.setUint32(0, params.numHeads, true);
      view.setUint32(4, params.numKVHeads, true);
      view.setUint32(8, params.headDim, true);
      view.setUint32(12, params.kvLen, true);
      view.setUint32(16, params.seqLen, true);
      view.setFloat32(20, params.scale, true);
      view.setUint32(24, params.causal ? 1 : 0, true);
      view.setUint32(28, params.startPos, true);
      view.setFloat32(32, params.attnSoftcap, true);
      view.setUint32(36, params.slidingWindow, true);
      view.setUint32(40, 0, true); // padding
      view.setUint32(44, 0, true); // padding
      view.setUint32(48, 0, true); // padding
      view.setUint32(52, 0, true); // padding
      view.setUint32(56, 0, true); // padding
      view.setUint32(60, 0, true); // padding
    },
    recorder,
    device
  );
}

/** @type {typeof import('./executors-components.js').resolveAttentionExecution} */
export function resolveAttentionExecution(recorder) {
  return {
    recorder: recorder || null,
    device: recorder?.device || getDevice(),
  };
}

/** @type {typeof import('./executors-components.js').assertAttentionBindGroupBuffer} */
export function assertAttentionBindGroupBuffer(kernelName, variant, bindingIndex, bindingLabel, buffer, details = []) {
  const isGpuBuffer = buffer && (
    typeof GPUBuffer === 'undefined'
      ? true
      : buffer instanceof GPUBuffer
  );
  if (isGpuBuffer) {
    return;
  }
  const detailText = details.filter(Boolean).join(', ');
  throw new Error(
    `[${kernelName}] variant="${variant}" binding ${bindingIndex} "${bindingLabel}" requires a GPUBuffer` +
    (detailText ? ` (${detailText})` : '') +
    '.'
  );
}

/** @type {typeof import('./executors-components.js').releaseAttentionUniform} */
export function releaseAttentionUniform(execution, uniformBuffer) {
  if (!execution.recorder) {
    releaseUniformBuffer(uniformBuffer);
  }
}

/** @type {typeof import('./executors-components.js').dispatchAttentionKernel} */
export function dispatchAttentionKernel(execution, kernel, pipeline, bindGroup, workgroups) {
  if (execution.recorder) {
    kernel.record(execution.recorder, pipeline, bindGroup, workgroups);
    return;
  }
  kernel.dispatch(pipeline, bindGroup, workgroups);
}
