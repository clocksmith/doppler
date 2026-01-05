/**
 * Dispatch Helpers - Simplified GPU kernel dispatch
 *
 * Provides helpers to reduce boilerplate for common dispatch patterns:
 * - Single submit dispatch
 * - CommandRecorder dispatch (batched)
 * - Multi-dimensional dispatch
 */

import type { CommandRecorder } from '../command-recorder.js';

/**
 * Dispatch a single compute pass and submit immediately
 * Use for standalone kernels that don't participate in batching
 */
export function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label: string = 'compute'
): void {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Record a compute pass to a CommandRecorder (no submit)
 * Use for kernels in the batched pipeline path
 */
export function recordDispatch(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label: string = 'compute'
): void {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  pass.end();
}

/**
 * Dispatch a single compute pass using an indirect dispatch buffer
 * Use when workgroup counts are produced on GPU
 */
export function dispatchIndirect(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset: number = 0,
  label: string = 'compute'
): void {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Record an indirect dispatch into a CommandRecorder (no submit)
 */
export function recordDispatchIndirect(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset: number = 0,
  label: string = 'compute'
): void {
  const pass = recorder.beginComputePass(label);
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroupsIndirect(indirectBuffer, indirectOffset);
  pass.end();
}

/**
 * Dispatch with multiple bind groups
 * For kernels that use multiple bind group sets
 */
export function dispatchMultiBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroups: GPUBindGroup[],
  workgroups: number | [number, number, number],
  label: string = 'compute'
): void {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const pass = encoder.beginComputePass({ label: `${label}_pass` });
  pass.setPipeline(pipeline);

  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Calculate workgroup count for 1D dispatch
 * @param totalThreads - Total number of threads needed
 * @param workgroupSize - Threads per workgroup (default: 256)
 * @returns Number of workgroups (rounded up)
 */
export function calculateWorkgroups1D(
  totalThreads: number,
  workgroupSize: number = 256
): number {
  return Math.ceil(totalThreads / workgroupSize);
}

/**
 * Calculate workgroup count for 2D dispatch
 * @param width - Width dimension (e.g., matrix columns)
 * @param height - Height dimension (e.g., matrix rows)
 * @param tileSize - Tile size per workgroup (default: 16)
 * @returns [workgroupsX, workgroupsY]
 */
export function calculateWorkgroups2D(
  width: number,
  height: number,
  tileSize: number = 16
): [number, number] {
  return [
    Math.ceil(width / tileSize),
    Math.ceil(height / tileSize),
  ];
}

/**
 * Calculate workgroup count for 3D dispatch
 * @param width - Width dimension
 * @param height - Height dimension
 * @param depth - Depth dimension
 * @param tileSizeX - Tile size in X (default: 16)
 * @param tileSizeY - Tile size in Y (default: 16)
 * @param tileSizeZ - Tile size in Z (default: 1)
 * @returns [workgroupsX, workgroupsY, workgroupsZ]
 */
export function calculateWorkgroups3D(
  width: number,
  height: number,
  depth: number,
  tileSizeX: number = 16,
  tileSizeY: number = 16,
  tileSizeZ: number = 1
): [number, number, number] {
  return [
    Math.ceil(width / tileSizeX),
    Math.ceil(height / tileSizeY),
    Math.ceil(depth / tileSizeZ),
  ];
}

/**
 * Dispatch options for advanced use cases
 */
export interface DispatchOptions {
  /** Custom label for encoder and pass */
  label?: string;

  /** Bind groups (default: single group at index 0) */
  bindGroups?: GPUBindGroup[];

  /** Push constants (if supported) */
  pushConstants?: ArrayBuffer;

  /** Timestamp queries (if available) */
  timestampWrites?: GPUComputePassTimestampWrites;
}

/**
 * Advanced dispatch with full control
 * Supports push constants, timestamps, and multiple bind groups
 */
export function dispatchAdvanced(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  workgroups: number | [number, number, number],
  options: DispatchOptions = {}
): void {
  const {
    label = 'compute',
    bindGroups = [],
    timestampWrites,
  } = options;

  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });
  const passDescriptor: GPUComputePassDescriptor = {
    label: `${label}_pass`,
  };

  if (timestampWrites) {
    passDescriptor.timestampWrites = timestampWrites;
  }

  const pass = encoder.beginComputePass(passDescriptor);
  pass.setPipeline(pipeline);

  // Set bind groups
  for (let i = 0; i < bindGroups.length; i++) {
    pass.setBindGroup(i, bindGroups[i]);
  }

  // Dispatch
  if (typeof workgroups === 'number') {
    pass.dispatchWorkgroups(workgroups);
  } else {
    pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  }

  pass.end();
  device.queue.submit([encoder.finish()]);
}

/**
 * Batch multiple dispatches in a single command buffer
 * Useful for multi-kernel operations that should be submitted together
 */
export function dispatchBatch(
  device: GPUDevice,
  batches: Array<{
    pipeline: GPUComputePipeline;
    bindGroup: GPUBindGroup;
    workgroups: number | [number, number, number];
    label?: string;
  }>,
  label: string = 'batch'
): void {
  const encoder = device.createCommandEncoder({ label: `${label}_encoder` });

  for (const batch of batches) {
    const pass = encoder.beginComputePass({ label: batch.label || `${label}_pass` });
    pass.setPipeline(batch.pipeline);
    pass.setBindGroup(0, batch.bindGroup);

    if (typeof batch.workgroups === 'number') {
      pass.dispatchWorkgroups(batch.workgroups);
    } else {
      pass.dispatchWorkgroups(batch.workgroups[0], batch.workgroups[1], batch.workgroups[2]);
    }

    pass.end();
  }

  device.queue.submit([encoder.finish()]);
}
