/**
 * Dispatch Helpers - Simplified GPU kernel dispatch
 *
 * Provides helpers to reduce boilerplate for common dispatch patterns:
 * - Single submit dispatch
 * - CommandRecorder dispatch (batched)
 * - Multi-dimensional dispatch
 */

import type { CommandRecorder } from '../command-recorder.js';

/** Diagnostic-only device lease. Caller must own all dispatches on this device;
 * concurrent leases reject. Each immediate kernel still submits separately.
 * Release before allowing another executor to use the device. */
export declare function beginImmediateDispatchProfile(
  device: GPUDevice,
  getRecorder: () => CommandRecorder
): () => void;

/** Borrow the active diagnostic owner for a compound immediate operator. */
export declare function getImmediateDispatchProfileRecorder(device: GPUDevice): CommandRecorder | null;

/** Validate a direct grid before allocating resources or recording into an existing pass. */
export declare function assertWorkgroupLimits(
  device: GPUDevice,
  workgroups: number | [number, number, number],
  label: string
): [number, number, number];

/**
 * Dispatch a single compute pass and submit immediately
 * Use for standalone kernels that don't participate in batching
 */
export declare function dispatch(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Polymorphic dispatch that picks between submit-and-dispatch (GPUDevice)
 * and record-only (CommandRecorder) based on the target's shape.
 */
export declare function dispatchKernel(
  target: GPUDevice | CommandRecorder | null,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Record a compute pass to a CommandRecorder (no submit)
 * Use for kernels in the batched pipeline path
 */
export declare function recordDispatch(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number | [number, number, number],
  label?: string
): void;

/**
 * Dispatch a single compute pass using an indirect dispatch buffer
 * Use when workgroup counts are produced on GPU
 */
export declare function dispatchIndirect(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset?: number,
  label?: string
): void;

/**
 * Record an indirect dispatch into a CommandRecorder (no submit)
 */
export declare function recordDispatchIndirect(
  recorder: CommandRecorder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  indirectBuffer: GPUBuffer,
  indirectOffset?: number,
  label?: string
): void;

