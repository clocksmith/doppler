/**
 * Kernel Base - Shared dispatch and pipeline helpers for kernel wrappers.
 */

import type { CommandRecorder } from '../command-recorder.js';

export declare abstract class KernelBase {
  protected readonly device: GPUDevice;

  constructor(device: GPUDevice);

  protected getPipelineFor(
    operation: string,
    variant: string,
    bindGroupLayout?: GPUBindGroupLayout | null,
    constants?: Record<string, number | boolean> | null
  ): Promise<GPUComputePipeline>;

  protected dispatchKernel(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | [number, number, number],
    label: string
  ): void;

  protected recordKernel(
    recorder: CommandRecorder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: number | [number, number, number],
    label: string
  ): void;
}
