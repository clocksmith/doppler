import type { CommandRecorder } from '../command-recorder.js';
import type { KernelConfig } from './utils.js';

export declare class MatmulKernel {
  constructor(device: GPUDevice);
  getPipeline(variant: string): Promise<GPUComputePipeline>;
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: GPUExtent3D): void;
  record(recorder: CommandRecorder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: GPUExtent3D): void;
}

export declare function calculateMatmulDispatch(
  variant: string,
  useQ4KFused: boolean,
  useGemv: boolean,
  M: number,
  N: number,
  config: KernelConfig
): { workgroups: GPUExtent3D; uniformWorkgroupsX?: number };

export declare function createMatmulUniformBuffer(
  label: string,
  M: number,
  N: number,
  K: number,
  alpha: number,
  useQ4KFused: boolean,
  transposeB: boolean,
  uniformWorkgroupsX: number | undefined,
  recorder: CommandRecorder | null,
  device: GPUDevice
): GPUBuffer;

export declare function createMatmulBindGroupLayout(): GPUBindGroupLayout;

export declare function getMatmulPipeline(
  variant: string,
  constants: Record<string, number | boolean> | null
): Promise<GPUComputePipeline>;
