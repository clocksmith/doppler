import type { CommandRecorder } from '../../command-recorder.js';
import { KernelBase } from '../kernel-base.js';

export type AttentionWorkgroups = number | [number, number, number];

/** One attention execution: recorded when a recorder is present, immediate otherwise. */
export interface AttentionExecution {
  recorder: CommandRecorder | null;
  /** The recorder's device, else the current device (null when none is active). */
  device: GPUDevice | null;
}

/** A kernel that both dispatches immediately and records into a recorder. */
export interface AttentionDispatcher {
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
  record(recorder: CommandRecorder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
}

export interface AttentionUniformParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  kvLen: number;
  seqLen: number;
  scale: number;
  causal: boolean;
  startPos: number;
  attnSoftcap: number;
  slidingWindow: number;
  /** 0 reads kvLen from the uniform, 1 from the kvLen buffer. */
  kvLenSource: number;
  kvStart?: number | null;
  pageSize?: number | null;
  kvLayout?: number | null;
  bidirectionalSpanStart?: number | null;
  bidirectionalSpanLength?: number | null;
}

export type BDPAAttentionUniformParams = Pick<AttentionUniformParams,
  'numHeads' | 'numKVHeads' | 'headDim' | 'kvLen' | 'seqLen' | 'scale' | 'causal' | 'startPos' | 'attnSoftcap' | 'slidingWindow'>;

export declare let kvLenFallbackBuffer: GPUBuffer | null;
export declare let kvLenFallbackBufferEpoch: number;
export declare const U32_BYTES: number;
export declare const F32_BYTES: number;
/** A one-word zero kvLen binding, recreated for each device epoch. */
export declare function getKvLenFallbackBuffer(device: GPUDevice): GPUBuffer;
export declare let pageTableFallbackBuffer: GPUBuffer | null;
export declare let pageTableFallbackBufferEpoch: number;
/** A one-word zero page-table binding, recreated for each device epoch. */
export declare function getPageTableFallbackBuffer(device: GPUDevice): GPUBuffer;

export declare class AttentionKernel extends KernelBase implements AttentionDispatcher {
  getPipeline(variant: string, constants?: Record<string, number | boolean> | null): Promise<GPUComputePipeline>;
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
  record(recorder: CommandRecorder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
}

export declare class AttentionBDPAKernel extends KernelBase implements AttentionDispatcher {
  getPipeline(variant: string): Promise<GPUComputePipeline>;
  dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
  record(recorder: CommandRecorder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups): void;
}

export declare function createAttentionUniformBuffer(
  device: GPUDevice, recorder: CommandRecorder | null, params: AttentionUniformParams
): GPUBuffer;
export declare function createBDPAAttentionUniformBuffer(
  device: GPUDevice, recorder: CommandRecorder | null, params: BDPAAttentionUniformParams
): GPUBuffer;
export declare function resolveAttentionExecution(recorder: CommandRecorder | null | undefined): AttentionExecution;
/** Throws unless `buffer` is a GPUBuffer, naming the kernel, variant and binding. */
export declare function assertAttentionBindGroupBuffer(
  kernelName: string, variant: string, bindingIndex: number, bindingLabel: string, buffer: unknown,
  details?: Array<string | null | undefined | false>
): void;
/** Immediate executions release their uniform; recorded ones leave it to the recorder. */
export declare function releaseAttentionUniform(execution: AttentionExecution, uniformBuffer: GPUBuffer): void;
export declare function dispatchAttentionKernel(
  execution: AttentionExecution, kernel: AttentionDispatcher, pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup, workgroups: AttentionWorkgroups
): void;
