import type { CommandRecorder } from '../../../gpu/command-recorder.js';

export declare function createDiffusionBufferReleaser(
  recorder: CommandRecorder | null | undefined
): (buffer: GPUBuffer | null | undefined) => void;

export declare function createDiffusionBufferDestroyer(
  recorder: CommandRecorder | null | undefined
): (buffer: GPUBuffer | null | undefined) => void;

export declare function createDiffusionIndexBuffer(
  device: GPUDevice,
  indices: Uint32Array,
  label: string
): GPUBuffer;
