import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { WeightBuffer } from '../../../gpu/weight-buffer.js';

export function resolveDiffusionActivationDtype(runtime: { latent?: { dtype?: string } } | null | undefined): 'f16' | 'f32';

export function createDiffusionBufferReleaser(
  recorder: CommandRecorder | null | undefined
): (buffer: GPUBuffer | null | undefined) => void;

export function createDiffusionBufferDestroyer(
  recorder: CommandRecorder | null | undefined
): (buffer: GPUBuffer | null | undefined) => void;

export function createDiffusionIndexBuffer(device: GPUDevice, indices: Uint32Array, label: string): GPUBuffer;

export function expectDiffusionWeight<T>(weight: T | null | undefined, label: string): T;

export function normalizeDiffusionLocationDtype(dtype: string | null | undefined): 'f16' | 'f32' | null;

export function normalizeDiffusionMatmulLocationDtype(dtype: string | null | undefined): string | null;

export function inferDiffusionMatmulDtypeFromBuffer(
  weight: GPUBuffer | WeightBuffer | null | undefined,
  N: number,
  K: number,
  preferred: string | null | undefined
): string | null | undefined;

export function sumDiffusionProfileTimings(timings: Record<string, number> | null | undefined): number | null;
