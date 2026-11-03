import type { CommandRecorder } from '../command-recorder.js';
/** Caller owns status (four u32 words), its initialization and the eventual readback. */
export function recordCheckFinitenessF32(recorder: CommandRecorder, input: GPUBuffer,
  size: number, status: GPUBuffer, layer: number, step: number, absThreshold: number): Promise<void>;
