/**
 * Kernel Option Types
 *
 * Shared option shapes used across kernel wrappers.
 */

export interface OutputBufferOptions {
  outputBuffer?: GPUBuffer | null;
}

export interface OutputOffsetOptions {
  outputOffset?: number;
}

export interface OutputDtypeOptions {
  outputDtype?: 'f16' | 'f32';
}

export interface Vec4Options {
  useVec4?: boolean;
}
