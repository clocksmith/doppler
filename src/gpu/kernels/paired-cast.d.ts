import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
/** Preserve source F32 bits and append a zero low component on the GPU. */
export declare function expandF32ToPaired(input: Tensor, options?: OutputBufferOptions): Promise<Tensor>;
export declare function recordExpandF32ToPaired(recorder: CommandRecorder, input: Tensor, options?: OutputBufferOptions): Promise<Tensor>;
/** Direct nearest-even F16 store of normalized pairs, with no F32 collapse. */
export declare function castPairedToF16(input: Tensor, options?: OutputBufferOptions): Promise<Tensor>;
export declare function recordCastPairedToF16(recorder: CommandRecorder, input: Tensor, options?: OutputBufferOptions): Promise<Tensor>;
