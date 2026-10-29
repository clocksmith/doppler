import type { Tensor } from '../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../gpu/command-recorder.js';
import type { DecodeBufferManager } from '../../decode-buffers.js';
export interface PrecomputedInputNormContext {
  __precomputedInputNorm?: { layerIdx: number; tensor: Tensor } | null;
  decodeBuffers?: DecodeBufferManager | null;
}
/** Consumes and releases any unused precomputed norm through its existing owner. */
export function releasePrecomputedInputNorm(context: PrecomputedInputNormContext, recorder: CommandRecorder | null): void;
/** Transfers the matching norm; rejects and releases a stale layer identity. */
export function takePrecomputedInputNorm(context: PrecomputedInputNormContext, layerIdx: number, recorder: CommandRecorder | null): Tensor | null;
