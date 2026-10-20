import type { LinearAttentionRuntime, LinearLayerRuntimeState } from '../linear-attention.js';
export function checkpointLinearAttentionRuntime(runtime: LinearAttentionRuntime): Promise<{ restore(): Promise<void>; close(lost?: boolean): void }>;

export interface SavedStateBuffer { source: GPUBuffer; saved: GPUBuffer; }
export interface SavedStateCursor { layer: LinearLayerRuntimeState; seqLen: number; warnedSeqMismatch: boolean; }
