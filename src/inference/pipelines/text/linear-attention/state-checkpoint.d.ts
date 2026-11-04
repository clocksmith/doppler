import type { LinearAttentionRuntime, LinearLayerRuntimeState } from '../linear-attention.js';
export interface RecordedLinearAttentionCheckpoint {
  /** Capture in the owner's existing command buffer, before tentative updates. No submission or host wait. */
  recordCapture(encoder: GPUCommandEncoder): void;
  /** Restore only after the capture is ordered before this command buffer; also restores saved host cursors. */
  recordRestore(encoder: GPUCommandEncoder): void;
  /** Caller retains buffers until all submitted capture/restore commands settle. */
  close(lost?: boolean): void;
}
export function createLinearAttentionCheckpoint(runtime: LinearAttentionRuntime): RecordedLinearAttentionCheckpoint;
export function checkpointLinearAttentionRuntime(runtime: LinearAttentionRuntime): Promise<{ restore(): Promise<void>; close(lost?: boolean): void }>;
/** Enqueue capture without a host fence. The caller must use the same GPU queue and settle later work before closing. */
export function queueLinearAttentionCheckpoint(runtime: LinearAttentionRuntime): Promise<{ restore(): Promise<void>; close(lost?: boolean): void }>;

export interface SavedStateBuffer { source: GPUBuffer; saved: GPUBuffer; }
export interface SavedStateCursor { layer: LinearLayerRuntimeState; seqLen: number; warnedSeqMismatch: boolean; }
