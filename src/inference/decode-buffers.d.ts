/**
 * Decode Buffer Manager
 *
 * Pre-allocates and reuses GPU buffers for the decode phase.
 * During decode (M=1), the same buffer sizes are needed every step.
 * Instead of acquiring from pool each time, we keep dedicated buffers.
 *
 * WebLLM-inspired optimization: decode uses fixed-size buffers that
 * can be reused across tokens without pool overhead.
 */

/**
 * Pre-allocated buffers for decode operations
 */
export interface DecodeBuffers {
  /** Hidden state buffer (1 x hiddenSize) */
  hidden: GPUBuffer;
  /** Attention output buffer (1 x hiddenSize) */
  attnOutput: GPUBuffer;
  /** FFN intermediate buffer (1 x intermediateSize) */
  ffnIntermediate: GPUBuffer;
  /** Alternate hidden buffer for ping-pong (optional, for 2C) */
  hiddenAlt?: GPUBuffer;
}

/**
 * Configuration for decode buffer sizes
 */
export interface DecodeBufferConfig {
  hiddenSize: number;
  intermediateSize: number;
  /** Enable ping-pong buffers (alternating between two hidden buffers) */
  enablePingPong?: boolean;
  /** Activation dtype for hidden buffers - 'f16' uses 2 bytes, 'f32' uses 4 bytes (default) */
  activationDtype?: 'f16' | 'f32';
}

/**
 * Manages pre-allocated buffers for efficient decode operations.
 *
 * Usage:
 * 1. Call ensureBuffers() after model config is known
 * 2. Use getHiddenBuffer() to get decode hidden state buffer
 * 3. Call release() when done with generation
 */
export declare class DecodeBufferManager {
  private buffers;
  private config;
  private pingPongIndex;

  /**
   * Ensure buffers are allocated for the given config.
   * No-op if already allocated with matching config.
   */
  ensureBuffers(config: DecodeBufferConfig): DecodeBuffers;

  /**
   * Get the current hidden state buffer.
   * If ping-pong is enabled, returns the current input buffer.
   */
  getHiddenBuffer(): GPUBuffer | null;

  /**
   * Get the output hidden state buffer for next layer.
   * If ping-pong is enabled, returns the alternate buffer.
   */
  getOutputHiddenBuffer(): GPUBuffer | null;

  /**
   * Swap ping-pong buffers (call after each layer).
   */
  swapPingPong(): void;

  /**
   * Reset ping-pong state (call at start of each decode step).
   */
  resetPingPong(): void;

  /**
   * Get attention output buffer.
   */
  getAttnOutputBuffer(): GPUBuffer | null;

  /**
   * Get FFN intermediate buffer.
   */
  getFFNIntermediateBuffer(): GPUBuffer | null;

  /**
   * Check if buffers are allocated.
   */
  hasBuffers(): boolean;

  /**
   * Get buffer sizes for debugging.
   */
  getStats(): { hiddenBytes: number; intermediateBytes: number; totalBytes: number; activationDtype: 'f16' | 'f32' } | null;

  /**
   * Check whether a buffer is managed by this decode buffer manager.
   */
  ownsBuffer(buffer: GPUBuffer): boolean;

  /**
   * Release all buffers.
   */
  release(): void;
}
