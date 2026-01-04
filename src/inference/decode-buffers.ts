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

import { getDevice } from '../gpu/device.js';

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
export class DecodeBufferManager {
  private buffers: DecodeBuffers | null = null;
  private config: DecodeBufferConfig | null = null;
  private pingPongIndex = 0;

  /**
   * Ensure buffers are allocated for the given config.
   * No-op if already allocated with matching config.
   */
  ensureBuffers(config: DecodeBufferConfig): DecodeBuffers {
    // Check if we already have matching buffers
    if (this.buffers && this.config &&
        this.config.hiddenSize === config.hiddenSize &&
        this.config.intermediateSize === config.intermediateSize &&
        this.config.activationDtype === config.activationDtype) {
      return this.buffers;
    }

    // Release old buffers if config changed
    if (this.buffers) {
      this.release();
    }

    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not initialized');
    }

    // Allocate buffers
    // For decode, we process 1 token at a time (M=1)
    // F16 activations use 2 bytes per element, F32 uses 4 bytes
    const bytesPerElement = config.activationDtype === 'f16' ? 2 : 4;
    const hiddenBytes = config.hiddenSize * bytesPerElement;
    const intermediateBytes = config.intermediateSize * bytesPerElement;

    const hidden = device.createBuffer({
      label: 'decode_hidden',
      size: hiddenBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const attnOutput = device.createBuffer({
      label: 'decode_attn_output',
      size: hiddenBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const ffnIntermediate = device.createBuffer({
      label: 'decode_ffn_intermediate',
      size: intermediateBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.buffers = { hidden, attnOutput, ffnIntermediate };
    this.config = config;

    // Allocate alternate hidden buffer for ping-pong if enabled
    if (config.enablePingPong) {
      this.buffers.hiddenAlt = device.createBuffer({
        label: 'decode_hidden_alt',
        size: hiddenBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    this.pingPongIndex = 0;

    return this.buffers;
  }

  /**
   * Get the current hidden state buffer.
   * If ping-pong is enabled, returns the current input buffer.
   */
  getHiddenBuffer(): GPUBuffer | null {
    if (!this.buffers) return null;
    if (this.buffers.hiddenAlt && this.pingPongIndex === 1) {
      return this.buffers.hiddenAlt;
    }
    return this.buffers.hidden;
  }

  /**
   * Get the output hidden state buffer for next layer.
   * If ping-pong is enabled, returns the alternate buffer.
   */
  getOutputHiddenBuffer(): GPUBuffer | null {
    if (!this.buffers) return null;
    if (this.buffers.hiddenAlt) {
      // Return the other buffer
      return this.pingPongIndex === 0 ? this.buffers.hiddenAlt : this.buffers.hidden;
    }
    return this.buffers.hidden;
  }

  /**
   * Swap ping-pong buffers (call after each layer).
   */
  swapPingPong(): void {
    if (this.buffers?.hiddenAlt) {
      this.pingPongIndex = 1 - this.pingPongIndex;
    }
  }

  /**
   * Reset ping-pong state (call at start of each decode step).
   */
  resetPingPong(): void {
    this.pingPongIndex = 0;
  }

  /**
   * Get attention output buffer.
   */
  getAttnOutputBuffer(): GPUBuffer | null {
    return this.buffers?.attnOutput ?? null;
  }

  /**
   * Get FFN intermediate buffer.
   */
  getFFNIntermediateBuffer(): GPUBuffer | null {
    return this.buffers?.ffnIntermediate ?? null;
  }

  /**
   * Check if buffers are allocated.
   */
  hasBuffers(): boolean {
    return this.buffers !== null;
  }

  /**
   * Get buffer sizes for debugging.
   */
  getStats(): { hiddenBytes: number; intermediateBytes: number; totalBytes: number; activationDtype: 'f16' | 'f32' } | null {
    if (!this.config) return null;
    const bytesPerElement = this.config.activationDtype === 'f16' ? 2 : 4;
    const hiddenBytes = this.config.hiddenSize * bytesPerElement;
    const intermediateBytes = this.config.intermediateSize * bytesPerElement;
    const bufferCount = this.buffers?.hiddenAlt ? 4 : 3;
    const totalBytes = hiddenBytes * (bufferCount - 1) + intermediateBytes;
    return { hiddenBytes, intermediateBytes, totalBytes, activationDtype: this.config.activationDtype ?? 'f32' };
  }

  /**
   * Release all buffers.
   */
  release(): void {
    if (this.buffers) {
      this.buffers.hidden.destroy();
      this.buffers.attnOutput.destroy();
      this.buffers.ffnIntermediate.destroy();
      this.buffers.hiddenAlt?.destroy();
      this.buffers = null;
    }
    this.config = null;
    this.pingPongIndex = 0;
  }
}
