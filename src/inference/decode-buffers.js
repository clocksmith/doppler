

import { getDevice } from '../gpu/device.js';




export class DecodeBufferManager {
  
  buffers = null;

  
  config = null;

  
  pingPongIndex = 0;

  
  ensureBuffers(config) {
    
    const normalizedConfig = {
      ...config,
      enablePingPong: config.enablePingPong ?? false,
      activationDtype: config.activationDtype ?? 'f32',
    };

    // Check if we already have matching buffers
    if (this.buffers && this.config &&
        this.config.hiddenSize === normalizedConfig.hiddenSize &&
        this.config.intermediateSize === normalizedConfig.intermediateSize &&
        this.config.activationDtype === normalizedConfig.activationDtype &&
        this.config.enablePingPong === normalizedConfig.enablePingPong) {
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
    const bytesPerElement = normalizedConfig.activationDtype === 'f16' ? 2 : 4;
    const hiddenBytes = normalizedConfig.hiddenSize * bytesPerElement;
    const intermediateBytes = normalizedConfig.intermediateSize * bytesPerElement;

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
    this.config = normalizedConfig;

    // Allocate alternate hidden buffer for ping-pong if enabled
    if (normalizedConfig.enablePingPong) {
      this.buffers.hiddenAlt = device.createBuffer({
        label: 'decode_hidden_alt',
        size: hiddenBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
    }

    this.pingPongIndex = 0;

    return this.buffers;
  }

  
  getHiddenBuffer() {
    if (!this.buffers) return null;
    if (this.buffers.hiddenAlt && this.pingPongIndex === 1) {
      return this.buffers.hiddenAlt;
    }
    return this.buffers.hidden;
  }

  
  getOutputHiddenBuffer() {
    if (!this.buffers) return null;
    if (this.buffers.hiddenAlt) {
      // Return the other buffer
      return this.pingPongIndex === 0 ? this.buffers.hiddenAlt : this.buffers.hidden;
    }
    return this.buffers.hidden;
  }

  
  swapPingPong() {
    if (this.buffers?.hiddenAlt) {
      this.pingPongIndex = 1 - this.pingPongIndex;
    }
  }

  
  resetPingPong() {
    this.pingPongIndex = 0;
  }

  
  getAttnOutputBuffer() {
    return this.buffers?.attnOutput ?? null;
  }

  
  getFFNIntermediateBuffer() {
    return this.buffers?.ffnIntermediate ?? null;
  }

  
  hasBuffers() {
    return this.buffers !== null;
  }

  
  getStats() {
    if (!this.config) return null;
    const bytesPerElement = this.config.activationDtype === 'f16' ? 2 : 4;
    const hiddenBytes = this.config.hiddenSize * bytesPerElement;
    const intermediateBytes = this.config.intermediateSize * bytesPerElement;
    const bufferCount = this.buffers?.hiddenAlt ? 4 : 3;
    const totalBytes = hiddenBytes * (bufferCount - 1) + intermediateBytes;
    return { hiddenBytes, intermediateBytes, totalBytes, activationDtype: this.config.activationDtype ?? 'f32' };
  }

  
  ownsBuffer(buffer) {
    if (!this.buffers) return false;
    return buffer === this.buffers.hidden
      || buffer === this.buffers.hiddenAlt
      || buffer === this.buffers.attnOutput
      || buffer === this.buffers.ffnIntermediate;
  }

  
  release() {
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
