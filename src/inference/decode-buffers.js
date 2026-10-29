

import { PreparedDecodeWorkspace } from '../gpu/prepared-decode-workspace.js';
import { getDevice } from '../gpu/device.js';
import { tensorElementBytes } from '../gpu/tensor.js';
import { padToQ4KBlock } from '../config/schema/index.js';

export class DecodeBufferManager {
  
  buffers = null;

  config = null;

  pingPongIndex = 0;

  workspace = null;

  workspaceDevice = null;

  ensureBuffers(config) {
    if (config.enablePingPong == null) {
      throw new Error('DecodeBufferManager requires enablePingPong in config.');
    }
    if (config.activationDtype !== 'f16' && config.activationDtype !== 'f32') {
      throw new Error('DecodeBufferManager requires activationDtype f16 or f32 in config.');
    }
    const normalizedConfig = { ...config };
    const bytesPerElement = tensorElementBytes(normalizedConfig.activationDtype, {
      precision: normalizedConfig.activationPrecision,
      components: normalizedConfig.activationComponents,
      elementStrideBytes: normalizedConfig.activationElementStrideBytes,
    });

    // Check if we already have matching buffers
    if (this.buffers && this.config &&
        this.config.hiddenSize === normalizedConfig.hiddenSize &&
        this.config.intermediateSize === normalizedConfig.intermediateSize &&
        this.config.activationDtype === normalizedConfig.activationDtype &&
        this.config.activationPrecision === normalizedConfig.activationPrecision &&
        this.config.activationComponents === normalizedConfig.activationComponents &&
        this.config.activationElementStrideBytes === normalizedConfig.activationElementStrideBytes &&
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
    // Storage size comes from the declared local activation representation.

    // Pad dimensions to Q4K super-block alignment for fused kernels.
    // Q4K kernels process 256 elements per block and read out of bounds
    // without padding. Extra padding elements remain zero (from WebGPU
    // buffer init) since all kernels write exactly hiddenSize/intermediateSize.
    const paddedHiddenSize = padToQ4KBlock(normalizedConfig.hiddenSize);
    const paddedIntermediateSize = padToQ4KBlock(normalizedConfig.intermediateSize);

    const hiddenBytes = paddedHiddenSize * bytesPerElement;
    const intermediateBytes = paddedIntermediateSize * bytesPerElement;

    const allocated = [];
    const create = descriptor => {
      const buffer = device.createBuffer(descriptor);
      allocated.push(buffer);
      return buffer;
    };
    try {
      const hidden = create({
        label: 'decode_hidden',
        size: hiddenBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      const attnOutput = create({
        label: 'decode_attn_output',
        size: hiddenBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      const ffnIntermediate = create({
        label: 'decode_ffn_intermediate',
        size: intermediateBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      const buffers = { hidden, attnOutput, ffnIntermediate };

      // Allocate alternate hidden buffer for ping-pong if enabled
      if (normalizedConfig.enablePingPong) {
        buffers.hiddenAlt = create({
          label: 'decode_hidden_alt',
          size: hiddenBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }

      this.buffers = buffers;
      this.config = normalizedConfig;
      this.pingPongIndex = 0;

      return this.buffers;
    } catch (error) {
      for (const buffer of allocated) buffer.destroy();
      throw error;
    }
  }

  getWorkspace(device) {
    if (this.workspace && this.workspaceDevice !== device) {
      throw new Error('Prepared decode workspace belongs to another GPUDevice; release the session first.');
    }
    this.workspace ??= new PreparedDecodeWorkspace(device);
    this.workspaceDevice = device;
    return this.workspace;
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
    const hiddenBytes = this.buffers.hidden.size;
    const intermediateBytes = this.buffers.ffnIntermediate.size;
    const bufferCount = this.buffers?.hiddenAlt ? 4 : 3;
    const workspace = this.workspace?.getStats() ?? null;
    const totalBytes = hiddenBytes * (bufferCount - 1) + intermediateBytes + (workspace?.bytes ?? 0);
    return { hiddenBytes, intermediateBytes, totalBytes, activationDtype: this.config.activationDtype, workspace };
  }

  ownsBuffer(buffer) {
    if (!this.buffers) return false;
    return buffer === this.buffers.hidden
      || buffer === this.buffers.hiddenAlt
      || buffer === this.buffers.attnOutput
      || buffer === this.buffers.ffnIntermediate
      || this.workspace?.ownsBuffer(buffer) === true;
  }

  release() {
    this.workspace?.release();
    this.workspace = null;
    this.workspaceDevice = null;
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
