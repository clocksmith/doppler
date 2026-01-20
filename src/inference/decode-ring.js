import { getDevice, getDeviceLimits } from '../gpu/device.js';

const TOKEN_BYTES = 4;

function clampRingSize(size) {
  if (size == null) return 0;
  if (!Number.isFinite(size) || size <= 0) {
    throw new Error('DecodeRing requires positive ring sizes or null.');
  }
  return Math.floor(size);
}

function assertBufferFits(label, size, isStorage, limits) {
  if (!limits) return;
  const maxBufferSize = limits.maxBufferSize ?? Infinity;
  if (size > maxBufferSize) {
    throw new Error(`DecodeRing ${label} size ${size} exceeds maxBufferSize (${maxBufferSize}).`);
  }
  if (isStorage) {
    const maxStorageSize = limits.maxStorageBufferBindingSize ?? Infinity;
    if (size > maxStorageSize) {
      throw new Error(
        `DecodeRing ${label} size ${size} exceeds maxStorageBufferBindingSize (${maxStorageSize}).`
      );
    }
  }
}

function sameConfig(a, b) {
  if (!a || !b) return false;
  return a.batchSize === b.batchSize
    && a.tokensPerInterval === b.tokensPerInterval
    && a.stopCheckMode === b.stopCheckMode
    && a.ringTokens === b.ringTokens
    && a.ringStop === b.ringStop
    && a.ringStaging === b.ringStaging;
}

export class DecodeRing {
  buffers = null;
  config = null;
  index = 0;
  ringSize = 0;
  zeroStopData = null;

  ensure(config) {
    if (!config) {
      throw new Error('DecodeRing requires config.');
    }
    if (!Number.isFinite(config.batchSize) || config.batchSize <= 0) {
      throw new Error('DecodeRing requires positive batchSize.');
    }
    if (!Number.isFinite(config.tokensPerInterval) || config.tokensPerInterval <= 0) {
      throw new Error('DecodeRing requires positive tokensPerInterval.');
    }
    if (!config.stopCheckMode) {
      throw new Error('DecodeRing requires stopCheckMode.');
    }

    const normalized = {
      batchSize: Math.floor(config.batchSize),
      tokensPerInterval: Math.floor(config.tokensPerInterval),
      stopCheckMode: config.stopCheckMode,
      ringTokens: clampRingSize(config.ringTokens),
      ringStop: clampRingSize(config.ringStop),
      ringStaging: clampRingSize(config.ringStaging),
    };

    if (this.buffers && sameConfig(this.config, normalized)) {
      return;
    }

    this.release();

    const device = getDevice();
    if (!device) {
      throw new Error('GPU device not initialized');
    }
    const limits = getDeviceLimits();

    const tokensBytes = (normalized.tokensPerInterval + 1) * TOKEN_BYTES;
    const stopBytes = normalized.tokensPerInterval * TOKEN_BYTES;
    const stagingBytes = normalized.tokensPerInterval * TOKEN_BYTES;

    assertBufferFits('tokens', tokensBytes, true, limits);
    assertBufferFits('stagingTokens', stagingBytes, false, limits);
    if (normalized.stopCheckMode === 'per-token') {
      assertBufferFits('stop', stopBytes, true, limits);
      assertBufferFits('stagingStop', stagingBytes, false, limits);
    }

    const buffers = {
      tokens: null,
      stop: null,
      stagingTokens: null,
      stagingStop: null,
    };

    if (normalized.ringTokens > 0) {
      buffers.tokens = Array.from({ length: normalized.ringTokens }, (_, i) => (
        device.createBuffer({
          label: `decode_ring_tokens_${i}`,
          size: tokensBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
      ));
    }

    if (normalized.ringStaging > 0) {
      buffers.stagingTokens = Array.from({ length: normalized.ringStaging }, (_, i) => (
        device.createBuffer({
          label: `decode_ring_staging_tokens_${i}`,
          size: stagingBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })
      ));
    }

    if (normalized.stopCheckMode === 'per-token' && normalized.ringStop > 0) {
      buffers.stop = Array.from({ length: normalized.ringStop }, (_, i) => (
        device.createBuffer({
          label: `decode_ring_stop_${i}`,
          size: stopBytes,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        })
      ));
    }

    if (normalized.stopCheckMode === 'per-token' && normalized.ringStaging > 0) {
      buffers.stagingStop = Array.from({ length: normalized.ringStaging }, (_, i) => (
        device.createBuffer({
          label: `decode_ring_staging_stop_${i}`,
          size: stagingBytes,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })
      ));
    }

    this.buffers = buffers;
    this.config = normalized;
    this.index = 0;
    this.ringSize = Math.max(
      1,
      normalized.ringTokens,
      normalized.ringStop,
      normalized.ringStaging
    );
    this.zeroStopData = normalized.stopCheckMode === 'per-token'
      ? new Uint32Array(normalized.tokensPerInterval)
      : null;
  }

  acquire() {
    if (!this.buffers || !this.config) return null;
    const idx = this.index;
    const tokens = this.buffers.tokens
      ? this.buffers.tokens[idx % this.buffers.tokens.length]
      : null;
    const stop = this.buffers.stop
      ? this.buffers.stop[idx % this.buffers.stop.length]
      : null;
    const stagingTokens = this.buffers.stagingTokens
      ? this.buffers.stagingTokens[idx % this.buffers.stagingTokens.length]
      : null;
    const stagingStop = this.buffers.stagingStop
      ? this.buffers.stagingStop[idx % this.buffers.stagingStop.length]
      : null;

    return {
      index: idx,
      tokens,
      stop,
      stagingTokens,
      stagingStop,
      tokensPerInterval: this.config.tokensPerInterval,
      zeroStopData: this.zeroStopData,
    };
  }

  advance() {
    if (!this.buffers) return;
    this.index = (this.index + 1) % this.ringSize;
  }

  reset() {
    this.index = 0;
  }

  release() {
    if (this.buffers) {
      this.buffers.tokens?.forEach((buffer) => buffer.destroy());
      this.buffers.stop?.forEach((buffer) => buffer.destroy());
      this.buffers.stagingTokens?.forEach((buffer) => buffer.destroy());
      this.buffers.stagingStop?.forEach((buffer) => buffer.destroy());
    }
    this.buffers = null;
    this.config = null;
    this.index = 0;
    this.ringSize = 0;
    this.zeroStopData = null;
  }
}
