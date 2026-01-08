import { getDevice, getDeviceLimits } from '../gpu/device.js';

const TOKEN_BYTES = 4;

function clampRingSize(size) {
  if (size == null) return 0;
  if (!Number.isFinite(size) || size <= 0) {
    throw new Error('DecodeRing requires positive ring sizes or null.');
  }
  return Math.floor(size);
}

function createSlotStats(allocated) {
  return { allocated, uses: 0, reuses: 0 };
}

function resetSlotStats(stats) {
  stats.uses = 0;
  stats.reuses = 0;
}

function trackSlotUse(stats) {
  stats.uses += 1;
  if (stats.uses > stats.allocated) {
    stats.reuses += 1;
  }
}

function createRingStats(buffers, config, ringSize) {
  return {
    tokens: createSlotStats(buffers.tokens?.length ?? 0),
    stop: createSlotStats(buffers.stop?.length ?? 0),
    stagingTokens: createSlotStats(buffers.stagingTokens?.length ?? 0),
    stagingStop: createSlotStats(buffers.stagingStop?.length ?? 0),
    acquires: 0,
    advances: 0,
    resets: 0,
    ringSize,
    tokensPerInterval: config.tokensPerInterval,
  };
}

function resetRingUsage(stats) {
  if (!stats) return;
  stats.acquires = 0;
  stats.advances = 0;
  stats.resets += 1;
  resetSlotStats(stats.tokens);
  resetSlotStats(stats.stop);
  resetSlotStats(stats.stagingTokens);
  resetSlotStats(stats.stagingStop);
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
  stats = null;

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
    const stopBytes = (normalized.tokensPerInterval + 1) * TOKEN_BYTES;
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
      ? new Uint32Array(normalized.tokensPerInterval + 1)
      : null;
    this.stats = createRingStats(buffers, normalized, this.ringSize);
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

    if (this.stats) {
      this.stats.acquires += 1;
      if (tokens) trackSlotUse(this.stats.tokens);
      if (stop) trackSlotUse(this.stats.stop);
      if (stagingTokens) trackSlotUse(this.stats.stagingTokens);
      if (stagingStop) trackSlotUse(this.stats.stagingStop);
    }

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
    if (this.stats) {
      this.stats.advances += 1;
    }
  }

  reset() {
    this.index = 0;
    resetRingUsage(this.stats);
  }

  getStats() {
    if (!this.stats) return null;
    return {
      tokens: { ...this.stats.tokens },
      stop: { ...this.stats.stop },
      stagingTokens: { ...this.stats.stagingTokens },
      stagingStop: { ...this.stats.stagingStop },
      acquires: this.stats.acquires,
      advances: this.stats.advances,
      resets: this.stats.resets,
      ringSize: this.stats.ringSize,
      tokensPerInterval: this.stats.tokensPerInterval,
    };
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
    this.stats = null;
  }
}
