import {
  DEFAULT_EXECUTION_V1_SESSION,
  DEFAULT_KVCACHE_CONFIG,
} from '../../src/config/schema/index.js';

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

export function createExecutionV1Session(overrides = {}) {
  const base = cloneJson(DEFAULT_EXECUTION_V1_SESSION);
  return {
    ...base,
    ...overrides,
    compute: {
      ...base.compute,
      ...(overrides.compute ?? {}),
      defaults: {
        ...base.compute.defaults,
        ...(overrides.compute?.defaults ?? {}),
      },
    },
    perLayerInputs: {
      ...base.perLayerInputs,
      ...(overrides.perLayerInputs ?? {}),
      rowCache: {
        ...base.perLayerInputs.rowCache,
        ...(overrides.perLayerInputs?.rowCache ?? {}),
      },
      prefetch: {
        ...base.perLayerInputs.prefetch,
        ...(overrides.perLayerInputs?.prefetch ?? {}),
      },
      gpuUpload: {
        ...base.perLayerInputs.gpuUpload,
        ...(overrides.perLayerInputs?.gpuUpload ?? {}),
      },
      hotCache: {
        ...base.perLayerInputs.hotCache,
        ...(overrides.perLayerInputs?.hotCache ?? {}),
      },
    },
  };
}

export function createExecutionContractSession(overrides = {}) {
  const baseKVCache = cloneJson(DEFAULT_KVCACHE_CONFIG);
  const overrideKVCache = overrides.kvcache ?? {};
  return createExecutionV1Session({
    ...overrides,
    kvcache: {
      ...baseKVCache,
      layout: 'contiguous',
      kvDtype: 'f16',
      ...overrideKVCache,
      tiering: {
        ...baseKVCache.tiering,
        mode: 'off',
        ...(overrideKVCache.tiering ?? {}),
        compression: {
          ...baseKVCache.tiering.compression,
          ...(overrideKVCache.tiering?.compression ?? {}),
        },
        gating: {
          ...baseKVCache.tiering.gating,
          ...(overrideKVCache.tiering?.gating ?? {}),
        },
      },
      quantization: {
        ...baseKVCache.quantization,
      mode: 'none',
        ...(overrideKVCache.quantization ?? {}),
      },
    },
    decodeLoop: {
      batchSize: 1,
      stopCheckMode: 'batch',
      readbackInterval: 1,
      readbackMode: 'sequential',
      ringTokens: 1,
      ringStop: 1,
      ringStaging: 1,
      disableCommandBatching: false,
      ...(overrides.decodeLoop ?? {}),
    },
  });
}
