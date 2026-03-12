import assert from 'node:assert/strict';

const { createKVCache } = await import('../../src/inference/pipelines/text/init.js');
const { setDevice } = await import('../../src/gpu/device.js');
const {
  DEFAULT_KVCACHE_CONFIG,
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} = await import('../../src/config/schema/index.js');
const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { SlidingWindowKVCache } = await import('../../src/inference/kv-cache/sliding-window.js');

function createMinimalDevice() {
  return {
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() {
        return Promise.resolve();
      },
    },
    features: new Set(),
    limits: {
      maxStorageBufferBindingSize: 1 << 20,
      maxBufferSize: 1 << 20,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 8,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() {
      return {};
    },
    createBuffer() {
      throw new Error('createBuffer should not be called in kvcache-layout-policy tests.');
    },
    createCommandEncoder() {
      throw new Error('createCommandEncoder should not be called in kvcache-layout-policy tests.');
    },
  };
}

function baseModelConfig(overrides = {}) {
  return {
    numLayers: 1,
    numKVHeads: 1,
    headDim: 8,
    maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
    slidingWindow: null,
    attnLogitSoftcapping: null,
    layerTypes: [],
    ...overrides,
  };
}

function baseRuntimeKV(overrides = {}) {
  return {
    ...DEFAULT_KVCACHE_CONFIG,
    maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
    ...overrides,
  };
}

setDevice(createMinimalDevice(), { platformConfig: null });

try {
  // === Regression: paged threshold must be guarded by !forceContiguousKVCache ===
  // The original bug: when maxSeqLen >= PAGED_LAYOUT_SEQ_LEN_THRESHOLD (8192),
  // contiguous layout was auto-upgraded to paged even for models with full-attention
  // layers (e.g. Gemma 3). This produced 64 <pad> tokens in browser because WebGPU
  // paged attention is broken for full-attention models.
  // Fix: the threshold upgrade is gated by !forceContiguousKVCache.

  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention', 'full_attention'],
        slidingWindow: 1024,
      }),
      false,
      false,
      baseRuntimeKV()
    );
    assert.equal(
      cache.layout,
      'contiguous',
      'Full-attention model must stay contiguous even when maxSeqLen >= threshold'
    );
  }

  // Sliding-window-only models should still auto-upgrade to paged
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention'],
        slidingWindow: 1024,
      }),
      false,
      false,
      baseRuntimeKV()
    );
    assert.equal(
      cache.layout,
      'paged',
      'Sliding-window-only model should auto-upgrade to paged at threshold'
    );
  }

  // Empty layerTypes should not force contiguous (no full-attention layers detected)
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: [],
      }),
      false,
      false,
      baseRuntimeKV()
    );
    assert.equal(
      cache.layout,
      'paged',
      'Empty layerTypes should allow paged upgrade'
    );
  }

  // === Explicit paged + forceContiguousKVCache must fail fast ===
  // When runtime config requests layout: 'paged' explicitly,
  // but model has full-attention layers, the runtime must reject the request
  // with an actionable error instead of silently proceeding.

  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['full_attention'],
        }),
        false,
        false,
        baseRuntimeKV({ layout: 'paged' })
      ),
      /Paged KV cache layout is not supported for models with full-attention layers/,
      'Explicit paged layout with full-attention layers must throw'
    );
  }

  // Mixed attention (full + sliding) must also reject explicit paged
  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['sliding_attention', 'full_attention'],
          slidingWindow: 1024,
        }),
        false,
        false,
        baseRuntimeKV({ layout: 'paged' })
      ),
      /Paged KV cache layout is not supported for models with full-attention layers/,
      'Explicit paged layout with mixed attention must throw'
    );
  }

  // === Threshold boundary: maxSeqLen just below threshold stays contiguous ===
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention'],
        maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD - 1,
      }),
      false,
      false,
      baseRuntimeKV({ maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD - 1 })
    );
    assert.equal(
      cache.layout,
      'contiguous',
      'maxSeqLen below threshold should stay contiguous'
    );
  }

  // maxSeqLen exactly at threshold triggers upgrade (sliding-window-only)
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention'],
        maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
      }),
      false,
      false,
      baseRuntimeKV({ maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD })
    );
    assert.equal(
      cache.layout,
      'paged',
      'maxSeqLen at threshold should upgrade to paged for sliding-window-only'
    );
  }

  // === Tiered layout gating ===
  // When tiering mode is not 'off' and layout is 'contiguous', the layout must
  // be upgraded to 'tiered'. The upgrade is verified indirectly: with useGPU=false,
  // kvDtype resolves to f32 (no shader-f16 capability), and TieredKVCache requires
  // f16. If the layout upgrade occurs (contiguous -> tiered), the f16 gate throws.
  // This proves the layout was upgraded before the cache was constructed.

  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['full_attention'],
        }),
        false,
        false,
        baseRuntimeKV({
          layout: 'contiguous',
          tiering: {
            ...DEFAULT_KVCACHE_CONFIG.tiering,
            mode: 'int8',
            gating: { mode: 'off', minAluBwRatio: 0 },
            compression: { mode: 'int8', blockSize: 1 },
          },
        })
      ),
      /Tiered KV cache requires kvDtype="f16"/,
      'Contiguous layout with active tiering must upgrade to tiered (then fail on f32 kvDtype)'
    );
  }

  // Paged layout with active tiering must fail fast (not contiguous, not tiered)
  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['full_attention'],
        }),
        false,
        false,
        baseRuntimeKV({
          layout: 'paged',
          tiering: {
            ...DEFAULT_KVCACHE_CONFIG.tiering,
            mode: 'int8',
            gating: { mode: 'off', minAluBwRatio: 0 },
            compression: { mode: 'int8', blockSize: 1 },
          },
        })
      ),
      /must be "tiered" when tiering\.mode is enabled/,
      'Non-contiguous, non-tiered layout with active tiering should throw'
    );
  }

  // tiering mode 'off' with contiguous layout should not upgrade
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['full_attention'],
      }),
      false,
      false,
      baseRuntimeKV({
        layout: 'contiguous',
        tiering: {
          ...DEFAULT_KVCACHE_CONFIG.tiering,
          mode: 'off',
        },
        maxSeqLen: 1024,
      })
    );
    assert.equal(
      cache.layout,
      'contiguous',
      'Tiering mode off should not upgrade contiguous layout'
    );
  }

  // === Sliding-window interaction ===
  // When model has sliding window and no full-attention layers,
  // cacheMaxSeqLen should be bounded by slidingWindow on contiguous layout.

  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention'],
        slidingWindow: 512,
        maxSeqLen: 2048,
      }),
      false,
      false,
      baseRuntimeKV({ maxSeqLen: 2048 })
    );
    assert.ok(
      cache instanceof SlidingWindowKVCache,
      'Sliding-window model below threshold should get SlidingWindowKVCache'
    );
    assert.equal(cache.maxSeqLen, 512, 'maxSeqLen should be bounded by slidingWindow');
  }

  // Mixed model (full+sliding) should NOT get SlidingWindowKVCache
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention', 'full_attention'],
        slidingWindow: 512,
        maxSeqLen: 2048,
      }),
      false,
      false,
      baseRuntimeKV({ maxSeqLen: 2048 })
    );
    assert.ok(
      cache instanceof KVCache,
      'Mixed model should get standard KVCache, not SlidingWindowKVCache'
    );
    assert.ok(
      !(cache instanceof SlidingWindowKVCache),
      'Mixed model must not be SlidingWindowKVCache'
    );
  }

  // === Required fields fail fast ===

  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({ maxSeqLen: NaN }),
        false,
        false,
        baseRuntimeKV()
      ),
      /maxSeqLen/,
      'Invalid maxSeqLen should throw'
    );
  }

  {
    assert.throws(
      () => createKVCache(
        baseModelConfig(),
        false,
        false,
        baseRuntimeKV({ layout: null })
      ),
      /layout is required/,
      'Missing layout should throw'
    );
  }

  // === Mixed layer type normalization ===
  // Various string aliases for sliding-window layers should all be recognized.

  {
    for (const slidingAlias of ['sliding_attention', 'local_attention', 'local', 'sliding']) {
      const cache = createKVCache(
        baseModelConfig({
          layerTypes: [slidingAlias],
          maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
        }),
        false,
        false,
        baseRuntimeKV()
      );
      assert.equal(
        cache.layout,
        'paged',
        `Sliding alias "${slidingAlias}" should allow paged upgrade`
      );
    }
  }

  // Full-attention layer types block paged threshold upgrade
  {
    for (const fullType of ['full_attention', 'global', 'standard', '']) {
      const cache = createKVCache(
        baseModelConfig({
          layerTypes: [fullType],
          maxSeqLen: PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
        }),
        false,
        false,
        baseRuntimeKV()
      );
      // hasFullAttentionLayers returns true for non-sliding types
      // Empty string or unknown types count as full-attention
      assert.equal(
        cache.layout,
        'contiguous',
        `Layer type "${fullType}" should be treated as full-attention, blocking paged upgrade`
      );
    }
  }
} finally {
  setDevice(null);
}

console.log('kvcache-layout-policy.test: ok');
