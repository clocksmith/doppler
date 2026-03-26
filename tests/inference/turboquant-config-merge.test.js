import assert from 'node:assert/strict';

globalThis.GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

const { mergeRuntimeValues } = await import('../../src/config/runtime-merge.js');
const { createKVCache } = await import('../../src/inference/pipelines/text/init.js');
const { setDevice } = await import('../../src/gpu/device.js');
const {
  DEFAULT_KVCACHE_CONFIG,
  PAGED_LAYOUT_SEQ_LEN_THRESHOLD,
} = await import('../../src/config/schema/index.js');
const { KVCache } = await import('../../src/inference/kv-cache/base.js');
const { QuantizedKVCache } = await import('../../src/inference/kv-cache/quantized.js');

// ---------------------------------------------------------------------------
// Mock device — extends the kvcache-layout-policy pattern with buffer creation
// so QuantizedKVCache can allocate its per-layer storage.
// ---------------------------------------------------------------------------
let nextBufferId = 0;

function createMockGPUBuffer(descriptor) {
  const size = descriptor?.size ?? 0;
  const ab = new ArrayBuffer(size);
  return {
    label: descriptor?.label ?? '',
    size,
    usage: descriptor?.usage ?? 0,
    mapState: 'unmapped',
    mappedAtCreation: descriptor?.mappedAtCreation ?? false,
    _arrayBuffer: ab,
    _id: nextBufferId++,
    getMappedRange() { return ab; },
    unmap() { this.mapState = 'unmapped'; },
    destroy() {},
    mapAsync() { return Promise.resolve(); },
  };
}

function createGPUDevice() {
  return {
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() { return Promise.resolve(); },
    },
    features: new Set(['shader-f16']),
    limits: {
      maxStorageBufferBindingSize: 1 << 28,
      maxBufferSize: 1 << 28,
      maxComputeWorkgroupSizeX: 256,
      maxComputeWorkgroupSizeY: 1,
      maxComputeWorkgroupSizeZ: 1,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16384,
      maxStorageBuffersPerShaderStage: 16,
      maxUniformBufferBindingSize: 65536,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    createBindGroup() { return {}; },
    createBuffer(desc) { return createMockGPUBuffer(desc); },
    createShaderModule() { return {}; },
    createCommandEncoder() {
      throw new Error('createCommandEncoder should not be called in turboquant-config-merge tests.');
    },
  };
}

function createMinimalDevice() {
  return {
    queue: {
      submit() {},
      writeBuffer() {},
      onSubmittedWorkDone() { return Promise.resolve(); },
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
    createBindGroup() { return {}; },
    createBuffer() {
      throw new Error('createBuffer should not be called with minimal device.');
    },
    createCommandEncoder() {
      throw new Error('createCommandEncoder should not be called in turboquant-config-merge tests.');
    },
  };
}

function baseModelConfig(overrides = {}) {
  return {
    numLayers: 1,
    numKVHeads: 1,
    headDim: 8,
    maxSeqLen: 1024,
    slidingWindow: null,
    attnLogitSoftcapping: null,
    layerTypes: [],
    ...overrides,
  };
}

function baseRuntimeKV(overrides = {}) {
  return {
    ...DEFAULT_KVCACHE_CONFIG,
    maxSeqLen: 1024,
    ...overrides,
  };
}

// ===========================================================================
// Part 1: mergeRuntimeValues deep merge of kvcache.quantization
// ===========================================================================

// Partial quantization override preserves unspecified fields from base
{
  const base = { ...DEFAULT_KVCACHE_CONFIG };
  const override = { quantization: { mode: 'turboquant' } };
  const merged = mergeRuntimeValues(base, override);
  assert.equal(merged.quantization.mode, 'turboquant', 'mode should be overridden');
  assert.equal(merged.quantization.bitWidth, 4, 'bitWidth should be preserved from base');
  assert.equal(merged.quantization.prodMode, false, 'prodMode should be preserved from base');
  assert.equal(merged.layout, 'contiguous', 'layout should be preserved from base');
}

// Full quantization override replaces all fields
{
  const base = { ...DEFAULT_KVCACHE_CONFIG };
  const override = { quantization: { mode: 'turboquant_prod', bitWidth: 3, prodMode: true } };
  const merged = mergeRuntimeValues(base, override);
  assert.equal(merged.quantization.mode, 'turboquant_prod');
  assert.equal(merged.quantization.bitWidth, 3);
  assert.equal(merged.quantization.prodMode, true);
}

// quantization.mode: 'none' override keeps defaults intact elsewhere
{
  const base = { ...DEFAULT_KVCACHE_CONFIG };
  const override = { quantization: { mode: 'none' } };
  const merged = mergeRuntimeValues(base, override);
  assert.equal(merged.quantization.mode, 'none');
  assert.equal(merged.quantization.bitWidth, 4);
  assert.equal(merged.layout, 'contiguous');
  assert.equal(merged.tiering.mode, 'off');
}

// tiering.compression override deep merges without clobbering quantization
{
  const base = { ...DEFAULT_KVCACHE_CONFIG };
  const override = {
    tiering: {
      compression: { mode: 'turboquant', blockSize: 1 },
    },
    quantization: { mode: 'turboquant', bitWidth: 4 },
  };
  const merged = mergeRuntimeValues(base, override);
  assert.equal(merged.tiering.compression.mode, 'turboquant');
  assert.equal(merged.tiering.mode, 'off', 'tiering.mode should be preserved from base');
  assert.equal(merged.tiering.hotWindow, 1024, 'tiering.hotWindow should be preserved');
  assert.equal(merged.quantization.mode, 'turboquant');
  assert.equal(merged.quantization.bitWidth, 4);
}

// Profile-style override: only mode and bitWidth
{
  const base = { ...DEFAULT_KVCACHE_CONFIG };
  const profile = {
    layout: 'contiguous',
    quantization: {
      mode: 'turboquant',
      bitWidth: 4,
    },
  };
  const merged = mergeRuntimeValues(base, profile);
  assert.equal(merged.layout, 'contiguous');
  assert.equal(merged.quantization.mode, 'turboquant');
  assert.equal(merged.quantization.bitWidth, 4);
  assert.equal(merged.quantization.prodMode, false, 'prodMode default preserved in profile merge');
}

// null override disables quantization entirely
{
  const base = {
    ...DEFAULT_KVCACHE_CONFIG,
    quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
  };
  const override = { quantization: null };
  const merged = mergeRuntimeValues(base, override);
  assert.equal(merged.quantization, null, 'null override should disable quantization');
}

console.log('turboquant-config-merge: merge tests passed');

// ===========================================================================
// Part 2: createKVCache layout resolution for contiguous_quantized
// ===========================================================================

// Full-attention model + quantization.mode='turboquant' auto-resolves to contiguous_quantized
setDevice(createGPUDevice(), { platformConfig: null });
try {
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['full_attention'],
      }),
      true,   // useGPU
      false,  // debug
      baseRuntimeKV({
        kvDtype: 'f16',
        quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
      })
    );
    assert.ok(
      cache instanceof QuantizedKVCache,
      'Full-attention + turboquant should create QuantizedKVCache'
    );
    assert.equal(cache.layout, 'contiguous_quantized');
    assert.equal(cache.quantMode, 'turboquant');
    assert.equal(cache.bitWidth, 4);
    assert.equal(cache.prodMode, false);
    assert.ok(cache.rotationMatrixBuffer, 'Should have rotation matrix buffer');
    assert.ok(cache.codebookCentroidsBuffer, 'Should have codebook centroids buffer');
    assert.ok(cache.codebookBoundariesBuffer, 'Should have codebook boundaries buffer');
    assert.equal(cache.qjlMatrixBuffer, null, 'Non-prod mode should not have QJL buffer');
    cache.destroy();
  }

  // Full-attention + turboquant_prod creates QuantizedKVCache with QJL buffer
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['full_attention'],
      }),
      true,
      false,
      baseRuntimeKV({
        kvDtype: 'f16',
        quantization: { mode: 'turboquant_prod', bitWidth: 4, prodMode: true },
      })
    );
    assert.ok(cache instanceof QuantizedKVCache);
    assert.equal(cache.layout, 'contiguous_quantized');
    assert.equal(cache.quantMode, 'turboquant_prod');
    assert.equal(cache.prodMode, true);
    assert.ok(cache.qjlMatrixBuffer, 'Prod mode should have QJL buffer');
    cache.destroy();
  }

  // Full-attention + quantization.mode='none' (default) stays contiguous
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['full_attention'],
      }),
      false,
      false,
      baseRuntimeKV({
        maxSeqLen: 1024,
      })
    );
    assert.ok(cache instanceof KVCache);
    assert.equal(cache.layout, 'contiguous', 'Default quantization.mode=none should stay contiguous');
    assert.ok(!(cache instanceof QuantizedKVCache));
  }

  // Mixed attention (full + sliding) + turboquant should also resolve to contiguous_quantized
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention', 'full_attention'],
        slidingWindow: 512,
      }),
      true,
      false,
      baseRuntimeKV({
        kvDtype: 'f16',
        quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
      })
    );
    assert.ok(cache instanceof QuantizedKVCache);
    assert.equal(cache.layout, 'contiguous_quantized');
    cache.destroy();
  }

  // Sliding-window-only model + turboquant should NOT get contiguous_quantized
  // (forceContiguousKVCache = false, so paged upgrade fires first if maxSeqLen >= threshold)
  {
    const cache = createKVCache(
      baseModelConfig({
        layerTypes: ['sliding_attention'],
        slidingWindow: 512,
      }),
      false,
      false,
      baseRuntimeKV({
        maxSeqLen: 1024,
        quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
      })
    );
    // Sliding-only below threshold stays contiguous (quantization.mode is set but
    // forceContiguousKVCache is false, so the contiguous_quantized auto-resolve doesn't fire).
    assert.equal(cache.layout, 'contiguous',
      'Sliding-only model should not auto-resolve to contiguous_quantized'
    );
    assert.ok(!(cache instanceof QuantizedKVCache));
  }

  // contiguous_quantized requires f16
  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['full_attention'],
        }),
        true,
        false,
        baseRuntimeKV({
          kvDtype: 'f32',
          quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
        })
      ),
      /Contiguous quantized KV cache requires kvDtype="f16"/,
      'contiguous_quantized with f32 kvDtype must throw'
    );
  }

  // contiguous_quantized requires GPU — when useGPU=false, kvDtype resolves to f32
  // (no shader-f16 capability), so the f16 gate fires first. Either error is acceptable
  // since both guard the contiguous_quantized path.
  {
    assert.throws(
      () => createKVCache(
        baseModelConfig({
          layerTypes: ['full_attention'],
        }),
        false,  // useGPU = false
        false,
        baseRuntimeKV({
          kvDtype: 'f16',
          quantization: { mode: 'turboquant', bitWidth: 4, prodMode: false },
        })
      ),
      /Contiguous quantized KV cache requires/,
      'contiguous_quantized without GPU must throw'
    );
  }

} finally {
  setDevice(null);
}

// ===========================================================================
// Part 3: Partial profile merge into full DEFAULT_KVCACHE_CONFIG
// Simulates what happens when a profile like turboquant-contiguous.json
// is merged into the default config.
// ===========================================================================

{
  const profile = {
    layout: 'contiguous',
    quantization: {
      mode: 'turboquant',
      bitWidth: 4,
      prodMode: false,
    },
  };
  const merged = mergeRuntimeValues(DEFAULT_KVCACHE_CONFIG, profile);

  // Quantization fields from profile are set
  assert.equal(merged.quantization.mode, 'turboquant');
  assert.equal(merged.quantization.bitWidth, 4);
  assert.equal(merged.quantization.prodMode, false);

  // All other DEFAULT_KVCACHE_CONFIG fields are preserved
  assert.equal(merged.kvDtype, 'f16');
  assert.equal(merged.pageSize, 256);
  assert.equal(merged.tiering.mode, 'off');
  assert.equal(merged.tiering.hotWindow, 1024);
  assert.equal(merged.tiering.compression.mode, 'none');
  assert.equal(merged.tiering.gating.mode, 'auto');
  assert.equal(merged.bdpaVocabSize, 2048);
}

// Prod profile merge
{
  const profile = {
    layout: 'contiguous',
    quantization: {
      mode: 'turboquant_prod',
      bitWidth: 4,
      prodMode: true,
    },
  };
  const merged = mergeRuntimeValues(DEFAULT_KVCACHE_CONFIG, profile);
  assert.equal(merged.quantization.mode, 'turboquant_prod');
  assert.equal(merged.quantization.prodMode, true);
  assert.equal(merged.tiering.mode, 'off', 'tiering.mode preserved after prod profile merge');
}

console.log('turboquant-config-merge: layout resolution tests passed');
console.log('turboquant-config-merge.test: ok');
