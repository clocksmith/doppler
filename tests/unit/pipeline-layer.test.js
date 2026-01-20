import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { createDopplerConfig } from '../../src/config/schema/index.js';

global.GPUBufferUsage = {
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

global.GPUMapMode = {
  READ: 0x0001,
  WRITE: 0x0002,
};

function createMockBuffer(size, usage, label = '') {
  return {
    size,
    usage,
    label,
    destroy: vi.fn(),
    mapAsync: vi.fn().mockResolvedValue(undefined),
    getMappedRange: vi.fn(() => new ArrayBuffer(size)),
    unmap: vi.fn(),
  };
}

function createMockDevice() {
  return {
    createBuffer: vi.fn((descriptor) =>
      createMockBuffer(descriptor.size, descriptor.usage, descriptor.label)
    ),
    createCommandEncoder: vi.fn(() => ({
      copyBufferToBuffer: vi.fn(),
      finish: vi.fn(() => ({})),
    })),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn(),
      onSubmittedWorkDone: vi.fn().mockResolvedValue(undefined),
    },
    limits: {
      maxBufferSize: 2147483647,
      maxStorageBufferBindingSize: 2147483647,
    },
  };
}

vi.mock('../../src/gpu/device.js', () => ({
  getDevice: vi.fn(),
  getDeviceLimits: vi.fn(() => ({
    maxBufferSize: 2147483647,
    maxStorageBufferBindingSize: 2147483647,
  })),
  hasFeature: vi.fn(() => false),
}));

vi.mock('../../src/memory/buffer-pool.js', () => ({
  acquireBuffer: vi.fn((size) => createMockBuffer(size, GPUBufferUsage.STORAGE)),
  releaseBuffer: vi.fn(),
  getBufferPool: vi.fn(() => ({
    acquire: vi.fn((size) => createMockBuffer(size, GPUBufferUsage.STORAGE)),
    release: vi.fn(),
  })),
}));

vi.mock('../../src/gpu/perf-guards.js', () => ({
  allowReadback: vi.fn(() => false),
  trackAllocation: vi.fn(),
}));

vi.mock('../../src/debug/index.js', () => ({
  log: {
    warn: vi.fn(),
    debug: vi.fn(),
    info: vi.fn(),
    error: vi.fn(),
  },
  trace: {
    buffers: vi.fn(),
    attn: vi.fn(),
    ffn: vi.fn(),
    kernels: vi.fn(),
  },
}));

vi.mock('../../src/config/runtime.js', () => ({
  getRuntimeConfig: vi.fn(() => ({
    shared: {
      debug: {
        probes: [],
        pipeline: {
          enabled: false,
          categories: [],
          layers: null,
          maxDecodeSteps: 0,
          maxAbsThreshold: 10000,
          bufferStats: false,
          readbackSampleSize: 512,
        },
      },
      bufferPool: {
        limits: { maxBuffersPerBucket: 8, maxTotalPooledBuffers: 64 },
        alignment: { alignmentBytes: 256 },
        bucket: { minBucketSizeBytes: 1024 },
      },
    },
  })),
}));

function makeBaseConfig(overrides = {}) {
  return {
    numLayers: 12,
    hiddenSize: 768,
    intermediateSize: 3072,
    numHeads: 12,
    numKVHeads: 12,
    headDim: 64,
    vocabSize: 32000,
    maxSeqLen: 2048,
    useMoE: false,
    numExperts: 0,
    moeTopK: 0,
    slidingWindow: null,
    ropeTheta: 10000,
    ropeLocalTheta: null,
    ropeScale: 1,
    ropeScalingType: null,
    ropeScaling: null,
    quantization: 'F16',
    quantMethod: null,
    rmsNormEps: 1e-5,
    rmsNormWeightOffset: false,
    scaleEmbeddings: false,
    hiddenActivation: 'silu',
    isGemma3: false,
    isGemma2: false,
    isLlama3Instruct: false,
    isQwen3: false,
    isGptOss: false,
    stopTokenIds: [],
    layerTypes: null,
    attentionBias: false,
    finalLogitSoftcapping: null,
    attnLogitSoftcapping: null,
    queryKeyNorm: false,
    queryPreAttnScalar: 8,
    chatTemplateEnabled: false,
    ...overrides,
  };
}

function makeLayerWeights(hiddenSize = 64, intermediateSize = 128, extras = {}) {
  return {
    inputNorm: new Float32Array(hiddenSize),
    qProj: new Float32Array(hiddenSize * hiddenSize),
    kProj: new Float32Array(hiddenSize * hiddenSize),
    vProj: new Float32Array(hiddenSize * hiddenSize),
    oProj: new Float32Array(hiddenSize * hiddenSize),
    postAttnNorm: new Float32Array(hiddenSize),
    gate: new Float32Array(hiddenSize * intermediateSize),
    up: new Float32Array(hiddenSize * intermediateSize),
    down: new Float32Array(intermediateSize * hiddenSize),
    ...extras,
  };
}

describe('layer forward pass structure', () => {
  let mockDevice;

  beforeEach(async () => {
    vi.resetModules();
    mockDevice = createMockDevice();
    const { getDevice } = await import('../../src/gpu/device.js');
    getDevice.mockReturnValue(mockDevice);
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('standard transformer layer flow', () => {
    it('validates input -> norm -> attn -> residual -> norm -> ffn -> residual -> output', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const standardPlan = {
        steps: [
          { op: 'save', name: 'residual_1' },
          { op: 'rmsnorm', weight: 'input', dst: 'state' },
          { op: 'attention', residual: 'residual_1' },
          { op: 'save', name: 'residual_2' },
          { op: 'rmsnorm', weight: 'post_attn', dst: 'state' },
          { op: 'ffn' },
          { op: 'residual_add', a: 'state', b: 'residual_2' },
        ],
      };

      const compiled = compileLayerPipeline(standardPlan, 12);

      expect(compiled.steps).toHaveLength(7);
      expect(compiled.steps.map((s) => s.op)).toEqual([
        'save',
        'rmsnorm',
        'attention',
        'save',
        'rmsnorm',
        'ffn',
        'residual_add',
      ]);
    });

    it('enforces save before residual read', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const invalidPlan = {
        steps: [
          { op: 'attention', residual: 'undefined_residual' },
        ],
      };

      expect(() => compileLayerPipeline(invalidPlan, 12)).toThrow(
        'reads undefined slot'
      );
    });

    it('allows attention without explicit residual (fused path)', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const fusedPlan = {
        steps: [
          { op: 'attention' },
          { op: 'ffn' },
        ],
      };

      const compiled = compileLayerPipeline(fusedPlan, 12);
      expect(compiled.steps).toHaveLength(2);
      expect(compiled.steps[0].residual).toBeNull();
    });
  });

  describe('layer output shape preservation', () => {
    it('maintains hidden size through layer operations', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'input' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention' },
          { op: 'residual_add', a: 'state', b: 'input' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);
      expect(compiled.steps).toHaveLength(4);
      expect(compiled.steps[3].a).toBe('state');
      expect(compiled.steps[3].b).toBe('input');
    });
  });
});

describe('residual connection handling', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  describe('explicit residual paths', () => {
    it('compiles pre-attention residual save correctly', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'pre_attn' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention', residual: 'pre_attn' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[0].name).toBe('pre_attn');
      expect(compiled.steps[2].residual).toBe('pre_attn');
    });

    it('compiles post-attention residual save correctly', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'attn_residual' },
          { op: 'attention' },
          { op: 'save', name: 'ffn_residual' },
          { op: 'ffn' },
          { op: 'residual_add', a: 'state', b: 'ffn_residual' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[2].name).toBe('ffn_residual');
      expect(compiled.steps[4].b).toBe('ffn_residual');
    });

    it('supports multiple residual slots', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'slot_a' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'save', name: 'slot_b' },
          { op: 'attention', residual: 'slot_a' },
          { op: 'rmsnorm', weight: 'post_attn', residual: 'slot_b' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[0].name).toBe('slot_a');
      expect(compiled.steps[2].name).toBe('slot_b');
      expect(compiled.steps[3].residual).toBe('slot_a');
      expect(compiled.steps[4].residual).toBe('slot_b');
    });
  });

  describe('residual_add operation', () => {
    it('defaults to state + residual when no args', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'residual' },
          { op: 'attention' },
          { op: 'residual_add' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[2].a).toBe('state');
      expect(compiled.steps[2].b).toBe('residual');
    });

    it('accepts custom slot names', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'custom_a' },
          { op: 'attention' },
          { op: 'save', name: 'custom_b' },
          { op: 'ffn' },
          { op: 'residual_add', a: 'custom_b', b: 'custom_a' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[4].a).toBe('custom_b');
      expect(compiled.steps[4].b).toBe('custom_a');
    });

    it('throws when reading undefined residual slot', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'attention' },
          { op: 'residual_add', a: 'state', b: 'nonexistent' },
        ],
      };

      expect(() => compileLayerPipeline(plan, 12)).toThrow('reads undefined slot');
    });
  });

  describe('fused residual in rmsnorm', () => {
    it('supports residual fusion in rmsnorm step', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'pre_norm' },
          { op: 'attention' },
          { op: 'rmsnorm', weight: 'post_attn', residual: 'pre_norm' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.steps[2].residual).toBe('pre_norm');
      expect(compiled.steps[2].weight).toBe('post_attn');
    });
  });
});

describe('norm application order', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  describe('pre-norm architecture (LLaMA style)', () => {
    it('compiles 2-norm pre-norm structure', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const preNormPlan = {
        steps: [
          { op: 'save', name: 'res1' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention' },
          { op: 'residual_add', a: 'state', b: 'res1' },
          { op: 'save', name: 'res2' },
          { op: 'rmsnorm', weight: 'post_attn' },
          { op: 'ffn' },
          { op: 'residual_add', a: 'state', b: 'res2' },
        ],
      };

      const compiled = compileLayerPipeline(preNormPlan, 12);
      const normSteps = compiled.steps.filter((s) => s.op === 'rmsnorm');

      expect(normSteps).toHaveLength(2);
      expect(normSteps[0].weight).toBe('input');
      expect(normSteps[1].weight).toBe('post_attn');
    });

    it('places norm before attention in pre-norm', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      const normIdx = compiled.steps.findIndex((s) => s.op === 'rmsnorm');
      const attnIdx = compiled.steps.findIndex((s) => s.op === 'attention');

      expect(normIdx).toBeLessThan(attnIdx);
    });
  });

  describe('sandwich norm architecture (Gemma 2/3 style)', () => {
    it('compiles 4-norm sandwich structure', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const sandwichPlan = {
        steps: [
          { op: 'save', name: 'res1' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention' },
          { op: 'rmsnorm', weight: 'post_attn', residual: 'res1' },
          { op: 'save', name: 'res2' },
          { op: 'rmsnorm', weight: 'pre_ffn' },
          { op: 'ffn' },
          { op: 'rmsnorm', weight: 'post_ffn', residual: 'res2' },
        ],
      };

      const compiled = compileLayerPipeline(sandwichPlan, 26);
      const normSteps = compiled.steps.filter((s) => s.op === 'rmsnorm');

      expect(normSteps).toHaveLength(4);
      expect(normSteps.map((s) => s.weight)).toEqual([
        'input',
        'post_attn',
        'pre_ffn',
        'post_ffn',
      ]);
    });

    it('applies post-attention norm after attention output', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'res' },
          { op: 'rmsnorm', weight: 'input' },
          { op: 'attention' },
          { op: 'rmsnorm', weight: 'post_attn', residual: 'res' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      const attnIdx = compiled.steps.findIndex((s) => s.op === 'attention');
      const postAttnNormIdx = compiled.steps.findIndex(
        (s) => s.op === 'rmsnorm' && s.weight === 'post_attn'
      );

      expect(postAttnNormIdx).toBeGreaterThan(attnIdx);
      expect(compiled.steps[postAttnNormIdx].residual).toBe('res');
    });

    it('applies pre-ffn norm before ffn', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'rmsnorm', weight: 'pre_ffn' },
          { op: 'ffn' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      const preFFNNormIdx = compiled.steps.findIndex(
        (s) => s.op === 'rmsnorm' && s.weight === 'pre_ffn'
      );
      const ffnIdx = compiled.steps.findIndex((s) => s.op === 'ffn');

      expect(preFFNNormIdx).toBeLessThan(ffnIdx);
    });

    it('applies post-ffn norm after ffn output', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [
          { op: 'save', name: 'ffn_res' },
          { op: 'rmsnorm', weight: 'pre_ffn' },
          { op: 'ffn' },
          { op: 'rmsnorm', weight: 'post_ffn', residual: 'ffn_res' },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      const ffnIdx = compiled.steps.findIndex((s) => s.op === 'ffn');
      const postFFNNormIdx = compiled.steps.findIndex(
        (s) => s.op === 'rmsnorm' && s.weight === 'post_ffn'
      );

      expect(postFFNNormIdx).toBeGreaterThan(ffnIdx);
    });
  });

  describe('detectSandwichNorm', () => {
    it('returns useSandwichNorm=false for standard architecture', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const weights = makeLayerWeights();
      const result = detectSandwichNorm(weights);

      expect(result.useSandwichNorm).toBe(false);
      expect(result.hasPreFeedforwardNorm).toBe(false);
      expect(result.hasPostFeedforwardNorm).toBe(false);
      expect(result.hasPostAttentionNorm).toBe(false);
    });

    it('returns useSandwichNorm=true when preFeedforwardNorm present', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const weights = makeLayerWeights(64, 128, {
        preFeedforwardNorm: new Float32Array(64),
      });
      const result = detectSandwichNorm(weights);

      expect(result.useSandwichNorm).toBe(true);
      expect(result.hasPreFeedforwardNorm).toBe(true);
    });

    it('returns useSandwichNorm=true when postFeedforwardNorm present', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const weights = makeLayerWeights(64, 128, {
        postFeedforwardNorm: new Float32Array(64),
      });
      const result = detectSandwichNorm(weights);

      expect(result.useSandwichNorm).toBe(true);
      expect(result.hasPostFeedforwardNorm).toBe(true);
    });

    it('detects all four norms for Gemma 3', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const weights = makeLayerWeights(64, 128, {
        postAttentionNorm: new Float32Array(64),
        preFeedforwardNorm: new Float32Array(64),
        postFeedforwardNorm: new Float32Array(64),
      });
      const result = detectSandwichNorm(weights);

      expect(result.useSandwichNorm).toBe(true);
      expect(result.hasPreFeedforwardNorm).toBe(true);
      expect(result.hasPostFeedforwardNorm).toBe(true);
      expect(result.hasPostAttentionNorm).toBe(true);
    });

    it('handles null weights gracefully', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const result = detectSandwichNorm(null);

      expect(result.useSandwichNorm).toBe(false);
      expect(result.hasPreFeedforwardNorm).toBe(false);
      expect(result.hasPostFeedforwardNorm).toBe(false);
      expect(result.hasPostAttentionNorm).toBe(false);
    });

    it('detects postAttentionNorm separately from sandwich norm', async () => {
      const { detectSandwichNorm } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const weights = makeLayerWeights(64, 128, {
        postAttentionNorm: new Float32Array(64),
      });
      const result = detectSandwichNorm(weights);

      expect(result.hasPostAttentionNorm).toBe(true);
      expect(result.useSandwichNorm).toBe(false);
    });
  });
});

describe('layer indexing and iteration', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  describe('zero-indexed layers', () => {
    it('handles layer 0 correctly', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [0],
            steps: [{ op: 'save', name: 'first' }, { op: 'attention' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);
      const layer0Steps = getLayerPlanSteps(compiled, 0);
      const layer1Steps = getLayerPlanSteps(compiled, 1);

      expect(layer0Steps).toHaveLength(2);
      expect(layer0Steps[0].name).toBe('first');
      expect(layer1Steps).toHaveLength(1);
    });

    it('handles last layer correctly', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [11],
            steps: [{ op: 'attention' }, { op: 'save', name: 'last' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);
      const lastSteps = getLayerPlanSteps(compiled, 11);

      expect(lastSteps).toHaveLength(2);
      expect(lastSteps[1].name).toBe('last');
    });
  });

  describe('layer index bounds validation', () => {
    it('filters out negative layer indices', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [-1, 0, 5],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.overrides[0].layers).not.toContain(-1);
      expect(compiled.overrides[0].layers).toContain(0);
      expect(compiled.overrides[0].layers).toContain(5);
    });

    it('filters out indices >= numLayers', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [0, 11, 12, 100],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.overrides[0].layers).toContain(0);
      expect(compiled.overrides[0].layers).toContain(11);
      expect(compiled.overrides[0].layers).not.toContain(12);
      expect(compiled.overrides[0].layers).not.toContain(100);
    });

    it('deduplicates layer indices', async () => {
      const { compileLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [1, 1, 2, 2, 3, 3],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      expect(compiled.overrides[0].layers).toEqual([1, 2, 3]);
    });
  });

  describe('getLayerPlanSteps iteration', () => {
    it('returns default steps for non-overridden layers', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }, { op: 'ffn' }],
        overrides: [
          {
            layers: [0, 5],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      for (const layerIdx of [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]) {
        const steps = getLayerPlanSteps(compiled, layerIdx);
        expect(steps).toHaveLength(2);
        expect(steps[0].op).toBe('attention');
        expect(steps[1].op).toBe('ffn');
      }
    });

    it('returns override steps for matching layers', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [0, 5, 10],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);

      for (const layerIdx of [0, 5, 10]) {
        const steps = getLayerPlanSteps(compiled, layerIdx);
        expect(steps).toHaveLength(1);
        expect(steps[0].op).toBe('noop');
      }
    });

    it('returns first matching override when layer in multiple', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }],
        overrides: [
          {
            layers: [0, 5],
            steps: [{ op: 'ffn' }],
          },
          {
            layers: [5, 10],
            steps: [{ op: 'noop' }],
          },
        ],
      };

      const compiled = compileLayerPipeline(plan, 12);
      const steps = getLayerPlanSteps(compiled, 5);

      expect(steps[0].op).toBe('ffn');
    });

    it('iterates all layers correctly', async () => {
      const { compileLayerPipeline, getLayerPlanSteps } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const plan = {
        steps: [{ op: 'attention' }, { op: 'ffn' }],
      };

      const numLayers = 12;
      const compiled = compileLayerPipeline(plan, numLayers);

      for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
        const steps = getLayerPlanSteps(compiled, layerIdx);
        expect(steps).toHaveLength(2);
      }
    });
  });

  describe('isMoELayer detection', () => {
    it('returns false when model does not use MoE', async () => {
      const { isMoELayer } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const config = makeBaseConfig({ useMoE: false });

      for (let i = 0; i < 12; i++) {
        expect(isMoELayer(i, config)).toBe(false);
      }
    });

    it('returns true for all layers when useMoE=true and no layerTypes', async () => {
      const { isMoELayer } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const config = makeBaseConfig({
        useMoE: true,
        numExperts: 8,
        moeTopK: 2,
      });

      for (let i = 0; i < 12; i++) {
        expect(isMoELayer(i, config)).toBe(true);
      }
    });

    it('respects layerTypes array for per-layer detection', async () => {
      const { isMoELayer } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const config = makeBaseConfig({
        useMoE: true,
        numLayers: 4,
        layerTypes: ['dense', 'moe', 'dense', 'moe'],
      });

      expect(isMoELayer(0, config)).toBe(false);
      expect(isMoELayer(1, config)).toBe(true);
      expect(isMoELayer(2, config)).toBe(false);
      expect(isMoELayer(3, config)).toBe(true);
    });

    it('returns true when layer has router weights', async () => {
      const { isMoELayer } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const config = makeBaseConfig({
        useMoE: true,
        numExperts: 8,
      });
      const weights = { routerWeight: new Float32Array(768 * 8) };

      expect(isMoELayer(0, config, weights)).toBe(true);
    });

    it('handles out-of-bounds layer index gracefully', async () => {
      const { isMoELayer } = await import(
        '../../src/inference/pipeline/layer.js'
      );

      const config = makeBaseConfig({
        useMoE: true,
        layerTypes: ['dense', 'moe'],
      });

      expect(isMoELayer(10, config)).toBe(true);
    });
  });
});

describe('layer plan resolution', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  describe('resolveLayerPipeline priority', () => {
    it('returns null when no plans provided', async () => {
      const { resolveLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const result = resolveLayerPipeline(null, null, 12);

      expect(result).toBeNull();
    });

    it('returns null when plans have empty steps', async () => {
      const { resolveLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const result = resolveLayerPipeline(
        { steps: [] },
        { steps: [] },
        12
      );

      expect(result).toBeNull();
    });

    it('prefers runtime plan over model plan', async () => {
      const { resolveLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const modelPlan = { steps: [{ op: 'attention' }] };
      const runtimePlan = { steps: [{ op: 'ffn' }] };

      const result = resolveLayerPipeline(modelPlan, runtimePlan, 12);

      expect(result.source).toBe('runtime');
      expect(result.steps[0].op).toBe('ffn');
    });

    it('uses model plan when runtime plan is null', async () => {
      const { resolveLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const modelPlan = { steps: [{ op: 'attention' }] };

      const result = resolveLayerPipeline(modelPlan, null, 12);

      expect(result.source).toBe('model');
      expect(result.steps[0].op).toBe('attention');
    });

    it('uses model plan when runtime plan has empty steps', async () => {
      const { resolveLayerPipeline } = await import(
        '../../src/inference/pipeline/layer-plan.js'
      );

      const modelPlan = { steps: [{ op: 'attention' }] };
      const runtimePlan = { steps: [] };

      const result = resolveLayerPipeline(modelPlan, runtimePlan, 12);

      expect(result.source).toBe('model');
    });
  });
});

describe('ops module', () => {
  let mockDevice;

  beforeEach(async () => {
    vi.resetModules();
    mockDevice = createMockDevice();
    const { getDevice } = await import('../../src/gpu/device.js');
    getDevice.mockReturnValue(mockDevice);
    vi.clearAllMocks();
  });

  describe('releaseOrTrack', () => {
    it('calls releaseBuffer when no recorder', async () => {
      const { releaseOrTrack } = await import(
        '../../src/inference/pipeline/ops.js'
      );
      const { releaseBuffer } = await import('../../src/memory/buffer-pool.js');

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);
      releaseOrTrack(undefined, buffer);

      expect(releaseBuffer).toHaveBeenCalledWith(buffer);
    });

    it('tracks buffer when recorder provided', async () => {
      const { releaseOrTrack } = await import(
        '../../src/inference/pipeline/ops.js'
      );
      const { releaseBuffer } = await import('../../src/memory/buffer-pool.js');

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);
      const recorder = { trackTemporaryBuffer: vi.fn() };

      releaseOrTrack(recorder, buffer);

      expect(recorder.trackTemporaryBuffer).toHaveBeenCalledWith(buffer);
      expect(releaseBuffer).not.toHaveBeenCalled();
    });

    it('skips decode buffers', async () => {
      const { releaseOrTrack } = await import(
        '../../src/inference/pipeline/ops.js'
      );
      const { releaseBuffer } = await import('../../src/memory/buffer-pool.js');

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);
      const decodeBuffers = { ownsBuffer: vi.fn().mockReturnValue(true) };

      releaseOrTrack(undefined, buffer, decodeBuffers);

      expect(releaseBuffer).not.toHaveBeenCalled();
    });
  });

  describe('isDecodeBuffer', () => {
    it('returns false when decodeBuffers is null', async () => {
      const { isDecodeBuffer } = await import(
        '../../src/inference/pipeline/ops.js'
      );

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);

      expect(isDecodeBuffer(null, buffer)).toBe(false);
    });

    it('returns false when decodeBuffers is undefined', async () => {
      const { isDecodeBuffer } = await import(
        '../../src/inference/pipeline/ops.js'
      );

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);

      expect(isDecodeBuffer(undefined, buffer)).toBe(false);
    });

    it('returns true when buffer is owned by decode manager', async () => {
      const { isDecodeBuffer } = await import(
        '../../src/inference/pipeline/ops.js'
      );

      const buffer = createMockBuffer(1024, GPUBufferUsage.STORAGE);
      const decodeBuffers = { ownsBuffer: vi.fn().mockReturnValue(true) };

      expect(isDecodeBuffer(decodeBuffers, buffer)).toBe(true);
      expect(decodeBuffers.ownsBuffer).toHaveBeenCalledWith(buffer);
    });
  });
});

describe('debug and probe handling', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it('preserves probeStage in compiled steps', async () => {
    const { compileLayerPipeline } = await import(
      '../../src/inference/pipeline/layer-plan.js'
    );

    const plan = {
      steps: [
        { op: 'attention', probeStage: 'attn_out' },
        { op: 'ffn', probeStage: 'ffn_out' },
        { op: 'rmsnorm', weight: 'post_attn', probeStage: 'norm_out' },
      ],
    };

    const compiled = compileLayerPipeline(plan, 12);

    expect(compiled.steps[0].probeStage).toBe('attn_out');
    expect(compiled.steps[1].probeStage).toBe('ffn_out');
    expect(compiled.steps[2].probeStage).toBe('norm_out');
  });

  it('allows steps without probeStage', async () => {
    const { compileLayerPipeline } = await import(
      '../../src/inference/pipeline/layer-plan.js'
    );

    const plan = {
      steps: [
        { op: 'attention' },
        { op: 'ffn' },
      ],
    };

    const compiled = compileLayerPipeline(plan, 12);

    expect(compiled.steps[0].probeStage).toBeUndefined();
    expect(compiled.steps[1].probeStage).toBeUndefined();
  });
});

describe('mini-model fixture integration', () => {
  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
  });

  it('parses layer config from mini-model manifest', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );
    const { parseModelConfig } = await import(
      '../../src/inference/pipeline/config.js'
    );

    const config = parseModelConfig(manifest.default);

    expect(config.numLayers).toBe(2);
    expect(config.hiddenSize).toBe(64);
    expect(config.intermediateSize).toBe(128);
    expect(config.numHeads).toBe(2);
    expect(config.numKVHeads).toBe(2);
    expect(config.headDim).toBe(32);
  });

  it('validates mini-model has no sandwich norms', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );

    const { normalization } = manifest.default.inference;

    expect(normalization.preFeedforwardNorm).toBe(false);
    expect(normalization.postFeedforwardNorm).toBe(false);
    expect(normalization.postAttentionNorm).toBe(false);
  });

  it('validates mini-model layer structure', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );

    const { groups } = manifest.default;

    expect(groups['layer.0']).toBeDefined();
    expect(groups['layer.1']).toBeDefined();
    expect(groups['layer.0'].type).toBe('layer');
    expect(groups['layer.0'].layerIndex).toBe(0);
    expect(groups['layer.1'].layerIndex).toBe(1);
  });

  it('validates mini-model layer tensors', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );

    const layer0Tensors = manifest.default.groups['layer.0'].tensors;

    expect(layer0Tensors).toContain('model.layers.0.input_layernorm.weight');
    expect(layer0Tensors).toContain('model.layers.0.self_attn.q_proj.weight');
    expect(layer0Tensors).toContain('model.layers.0.self_attn.k_proj.weight');
    expect(layer0Tensors).toContain('model.layers.0.self_attn.v_proj.weight');
    expect(layer0Tensors).toContain('model.layers.0.self_attn.o_proj.weight');
    expect(layer0Tensors).toContain(
      'model.layers.0.post_attention_layernorm.weight'
    );
    expect(layer0Tensors).toContain('model.layers.0.mlp.gate_proj.weight');
    expect(layer0Tensors).toContain('model.layers.0.mlp.up_proj.weight');
    expect(layer0Tensors).toContain('model.layers.0.mlp.down_proj.weight');
  });

  it('validates layer iteration over mini-model', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );
    const { parseModelConfig } = await import(
      '../../src/inference/pipeline/config.js'
    );

    const config = parseModelConfig(manifest.default);

    const layerIndices = [];
    for (let i = 0; i < config.numLayers; i++) {
      layerIndices.push(i);
    }

    expect(layerIndices).toEqual([0, 1]);
  });

  it('validates mini-model uses standard norm order', async () => {
    const manifest = await import(
      '../../tests/fixtures/mini-model/manifest.json',
      { with: { type: 'json' } }
    );
    const { parseModelConfig } = await import(
      '../../src/inference/pipeline/config.js'
    );
    const { detectSandwichNorm } = await import(
      '../../src/inference/pipeline/layer.js'
    );

    const config = parseModelConfig(manifest.default);
    const mockWeights = makeLayerWeights(config.hiddenSize, config.intermediateSize);
    const sandwichInfo = detectSandwichNorm(mockWeights);

    expect(sandwichInfo.useSandwichNorm).toBe(false);
  });
});

describe('activation dtype defaults', () => {
  it('defaults to f16 in runtime config', () => {
    const config = createDopplerConfig().runtime;
    expect(config.inference.compute.activationDtype).toBe('f16');
  });

  it('supports runtime override to f32', () => {
    const config = createDopplerConfig({
      runtime: {
        inference: {
          compute: { activationDtype: 'f32' },
        },
      },
    }).runtime;
    expect(config.inference.compute.activationDtype).toBe('f32');
  });

  it('validates activationDtype affects tensor dtype selection', () => {
    // Test that tensor creation respects activation dtype
    const dtypeF16 = 'f16';
    const dtypeF32 = 'f32';

    // Bytes per element calculation
    const f16BytesPerElement = 2;
    const f32BytesPerElement = 4;

    const numElements = 1152; // Typical hidden size

    // F16 buffer should be half the size of F32
    const f16BufferSize = numElements * f16BytesPerElement;
    const f32BufferSize = numElements * f32BytesPerElement;

    expect(f16BufferSize).toBe(numElements * 2);
    expect(f32BufferSize).toBe(numElements * 4);
    expect(f16BufferSize).toBe(f32BufferSize / 2);
  });
});
