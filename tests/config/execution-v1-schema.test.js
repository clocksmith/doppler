import {
  expandExecutionV1,
  hasExecutionV1,
  EXECUTION_V1_SCHEMA_ID,
  DEFAULT_EXECUTION_V1_POLICIES,
} from '../../src/config/schema/index.js';
import assert from 'node:assert/strict';
import { createDopplerConfig } from '../../src/config/schema/doppler.schema.js';
import { applyExecutionV1RuntimeConfig, compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { readFileSync } from 'fs';

const D = (c) => 'sha256:' + c.repeat(64);

function makeKernels() {
  return {
    rmsnorm: { kernel: 'rmsnorm.wgsl', entry: 'main', digest: D('a') },
    gemv: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4', digest: D('b') },
    attn: { kernel: 'attention_decode_online_f16kv.wgsl', entry: 'main', digest: D('c'), precision: { kvDtype: 'f16' } },
    gelu: { kernel: 'gelu.wgsl', entry: 'main', digest: D('d'), constants: { HAS_GATE: true } },
    residual: { kernel: 'residual.wgsl', entry: 'main', digest: D('e') },
    rope: { kernel: 'rope.wgsl', entry: 'main', digest: D('f') },
    embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: D('1') },
    lm_head: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_multicol', digest: D('2') },
    sample: { kernel: 'sample.wgsl', entry: 'sample_single_pass', digest: D('3') },
  };
}

function makeGraph() {
  return {
    kernels: makeKernels(),
    preLayer: [['embed', 'embed', 'embed_tokens']],
    decode: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
      ['k_proj', 'gemv', 'layer.{L}.self_attn.k_proj'],
      ['v_proj', 'gemv', 'layer.{L}.self_attn.v_proj'],
      ['rope_q', 'rope'],
      ['rope_k', 'rope'],
      ['attention', 'attn'],
      ['o_proj', 'gemv', 'layer.{L}.self_attn.o_proj'],
      ['attn_residual', 'residual'],
      ['post_attn_norm', 'rmsnorm'],
      ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
      ['up_proj', 'gemv', 'layer.{L}.mlp.up_proj'],
      ['activation', 'gelu'],
      ['down_proj', 'gemv', 'layer.{L}.mlp.down_proj'],
      ['ffn_residual', 'residual'],
    ],
    prefill: [
      ['input_norm', 'rmsnorm'],
      ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
      ['attention', 'attn'],
      ['o_proj', 'gemv', 'layer.{L}.self_attn.o_proj'],
      ['ffn_residual', 'residual'],
    ],
    postLayer: [
      ['final_norm', 'rmsnorm'],
      ['lm_head', 'lm_head', 'lm_head'],
      ['sample', 'sample'],
    ],
    policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
  };
}

// === hasExecutionV1 ===
const v1Inference = { schema: EXECUTION_V1_SCHEMA_ID, execution: { kernels: {} } };
if (!hasExecutionV1(v1Inference)) throw new Error('hasExecutionV1 should detect v1');
if (hasExecutionV1({ schema: 'doppler.execution/v0', execution: { steps: [] } })) {
  throw new Error('hasExecutionV1 should not detect v0');
}
if (hasExecutionV1(null)) throw new Error('hasExecutionV1 should handle null');
if (hasExecutionV1({})) throw new Error('hasExecutionV1 should handle empty');

// === expandExecutionV1 ===
const graph = makeGraph();
const expanded = expandExecutionV1(graph);
if (expanded.length !== 24) throw new Error(`Expected 24 steps, got ${expanded.length}`);

const decodeSteps = expanded.filter((s) => s.phase === 'decode');
const prefillSteps = expanded.filter((s) => s.phase === 'prefill');
const bothSteps = expanded.filter((s) => s.phase === 'both');
if (decodeSteps.length !== 15) throw new Error(`Expected 15 decode, got ${decodeSteps.length}`);
if (prefillSteps.length !== 5) throw new Error(`Expected 5 prefill, got ${prefillSteps.length}`);
if (bothSteps.length !== 4) throw new Error(`Expected 4 both, got ${bothSteps.length}`);

// First step is embed (preLayer, both)
if (expanded[0].op !== 'embed') throw new Error(`First op should be embed, got ${expanded[0].op}`);
if (expanded[0].phase !== 'both') throw new Error('embed should be phase both');
if (expanded[0].section !== 'preLayer') throw new Error('embed should be section preLayer');
if (expanded[0].weights !== 'embed_tokens') throw new Error('embed should have weights');

// Decode q_proj
const qProj = expanded.find((s) => s.phase === 'decode' && s.op === 'q_proj');
if (!qProj) throw new Error('Missing decode q_proj');
if (qProj.kernel !== 'matmul_gemv_subgroup.wgsl') throw new Error('Wrong kernel for q_proj');
if (qProj.entry !== 'main_vec4') throw new Error('Wrong entry for q_proj');
if (qProj.weights !== 'layer.{L}.self_attn.q_proj') throw new Error('Wrong weights for q_proj');
if (qProj.src !== 'state') throw new Error('Expanded q_proj should carry explicit src=state');
if (qProj.dst !== 'state') throw new Error('Expanded q_proj should carry explicit dst=state');

const decodeAttention = expanded.find((s) => s.phase === 'decode' && s.op === 'attention');
if (!decodeAttention) throw new Error('Missing decode attention');
if (decodeAttention.precision?.kvDtype !== 'f16') {
  throw new Error(`Expected attention precision.kvDtype=f16, got ${JSON.stringify(decodeAttention.precision)}`);
}

const castGraph = makeGraph();
castGraph.kernels.cast = {
  kernel: 'cast_f32_to_f16.wgsl',
  entry: 'main',
  digest: D('4'),
  precision: { inputDtype: 'f32', outputDtype: 'f16' },
};
castGraph.preLayer.push(['cast', 'cast']);
const castStep = expandExecutionV1(castGraph).find((s) => s.section === 'preLayer' && s.op === 'cast');
if (!castStep) throw new Error('Missing expanded cast step');
if (castStep.fromDtype !== 'f32' || castStep.toDtype !== 'f16') {
  throw new Error(`Expected cast f32->f16, got ${castStep.fromDtype}->${castStep.toDtype}`);
}

const invalidCastGraph = makeGraph();
invalidCastGraph.kernels.cast = {
  kernel: 'cast_f32_to_f16.wgsl',
  entry: 'main',
  digest: D('5'),
};
invalidCastGraph.preLayer.push(['cast', 'cast']);
assert.throws(
  () => expandExecutionV1(invalidCastGraph),
  /cast steps require kernel precision\.inputDtype and precision\.outputDtype/
);

// Constants propagation (gelu has HAS_GATE)
const activation = expanded.find((s) => s.phase === 'decode' && s.op === 'activation');
if (!activation) throw new Error('Missing decode activation');
if (!activation.constants?.HAS_GATE) throw new Error('activation should have HAS_GATE constant');

// === Hybrid layer groups ===
const hybridGraph = {
  kernels: makeKernels(),
  preLayer: [],
  decode: [
    ['input_norm', 'rmsnorm'],
    { layers: [3, 7, 11], steps: [
      ['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj'],
      ['attention', 'attn'],
    ] },
    { layers: [0, 1, 2, 4, 5, 6], steps: [
      ['gate_proj', 'gemv', 'layer.{L}.mlp.gate_proj'],
    ] },
    ['ffn_residual', 'residual'],
  ],
  prefill: [],
  postLayer: [],
  policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
};

const hybridExpanded = expandExecutionV1(hybridGraph);
if (hybridExpanded.length !== 5) throw new Error(`Expected 5 hybrid steps, got ${hybridExpanded.length}`);

const attnStep = hybridExpanded.find((s) => s.op === 'attention');
if (!attnStep) throw new Error('Missing attention in hybrid');
if (!Array.isArray(attnStep.layers) || attnStep.layers.length !== 3) {
  throw new Error(`attention layers should be [3,7,11], got ${JSON.stringify(attnStep.layers)}`);
}

const gateStep = hybridExpanded.find((s) => s.op === 'gate_proj');
if (!gateStep) throw new Error('Missing gate_proj in hybrid');
if (!Array.isArray(gateStep.layers) || gateStep.layers.length !== 6) {
  throw new Error(`gate_proj layers should be 6 elements, got ${JSON.stringify(gateStep.layers)}`);
}

const normStep = hybridExpanded.find((s) => s.op === 'input_norm');
if (normStep.layers !== 'all') throw new Error('input_norm should target all layers');

// === Validation: bad kernel ref ===
let threw = false;
try {
  expandExecutionV1({
    kernels: makeKernels(),
    preLayer: [],
    decode: [['q_proj', 'nonexistent_kernel']],
    prefill: [],
    postLayer: [],
    policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
  });
} catch {
  threw = true;
}
if (!threw) throw new Error('Should throw on unknown kernel key');

// === Validation: bad digest ===
threw = false;
try {
  expandExecutionV1({
    kernels: { bad: { kernel: 'x.wgsl', entry: 'main', digest: 'not-a-digest' } },
    preLayer: [],
    decode: [['op', 'bad']],
    prefill: [],
    postLayer: [],
    policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
  });
} catch {
  threw = true;
}
if (!threw) throw new Error('Should throw on bad digest');

// === Validation: bad precision dtype ===
threw = false;
try {
  expandExecutionV1({
    kernels: {
      bad: {
        kernel: 'attention_decode_online_f16kv.wgsl',
        entry: 'main',
        digest: D('d'),
        precision: { kvDtype: 'bf16' },
      },
    },
    preLayer: [],
    decode: [['attention', 'bad']],
    prefill: [],
    postLayer: [],
    policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
  });
} catch {
  threw = true;
}
if (!threw) throw new Error('Should throw on invalid precision dtype');

// === Validation: bad inlineKernelPath ===
threw = false;
try {
  expandExecutionV1({
    kernels: makeKernels(),
    inlineKernelPath: 'no',
    preLayer: [],
    decode: [['q_proj', 'gemv', 'layer.{L}.self_attn.q_proj']],
    prefill: [],
    postLayer: [],
    policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
  });
} catch {
  threw = true;
}
if (!threw) throw new Error('Should throw on non-boolean inlineKernelPath');

// === compileExecutionV1 ===
const compiled = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
  },
  modelId: 'test-model',
  numLayers: 26,
});

if (!compiled.resolvedSteps) throw new Error('Missing resolvedSteps');
if (!compiled.runtimeInferencePatch) throw new Error('Missing runtimeInferencePatch');
if (compiled.resolvedSteps.all.length !== 24) {
  throw new Error(`Expected 24 resolved steps, got ${compiled.resolvedSteps.all.length}`);
}
if (!compiled.runtimeInferencePatch.kernelPath) {
  throw new Error('Expected inline kernelPath in runtimeInferencePatch');
}
if (compiled.runtimeInferencePatch.kernelPathSource !== 'execution-v1') {
  throw new Error('Expected kernelPathSource execution-v1');
}
if (!compiled.runtimeInferencePatch.compute?.activationDtype) {
  throw new Error('Expected compute.activationDtype in patch');
}
if (!compiled.runtimeInferencePatch.session?.kvcache?.kvDtype) {
  throw new Error('Expected session.kvcache.kvDtype in patch');
}
if (compiled.runtimeInferencePatch.kernelPath.decode.steps.find((step) => step.op === 'attention')?.precision?.kvDtype !== 'f16') {
  throw new Error('Expected inline kernelPath attention precision.kvDtype to survive compileExecutionV1');
}
if (compiled.runtimeInferencePatch.kvcache) {
  throw new Error('Execution-v1 runtime patch must not mirror session.kvcache to top-level kvcache');
}
if (compiled.runtimeInferencePatch.batching) {
  throw new Error('Execution-v1 runtime patch must not pre-apply decodeLoop batching defaults');
}

const configuredKernelPath = {
  ...structuredClone(compiled.runtimeInferencePatch.kernelPath),
  id: 'runtime-configured-kernel-path',
};
const configuredKernelPathOverrides = {
  inference: {
    kernelPath: configuredKernelPath,
    compute: {
      activationDtype: 'f16',
    },
    session: {
      compute: {
        defaults: {
          activationDtype: 'f16',
          mathDtype: 'f32',
          accumDtype: 'f32',
          outputDtype: 'f16',
        },
      },
      kvcache: {
        kvDtype: 'f16',
      },
    },
  },
};
const appliedConfiguredKernelPath = applyExecutionV1RuntimeConfig({
  runtimeConfig: createDopplerConfig({ runtime: configuredKernelPathOverrides }).runtime,
  runtimeOverrides: configuredKernelPathOverrides,
  manifest: {
    modelId: 'qwen-3-5-2b-q4k-ehaf16',
    architecture: {
      numLayers: 26,
      headDim: 256,
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: makeGraph(),
      layerPattern: {
        layerTypes: Array(26).fill('full_attention'),
      },
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: { kvDtype: 'f16' },
        decodeLoop: null,
      },
    },
  },
  modelId: 'qwen-3-5-2b-q4k-ehaf16',
  numLayers: 26,
  capabilities: {
    hasSubgroups: true,
    hasF16: true,
    maxWorkgroupStorageSize: 32768,
  },
  platform: {
    id: 'metal',
    vendor: 'apple',
    architecture: 'metal-3',
  },
});
assert.equal(
  appliedConfiguredKernelPath.runtimeConfig.inference.kernelPath,
  configuredKernelPath,
  'explicit runtime kernelPath must retain precedence over the execution-v1 compiled path'
);
assert.equal(
  appliedConfiguredKernelPath.runtimeConfig.inference.kernelPathSource,
  'config',
  'explicit runtime kernelPath must be identified as config-owned'
);
assert.equal(
  appliedConfiguredKernelPath.runtimeConfig.inference.compute.activationDtype,
  'f16',
  'config-owned kernelPath activation dtype must not be rewritten by execution-v1 compilation'
);
assert.equal(
  appliedConfiguredKernelPath.runtimeConfig.inference.session.compute.defaults.outputDtype,
  'f16',
  'config-owned kernelPath output dtype must not be rewritten by execution-v1 compilation'
);
assert.notEqual(
  appliedConfiguredKernelPath.executionV1State.runtimeInferencePatch.kernelPath.id,
  configuredKernelPath.id,
  'execution-v1 state must retain its separately compiled manifest path'
);

const greedyLmHeadFusionConfig = createDopplerConfig({
  runtime: {
    inference: {
      session: {
        useGreedyLmHeadArgmaxFusion: true,
        useWideTileQ4KDecode: true,
        useSandwichRMSNormPairFusion: true,
        usePostFfnNextInputRMSNormPairFusion: true,
        usePostAttnNormFusedGateUp: true,
        fusedFfnQ4K: {
          decode: {
            variant: 'q4k_metal_simd16',
            pipelineConstants: {
              COLS_PER_WG: 64,
              THREADS_PER_COL: 4,
            },
          },
        },
        lmHeadArgmaxQ4K: {
          useFullBlockFastPath: true,
          colsPerWorkgroup: 128,
          threadsPerCol: 2,
        },
        attentionDecodeOnline: {
          workgroupSize: 128,
          useDirectContiguousKVLayout: true,
          useOutputGateFusion: true,
        },
        useLinearAttentionABProjectionFusion: true,
        useLinearAttentionQKVZProjectionFusion: true,
        useLinearAttentionFusedDecodeCore: true,
        useWideTileResidualFusion: true,
        useFusedRmsnormWideTile: true,
        useFusedQKVSplitQKNorm: true,
        useFusedQKVSplitQKNormRoPE: true,
        useLargeBatchF16F32FusedGateUp: true,
        skipEmbeddingKVCacheWrites: true,
      },
    },
  },
});
if (createDopplerConfig().runtime.inference.session.useGreedyLmHeadArgmaxFusion !== false) {
  throw new Error('useGreedyLmHeadArgmaxFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useGreedyLmHeadArgmaxFusion !== true) {
  throw new Error('runtime session override must enable useGreedyLmHeadArgmaxFusion');
}
if (createDopplerConfig().runtime.inference.session.useWideTileQ4KDecode !== false) {
  throw new Error('useWideTileQ4KDecode must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useWideTileQ4KDecode !== true) {
  throw new Error('runtime session override must enable useWideTileQ4KDecode');
}
if (createDopplerConfig().runtime.inference.session.useSandwichRMSNormPairFusion !== false) {
  throw new Error('useSandwichRMSNormPairFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useSandwichRMSNormPairFusion !== true) {
  throw new Error('runtime session override must enable useSandwichRMSNormPairFusion');
}
if (createDopplerConfig().runtime.inference.session.usePostFfnNextInputRMSNormPairFusion !== false) {
  throw new Error('usePostFfnNextInputRMSNormPairFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.usePostFfnNextInputRMSNormPairFusion !== true) {
  throw new Error('runtime session override must enable usePostFfnNextInputRMSNormPairFusion');
}
if (createDopplerConfig().runtime.inference.session.usePostAttnNormFusedGateUp !== false) {
  throw new Error('usePostAttnNormFusedGateUp must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.usePostAttnNormFusedGateUp !== true) {
  throw new Error('runtime session override must enable usePostAttnNormFusedGateUp');
}
if (createDopplerConfig().runtime.inference.session.fusedFfnQ4K !== null) {
  throw new Error('fusedFfnQ4K must default null');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.fusedFfnQ4K.decode.pipelineConstants.COLS_PER_WG !== 64) {
  throw new Error('runtime session override must enable fusedFfnQ4K decode constants');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.fusedFfnQ4K.decode.variant !== 'q4k_metal_simd16') {
  throw new Error('runtime session override must enable the explicit fusedFfnQ4K decode variant');
}
if (createDopplerConfig().runtime.inference.session.lmHeadArgmaxQ4K !== null) {
  throw new Error('lmHeadArgmaxQ4K must default null');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.lmHeadArgmaxQ4K.useFullBlockFastPath !== true) {
  throw new Error('runtime session override must enable lmHeadArgmaxQ4K full-block fast path');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.lmHeadArgmaxQ4K.colsPerWorkgroup !== 128) {
  throw new Error('runtime session override must enable lmHeadArgmaxQ4K column grouping');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.lmHeadArgmaxQ4K.threadsPerCol !== 2) {
  throw new Error('runtime session override must enable lmHeadArgmaxQ4K threads-per-column');
}
if (createDopplerConfig().runtime.inference.session.attentionDecodeOnline !== null) {
  throw new Error('attentionDecodeOnline must default null');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.attentionDecodeOnline.workgroupSize !== 128) {
  throw new Error('runtime session override must enable attentionDecodeOnline workgroup sizing');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.attentionDecodeOnline.useDirectContiguousKVLayout !== true) {
  throw new Error('runtime session override must enable attentionDecodeOnline direct contiguous KV layout');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.attentionDecodeOnline.useOutputGateFusion !== true) {
  throw new Error('runtime session override must enable attentionDecodeOnline output gate fusion');
}
if (createDopplerConfig().runtime.inference.session.useLinearAttentionABProjectionFusion !== false) {
  throw new Error('useLinearAttentionABProjectionFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useLinearAttentionABProjectionFusion !== true) {
  throw new Error('runtime session override must enable useLinearAttentionABProjectionFusion');
}
if (createDopplerConfig().runtime.inference.session.useLinearAttentionQKVZProjectionFusion !== false) {
  throw new Error('useLinearAttentionQKVZProjectionFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useLinearAttentionQKVZProjectionFusion !== true) {
  throw new Error('runtime session override must enable useLinearAttentionQKVZProjectionFusion');
}
if (createDopplerConfig().runtime.inference.session.useLinearAttentionFusedDecodeCore !== false) {
  throw new Error('useLinearAttentionFusedDecodeCore must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useLinearAttentionFusedDecodeCore !== true) {
  throw new Error('runtime session override must enable useLinearAttentionFusedDecodeCore');
}
if (createDopplerConfig().runtime.inference.session.useWideTileResidualFusion !== false) {
  throw new Error('useWideTileResidualFusion must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useWideTileResidualFusion !== true) {
  throw new Error('runtime session override must enable useWideTileResidualFusion');
}
if (createDopplerConfig().runtime.inference.session.useFusedRmsnormWideTile !== false) {
  throw new Error('useFusedRmsnormWideTile must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useFusedRmsnormWideTile !== true) {
  throw new Error('runtime session override must enable useFusedRmsnormWideTile');
}
if (createDopplerConfig().runtime.inference.session.useFusedQKVSplitQKNorm !== false) {
  throw new Error('useFusedQKVSplitQKNorm must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useFusedQKVSplitQKNorm !== true) {
  throw new Error('runtime session override must enable useFusedQKVSplitQKNorm');
}
if (createDopplerConfig().runtime.inference.session.useFusedQKVSplitQKNormRoPE !== false) {
  throw new Error('useFusedQKVSplitQKNormRoPE must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useFusedQKVSplitQKNormRoPE !== true) {
  throw new Error('runtime session override must enable useFusedQKVSplitQKNormRoPE');
}
if (createDopplerConfig().runtime.inference.session.useLargeBatchF16F32FusedGateUp !== false) {
  throw new Error('useLargeBatchF16F32FusedGateUp must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.useLargeBatchF16F32FusedGateUp !== true) {
  throw new Error('runtime session override must enable useLargeBatchF16F32FusedGateUp');
}
if (createDopplerConfig().runtime.inference.session.skipEmbeddingKVCacheWrites !== false) {
  throw new Error('skipEmbeddingKVCacheWrites must default false');
}
if (greedyLmHeadFusionConfig.runtime.inference.session.skipEmbeddingKVCacheWrites !== true) {
  throw new Error('runtime session override must enable skipEmbeddingKVCacheWrites');
}
const patchedRuntimeConfig = applyExecutionV1RuntimeConfig({
  runtimeConfig: createDopplerConfig({
    runtime: {
      inference: {
        executionPatch: {
          addKernels: [{
            key: 'attn_chunked',
            kernel: {
              kernel: 'attention_decode_chunked_f16kv.wgsl',
              entry: 'main',
              digest: D('4'),
              precision: { kvDtype: 'f16' },
            },
          }],
          set: [{ section: 'decode', op: 'attention', kernelKey: 'attn_chunked' }],
        },
      },
    },
  }).runtime,
  runtimeOverrides: null,
  manifest: {
    modelId: 'test-model-execution-patch',
    architecture: {
      numLayers: 26,
      headDim: 256,
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: makeGraph(),
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: { kvDtype: 'f16' },
        decodeLoop: null,
      },
    },
  },
  modelId: 'test-model-execution-patch',
  numLayers: 26,
});

const patchedDecodeAttention = patchedRuntimeConfig.runtimeConfig
  .inference?.kernelPath?.decode?.steps?.find((step) => step.op === 'attention');
if (patchedDecodeAttention?.kernel !== 'attention_decode_chunked_f16kv.wgsl') {
  throw new Error('executionPatch.set must update section-scoped decode attention kernel');
}
const patchedPrefillAttention = patchedRuntimeConfig.runtimeConfig
  .inference?.kernelPath?.prefill?.steps?.find((step) => step.op === 'attention');
if (patchedPrefillAttention?.kernel !== 'attention_decode_online_f16kv.wgsl') {
  throw new Error('executionPatch.set with section=decode must not update prefill attention');
}

assert.throws(
  () => applyExecutionV1RuntimeConfig({
    runtimeConfig: createDopplerConfig({
      runtime: {
        inference: {
          executionPatch: {
            remove: [{ op: 'attention' }],
          },
        },
      },
    }).runtime,
    runtimeOverrides: null,
    manifest: {
      modelId: 'test-model-execution-patch-remove',
      architecture: {
        numLayers: 26,
        headDim: 256,
      },
      inference: {
        schema: EXECUTION_V1_SCHEMA_ID,
        execution: makeGraph(),
        session: {
          compute: {
            defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
          },
          kvcache: { kvDtype: 'f16' },
          decodeLoop: null,
        },
      },
    },
    modelId: 'test-model-execution-patch-remove',
    numLayers: 26,
  }),
  /executionPatch\.remove is not supported/
);

const f16Graph = makeGraph();
f16Graph.kernels.attn = {
  kernel: 'attention_streaming_f16.wgsl',
  entry: 'main',
  digest: D('9'),
  precision: { kvDtype: 'f16' },
};

const runtimeSessionActivationOverride = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: f16Graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-f16-runtime-compute',
  numLayers: 26,
  runtimeSession: {
    compute: {
      defaults: {
        activationDtype: 'f16',
        mathDtype: 'f16',
        accumDtype: 'f16',
        outputDtype: 'f16',
      },
    },
  },
});

if (runtimeSessionActivationOverride.session.compute.defaults.activationDtype !== 'f16') {
  throw new Error('Expected runtimeSession activation override to update execution-v1 session activationDtype');
}
if (runtimeSessionActivationOverride.session.compute.defaults.outputDtype !== 'f16') {
  throw new Error('Expected runtimeSession activation override to update execution-v1 session outputDtype');
}
if (runtimeSessionActivationOverride.runtimeInferencePatch.compute?.activationDtype !== 'f16') {
  throw new Error('Expected runtimeInferencePatch.compute.activationDtype=f16 after runtimeSession override');
}

const ignoredRuntimeComputeActivationOverride = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-runtime-compute-ignored',
  numLayers: 26,
  runtimeCompute: {
    activationDtype: 'f16',
  },
});

if (ignoredRuntimeComputeActivationOverride.session.compute.defaults.activationDtype !== 'f32') {
  throw new Error('runtimeCompute.activationDtype must not mutate execution-v1 session activationDtype');
}
if (ignoredRuntimeComputeActivationOverride.runtimeInferencePatch.compute?.activationDtype !== 'f32') {
  throw new Error('runtimeCompute.activationDtype must not mutate runtimeInferencePatch.compute.activationDtype');
}
if (compiled.runtimeInferencePatch.generation) {
  throw new Error('Execution-v1 runtime patch must not pre-apply decodeLoop generation defaults');
}

threw = false;
try {
  compileExecutionV1({
    manifestInference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: graph,
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: { layout: 'contiguous' },
        decodeLoop: null,
      },
    },
    modelId: 'test-model-missing-kv-dtype',
    numLayers: 26,
  });
} catch (error) {
  threw = /session\.kvcache\.kvDtype is required/.test(String(error?.message ?? error));
}
if (!threw) {
  throw new Error('Expected compileExecutionV1 to fail fast when session.kvcache.kvDtype is missing');
}

const compiledTurboQuant = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: {
        layout: 'contiguous',
        kvDtype: 'f16',
        maxSeqLen: 2048,
        tiering: { mode: 'off' },
        quantization: {
          mode: 'turboquant',
          bitWidth: 4,
          prodMode: false,
        },
      },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-turboquant',
  numLayers: 26,
});

if (compiledTurboQuant.runtimeInferencePatch.session?.kvcache?.quantization?.mode !== 'turboquant') {
  throw new Error('Execution-v1 session patch must preserve quantization mode');
}
if (compiledTurboQuant.runtimeInferencePatch.session?.kvcache?.maxSeqLen !== 2048) {
  throw new Error('Execution-v1 session patch must preserve maxSeqLen');
}
if (compiledTurboQuant.runtimeInferencePatch.kvcache) {
  throw new Error('Execution-v1 TurboQuant patch must not mirror session.kvcache to top-level kvcache');
}

const compiledTieredTurboQuant = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: {
        layout: 'tiered',
        kvDtype: 'f16',
        maxSeqLen: 2048,
        tiering: {
          mode: 'turboquant',
          hotWindow: 256,
          coldPageSize: 64,
          coldDtype: 'f16',
          compression: {
            mode: 'turboquant',
            blockSize: 1,
            bitWidth: 4,
            prodMode: false,
          },
          gating: {
            mode: 'force_on',
          },
        },
      },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-tiered-turboquant',
  numLayers: 26,
});

if (compiledTieredTurboQuant.runtimeInferencePatch.session?.kvcache?.tiering?.mode !== 'turboquant') {
  throw new Error('Execution-v1 session patch must preserve tiered TurboQuant mode');
}
if (compiledTieredTurboQuant.runtimeInferencePatch.session?.kvcache?.tiering?.compression?.mode !== 'turboquant') {
  throw new Error('Execution-v1 session patch must preserve tiered TurboQuant compression mode');
}
if (compiledTieredTurboQuant.runtimeInferencePatch.kvcache) {
  throw new Error('Execution-v1 tiered TurboQuant patch must not mirror session.kvcache to top-level kvcache');
}

const compiledWithoutInlineKernelPath = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: {
      ...graph,
      inlineKernelPath: false,
    },
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-no-inline-kernel-path',
  numLayers: 26,
});

if (compiledWithoutInlineKernelPath.runtimeInferencePatch.kernelPath) {
  throw new Error('inlineKernelPath=false should suppress inline kernelPath generation');
}
if (compiledWithoutInlineKernelPath.runtimeInferencePatch.kernelPathSource) {
  throw new Error('inlineKernelPath=false should not stamp kernelPathSource');
}

const compiledWithRuntimeSessionDefaults = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: graph,
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16', layout: 'contiguous' },
      decodeLoop: null,
    },
  },
  modelId: 'test-model-runtime-session-defaults',
  numLayers: 26,
  runtimeSession: {
    compute: {
      defaults: { activationDtype: 'f16' },
    },
    kvcache: { kvDtype: 'f16', layout: 'contiguous', maxSeqLen: 4096 },
  },
  capabilities: { hasF16: true, hasSubgroups: true },
  platform: { id: 'test', vendor: 'test', architecture: 'test' },
  kernelPathPolicy: {
    mode: 'capability-aware',
    sourceScope: ['manifest', 'model', 'config'],
    allowSources: ['manifest', 'model', 'config'],
    onIncompatible: 'remap',
  },
});

if (compiledWithRuntimeSessionDefaults.session?.compute?.defaults?.activationDtype !== 'f16') {
  throw new Error('Execution-v1 must let runtime session activationDtype override manifest defaults');
}
if (compiledWithRuntimeSessionDefaults.session?.kvcache?.kvDtype !== 'f16') {
  throw new Error('Execution-v1 must retain the runtime-overridden kvDtype');
}
if (compiledWithRuntimeSessionDefaults.runtimeInferencePatch.kernelPath?.activationDtype !== 'f16') {
  throw new Error('Execution-v1 inline kernel path must reflect the runtime activationDtype');
}
if (!compiledWithRuntimeSessionDefaults.appliedTransforms?.includes('narrowToF16Activations')) {
  throw new Error('Execution-v1 must remap the graph onto f16 kernels when runtime requests f16 activations');
}
if (
  compiledWithRuntimeSessionDefaults.runtimeInferencePatch.kernelPath?.decode?.steps?.find((step) => step.op === 'q_proj')?.kernel
  !== 'matmul_gemv_subgroup_f16a.wgsl'
) {
  throw new Error('Execution-v1 f16 runtime path must remap decode projections onto f16 GEMV kernels');
}
if (
  compiledWithRuntimeSessionDefaults.runtimeInferencePatch.kernelPath?.decode?.steps?.find((step) => step.op === 'attention')?.kernel
  !== 'attention_decode_online_f16.wgsl'
) {
  throw new Error('Execution-v1 f16 runtime path must remap attention onto f16 kernels');
}
if (compiledWithRuntimeSessionDefaults.runtimeInferencePatch.session?.kvcache?.maxSeqLen !== 4096) {
  throw new Error('Execution-v1 session merge must still retain non-dtype runtime session fields');
}

const f32AttentionGraph = makeGraph();
f32AttentionGraph.kernels.attn = {
  kernel: 'attention_decode.wgsl',
  entry: 'main',
  digest: D('8'),
};

const explicitRuntimeOverridesWithoutSession = {
  inference: {
    compute: {
      activationDtype: 'f32',
    },
  },
};

const appliedRuntimeConfig = applyExecutionV1RuntimeConfig({
  runtimeConfig: createDopplerConfig({
    runtime: explicitRuntimeOverridesWithoutSession,
  }).runtime,
  runtimeOverrides: explicitRuntimeOverridesWithoutSession,
  manifest: {
    modelId: 'test-model-explicit-runtime-without-session',
    architecture: {
      numLayers: 26,
      headDim: 128,
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: f32AttentionGraph,
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
        },
        decodeLoop: null,
      },
    },
  },
  modelId: 'test-model-explicit-runtime-without-session',
  numLayers: 26,
  capabilities: { hasF16: true, hasSubgroups: true },
  platform: { id: 'test', vendor: 'test', architecture: 'test' },
});

if (appliedRuntimeConfig.executionV1State?.session?.kvcache?.kvDtype !== 'f32') {
  throw new Error(
    'Execution-v1 must preserve manifest session.kvcache.kvDtype when runtime overrides omit inference.session'
  );
}
if (appliedRuntimeConfig.runtimeConfig.inference?.session?.kvcache?.kvDtype !== 'f32') {
  throw new Error(
    'Execution-v1 runtime patch must not let default runtime session.kvcache override manifest session.kvcache'
  );
}

const decodeLoopOverrideRuntimeConfig = createDopplerConfig({
  runtime: {
    inference: {
      session: {
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
        },
        decodeLoop: {
          batchSize: 1,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          readbackMode: 'sequential',
          ringTokens: 1,
          ringStop: 1,
          ringStaging: 2,
          disableCommandBatching: true,
        },
      },
    },
  },
}).runtime;

const appliedDecodeLoopOverrideRuntimeConfig = applyExecutionV1RuntimeConfig({
  runtimeConfig: decodeLoopOverrideRuntimeConfig,
  runtimeOverrides: null,
  manifest: {
    modelId: 'test-model-runtime-decode-loop-preserved',
    architecture: {
      numLayers: 26,
      headDim: 128,
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: f32AttentionGraph,
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
        },
        decodeLoop: {
          batchSize: 8,
          stopCheckMode: 'batch',
          readbackInterval: 8,
          readbackMode: 'sequential',
          ringTokens: 1,
          ringStop: 1,
          ringStaging: 1,
          disableCommandBatching: false,
        },
      },
    },
  },
  modelId: 'test-model-runtime-decode-loop-preserved',
  numLayers: 26,
  capabilities: { hasF16: true, hasSubgroups: true },
  platform: { id: 'test', vendor: 'test', architecture: 'test' },
});

const appliedDecodeLoop = appliedDecodeLoopOverrideRuntimeConfig
  .runtimeConfig.inference?.session?.decodeLoop;
if (appliedDecodeLoop?.disableCommandBatching !== true) {
  throw new Error(
    'Execution-v1 runtime patch must preserve runtime session.decodeLoop.disableCommandBatching'
  );
}
if (appliedDecodeLoop?.batchSize !== 1 || appliedDecodeLoop?.readbackInterval !== 1) {
  throw new Error('Execution-v1 runtime patch must preserve runtime session.decodeLoop fields');
}

const nullRuntimeOverridesConfig = createDopplerConfig({
  runtime: {
    inference: {
      session: {
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
        },
        decodeLoop: {
          batchSize: 32,
          stopCheckMode: 'batch',
          readbackInterval: 64,
          readbackMode: 'sequential',
          ringTokens: 1,
          ringStop: 1,
          ringStaging: 1,
          disableCommandBatching: false,
        },
      },
    },
  },
}).runtime;

const appliedNullRuntimeOverridesConfig = applyExecutionV1RuntimeConfig({
  runtimeConfig: nullRuntimeOverridesConfig,
  runtimeOverrides: null,
  manifest: {
    modelId: 'test-model-null-runtime-overrides',
    architecture: {
      numLayers: 26,
      headDim: 128,
    },
    inference: {
      schema: EXECUTION_V1_SCHEMA_ID,
      execution: f32AttentionGraph,
      session: {
        compute: {
          defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
        },
        kvcache: {
          kvDtype: 'f32',
          layout: 'contiguous',
        },
        decodeLoop: {
          batchSize: 12,
          stopCheckMode: 'batch',
          readbackInterval: 32,
          readbackMode: 'sequential',
        },
      },
    },
  },
  modelId: 'test-model-null-runtime-overrides',
  numLayers: 26,
  capabilities: { hasF16: true, hasSubgroups: true },
  platform: { id: 'test', vendor: 'test', architecture: 'test' },
});

if (appliedNullRuntimeOverridesConfig.executionV1State?.session?.decodeLoop?.batchSize !== 32) {
  throw new Error(
    'Execution-v1 must treat runtimeOverrides=null as absent and keep resolved runtime session decodeLoop overrides'
  );
}
if (appliedNullRuntimeOverridesConfig.executionV1State?.session?.decodeLoop?.readbackInterval !== 64) {
  throw new Error(
    'Execution-v1 must not replace resolved runtime decodeLoop with manifest decodeLoop when runtimeOverrides is null'
  );
}

const compiledPipeline = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: {
      kernels: {
        attn: {
          kernel: 'attention_streaming_f16kv.wgsl',
          entry: 'main',
          digest: D('9'),
          precision: { kvDtype: 'f16' },
        },
      },
      preLayer: [],
      decode: [['attention', 'attn']],
      prefill: [['attention', 'attn']],
      postLayer: [],
      policies: { ...DEFAULT_EXECUTION_V1_POLICIES },
    },
    session: {
      compute: {
        defaults: { activationDtype: 'f32', mathDtype: 'f32', accumDtype: 'f32', outputDtype: 'f32' },
      },
      kvcache: { kvDtype: 'f16' },
      decodeLoop: null,
    },
  },
  modelId: 'test-pipeline-precision',
  numLayers: 2,
});

if (compiledPipeline.runtimeInferencePatch?.pipeline?.steps?.find((step) => step.op === 'attention')?.kvDtype !== 'f16') {
  throw new Error('Expected layer pipeline attention step to preserve kvDtype precision');
}

// === Real config file ===
const realConfig = JSON.parse(readFileSync('src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json', 'utf8'));
const realExpanded = expandExecutionV1(realConfig.execution);
if (realExpanded.length !== 35) throw new Error(`Real config: expected 35 steps, got ${realExpanded.length}`);

console.log('execution-v1-schema.test: ok');
