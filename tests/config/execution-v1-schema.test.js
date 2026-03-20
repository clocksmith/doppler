import {
  expandExecutionV1,
  hasExecutionV1,
  EXECUTION_V1_SCHEMA_ID,
  DEFAULT_EXECUTION_V1_POLICIES,
} from '../../src/config/schema/index.js';
import { compileExecutionV1 } from '../../src/inference/pipelines/text/execution-v1.js';
import { readFileSync } from 'fs';

const D = (c) => 'sha256:' + c.repeat(64);

function makeKernels() {
  return {
    rmsnorm: { kernel: 'rmsnorm.wgsl', entry: 'main', digest: D('a') },
    gemv: { kernel: 'matmul_gemv_subgroup.wgsl', entry: 'main_vec4', digest: D('b') },
    attn: { kernel: 'attention_decode_online_f16kv.wgsl', entry: 'main', digest: D('c') },
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
    sessionDefaults: {
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
if (!compiled.runtimeInferencePatch.kvcache?.kvDtype) {
  throw new Error('Expected kvcache.kvDtype in patch');
}

const compiledWithoutInlineKernelPath = compileExecutionV1({
  manifestInference: {
    schema: EXECUTION_V1_SCHEMA_ID,
    execution: {
      ...graph,
      inlineKernelPath: false,
    },
    sessionDefaults: {
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

// === Real config file ===
const realConfig = JSON.parse(readFileSync('src/config/conversion/gemma3/gemma-3-1b-it-q4k-ehf16-af32.json', 'utf8'));
const realExpanded = expandExecutionV1(realConfig.execution);
if (realExpanded.length !== 35) throw new Error(`Real config: expected 35 steps, got ${realExpanded.length}`);

console.log('execution-v1-schema.test: ok');
