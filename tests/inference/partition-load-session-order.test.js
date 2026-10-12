import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

const target = new URL('../../src/inference/pipelines/text/lifecycle.js', import.meta.url).href;
const partition = new URL('../../src/inference/pipelines/text/partition-execution.js', import.meta.url).href;
const planModule = `
  export const compileExecutionPlanState = ({ runtimeConfig }) => ({ active: {
    id: 'test-plan', activationDtype: 'f32',
    ...(runtimeConfig.inference.session.intermediatePrecision === 'two-f32'
      ? { activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 } : {})
  } });
  export const resolveActiveExecutionPlan = state => (state.executionPlanState ?? state).active;
  export const resolvePrefillRecorderChunkLayers = () => 1;
`;
const modules = {
  './config.js': `export const parseModelConfig = manifest => manifest.testConfig;`,
  './model-load.js': `
    export const runKernelWarmup = async () => {};
    export const applyModelBatchingRuntimeDefaults = config => config;
    export const resolveKernelPathState = ({ runtimeConfig }) => ({ runtimeConfig, resolvedKernelPath: null, kernelPathSource: 'test' });
    export const initTokenizerFromManifest = async () => ({ getVocabSize: () => 32 });
    export const assertManifestComputeLaneBinding = () => {};
  `,
  './execution-v1.js': `export const hasExecutionV1 = () => false; export const applyExecutionV1RuntimeConfig = () => {};`,
  './init.js': `
    export const createKVCache = () => ({ maxSeqLen: 32 });
    export const loadWeights = () => {};
    export const initMoERouter = () => {};
    export const initSpeculativeDecoder = () => {};
    export const fuseQKVWeights = () => {};
    export const initEmulation = () => {};
    export const destroyEmulation = () => {};
  `,
  './layer-partition-contract.js': `export const resolveLayerPartition = () => ({ layerRange: [0, 0] });`,
  './execution-plan.js': planModule,
  './dtype-contract.js': `export const assertDtypeConsistency = () => {};`,
  './resolved-runtime-session.js': `export const createResolvedRuntimeSession = ({ runtimeConfig }) =>
    Object.freeze({ runtime: Object.freeze({ session: Object.freeze({ ...runtimeConfig.inference.session }) }) });`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  const source = context.parentURL === target ? modules[specifier]
    : context.parentURL === partition && specifier === './execution-plan.js' ? planModule : undefined;
  return source ? { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true }
    : next(specifier, context);
} });
const reachedWeights = new Error('test reached weight loading');
try {
  const { loadModel } = await import(target);
  for (const precision of ['f32', 'two-f32']) {
    for (const mismatch of [false, true]) {
      let weightLoads = 0;
      const planPrecision = mismatch ? (precision === 'f32' ? 'two-f32' : 'f32') : precision;
      const state = {
        stats: {}, useGPU: true, runtimeConfig: { inference: { session: { intermediatePrecision: precision } } },
        modelPartition: { plan: { activationDtype: 'f32', ...(planPrecision === 'two-f32'
          ? { activationPrecision: 'two-f32', activationComponents: 2, activationElementStrideBytes: 8 } : {}) } },
        _resolveLayerPipeline() {},
        async _loadWeights() { weightLoads++; assert.ok(Object.isFrozen(this.resolvedRuntimeSession.runtime.session)); throw reachedWeights; },
      };
      const manifest = { modelId: 'lifecycle-order-fixture', testConfig: {
        numLayers: 1, hiddenSize: 16, numHeads: 1, headDim: 16, vocabSize: 32,
        useMoE: false, numKvSharedLayers: 0, hiddenSizePerLayerInput: null,
        decodeStrategy: 'incremental', causalAttention: true, layerTypes: ['full_attention'],
      } };
      await assert.rejects(loadModel.call(state, manifest), mismatch
        ? /Partition activation precision must match/ : error => error.cause === reachedWeights);
      assert.equal(weightLoads, mismatch ? 0 : 1, 'Admission must consume resolved policy before weight loading');
    }
  }
} finally { hooks.deregister(); }
console.log('partition-load-session-order.test: scalar/paired admission and pre-weight mismatch rejection passed');
