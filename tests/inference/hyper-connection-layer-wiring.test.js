import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';

// Boundary test: real layer adapter, substituted arithmetic. Numerical operator
// proof is separate (the retained upstream composition and physical logits test).
const target = new URL('../../src/inference/pipelines/text/hyper-connection/layer-executor.js', import.meta.url).href;
const events = [];
const buffer = label => ({ __dopplerFakeGPUBuffer: true, size: 64, label, destroy() {} });
const tensor = label => ({ buffer: buffer(label), shape: [1, 4], dtype: 'f32' });
globalThis.hyperWiring = { events, tensor, residual: false };
const mocks = {
  './decoder-layer.js': `const p = globalThis.hyperWiring;
    export const replicateEmbeddings = async (_target, input, options) => { p.events.push('replicate'); return { ...input, shape: [1, options.hyper.streamCount * options.hyper.streamWidth] }; };
    export const runHyperConnectionDecoderLayer = async (_target, call) => {
      if (call.ple) await call.ple(call.streams);
      await call.attentionBranch(p.tensor('attention input'));
      await call.mlpBranch(p.tensor('mlp input'));
      return p.tensor('layer result');
    };`,
  '../ops.js': `const p = globalThis.hyperWiring;
    export const releaseOrTrack = (_recorder, value) => { p.events.push('release:' + value.label); };
    export const doAttention = async (_input, weights, config, state) => {
      p.attention = { weights, config, state }; p.events.push('attention');
      return { output: p.tensor('attention output'), residualFused: p.residual };
    };`,
  '../linear-attention.js': `const p = globalThis.hyperWiring;
    export const runLinearAttentionLayer = async (_input, weights, options) => {
      p.linear = { weights, options }; p.events.push('linear'); return p.tensor('linear output');
    };`,
  '../ffn/moe.js': `export const runMoEFFNGPU = async (_layer, input) => { globalThis.hyperWiring.events.push('moe'); return input; };`,
  '../probes.js': 'export const runProbes = async () => {};',
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  const source = context.parentURL === target ? mocks[specifier] : null;
  return source ? { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true } : next(specifier, context);
} });
try {
  const { processHyperConnectionLayer } = await import(target);
  const hyper = { streamCount: 4, streamWidth: 4, normWeightOffset: true, projectionRank: 2, normEps: 1e-6 };
  const weights = { inputNorm: buffer('ordinary norm'), hyperConnections: { attention: {}, mlp: {} }, qsaIndexer: {} };
  const config = { rmsNormWeightOffset: true, hyperConnections: hyper, hiddenSize: 4, numLayers: 4, layerTypes: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
    numHeads: 2, numKVHeads: 1, headDim: 2, rmsNormEps: 1e-6, slidingWindow: null,
    attnLogitSoftcapping: null, queryPreAttnScalar: null, queryScale: 1, queryKeyNorm: true,
    queryKeyNormType: 'rmsnorm', queryKeyNormAxis: 'head', queryKeyNormWeightLayers: null, valueNorm: false,
    attentionOutputGate: true, outputGateType: 'sigmoid', causalAttention: true, rmsNormWeightOffset: true,
    normalizationType: 'rmsnorm', ropeRotaryDim: 2, ropeFrequencyBaseDim: 2, ropeInterleaved: false,
    sessionSettings: { intermediatePrecision: 'f32' } };
  const session = {
    prepareQsa: async (layerIdx, _input, _weights, context) => {
      events.push('indexer'); return { layerIdx, numTokens: 1, visibleRows: context.currentSeqLen + 1, selection: buffer('selection') };
    },
    releaseQsa: () => events.push('release selection'),
    runPle: async (_index, input) => { events.push('ple'); return input; },
  };
  const context = { config, hyperConnectionSession: session, activationDtype: 'f32', weights: new Map([['layer_0', weights], ['layer_1', { ...weights, ple: {} }], ['layer_3', weights]]),
    phase: 'prefill', currentSeqLen: 3000, currentTokenIds: [7], kvCache: {},
    kernelPath: { id: 'wiring' }, ropeFreqsCos: buffer('cos'), ropeFreqsSin: buffer('sin'),
    resolvedRuntimeSession: { runtime: { session: {} } }, weightConfig: {} };
  await processHyperConnectionLayer(3, buffer('existing streams'), 1, true, context);
  assert.deepEqual(events, ['indexer', 'attention', 'release selection', 'moe']);
  assert.equal(globalThis.hyperWiring.attention.config.skipInputNorm, true);
  assert.equal(globalThis.hyperWiring.attention.config.residualTensor, null);
  assert.equal(globalThis.hyperWiring.attention.state.qsa.visibleRows, 3001, 'decode-style query count does not replace visible KV count');
  events.length = 0;
  await processHyperConnectionLayer(0, buffer('embeddings'), 1, true, context);
  assert.deepEqual(events, ['replicate', 'linear', 'moe', 'release:embeddings']);
  assert.equal(globalThis.hyperWiring.linear.weights.inputNorm, null);
  assert.equal(weights.inputNorm.label, 'ordinary norm', 'branch normalization overrides do not mutate loaded weights');
  assert.equal(globalThis.hyperWiring.linear.options.executionPhase, 'prefill', 'one-token prefill retains its phase');
  events.length = 0;
  await processHyperConnectionLayer(1, buffer('streams'), 1, true, context);
  assert.deepEqual(events, ['ple', 'linear', 'moe']);
  events.length = 0; globalThis.hyperWiring.residual = true;
  await assert.rejects(processHyperConnectionLayer(3, buffer('streams'), 1, true, context), /ordinary residual/);
  assert.ok(events.includes('release selection')); assert.ok(!events.includes('moe'));
} finally { hooks.deregister(); delete globalThis.hyperWiring; }
console.log('hyper-connection layer wiring: branch ownership, PLE order, selection and explicit phase passed');
