import assert from 'node:assert/strict';
import { registerHooks } from 'node:module';
import { samplingSeedForToken } from '../../src/inference/token-sampling.js';

// Exercise both real batch control loops, replacing only GPU/storage operations.
// This checks the seed crossing the recorded-sampling boundary, not GPU math.
const owner = new URL('../../src/inference/pipelines/text/generator-steps.js', import.meta.url).href;
const previous = Object.fromEntries(['GPUBufferUsage', 'GPUMapMode', '__seedBatchFixture']
  .map(key => [key, globalThis[key]]));
class Buffer {
  constructor({ size }) { this.size = size; this.bytes = new ArrayBuffer(size); }
  async mapAsync() {}
  getMappedRange() { return this.bytes; }
  unmap() {}
  destroy() {}
}
const tokenForSeed = seed => seed === undefined ? 1 : Math.floor(seed % 7) + 1;
const fixture = {
  rangeBacked: false, seeds: [],
  device: {
    createBuffer: options => new Buffer(options),
    queue: { writeBuffer(target, offset, value) {
      new Uint8Array(target.bytes, offset, value.byteLength)
        .set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    } },
  },
  recorder: () => ({
    trackTemporaryBuffer() {}, submitPartial() {}, submit() {}, async abort() {},
    async completeDeferredCleanup() {}, getSubmitLatencyMs: () => null,
    getStats: () => ({ opCount: 0, computePassCount: 0, submissionCount: 1 }),
    getEncoder: () => ({ copyBufferToBuffer(source, from, target, to, size) {
      new Uint8Array(target.bytes, to, size).set(new Uint8Array(source.bytes, from, size));
    } }),
  }),
  sample(options) {
    this.seeds.push(options.randomSeed);
    new Uint32Array(options.outputBuffer.bytes)[options.outputIndex] = tokenForSeed(options.randomSeed);
  },
};
globalThis.GPUBufferUsage = { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };
globalThis.GPUMapMode = { READ: 1 };
globalThis.__seedBatchFixture = fixture;
const overrides = {
  '../../../gpu/device.js': 'export const getDevice = () => f.device;',
  '../../../gpu/command-recorder.js': 'export const createCommandRecorder = () => f.recorder();',
  '../../../gpu/kernels/sample.js': `export const recordGPUSample = async (r, l, v, o) => f.sample(o);
    export const recordArgmax = async (r, l, v, o) => {
      new Uint32Array(o.outputBuffer.bytes)[o.outputIndex] = 1;
    };`,
  '../../../gpu/weight-buffer.js': `export const isGpuBufferInstance = x => !!x?.bytes;
    export const isCpuWeightBuffer = () => false; export const isWeightBuffer = () => false;
    export const isSplitWeightBuffer = () => false; export const getWeightDtype = () => 'f32';
    export const getWeightMetadata = () => null;`,
  '../../../debug/perf-guards.js': 'export const allowReadback = () => true;',
  '../../../gpu/uniform-cache.js': 'export const getUniformCache = () => ({flushPendingDestruction(){}});',
  './embed.js': 'export const embed = async () => ({buffer:f.device.createBuffer({size:4})});',
  './logits/index.js': 'export const recordLogitsGPU = async () => ({logitsBuffer:{},vocabSize:10,logitsDtype:"f32"});',
  './per-layer-inputs.js': `export const hasRangeBackedPerLayerInputEmbeddings = () => f.rangeBacked;
    export const hasGpuSplitPerLayerInputEmbeddings = () => false;
    export const getPleHotVocabularyRuntime = () => null;
    export const preparePerLayerInputs = async () => null;`,
  './generator/session-context.js': 'export const resolvePerLayerInputsSession = () => null;',
  './generator/decode.js': `export const getEffectiveActivationDtype = () => 'f32';
    export const decodeStep = async (state, ids, options) => {
      f.seeds.push(options.seed); state.currentSeqLen++;
      return options.temperature === 0 ? 1 : (options.seed === undefined ? 1 : Math.floor(options.seed % 7) + 1);
    };`,
};
const hooks = registerHooks({ resolve(specifier, context, next) {
  if (context.parentURL === owner && overrides[specifier]) {
    const original = new URL(specifier, owner).href;
    const source = `export * from ${JSON.stringify(original)};
      const f = globalThis.__seedBatchFixture; ${overrides[specifier]}`;
    return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true };
  }
  return next(specifier, context);
} });
try {
  const { generateNTokensGPU } = await import(owner);
  function state() {
    return {
      modelConfig: { layerTypes: [], numLayers: 0, hiddenSize: 1, vocabSize: 10,
        stopTokenIds: [9], finalLogitSoftcapping: null },
      runtimeConfig: { shared: { debug: { probes: [] } }, inference: {
        sampling: { greedyThreshold: 0.01, repetitionPenalty: 1, presencePenalty: 0, repetitionPenaltyWindow: 0 },
        batching: { batchSize: 8, readbackInterval: 1, stopCheckMode: 'batch' }, session: {},
      } },
      tokenizer: { getSpecialTokens: () => ({ eos: 9, pad: 0 }) },
      weights: new Map([['embed', new Buffer({ size: 4 })]]),
      currentSeqLen: 4, batchingStats: { batchedForwardCalls: 0, totalBatchedTimeMs: 0, gpuSubmissions: 0 },
      stats: {},
    };
  }
  const helpers = { buildLayerContext: () => ({}), getLogitsWeights: () => ({}), getLogitsConfig: () => ({}) };
  for (const rangeBacked of [false, true]) {
    fixture.rangeBacked = rangeBacked;
    for (const seed of [undefined, 0, 17, 0.25, 4294967295, Number.MAX_VALUE]) {
      const options = Object.freeze({ seed, promptTokenCount: 2, temperature: 1,
        topK: 4, topP: 1, maxTokens: 16, repetitionPenalty: 1, presencePenalty: 0 });
      fixture.seeds = [];
      await generateNTokensGPU(state(), 2, 1, [1, 2], options, helpers);
      assert.deepEqual(fixture.seeds, [seed], 'first draw preserves the original seed');
      const ids = [1, 2, 3, 4, 5]; // Three already emitted tokens, including any forced output.
      const expectedSeeds = Array.from({ length: 7 }, (_, i) => samplingSeedForToken(seed, 3 + i));
      fixture.seeds = [];
      const whole = await generateNTokensGPU(state(), ids.at(-1), 7, ids, options, helpers);
      assert.deepEqual(fixture.seeds, expectedSeeds, `rangeBacked=${rangeBacked}, seed=${seed}`);
      assert.deepEqual(Array.from(whole.tokens), expectedSeeds.map(tokenForSeed));
      const splitState = state();
      fixture.seeds = [];
      const first = await generateNTokensGPU(splitState, ids.at(-1), 4, ids, options, helpers);
      const continuedIds = [...ids, ...first.tokens];
      const second = await generateNTokensGPU(splitState, continuedIds.at(-1), 3, continuedIds, options, helpers);
      assert.deepEqual(fixture.seeds, expectedSeeds, 'splitting recording intervals cannot reset the stream');
      assert.deepEqual([...first.tokens, ...second.tokens], Array.from(whole.tokens));
      assert.equal(options.seed, seed, 'original generation identity remains unchanged');
      fixture.seeds = [];
      const retry = await generateNTokensGPU(state(), ids.at(-1), 7, ids, options, helpers);
      assert.deepEqual(Array.from(retry.tokens), Array.from(whole.tokens), 'same attempt index reuses the draw');
      const greedy = await generateNTokensGPU(state(), ids.at(-1), 7, ids, { ...options, temperature: 0 }, helpers);
      assert.deepEqual(Array.from(greedy.tokens), Array(7).fill(1));
    }
  }
} finally {
  hooks.deregister();
  for (const [key, value] of Object.entries(previous)) {
    if (value === undefined) delete globalThis[key]; else globalThis[key] = value;
  }
}
console.log('seeded-batch-generation: recorded GPU and range-backed PLE use output-indexed draws');
