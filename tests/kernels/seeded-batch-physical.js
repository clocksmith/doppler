/** Real batch/sampling/cleanup owners; fixed GPU logits replace model computation. */
export async function auditSeededBatches() {
  const { initDevice, getKernelCapabilities, destroyDevice } = await import('/src/gpu/device.js');
  const { acquireBuffer, getBufferPool, destroyBufferPool } = await import('/src/memory/buffer-pool.js');
  const { getRuntimeConfig } = await import('/src/config/runtime.js');
  const { generateNTokensGPU } = await import('/src/inference/pipelines/text/generator-steps.js');
  const { PipelineGenerator } = await import('/src/inference/pipelines/text/generator.js');
  const device = await initDevice();
  const adapter = getKernelCapabilities().adapterInfo;
  if (/software|swiftshader|llvmpipe/i.test(JSON.stringify(adapter))) throw new Error('Physical adapter required');
  const errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  const values = new Float32Array(512).fill(-100); values.set([3, 2, 1, 0]);
  const createInput = (data, label) => {
    const buffer = acquireBuffer(data.byteLength, undefined, label);
    device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  globalThis.__samplingFixture = {
    embed: () => ({ buffer: createInput(new Float32Array([1]), 'fixture_hidden') }),
    logits: () => ({ logitsBuffer: createInput(values, 'fixture_logits'), vocabSize: 512, logitsDtype: 'f32' }),
  };
  const embed = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE });
  const rows = [];
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  const equal = (left, right, message) => check(JSON.stringify(left) === JSON.stringify(right), message);
  function expected(seed, index) {
    // Independent scalar oracle: freeze the documented counter and inverse CDF.
    const drawSeed = index === 0 ? seed : ((seed % 4294967296) + (index % 4294967296)) % 4294967296;
    const random = Math.sin(drawSeed) * 10000;
    const uniform = Math.fround(random - Math.floor(random));
    const weights = [0, -1, -2, -3].map(Math.exp);
    const total = weights.reduce((a, b) => a + b, 0);
    let cumulative = 0;
    for (let token = 0; token < 4; token++) {
      cumulative += weights[token] / total;
      if (uniform < cumulative) return token;
    }
    return 3;
  }
  function state() {
    const runtimeConfig = structuredClone(getRuntimeConfig());
    runtimeConfig.inference.compute.activationDtype = 'f32';
    runtimeConfig.inference.session.useGreedyLmHeadArgmaxFusion = false;
    return { modelConfig: { layerTypes: [], numLayers: 0, hiddenSize: 1, vocabSize: 512,
      stopTokenIds: [511], finalLogitSoftcapping: null }, runtimeConfig,
      tokenizer: { getSpecialTokens: () => ({ eos: 511, pad: 510 }) },
      weights: new Map([['embed', embed]]), useGPU: true, currentSeqLen: 4, decodeStepCount: 0,
      batchingStats: { batchedForwardCalls: 0, totalBatchedTimeMs: 0, gpuSubmissions: 0 }, stats: {} };
  }
  const helpers = { buildLayerContext: () => ({ config: { hiddenSizePerLayerInput: 0 } }), getLogitsWeights: () => ({}), getLogitsConfig: () => ({}) };
  const executionPlan = Object.freeze({ activationDtype: 'f32', batchSize: 4, readbackInterval: 2,
    stopCheckMode: 'batch', disableCommandBatching: false, disableMultiTokenDecode: false,
    maxBatchDecodeTokens: null, maxTokens: 16 });
  const options = seed => Object.freeze({ seed, promptTokenCount: 2, temperature: 1, topK: 4, topP: 1,
    maxTokens: 16, repetitionPenalty: 1, repetitionPenaltyWindow: 0, presencePenalty: 0,
    stopSequences: [], suppressTokenIds: [], executionPlan });
  const settle = async () => {
    await device.queue.onSubmittedWorkDone();
    // Pool retirement schedules completion per captured retirement wave. A later
    // wave is owned by its own queue completion, not the first promise above.
    let resources = getBufferPool().getStats().resources;
    for (let wave = 0; resources.deferredCleanup.bytes > 0 && wave < 32; wave++) {
      await device.queue.onSubmittedWorkDone();
      resources = getBufferPool().getStats().resources;
    }
    check(resources.active.bytes === 0 && resources.deferredCleanup.bytes === 0, 'Owned transient buffers must settle: ' + JSON.stringify(resources));
    return resources;
  };
  try {
    for (const seed of [0, 17, 0.25, 4294967295, Number.MAX_VALUE]) {
      const ids = [1, 2, 3, 4, 5], opts = options(seed);
      const oracle = Array.from({ length: 7 }, (_, i) => expected(seed, i + 3));
      const whole = await generateNTokensGPU(state(), ids.at(-1), 7, ids, opts, helpers);
      equal(Array.from(whole.tokens), oracle, 'Whole GPU batch differs from independent draws');
      const split = state();
      const a = await generateNTokensGPU(split, ids.at(-1), 4, ids, opts, helpers);
      const nextIds = [...ids, ...a.tokens];
      const b = await generateNTokensGPU(split, nextIds.at(-1), 3, nextIds, opts, helpers);
      equal([...a.tokens, ...b.tokens], oracle, 'Split GPU batches changed draws');
      const retry = await generateNTokensGPU(state(), ids.at(-1), 7, ids, opts, helpers);
      equal(Array.from(retry.tokens), oracle, 'Retry changed indexed GPU draws');
      check(opts.seed === seed, 'Caller generation seed mutated');
      rows.push({ seed, expected: oracle, whole: Array.from(whole.tokens), split: [...a.tokens, ...b.tokens],
        retry: Array.from(retry.tokens), resources: await settle() });
    }
    const makeGenerator = () => {
      const pipeline = Object.create(PipelineGenerator.prototype); pipeline._state = state();
      pipeline._getDecodeHelpers = () => helpers;
      pipeline._hasFinitenessFallbackWindow = () => false;
      pipeline._consumeFinitenessFallbackToken = () => {};
      pipeline._recordStopReason = reason => { pipeline._state.stats.stopReason = reason; };
      return pipeline;
    };
    const abort = new AbortController(), pipeline = makeGenerator();
    const opts = { ...options(17), maxTokens: 9 };
    const ids = [1, 2, expected(17, 0)];
    const stream = pipeline._runDecodeLoop(ids, opts, { signal: abort.signal }, {
      stopTokenIds: [511], eosToken: 511, stopSequenceStart: 2, decodeToken: String, emitMode: 'token' });
    const first = await stream.next(); check(!first.done, 'Batch must deliver before cancellation');
    abort.abort();
    const afterAbort = await stream.next();
    rows.push({ cancellation: { first, afterAbort, stopReason: pipeline._state.stats.stopReason,
      resources: await settle(), generatedIds: ids } });
    await stream.return();
    check(afterAbort.done, 'Cancellation after one delivered batch token must suppress remaining output');
    const fresh = makeGenerator(), freshIds = [1, 2, expected(17, 0)];
    const tokens = await Array.fromAsync(fresh._runDecodeLoop(freshIds, opts, {}, {
      stopTokenIds: [511], eosToken: 511, stopSequenceStart: 2, decodeToken: String, emitMode: 'token' }));
    equal(tokens, Array.from({ length: 8 }, (_, i) => expected(17, i + 1)), 'New attempt after cancellation failed');
    rows.push({ freshAttempt: tokens, resources: await settle() });
    for (const boundary of ['onToken', 'onBatch']) {
      const failed = makeGenerator(), failedIds = [1, 2, expected(17, 0)];
      const sentinel = new Error('consumer:' + boundary);
      let observed;
      try {
        await Array.fromAsync(failed._runDecodeLoop(failedIds, opts,
          { [boundary]: () => { throw sentinel; } }, {
            stopTokenIds: [511], eosToken: 511, stopSequenceStart: 2, decodeToken: String, emitMode: 'token' }));
      } catch (error) { observed = error; }
      check(observed === sentinel, 'Consumer failure must propagate without GPU fallback');
      rows.push({ deliveryFailure: boundary, resources: await settle() });
    }
    equal(errors, [], 'Uncaptured WebGPU errors');
    return { qualified: true, scope: 'Physical sampler and batch lifecycle with fixed logits; no model or release qualification', adapter, rows, errors };
  } catch (error) {
    return { qualified: false, adapter, rows, errors, failure: error.stack ?? error.message };
  } finally { embed.destroy(); destroyBufferPool(); destroyDevice(); delete globalThis.__samplingFixture; }
}
