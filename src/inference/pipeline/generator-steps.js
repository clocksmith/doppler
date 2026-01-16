import { getDevice, setTrackSubmits } from '../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runArgmax, runGPUSample, recordArgmax, recordGPUSample, isGPUSamplingAvailable } from '../../gpu/kernels/sample.js';
import { recordCheckStop } from '../../gpu/kernels/check-stop.js';
import { resetSubmitStats, logSubmitStats } from '../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../gpu/command-recorder.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { getUniformCache } from '../../gpu/uniform-cache.js';
import { log } from '../../debug/index.js';
import { getRuntimeConfig } from '../../config/runtime.js';

import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, applySoftcapping } from './logits.js';
import { isWeightBuffer, isCpuWeightBuffer, getWeightDtype } from '../../gpu/weight-buffer.js';
import { decodeReadback } from './debug-utils.js';

export function sumProfileTimings(timings) {
  if (!timings || Object.keys(timings).length === 0) return null;
  let total = 0;
  for (const value of Object.values(timings)) {
    if (Number.isFinite(value)) {
      total += value;
    }
  }
  return total;
}

export async function decodeStep(state, currentIds, opts, helpers) {
  const lastToken = currentIds[currentIds.length - 1];
  const numTokens = 1;
  const config = state.modelConfig;
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  state.decodeStepCount++;
  const isDebugStep = opts.debug && state.decodeStepCount <= 5;
  if (isDebugStep) {
    const tokenText = state.tokenizer?.decode?.([lastToken]) || '?';
    log.debug('Decode', `[${state.decodeStepCount}] token="${tokenText}" pos=${state.currentSeqLen}`);
  }

  const device = getDevice();

  let recorder;
  if (device && !opts.debug && !opts.disableCommandBatching) {
    recorder = opts.profile
      ? createProfilingRecorder('decode')
      : createCommandRecorder('decode');
  }
  if (state.decodeStepCount === 1) {
    const path = recorder ? 'fused' : 'debug-sync';
    log.debug('Decode', `Using ${path} path (recorder=${!!recorder}, debug=${opts.debug})`);
  }
  const context = helpers.buildLayerContext(recorder, true, opts.debugLayers);

  state.decodeBuffers.resetPingPong();

  const decodeHiddenBuffer = state.decodeBuffers.getHiddenBuffer();
  const decodeAltBuffer = state.decodeBuffers.getOutputHiddenBuffer();

  const embedBufferRaw = state.weights.get('embed');
  if (!(embedBufferRaw instanceof GPUBuffer) && !isWeightBuffer(embedBufferRaw) && !isCpuWeightBuffer(embedBufferRaw) && !(embedBufferRaw instanceof Float32Array)) {
    throw new Error('Embed buffer not found or not a supported buffer type');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isWeightBuffer(embedBufferRaw)
    ? getWeightDtype(embedBufferRaw)
    : isCpuWeightBuffer(embedBufferRaw)
      ? embedBufferRaw.dtype
      : null;
  const activationDtype = state.runtimeConfig.inference.compute.activationDtype;
  const activationBytes = activationDtype === 'f16' ? 2 : 4;

  const embedTensor = await embed([lastToken], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    recorder,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    activationDtype,
    embeddingDtype: embedDtype === 'f16' ? 'f16' : 'f32',
  });

  let hiddenStates = embedTensor.buffer;

  if (opts.debug && state.decodeStepCount === 1) {
    const validSize = config.hiddenSize * activationBytes;
    const embedData = await readBuffer(hiddenStates, validSize);
    const embedArr = decodeReadback(embedData, activationDtype);
    const sample = embedArr.slice(0, 5);
    const maxAbs = Math.max(...embedArr.map(Math.abs));
    const nonZero = embedArr.filter(x => Math.abs(x) > 1e-10).length;
    log.debug('Decode', `[1] Embed check: maxAbs=${maxAbs.toFixed(2)}, nonZero=${nonZero}/${embedArr.length}, sample=[${Array.from(sample).map(v => v.toFixed(3)).join(', ')}]`);
  }

  const benchmarkSubmits = state.decodeStepCount <= 3 && opts.debug;
  if (benchmarkSubmits) {
    setTrackSubmits(true);
    resetSubmitStats();
  }

  const hasGPUCache = context.kvCache?.hasGPUCache?.() ?? false;
  if (opts.debug && state.decodeStepCount === 1) {
    log.debug('Decode', `KV cache check: hasGPUCache=${hasGPUCache}, currentSeqLen=${context.currentSeqLen}`);
  }

  for (let l = 0; l < config.numLayers; l++) {
    const prevStates = hiddenStates;
    hiddenStates =  (await processLayer(l, hiddenStates, numTokens, false, context));

    state.decodeBuffers.swapPingPong();

    if (prevStates instanceof GPUBuffer && prevStates !== hiddenStates) {
      const isPreAllocated = prevStates === decodeHiddenBuffer || prevStates === decodeAltBuffer;
      if (!isPreAllocated) {
        if (recorder) {
          recorder.trackTemporaryBuffer(prevStates);
        } else {
          releaseBuffer(prevStates);
        }
      }
    }
  }

  const logitSoftcap = config.finalLogitSoftcapping ?? 0;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad;
  const lmHeadIsCpu = isCpuWeightBuffer(state.weights.get('lm_head'));
  const useGPUSampling = state.useGPU && isGPUSamplingAvailable() && !lmHeadIsCpu;
  const useFusedDecode = recorder && useGPUSampling && !state.disableFusedDecode;

  if (useFusedDecode) {
    const { logitsBuffer, vocabSize, logitsDtype } = await recordLogitsGPU(
      recorder,
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
    );

    const sampleOutputBuffer = opts.temperature < samplingDefaults.greedyThreshold
      ? await recordArgmax(recorder, logitsBuffer, vocabSize, { padTokenId, logitSoftcap, logitsDtype })
      : await recordGPUSample(recorder, logitsBuffer, vocabSize, {
          temperature: opts.temperature,
          topK: opts.topK,
          padTokenId,
          logitSoftcap,
          logitsDtype,
        });

    const isPreAllocated = hiddenStates === decodeHiddenBuffer || hiddenStates === decodeAltBuffer;

    recorder.submit();

    if (!allowReadback('pipeline.decode.sample')) {
      throw new Error('[Pipeline] GPU readback disabled for sampling');
    }

    const stagingBuffer = device.createBuffer({
      label: 'sample_staging',
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const copyEncoder = device.createCommandEncoder({ label: 'sample_copy' });
    copyEncoder.copyBufferToBuffer(sampleOutputBuffer, 0, stagingBuffer, 0, 4);
    device.queue.submit([copyEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const nextToken = new Uint32Array(stagingBuffer.getMappedRange())[0];
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    log.debug('Decode', `Step ${state.decodeStepCount}: token=${nextToken} (vocabSize=${config.vocabSize})`);

    const invalidToken = nextToken >= config.vocabSize
      || (padTokenId !== undefined && nextToken === padTokenId)
      || (padTokenId === undefined && nextToken === 0);
    if (invalidToken) {
      log.warn('Decode', `Suspicious token ${nextToken} (vocabSize=${config.vocabSize}, step=${state.decodeStepCount})`);
      if (allowReadback('pipeline.decode.debug-logits')) {
        try {
          const logitsBytes = logitsDtype === 'f16' ? 2 : 4;
          const logitSample = await readBuffer(logitsBuffer, Math.min(config.vocabSize * logitsBytes, 4096));
          const logitArr = decodeReadback(logitSample, logitsDtype);
          const maxLogit = Math.max(...logitArr);
          const minLogit = Math.min(...logitArr);
          const hasNaN = logitArr.some((v) => Number.isNaN(v));
          const hasInf = logitArr.some((v) => !Number.isFinite(v));
          let argmaxIdx = 0;
          let argmaxVal = logitArr[0];
          for (let i = 1; i < logitArr.length; i++) {
            if (logitArr[i] > argmaxVal) {
              argmaxVal = logitArr[i];
              argmaxIdx = i;
            }
          }
          log.warn('Decode', `Logits: max=${maxLogit.toFixed(4)} at [${argmaxIdx}], min=${minLogit.toFixed(4)}, hasNaN=${hasNaN}, hasInf=${hasInf}`);
          log.warn('Decode', `First 10 logits: ${Array.from(logitArr.slice(0, 10)).map((v) => v.toFixed(4)).join(', ')}`);
          log.warn('Decode', `Logit[0] (pad): ${logitArr[0].toFixed(4)}, Logit[${argmaxIdx}]: ${argmaxVal.toFixed(4)}`);
        } catch (e) {
          log.warn('Decode', `Failed to read logits: ${ (e).message}`);
        }
      }
    }

    releaseBuffer(logitsBuffer);
    releaseBuffer(sampleOutputBuffer);

    if (benchmarkSubmits) {
      logSubmitStats(`Decode step ${state.decodeStepCount} (${config.numLayers} layers, fused)`);
      setTrackSubmits(false);
    }

    if (opts.profile && recorder.isProfilingEnabled()) {
      const timings = await recorder.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
      }
      if (timings) {
        log.warn('Profile', `Decode step ${state.decodeStepCount}:`);
        log.warn('Profile', CommandRecorder.formatProfileReport(timings));
      }
    }

    if (invalidToken) {
      state.disableFusedDecode = true;
      log.warn('Decode', 'Fused sampling produced invalid token; falling back to CPU sampling.');
      const fallbackLogits = await computeLogits(
        hiddenStates,
        numTokens,
        helpers.getLogitsWeights(),
        helpers.getLogitsConfig(),
        state.useGPU,
        state.debugFlags,
        undefined,
        debugCheckBuffer,
        state.runtimeConfig.shared.debug.probes
      );
      applyRepetitionPenalty(fallbackLogits, currentIds, opts.repetitionPenalty);
      const fallbackToken = sample(fallbackLogits, {
        temperature: opts.temperature,
        topP: opts.topP,
        topK: opts.topK,
        padTokenId,
      });
      if (!isPreAllocated) {
        releaseBuffer(hiddenStates);
      }
      state.currentSeqLen++;
      return fallbackToken;
    }

    if (!isPreAllocated) {
      releaseBuffer(hiddenStates);
    }

    state.currentSeqLen++;
    return nextToken;
  }

  if (recorder) {
    await recorder.submitAndWait();

    if (opts.profile && recorder.isProfilingEnabled()) {
      const timings = await recorder.resolveProfileTimings();
      const total = sumProfileTimings(timings);
      if (total !== null) {
        state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
      }
      if (timings) {
        log.warn('Profile', `Decode step ${state.decodeStepCount} (layers only):`);
        log.warn('Profile', CommandRecorder.formatProfileReport(timings));
      }
    }
  }

  if (benchmarkSubmits) {
    logSubmitStats(`Decode step ${state.decodeStepCount} (${config.numLayers} layers)`);
    setTrackSubmits(false);
  }

  if (opts.debug && state.decodeStepCount === 1 && hiddenStates instanceof GPUBuffer) {
    const debugDevice = getDevice();
    if (debugDevice) {
      if (allowReadback('pipeline.decode.debug-hidden')) {
        const debugReadbackSize = state.runtimeConfig.shared.debug.pipeline.readbackSampleSize;
        const sampleSize = Math.min(debugReadbackSize, hiddenStates.size);
        const staging = debugDevice.createBuffer({
          size: sampleSize,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const enc = debugDevice.createCommandEncoder();
        enc.copyBufferToBuffer(hiddenStates, 0, staging, 0, sampleSize);
        debugDevice.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        const nanCount = Array.from(data).filter(x => !Number.isFinite(x)).length;
        const nonZero = Array.from(data).filter(x => Number.isFinite(x) && x !== 0).slice(0, 5);
        log.debug('Decode', `[1] HIDDEN_AFTER_LAYERS: nan=${nanCount}/${data.length}, nonZero=${nonZero.length}, sample=[${nonZero.map(x => x.toFixed(4)).join(', ')}]`);
      }
    }
  }

  if (useGPUSampling) {
    const logitsResult = await computeLogitsGPU(
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.debugFlags
    );
    if (logitsResult) {
      const { logitsBuffer, vocabSize, logitsDtype } = logitsResult;

      const nextToken = opts.temperature < samplingDefaults.greedyThreshold
        ? await runArgmax(logitsBuffer, vocabSize, { padTokenId, logitSoftcap, logitsDtype })
        : await runGPUSample(logitsBuffer, vocabSize, {
            temperature: opts.temperature,
            topK: opts.topK,
            padTokenId,
            logitSoftcap,
            logitsDtype,
          });

      releaseBuffer(logitsBuffer);
      if (!context.decodeBuffers?.ownsBuffer(hiddenStates)) {
        releaseBuffer(hiddenStates);
      }
      state.currentSeqLen++;
      return nextToken;
    }
  }

  const logits = await computeLogits(
    hiddenStates,
    numTokens,
    helpers.getLogitsWeights(),
    helpers.getLogitsConfig(),
    state.useGPU,
    state.debugFlags,
    undefined,
    debugCheckBuffer,
    state.runtimeConfig.shared.debug.probes
  );

  if (!context.decodeBuffers?.ownsBuffer(hiddenStates)) {
    releaseBuffer(hiddenStates);
  }

  if (isDebugStep) {
    logitsSanity(logits, `Decode[${state.decodeStepCount}]`, opts.decode);
  }

  applyRepetitionPenalty(logits, currentIds, opts.repetitionPenalty);
  const nextToken = sample(logits, {
    temperature: opts.temperature,
    topP: opts.topP,
    topK: opts.topK,
    padTokenId,
  });

  state.currentSeqLen++;
  return nextToken;
}

export async function generateNTokensGPU(state, startToken, N, currentIds, opts, helpers) {
  const device = getDevice();
  const config = state.modelConfig;
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const recorder = opts.profile
    ? createProfilingRecorder('batch_decode')
    : createCommandRecorder('batch_decode');
  const lmHead = state.weights.get('lm_head');
  if (lmHead && isCpuWeightBuffer(lmHead)) {
    throw new Error('[Pipeline] GPU-only decode not supported with CPU-resident LM head.');
  }

  const stopCheckMode = opts.stopCheckMode ?? 'per-token';

  const stopBufferSize = stopCheckMode === 'per-token' ? N * 4 : 0;
  const stopBuffer = stopCheckMode === 'per-token'
    ? device.createBuffer({
      size: stopBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    })
    : null;

  const stopTokenIds = config.stopTokenIds || [];
  const eosToken = state.tokenizer?.getSpecialTokens?.()?.eos;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad;
  const logitSoftcap = config.finalLogitSoftcapping ?? 0;
  const eosTokenId = eosToken ?? stopTokenIds[0] ?? 1;
  const maxTokens = opts.maxTokens || getRuntimeConfig().inference.batching.maxTokens;

  const tokensBuffer = device.createBuffer({
    size: (N + 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(tokensBuffer, 0, new Uint32Array([startToken]));

  const context = helpers.buildLayerContext(recorder, true, opts.debugLayers);
  const embedBufferRaw = state.weights.get('embed');
  if (isCpuWeightBuffer(embedBufferRaw)) {
    throw new Error('[Pipeline] GPU-only decode not supported with CPU-resident embeddings.');
  }
  if (!(embedBufferRaw instanceof GPUBuffer) && !isWeightBuffer(embedBufferRaw)) {
    throw new Error('Embed buffer not found or not a GPUBuffer/WeightBuffer');
  }
  const embedBuffer = isWeightBuffer(embedBufferRaw) ? embedBufferRaw.buffer : embedBufferRaw;
  const embedDtype = isWeightBuffer(embedBufferRaw) ? getWeightDtype(embedBufferRaw) : null;
  const activationDtype = state.runtimeConfig.inference.compute.activationDtype;

  for (let i = 0; i < N; i++) {
    const currentPos = state.currentSeqLen + i;
    context.currentSeqLen = currentPos;
    context.decodeBuffers?.resetPingPong();

    const hiddenTensor = await embed(tokensBuffer, embedBuffer, {
      hiddenSize: config.hiddenSize,
      vocabSize: config.vocabSize,
      scaleEmbeddings: config.scaleEmbeddings,
      recorder,
      transpose: state.embeddingTranspose,
      debugProbes: state.runtimeConfig.shared.debug.probes,
      activationDtype,
      embeddingDtype: embedDtype === 'f16' ? 'f16' : 'f32',
      numTokens: 1,
      indexOffset: i,
    });

    let hiddenStatesBuffer = hiddenTensor.buffer;
    for (let l = 0; l < config.numLayers; l++) {
      const prevStates = hiddenStatesBuffer;
      hiddenStatesBuffer = (await processLayer(l, hiddenStatesBuffer, 1, false, context));
      context.decodeBuffers?.swapPingPong();
      if (prevStates instanceof GPUBuffer && prevStates !== hiddenStatesBuffer) {
        const ownsBuffer = context.decodeBuffers?.ownsBuffer(prevStates);
        if (!ownsBuffer) {
          recorder.trackTemporaryBuffer(prevStates);
        }
      }
    }

    const logits = await recordLogitsGPU(
      recorder,
      hiddenStatesBuffer,
      1,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig()
    );
    const { logitsBuffer, vocabSize, logitsDtype } = logits;

    const nextToken = opts.temperature < samplingDefaults.greedyThreshold
      ? await recordArgmax(recorder, logitsBuffer, vocabSize, { padTokenId, logitSoftcap, logitsDtype })
      : await recordGPUSample(recorder, logitsBuffer, vocabSize, {
          temperature: opts.temperature,
          topK: opts.topK,
          padTokenId,
          logitSoftcap,
          logitsDtype,
        });

    const stopCheck = stopCheckMode === 'per-token'
      ? await recordCheckStop(recorder, nextToken, stopBuffer, i, { eosTokenId })
      : null;

    if (hiddenStatesBuffer instanceof GPUBuffer && !context.decodeBuffers?.ownsBuffer(hiddenStatesBuffer)) {
      recorder.trackTemporaryBuffer(hiddenStatesBuffer);
    }
    if (logitsBuffer instanceof GPUBuffer) {
      recorder.trackTemporaryBuffer(logitsBuffer);
    }
    if (nextToken instanceof GPUBuffer) {
      recorder.trackTemporaryBuffer(nextToken);
    }
    if (stopCheck instanceof GPUBuffer) {
      recorder.trackTemporaryBuffer(stopCheck);
    }
  }

  recorder.submit();

  if (!allowReadback('pipeline.decode.sample')) {
    throw new Error('[Pipeline] GPU readback disabled for sampling');
  }

  const tokensStagingBuffer = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const copyEncoder = device.createCommandEncoder({ label: 'tokens_copy' });
  copyEncoder.copyBufferToBuffer(tokensBuffer, 4, tokensStagingBuffer, 0, N * 4);

  let stopStagingBuffer = null;
  if (stopCheckMode === 'per-token' && stopBuffer) {
    stopStagingBuffer = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    copyEncoder.copyBufferToBuffer(stopBuffer, 0, stopStagingBuffer, 0, N * 4);
  }
  device.queue.submit([copyEncoder.finish()]);

  const mapPromises = [tokensStagingBuffer.mapAsync(GPUMapMode.READ)];
  if (stopStagingBuffer) {
    mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
  }
  await Promise.all(mapPromises);

  getUniformCache().flushPendingDestruction();

  const tokensView = new Uint32Array(tokensStagingBuffer.getMappedRange());
  const tokens = Array.from(tokensView);

  let actualCount = N;
  if (stopCheckMode === 'per-token' && stopStagingBuffer) {
    const stopFlags = new Uint32Array(stopStagingBuffer.getMappedRange().slice(0));
    log.debug('Pipeline', `[STOP] N=${N} flags=[${Array.from(stopFlags).join(',')}] tokens=[${tokens.join(',')}] eos=${eosTokenId}`);
    for (let i = 0; i < N; i++) {
      if (stopFlags[i] === 1) {
        actualCount = i + 1;
        break;
      }
    }
    stopStagingBuffer.unmap();
  } else {
    for (let i = 0; i < N; i++) {
      if (isStopToken(tokens[i], stopTokenIds, eosToken)) {
        actualCount = i + 1;
        break;
      }
    }
  }

  tokensStagingBuffer.unmap();

  const generatedTokens = tokens.slice(0, actualCount);

  tokensBuffer.destroy();
  stopBuffer?.destroy();
  tokensStagingBuffer.destroy();
  if (stopStagingBuffer) stopStagingBuffer.destroy();

  if (opts.profile && recorder.isProfilingEnabled()) {
    const timings = await recorder.resolveProfileTimings();
    const total = sumProfileTimings(timings);
    if (total !== null) {
      state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
    }
    if (timings) {
      log.warn('Profile', `Batch decode (N=${N}):`);
      log.warn('Profile', CommandRecorder.formatProfileReport(timings));
    }
  }

  state.currentSeqLen += actualCount;

  return { tokens: generatedTokens, actualCount };
}
