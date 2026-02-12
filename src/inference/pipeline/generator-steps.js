import { getDevice, setTrackSubmits } from '../../gpu/device.js';
import { releaseBuffer, readBuffer } from '../../memory/buffer-pool.js';
import { runArgmax, runGPUSample, recordArgmax, recordGPUSample, isGPUSamplingAvailable } from '../../gpu/kernels/sample.js';
import { recordCheckStop } from '../../gpu/kernels/check-stop.js';
import { resetSubmitStats, logSubmitStats } from '../../gpu/submit-tracker.js';
import { createCommandRecorder, createProfilingRecorder, CommandRecorder } from '../../gpu/command-recorder.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { getUniformCache } from '../../gpu/uniform-cache.js';
import { log } from '../../debug/index.js';
import { selectRuleValue } from '../../rules/rule-registry.js';

import { sample, applyRepetitionPenalty, logitsSanity, getTopK } from './sampling.js';
import { isStopToken } from './init.js';
import { embed } from './embed.js';
import { processLayer } from './layer.js';
import { computeLogits, computeLogitsGPU, recordLogitsGPU, extractLastPositionLogits, finalizeLogits, applySoftcapping } from './logits.js';
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

function shouldLogProfileStep(state, step) {
  const profilerConfig = state.runtimeConfig?.shared?.debug?.profiler;
  const every = profilerConfig?.logEveryDecodeSteps ?? 1;
  if (!Number.isFinite(every) || every <= 1) return true;
  return step === 1 || step % every === 0;
}

function recordDecodeProfileStep(state, entry) {
  if (!entry || !entry.timings) return;
  if (!state.stats.decodeProfileSteps) {
    state.stats.decodeProfileSteps = [];
  }
  state.stats.decodeProfileSteps.push(entry);
}

export function shouldUseBatchDecode(config) {
  return config.batchSize > 1
    && config.useGPU
    && config.gpuSamplingAvailable
    && !config.disableMultiTokenDecode
    && !config.disableCommandBatching;
}

export function resolveBatchStop(tokens, stopFlags, stopTokenIds, eosTokenId) {
  let actualCount = tokens.length;
  if (stopFlags) {
    const maxFlags = Math.min(stopFlags.length, tokens.length);
    for (let i = 0; i < maxFlags; i++) {
      if (stopFlags[i] === 1) {
        actualCount = i + 1;
        break;
      }
    }
  }

  for (let i = 0; i < actualCount; i++) {
    if (isStopToken(tokens[i], stopTokenIds, eosTokenId)) {
      actualCount = i + 1;
      break;
    }
  }

  return actualCount;
}

async function runDecodeLayers(state, tokenId, opts, helpers) {
  const config = state.modelConfig;
  const debugCheckBuffer = state.debug ? helpers.debugCheckBuffer : undefined;

  const context = helpers.buildLayerContext(undefined, true, opts.debugLayers);

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

  const embedTensor = await embed([tokenId], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
  });

  let hiddenStates = embedTensor.buffer;

  for (let l = 0; l < config.numLayers; l++) {
    const prevStates = hiddenStates;
    hiddenStates = (await processLayer(l, hiddenStates, 1, false, context));
    state.decodeBuffers.swapPingPong();
    if (prevStates instanceof GPUBuffer && prevStates !== hiddenStates) {
      const isPreAllocated = prevStates === decodeHiddenBuffer || prevStates === decodeAltBuffer;
      if (!isPreAllocated) {
        releaseBuffer(prevStates);
      }
    }
  }

  return { hiddenStates, decodeHiddenBuffer, decodeAltBuffer, debugCheckBuffer, context };
}

export async function decodeStep(state, currentIds, opts, helpers) {
  const lastToken = currentIds[currentIds.length - 1];
  const numTokens = 1;
  const config = state.modelConfig;
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const batchingConfig = state.runtimeConfig.inference.batching;
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
    const path = selectRuleValue('inference', 'config', 'tracePath', { useRecorder: Boolean(recorder) });
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
  const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });

  const embedTensor = await embed([lastToken], embedBuffer, {
    hiddenSize: config.hiddenSize,
    vocabSize: config.vocabSize,
    scaleEmbeddings: config.scaleEmbeddings,
    recorder,
    outputBuffer: decodeHiddenBuffer ?? undefined,
    transpose: state.embeddingTranspose,
    debugProbes: state.runtimeConfig.shared.debug.probes,
    activationDtype,
    embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
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

  const logitSoftcap = config.finalLogitSoftcapping === null
    ? 0
    : config.finalLogitSoftcapping;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad ?? null;
  const lmHeadIsCpu = isCpuWeightBuffer(state.weights.get('lm_head'));
  const useGPUSampling = state.useGPU && isGPUSamplingAvailable() && !lmHeadIsCpu;
  const useFusedDecode = recorder && useGPUSampling && !state.disableFusedDecode;

  if (useFusedDecode) {
    const ring = state.decodeRing;
    let ringSlot = null;
    if (ring) {
      ring.ensure({
        batchSize: 1,
        tokensPerInterval: 1,
        stopCheckMode: batchingConfig.stopCheckMode,
        ringTokens: batchingConfig.ringTokens,
        ringStop: batchingConfig.ringStop,
        ringStaging: batchingConfig.ringStaging,
      });
      ringSlot = ring.acquire();
    }

    const { logitsBuffer, vocabSize, logitsDtype } = await recordLogitsGPU(
      recorder,
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
    );

    const ringTokensBuffer = ringSlot?.tokens ?? null;
    const sampleOutputBuffer = opts.temperature < samplingDefaults.greedyThreshold
      ? await recordArgmax(recorder, logitsBuffer, vocabSize, {
          padTokenId,
          logitSoftcap,
          logitsDtype,
          outputBuffer: ringTokensBuffer ?? undefined,
          outputIndex: 0,
        })
      : await recordGPUSample(recorder, logitsBuffer, vocabSize, {
          temperature: opts.temperature,
          topK: opts.topK,
          padTokenId,
          logitSoftcap,
          logitsDtype,
          outputBuffer: ringTokensBuffer ?? undefined,
          outputIndex: 0,
          greedyThreshold: samplingDefaults.greedyThreshold,
        });

    const ringStagingBuffer = ringSlot?.stagingTokens ?? null;
    const stagingBuffer = ringStagingBuffer ?? device.createBuffer({
      label: 'sample_staging',
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const ownsStagingBuffer = !ringStagingBuffer;
    const ownsSampleOutputBuffer = !ringTokensBuffer || sampleOutputBuffer !== ringTokensBuffer;

    const isPreAllocated = hiddenStates === decodeHiddenBuffer || hiddenStates === decodeAltBuffer;
    const encoder = recorder.getEncoder();
    encoder.copyBufferToBuffer(sampleOutputBuffer, 0, stagingBuffer, 0, 4);

    recorder.submit();

    if (!allowReadback('pipeline.decode.sample')) {
      throw new Error('[Pipeline] GPU readback disabled for sampling');
    }

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const nextToken = new Uint32Array(stagingBuffer.getMappedRange())[0];
    stagingBuffer.unmap();
    if (ownsStagingBuffer) {
      stagingBuffer.destroy();
    }
    ring?.advance();

    log.debug('Decode', `Step ${state.decodeStepCount}: token=${nextToken} (vocabSize=${config.vocabSize})`);

    const invalidToken = nextToken >= config.vocabSize
      || (padTokenId != null && nextToken === padTokenId)
      || (padTokenId == null && nextToken === 0);
    if (invalidToken) {
      log.warn('Decode', `Suspicious token ${nextToken} (vocabSize=${config.vocabSize}, step=${state.decodeStepCount})`);
      if (allowReadback('pipeline.decode.debug-logits')) {
        try {
          const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
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
    if (ownsSampleOutputBuffer) {
      releaseBuffer(sampleOutputBuffer);
    }

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
        recordDecodeProfileStep(state, { step: state.decodeStepCount, timings, totalMs: total ?? undefined });
        if (shouldLogProfileStep(state, state.decodeStepCount)) {
          log.warn('Profile', `Decode step ${state.decodeStepCount}:`);
          log.warn('Profile', CommandRecorder.formatProfileReport(timings));
        }
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
        recordDecodeProfileStep(state, { step: state.decodeStepCount, timings, totalMs: total ?? undefined });
        if (shouldLogProfileStep(state, state.decodeStepCount)) {
          log.warn('Profile', `Decode step ${state.decodeStepCount} (layers only):`);
          log.warn('Profile', CommandRecorder.formatProfileReport(timings));
        }
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
        ? await runArgmax(logitsBuffer, vocabSize, { padTokenId, logitSoftcap, logitsDtype, outputIndex: 0 })
        : await runGPUSample(logitsBuffer, vocabSize, {
            temperature: opts.temperature,
            topK: opts.topK,
            padTokenId,
            logitSoftcap,
            logitsDtype,
            outputIndex: 0,
            greedyThreshold: samplingDefaults.greedyThreshold,
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

export async function decodeStepLogits(state, currentIds, opts, helpers) {
  const lastToken = currentIds[currentIds.length - 1];
  const numTokens = 1;
  const config = state.modelConfig;

  state.decodeStepCount++;

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer, debugCheckBuffer } = await runDecodeLayers(
    state,
    lastToken,
    opts,
    helpers
  );

  let logitsBuffer = null;
  let logitsDtype = null;
  let rawVocabSize = config.vocabSize;
  let logits = null;

  if (state.useGPU && !isCpuWeightBuffer(state.weights.get('lm_head'))) {
    const logitsResult = await computeLogitsGPU(
      hiddenStates,
      numTokens,
      helpers.getLogitsWeights(),
      helpers.getLogitsConfig(),
      state.debugFlags
    );

    if (logitsResult) {
      logitsBuffer = logitsResult.logitsBuffer;
      logitsDtype = logitsResult.logitsDtype;
      rawVocabSize = logitsResult.vocabSize;

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsDtype });
      const logitsData = await readBuffer(logitsBuffer, numTokens * rawVocabSize * logitsBytes);
      const rawLogits = decodeReadback(logitsData, logitsDtype);
      const finalized = await finalizeLogits(
        rawLogits,
        numTokens,
        rawVocabSize,
        config.vocabSize,
        config,
        state.runtimeConfig.shared.debug.probes
      );
      logits = extractLastPositionLogits(finalized, numTokens, config.vocabSize);
    }
  }

  if (!logits) {
    const rawLogits = await computeLogits(
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
    logits = extractLastPositionLogits(rawLogits, numTokens, config.vocabSize);
  }

  const isPreAllocated = hiddenStates === decodeHiddenBuffer || hiddenStates === decodeAltBuffer;
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;

  return {
    logits,
    logitsBuffer,
    logitsDtype,
    rawVocabSize,
    vocabSize: config.vocabSize,
  };
}

export async function advanceWithToken(state, tokenId, opts, helpers) {
  state.decodeStepCount++;

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  const isPreAllocated = hiddenStates === decodeHiddenBuffer || hiddenStates === decodeAltBuffer;
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;
}

export async function advanceWithTokenAndEmbedding(state, tokenId, opts, helpers, embeddingMode) {
  if (embeddingMode !== 'last') {
    throw new Error(`advanceWithTokenAndEmbedding: unsupported embeddingMode "${embeddingMode}" (v0 supports "last" only)`);
  }

  state.decodeStepCount++;

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  if (!allowReadback('pipeline.advance.embedding')) {
    throw new Error('GPU readback disabled; cannot return embedding');
  }

  const device = getDevice();
  if (!device) {
    throw new Error('GPU device not available');
  }

  const config = state.modelConfig;
  const activationDtype = state.runtimeConfig.inference.compute.activationDtype;
  const activationBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: activationDtype });

  const sampleSize = config.hiddenSize * activationBytes;
  const staging = device.createBuffer({
    size: sampleSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const enc = device.createCommandEncoder({ label: 'advance_with_embedding_copy' });
  enc.copyBufferToBuffer(hiddenStates, 0, staging, 0, sampleSize);
  device.queue.submit([enc.finish()]);

  await staging.mapAsync(GPUMapMode.READ);
  const embedding = decodeReadback(staging.getMappedRange().slice(0), activationDtype);
  staging.unmap();
  staging.destroy();

  const isPreAllocated = hiddenStates === decodeHiddenBuffer || hiddenStates === decodeAltBuffer;
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;

  return {
    embedding,
    embeddingMode,
    seqLen: state.currentSeqLen,
  };
}

export async function generateNTokensGPU(state, startToken, N, currentIds, opts, helpers) {
  const device = getDevice();
  const config = state.modelConfig;
  const samplingDefaults = state.runtimeConfig.inference.sampling;
  const batchingConfig = state.runtimeConfig.inference.batching;
  const batchSize = opts.batchSize ?? batchingConfig.batchSize;
  const readbackInterval = batchingConfig.readbackInterval == null ? 1 : batchingConfig.readbackInterval;
  const stopCheckMode = opts.stopCheckMode ?? batchingConfig.stopCheckMode;
  const effectiveStopCheckMode = readbackInterval > 1 ? 'per-token' : stopCheckMode;
  const tokensPerInterval = batchSize * readbackInterval;
  const recorder = opts.profile
    ? createProfilingRecorder('batch_decode')
    : createCommandRecorder('batch_decode');
  const lmHead = state.weights.get('lm_head');
  if (lmHead && isCpuWeightBuffer(lmHead)) {
    throw new Error('[Pipeline] GPU-only decode not supported with CPU-resident LM head.');
  }

  if (!Number.isFinite(N) || N <= 0) {
    throw new Error('[Pipeline] generateNTokensGPU requires N > 0.');
  }
  if (N > tokensPerInterval) {
    throw new Error('[Pipeline] Batch size exceeds decode ring capacity.');
  }

  const stopTokenIds = config.stopTokenIds;
  const eosToken = state.tokenizer?.getSpecialTokens?.()?.eos;
  const padTokenId = state.tokenizer?.getSpecialTokens?.()?.pad ?? null;
  const logitSoftcap = config.finalLogitSoftcapping === null
    ? 0
    : config.finalLogitSoftcapping;
  if (eosToken == null && stopTokenIds.length === 0) {
    throw new Error('[Pipeline] Missing EOS token. Ensure tokenizer or manifest provides stop tokens.');
  }
  const eosTokenId = eosToken ?? stopTokenIds[0];
  if (eosTokenId == null) {
    throw new Error('[Pipeline] Missing EOS token. Ensure tokenizer or manifest provides stop tokens.');
  }
  const maxTokens = opts.maxTokens ?? batchingConfig.maxTokens;
  const maxSeqLen = state.currentSeqLen + maxTokens;

  const recordStart = performance.now();

  const ring = state.decodeRing;
  let ringSlot = null;
  if (ring) {
    ring.ensure({
      batchSize,
      tokensPerInterval,
      stopCheckMode: effectiveStopCheckMode,
      ringTokens: batchingConfig.ringTokens,
      ringStop: batchingConfig.ringStop,
      ringStaging: batchingConfig.ringStaging,
    });
    ringSlot = ring.acquire();
  }

  const tokenCapacity = ringSlot?.tokens ? ringSlot.tokensPerInterval : N;
  const tokensBuffer = ringSlot?.tokens ?? device.createBuffer({
    size: (tokenCapacity + 1) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const ownsTokensBuffer = !ringSlot?.tokens;

  const stopCapacity = ringSlot?.stop ? ringSlot.tokensPerInterval + 1 : N + 1;
  const stopBuffer = effectiveStopCheckMode === 'per-token'
    ? ringSlot?.stop ?? device.createBuffer({
        size: stopCapacity * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      })
    : null;
  const ownsStopBuffer = effectiveStopCheckMode === 'per-token' && !ringSlot?.stop;

  const tokensStagingBuffer = ringSlot?.stagingTokens ?? device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const ownsTokensStaging = !ringSlot?.stagingTokens;

  const stopStagingBuffer = effectiveStopCheckMode === 'per-token'
    ? ringSlot?.stagingStop ?? device.createBuffer({
        size: N * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
    : null;
  const ownsStopStaging = effectiveStopCheckMode === 'per-token' && !ringSlot?.stagingStop;

  device.queue.writeBuffer(tokensBuffer, 0, new Uint32Array([startToken]));
  if (stopBuffer) {
    const stopElements = stopBuffer.size / 4;
    const zeroStopData = ringSlot?.zeroStopData;
    const clearData = zeroStopData && zeroStopData.length <= stopElements
      ? zeroStopData
      : new Uint32Array(stopElements);
    device.queue.writeBuffer(stopBuffer, 0, clearData);
  }

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
      embeddingDtype: selectRuleValue('inference', 'dtype', 'f16OrF32FromDtype', { dtype: embedDtype }),
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

    const outputIndex = i + 1;
    if (opts.temperature < samplingDefaults.greedyThreshold) {
      await recordArgmax(recorder, logitsBuffer, vocabSize, {
        padTokenId,
        logitSoftcap,
        logitsDtype,
        outputBuffer: tokensBuffer,
        outputIndex,
      });
    } else {
      await recordGPUSample(recorder, logitsBuffer, vocabSize, {
        temperature: opts.temperature,
        topK: opts.topK,
        padTokenId,
        logitSoftcap,
        logitsDtype,
        outputBuffer: tokensBuffer,
        outputIndex,
        greedyThreshold: samplingDefaults.greedyThreshold,
      });
    }

    const stopCheck = effectiveStopCheckMode === 'per-token'
      ? recordCheckStop(recorder, {
          sampledTokenBuffer: tokensBuffer,
          shouldStopBuffer: stopBuffer,
          tokenIndex: outputIndex,
          eosTokenId,
          maxTokens: maxSeqLen,
          currentPos,
        })
      : null;

    if (hiddenStatesBuffer instanceof GPUBuffer && !context.decodeBuffers?.ownsBuffer(hiddenStatesBuffer)) {
      recorder.trackTemporaryBuffer(hiddenStatesBuffer);
    }
    if (logitsBuffer instanceof GPUBuffer) {
      recorder.trackTemporaryBuffer(logitsBuffer);
    }
    if (stopCheck instanceof GPUBuffer && stopCheck !== stopBuffer) {
      recorder.trackTemporaryBuffer(stopCheck);
    }
  }

  const recordMs = performance.now() - recordStart;
  state.stats.decodeRecordMs = (state.stats.decodeRecordMs ?? 0) + recordMs;

  const encoder = recorder.getEncoder();
  encoder.copyBufferToBuffer(tokensBuffer, 4, tokensStagingBuffer, 0, N * 4);
  if (effectiveStopCheckMode === 'per-token' && stopBuffer && stopStagingBuffer) {
    encoder.copyBufferToBuffer(stopBuffer, 4, stopStagingBuffer, 0, N * 4);
  }

  recorder.submit();

  if (!allowReadback('pipeline.decode.sample')) {
    throw new Error('[Pipeline] GPU readback disabled for sampling');
  }

  const readbackStart = performance.now();
  const mapPromises = [tokensStagingBuffer.mapAsync(GPUMapMode.READ)];
  if (stopStagingBuffer) {
    mapPromises.push(stopStagingBuffer.mapAsync(GPUMapMode.READ));
  }
  await Promise.all(mapPromises);
  const readbackWaitMs = performance.now() - readbackStart;
  state.stats.decodeReadbackWaitMs = (state.stats.decodeReadbackWaitMs ?? 0) + readbackWaitMs;

  const submitWaitMs = recorder.getSubmitLatencyMs();
  if (submitWaitMs != null) {
    state.stats.decodeSubmitWaitMs = (state.stats.decodeSubmitWaitMs ?? 0) + submitWaitMs;
  }

  getUniformCache().flushPendingDestruction();

  const tokensView = new Uint32Array(tokensStagingBuffer.getMappedRange());
  const tokens = Array.from(tokensView.subarray(0, N));

  const stopFlags = stopStagingBuffer
    ? new Uint32Array(stopStagingBuffer.getMappedRange().slice(0, N * 4))
    : null;

  if (stopFlags) {
    log.debug('Pipeline', `[STOP] N=${N} flags=[${Array.from(stopFlags).join(',')}] tokens=[${tokens.join(',')}] eos=${eosTokenId}`);
  }

  const actualCount = resolveBatchStop(tokens, stopFlags, stopTokenIds, eosToken);

  tokensStagingBuffer.unmap();
  if (stopStagingBuffer) {
    stopStagingBuffer.unmap();
  }

  const generatedTokens = tokens.slice(0, actualCount);

  if (ownsTokensBuffer) tokensBuffer.destroy();
  if (ownsStopBuffer) stopBuffer?.destroy();
  if (ownsTokensStaging) tokensStagingBuffer.destroy();
  if (ownsStopStaging) stopStagingBuffer?.destroy();

  if (opts.profile && recorder.isProfilingEnabled()) {
    const timings = await recorder.resolveProfileTimings();
    const total = sumProfileTimings(timings);
    if (total !== null) {
      state.stats.gpuTimeDecodeMs = (state.stats.gpuTimeDecodeMs ?? 0) + total;
    }
    if (timings) {
      recordDecodeProfileStep(state, {
        batch: true,
        stepStart: state.decodeStepCount + 1,
        stepCount: actualCount,
        timings,
        totalMs: total ?? undefined,
      });
      const stepStart = state.decodeStepCount + 1;
      if (shouldLogProfileStep(state, stepStart)) {
        log.warn('Profile', `Batch decode (N=${N}):`);
        log.warn('Profile', CommandRecorder.formatProfileReport(timings));
      }
    }
  }

  state.currentSeqLen += actualCount;
  ring?.advance();

  return { tokens: generatedTokens, actualCount };
}
