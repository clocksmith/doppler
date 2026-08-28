

import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { runGather, recordGather, runGatherSplit, recordGatherSplit, runScale, recordScale } from '../../../gpu/kernel-selector.js';
import { trace } from '../../../debug/index.js';
import { decodeReadback } from './debug-utils/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { isCpuWeightBuffer, isGpuBufferInstance, isSplitWeightBuffer } from '../../../gpu/weight-buffer.js';
import { f16ToF32 } from '../../../loader/dtype-utils.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { resolveEmbeddingNormalization } from './embedding-contract.js';
import { finalizeEmbeddingOutput } from './embedding-normalization.js';
import { f32ToF16Array } from '../../kv-cache/types.js';
import { runProbes } from './probes.js';

const bf16ScratchU32 = new Uint32Array(1);
const bf16ScratchF32 = new Float32Array(bf16ScratchU32.buffer);

function bf16ToF32(value) {
  bf16ScratchU32[0] = (value & 0xffff) << 16;
  return bf16ScratchF32[0];
}

export function isRangeBackedCpuEmbeddingSource(value) {
  return (
    typeof value === 'object'
    && value !== null
    && value.kind === 'tensor_range_source'
    && typeof value.loadRange === 'function'
  );
}

export function normalizeRangeBytes(value, label) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error(`[Embed] ${label} returned unsupported byte payload type "${value?.constructor?.name ?? typeof value}".`);
}

export function decodeRangeChunkIntoOutput(bytes, sourceDtype, output, dstOffset, hiddenSize) {
  if (sourceDtype === 'f16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, hiddenSize);
    for (let i = 0; i < hiddenSize; i++) {
      output[dstOffset + i] = f16ToF32(values[i]);
    }
    return;
  }
  if (sourceDtype === 'bf16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, hiddenSize);
    for (let i = 0; i < hiddenSize; i++) {
      output[dstOffset + i] = bf16ToF32(values[i]);
    }
    return;
  }
  if (((bytes.byteOffset % 4) === 0) && ((bytes.byteLength % 4) === 0)) {
    output.set(new Float32Array(bytes.buffer, bytes.byteOffset, hiddenSize), dstOffset);
    return;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < hiddenSize; i++) {
    output[dstOffset + i] = view.getFloat32(i * 4, true);
  }
}

export function resolveEmbeddingScale(config, hiddenSize) {
  const embeddingScale = config.embeddingScale;
  const scaleEmbeddings = config.scaleEmbeddings;
  if (embeddingScale === undefined) {
    throw new Error('[Embed] embeddingScale must be explicitly set (null to use scaleEmbeddings semantics).');
  }
  if (scaleEmbeddings == null) {
    throw new Error('[Embed] scaleEmbeddings is required.');
  }
  if (embeddingScale !== null) {
    const value = Number(embeddingScale);
    if (!Number.isFinite(value) || value <= 0) {
      throw new Error(`[Embed] embeddingScale must be a positive finite number or null; got "${String(embeddingScale)}".`);
    }
    if (scaleEmbeddings === true) {
      throw new Error('[Embed] embeddingScale cannot be set when scaleEmbeddings is true.');
    }
    return value;
  }
  return scaleEmbeddings === true ? Math.sqrt(hiddenSize) : 1;
}

export function selectPreloadedCpuEmbeddingValues({
  preloadedCpuRow = null,
  preloadedCpuBatchedRows = null,
  numTokens,
  inputHiddenSize,
  hiddenSize,
  hiddenOffset,
}) {
  if (preloadedCpuRow == null && preloadedCpuBatchedRows == null) {
    return null;
  }
  if (preloadedCpuRow != null && preloadedCpuBatchedRows != null) {
    throw new Error('[Embed] preloadedCpuRow and preloadedCpuBatchedRows are mutually exclusive.');
  }
  if (!Number.isInteger(numTokens) || numTokens <= 0) {
    throw new Error('[Embed] preloaded CPU embeddings require a positive integer numTokens.');
  }
  if (!Number.isInteger(inputHiddenSize) || inputHiddenSize <= 0) {
    throw new Error('[Embed] preloaded CPU embeddings require a positive integer inputHiddenSize.');
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw new Error('[Embed] preloaded CPU embeddings require a positive integer hiddenSize.');
  }
  if (!Number.isInteger(hiddenOffset) || hiddenOffset < 0 || hiddenOffset + hiddenSize > inputHiddenSize) {
    throw new Error(
      `[Embed] preloaded CPU embedding slice [${hiddenOffset}, ${hiddenOffset + hiddenSize}) ` +
      `exceeds inputHiddenSize=${inputHiddenSize}.`
    );
  }

  if (preloadedCpuRow != null) {
    if (!(preloadedCpuRow instanceof Float32Array)) {
      throw new Error('[Embed] preloadedCpuRow must be a Float32Array.');
    }
    if (numTokens !== 1) {
      throw new Error('[Embed] preloadedCpuRow requires numTokens=1.');
    }
    if (preloadedCpuRow.length !== inputHiddenSize) {
      throw new Error(
        `[Embed] preloadedCpuRow length=${preloadedCpuRow.length}, expected inputHiddenSize=${inputHiddenSize}.`
      );
    }
    return preloadedCpuRow.slice(hiddenOffset, hiddenOffset + hiddenSize);
  }

  if (!(preloadedCpuBatchedRows instanceof Float32Array)) {
    throw new Error('[Embed] preloadedCpuBatchedRows must be a Float32Array.');
  }
  const expectedLength = numTokens * inputHiddenSize;
  if (preloadedCpuBatchedRows.length !== expectedLength) {
    throw new Error(
      `[Embed] preloadedCpuBatchedRows length=${preloadedCpuBatchedRows.length}, ` +
      `expected numTokens*inputHiddenSize=${expectedLength}.`
    );
  }
  const selected = new Float32Array(numTokens * hiddenSize);
  for (let tokenIndex = 0; tokenIndex < numTokens; tokenIndex++) {
    const sourceStart = tokenIndex * inputHiddenSize + hiddenOffset;
    const targetStart = tokenIndex * hiddenSize;
    selected.set(
      preloadedCpuBatchedRows.subarray(sourceStart, sourceStart + hiddenSize),
      targetStart
    );
  }
  return selected;
}

function uploadPreloadedCpuEmbedding(device, values, dtype, shape, outputBuffer = null) {
  const payload = dtype === 'f16' ? f32ToF16Array(values) : values;
  const requiredBytes = payload.byteLength;
  if (outputBuffer && outputBuffer.size < requiredBytes) {
    throw new Error(
      `[Embed] preallocated output has ${outputBuffer.size} bytes; ` +
      `preloaded CPU embedding requires ${requiredBytes}.`
    );
  }
  const ownsBuffer = outputBuffer == null;
  const buffer = outputBuffer ?? acquireBuffer(requiredBytes, undefined, 'embed_preloaded_cpu');
  try {
    device.queue.writeBuffer(buffer, 0, payload);
    return createTensor(buffer, dtype, shape, 'embed_preloaded_cpu');
  } catch (error) {
    if (ownsBuffer) {
      releaseBuffer(buffer);
    }
    throw error;
  }
}

export async function embed(tokenIds, embedBuffer, config) {
  const {
    hiddenSize,
    vocabSize,
    scaleEmbeddings,
    embeddingScale,
    embeddingNormalization = null,
    debug = false,
    recorder,
    outputBuffer: preAllocatedOutput,
    transpose = false,
    activationDtype,
    embeddingDtype,
    operatorDiagnostics,
    probeStage = 'embed_out',
    inputHiddenSize = hiddenSize,
    hiddenOffset = 0,
    embeddingStorageEncoding = null,
    preloadedCpuRow = null,
    preloadedCpuBatchedRows = null,
  } = config;
  const device = getDevice();
  const resolvedEmbeddingScale = resolveEmbeddingScale({ scaleEmbeddings, embeddingScale }, hiddenSize);
  const resolvedEmbeddingNormalization = resolveEmbeddingNormalization(embeddingNormalization);
  const intermediateOutputBuffer = resolvedEmbeddingNormalization ? null : preAllocatedOutput;
  const tokenBufferInput = isGpuBufferInstance(tokenIds);
  let tokenIdArray = tokenBufferInput ? null :  (tokenIds);
  const numTokens = tokenBufferInput
    ? (config.numTokens ?? 0)
    : (tokenIdArray?.length ?? 0);
  const indexOffset = tokenBufferInput ? (config.indexOffset ?? 0) : 0;

  if (!device) throw new Error('GPU device not available');
  if (!activationDtype || !embeddingDtype) {
    throw new Error('[Embed] activationDtype and embeddingDtype are required.');
  }

  // Check if F16 output is requested and supported
  const caps = getKernelCapabilities();
  const requiresF16Output = activationDtype === 'f16';
  if (requiresF16Output && !caps.hasF16) {
    throw new Error('[Embed] activationDtype="f16" requires shader-f16 support.');
  }
  const useF16 = requiresF16Output;
  
  const dtype = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16 });

  if (debug) {
    trace.embed(
      `tokens=${numTokens}, hidden=${hiddenSize}, vocab=${vocabSize}, scaleEmbeddings=${scaleEmbeddings}, ` +
      `transpose=${transpose}, indexOffset=${indexOffset}, inputHiddenSize=${inputHiddenSize}, ` +
      `hiddenOffset=${hiddenOffset}, activationDtype=${activationDtype}, useF16=${useF16}`
    );
    if (tokenBufferInput) {
      trace.embed('TOKEN_IDS: [gpu-buffer]');
    } else {
      trace.embed(`TOKEN_IDS: [${Array.from(tokenIdArray ?? []).join(', ')}]`);
    }
  }

  if (tokenBufferInput && numTokens <= 0) {
    throw new Error('[Embed] numTokens must be provided when tokenIds is a GPUBuffer.');
  }

  const preloadedCpuValues = selectPreloadedCpuEmbeddingValues({
    preloadedCpuRow,
    preloadedCpuBatchedRows,
    numTokens,
    inputHiddenSize,
    hiddenSize,
    hiddenOffset,
  });

  if (isGpuBufferInstance(embedBuffer) && transpose === false) {
    await runProbes('embed_weight_row', embedBuffer, {
      numTokens: vocabSize,
      hiddenSize: inputHiddenSize,
      probes: config.debugProbes,
      recorder,
      operatorDiagnostics,
      dtype: embeddingDtype,
    });
  }

  // Use pre-allocated output buffer if provided, otherwise acquire from pool
  // Pass outputDtype to enable F16 output when in F16 activation mode
  // Pass embeddingDtype so gather kernel uses correct input format
  const gatherOptions = {
    outputBuffer: intermediateOutputBuffer,
    transpose,
    outputDtype: selectRuleValue('shared', 'dtype', 'f16OrF32', { useF16 }),
    embeddingDtype,
    storageEncoding: embeddingStorageEncoding,
    indexOffset,
    inputHiddenSize,
    hiddenOffset,
  };
  let gatherOutput;
  if (preloadedCpuValues) {
    if (!isCpuWeightBuffer(embedBuffer) || !isRangeBackedCpuEmbeddingSource(embedBuffer.data)) {
      throw new Error('[Embed] preloaded CPU rows require a range-backed CpuWeightBuffer source.');
    }
    if (tokenBufferInput) {
      throw new Error('[Embed] preloaded CPU rows require CPU token IDs.');
    }
    gatherOutput = uploadPreloadedCpuEmbedding(
      device,
      preloadedCpuValues,
      dtype,
      [numTokens, hiddenSize],
      intermediateOutputBuffer
    );
  } else {
    if (isCpuWeightBuffer(embedBuffer) || embedBuffer instanceof Float32Array) {
      throw new Error(
        '[Embed] CPU-resident embedding gather requires a verified preloaded row; ' +
        'materialize a GPU or split embedding weight.'
      );
    }
    if (!isGpuBufferInstance(embedBuffer) && !isSplitWeightBuffer(embedBuffer)) {
      throw new Error('[Embed] GPU embeddings required for gather path.');
    }
    const tokenIdBuffer = tokenBufferInput
      ? tokenIds
      : acquireBuffer(Math.max(numTokens * 4, 256), undefined, 'embed_tokens');
    if (!tokenBufferInput) {
      device.queue.writeBuffer(tokenIdBuffer, 0, new Uint32Array( (tokenIdArray)));
    }
    try {
      gatherOutput = isSplitWeightBuffer(embedBuffer)
        ? (
          recorder
            ? await recordGatherSplit(recorder, tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
            : await runGatherSplit(tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
        )
        : (
          recorder
            ? await recordGather(recorder, tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
            : await runGather(tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
        );
    } finally {
      if (!tokenBufferInput) {
        if (recorder) {
          recorder.trackTemporaryBuffer(tokenIdBuffer);
        } else {
          releaseBuffer(tokenIdBuffer);
        }
      }
    }
  }

  // Debug: Verify first token embedding
  if (debug && !recorder && tokenIdArray && tokenIdArray.length > 0) {
    const firstTokenId = tokenIdArray[0];
    const bytesPerElement = useF16 ? 2 : 4;
    const sampleSize = Math.min(32 * bytesPerElement, hiddenSize * bytesPerElement);
    const readback = await readBuffer(gatherOutput.buffer, sampleSize);
    const data = decodeReadback(readback, gatherOptions.outputDtype);

    // Compute statistics
    let sum = 0, sumSq = 0;
    for (const v of data) { sum += v; sumSq += v * v; }
    const mean = sum / data.length;
    const variance = (sumSq / data.length) - (mean * mean);
    const std = Math.sqrt(variance);
    let maxAbs = 0;
    for (let i = 0; i < data.length; i++) {
      const abs = Math.abs(data[i]);
      if (abs > maxAbs) maxAbs = abs;
    }

    trace.embed(`FIRST_TOKEN[${firstTokenId}]: maxAbs=${maxAbs.toFixed(4)}, mean=${mean.toFixed(4)}, std=${std.toFixed(4)}, first8=[${Array.from(data).slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
  }
  if (resolvedEmbeddingScale === 1) {
    return finalizeEmbeddingOutput(gatherOutput, resolvedEmbeddingNormalization, {
      recorder, numTokens, hiddenSize, outputBuffer: preAllocatedOutput,
      probeStage, debugProbes: config.debugProbes, operatorDiagnostics,
    });
  }

  // Debug: check raw embedding values before scaling
  if (debug && !recorder) {
    const bytesPerElement = gatherOptions.outputDtype === 'f16' ? 2 : 4;
    const sampleBytes = Math.min(gatherOutput.buffer.size, numTokens * hiddenSize * bytesPerElement);
    const sample = await readBuffer(gatherOutput.buffer, sampleBytes);
    const f32 = decodeReadback(sample, gatherOptions.outputDtype);
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
      const abs = Math.abs(f32[i]);
      if (abs > maxAbs) maxAbs = abs;
    }
    trace.embed(`RAW (before scale): maxAbs=${maxAbs.toFixed(4)}, scaleFactor=${resolvedEmbeddingScale.toFixed(4)}`);
  }

  const gatheredTensor = createTensor(
    gatherOutput.buffer,
    gatherOptions.outputDtype,
    [numTokens, hiddenSize],
    'embed_gather_output'
  );
  const scaledTensor = recorder
    ? await recordScale(recorder, gatheredTensor, resolvedEmbeddingScale, {
      count: numTokens * hiddenSize,
    })
    : await runScale(gatheredTensor, resolvedEmbeddingScale, {
      count: numTokens * hiddenSize,
    });
  const scaledBuffer = scaledTensor.buffer;
  if (recorder) {
    // Only track if we created this buffer (not pre-allocated)
    // Pre-allocated buffers are managed by the caller (e.g., DecodeBufferManager)
    if (!intermediateOutputBuffer) {
      recorder.trackTemporaryBuffer(gatherOutput.buffer);
    }
  } else {
    // For sync path: only release if not pre-allocated
    if (!intermediateOutputBuffer) {
      releaseBuffer(gatherOutput.buffer);
    }
  }

  if (debug && !recorder) {
    const bytesPerElement = dtype === 'f16' ? 2 : 4;
    const sampleBytes = Math.min(scaledBuffer.size, numTokens * hiddenSize * bytesPerElement);
    const sample = await readBuffer(scaledBuffer, sampleBytes);
    const f32 = decodeReadback(sample, dtype);
    let maxAbs = 0;
    for (let i = 0; i < f32.length; i++) {
      const abs = Math.abs(f32[i]);
      if (abs > maxAbs) maxAbs = abs;
    }
    trace.embed(`SCALED (after *${resolvedEmbeddingScale.toFixed(2)}): maxAbs=${maxAbs.toFixed(4)}, buffer.label=${scaledBuffer.label}, buffer.size=${scaledBuffer.size}`);
    trace.embed(`RETURNING buffer with first8=[${Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
    if (f32.some(x => !Number.isFinite(x))) {
      throw new Error('[Embed] Scaled embedding contains NaN/Inf');
    }
  }
  return finalizeEmbeddingOutput(
    createTensor(scaledBuffer, dtype, [numTokens, hiddenSize], 'embed_output'),
    resolvedEmbeddingNormalization,
    {
      recorder, numTokens, hiddenSize, outputBuffer: preAllocatedOutput,
      probeStage, debugProbes: config.debugProbes, operatorDiagnostics,
    }
  );
}
