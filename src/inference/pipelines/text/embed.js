

import { getDevice, getKernelCapabilities } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { runGather, recordGather, runScale, recordScale } from '../../../gpu/kernel-selector.js';
import { log, trace } from '../../../debug/index.js';
import { runProbes } from './probes.js';
import { decodeReadback } from './debug-utils/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { castF32ToF16, recordCastF32ToF16 } from '../../../gpu/kernels/cast.js';
import { isCpuWeightBuffer, isGpuBufferInstance } from '../../../gpu/weight-buffer.js';
import { f16ToF32 } from '../../../loader/dtype-utils.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

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

export async function embed(tokenIds, embedBuffer, config) {
  const {
    hiddenSize,
    vocabSize,
    scaleEmbeddings,
    debug = false,
    recorder,
    outputBuffer: preAllocatedOutput,
    transpose = false,
    activationDtype,
    embeddingDtype,
    operatorDiagnostics,
    inputHiddenSize = hiddenSize,
    hiddenOffset = 0,
  } = config;
  const device = getDevice();
  const tokenBufferInput = isGpuBufferInstance(tokenIds);
  const tokenIdArray = tokenBufferInput ? null :  (tokenIds);
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
  const useF16 = activationDtype === 'f16' && caps.hasF16;
  
  const dtype = selectRuleValue('inference', 'dtype', 'f16OrF32', { useF16 });

  let cpuEmbeddings = null;
  if (isCpuWeightBuffer(embedBuffer)) {
    const bufDtype = embedBuffer.dtype;
    if (bufDtype !== 'f32' && bufDtype !== 'f16') {
      throw new Error(
        `[Embed] CPU embedding buffer has unsupported dtype '${bufDtype}'; ` +
        `only 'f32' and 'f16' are supported in the CPU gather path.`
      );
    }
    cpuEmbeddings = embedBuffer.data;
  } else if (embedBuffer instanceof Float32Array) {
    cpuEmbeddings = embedBuffer;
  }

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

  if (cpuEmbeddings) {
    if (tokenBufferInput) {
      throw new Error('[Embed] GPU token buffer requires GPU-resident embeddings.');
    }
    if (debug) {
      trace.embed('Using CPU embedding gather (oversized embedding)');
    }

    // Bounds check: warn (not throw) for token IDs outside vocab range.
    // Some tokenizers intentionally produce special OOV token IDs beyond vocabSize.
    if (tokenIdArray) {
      for (let t = 0; t < tokenIdArray.length; t++) {
        const tid = tokenIdArray[t];
        if (tid < 0 || tid >= vocabSize) {
          log.warn(
            'Embed',
            `Token ID ${tid} at position ${t} is outside vocab range [0, ${vocabSize}). ` +
            'This may produce incorrect embeddings.'
          );
          break;
        }
      }
    }

    const output = new Float32Array(numTokens * hiddenSize);

    // Step 6: Batched prefill rows — per-token pre-decoded PLE data for all layers.
    // Layout: Float32Array[numTokens × inputHiddenSize], indexed by
    // [t * inputHiddenSize + hiddenOffset] to select each token's layer slice.
    if (config.preloadedCpuBatchedRows) {
      for (let t = 0; t < numTokens; t++) {
        const srcOffset = t * inputHiddenSize + hiddenOffset;
        const dstOffset = t * hiddenSize;
        output.set(config.preloadedCpuBatchedRows.subarray(srcOffset, srcOffset + hiddenSize), dstOffset);
      }
    } else

    // Fast path: use pre-loaded and pre-decoded PLE row (coalesced read optimization).
    // preloadedCpuRow is a Float32Array containing the full PLE row for the token,
    // indexed by hiddenOffset to select the correct layer's slice.
    if (config.preloadedCpuRow) {
      const rowSlice = config.preloadedCpuRow.subarray(hiddenOffset, hiddenOffset + hiddenSize);
      for (let t = 0; t < numTokens; t++) {
        output.set(rowSlice, t * hiddenSize);
      }
    } else

    // Range-backed path: per-element loadRange calls (original path)
    {
    const rangeBackedSource = isRangeBackedCpuEmbeddingSource(cpuEmbeddings)
      ? cpuEmbeddings
      : null;
    if (rangeBackedSource) {
      const sourceDtype = String(rangeBackedSource.sourceDtype ?? embedBuffer.dtype ?? 'f32').toLowerCase();
      const bytesPerElement = sourceDtype === 'f16' || sourceDtype === 'bf16' ? 2 : 4;
      if (!transpose) {
        for (let t = 0; t < numTokens; t++) {
          const tokenId = (tokenIdArray)[t];
          const srcOffset = tokenId * inputHiddenSize + hiddenOffset;
          const chunk = normalizeRangeBytes(
            await rangeBackedSource.loadRange(srcOffset * bytesPerElement, hiddenSize * bytesPerElement),
            'CPU embedding range source'
          );
          if (chunk.byteLength !== hiddenSize * bytesPerElement) {
            throw new Error(
              `[Embed] CPU embedding range source returned ${chunk.byteLength} bytes, ` +
              `expected ${hiddenSize * bytesPerElement}.`
            );
          }
          decodeRangeChunkIntoOutput(chunk, sourceDtype, output, t * hiddenSize, hiddenSize);
        }
      } else {
        for (let t = 0; t < numTokens; t++) {
          const tokenId = (tokenIdArray)[t];
          const dstOffset = t * hiddenSize;
          for (let h = 0; h < hiddenSize; h++) {
            const chunk = normalizeRangeBytes(
              await rangeBackedSource.loadRange(
                ((hiddenOffset + h) * vocabSize + tokenId) * bytesPerElement,
                bytesPerElement
              ),
              'CPU embedding range source'
            );
            decodeRangeChunkIntoOutput(chunk, sourceDtype, output, dstOffset + h, 1);
          }
        }
      }
    } else {
    // Check actual data type: loader's f16_to_f32 CPU path already decodes F16 into Float32Array,
    // so dtype='f16' does not reliably indicate raw F16 bytes. Only Uint16Array needs per-element decoding.
      const isF16Cpu = cpuEmbeddings instanceof Uint16Array;
      if (!transpose) {
        for (let t = 0; t < numTokens; t++) {
          const tokenId =  (tokenIdArray)[t];
          const srcOffset = tokenId * inputHiddenSize + hiddenOffset;
          if (isF16Cpu) {
            for (let h = 0; h < hiddenSize; h++) {
              output[t * hiddenSize + h] = f16ToF32(cpuEmbeddings[srcOffset + h]);
            }
          } else {
            output.set(cpuEmbeddings.subarray(srcOffset, srcOffset + hiddenSize), t * hiddenSize);
          }
        }
      } else {
        for (let t = 0; t < numTokens; t++) {
          const tokenId =  (tokenIdArray)[t];
          const dstOffset = t * hiddenSize;
          for (let h = 0; h < hiddenSize; h++) {
            const raw = cpuEmbeddings[(hiddenOffset + h) * vocabSize + tokenId];
            output[dstOffset + h] = isF16Cpu ? f16ToF32(raw) : raw;
          }
        }
      }
    }
    } // end else (non-preloaded path)

    if (scaleEmbeddings) {
      const scaleFactor = Math.sqrt(hiddenSize);
      for (let i = 0; i < output.length; i++) {
        output[i] *= scaleFactor;
      }
    }

    if (useF16) {
      const f32Buffer = acquireBuffer(output.byteLength, undefined, 'embed_cpu_f32');
      device.queue.writeBuffer(f32Buffer, 0, output);
      const f32Tensor = createTensor(f32Buffer, 'f32', [numTokens, hiddenSize], 'embed_cpu_f32');
      const outputBytes = numTokens * hiddenSize * 2;
      const outputBuffer = preAllocatedOutput && preAllocatedOutput.size >= outputBytes ? preAllocatedOutput : null;
      const f16Tensor = recorder
        ? await recordCastF32ToF16(recorder, f32Tensor, { outputBuffer })
        : await castF32ToF16(f32Tensor, { outputBuffer });
      if (recorder) {
        recorder.trackTemporaryBuffer(f32Buffer);
      } else {
        releaseBuffer(f32Buffer);
      }
      await runProbes('embed_out', f16Tensor.buffer, {
        numTokens,
        hiddenSize,
        probes: config.debugProbes,
        recorder,
        operatorDiagnostics,
        dtype: 'f16',
      });
      return f16Tensor;
    }

    const outputBytes = output.byteLength;
    const outputBuffer = preAllocatedOutput && preAllocatedOutput.size >= outputBytes
      ? preAllocatedOutput
      : acquireBuffer(outputBytes, undefined, 'embed_cpu_f32_out');
    device.queue.writeBuffer(outputBuffer, 0, output);
    await runProbes('embed_out', outputBuffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes,
      recorder,
      operatorDiagnostics,
    });
    return createTensor(outputBuffer, dtype, [numTokens, hiddenSize], 'embed_output');
  }

  if (tokenBufferInput && numTokens <= 0) {
    throw new Error('[Embed] numTokens must be provided when tokenIds is a GPUBuffer.');
  }
  const tokenIdBuffer = tokenBufferInput
    ? tokenIds
    : acquireBuffer(Math.max(numTokens * 4, 256), undefined, 'embed_tokens');
  if (!tokenBufferInput) {
    device.queue.writeBuffer(tokenIdBuffer, 0, new Uint32Array( (tokenIdArray)));
  }

  // Use pre-allocated output buffer if provided, otherwise acquire from pool
  // Pass outputDtype to enable F16 output when in F16 activation mode
  // Pass embeddingDtype so gather kernel uses correct input format
  const gatherOptions = {
    outputBuffer: preAllocatedOutput,
    transpose,
    outputDtype: selectRuleValue('shared', 'dtype', 'f16OrF32', { useF16 }),
    embeddingDtype,
    indexOffset,
    inputHiddenSize,
    hiddenOffset,
  };
  if (!isGpuBufferInstance(embedBuffer)) {
    throw new Error('[Embed] GPU embeddings required for gather path.');
  }
  const gatherOutput = recorder
    ? await recordGather(recorder, tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions)
    : await runGather(tokenIdBuffer, embedBuffer, numTokens, hiddenSize, vocabSize, gatherOptions);

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
  if (!tokenBufferInput) {
    if (recorder) {
      recorder.trackTemporaryBuffer(tokenIdBuffer);
    } else {
      releaseBuffer(tokenIdBuffer);
    }
  }

  if (!scaleEmbeddings) {
    await runProbes('embed_out', gatherOutput.buffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes,
      recorder,
      operatorDiagnostics,
      dtype: gatherOptions.outputDtype,
    });
    return gatherOutput;
  }

  // Apply embedding scaling: sqrt(hiddenSize)
  const scaleFactor = Math.sqrt(hiddenSize);

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
    trace.embed(`RAW (before scale): maxAbs=${maxAbs.toFixed(4)}, scaleFactor=${scaleFactor.toFixed(4)}`);
  }

  const gatheredTensor = createTensor(
    gatherOutput.buffer,
    gatherOptions.outputDtype,
    [numTokens, hiddenSize],
    'embed_gather_output'
  );
  const scaledTensor = recorder
    ? await recordScale(recorder, gatheredTensor, scaleFactor, {
      count: numTokens * hiddenSize,
    })
    : await runScale(gatheredTensor, scaleFactor, {
      count: numTokens * hiddenSize,
    });
  const scaledBuffer = scaledTensor.buffer;
  if (recorder) {
    // Only track if we created this buffer (not pre-allocated)
    // Pre-allocated buffers are managed by the caller (e.g., DecodeBufferManager)
    if (!preAllocatedOutput) {
      recorder.trackTemporaryBuffer(gatherOutput.buffer);
    }
  } else {
    // For sync path: only release if not pre-allocated
    if (!preAllocatedOutput) {
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
    trace.embed(`SCALED (after *${scaleFactor.toFixed(2)}): maxAbs=${maxAbs.toFixed(4)}, buffer.label=${scaledBuffer.label}, buffer.size=${scaledBuffer.size}`);
    trace.embed(`RETURNING buffer with first8=[${Array.from(f32).slice(0, 8).map(x => x.toFixed(4)).join(', ')}]`);
    if (f32.some(x => !Number.isFinite(x))) {
      throw new Error('[Embed] Scaled embedding contains NaN/Inf');
    }
  }
  await runProbes('embed_out', scaledBuffer, {
    numTokens,
    hiddenSize,
    probes: config.debugProbes,
    recorder,
    operatorDiagnostics,
    dtype,
  });

  return createTensor(scaledBuffer, dtype, [numTokens, hiddenSize], 'embed_output');
}


export async function validateEmbedding(buffer, label, numTokens, hiddenSize) {
  const device = getDevice();
  if (!device) return null;

  const sampleSize = Math.min(1024 * 4, buffer.size);
  const sample = await readBuffer(buffer, sampleSize);
  const f32 = new Float32Array(sample);

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let zeros = 0;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < f32.length; i++) {
    const v = f32[i];
    if (Number.isNaN(v)) {
      nanCount++;
    } else if (!Number.isFinite(v)) {
      infCount++;
    } else {
      if (v === 0) zeros++;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
  }

  const mean = sum / f32.length;

  trace.embed(`${label}: tokens=${numTokens}, hidden=${hiddenSize}`);
  trace.embed(`${label}: min=${min.toFixed(4)}, max=${max.toFixed(4)}, mean=${mean.toFixed(4)}`);
  trace.embed(`${label}: zeros=${zeros}/${f32.length}, NaN=${nanCount}, Inf=${infCount}`);

  return { min, max, mean, zeros, nanCount, infCount };
}
