import { getDevice, getKernelCapabilities } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm, runScale } from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { castF16ToF32, castF32ToF16 } from '../../../../gpu/kernels/cast.js';
import {
  createWeightBuffer,
  createSplitWeightBuffer,
  isWeightBuffer,
  isCpuWeightBuffer,
  isGpuBufferInstance,
  isSplitWeightBuffer,
} from '../../../../gpu/weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { getKernelPathMatmulPrecision, getKernelPathStepPrecision } from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { runProbes } from '../probes.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { f16BufferToF32 } from './cpu.js';
import { readBufferWithCleanup } from './readback.js';
import { resolveLogitInputScale } from './scale-policy.js';
import { finalizeLogitOutputTensor } from './output-transform.js';
import { shouldForceStableF32Logits, createStableF32LogitsKernelPath } from './precision-policy.js';
import {
  extractLmHeadChunk,
  isRangeBackedCpuWeightSource,
  normalizeRangeBytes,
  resolveCpuWeightDims,
  shouldMaterializeSplitLmHeadGPU,
  writeChunkLogits,
} from './plan.js';
import {
  coerceTensorDtype,
  recordGreedyLmHeadArgmaxGPU,
  recordLogitsGPU,
  resolveFinalNormGpuBuffer,
  resolveLmHeadMatmulRole,
  resolveMatmulStepDtype,
  resolvePostLayerStepDtype,
} from './gpu-executor.js';

export {
  extractLmHeadChunk,
  resolveCpuWeightDims,
  shouldMaterializeSplitLmHeadGPU,
  writeChunkLogits,
} from './plan.js';
export {
  recordGreedyLmHeadArgmaxGPU,
  recordLogitsGPU,
} from './gpu-executor.js';

const SPLIT_UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024;


function alignByteLength(byteLength) {
  return Math.ceil(byteLength / 4) * 4;
}

function writeBufferInChunks(queue, buffer, bytes) {
  for (let offset = 0; offset < bytes.byteLength; offset += SPLIT_UPLOAD_CHUNK_BYTES) {
    const end = Math.min(offset + SPLIT_UPLOAD_CHUNK_BYTES, bytes.byteLength);
    queue.writeBuffer(buffer, offset, bytes, offset, end - offset);
  }
}

export function resolveLmHeadChunkRows(
  device,
  numTokens,
  hiddenSize,
  config
) {
  const resolved = config ?? getRuntimeConfig().inference.largeWeights;
  if (resolved.safetyRatio == null) {
    throw new Error('runtime.inference.largeWeights.safetyRatio is required.');
  }
  const safety = Math.min(Math.max(resolved.safetyRatio, 0.1), 1);
  const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxBytes = Math.floor(maxBinding * safety);

  const maxRowsByWeight = Math.floor(maxBytes / (hiddenSize * 4));
  const maxRowsByOutput = Math.floor(maxBytes / (numTokens * 4));
  const maxRows = Math.min(maxRowsByWeight, maxRowsByOutput);

  if (!Number.isFinite(maxRows) || maxRows <= 0) {
    throw new Error(
      `[Logits] LM head chunk size underflow (maxBytes=${maxBytes}, hiddenSize=${hiddenSize}, numTokens=${numTokens}).`
    );
  }

  const override = resolved.lmHeadChunkRows ?? null;
  if (override && override > 0) {
    return Math.min(override, maxRows);
  }
  return maxRows;
}


function resolveSplitLmHeadRows(device, hiddenSize, largeWeightConfig) {
  if (largeWeightConfig.safetyRatio == null) {
    throw new Error('runtime.inference.largeWeights.safetyRatio is required.');
  }
  const safety = Math.min(Math.max(largeWeightConfig.safetyRatio, 0.1), 1);
  const maxBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const maxBytes = Math.floor(maxBinding * safety);
  const rowBytes = hiddenSize * 2;
  const rowsByBinding = Math.floor(maxBytes / rowBytes);
  const requested = largeWeightConfig.lmHeadChunkRows;
  const rows = Number.isInteger(requested) && requested > 0
    ? Math.min(requested, rowsByBinding)
    : rowsByBinding;
  if (!Number.isFinite(rows) || rows <= 0) {
    throw new Error(
      `[Logits] split LM head row size underflow (maxBytes=${maxBytes}, hiddenSize=${hiddenSize}).`
    );
  }
  return rows;
}


function destroySplitWeightBuffer(splitWeight) {
  if (!splitWeight) {
    return;
  }
  for (const section of splitWeight.sections ?? []) {
    try {
      section.buffer.destroy();
    } catch {}
  }
}

async function readSplitLmHeadSectionBytes(lmHead, hiddenSize, rowStart, rowCount) {
  const byteOffset = rowStart * hiddenSize * 2;
  const byteLength = rowCount * hiddenSize * 2;
  const data = lmHead.data;
  if (isRangeBackedCpuWeightSource(data)) {
    const bytes = normalizeRangeBytes(
      await data.loadRange(byteOffset, byteLength),
      'CPU LM head split range source'
    );
    if (bytes.byteLength !== byteLength) {
      throw new Error(
        `[Logits] CPU LM head split source returned ${bytes.byteLength} bytes, expected ${byteLength}.`
      );
    }
    return bytes;
  }
  if (data instanceof Uint16Array) {
    return new Uint8Array(data.buffer, data.byteOffset + byteOffset, byteLength);
  }
  return null;
}

async function materializeSplitLmHeadGPU(lmHead, hiddenSize, weightVocabSize, largeWeightConfig) {
  if (!shouldMaterializeSplitLmHeadGPU(lmHead, largeWeightConfig)) {
    if (lmHead.gpuSplitWeight) {
      destroySplitWeightBuffer(lmHead.gpuSplitWeight);
      lmHead.gpuSplitWeight = null;
    }
    return null;
  }
  if (lmHead.gpuSplitWeight) {
    return lmHead.gpuSplitWeight;
  }
  if (lmHead.layout !== 'row' || lmHead.dtype !== 'f16') {
    return null;
  }

  const device = getDevice();
  if (!device) {
    throw new Error('[Logits] GPU device not available for split LM head materialization.');
  }

  const rowsPerSection = resolveSplitLmHeadRows(device, hiddenSize, largeWeightConfig);
  const createdBuffers = [];
  try {
    const sections = [];
    for (let rowStart = 0; rowStart < weightVocabSize; rowStart += rowsPerSection) {
      const rowCount = Math.min(rowsPerSection, weightVocabSize - rowStart);
      const bytes = await readSplitLmHeadSectionBytes(lmHead, hiddenSize, rowStart, rowCount);
      if (!bytes) {
        for (const buffer of createdBuffers) {
          buffer.destroy();
        }
        return null;
      }
      const buffer = device.createBuffer({
        label: `${lmHead.label ?? 'lm_head'}:lazy_split:${sections.length}`,
        size: alignByteLength(bytes.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      createdBuffers.push(buffer);
      writeBufferInChunks(device.queue, buffer, bytes);
      sections.push({ buffer, rowStart, rowCount });
    }
    const splitWeight = createSplitWeightBuffer(
      sections,
      lmHead.dtype,
      lmHead.layout,
      lmHead.shape,
      lmHead.label
    );
    log.warn(
      'Logits',
      `LM head "${lmHead.label ?? 'lm_head'}" materialized as lazy split GPU sections ` +
      `(${sections.length} sections, dtype=${lmHead.dtype}, layout=${lmHead.layout}).`
    );
    Object.defineProperty(lmHead, 'gpuSplitWeight', {
      value: splitWeight,
      configurable: true,
      enumerable: false,
      writable: true,
    });
    return splitWeight;
  } catch (error) {
    for (const buffer of createdBuffers) {
      try {
        buffer.destroy();
      } catch {}
    }
    throw error;
  }
}


export async function computeChunkedLogitsGPU(
  normedTensor,
  lmHead,
  numTokens,
  hiddenSize,
  vocabSize,
  weightVocabSize,
  debugProbes,
  operatorDiagnostics,
  largeWeightConfig,
  kernelPath = null,
  executionPolicies = null
) {
  const device = getDevice();
  if (!device) {
    throw new Error('[Logits] GPU device not available for chunked LM head.');
  }
  if (!largeWeightConfig) {
    throw new Error('[Logits] largeWeights config is required for chunked LM head.');
  }

  const splitLmHead = await materializeSplitLmHeadGPU(lmHead, hiddenSize, weightVocabSize, largeWeightConfig);
  if (splitLmHead) {
    return computeSplitLogitsGPU(
      normedTensor,
      splitLmHead,
      numTokens,
      hiddenSize,
      vocabSize,
      weightVocabSize,
      debugProbes,
      operatorDiagnostics,
      kernelPath,
      executionPolicies
    );
  }

  const chunkRows = resolveLmHeadChunkRows(device, numTokens, hiddenSize, largeWeightConfig);
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const lmHeadRole = resolveLmHeadMatmulRole(phase);
  const lmHeadInputDtype = resolveMatmulStepDtype(lmHeadRole, phase, kernelPath, normedTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = resolveMatmulStepDtype(lmHeadRole, phase, kernelPath, normedTensor.dtype, 'outputDtype');
  const caps = getKernelCapabilities();
  const weightDtype = selectRuleValue('inference', 'dtype', 'lmHeadChunkWeightDtype', {
    preferF16: largeWeightConfig.preferF16,
    lmHeadDtype: lmHead.dtype,
    hasF16: caps.hasF16,
  });
  const preferF16 = weightDtype === 'f16';
  const logits = new Float32Array(numTokens * vocabSize);

  if (isTraceEnabled('logits')) {
    trace.logits(`LM_HEAD_CHUNKED: vocab=${vocabSize}, chunkRows=${chunkRows}, layout=${lmHead.layout}, f16=${preferF16}`);
  }

  const matmulInput = lmHeadInputDtype !== normedTensor.dtype
    ? await coerceTensorDtype(normedTensor, lmHeadInputDtype, null, {
      executionPolicies,
      op: 'lm_head',
      transitionDeclaredBy: 'step_precision',
    })
    : normedTensor;

  for (let rowOffset = 0; rowOffset < vocabSize; rowOffset += chunkRows) {
    const rowCount = Math.min(chunkRows, vocabSize - rowOffset);
    const chunkShape = lmHead.layout === 'column'
      ? [hiddenSize, rowCount]
      : [rowCount, hiddenSize];

    let weightBuffer;
    if (preferF16 && lmHead.layout === 'row' && lmHead.dtype === 'f16') {
      const chunkBytes = await readSplitLmHeadSectionBytes(lmHead, hiddenSize, rowOffset, rowCount);
      if (!chunkBytes) {
        throw new Error('[Logits] F16 LM head chunk source is not range-readable.');
      }
      const f16Buffer = acquireBuffer(alignByteLength(chunkBytes.byteLength), undefined, 'lm_head_chunk_f16');
      writeBufferInChunks(device.queue, f16Buffer, chunkBytes);
      weightBuffer = createWeightBuffer(f16Buffer, 'f16', lmHead.layout, chunkShape, 'lm_head_chunk_f16');
    } else {
      const chunkData = await extractLmHeadChunk(
        lmHead.data,
        lmHead.layout,
        hiddenSize,
        weightVocabSize,
        rowOffset,
        rowCount,
        lmHead.dtype
      );

      const f32Buffer = acquireBuffer(chunkData.byteLength, undefined, 'lm_head_chunk_f32');
      device.queue.writeBuffer(
        f32Buffer,
        0,
        chunkData.buffer,
        chunkData.byteOffset,
        chunkData.byteLength
      );

      weightBuffer = createWeightBuffer(f32Buffer, 'f32', lmHead.layout, chunkShape, 'lm_head_chunk_f32');

      if (preferF16) {
        const f32Tensor = createTensor(f32Buffer, 'f32', chunkShape, 'lm_head_chunk_f32');
        const f16Tensor = await castF32ToF16(f32Tensor);
        releaseBuffer(f32Buffer);
        weightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', lmHead.layout, chunkShape, 'lm_head_chunk_f16');
      }
    }

    const logitsTensor = await runMatmul(matmulInput, weightBuffer, numTokens, rowCount, hiddenSize, {
      transposeB: 'auto',
      role: lmHeadRole,
      kernelPath,
      outputDtype: lmHeadOutputDtype,
      executionPolicies,
    });

    if (debugProbes?.length || operatorDiagnostics?.enabled) {
      await runProbes('logits', logitsTensor.buffer, {
        numTokens,
        hiddenSize: rowCount,
        probes: debugProbes,
        operatorDiagnostics,
        dtype: logitsTensor.dtype,
      });
    }

    const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsTensor.dtype });
    const chunkLogitsData = await readBufferWithCleanup(
      logitsTensor.buffer,
      numTokens * rowCount * logitsBytes,
      () => {
        releaseBuffer(logitsTensor.buffer);
        releaseBuffer(weightBuffer.buffer);
      }
    );
    const chunkLogits = logitsTensor.dtype === 'f16'
      ? f16BufferToF32(chunkLogitsData)
      : new Float32Array(chunkLogitsData);
    writeChunkLogits(logits, chunkLogits, numTokens, vocabSize, rowOffset, rowCount);
  }

  if (matmulInput !== normedTensor) {
    releaseBuffer(matmulInput.buffer);
  }

  return logits;
}


export async function computeSplitLogitsGPU(
  normedTensor,
  lmHead,
  numTokens,
  hiddenSize,
  vocabSize,
  weightVocabSize,
  debugProbes,
  operatorDiagnostics,
  kernelPath = null,
  executionPolicies = null
) {
  const device = getDevice();
  if (!device) {
    throw new Error('[Logits] GPU device not available for split LM head.');
  }
  if (lmHead.layout !== 'row') {
    throw new Error(`[Logits] split LM head requires row layout, got "${lmHead.layout}".`);
  }

  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const lmHeadRole = resolveLmHeadMatmulRole(phase);
  const lmHeadInputDtype = resolveMatmulStepDtype(lmHeadRole, phase, kernelPath, normedTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = resolveMatmulStepDtype(lmHeadRole, phase, kernelPath, normedTensor.dtype, 'outputDtype');
  const logits = new Float32Array(numTokens * vocabSize);
  let matmulInput = normedTensor;
  let matmulInputOwned = false;

  try {
    matmulInput = lmHeadInputDtype !== normedTensor.dtype
      ? await coerceTensorDtype(normedTensor, lmHeadInputDtype, null, {
        executionPolicies,
        op: 'lm_head',
        transitionDeclaredBy: 'step_precision',
      })
      : normedTensor;
    matmulInputOwned = matmulInput !== normedTensor;

    for (const section of lmHead.sections) {
      if (section.rowStart >= vocabSize) {
        continue;
      }
      if (section.rowStart + section.rowCount > weightVocabSize) {
        throw new Error(
          `[Logits] split LM head section exceeds weight vocab: rowStart=${section.rowStart}, ` +
          `rowCount=${section.rowCount}, weightVocabSize=${weightVocabSize}.`
        );
      }

      const rowCount = Math.min(section.rowCount, vocabSize - section.rowStart);
      const weightBuffer = createWeightBuffer(
        section.buffer,
        lmHead.dtype,
        lmHead.layout,
        [section.rowCount, hiddenSize],
        `${lmHead.label ?? 'lm_head'}:split:${section.rowStart}`
      );
      const logitsTensor = await runMatmul(matmulInput, weightBuffer, numTokens, rowCount, hiddenSize, {
        transposeB: 'auto',
        role: lmHeadRole,
        kernelPath,
        outputDtype: lmHeadOutputDtype,
        executionPolicies,
      });

      if (debugProbes?.length || operatorDiagnostics?.enabled) {
        await runProbes('logits', logitsTensor.buffer, {
          numTokens,
          hiddenSize: rowCount,
          probes: debugProbes,
          operatorDiagnostics,
          dtype: logitsTensor.dtype,
        });
      }

      const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: logitsTensor.dtype });
      const chunkLogitsData = await readBufferWithCleanup(
        logitsTensor.buffer,
        numTokens * rowCount * logitsBytes,
        () => {
          releaseBuffer(logitsTensor.buffer);
        }
      );
      const chunkLogits = logitsTensor.dtype === 'f16'
        ? f16BufferToF32(chunkLogitsData)
        : new Float32Array(chunkLogitsData);
      writeChunkLogits(logits, chunkLogits, numTokens, vocabSize, section.rowStart, rowCount);
    }
  } finally {
    if (matmulInputOwned) {
      releaseBuffer(matmulInput.buffer);
    }
  }

  return logits;
}


export async function computeLogitsGPU(
  hiddenStates,
  numTokens,
  weights,
  config,
  debugFlags,
  operatorDiagnostics = null,
) {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype,
  } = config;
  const { finalNorm, lmHead } = weights;
  const device = getDevice();

  if (!device) {
    return null;
  }
  if (!activationDtype) {
    throw new Error('[Logits] activationDtype is required.');
  }

  if (!finalNorm || !lmHead) {
    log.warn('Pipeline', 'Final norm or LM head not loaded');
    return null;
  }
  if (isCpuWeightBuffer(lmHead) || isSplitWeightBuffer(lmHead)) {
    return null;
  }

  // Get or create input buffer

  let inputBuffer;
  let inputBufferOwned = false;
  let normWeightBuffer;
  let normWeightBufferOwned = false;
  let normInputTensor;
  let normInputOwned = false;
  let normedTensor;
  let finalNormTensor;
  let logitInputTensor;
  let logitInputOwned = false;
  let lmHeadInputTensor;
  let lmHeadInputOwned = false;
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;

  try {
    if (isGpuBufferInstance(hiddenStates)) {
      inputBuffer = hiddenStates;
    } else {
      inputBuffer = acquireBuffer( (hiddenStates).byteLength, undefined, 'logits_input');
      device.queue.writeBuffer(inputBuffer, 0,  (hiddenStates));
      inputBufferOwned = true;
    }

    // Apply final RMSNorm
    const resolvedFinalNorm = resolveFinalNormGpuBuffer(finalNorm, device.queue, 'final_norm_w');
    normWeightBuffer = resolvedFinalNorm.buffer;
    normWeightBufferOwned = resolvedFinalNorm.owned;

    const inputDtype = isGpuBufferInstance(hiddenStates) ? activationDtype : 'f32';
    const inputTensor = createTensor(inputBuffer, inputDtype, [numTokens, hiddenSize], 'logits_input');
    const phase = numTokens === 1 ? 'decode' : 'prefill';
    const kernelPath = config.kernelPath ?? null;
    const finalNormPrecision = getKernelPathStepPrecision('final_norm', 'postLayer', phase, 0, kernelPath);
    const hasExplicitFinalNormPrecision = finalNormPrecision?.inputDtype != null || finalNormPrecision?.outputDtype != null;
    await runProbes('pre_final_norm', inputBuffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes ?? null,
      operatorDiagnostics,
      dtype: inputDtype,
    });
    const forceStableF32Logits = !hasExplicitFinalNormPrecision && shouldForceStableF32Logits(config, inputDtype);
    const stableKernelPath = forceStableF32Logits
      ? createStableF32LogitsKernelPath(kernelPath)
      : kernelPath;
    normInputTensor = inputTensor;
    if (forceStableF32Logits) {
      assertImplicitDtypeTransitionAllowed({
        executionPolicies: config.executionPolicies ?? null,
        fromDtype: inputTensor.dtype,
        toDtype: 'f32',
        op: 'logits_final_norm',
        detail: 'Stable logits mode would widen activations implicitly before final RMSNorm.',
      });
      normInputTensor = await castF16ToF32(inputTensor);
      normInputOwned = true;
    } else {
      const finalNormInputDtype = resolvePostLayerStepDtype('final_norm', phase, stableKernelPath, inputTensor.dtype, 'inputDtype');
      normInputTensor = finalNormInputDtype !== inputTensor.dtype
        ? await coerceTensorDtype(inputTensor, finalNormInputDtype, null, {
          executionPolicies: config.executionPolicies ?? null,
          op: 'final_norm',
          transitionDeclaredBy: 'step_precision',
        })
        : inputTensor;
      normInputOwned = normInputTensor !== inputTensor;
    }
    normedTensor = await runRMSNorm(normInputTensor, normWeightBuffer, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: config.rmsNormWeightOffset,
    });
    finalNormTensor = normedTensor;
    if (!forceStableF32Logits) {
      const finalNormOutputDtype = resolvePostLayerStepDtype(
        'final_norm',
        phase,
        stableKernelPath,
        normedTensor.dtype,
        'outputDtype'
      );
      finalNormTensor = finalNormOutputDtype !== normedTensor.dtype
        ? await coerceTensorDtype(normedTensor, finalNormOutputDtype, null, {
          executionPolicies: config.executionPolicies ?? null,
          op: 'final_norm',
          transitionDeclaredBy: 'step_precision',
        })
        : normedTensor;
    }
    if (finalNormTensor !== normedTensor) {
      releaseBuffer(normedTensor.buffer);
      normedTensor = null;
    }
    await runProbes('final_norm', finalNormTensor.buffer, {
      numTokens,
      hiddenSize,
      probes: config.debugProbes ?? null,
      operatorDiagnostics,
      dtype: finalNormTensor.dtype,
    });
    if (normInputOwned) {
      releaseBuffer(normInputTensor.buffer);
      normInputOwned = false;
    }
    const logitInputScale = resolveLogitInputScale(config);
    logitInputTensor = finalNormTensor;
    if (logitInputScale !== 1) {
      logitInputTensor = await runScale(finalNormTensor, logitInputScale, {
        count: numTokens * hiddenSize,
      });
      logitInputOwned = true;
    }
    const lmHeadRole = resolveLmHeadMatmulRole(phase);
    const lmHeadInputDtype = forceStableF32Logits
      ? logitInputTensor.dtype
      : resolveMatmulStepDtype(lmHeadRole, phase, stableKernelPath, logitInputTensor.dtype, 'inputDtype');
    const lmHeadOutputDtype = forceStableF32Logits
      ? logitInputTensor.dtype
      : resolveMatmulStepDtype(lmHeadRole, phase, stableKernelPath, logitInputTensor.dtype, 'outputDtype');
    lmHeadInputTensor = lmHeadInputDtype !== logitInputTensor.dtype
      ? await coerceTensorDtype(logitInputTensor, lmHeadInputDtype, null, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'lm_head',
        transitionDeclaredBy: 'step_precision',
      })
      : logitInputTensor;
    lmHeadInputOwned = lmHeadInputTensor !== logitInputTensor;

    // Project to vocab via LM head
    if (isGpuBufferInstance(lmHead)) {
      lmHeadBuffer = lmHead;
    } else if (isWeightBuffer(lmHead)) {
      lmHeadBuffer = lmHead;
    } else {
      const rawBuffer = acquireBuffer( (lmHead).byteLength, undefined, 'lm_head_w');
      device.queue.writeBuffer(rawBuffer, 0,  (lmHead));
      lmHeadBuffer = rawBuffer;
      lmHeadBufferOwned = true;
    }

    const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize
      ? embeddingVocabSize
      : vocabSize;

    let logitsTensor = await runMatmul(lmHeadInputTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
      transposeB: 'auto',
      role: lmHeadRole,
      kernelPath: stableKernelPath,
      outputDtype: lmHeadOutputDtype,
      executionPolicies: config.executionPolicies ?? null,
    });
    logitsTensor = await finalizeLogitOutputTensor(logitsTensor, config, {
      numTokens, vocabSize: matmulVocabSize, operatorDiagnostics,
    });

    // Cleanup intermediate buffers (but keep logitsBuffer)
    if (inputBufferOwned) { releaseBuffer(inputBuffer); inputBufferOwned = false; }
    if (lmHeadInputOwned) { releaseBuffer(lmHeadInputTensor.buffer); lmHeadInputOwned = false; }
    if (logitInputOwned) { releaseBuffer(logitInputTensor.buffer); logitInputOwned = false; }
    if (finalNormTensor) {
      releaseBuffer(finalNormTensor.buffer);
      finalNormTensor = null;
      normedTensor = null;
    }
    if (normWeightBufferOwned) { releaseBuffer(normWeightBuffer); normWeightBufferOwned = false; }
    if (lmHeadBufferOwned) { releaseBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer); lmHeadBufferOwned = false; }

    return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize, logitsDtype: logitsTensor.dtype };
  } finally {
    if (inputBufferOwned && inputBuffer) releaseBuffer(inputBuffer);
    if (normInputOwned && normInputTensor) releaseBuffer(normInputTensor.buffer);
    if (lmHeadInputOwned && lmHeadInputTensor) releaseBuffer(lmHeadInputTensor.buffer);
    if (logitInputOwned && logitInputTensor) releaseBuffer(logitInputTensor.buffer);
    if (finalNormTensor) releaseBuffer(finalNormTensor.buffer);
    if (normedTensor) releaseBuffer(normedTensor.buffer);
    if (normWeightBufferOwned && normWeightBuffer) releaseBuffer(normWeightBuffer);
    if (lmHeadBufferOwned && lmHeadBuffer) releaseBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);
  }
}
