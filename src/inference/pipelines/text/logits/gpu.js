

import { getDevice, getKernelCapabilities } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import { runMatmul, runRMSNorm } from '../../../../gpu/kernel-selector.js';
import { recordMatmul } from '../../../../gpu/kernels/matmul.js';
import { recordRMSNorm } from '../../../../gpu/kernels/rmsnorm.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
} from '../../../../gpu/kernels/cast.js';
import { createWeightBuffer, isWeightBuffer, isCpuWeightBuffer, isGpuBufferInstance } from '../../../../gpu/weight-buffer.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { getKernelPathMatmulPrecision, getKernelPathStepPrecision } from '../../../../config/kernel-path-loader.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { runProbes } from '../probes.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { f16BufferToF32 } from './cpu.js';
import { readBufferWithCleanup } from './utils.js';
import { f16ToF32 } from '../../../../loader/dtype-utils.js';

function shouldForceStableF32Logits(config, inputDtype) {
  if (inputDtype !== 'f16') {
    return false;
  }
  // Softcapped output heads are numerically sensitive in pure F16 on the
  // final RMSNorm + LM-head path. Widen only the logits tail so the main
  // layer stack and KV cache can stay on the faster F16 lane.
  if (Number.isFinite(config.finalLogitSoftcapping) && config.finalLogitSoftcapping > 0) {
    return true;
  }
  // Small Gemma-family checkpoints can also overflow in pure F16 logits path
  // after RMSNorm offset even without output softcapping.
  return config.rmsNormWeightOffset === true
    && Number.isFinite(config.hiddenSize)
    && config.hiddenSize <= 768;
}

function resolvePrecisionFieldDtype(precision, fallback, field) {
  const requested = precision?.[field] ?? fallback;
  if (requested == null) {
    return fallback;
  }
  return selectRuleValue('shared', 'dtype', 'f16OrF32FromDtype', { dtype: requested });
}

function resolveMatmulStepDtype(role, phase, kernelPath, fallback, field) {
  const precision = getKernelPathMatmulPrecision(role, phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

function resolvePostLayerStepDtype(op, phase, kernelPath, fallback, field) {
  const precision = getKernelPathStepPrecision(op, 'postLayer', phase, 0, kernelPath);
  return resolvePrecisionFieldDtype(precision, fallback, field);
}

async function coerceTensorDtype(tensor, targetDtype, recorder = null, options = {}) {
  if (!targetDtype || tensor.dtype === targetDtype) {
    return tensor;
  }
  assertImplicitDtypeTransitionAllowed({
    executionPolicies: options.executionPolicies ?? null,
    fromDtype: tensor.dtype,
    toDtype: targetDtype,
    op: options.op ?? 'logits',
    detail: 'The execution graph must declare this cast explicitly.',
    transitionDeclaredBy: options.transitionDeclaredBy ?? null,
  });
  if (tensor.dtype === 'f32' && targetDtype === 'f16') {
    return recorder ? await recordCastF32ToF16(recorder, tensor) : await castF32ToF16(tensor);
  }
  if (tensor.dtype === 'f16' && targetDtype === 'f32') {
    return recorder ? await recordCastF16ToF32(recorder, tensor) : await castF16ToF32(tensor);
  }
  throw new Error(`Unsupported logits matmul dtype coercion: ${tensor.dtype} -> ${targetDtype}`);
}

const STABLE_F32_LOGITS_KERNEL_MAP = new Map([
  ['matmul_gemv_subgroup_f16a.wgsl', 'matmul_gemv_subgroup.wgsl'],
  ['matmul_f16.wgsl', 'matmul_f16w_f32a.wgsl'],
  ['matmul_f16_tiled.wgsl', 'matmul_f16w_f32a_tiled.wgsl'],
]);

function createStableF32LogitsKernelPath(kernelPath) {
  if (!kernelPath?.postLayer) {
    return kernelPath;
  }
  let changed = false;
  const postLayer = kernelPath.postLayer.map((step) => {
    if (step?.op === 'final_norm') {
      const precision = {
        ...(step.precision ?? {}),
        inputDtype: 'f32',
        outputDtype: 'f32',
      };
      if (
        step.precision?.inputDtype === precision.inputDtype
        && step.precision?.outputDtype === precision.outputDtype
      ) {
        return step;
      }
      changed = true;
      return {
        ...step,
        precision,
      };
    }
    if (step?.op !== 'lm_head' && step?.op !== 'lm_head_prefill') {
      return step;
    }
    const replacement = STABLE_F32_LOGITS_KERNEL_MAP.get(step.kernel) ?? step.kernel;
    const precision = {
      ...(step.precision ?? {}),
      inputDtype: 'f32',
      outputDtype: 'f32',
    };
    if (
      replacement === step.kernel
      && step.precision?.inputDtype === precision.inputDtype
      && step.precision?.outputDtype === precision.outputDtype
    ) {
      return step;
    }
    changed = true;
    return {
      ...step,
      kernel: replacement,
      precision,
    };
  });
  if (!changed) {
    return kernelPath;
  }
  return {
    ...kernelPath,
    postLayer,
  };
}

const bf16ScratchU32 = new Uint32Array(1);
const bf16ScratchF32 = new Float32Array(bf16ScratchU32.buffer);

function bf16ToF32(value) {
  bf16ScratchU32[0] = (value & 0xffff) << 16;
  return bf16ScratchF32[0];
}

function isRangeBackedCpuWeightSource(value) {
  return (
    typeof value === 'object'
    && value !== null
    && value.kind === 'tensor_range_source'
    && typeof value.loadRange === 'function'
  );
}

function normalizeRangeBytes(value, label) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error(
    `[Logits] ${label} returned unsupported byte payload type "${value?.constructor?.name ?? typeof value}".`
  );
}

function decodeChunkIntoOutput(bytes, sourceDtype, output, dstOffset, valueCount) {
  if (sourceDtype === 'f16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, valueCount);
    for (let index = 0; index < valueCount; index += 1) {
      output[dstOffset + index] = f16ToF32(values[index]);
    }
    return;
  }
  if (sourceDtype === 'bf16') {
    const values = new Uint16Array(bytes.buffer, bytes.byteOffset, valueCount);
    for (let index = 0; index < valueCount; index += 1) {
      output[dstOffset + index] = bf16ToF32(values[index]);
    }
    return;
  }
  if (((bytes.byteOffset % 4) === 0) && ((bytes.byteLength % 4) === 0)) {
    output.set(new Float32Array(bytes.buffer, bytes.byteOffset, valueCount), dstOffset);
    return;
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < valueCount; index += 1) {
    output[dstOffset + index] = view.getFloat32(index * 4, true);
  }
}

function readTypedLmHeadChunk(data, layout, hiddenSize, vocabSize, rowOffset, rowCount, sourceDtype) {
  if (data instanceof Float32Array) {
    if (layout === 'row') {
      const start = rowOffset * hiddenSize;
      return data.subarray(start, start + rowCount * hiddenSize);
    }
    const chunk = new Float32Array(hiddenSize * rowCount);
    for (let k = 0; k < hiddenSize; k++) {
      const srcOffset = k * vocabSize + rowOffset;
      const dstOffset = k * rowCount;
      chunk.set(data.subarray(srcOffset, srcOffset + rowCount), dstOffset);
    }
    return chunk;
  }

  if (!(data instanceof Uint16Array)) {
    throw new Error(
      `[Logits] Unsupported CPU LM head chunk source type "${data?.constructor?.name ?? typeof data}".`
    );
  }

  const chunk = new Float32Array(hiddenSize * rowCount);
  if (layout === 'row') {
    const start = rowOffset * hiddenSize;
    for (let index = 0; index < rowCount * hiddenSize; index += 1) {
      const raw = data[start + index];
      chunk[index] = sourceDtype === 'bf16' ? bf16ToF32(raw) : f16ToF32(raw);
    }
    return chunk;
  }

  for (let k = 0; k < hiddenSize; k += 1) {
    const srcOffset = k * vocabSize + rowOffset;
    const dstOffset = k * rowCount;
    for (let index = 0; index < rowCount; index += 1) {
      const raw = data[srcOffset + index];
      chunk[dstOffset + index] = sourceDtype === 'bf16' ? bf16ToF32(raw) : f16ToF32(raw);
    }
  }
  return chunk;
}


export function resolveCpuWeightDims(lmHead) {
  if (lmHead.shape.length !== 2) {
    throw new Error(`[Logits] CPU LM head shape must be 2D, got [${lmHead.shape.join(', ')}]`);
  }
  if (lmHead.layout === 'column') {
    return { hiddenSize: lmHead.shape[0], vocabSize: lmHead.shape[1] };
  }
  return { vocabSize: lmHead.shape[0], hiddenSize: lmHead.shape[1] };
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


export async function extractLmHeadChunk(
  data,
  layout,
  hiddenSize,
  vocabSize,
  rowOffset,
  rowCount,
  sourceDtype = 'f32'
) {
  const normalizedSourceDtype = String(sourceDtype ?? 'f32').toLowerCase();
  if (normalizedSourceDtype !== 'f32' && normalizedSourceDtype !== 'f16' && normalizedSourceDtype !== 'bf16') {
    throw new Error(`[Logits] Unsupported CPU LM head source dtype "${sourceDtype}".`);
  }
  if (!isRangeBackedCpuWeightSource(data)) {
    return readTypedLmHeadChunk(
      data,
      layout,
      hiddenSize,
      vocabSize,
      rowOffset,
      rowCount,
      normalizedSourceDtype
    );
  }

  const bytesPerElement = normalizedSourceDtype === 'f32' ? 4 : 2;
  const chunk = new Float32Array(hiddenSize * rowCount);
  if (layout === 'row') {
    const byteOffset = rowOffset * hiddenSize * bytesPerElement;
    const byteLength = rowCount * hiddenSize * bytesPerElement;
    const bytes = normalizeRangeBytes(
      await data.loadRange(byteOffset, byteLength),
      'CPU LM head range source'
    );
    if (bytes.byteLength !== byteLength) {
      throw new Error(
        `[Logits] CPU LM head range source returned ${bytes.byteLength} bytes, expected ${byteLength}.`
      );
    }
    decodeChunkIntoOutput(bytes, normalizedSourceDtype, chunk, 0, rowCount * hiddenSize);
    return chunk;
  }

  for (let k = 0; k < hiddenSize; k += 1) {
    const byteOffset = (k * vocabSize + rowOffset) * bytesPerElement;
    const byteLength = rowCount * bytesPerElement;
    const bytes = normalizeRangeBytes(
      await data.loadRange(byteOffset, byteLength),
      `CPU LM head range source column ${k}`
    );
    if (bytes.byteLength !== byteLength) {
      throw new Error(
        `[Logits] CPU LM head range source returned ${bytes.byteLength} bytes for column ${k}, expected ${byteLength}.`
      );
    }
    decodeChunkIntoOutput(bytes, normalizedSourceDtype, chunk, k * rowCount, rowCount);
  }
  return chunk;
}


export function writeChunkLogits(
  target,
  chunk,
  numTokens,
  vocabSize,
  rowOffset,
  rowCount
) {
  for (let t = 0; t < numTokens; t++) {
    const srcOffset = t * rowCount;
    const dstOffset = t * vocabSize + rowOffset;
    target.set(chunk.subarray(srcOffset, srcOffset + rowCount), dstOffset);
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

  const chunkRows = resolveLmHeadChunkRows(device, numTokens, hiddenSize, largeWeightConfig);
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const lmHeadInputDtype = resolveMatmulStepDtype('lm_head', phase, kernelPath, normedTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = resolveMatmulStepDtype('lm_head', phase, kernelPath, normedTensor.dtype, 'outputDtype');
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

    const chunkShape = lmHead.layout === 'column'
      ? [hiddenSize, rowCount]
      : [rowCount, hiddenSize];

    let weightBuffer = createWeightBuffer(f32Buffer, 'f32', lmHead.layout, chunkShape, 'lm_head_chunk_f32');

    if (preferF16) {
      const f32Tensor = createTensor(f32Buffer, 'f32', chunkShape, 'lm_head_chunk_f32');
      const f16Tensor = await castF32ToF16(f32Tensor);
      releaseBuffer(f32Buffer);
      weightBuffer = createWeightBuffer(f16Tensor.buffer, 'f16', lmHead.layout, chunkShape, 'lm_head_chunk_f16');
    }

    const logitsTensor = await runMatmul(matmulInput, weightBuffer, numTokens, rowCount, hiddenSize, {
      transposeB: 'auto',
      role: 'lm_head',
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
  if (isCpuWeightBuffer(lmHead)) {
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
    if (isGpuBufferInstance(finalNorm)) {
      normWeightBuffer = finalNorm;
    } else {
      normWeightBuffer = acquireBuffer( (finalNorm).byteLength, undefined, 'final_norm_w');
      device.queue.writeBuffer(normWeightBuffer, 0,  (finalNorm));
      normWeightBufferOwned = true;
    }

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
    const lmHeadInputDtype = forceStableF32Logits
      ? finalNormTensor.dtype
      : resolveMatmulStepDtype('lm_head', phase, stableKernelPath, finalNormTensor.dtype, 'inputDtype');
    const lmHeadOutputDtype = forceStableF32Logits
      ? finalNormTensor.dtype
      : resolveMatmulStepDtype('lm_head', phase, stableKernelPath, finalNormTensor.dtype, 'outputDtype');
    lmHeadInputTensor = lmHeadInputDtype !== finalNormTensor.dtype
      ? await coerceTensorDtype(finalNormTensor, lmHeadInputDtype, null, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'lm_head',
        transitionDeclaredBy: 'step_precision',
      })
      : finalNormTensor;
    lmHeadInputOwned = lmHeadInputTensor !== finalNormTensor;

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

    const logitsTensor = await runMatmul(lmHeadInputTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
      transposeB: 'auto',
      role: 'lm_head',
      kernelPath: stableKernelPath,
      outputDtype: lmHeadOutputDtype,
      executionPolicies: config.executionPolicies ?? null,
    });
    await runProbes('logits', logitsTensor.buffer, {
      numTokens,
      hiddenSize: matmulVocabSize,
      operatorDiagnostics,
      dtype: logitsTensor.dtype,
    });

    // Cleanup intermediate buffers (but keep logitsBuffer)
    if (inputBufferOwned) { releaseBuffer(inputBuffer); inputBufferOwned = false; }
    if (lmHeadInputOwned) { releaseBuffer(lmHeadInputTensor.buffer); lmHeadInputOwned = false; }
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
    if (finalNormTensor) releaseBuffer(finalNormTensor.buffer);
    if (normedTensor) releaseBuffer(normedTensor.buffer);
    if (normWeightBufferOwned && normWeightBuffer) releaseBuffer(normWeightBuffer);
    if (lmHeadBufferOwned && lmHeadBuffer) releaseBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);
  }
}


export async function recordLogitsGPU(
  recorder,
  hiddenStates,
  numTokens,
  weights,
  config,
  operatorDiagnostics = null,
) {
  const {
    hiddenSize,
    vocabSize,
    rmsNormEps,
    useTiedEmbeddings,
    embeddingVocabSize,
    activationDtype = 'f32',
  } = config;
  const { finalNorm, lmHead } = weights;
  const matmulVocabSize = useTiedEmbeddings && embeddingVocabSize ? embeddingVocabSize : vocabSize;

  if (!finalNorm || !lmHead) {
    throw new Error('[recordLogitsGPU] Final norm or LM head not loaded');
  }
  if (isCpuWeightBuffer(lmHead)) {
    throw new Error('[recordLogitsGPU] CPU-resident LM head not supported in recorded path');
  }

  // Get norm weight buffer
  
  let normWeightBuffer;
  let normWeightOwned = false;
  if (isGpuBufferInstance(finalNorm)) {
    normWeightBuffer = finalNorm;
  } else {
    normWeightBuffer = acquireBuffer( (finalNorm).byteLength, undefined, 'final_norm_w');
    recorder.device.queue.writeBuffer(normWeightBuffer, 0,  (finalNorm));
    normWeightOwned = true;
  }

  
  const inputDtype = activationDtype;
  // Wrap input buffer as Tensor for RMSNorm
  const inputTensor = createTensor(hiddenStates, inputDtype, [numTokens, hiddenSize], 'logits_input');
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const kernelPath = config.kernelPath ?? null;
  const finalNormPrecision = getKernelPathStepPrecision('final_norm', 'postLayer', phase, 0, kernelPath);
  const hasExplicitFinalNormPrecision = finalNormPrecision?.inputDtype != null || finalNormPrecision?.outputDtype != null;
  await runProbes('pre_final_norm', hiddenStates, {
    numTokens,
    hiddenSize,
    recorder,
    operatorDiagnostics,
    dtype: inputDtype,
  });
  const forceStableF32Logits = !hasExplicitFinalNormPrecision && shouldForceStableF32Logits(config, inputDtype);
  const stableKernelPath = forceStableF32Logits
    ? createStableF32LogitsKernelPath(kernelPath)
    : kernelPath;
  let normInputTensor = inputTensor;
  let normInputOwned = false;
  if (forceStableF32Logits) {
    assertImplicitDtypeTransitionAllowed({
      executionPolicies: config.executionPolicies ?? null,
      fromDtype: inputTensor.dtype,
      toDtype: 'f32',
      op: 'logits_final_norm',
      detail: 'Stable logits mode would widen activations implicitly before final RMSNorm.',
    });
    normInputTensor = await recordCastF16ToF32(recorder, inputTensor);
    normInputOwned = true;
  } else {
    const finalNormInputDtype = resolvePostLayerStepDtype('final_norm', phase, stableKernelPath, inputTensor.dtype, 'inputDtype');
    normInputTensor = finalNormInputDtype !== inputTensor.dtype
      ? await coerceTensorDtype(inputTensor, finalNormInputDtype, recorder, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : inputTensor;
    normInputOwned = normInputTensor !== inputTensor;
  }
  // Record RMSNorm (no submit)
  const normedTensor = await recordRMSNorm(recorder, normInputTensor, normWeightBuffer, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  });
  let finalNormTensor = normedTensor;
  if (!forceStableF32Logits) {
    const finalNormOutputDtype = resolvePostLayerStepDtype(
      'final_norm',
      phase,
      stableKernelPath,
      normedTensor.dtype,
      'outputDtype'
    );
    finalNormTensor = finalNormOutputDtype !== normedTensor.dtype
      ? await coerceTensorDtype(normedTensor, finalNormOutputDtype, recorder, {
        executionPolicies: config.executionPolicies ?? null,
        op: 'final_norm',
        transitionDeclaredBy: 'step_precision',
      })
      : normedTensor;
  }
  await runProbes('final_norm', finalNormTensor.buffer, {
    numTokens,
    hiddenSize,
    recorder,
    operatorDiagnostics,
    dtype: finalNormTensor.dtype,
  });
  const lmHeadInputDtype = forceStableF32Logits
    ? finalNormTensor.dtype
    : resolveMatmulStepDtype('lm_head', phase, stableKernelPath, finalNormTensor.dtype, 'inputDtype');
  const lmHeadOutputDtype = forceStableF32Logits
    ? finalNormTensor.dtype
    : resolveMatmulStepDtype('lm_head', phase, stableKernelPath, finalNormTensor.dtype, 'outputDtype');
  const lmHeadInputTensor = lmHeadInputDtype !== finalNormTensor.dtype
    ? await coerceTensorDtype(finalNormTensor, lmHeadInputDtype, recorder, {
      executionPolicies: config.executionPolicies ?? null,
      op: 'lm_head',
      transitionDeclaredBy: 'step_precision',
    })
    : finalNormTensor;

  // Get LM head buffer
  
  let lmHeadBuffer;
  let lmHeadBufferOwned = false;
  if (isGpuBufferInstance(lmHead)) {
    lmHeadBuffer = lmHead;
  } else if (isWeightBuffer(lmHead)) {
    lmHeadBuffer = lmHead;
  } else {
    const rawBuffer = acquireBuffer( (lmHead).byteLength, undefined, 'lm_head_w');
    recorder.device.queue.writeBuffer(rawBuffer, 0,  (lmHead));
    lmHeadBuffer = rawBuffer;
    lmHeadBufferOwned = true;
  }

  // Record matmul (no submit)
  const logitsTensor = await recordMatmul(recorder, lmHeadInputTensor, lmHeadBuffer, numTokens, matmulVocabSize, hiddenSize, {
    transposeB: 'auto',
    role: 'lm_head',
    kernelPath: stableKernelPath,
    outputDtype: lmHeadOutputDtype,
    executionPolicies: config.executionPolicies ?? null,
  });
  await runProbes('logits', logitsTensor.buffer, {
    numTokens,
    hiddenSize: matmulVocabSize,
    recorder,
    operatorDiagnostics,
    dtype: logitsTensor.dtype,
  });

  // Track intermediate buffer for cleanup after submit
  const trackedTempBuffers = new Set();
  const trackTempBufferOnce = (buffer) => {
    if (!buffer || trackedTempBuffers.has(buffer)) {
      return;
    }
    trackedTempBuffers.add(buffer);
    recorder.trackTemporaryBuffer(buffer);
  };
  if (finalNormTensor !== normedTensor) {
    trackTempBufferOnce(normedTensor.buffer);
  }
  trackTempBufferOnce(finalNormTensor.buffer);
  if (lmHeadInputTensor !== finalNormTensor) {
    trackTempBufferOnce(lmHeadInputTensor.buffer);
  }
  if (normWeightOwned) {
    recorder.trackTemporaryBuffer(normWeightBuffer);
  }
  if (normInputOwned) {
    recorder.trackTemporaryBuffer(normInputTensor.buffer);
  }
  if (lmHeadBufferOwned) {
    recorder.trackTemporaryBuffer(isWeightBuffer(lmHeadBuffer) ? lmHeadBuffer.buffer : lmHeadBuffer);
  }

  return { logitsBuffer: logitsTensor.buffer, vocabSize: matmulVocabSize, logitsDtype: logitsTensor.dtype };
}
