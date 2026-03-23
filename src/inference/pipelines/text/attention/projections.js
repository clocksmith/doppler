import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { isWeightBuffer, getLayout, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import {
  runMatmul,
  recordMatmul,
  runSplitQKV,
  recordSplitQKV,
  runSplitQG,
  recordSplitQG,
  runRMSNorm,
  recordRMSNorm,
} from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../../config/schema/index.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';

function getMatmulRunner(recorder) {
  if (!recorder) {
    return (input, weight, M, N, K, options) => runMatmul(input, weight, M, N, K, options);
  }
  return (input, weight, M, N, K, options) => recordMatmul(recorder, input, weight, M, N, K, options);
}

function getSplitRunner(recorder) {
  if (!recorder) {
    return (qkvTensor, options) => runSplitQKV(qkvTensor, options);
  }
  return (qkvTensor, options) => recordSplitQKV(recorder, qkvTensor, options);
}

function getSplitQGRunner(recorder) {
  if (!recorder) {
    return (qgTensor, options) => runSplitQG(qgTensor, options);
  }
  return (qgTensor, options) => recordSplitQG(recorder, qgTensor, options);
}

function getRmsNormRunner(recorder) {
  if (!recorder) {
    return (input, weight, eps, options) => runRMSNorm(input, weight, eps, options);
  }
  return (input, weight, eps, options) => recordRMSNorm(recorder, input, weight, eps, options);
}

function releaseOwnedWeightBuffer(layerWeight, resolvedWeightBuffer, releaseTemporary) {
  if ((typeof GPUBuffer !== 'undefined' && layerWeight instanceof GPUBuffer) || isWeightBuffer(layerWeight)) {
    return;
  }
  if (!resolvedWeightBuffer) {
    return;
  }
  const buffer = isWeightBuffer(resolvedWeightBuffer) ? resolvedWeightBuffer.buffer : resolvedWeightBuffer;
  releaseTemporary(buffer);
}

async function projectSingleQkvTensor({
  recorder,
  normed,
  layerWeights,
  weightKey,
  role,
  outputSize,
  outputLabel,
  loraKey,
  numTokens,
  hiddenSize,
  layerIdx,
  kernelPath,
  matmulOutputDtype,
  getWeightBuffer,
  lora,
  matmulDebug,
  releaseTemporary,
}) {
    const runMatmulForMode = getMatmulRunner(recorder);
  const layerWeight = layerWeights?.[weightKey];
  if (!layerWeight) {
    throw new Error(`Attention projection requires ${weightKey}.`);
  }
  if (!getWeightBuffer) {
    throw new Error(`Attention projection requires getWeightBuffer for ${role}.`);
  }

  let projected;
  const projBuffer = getWeightBuffer(layerWeight, role);
  try {
    projected = await runMatmulForMode(normed, projBuffer, numTokens, outputSize, hiddenSize, {
      transposeB: 'auto',
      role,
      layerIdx,
      kernelPath,
      outputDtype: matmulOutputDtype,
      matmulDebug,
    });
  } finally {
    releaseOwnedWeightBuffer(layerWeight, projBuffer, releaseTemporary);
  }

  const loraModule = getLoRAModule(lora, layerIdx, loraKey);
  if (loraModule && getWeightBuffer) {
    try {
      const combined = await applyLoRA(
        normed,
        projected,
        loraModule,
        { M: numTokens, N: outputSize, K: hiddenSize },
        getWeightBuffer,
        recorder ?? undefined,
        { kernelPath }
      );
      if (combined.buffer !== projected.buffer) {
        releaseTemporary(projected.buffer);
        projected = combined;
      }
    } catch (error) {
      if (projected?.buffer) {
        releaseTemporary(projected.buffer);
      }
      throw error;
    }
  }

  return projected;
}

function resolveProjectionOutputSize(layerWeight, hiddenSize) {
  if (!isWeightBuffer(layerWeight) || !Array.isArray(layerWeight.shape) || layerWeight.shape.length < 2) {
    return null;
  }
  const dim0 = Number(layerWeight.shape[0]);
  const dim1 = Number(layerWeight.shape[1]);
  if (!Number.isFinite(dim0) || !Number.isFinite(dim1)) {
    return null;
  }
  if (dim1 === hiddenSize) {
    return Math.trunc(dim0);
  }
  if (dim0 === hiddenSize) {
    return Math.trunc(dim1);
  }
  return null;
}

export function resolveProjectionSliceOffsetBytes(weightBuffer, outputRows, inputCols) {
  const safeRows = Number.isFinite(outputRows) ? Math.max(0, Math.floor(outputRows)) : 0;
  const safeCols = Number.isFinite(inputCols) ? Math.max(0, Math.floor(inputCols)) : 0;
  if (safeRows === 0 || safeCols === 0) {
    return 0;
  }

  const dtype = String(getWeightDtype(weightBuffer) ?? '').toLowerCase();
  if (dtype === 'q4k') {
    const layout = String(getLayout(weightBuffer) ?? 'row').toLowerCase();
    if (layout !== 'row') {
      throw new Error(`resolveProjectionSliceOffsetBytes: unsupported q4k layout "${layout}" for projection slicing.`);
    }
    const blocksPerRow = Math.ceil(safeCols / QK_K);
    const bytesPerRow = blocksPerRow * Q4K_BLOCK_BYTES;
    return safeRows * bytesPerRow;
  }

  if (dtype === 'f16' || dtype === 'bf16') {
    return safeRows * safeCols * 2;
  }
  return safeRows * safeCols * 4;
}

async function projectQueryWithOptionalGate({
  recorder,
  normed,
  layerWeights,
  numTokens,
  numHeads,
  headDim,
  hiddenSize,
  layerIdx,
  kernelPath,
  matmulOutputDtype,
  getWeightBuffer,
  lora,
  matmulDebug,
  releaseTemporary,
  attentionOutputGate,
}) {
  const qSize = numHeads * headDim;
  const qWeight = layerWeights?.qProj;
  const hasGateProjection = attentionOutputGate === true
    && !!qWeight
    && !!getWeightBuffer
    && (resolveProjectionOutputSize(qWeight, hiddenSize) ?? 0) >= (qSize * 2);

  if (!hasGateProjection) {
    const qTensor = await projectSingleQkvTensor({
      recorder,
      normed,
      layerWeights,
      weightKey: 'qProj',
      role: 'q_proj',
      outputSize: qSize,
      outputLabel: 'Q',
      loraKey: 'q_proj',
      numTokens,
      hiddenSize,
      layerIdx,
      kernelPath,
      matmulOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
    });
    return { qTensor, qGateTensor: null };
  }

  // q_proj weights are stored with interleaved head layout: for head h,
  // rows [h*headDim*2 : h*headDim*2+headDim] = Q, rows [h*headDim*2+headDim : (h+1)*headDim*2] = gate.
  // Compute the full 2*qSize matmul, then de-interleave into separate Q and gate tensors.
  const runMatmulForMode = getMatmulRunner(recorder);
  const runSplitQGForMode = getSplitQGRunner(recorder);
  const qWeightBuffer = getWeightBuffer(qWeight, 'q_proj');
  let fullQGTensor = null;
  let qTensor = null;
  let qGateTensor = null;
  try {
    fullQGTensor = await runMatmulForMode(normed, qWeightBuffer, numTokens, qSize * 2, hiddenSize, {
      transposeB: 'auto',
      role: 'q_proj',
      layerIdx,
      kernelPath,
      outputDtype: matmulOutputDtype,
      matmulDebug,
    });

    const split = await runSplitQGForMode(fullQGTensor, {
      numTokens,
      numHeads,
      headDim,
    });
    releaseTemporary(fullQGTensor.buffer);
    fullQGTensor = null;
    qTensor = split.Q;
    qGateTensor = split.G;
  } catch (error) {
    if (fullQGTensor) {
      releaseTemporary(fullQGTensor.buffer);
    }
    if (qTensor) {
      releaseTemporary(qTensor.buffer);
    }
    if (qGateTensor) {
      releaseTemporary(qGateTensor.buffer);
    }
    throw error;
  } finally {
    releaseOwnedWeightBuffer(qWeight, qWeightBuffer, releaseTemporary);
  }

  const loraModule = getLoRAModule(lora, layerIdx, 'q_proj');
  if (loraModule && getWeightBuffer) {
    try {
      const combined = await applyLoRA(
        normed,
        qTensor,
        loraModule,
        { M: numTokens, N: qSize, K: hiddenSize },
        getWeightBuffer,
        recorder ?? undefined,
        { kernelPath }
      );
      if (combined.buffer !== qTensor.buffer) {
        releaseTemporary(qTensor.buffer);
        qTensor = combined;
      }
    } catch (error) {
      if (qTensor?.buffer) {
        releaseTemporary(qTensor.buffer);
      }
      if (qGateTensor?.buffer) {
        releaseTemporary(qGateTensor.buffer);
      }
      throw error;
    }
  }

  return { qTensor, qGateTensor };
}

export function recordAttentionInputs(state, info) {
  if (!state?.stats || !info) return;
  if (!state.stats.attentionInputs) {
    state.stats.attentionInputs = [];
  }
  const exists = state.stats.attentionInputs.some(
    (entry) => entry.phase === info.phase && entry.layerIdx === info.layerIdx
  );
  if (exists) return;
  state.stats.attentionInputs.push(info);
}

export function shouldForceF32AttentionProjectionForRoPE({
  attentionInputDtype,
  headDim,
  rotaryDim = headDim,
  interleaved = false,
  kernelPathIsF16 = false,
}) {
  // When the execution graph specifies f16 matmul kernels for Q/K/V projections,
  // the graph is authoritative. The f16 RoPE kernel handles partial rotation and
  // interleaving at f16 precision. Do not override to f32.
  if (kernelPathIsF16) return false;
  return attentionInputDtype === 'f16'
    && Number.isFinite(headDim)
    && Number.isFinite(rotaryDim)
    && (rotaryDim !== headDim || interleaved === true);
}

export function resolveAttentionProjectionOutputDtype(attentionInputDtype, options = {}) {
  const useF16Activations = attentionInputDtype === 'f16';
  return selectRuleValue('inference', 'dtype', 'attentionProjectionOutputDtype', {
    forceF32: options.forceF32 === true,
    useF16: useF16Activations,
    fallback: attentionInputDtype,
  });
}

export async function projectAttentionQKV({
  recorder = null,
  normed,
  layerWeights,
  numTokens,
  numHeads,
  numKVHeads,
  headDim,
  hiddenSize,
  layerIdx,
  kernelPath,
  matmulOutputDtype,
  getWeightBuffer,
  lora,
  matmulDebug,
  releaseTemporary,
  onFusedQKV = null,
  attentionOutputGate = false,
}) {
  const runMatmulForMode = getMatmulRunner(recorder);
  const runSplitForMode = getSplitRunner(recorder);

  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj')
    || getLoRAModule(lora, layerIdx, 'k_proj')
    || getLoRAModule(lora, layerIdx, 'v_proj');
  const forceSplitQKV = Boolean(matmulDebug?.enabled) && matmulDebug?.forceSplitQKV === true;
  const useFusedQKV = !forceSplitQKV && selectRuleValue('inference', 'attention', 'useFusedQkv', {
    hasQkvProj: Boolean(layerWeights.qkvProj),
    hasQkvSizes: Boolean(layerWeights.qkvSizes),
    hasLoRA: Boolean(hasLoRA),
  });

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    const [qSizeFused, kSizeFused, vSizeFused] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSizeFused + kSizeFused + vSizeFused;
    let qkvTensor = null;
    try {
      qkvTensor = await runMatmulForMode(normed, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
        transposeB: 'auto',
        role: 'qkv_proj',
        layerIdx,
        kernelPath,
        outputDtype: matmulOutputDtype,
        matmulDebug,
      });
      const split = await runSplitForMode(qkvTensor, {
        numTokens,
        qSize: qSizeFused,
        kSize: kSizeFused,
        vSize: vSizeFused,
      });
      releaseTemporary(qkvTensor.buffer);
      if (onFusedQKV) {
        onFusedQKV({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize: qkvSizeTotal });
      }
      return { qTensor: split.Q, qGateTensor: null, kTensor: split.K, vTensor: split.V, usedFusedQKV: true };
    } catch (error) {
      if (qkvTensor) {
        releaseTemporary(qkvTensor.buffer);
      }
      throw error;
    }
  }

  let qTensor = null;
  let qGateTensor = null;
  let kTensor = null;
  let vTensor = null;
  try {
    ({ qTensor, qGateTensor } = await projectQueryWithOptionalGate({
      recorder,
      normed,
      layerWeights,
      numTokens,
      numHeads,
      headDim,
      hiddenSize,
      layerIdx,
      kernelPath,
      matmulOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
      attentionOutputGate,
    }));

    kTensor = await projectSingleQkvTensor({
      recorder,
      normed,
      layerWeights,
      weightKey: 'kProj',
      role: 'k_proj',
      outputSize: numKVHeads * headDim,
      outputLabel: 'K',
      loraKey: 'k_proj',
      numTokens,
      hiddenSize,
      layerIdx,
      kernelPath,
      matmulOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
    });

    vTensor = await projectSingleQkvTensor({
      recorder,
      normed,
      layerWeights,
      weightKey: 'vProj',
      role: 'v_proj',
      outputSize: numKVHeads * headDim,
      outputLabel: 'V',
      loraKey: 'v_proj',
      numTokens,
      hiddenSize,
      layerIdx,
      kernelPath,
      matmulOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
    });

    return { qTensor, qGateTensor, kTensor, vTensor, usedFusedQKV: false };
  } catch (error) {
    for (const tensor of [qTensor, qGateTensor, kTensor, vTensor]) {
      if (tensor?.buffer) {
        releaseTemporary(tensor.buffer);
      }
    }
    throw error;
  }
}

export async function applyAttentionQKNorm({
  recorder = null,
  qTensor,
  kTensor,
  layerWeights,
  getNormWeightBuffer,
  rmsNormEps,
  numTokens,
  numHeads,
  numKVHeads,
  headDim,
  rmsNormWeightOffset = false,
  releaseTemporary,
  onQNormApplied = null,
  onKNormApplied = null,
}) {
  const runRmsNormForMode = getRmsNormRunner(recorder);
  let nextQ = qTensor;
  let nextK = kTensor;

  if (layerWeights.qNorm && getNormWeightBuffer) {
    const qNormBuf = getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
    const qElemsF32 = qNormBuf.size / 4;
    const qElemsF16 = qNormBuf.size / 2;
    const qElems = qElemsF32 === headDim ? qElemsF32 : qElemsF16;
    if (qElems === headDim) {
      const qNormedTensor = await runRmsNormForMode(nextQ, qNormBuf, rmsNormEps, {
        batchSize: numTokens * numHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset,
      });
      releaseTemporary(nextQ.buffer);
      nextQ = qNormedTensor;
      if (onQNormApplied) {
        await onQNormApplied(nextQ);
      }
    }
    if (!(layerWeights.qNorm instanceof GPUBuffer)) {
      releaseTemporary(qNormBuf);
    }
  }

  if (layerWeights.kNorm && getNormWeightBuffer) {
    const kNormBuf = getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
    const kElemsF32 = kNormBuf.size / 4;
    const kElemsF16 = kNormBuf.size / 2;
    const kElems = kElemsF32 === headDim ? kElemsF32 : kElemsF16;
    if (kElems === headDim) {
      const kNormedTensor = await runRmsNormForMode(nextK, kNormBuf, rmsNormEps, {
        batchSize: numTokens * numKVHeads,
        hiddenSize: headDim,
        rmsNormWeightOffset,
      });
      releaseTemporary(nextK.buffer);
      nextK = kNormedTensor;
      if (onKNormApplied) {
        await onKNormApplied(nextK);
      }
    }
    if (!(layerWeights.kNorm instanceof GPUBuffer)) {
      releaseTemporary(kNormBuf);
    }
  }

  return { qTensor: nextQ, kTensor: nextK };
}
