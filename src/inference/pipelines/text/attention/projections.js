import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import {
  runMatmul,
  recordMatmul,
  runSplitQKV,
  recordSplitQKV,
  runRMSNorm,
  recordRMSNorm,
} from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
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

function getRmsNormRunner(recorder) {
  if (!recorder) {
    return (input, weight, eps, options) => runRMSNorm(input, weight, eps, options);
  }
  return (input, weight, eps, options) => recordRMSNorm(recorder, input, weight, eps, options);
}

function releaseOwnedWeightBuffer(layerWeight, resolvedWeightBuffer, releaseTemporary) {
  if (layerWeight instanceof GPUBuffer || isWeightBuffer(layerWeight)) {
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
  matmulOutputDtype,
  getWeightBuffer,
  lora,
  releaseTemporary,
}) {
  const runMatmulForMode = getMatmulRunner(recorder);
  const layerWeight = layerWeights?.[weightKey];
  let projected;

  if (layerWeight && getWeightBuffer) {
    const projBuffer = getWeightBuffer(layerWeight, role);
    projected = await runMatmulForMode(normed, projBuffer, numTokens, outputSize, hiddenSize, {
      transposeB: 'auto',
      role,
      layerIdx,
      outputDtype: matmulOutputDtype,
    });
    releaseOwnedWeightBuffer(layerWeight, projBuffer, releaseTemporary);
  } else {
    const fallback = acquireBuffer(numTokens * outputSize * 4, undefined, outputLabel);
    projected = createTensor(fallback, normed.dtype, [numTokens, outputSize], outputLabel);
  }

  const loraModule = getLoRAModule(lora, layerIdx, loraKey);
  if (loraModule && getWeightBuffer) {
    const combined = await applyLoRA(
      normed,
      projected,
      loraModule,
      { M: numTokens, N: outputSize, K: hiddenSize },
      getWeightBuffer,
      recorder ?? undefined
    );
    if (combined.buffer !== projected.buffer) {
      releaseTemporary(projected.buffer);
      projected = combined;
    }
  }

  return projected;
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

export function resolveAttentionProjectionOutputDtype(attentionInputDtype) {
  const useF16Activations = attentionInputDtype === 'f16';
  return selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
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
  matmulOutputDtype,
  getWeightBuffer,
  lora,
  releaseTemporary,
  onFusedQKV = null,
}) {
  const runMatmulForMode = getMatmulRunner(recorder);
  const runSplitForMode = getSplitRunner(recorder);

  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj')
    || getLoRAModule(lora, layerIdx, 'k_proj')
    || getLoRAModule(lora, layerIdx, 'v_proj');
  const useFusedQKV = selectRuleValue('inference', 'attention', 'useFusedQkv', {
    hasQkvProj: Boolean(layerWeights.qkvProj),
    hasQkvSizes: Boolean(layerWeights.qkvSizes),
    hasLoRA: Boolean(hasLoRA),
  });

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    const [qSizeFused, kSizeFused, vSizeFused] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSizeFused + kSizeFused + vSizeFused;
    const qkvTensor = await runMatmulForMode(normed, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
      transposeB: 'auto',
      role: 'qkv_proj',
      layerIdx,
      outputDtype: matmulOutputDtype,
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
    return { qTensor: split.Q, kTensor: split.K, vTensor: split.V, usedFusedQKV: true };
  }

  const qTensor = await projectSingleQkvTensor({
    recorder,
    normed,
    layerWeights,
    weightKey: 'qProj',
    role: 'q_proj',
    outputSize: numHeads * headDim,
    outputLabel: 'Q',
    loraKey: 'q_proj',
    numTokens,
    hiddenSize,
    layerIdx,
    matmulOutputDtype,
    getWeightBuffer,
    lora,
    releaseTemporary,
  });

  const kTensor = await projectSingleQkvTensor({
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
    matmulOutputDtype,
    getWeightBuffer,
    lora,
    releaseTemporary,
  });

  const vTensor = await projectSingleQkvTensor({
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
    matmulOutputDtype,
    getWeightBuffer,
    lora,
    releaseTemporary,
  });

  return { qTensor, kTensor, vTensor, usedFusedQKV: false };
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
