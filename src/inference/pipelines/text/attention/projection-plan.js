import { releaseBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBufferInstance, isWeightBuffer, getLayout, getWeightDtype } from '../../../../gpu/weight-buffer.js';
import {
  runMatmul,
  recordMatmul,
  runSplitQKV,
  recordSplitQKV,
  runSplitQG,
  recordSplitQG,
  runRMSNorm,
  recordRMSNorm,
  runLayerNorm,
  recordLayerNorm,
  canUseRMSNormQK,
  runRMSNormQK,
  recordRMSNormQK,
  canUseSplitQKVRMSNormQK,
  runSplitQKVRMSNormQK,
  recordSplitQKVRMSNormQK,
  canUseSplitQKVRMSNormRoPEQK,
  runSplitQKVRMSNormRoPEQK,
  recordSplitQKVRMSNormRoPEQK,
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
  runBiasAdd,
  recordBiasAdd,
} from '../../../../gpu/kernel-selector.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { QK_K, Q4K_BLOCK_BYTES } from '../../../../config/schema/index.js';
import { getKernelPathMatmulPrecision } from '../../../../config/kernel-path-loader.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getQKNormOnesBuffer, getQKNormZerosBuffer } from './types.js';
import { getVectorTensor } from '../weights.js';
import { projectSeparateAttentionGate } from './gate-projection.js';

export function getRmsNormRunner(recorder) {
  if (!recorder) {
    return (input, weight, eps, options) => runRMSNorm(input, weight, eps, options);
  }
  return (input, weight, eps, options) => recordRMSNorm(recorder, input, weight, eps, options);
}

export function getLayerNormRunner(recorder) {
  if (!recorder) {
    return (input, weight, bias, eps, options) => runLayerNorm(input, weight, bias, eps, options);
  }
  return (input, weight, bias, eps, options) => recordLayerNorm(
    recorder,
    input,
    weight,
    bias,
    eps,
    options
  );
}

export function getRmsNormQKRunner(recorder) {
  if (!recorder) {
    return (q, k, qWeight, kWeight, eps, options) => runRMSNormQK(q, k, qWeight, kWeight, eps, options);
  }
  return (q, k, qWeight, kWeight, eps, options) => recordRMSNormQK(recorder, q, k, qWeight, kWeight, eps, options);
}

export function normBufferMatchesSize(buffer, expectedSize, layerWeight = null) {
  // Logical shape proves how many elements a loaded weight represents; it does
  // not prove that this invocation actually resolved a buffer. Reused-KV
  // layers deliberately leave kNormBuf null, so accepting shape alone would
  // dispatch K normalization with a missing binding.
  if (!buffer || !Number.isFinite(buffer.size)) {
    return false;
  }
  if (Array.isArray(layerWeight?.shape)) {
    const logicalElements = layerWeight.shape.reduce((product, dimension) => product * dimension, 1);
    return logicalElements === expectedSize;
  }
  if (ArrayBuffer.isView(layerWeight)) {
    return layerWeight.length === expectedSize;
  }
  const elemsF32 = buffer.size / 4;
  const elemsF16 = buffer.size / 2;
  return elemsF32 === expectedSize || elemsF16 === expectedSize;
}

export function ownsNormBuffer(layerWeight) {
  return layerWeight && !isGpuBufferInstance(layerWeight) && !isWeightBuffer(layerWeight);
}

export function releaseOwnedNormBuffer(buffer, owned, releaseTemporary, releasedBuffers) {
  if (!owned || !buffer || releasedBuffers.has(buffer)) {
    return;
  }
  releasedBuffers.add(buffer);
  releaseTemporary(buffer);
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
  normalizationType = 'rmsnorm',
  normalizationAxis = 'head',
  releaseTemporary,
  onQNormApplied = null,
  onKNormApplied = null,
  skipKNorm = false,
  retainKInput = false,
  allowUnitQKNorm = false,
}) {
  const runRmsNormForMode = getRmsNormRunner(recorder);
  const runRmsNormQKForMode = getRmsNormQKRunner(recorder);
  const runLayerNormForMode = getLayerNormRunner(recorder);
  if (normalizationType !== 'rmsnorm' && normalizationType !== 'layernorm') {
    throw new Error(`Unsupported Q/K normalization type "${normalizationType}".`);
  }
  if (normalizationAxis !== 'head' && normalizationAxis !== 'projection') {
    throw new Error(`Unsupported Q/K normalization axis "${normalizationAxis}".`);
  }
  const qNormSize = normalizationAxis === 'projection' ? numHeads * headDim : headDim;
  const kNormSize = normalizationAxis === 'projection' ? numKVHeads * headDim : headDim;
  const qNormBatchSize = normalizationAxis === 'projection' ? numTokens : numTokens * numHeads;
  const kNormBatchSize = normalizationAxis === 'projection' ? numTokens : numTokens * numKVHeads;
  const effectiveRmsNormWeightOffset = allowUnitQKNorm ? false : rmsNormWeightOffset;
  let nextQ = qTensor;
  let nextK = kTensor;
  let qNormBuf = null;
  let kNormBuf = null;
  const releasedNormBuffers = new Set();

  try {
    const wantsQNorm = (layerWeights.qNorm && getNormWeightBuffer) || allowUnitQKNorm;
    const wantsKNorm = !skipKNorm && ((layerWeights.kNorm && getNormWeightBuffer) || allowUnitQKNorm);

    if (wantsQNorm) {
      qNormBuf = layerWeights.qNorm && getNormWeightBuffer
        ? getNormWeightBuffer(layerWeights.qNorm, 'q_norm')
        : getQKNormOnesBuffer(qNormSize);
    }
    if (wantsKNorm) {
      kNormBuf = layerWeights.kNorm && getNormWeightBuffer
        ? getNormWeightBuffer(layerWeights.kNorm, 'k_norm')
        : getQKNormOnesBuffer(kNormSize);
    }

    const qNormApplies = normBufferMatchesSize(qNormBuf, qNormSize, layerWeights.qNorm);
    const kNormApplies = normBufferMatchesSize(kNormBuf, kNormSize, layerWeights.kNorm);
    if (wantsQNorm && !qNormApplies) {
      throw new Error(
        `Q normalization weight has ${String(qNormBuf?.size ?? 'unknown')} bytes; ` +
        `expected ${qNormSize} f16 or f32 elements.`
      );
    }
    if (wantsKNorm && !kNormApplies) {
      throw new Error(
        `K normalization weight has ${String(kNormBuf?.size ?? 'unknown')} bytes; ` +
        `expected ${kNormSize} f16 or f32 elements.`
      );
    }
    if (
      normalizationType === 'rmsnorm'
      && normalizationAxis === 'head'
      && qNormApplies
      && kNormApplies
      && canUseRMSNormQK(nextQ, nextK, { skipKNorm })
    ) {
      const fused = await runRmsNormQKForMode(nextQ, nextK, qNormBuf, kNormBuf, rmsNormEps, {
        numTokens,
        numHeads,
        numKVHeads,
        headDim,
        rmsNormWeightOffset: effectiveRmsNormWeightOffset,
      });
      releaseTemporary(nextQ.buffer);
      if (!retainKInput) {
        releaseTemporary(nextK.buffer);
      }
      nextQ = fused.q;
      nextK = fused.k;
      if (onQNormApplied) {
        await onQNormApplied(nextQ);
      }
      if (onKNormApplied) {
        await onKNormApplied(nextK);
      }
      return { qTensor: nextQ, kTensor: nextK };
    }

    if (qNormApplies) {
      const qNormedTensor = normalizationType === 'layernorm'
        ? await runLayerNormForMode(
          nextQ,
          qNormBuf,
          getQKNormZerosBuffer(qNormSize),
          rmsNormEps,
          { batchSize: qNormBatchSize, hiddenSize: qNormSize, label: 'q_norm' }
        )
        : await runRmsNormForMode(nextQ, qNormBuf, rmsNormEps, {
          batchSize: qNormBatchSize,
          hiddenSize: qNormSize,
          rmsNormWeightOffset: effectiveRmsNormWeightOffset,
          label: 'q_norm',
        });
      releaseTemporary(nextQ.buffer);
      nextQ = qNormedTensor;
      if (onQNormApplied) {
        await onQNormApplied(nextQ);
      }
    }

    if (kNormApplies) {
      const kNormedTensor = normalizationType === 'layernorm'
        ? await runLayerNormForMode(
          nextK,
          kNormBuf,
          getQKNormZerosBuffer(kNormSize),
          rmsNormEps,
          { batchSize: kNormBatchSize, hiddenSize: kNormSize, label: 'k_norm' }
        )
        : await runRmsNormForMode(nextK, kNormBuf, rmsNormEps, {
          batchSize: kNormBatchSize,
          hiddenSize: kNormSize,
          rmsNormWeightOffset: effectiveRmsNormWeightOffset,
          label: 'k_norm',
        });
      if (!retainKInput) {
        releaseTemporary(nextK.buffer);
      }
      nextK = kNormedTensor;
      if (onKNormApplied) {
        await onKNormApplied(nextK);
      }
    }
    return { qTensor: nextQ, kTensor: nextK };
  } finally {
    releaseOwnedNormBuffer(qNormBuf, ownsNormBuffer(layerWeights?.qNorm), releaseTemporary, releasedNormBuffers);
    releaseOwnedNormBuffer(kNormBuf, ownsNormBuffer(layerWeights?.kNorm), releaseTemporary, releasedNormBuffers);
  }
}
