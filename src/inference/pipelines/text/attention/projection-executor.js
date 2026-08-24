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
import { getRmsNormRunner, normBufferMatchesSize, ownsNormBuffer, releaseOwnedNormBuffer } from './projection-plan.js';

export function getMatmulRunner(recorder) {
  if (!recorder) {
    return (input, weight, M, N, K, options) => runMatmul(input, weight, M, N, K, options);
  }
  return (input, weight, M, N, K, options) => recordMatmul(recorder, input, weight, M, N, K, options);
}

export function getSplitQGRunner(recorder) {
  if (!recorder) {
    return (qgTensor, options) => runSplitQG(qgTensor, options);
  }
  return (qgTensor, options) => recordSplitQG(recorder, qgTensor, options);
}

export function getBiasAddRunner(recorder) {
  if (!recorder) {
    return (data, bias, numTokens, dim, options) => runBiasAdd(data, bias, numTokens, dim, options);
  }
  return (data, bias, numTokens, dim, options) => recordBiasAdd(
    recorder,
    data,
    bias,
    numTokens,
    dim,
    options
  );
}

export function releaseOwnedWeightBuffer(layerWeight, resolvedWeightBuffer, releaseTemporary) {
  if (isGpuBufferInstance(layerWeight) || isWeightBuffer(layerWeight)) {
    return;
  }
  if (!resolvedWeightBuffer) {
    return;
  }
  const buffer = isWeightBuffer(resolvedWeightBuffer) ? resolvedWeightBuffer.buffer : resolvedWeightBuffer;
  releaseTemporary(buffer);
}

export async function projectSingleQkvTensor({
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
  executionPolicies = null,
  fusedNormWeight = null,
  fusedNormEps = null,
  fusedNormOffset = false,
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
      executionPolicies,
      normWeight: fusedNormWeight,
      rmsNormEps: fusedNormEps,
      rmsNormOffset: fusedNormOffset,
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

  const biasWeight = layerWeights?.[`${weightKey}Bias`];
  if (biasWeight) {
    const { tensor: biasTensor, owned } = getVectorTensor(
      biasWeight,
      `${role}_bias`,
      outputSize,
      {},
      {}
    );
    try {
      projected = await getBiasAddRunner(recorder)(
        projected,
        biasTensor,
        numTokens,
        outputSize,
        {
          label: `L${layerIdx}.${role}_bias`,
          layerIdx,
          executionPolicies,
        }
      );
    } finally {
      if (owned) releaseTemporary(biasTensor.buffer);
    }
  }

  return projected;
}

export function resolveProjectionOutputSize(layerWeight, hiddenSize) {
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

export async function projectQueryWithOptionalGate({
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
  executionPolicies = null,
  fusedNormWeight = null,
  fusedNormEps = null,
  fusedNormOffset = false,
}) {
  const qSize = numHeads * headDim;
  const qWeight = layerWeights?.qProj;
  const separateGateWeight = attentionOutputGate === true ? layerWeights?.qGateProj : null;
  const hasGateProjection = !separateGateWeight && attentionOutputGate === true
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
      fusedNormWeight,
      fusedNormEps,
      fusedNormOffset,
    });
    if (!separateGateWeight) return { qTensor, qGateTensor: null };
    try {
      const qGateTensor = await projectSeparateAttentionGate({
        runMatmul: getMatmulRunner(recorder), projectionInput: normed, gateWeight: separateGateWeight,
        numTokens, outputSize: qSize, hiddenSize, layerIdx, kernelPath,
        outputDtype: matmulOutputDtype, matmulDebug, executionPolicies,
        fusedNormWeight, fusedNormEps, fusedNormOffset,
      });
      return { qTensor, qGateTensor };
    } catch (error) {
      releaseTemporary(qTensor.buffer);
      throw error;
    }
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
      executionPolicies,
    });

    const loraModule = getLoRAModule(lora, layerIdx, 'q_proj');
    if (loraModule && getWeightBuffer) {
      const combined = await applyLoRA(
        normed,
        fullQGTensor,
        loraModule,
        { M: numTokens, N: qSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder ?? undefined,
        { kernelPath }
      );
      if (combined.buffer !== fullQGTensor.buffer) {
        releaseTemporary(fullQGTensor.buffer);
        fullQGTensor = combined;
      }
    }

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

  return { qTensor, qGateTensor };
}
