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
import { getBiasAddRunner, getMatmulRunner, getSplitQGRunner, projectQueryWithOptionalGate, projectSingleQkvTensor } from './projection-executor.js';
export { applyAttentionQKNorm, normBufferMatchesSize } from './projection-plan.js';

function getSplitRunner(recorder) {
  if (!recorder) {
    return (qkvTensor, options) => runSplitQKV(qkvTensor, options);
  }
  return (qkvTensor, options) => recordSplitQKV(recorder, qkvTensor, options);
}

function getSplitQKVRMSNormQKRunner(recorder) {
  if (!recorder) {
    return (qkvTensor, qWeight, kWeight, eps, options) => runSplitQKVRMSNormQK(qkvTensor, qWeight, kWeight, eps, options);
  }
  return (qkvTensor, qWeight, kWeight, eps, options) => recordSplitQKVRMSNormQK(
    recorder,
    qkvTensor,
    qWeight,
    kWeight,
    eps,
    options
  );
}

function getSplitQKVRMSNormRoPEQKRunner(recorder) {
  if (!recorder) {
    return (qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options) => runSplitQKVRMSNormRoPEQK(
      qkvTensor,
      qWeight,
      kWeight,
      freqsCos,
      freqsSin,
      eps,
      options
    );
  }
  return (qkvTensor, qWeight, kWeight, freqsCos, freqsSin, eps, options) => recordSplitQKVRMSNormRoPEQK(
    recorder,
    qkvTensor,
    qWeight,
    kWeight,
    freqsCos,
    freqsSin,
    eps,
    options
  );
}

export function hasAttentionProjectionDiagnostics(state) {
  return hasAttentionStageDiagnostics(state, ['q_proj', 'k_proj', 'v_proj']);
}

export function hasAttentionStageDiagnostics(state, stages) {
  const diagnostics = state?.operatorDiagnostics ?? null;
  if (diagnostics?.enabled || diagnostics?.tsirFixture?.dir) {
    return true;
  }
  const stageSet = new Set(stages);
  const probes = state?.debugProbes;
  return Array.isArray(probes) && probes.some((probe) => stageSet.has(probe?.stage));
}

export function resolveAttentionQKNormState({ config, layerWeights, layerIdx, reusesSharedKV }) {
  const wantsQKNorm = config.queryKeyNorm === true;
  const hasQNorm = !!layerWeights.qNorm;
  const hasKNorm = !!layerWeights.kNorm;
  const qkNormWeightLayers = Array.isArray(config.queryKeyNormWeightLayers)
    ? config.queryKeyNormWeightLayers
    : null;
  const expectsWeightedQKNorm = qkNormWeightLayers
    ? qkNormWeightLayers.includes(layerIdx)
    : true;
  const allowUnitQKNorm = wantsQKNorm && qkNormWeightLayers !== null && !expectsWeightedQKNorm;
  if (wantsQKNorm && allowUnitQKNorm && (hasQNorm || hasKNorm)) {
    throw new Error(
      `Layer ${layerIdx} declares unit-scale Q/K norm but companion weights are present ` +
      `(hasQ=${hasQNorm}, hasK=${hasKNorm}). Check manifest.inference.attention.queryKeyNormWeightLayers.`
    );
  }
  if (wantsQKNorm && expectsWeightedQKNorm && (!hasQNorm || (!hasKNorm && !reusesSharedKV))) {
    throw new Error(
      `Layer ${layerIdx} requested Q/K norm but companion weights are missing ` +
      `(hasQ=${hasQNorm}, hasK=${hasKNorm}). Check manifest.inference.attention.queryKeyNormWeightLayers.`
    );
  }
  return {
    wantsQKNorm,
    hasQNorm,
    hasKNorm,
    allowUnitQKNorm,
    skipKNorm: reusesSharedKV,
    rmsNormWeightOffset: allowUnitQKNorm ? false : config.rmsNormWeightOffset === true,
  };
}

function normalizeProjectionMatmulDtype(value, precisionField = 'dtype') {
  if (value == null || value === '') {
    return null;
  }
  const normalized = String(value).trim().toLowerCase();
  if (normalized !== 'f16' && normalized !== 'f32') {
    throw new Error(
      `[ExecutionV1] attention projection ${precisionField} must be "f16" or "f32"; got "${value}".`
    );
  }
  return normalized;
}

async function coerceProjectionInputTensor(recorder, tensor, targetDtype) {
  if (!targetDtype || tensor.dtype === targetDtype) {
    return tensor;
  }
  if (targetDtype === 'f16') {
    return recorder
      ? recordCastF32ToF16(recorder, tensor)
      : castF32ToF16(tensor);
  }
  return recorder
    ? recordCastF16ToF32(recorder, tensor)
    : castF16ToF32(tensor);
}

export function resolveProjectionMatmulDtype({
  useFusedQKV,
  phase,
  layerIdx,
  kernelPath,
  precisionField,
  fallbackDtype,
}) {
  const roles = useFusedQKV ? ['qkv_proj'] : ['q_proj', 'k_proj', 'v_proj'];
  const explicitInputDtypes = roles
    .map((role) => normalizeProjectionMatmulDtype(
      getKernelPathMatmulPrecision(role, phase, layerIdx, kernelPath)?.[precisionField] ?? null,
      precisionField
    ))
    .filter(Boolean);
  if (explicitInputDtypes.length === 0) {
    return fallbackDtype;
  }
  const [resolvedInputDtype] = explicitInputDtypes;
  if (explicitInputDtypes.some((dtype) => dtype !== resolvedInputDtype)) {
    throw new Error(
      `[ExecutionV1] attention projection steps resolved conflicting ${precisionField} values at layer ${layerIdx}: ` +
      `${explicitInputDtypes.join(', ')}.`
    );
  }
  return resolvedInputDtype;
}

function resolveProjectionInputDtype({ useFusedQKV, phase, layerIdx, kernelPath, fallbackDtype }) {
  return resolveProjectionMatmulDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    precisionField: 'inputDtype',
    fallbackDtype,
  });
}

function resolveProjectionOutputDtype({ useFusedQKV, phase, layerIdx, kernelPath, fallbackDtype }) {
  return resolveProjectionMatmulDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    precisionField: 'outputDtype',
    fallbackDtype,
  });
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
  sharedKTensor = null,
  sharedVTensor = null,
  executionPolicies = null,
  fusedNormWeight = null,
  fusedNormEps = null,
  fusedNormOffset = false,
  qkNormFusion = null,
  qkNormRoPEFusion = null,
}) {
  const runMatmulForMode = getMatmulRunner(recorder);
  const runSplitForMode = getSplitRunner(recorder);
  const runSplitQGForMode = getSplitQGRunner(recorder);
  const runSplitQKVRMSNormQKForMode = getSplitQKVRMSNormQKRunner(recorder);
  const runSplitQKVRMSNormRoPEQKForMode = getSplitQKVRMSNormRoPEQKRunner(recorder);
  const reuseSharedKV = sharedKTensor != null || sharedVTensor != null;
  if (reuseSharedKV && (!sharedKTensor || !sharedVTensor)) {
    throw new Error('projectAttentionQKV requires both sharedKTensor and sharedVTensor when reusing shared KV.');
  }

  const hasLoRA = getLoRAModule(lora, layerIdx, 'q_proj')
    || getLoRAModule(lora, layerIdx, 'k_proj')
    || getLoRAModule(lora, layerIdx, 'v_proj');
  const forceSplitQKV = Boolean(matmulDebug?.enabled) && matmulDebug?.forceSplitQKV === true;
  const useFusedQKV = !reuseSharedKV && !forceSplitQKV && selectRuleValue('inference', 'attention', 'useFusedQkv', {
    hasQkvProj: Boolean(layerWeights.qkvProj),
    hasQkvSizes: Boolean(layerWeights.qkvSizes),
    hasLoRA: Boolean(hasLoRA),
  });
  const phase = numTokens === 1 ? 'decode' : 'prefill';
  const projectionInputDtype = resolveProjectionInputDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    fallbackDtype: normed.dtype,
  });
  const projectionOutputDtype = resolveProjectionOutputDtype({
    useFusedQKV,
    phase,
    layerIdx,
    kernelPath,
    fallbackDtype: matmulOutputDtype,
  });
  let projectionInput = normed;
  let projectionInputOwned = false;
  if (projectionInputDtype && projectionInputDtype !== normed.dtype) {
    projectionInput = await coerceProjectionInputTensor(recorder, normed, projectionInputDtype);
    projectionInputOwned = projectionInput !== normed;
  }

  if (useFusedQKV && layerWeights.qkvProj && layerWeights.qkvSizes) {
    const [qSizeFused, kSizeFused, vSizeFused] = layerWeights.qkvSizes;
    const qkvSizeTotal = qSizeFused + kSizeFused + vSizeFused;
    const qProjectionSize = numHeads * headDim;
    const qProjectionContainsGate = attentionOutputGate === true;
    const hasSeparateGateProjection = qProjectionContainsGate && layerWeights.qGateProj;
    if (hasSeparateGateProjection && qSizeFused !== qProjectionSize) {
      throw new Error(
        `Fused QKV for attention-output-gate layer ${layerIdx} with qGateProj must store ` +
        `Q-only first slice with ${qProjectionSize} columns; got ${qSizeFused}.`
      );
    }
    if (qProjectionContainsGate && !hasSeparateGateProjection && qSizeFused !== qProjectionSize * 2) {
      throw new Error(
        `Fused QKV for attention-output-gate layer ${layerIdx} must store Q+gate first slice ` +
        `with ${qProjectionSize * 2} columns; got ${qSizeFused}.`
      );
    }
    let qkvTensor = null;
    let qNormBuf = null;
    let kNormBuf = null;
    const releasedNormBuffers = new Set();
    try {
      qkvTensor = await runMatmulForMode(projectionInput, layerWeights.qkvProj, numTokens, qkvSizeTotal, hiddenSize, {
        transposeB: 'auto',
        role: 'qkv_proj',
        layerIdx,
        kernelPath,
        outputDtype: projectionOutputDtype,
        matmulDebug,
        executionPolicies,
        // Forward fused-rmsnorm params so the combined QKV matmul runs the
        // input_norm prologue internally, eliminating the standalone rmsnorm
        // dispatch upstream for layers using the useFusedQKV path.
        normWeight: fusedNormWeight,
        rmsNormEps: fusedNormEps,
        rmsNormOffset: fusedNormOffset,
      });
      if (layerWeights.qkvProjBias) {
        const { tensor: qkvBiasTensor, owned: qkvBiasOwned } = getVectorTensor(
          layerWeights.qkvProjBias,
          `L${layerIdx}.qkv_proj_bias`,
          qkvSizeTotal,
          {},
          {}
        );
        try {
          qkvTensor = await getBiasAddRunner(recorder)(
            qkvTensor,
            qkvBiasTensor,
            numTokens,
            qkvSizeTotal,
            {
              label: `L${layerIdx}.qkv_proj_bias`,
              layerIdx,
              executionPolicies,
            }
          );
        } finally {
          if (qkvBiasOwned) releaseTemporary(qkvBiasTensor.buffer);
        }
      }
      const qkNormRoPEOptions = qkNormRoPEFusion
        ? { ...qkNormRoPEFusion, headDim }
        : null;
      const canFuseSplitQKNormAndRoPE = qkNormRoPEFusion?.enabled === true
        && (!qProjectionContainsGate || hasSeparateGateProjection)
        && qkNormRoPEFusion.projectionDiagnosticsEnabled !== true
        && qkNormRoPEFusion.skipKNorm !== true
        && qkNormRoPEFusion.allowUnitQKNorm !== true
        && layerWeights.qNorm
        && layerWeights.kNorm
        && qkNormRoPEFusion.getNormWeightBuffer
        && qkNormRoPEFusion.freqsCos
        && qkNormRoPEFusion.freqsSin
        && canUseSplitQKVRMSNormRoPEQK(qkvTensor, qkNormRoPEOptions);
      if (canFuseSplitQKNormAndRoPE) {
        qNormBuf = qkNormRoPEFusion.getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
        kNormBuf = qkNormRoPEFusion.getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
        const qNormApplies = normBufferMatchesSize(qNormBuf, headDim, layerWeights.qNorm);
        const kNormApplies = normBufferMatchesSize(kNormBuf, headDim, layerWeights.kNorm);
        if (qNormApplies && kNormApplies) {
          const fused = await runSplitQKVRMSNormRoPEQKForMode(
            qkvTensor,
            qNormBuf,
            kNormBuf,
            qkNormRoPEFusion.freqsCos,
            qkNormRoPEFusion.freqsSin,
            qkNormRoPEFusion.rmsNormEps,
            {
              numTokens,
              numHeads,
              numKVHeads,
              headDim,
              qSize: qSizeFused,
              kSize: kSizeFused,
              vSize: vSizeFused,
              startPos: qkNormRoPEFusion.startPos,
              rotaryDim: qkNormRoPEFusion.rotaryDim,
              pairSpanDim: qkNormRoPEFusion.pairSpanDim,
              interleaved: qkNormRoPEFusion.interleaved,
              rmsNormWeightOffset: qkNormRoPEFusion.rmsNormWeightOffset === true,
              f16KVCacheWrite: qkNormRoPEFusion.f16KVCacheWrite ?? null,
            }
          );
          releaseTemporary(qkvTensor.buffer);
          qkvTensor = null;
          let qGateTensor = null;
          if (hasSeparateGateProjection) {
            try {
              qGateTensor = await projectSeparateAttentionGate({
                runMatmul: runMatmulForMode,
                projectionInput,
                gateWeight: layerWeights.qGateProj,
                numTokens,
                outputSize: qProjectionSize,
                hiddenSize,
                layerIdx,
                kernelPath,
                outputDtype: projectionOutputDtype,
                matmulDebug,
                executionPolicies,
                fusedNormWeight,
                fusedNormEps,
                fusedNormOffset,
              });
            } catch (error) {
              if (fused.Q?.buffer) {
                releaseTemporary(fused.Q.buffer);
              }
              if (fused.K?.buffer) {
                releaseTemporary(fused.K.buffer);
              }
              if (fused.V?.buffer) {
                releaseTemporary(fused.V.buffer);
              }
              throw error;
            }
          }
          if (onFusedQKV) {
            onFusedQKV({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize: qkvSizeTotal });
          }
          return {
            qTensor: fused.Q,
            qGateTensor,
            kTensor: fused.K,
            vTensor: fused.V,
            usedFusedQKV: true,
            valueAliasesKey: false,
            qkNormApplied: true,
            ropeApplied: true,
            kvCacheWriteFused: fused.wroteF16KVCache === true,
          };
        }
      }
      const canFuseSplitAndQKNorm = qkNormFusion?.enabled === true
        && (!qProjectionContainsGate || hasSeparateGateProjection)
        && qkNormFusion.projectionDiagnosticsEnabled !== true
        && qkNormFusion.skipKNorm !== true
        && qkNormFusion.allowUnitQKNorm !== true
        && layerWeights.qNorm
        && layerWeights.kNorm
        && qkNormFusion.getNormWeightBuffer
        && canUseSplitQKVRMSNormQK(qkvTensor, qkNormFusion);
      if (canFuseSplitAndQKNorm) {
        qNormBuf = qkNormFusion.getNormWeightBuffer(layerWeights.qNorm, 'q_norm');
        kNormBuf = qkNormFusion.getNormWeightBuffer(layerWeights.kNorm, 'k_norm');
        const qNormApplies = normBufferMatchesSize(qNormBuf, headDim, layerWeights.qNorm);
        const kNormApplies = normBufferMatchesSize(kNormBuf, headDim, layerWeights.kNorm);
        if (qNormApplies && kNormApplies) {
          const fused = await runSplitQKVRMSNormQKForMode(
            qkvTensor,
            qNormBuf,
            kNormBuf,
            qkNormFusion.rmsNormEps,
            {
              numTokens,
              numHeads,
              numKVHeads,
              headDim,
              qSize: qSizeFused,
              kSize: kSizeFused,
              vSize: vSizeFused,
              rmsNormWeightOffset: qkNormFusion.rmsNormWeightOffset === true,
            }
          );
          releaseTemporary(qkvTensor.buffer);
          qkvTensor = null;
          let qGateTensor = null;
          if (hasSeparateGateProjection) {
            try {
              qGateTensor = await projectSeparateAttentionGate({
                runMatmul: runMatmulForMode,
                projectionInput,
                gateWeight: layerWeights.qGateProj,
                numTokens,
                outputSize: qProjectionSize,
                hiddenSize,
                layerIdx,
                kernelPath,
                outputDtype: projectionOutputDtype,
                matmulDebug,
                executionPolicies,
                fusedNormWeight,
                fusedNormEps,
                fusedNormOffset,
              });
            } catch (error) {
              if (fused.Q?.buffer) {
                releaseTemporary(fused.Q.buffer);
              }
              if (fused.K?.buffer) {
                releaseTemporary(fused.K.buffer);
              }
              if (fused.V?.buffer) {
                releaseTemporary(fused.V.buffer);
              }
              throw error;
            }
          }
          if (onFusedQKV) {
            onFusedQKV({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize: qkvSizeTotal });
          }
          return {
            qTensor: fused.Q,
            qGateTensor,
            kTensor: fused.K,
            vTensor: fused.V,
            usedFusedQKV: true,
            valueAliasesKey: false,
            qkNormApplied: true,
            ropeApplied: false,
            kvCacheWriteFused: false,
          };
        }
      }
      const split = await runSplitForMode(qkvTensor, {
        numTokens,
        qSize: qSizeFused,
        kSize: kSizeFused,
        vSize: vSizeFused,
      });
      releaseTemporary(qkvTensor.buffer);
      qkvTensor = null;
      let qTensor = split.Q;
      let qGateTensor = null;
      if (hasSeparateGateProjection) {
        try {
          qGateTensor = await projectSeparateAttentionGate({
            runMatmul: runMatmulForMode,
            projectionInput,
            gateWeight: layerWeights.qGateProj,
            numTokens,
            outputSize: qProjectionSize,
            hiddenSize,
            layerIdx,
            kernelPath,
            outputDtype: projectionOutputDtype,
            matmulDebug,
            executionPolicies,
            fusedNormWeight,
            fusedNormEps,
            fusedNormOffset,
          });
        } catch (error) {
          if (split.Q?.buffer) {
            releaseTemporary(split.Q.buffer);
          }
          if (split.K?.buffer) {
            releaseTemporary(split.K.buffer);
          }
          if (split.V?.buffer) {
            releaseTemporary(split.V.buffer);
          }
          throw error;
        }
      } else if (qProjectionContainsGate) {
        const qgTensor = split.Q;
        try {
          const qg = await runSplitQGForMode(qgTensor, {
            numTokens,
            numHeads,
            headDim,
          });
          qTensor = qg.Q;
          qGateTensor = qg.G;
          releaseTemporary(qgTensor.buffer);
        } catch (error) {
          releaseTemporary(qgTensor.buffer);
          if (split.K?.buffer) {
            releaseTemporary(split.K.buffer);
          }
          if (split.V?.buffer) {
            releaseTemporary(split.V.buffer);
          }
          throw error;
        }
      }
      if (onFusedQKV) {
        onFusedQKV({ qSize: qSizeFused, kSize: kSizeFused, vSize: vSizeFused, totalSize: qkvSizeTotal });
      }
      return {
        qTensor,
        qGateTensor,
        kTensor: split.K,
        vTensor: split.V,
        usedFusedQKV: true,
        valueAliasesKey: false,
        qkNormApplied: false,
        ropeApplied: false,
        kvCacheWriteFused: false,
      };
    } catch (error) {
      if (qkvTensor) {
        releaseTemporary(qkvTensor.buffer);
      }
      throw error;
    } finally {
      releaseOwnedNormBuffer(qNormBuf, ownsNormBuffer(layerWeights.qNorm), releaseTemporary, releasedNormBuffers);
      releaseOwnedNormBuffer(kNormBuf, ownsNormBuffer(layerWeights.kNorm), releaseTemporary, releasedNormBuffers);
      if (projectionInputOwned) {
        releaseTemporary(projectionInput.buffer);
      }
    }
  }

  let qTensor = null;
  let qGateTensor = null;
  let kTensor = null;
  let vTensor = null;
  try {
    ({ qTensor, qGateTensor } = await projectQueryWithOptionalGate({
      recorder,
      normed: projectionInput,
      layerWeights,
      numTokens,
      numHeads,
      headDim,
      hiddenSize,
      layerIdx,
      kernelPath,
      matmulOutputDtype: projectionOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
      attentionOutputGate,
      executionPolicies,
      fusedNormWeight,
      fusedNormEps,
      fusedNormOffset,
    }));

    if (reuseSharedKV) {
      return {
        qTensor,
        qGateTensor,
        kTensor: sharedKTensor,
        vTensor: sharedVTensor,
        usedFusedQKV: false,
        valueAliasesKey: false,
        qkNormApplied: false,
        ropeApplied: false,
        kvCacheWriteFused: false,
      };
    }

    kTensor = await projectSingleQkvTensor({
      recorder,
      normed: projectionInput,
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
      matmulOutputDtype: projectionOutputDtype,
      getWeightBuffer,
      lora,
      matmulDebug,
      releaseTemporary,
      executionPolicies,
      fusedNormWeight,
      fusedNormEps,
      fusedNormOffset,
    });

    let valueAliasesKey = false;
    if (layerWeights.vProj) {
      vTensor = await projectSingleQkvTensor({
        recorder,
        normed: projectionInput,
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
        matmulOutputDtype: projectionOutputDtype,
        getWeightBuffer,
        lora,
        matmulDebug,
        releaseTemporary,
        executionPolicies,
        fusedNormWeight,
        fusedNormEps,
        fusedNormOffset,
      });
    } else {
      vTensor = kTensor;
      valueAliasesKey = true;
    }

    return {
      qTensor,
      qGateTensor,
      kTensor,
      vTensor,
      usedFusedQKV: false,
      valueAliasesKey,
      qkNormApplied: false,
      ropeApplied: false,
      kvCacheWriteFused: false,
    };
  } catch (error) {
    for (const tensor of [qTensor, qGateTensor]) {
      if (tensor?.buffer) {
        releaseTemporary(tensor.buffer);
      }
    }
    for (const tensor of [kTensor, vTensor]) {
      if (tensor?.buffer && tensor !== sharedKTensor && tensor !== sharedVTensor) {
        releaseTemporary(tensor.buffer);
      }
    }
    throw error;
  } finally {
    if (projectionInputOwned) {
      releaseTemporary(projectionInput.buffer);
    }
  }
}

export async function applyAttentionValueNorm({
  recorder = null,
  vTensor,
  rmsNormEps,
  numTokens,
  numKVHeads,
  headDim,
  releaseTemporary,
  onVNormApplied = null,
}) {
  const runRmsNormForMode = getRmsNormRunner(recorder);
  const vNormBuf = getQKNormOnesBuffer(headDim);
  const nextV = await runRmsNormForMode(vTensor, vNormBuf, rmsNormEps, {
    batchSize: numTokens * numKVHeads,
    hiddenSize: headDim,
    rmsNormWeightOffset: false,
  });
  releaseTemporary(vTensor.buffer);
  if (onVNormApplied) {
    await onVNormApplied(nextV);
  }
  return nextV;
}
