

import {
  doCast,
  doLayerNorm,
  doRMSNorm,
  doRMSNormStats,
  doResidualAdd,
  doResidualNextRMSNormPair,
  releaseOrTrack,
} from '../ops.js';
import { getNormWeightBuffer, getVectorTensor } from '../weights.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal } from './types.js';
import { runDenseFFNGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';
import { resolveQwenNormRouter } from './qwen-shared-expert.js';
import { acquireBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import { runScale, recordScale } from '../../../../gpu/kernels/scale.js';
import { isGpuBufferInstance, isWeightBuffer, requireWeightDtype } from '../../../../gpu/weight-buffer.js';
import { shouldDebugLayerOutput } from '../debug-utils/config.js';
import { decodeReadback } from '../debug-utils/readback-decoding.js';
import { getLogitsHealth } from '../debug-utils/health-metrics.js';
import { trace, isTraceEnabled } from '../../../../debug/index.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { createTensor, tensorElementBytes } from '../../../../gpu/tensor.js';

function shouldUsePostAttnNormFusedGateUpPrelude({ session, postAttn, fusedResidualInput, numTokens, context }) {
  if (postAttn.precision !== undefined || fusedResidualInput?.precision !== undefined) return false;
  if (context.config.normalizationType !== 'rmsnorm') return false;
  if (session?.usePostAttnNormFusedGateUp !== true) return false;
  if (numTokens !== 1) return false;
  if (postAttn?.dtype !== 'f32' || fusedResidualInput?.dtype !== 'f32') return false;
  if (context.debugProbes?.length) return false;
  if (context.debugCheckBuffer) return false;
  if (isTraceEnabled('logits')) return false;
  return true;
}

async function debugFFNBuffer(context, layerIdx, label, tensor, numTokens, hiddenSize) {
  if (!context.debugCheckBuffer) return;
  if (!isGpuBufferInstance(tensor?.buffer)) return;
  if (!shouldDebugLayerOutput(layerIdx, context.debugLayers)) return;
  await context.debugCheckBuffer(tensor.buffer, `L${layerIdx} ${label} (GPU)`, numTokens, hiddenSize, tensor);
}

function enqueueRecordedFFNHealth(context, layerIdx, label, tensor, elementCount) {
  const recorder = context.recorder ?? null;
  if (!recorder || !isTraceEnabled('logits') || !shouldDebugLayerOutput(layerIdx, context.debugLayers)) {
    return;
  }
  if (!tensor?.buffer || !Number.isFinite(elementCount) || elementCount <= 0) {
    return;
  }
  const dtype = tensor.dtype;
  const bytesPerElement = tensorElementBytes(dtype, tensor);
  recorder.enqueueCompletionTask(async () => {
    const data = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
    trace.logits(`L${layerIdx}.${label}_HEALTH`, getLogitsHealth(decodeReadback(data, dtype, tensor)));
  });
}


export async function processFFNStandard(
  layerIdx,
  postAttn,
  numTokens,
  size,
  context,
  layerWeights,
  fusedResidualInput,
  finalOutputScale = 1,
  residualBranchScale = 1,
  nextInputNormFusion = null
) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;
  const requestedFinalOutputScale = finalOutputScale == null ? 1 : Number(finalOutputScale);
  if (!Number.isFinite(requestedFinalOutputScale)) {
    throw new Error(`Layer ${layerIdx} finalOutputScale must be finite; got "${String(finalOutputScale)}".`);
  }
  const requestedResidualBranchScale = Number(residualBranchScale);
  if (!Number.isFinite(requestedResidualBranchScale) || requestedResidualBranchScale <= 0) {
    throw new Error(
      `Layer ${layerIdx} residualBranchScale must be a positive finite number; ` +
      `got "${String(residualBranchScale)}".`
    );
  }
  context.__layerScalarFusedFired = false;

  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;

  const useLegacyMoeOnlyFFN = config.ffnBranchMode !== 'dense'
    && config.ffnBranchMode !== 'dense_plus_moe'
    && config.useMoE
    && isMoELayerLocal(layerIdx, config, layerWeights);
  // A declared moe_norm_router step moves this norm into the MoE's router dispatch.
  const normRouterDeclared = useLegacyMoeOnlyFFN
    && resolveQwenNormRouter({ kernelPath: context.kernelPath ?? null, phase: context.phase, layerIdx });
  let pendingNorm = null;

  // 1. Post-attention norm (optionally fuses upstream residual add via PRE_RESIDUAL)
  let normedTensor = postAttn;
  let prenormSumBuffer = null;
  let postAttnNormStats = null;
  let postAttnNormWeightBuf = null;
  let releasePostAttnNormWeight = false;
  let usedPostAttnNormStatsPrelude = false;
  if (layerWeights?.postAttnNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttnNorm, 'post_attn_norm', weightConfig, debugFlags);
    postAttnNormWeightBuf = normWeightBuf;
    releasePostAttnNormWeight = !isGpuBufferInstance(layerWeights.postAttnNorm) && !isWeightBuffer(layerWeights.postAttnNorm);
    const session = getRuntimeConfig().inference?.session;

    if (config.normalizationType === 'layernorm') {
      if (!layerWeights.postAttentionNormBias) {
        throw new Error(`Layer ${layerIdx} requires post_attention_layernorm.bias for LayerNorm execution.`);
      }
      let normInput = postAttn;
      if (fusedResidualInput) {
        normInput = await doResidualAdd(postAttn, fusedResidualInput, size, recorder, {
          label: `L${layerIdx}.attention_residual`,
          layerIdx,
          executionPolicies: context.executionPolicies ?? null,
        });
        prenormSumBuffer = normInput.buffer;
      }
      const { tensor: normBiasTensor, owned: normBiasOwned } = getVectorTensor(
        layerWeights.postAttentionNormBias,
        `L${layerIdx}.post_attn_norm_bias`,
        hiddenSize,
        weightConfig,
        debugFlags
      );
      try {
        normedTensor = await doLayerNorm(
          normInput,
          normWeightBuf,
          normBiasTensor.buffer,
          rmsNormEps,
          {
            batchSize: numTokens,
            hiddenSize,
            label: `L${layerIdx}.post_attn_norm`,
            layerIdx,
            normWeightDtype: requireWeightDtype(normWeightBuf, `L${layerIdx}.post_attn_norm`),
          },
          recorder
        );
      } finally {
        if (normBiasOwned) releaseOrTrack(recorder, normBiasTensor.buffer, decodeBuffers);
      }
    } else if (fusedResidualInput) {
      const bytesPerElement = tensorElementBytes(postAttn.dtype, postAttn);
      prenormSumBuffer = acquireBuffer(size * bytesPerElement, undefined, 'fused_prenorm_sum');
      if (normRouterDeclared) {
        if (numTokens !== 1 || postAttn.dtype !== 'f32' || postAttn.precision !== undefined
          || fusedResidualInput.dtype !== 'f32' || fusedResidualInput.precision !== undefined
          || shouldUsePostAttnNormFusedGateUpPrelude({ session, postAttn, fusedResidualInput, numTokens, context })) {
          throw new Error(`Layer ${layerIdx} declares moe_norm_router, which covers single-token scalar F32 `
            + 'post-attention norms without the fused gate/up prelude.');
        }
        // The MoE's norm + router dispatch writes the normalized row and pre-norm sum.
        normedTensor = createTensor(acquireBuffer(size * 4, undefined, 'post_attn_norm_output'), 'f32',
          postAttn.shape, 'post_attn_norm_output');
        pendingNorm = {
          input: postAttn, residual: fusedResidualInput, weight: normWeightBuf,
          rmsNormWeightOffset: Boolean(weightConfig.rmsNormWeightOffset), eps: rmsNormEps, prenorm: prenormSumBuffer,
        };
      } else if (shouldUsePostAttnNormFusedGateUpPrelude({ session, postAttn, fusedResidualInput, numTokens, context })) {
        postAttnNormStats = await doRMSNormStats(postAttn, fusedResidualInput, rmsNormEps, {
          batchSize: numTokens,
          hiddenSize,
          outputBuffer: prenormSumBuffer,
          label: `L${layerIdx}.post_attn_norm_stats`,
        }, recorder);
        normedTensor = postAttnNormStats.prenormSum;
        usedPostAttnNormStatsPrelude = true;
      } else {
        normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
          kernelPath: context.kernelPath, phase: context.phase, role: 'post_attn_norm', section: 'layer',
          batchSize: numTokens,
          hiddenSize,
          preResidual: fusedResidualInput,
          residualSumOutput: prenormSumBuffer,
          label: `L${layerIdx}.post_attn_norm`,
          layerIdx,
          rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
        }, recorder);
      }
    } else {
      normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
        kernelPath: context.kernelPath, phase: context.phase, role: 'post_attn_norm', section: 'layer',
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
    }
  }
  if (normRouterDeclared && !pendingNorm) {
    throw new Error(`Layer ${layerIdx} declares moe_norm_router, which needs an RMS post-attention norm `
      + 'with the folded attention residual.');
  }
  const probeFfnInput = async () => {
    await runProbes('ffn_in', normedTensor.buffer, {
      layerIdx,
      numTokens,
      hiddenSize,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: normedTensor.dtype,
      precision: normedTensor.precision, components: normedTensor.components,
      elementStrideBytes: normedTensor.elementStrideBytes,
    });
    await debugFFNBuffer(context, layerIdx, 'FFN input', normedTensor, numTokens, hiddenSize);
    enqueueRecordedFFNHealth(context, layerIdx, 'ffn_in', normedTensor, numTokens * hiddenSize);
  };
  // A deferred norm is written inside the FFN; its probes follow the FFN in stream order.
  if (!usedPostAttnNormStatsPrelude && !pendingNorm) await probeFfnInput();

  // 2. FFN
  // Stage the residual tensor so that runDenseFFNGPU's ffn_down matmul can
  // opportunistically fuse with it (WideTile+residual kernel). The flag
  // `context.__ffnResidualFusedFired` will be set to true by dense.js if
  // the fusion actually fired; we read it below to skip the separate add.
  // If the pre-norm residual sum was captured (fused RMSNorm path), use that
  // buffer directly — it IS the correct residual for ffn_residual because
  // ffn_residual adds to the pre-rmsnorm tensor.
  const pendingResidualForFfn = prenormSumBuffer
    ? createTensor(prenormSumBuffer, postAttn.dtype, postAttn.shape, 'ffn_prenorm_sum', postAttn)
    : postAttn;
  context.__pendingFfnResidualTensor = requestedResidualBranchScale === 1
    ? pendingResidualForFfn
    : null;
  // Where a consumer that adds the residual writes the layer output: the same decode
  // buffer the separate residual add below would write.
  context.__pendingFfnResidualOutputBuffer = context.__pendingFfnResidualTensor ? decodeOutputBuffer : null;
  context.__pendingFfnInputNormStats = postAttnNormStats
    ? {
      invRmsBuffer: postAttnNormStats.invRmsBuffer,
      normWeight: postAttnNormWeightBuf,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }
    : null;
  context.__ffnResidualFusedFired = false;

  let ffnOutput;
  context.__pendingFfnNorm = pendingNorm;
  context.__ffnNormFusedFired = false;
  try {
    if (useLegacyMoeOnlyFFN) {
      ffnOutput = await runMoEFFNGPU(layerIdx, normedTensor, numTokens, context);
    } else {
      ffnOutput = await runDenseFFNGPU(layerIdx, normedTensor, numTokens, context, layerWeights);
    }
  } finally {
    context.__pendingFfnResidualTensor = null;
    context.__pendingFfnResidualOutputBuffer = null;
    context.__pendingFfnInputNormStats = null;
    context.__pendingFfnNorm = null;
  }
  if (pendingNorm && context.__ffnNormFusedFired !== true) {
    throw new Error(`Layer ${layerIdx} handed its post-attention norm to the MoE router, which did not run it.`);
  }
  context.__ffnNormFusedFired = false;
  if (pendingNorm) await probeFfnInput();
  const ffnResidualFused = context.__ffnResidualFusedFired === true;
  context.__ffnResidualFusedFired = false;
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: ffnOutput.dtype,
    precision: ffnOutput.precision, components: ffnOutput.components,
    elementStrideBytes: ffnOutput.elementStrideBytes,
  });
  await debugFFNBuffer(context, layerIdx, 'FFN output', ffnOutput, numTokens, hiddenSize);
  enqueueRecordedFFNHealth(context, layerIdx, 'ffn_out', ffnOutput, numTokens * hiddenSize);

  // 3. Residual add (uses prenorm sum when fused, otherwise postAttn)
  // When WideTile+residual fusion fired inside runDenseFFNGPU, ffnOutput
  // IS already post-residual — skip the separate add entirely.
  let output;
  let residualInput = ffnOutput;
  let residualInputOwned = false;
  if (ffnResidualFused) {
    output = ffnOutput;
  } else {
    const residualTensor = prenormSumBuffer
      ? createTensor(prenormSumBuffer, postAttn.dtype, postAttn.shape, 'ffn_prenorm_sum', postAttn)
      : postAttn;
    residualInputOwned = ffnOutput.dtype !== residualTensor.dtype;
    if (residualInputOwned) {
      residualInput = await doCast(ffnOutput, residualTensor.dtype, recorder);
    }
    if (requestedResidualBranchScale !== 1) {
      const unscaledResidualInput = residualInput;
      residualInput = recorder
        ? await recordScale(recorder, unscaledResidualInput, requestedResidualBranchScale, { count: size })
        : await runScale(unscaledResidualInput, requestedResidualBranchScale, { count: size });
      if (residualInputOwned) {
        releaseOrTrack(recorder, unscaledResidualInput.buffer, decodeBuffers);
      } else if (unscaledResidualInput.buffer !== ffnOutput.buffer) {
        releaseOrTrack(recorder, unscaledResidualInput.buffer, decodeBuffers);
      }
      residualInputOwned = true;
    }
    const residualOutputScale = requestedFinalOutputScale !== 1
      && residualTensor.dtype === context.activationDtype
      && !context.debugProbes?.length
      ? requestedFinalOutputScale
      : 1;
    if (
      nextInputNormFusion
      && config.normalizationType === 'rmsnorm'
      && residualInput.precision === undefined
      && residualTensor.precision === undefined
      && residualInput.dtype === 'f32'
      && residualTensor.dtype === 'f32'
    ) {
      const normWeightBuf = getNormWeightBuffer(
        nextInputNormFusion.weight,
        'next_input_norm',
        weightConfig,
        debugFlags
      );
      const pair = await doResidualNextRMSNormPair(
        residualInput,
        residualTensor,
        normWeightBuf,
        rmsNormEps,
        {
          batchSize: numTokens,
          hiddenSize,
          residualOutputBuffer: decodeOutputBuffer,
          outputScale: residualOutputScale,
          label: `L${layerIdx}.ffn_residual_next_input_norm`,
          layerIdx,
          rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
        },
        recorder
      );
      output = pair.residual;
      context.__precomputedInputNorm = {
        layerIdx: nextInputNormFusion.layerIdx,
        tensor: pair.nextInput,
      };
      if (!isGpuBufferInstance(nextInputNormFusion.weight) && !isWeightBuffer(nextInputNormFusion.weight)) {
        releaseOrTrack(recorder, normWeightBuf);
      }
    } else {
      output = await doResidualAdd(residualInput, residualTensor, size, recorder, {
        label: `L${layerIdx}.ffn_residual`,
        layerIdx,
        outputBuffer: decodeOutputBuffer,
        outputScale: residualOutputScale,
        executionPolicies: context.executionPolicies ?? null,
      });
    }
    if (residualOutputScale !== 1) {
      context.__layerScalarFusedFired = true;
    }
  }
  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: output.dtype,
    precision: output.precision, components: output.components,
    elementStrideBytes: output.elementStrideBytes,
  });
  await debugFFNBuffer(context, layerIdx, 'layer output', output, numTokens, hiddenSize);
  enqueueRecordedFFNHealth(context, layerIdx, 'layer_out', output, numTokens * hiddenSize);

  if (normedTensor !== postAttn && !usedPostAttnNormStatsPrelude) {
    releaseOrTrack(recorder, normedTensor.buffer, decodeBuffers);
  }
  if (postAttnNormStats?.invRmsBuffer) {
    releaseOrTrack(recorder, postAttnNormStats.invRmsBuffer, decodeBuffers);
  }
  if (releasePostAttnNormWeight && postAttnNormWeightBuf) {
    releaseOrTrack(recorder, postAttnNormWeightBuf, decodeBuffers);
  }
  releaseOrTrack(recorder, postAttn.buffer, decodeBuffers);
  if (prenormSumBuffer) {
    releaseOrTrack(recorder, prenormSumBuffer, decodeBuffers);
  }
  if (residualInputOwned) {
    releaseOrTrack(recorder, residualInput.buffer, decodeBuffers);
  }
  // When fusion fired, `output === ffnOutput` and we must NOT release it
  // here — it's our return value.
  if (!ffnResidualFused) {
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
  }

  return output;
}
