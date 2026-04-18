

import { doCast, doRMSNorm, doResidualAdd, releaseOrTrack } from '../ops.js';
import { getNormWeightBuffer } from '../weights.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal } from './types.js';
import { runDenseFFNGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { shouldDebugLayerOutput } from '../debug-utils/index.js';

async function debugFFNBuffer(context, layerIdx, label, tensor, numTokens, hiddenSize) {
  if (!context.debugCheckBuffer) return;
  if (!isGpuBufferInstance(tensor?.buffer)) return;
  if (!shouldDebugLayerOutput(layerIdx, context.debugLayers)) return;
  await context.debugCheckBuffer(tensor.buffer, `L${layerIdx} ${label} (GPU)`, numTokens, hiddenSize);
}


export async function processFFNStandard(
  layerIdx,
  postAttn,
  numTokens,
  size,
  context,
  layerWeights,
  fusedResidualInput
) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;

  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;

  // 1. Post-attention norm (optionally fuses upstream residual add via PRE_RESIDUAL)
  let normedTensor = postAttn;
  let prenormSumBuffer = null;
  if (layerWeights?.postAttnNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttnNorm, 'post_attn_norm', weightConfig, debugFlags);

    if (fusedResidualInput) {
      // Fused path: rmsnorm(postAttn + fusedResidualInput) and write pre-norm sum
      const bytesPerElement = postAttn.dtype === 'f16' ? 2 : 4;
      prenormSumBuffer = acquireBuffer(size * bytesPerElement, undefined, 'fused_prenorm_sum');
      normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        preResidual: fusedResidualInput,
        residualSumOutput: prenormSumBuffer,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
    } else {
      normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
    }

    if (!isGpuBufferInstance(layerWeights.postAttnNorm) && !isWeightBuffer(layerWeights.postAttnNorm)) releaseOrTrack(recorder, normWeightBuf);
  }
  await runProbes('ffn_in', normedTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: normedTensor.dtype,
  });
  await debugFFNBuffer(context, layerIdx, 'FFN input', normedTensor, numTokens, hiddenSize);

  // 2. FFN
  // Stage the residual tensor so that runDenseFFNGPU's ffn_down matmul can
  // opportunistically fuse with it (WideTile+residual kernel). The flag
  // `context.__ffnResidualFusedFired` will be set to true by dense.js if
  // the fusion actually fired; we read it below to skip the separate add.
  // If the pre-norm residual sum was captured (fused RMSNorm path), use that
  // buffer directly — it IS the correct residual for ffn_residual because
  // ffn_residual adds to the pre-rmsnorm tensor.
  const pendingResidualForFfn = prenormSumBuffer
    ? { buffer: prenormSumBuffer, dtype: postAttn.dtype, shape: postAttn.shape }
    : postAttn;
  context.__pendingFfnResidualTensor = pendingResidualForFfn;
  context.__ffnResidualFusedFired = false;

  let ffnOutput;
  if (config.useMoE && isMoELayerLocal(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, normedTensor, numTokens, context);
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, normedTensor, numTokens, context, layerWeights);
  }
  const ffnResidualFused = context.__ffnResidualFusedFired === true;
  context.__pendingFfnResidualTensor = null;
  context.__ffnResidualFusedFired = false;
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: ffnOutput.dtype,
  });
  await debugFFNBuffer(context, layerIdx, 'FFN output', ffnOutput, numTokens, hiddenSize);

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
      ? { buffer: prenormSumBuffer, dtype: postAttn.dtype, shape: postAttn.shape }
      : postAttn;
    residualInputOwned = ffnOutput.dtype !== residualTensor.dtype;
    if (residualInputOwned) {
      residualInput = await doCast(ffnOutput, residualTensor.dtype, recorder);
    }
    output = await doResidualAdd(residualInput, residualTensor, size, recorder, {
      label: `L${layerIdx}.ffn_residual`,
      layerIdx,
      outputBuffer: decodeOutputBuffer,
      executionPolicies: context.executionPolicies ?? null,
    });
  }
  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: output.dtype,
  });
  await debugFFNBuffer(context, layerIdx, 'layer output', output, numTokens, hiddenSize);

  if (normedTensor !== postAttn) {
    releaseOrTrack(recorder, normedTensor.buffer, decodeBuffers);
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
