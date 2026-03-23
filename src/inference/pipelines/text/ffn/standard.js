

import { doRMSNorm, doResidualAdd, releaseOrTrack } from '../ops.js';
import { getNormWeightBuffer } from '../weights.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal } from './types.js';
import { runDenseFFNGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';
import { acquireBuffer } from '../../../../memory/buffer-pool.js';


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

    if (!(layerWeights.postAttnNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
  }
  await runProbes('ffn_in', normedTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    dtype: normedTensor.dtype,
  });

  // 2. FFN

  let ffnOutput;
  if (config.useMoE && isMoELayerLocal(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, normedTensor, numTokens, context);
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, normedTensor, numTokens, context, layerWeights);
  }
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    dtype: ffnOutput.dtype,
  });

  // 3. Residual add (uses prenorm sum when fused, otherwise postAttn)
  const residualTensor = prenormSumBuffer
    ? { buffer: prenormSumBuffer, dtype: postAttn.dtype }
    : postAttn;
  const output = await doResidualAdd(ffnOutput, residualTensor, size, recorder, {
    label: `L${layerIdx}.ffn_residual`,
    layerIdx,
    outputBuffer: decodeOutputBuffer,
  });
  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    dtype: output.dtype,
  });

  if (normedTensor !== postAttn) {
    releaseOrTrack(recorder, normedTensor.buffer, decodeBuffers);
  }
  releaseOrTrack(recorder, postAttn.buffer, decodeBuffers);
  if (prenormSumBuffer) {
    releaseOrTrack(recorder, prenormSumBuffer, decodeBuffers);
  }
  releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);

  return output;
}
