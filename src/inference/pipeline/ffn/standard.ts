/**
 * Standard FFN Processing
 *
 * Handles FFN with standard architecture (LLaMA-style) where
 * post-attention norm precedes the FFN block.
 *
 * @module inference/pipeline/ffn/standard
 */

import { doRMSNorm, doResidualAdd, releaseOrTrack } from '../ops.js';
import type { Tensor } from '../../../gpu/tensor.js';
import { getNormWeightBuffer } from '../weights.js';
import type { LayerContext, LayerWeights } from '../types.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal } from './types.js';
import { runDenseFFNGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';

/**
 * Process FFN with standard architecture (LLaMA-style).
 */
export async function processFFNStandard(
  layerIdx: number,
  postAttn: Tensor,
  numTokens: number,
  size: number,
  context: LayerContext,
  layerWeights: LayerWeights | undefined
): Promise<Tensor> {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;

  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;

  // 1. Post-attention norm
  let normedTensor = postAttn;
  if (layerWeights?.postAttnNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttnNorm, 'post_attn_norm', weightConfig, debugFlags);
    normedTensor = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
    }, recorder);
    if (!(layerWeights.postAttnNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
  }
  await runProbes('ffn_in', normedTensor.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 2. FFN
  let ffnOutput: Tensor;
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
  });

  // 3. Residual add
  const output = await doResidualAdd(ffnOutput, postAttn, size, recorder, {
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
  });

  if (normedTensor !== postAttn) {
    releaseOrTrack(recorder, normedTensor.buffer, decodeBuffers);
  }
  releaseOrTrack(recorder, postAttn.buffer, decodeBuffers);
  releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);

  return output;
}
