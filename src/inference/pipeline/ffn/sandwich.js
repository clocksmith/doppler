/**
 * Sandwich Norm FFN Processing
 *
 * Handles FFN with sandwich norm architecture (Gemma 3 style) where
 * pre/post FFN norms wrap the FFN block.
 *
 * @module inference/pipeline/ffn/sandwich
 */

import { doRMSNorm, doResidualAdd, releaseOrTrack } from '../ops.js';
import { getLayout, getWeightDtype, isCpuWeightBuffer } from '../../../gpu/weight-buffer.js';
import { trace } from '../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector, logFFN, getBufferStats } from '../debug-utils.js';
import { getNormWeightBuffer } from '../weights.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal, hasLoggedFusedDownNorm, setLoggedFusedDownNorm } from './types.js';
import { runDenseFFNGPU, runDenseFFNWithFusedPostNormGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';

/**
 * Process FFN with sandwich norm architecture (Gemma 3).
 * Input and output are Tensor for dtype-aware processing.
 * @param {number} layerIdx
 * @param {import('../../../gpu/tensor.js').Tensor} postAttn
 * @param {number} numTokens
 * @param {number} size
 * @param {import('../types.js').LayerContext} context
 * @param {import('../types.js').LayerWeights | undefined} layerWeights
 * @param {import('../types.js').SandwichNormInfo} sandwichNorm
 * @returns {Promise<import('../../../gpu/tensor.js').Tensor>}
 */
export async function processFFNWithSandwichNorm(
  layerIdx,
  postAttn,
  numTokens,
  size,
  context,
  layerWeights,
  sandwichNorm
) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;

  // For decode (M=1), get pre-allocated output buffer to avoid allocation
  const decodeOutputBuffer = numTokens === 1 && decodeBuffers
    ? decodeBuffers.getOutputHiddenBuffer()
    : null;
  const lastTokenIdx = Math.max(0, numTokens - 1);

  // 1. Pre-FFN norm (applied to residual stream before FFN)
  let ffnInput = postAttn;
  if (sandwichNorm.hasPreFeedforwardNorm && layerWeights?.preFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.preFeedforwardNorm, 'pre_feedforward_norm', weightConfig, debugFlags);

    ffnInput = await doRMSNorm(postAttn, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      label: `L${layerIdx}.pre_ffn_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);
    if (!(layerWeights.preFeedforwardNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
  }

  await runProbes('ffn_in', ffnInput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnInput.buffer, 'pre_ffn_norm_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: ffnInput.dtype,
    });
  }

  // 2. FFN (or MoE FFN)
  const downWeight = layerWeights?.down;
  const downWeightIsColumnMajor = downWeight && !(downWeight instanceof Float32Array) && !isCpuWeightBuffer(downWeight)
    ? getLayout(downWeight) === 'column'
    : false;

  const downWeightDtype = downWeight && !(downWeight instanceof Float32Array)
    ? (isCpuWeightBuffer(downWeight) ? downWeight.dtype : getWeightDtype(downWeight))
    : 'f32';
  const downWeightIsF32 = downWeightDtype === 'f32' || downWeightDtype === null;
  const downWeightIsF16 = downWeightDtype === 'f16';

  // Fused kernel requires matching dtypes: both F32 or both F16
  const dtypesMatchForFusion = (ffnInput.dtype === 'f32' && downWeightIsF32)
    || (ffnInput.dtype === 'f16' && downWeightIsF16);

  const canUseFusedDownNorm = numTokens === 1
    && !config.useMoE
    && !isMoELayerLocal(layerIdx, config, layerWeights)
    && sandwichNorm.hasPostFeedforwardNorm
    && layerWeights?.postFeedforwardNorm
    && layerWeights?.down
    && dtypesMatchForFusion
    && (await import('../../../gpu/kernel-selector.js')).shouldUseFusedMatmulRMSNorm(
      numTokens,
      hiddenSize,
      config.intermediateSize
    );

  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let ffnOutput;
  let usedFusedDownNorm = false;

  if (config.useMoE && isMoELayerLocal(layerIdx, config, layerWeights)) {
    ffnOutput = await runMoEFFNGPU(layerIdx, ffnInput, numTokens, context);
  } else if (canUseFusedDownNorm && layerWeights?.down && layerWeights?.postFeedforwardNorm &&
             (layerWeights?.gateUp || (layerWeights?.gate && layerWeights?.up))) {
    if (layerIdx === 0 && !hasLoggedFusedDownNorm()) {
      trace.ffn(0, `Using fused down+norm kernel (dtype=${ffnInput.dtype}, transposeB=${!downWeightIsColumnMajor})`);
      setLoggedFusedDownNorm(true);
    }
    ffnOutput = await runDenseFFNWithFusedPostNormGPU(
      layerIdx, ffnInput, numTokens, context, layerWeights,
      postAttn,
      rmsNormEps,
      !downWeightIsColumnMajor,
      decodeOutputBuffer
    );
    usedFusedDownNorm = true;
  } else {
    ffnOutput = await runDenseFFNGPU(layerIdx, ffnInput, numTokens, context, layerWeights);
  }
  await runProbes('ffn_out', ffnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(ffnOutput.buffer, 'ffn_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: ffnOutput.dtype,
    });
  }

  // Track for cleanup after submit if using recorder, otherwise release immediately
  if (ffnInput !== postAttn) {
    releaseOrTrack(recorder, ffnInput.buffer, decodeBuffers);
  }

  // Debug: trace FFN output
  const ffnStats = await getBufferStats(ffnOutput.buffer);
  if (ffnStats) logFFN(layerIdx, { maxAbsOut: ffnStats.maxAbs });

  // 3. Post-FFN norm
  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let output;
  if (usedFusedDownNorm) {
    output = ffnOutput;
  } else if (sandwichNorm.hasPostFeedforwardNorm && layerWeights?.postFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

    output = await doRMSNorm(ffnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: postAttn,
      outputBuffer: decodeOutputBuffer,
      label: `L${layerIdx}.post_ffn_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);

    if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
  } else {
    output = await doResidualAdd(ffnOutput, postAttn, size, recorder, {
      label: `L${layerIdx}.post_ffn_residual`,
      layerIdx,
      outputBuffer: decodeOutputBuffer,
    });
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
  }

  await runProbes('layer_out', output.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output.buffer, 'layer_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: output.dtype,
    });
  }

  releaseOrTrack(recorder, postAttn.buffer, decodeBuffers);

  return output;
}
