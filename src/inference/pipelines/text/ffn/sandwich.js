

import { doRMSNorm, doResidualAdd, releaseOrTrack } from '../ops.js';
import { getLayout, getWeightDtype, isCpuWeightBuffer, isGpuBufferInstance, isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { trace } from '../../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector, logFFN, getBufferStats } from '../debug-utils.js';
import { shouldDebugLayerOutput } from '../debug-utils/index.js';
import { getNormWeightBuffer } from '../weights.js';
import { runProbes } from '../probes.js';
import { isMoELayerLocal, hasLoggedFusedDownNorm, setLoggedFusedDownNorm } from './types.js';
import { runDenseFFNGPU, runDenseFFNWithFusedPostNormGPU } from './dense.js';
import { runMoEFFNGPU } from './moe.js';
import { resolveLayerIntermediateSize } from '../config.js';

async function debugFFNBuffer(context, layerIdx, label, tensor, numTokens, hiddenSize) {
  if (!context.debugCheckBuffer) return;
  if (!isGpuBufferInstance(tensor?.buffer)) return;
  if (!shouldDebugLayerOutput(layerIdx, context.debugLayers)) return;
  await context.debugCheckBuffer(tensor.buffer, `L${layerIdx} ${label} (GPU)`, numTokens, hiddenSize);
}

function shouldUseDensePlusMoeFFN(layerIdx, config, layerWeights) {
  return config.ffnBranchMode === 'dense_plus_moe'
    && config.useMoE
    && isMoELayerLocal(layerIdx, config, layerWeights);
}

function shouldUseLegacyMoeOnlyFFN(layerIdx, config, layerWeights) {
  return config.ffnBranchMode !== 'dense'
    && config.ffnBranchMode !== 'dense_plus_moe'
    && config.useMoE
    && isMoELayerLocal(layerIdx, config, layerWeights);
}

function requireNormWeight(layerIdx, layerWeights, key) {
  const weight = layerWeights?.[key];
  if (!weight) {
    throw new Error(`Layer ${layerIdx} dense_plus_moe FFN is missing ${key}. Re-convert the model.`);
  }
  return weight;
}

async function applyBranchRMSNorm(layerIdx, tensor, layerWeights, key, label, context, numTokens, hiddenSize) {
  const { config, weightConfig, debugFlags, recorder } = context;
  const weight = requireNormWeight(layerIdx, layerWeights, key);
  const normWeightBuf = getNormWeightBuffer(weight, label, weightConfig, debugFlags);
  const output = await doRMSNorm(tensor, normWeightBuf, config.rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    label: `L${layerIdx}.${label}`,
    layerIdx,
    rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
  }, recorder);
  if (!isGpuBufferInstance(weight) && !isWeightBuffer(weight)) {
    releaseOrTrack(recorder, normWeightBuf);
  }
  return output;
}

async function runDensePlusMoeFFN(
  layerIdx,
  postAttn,
  ffnInput,
  numTokens,
  size,
  context,
  layerWeights,
  decodeOutputBuffer,
  finalOutputScale
) {
  const { config, recorder, weightConfig, debugFlags } = context;
  const { hiddenSize, rmsNormEps } = config;
  const denseRaw = await runDenseFFNGPU(layerIdx, ffnInput, numTokens, context, layerWeights);
  const denseNorm = await applyBranchRMSNorm(
    layerIdx,
    denseRaw,
    layerWeights,
    'postFeedforwardNorm1',
    'post_feedforward_norm_1',
    context,
    numTokens,
    hiddenSize
  );
  releaseOrTrack(recorder, denseRaw.buffer, context.decodeBuffers);

  const expertInput = await applyBranchRMSNorm(
    layerIdx,
    postAttn,
    layerWeights,
    'preFeedforwardNorm2',
    'pre_feedforward_norm_2',
    context,
    numTokens,
    hiddenSize
  );
  const expertRaw = await runMoEFFNGPU(layerIdx, expertInput, numTokens, context, {
    routerInputTensor: postAttn,
  });
  releaseOrTrack(recorder, expertInput.buffer, context.decodeBuffers);

  const expertNorm = await applyBranchRMSNorm(
    layerIdx,
    expertRaw,
    layerWeights,
    'postFeedforwardNorm2',
    'post_feedforward_norm_2',
    context,
    numTokens,
    hiddenSize
  );
  releaseOrTrack(recorder, expertRaw.buffer, context.decodeBuffers);

  const combined = await doResidualAdd(denseNorm, expertNorm, size, recorder, {
    label: `L${layerIdx}.ffn_dense_plus_moe`,
    layerIdx,
    executionPolicies: context.executionPolicies ?? null,
  });
  releaseOrTrack(recorder, denseNorm.buffer, context.decodeBuffers);
  releaseOrTrack(recorder, expertNorm.buffer, context.decodeBuffers);

  const finalNormWeight = requireNormWeight(layerIdx, layerWeights, 'postFeedforwardNorm');
  const finalNormWeightBuf = getNormWeightBuffer(finalNormWeight, 'post_feedforward_norm', weightConfig, debugFlags);
  const rmsNormOutputScale = finalOutputScale !== 1
    && combined.dtype === context.activationDtype
    && !context.debugProbes?.length
    ? finalOutputScale
    : 1;
  const output = await doRMSNorm(combined, finalNormWeightBuf, rmsNormEps, {
    batchSize: numTokens,
    hiddenSize,
    residual: postAttn,
    outputBuffer: decodeOutputBuffer,
    outputScale: rmsNormOutputScale,
    label: `L${layerIdx}.post_ffn_norm`,
    layerIdx,
    rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
  }, recorder);
  if (rmsNormOutputScale !== 1) {
    context.__layerScalarFusedFired = true;
  }
  if (!isGpuBufferInstance(finalNormWeight) && !isWeightBuffer(finalNormWeight)) {
    releaseOrTrack(recorder, finalNormWeightBuf);
  }

  return {
    ffnOutput: combined,
    output,
  };
}


export async function processFFNWithSandwichNorm(
  layerIdx,
  postAttn,
  numTokens,
  size,
  context,
  layerWeights,
  sandwichNorm,
  finalOutputScale = 1
) {
  const { config, weightConfig, debugFlags, recorder, decodeBuffers } = context;
  const { hiddenSize, rmsNormEps } = config;
  const requestedFinalOutputScale = finalOutputScale == null ? 1 : Number(finalOutputScale);
  if (!Number.isFinite(requestedFinalOutputScale)) {
    throw new Error(`Layer ${layerIdx} finalOutputScale must be finite; got "${String(finalOutputScale)}".`);
  }
  context.__layerScalarFusedFired = false;

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
    if (!isGpuBufferInstance(layerWeights.preFeedforwardNorm) && !isWeightBuffer(layerWeights.preFeedforwardNorm)) releaseOrTrack(recorder, normWeightBuf);
  }

  await runProbes('ffn_in', ffnInput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: ffnInput.dtype,
  });
  await debugFFNBuffer(context, layerIdx, 'FFN input', ffnInput, numTokens, hiddenSize);

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

  let canUseFusedDownNorm = numTokens === 1
    && !config.useMoE
    && !isMoELayerLocal(layerIdx, config, layerWeights)
    && sandwichNorm.hasPostFeedforwardNorm
    && layerWeights?.postFeedforwardNorm
    && layerWeights?.down
    && dtypesMatchForFusion;

  if (canUseFusedDownNorm) {
    canUseFusedDownNorm = (await import('../../../../gpu/kernel-selector.js')).shouldUseFusedMatmulRMSNorm(
      numTokens,
      hiddenSize,
      resolveLayerIntermediateSize(config, layerIdx)
    );
  }

  let ffnOutput;
  let combinedDenseMoeOutput = null;
  let usedFusedDownNorm = false;

  if (shouldUseDensePlusMoeFFN(layerIdx, config, layerWeights)) {
    const result = await runDensePlusMoeFFN(
      layerIdx,
      postAttn,
      ffnInput,
      numTokens,
      size,
      context,
      layerWeights,
      decodeOutputBuffer,
      requestedFinalOutputScale
    );
    ffnOutput = result.ffnOutput;
    combinedDenseMoeOutput = result.output;
  } else if (shouldUseLegacyMoeOnlyFFN(layerIdx, config, layerWeights)) {
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
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: ffnOutput.dtype,
  });
  await debugFFNBuffer(context, layerIdx, 'FFN output', ffnOutput, numTokens, hiddenSize);

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

  let output;
  if (combinedDenseMoeOutput) {
    output = combinedDenseMoeOutput;
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
  } else if (usedFusedDownNorm) {
    output = ffnOutput;
  } else if (sandwichNorm.hasPostFeedforwardNorm && layerWeights?.postFeedforwardNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);
    const rmsNormOutputScale = requestedFinalOutputScale !== 1
      && ffnOutput.dtype === context.activationDtype
      && !context.debugProbes?.length
      ? requestedFinalOutputScale
      : 1;

    output = await doRMSNorm(ffnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: postAttn,
      outputBuffer: decodeOutputBuffer,
      outputScale: rmsNormOutputScale,
      label: `L${layerIdx}.post_ffn_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);
    if (rmsNormOutputScale !== 1) {
      context.__layerScalarFusedFired = true;
    }

    if (!isGpuBufferInstance(layerWeights.postFeedforwardNorm) && !isWeightBuffer(layerWeights.postFeedforwardNorm)) releaseOrTrack(recorder, normWeightBuf);
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
  } else {
    const residualOutputScale = requestedFinalOutputScale !== 1
      && ffnOutput.dtype === context.activationDtype
      && postAttn.dtype === context.activationDtype
      && !context.debugProbes?.length
      ? requestedFinalOutputScale
      : 1;
    output = await doResidualAdd(ffnOutput, postAttn, size, recorder, {
      label: `L${layerIdx}.post_ffn_residual`,
      layerIdx,
      outputBuffer: decodeOutputBuffer,
      outputScale: residualOutputScale,
      executionPolicies: context.executionPolicies ?? null,
    });
    if (residualOutputScale !== 1) {
      context.__layerScalarFusedFired = true;
    }
    releaseOrTrack(recorder, ffnOutput.buffer, decodeBuffers);
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
