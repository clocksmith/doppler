import { createTensor } from '../../../../gpu/tensor.js';
import { isGpuBufferInstance } from '../../../../gpu/weight-buffer.js';
import { doAttention, releaseOrTrack } from '../ops.js';
import { getWeightBuffer, getNormWeightBuffer } from '../weights.js';
import { runLinearAttentionLayer } from '../linear-attention.js';
import { runMoEFFNGPU } from '../ffn/moe.js';
import { validateAttnConfig } from '../attention/attn-config.js';
import { runProbes } from '../probes.js';
import { replicateEmbeddings, runHyperConnectionDecoderLayer } from './decoder-layer.js';
import { createHyperConnectionProjection } from './projection.js';

/** @type {typeof import('./layer-executor.js').processHyperConnectionLayer} */
export async function processHyperConnectionLayer(layerIdx, inputBuffer, numTokens, isPrefill, context) {
  const { config, recorder, hyperConnectionSession: session } = context;
  const hyper = config.hyperConnections;
  const resolvedRuntimeSession = context.resolvedRuntimeSession;
  const intermediatePrecision = config.sessionSettings.intermediatePrecision;
  const weights = /** @type {import('../weights.js').LayerWeights | undefined} */ (context.weights.get(`layer_${layerIdx}`));
  if (!hyper || !session || !weights?.hyperConnections || context.activationDtype !== 'f32'
    || context.activationPrecision || context.pipelinePlan || context.lora || context.skipKVCacheWrites
    || !resolvedRuntimeSession || intermediatePrecision !== 'f32') {
    throw new Error('[HyperConnection] Requires initialized state, gated weights and ordinary f32 text execution without layer-plan/LoRA overrides.');
  }
  context.signal?.throwIfAborted();
  const width = hyper.streamCount * hyper.streamWidth;
  const project = createHyperConnectionProjection({ recorder, kernelPath: context.kernelPath, phase: context.phase, layerIdx });
  let streams = createTensor(inputBuffer, 'f32', [numTokens, layerIdx === 0 ? hyper.streamWidth : width], 'hc_input');
  const ownsInput = layerIdx === 0;
  if (ownsInput) streams = await replicateEmbeddings(recorder ?? null, streams, { hyper, numTokens });
  try {
    await runProbes('layer_in', streams.buffer, { layerIdx, numTokens, hiddenSize: width,
      probes: context.debugProbes, recorder, operatorDiagnostics: context.operatorDiagnostics, dtype: 'f32' });
    const result = await runHyperConnectionDecoderLayer(recorder ?? null, {
      streams, numTokens, hyper, weights: weights.hyperConnections, project,
      ple: weights.ple ? input => session.runPle(layerIdx, input, weights.ple, context, project) : null,
      attentionBranch: async input => {
        if (config.layerTypes?.[layerIdx] === 'linear_attention') {
          return runLinearAttentionLayer(input, { ...weights, inputNorm: null }, {
            layerIdx, executionPhase: context.phase ?? undefined, numTokens, hiddenSize: config.hiddenSize, config: { ...config, sessionSettings: { ...config.sessionSettings, intermediatePrecision } },
            currentSeqLen: context.currentSeqLen, activationDtype: input.dtype, kernelPath: context.kernelPath,
            linearRuntime: context.linearAttentionRuntime,
            runtimeSession: /** @type {import('../linear-attention.js').RunLinearAttentionLayerOptions['runtimeSession']} */ (resolvedRuntimeSession.runtime.session),
            getWeightBuffer, getNormWeightBuffer: (weight, label) => getNormWeightBuffer(weight, label, context.weightConfig, context.debugFlags),
            debugProbes: context.debugProbes, operatorDiagnostics: context.operatorDiagnostics, recorder: recorder ?? null,
          });
        }
        if (config.layerTypes?.[layerIdx] !== 'full_attention' || !weights.qsaIndexer) {
          throw new Error(`[HyperConnection] Layer ${layerIdx} requires a declared linear or indexed full-attention branch.`);
        }
        const cos = context.ropeFreqsCos, sin = context.ropeFreqsSin;
        if (!isGpuBufferInstance(cos) || !isGpuBufferInstance(sin)) throw new Error('[HyperConnection] Indexed attention requires GPU rotary tables.');
        const selection = await session.prepareQsa(layerIdx, input, weights.qsaIndexer, context, project);
        try {
          context.signal?.throwIfAborted();
          const attnConfig = {
            layerIdx, numTokens, isPrefill, numHeads: config.numHeads, numKVHeads: config.numKVHeads,
            headDim: config.headDim, hiddenSize: config.hiddenSize, rmsNormEps: config.rmsNormEps,
            currentSeqLen: context.currentSeqLen, activationDtype: input.dtype, slidingWindow: config.slidingWindow,
            layerType: config.layerTypes[layerIdx], residualTensor: null, skipInputNorm: true,
            attnSoftcap: config.attnLogitSoftcapping === null ? 0 : config.attnLogitSoftcapping,
            queryPreAttnScalar: config.queryPreAttnScalar, queryScale: config.queryScale, queryKeyNorm: config.queryKeyNorm,
            queryKeyNormType: config.queryKeyNormType, queryKeyNormAxis: config.queryKeyNormAxis,
            queryKeyNormWeightLayers: config.queryKeyNormWeightLayers, valueNorm: config.valueNorm,
            attentionOutputGate: config.attentionOutputGate, outputGateType: config.outputGateType,
            causalAttention: config.causalAttention, rmsNormWeightOffset: config.rmsNormWeightOffset,
            normalizationType: config.normalizationType, ropeRotaryDim: config.ropeRotaryDim,
            ropeFrequencyBaseDim: config.ropeFrequencyBaseDim, ropeInterleaved: config.ropeInterleaved,
            tokenIds: context.currentTokenIds, kernelPath: context.kernelPath, sessionSettings: config.sessionSettings,
          };
          validateAttnConfig(attnConfig, `L${layerIdx}`);
          const output = await doAttention(input, weights, attnConfig, {
            ropeFreqsCos: cos, ropeFreqsSin: sin, kvCache: context.kvCache,
            stats: context.stats, debugProbes: context.debugProbes, operatorDiagnostics: context.operatorDiagnostics,
            resolvedRuntimeSession, executionPolicies: context.executionPolicies,
            observationContext: context.observationContext, qsa: selection,
          }, context.debug, { debugLayers: context.debugLayers }, getWeightBuffer,
          (weight, label) => getNormWeightBuffer(weight, label, context.weightConfig, context.debugFlags),
          context.debugCheckBuffer, recorder);
          if (output.residualFused) {
            releaseOrTrack(recorder, output.output.buffer);
            throw new Error('[HyperConnection] Attention must not add an ordinary residual.');
          }
          return output.output;
        } finally {
          session.releaseQsa(selection, recorder ?? null);
        }
      },
      mlpBranch: input => {
        context.signal?.throwIfAborted();
        return runMoEFFNGPU(layerIdx, input, numTokens, context);
      },
    });
    try {
      await runProbes('layer_out', result.buffer, { layerIdx, numTokens, hiddenSize: width,
        probes: context.debugProbes, recorder, operatorDiagnostics: context.operatorDiagnostics, dtype: 'f32' });
      return result.buffer;
    } catch (error) {
      releaseOrTrack(recorder, result.buffer);
      throw error;
    }
  } finally {
    if (ownsInput && isGpuBufferInstance(streams.buffer)) releaseOrTrack(recorder, streams.buffer);
  }
}
