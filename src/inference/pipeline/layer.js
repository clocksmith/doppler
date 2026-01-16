

import { log, trace } from '../../debug/index.js';
import { getDevice } from '../../gpu/device.js';
import { releaseBuffer } from '../../gpu/buffer-pool.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import { createTensor } from '../../gpu/tensor.js';
import {
  doAttention, doRMSNorm, doResidualAdd,
  releaseOrTrack
} from './ops.js';
import {
  processFFNWithSandwichNorm,
  processFFNStandard
} from './ffn.js';
import { getWeightBuffer, getNormWeightBuffer } from './weights.js';
import { logLayer, logAttn, getBufferStats, isKernelDebugEnabled, dumpTokenVector, logKernelStep, shouldDebugLayerOutput } from './debug-utils.js';
import { runProbes } from './probes.js';
import { getLayerPlanSteps } from './layer-plan.js';

// ============================================================================
// Architecture Detection
// ============================================================================


export function detectSandwichNorm(layerWeights) {
  const hasPreFeedforwardNorm = Boolean(layerWeights?.preFeedforwardNorm);
  const hasPostFeedforwardNorm = Boolean(layerWeights?.postFeedforwardNorm);
  const hasPostAttentionNorm = Boolean(layerWeights?.postAttentionNorm);

  return {
    useSandwichNorm: hasPreFeedforwardNorm || hasPostFeedforwardNorm,
    hasPreFeedforwardNorm,
    hasPostFeedforwardNorm,
    hasPostAttentionNorm,
  };
}


export function isMoELayer(layerIdx, config, layerWeights) {
  if (!config.useMoE) return false;

  // Check if layer has router weights
  if (layerWeights?.routerWeight) return true;

  // Fall back to layer_types array if available
  const layerTypes = config.layerTypes;
  if (Array.isArray(layerTypes) && layerIdx < layerTypes.length) {
    return layerTypes[layerIdx] === 'moe';
  }

  // Default: assume all layers are MoE if model uses MoE
  return true;
}

// ============================================================================
// Main Layer Processing
// ============================================================================


export async function processLayer(layerIdx, hiddenStates, numTokens, isPrefill, context) {
  const { config, useGPU } = context;
  const { hiddenSize } = config;

  // Debug routing (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  // Debug: check path being taken for layer 0
  if (context.debug && layerIdx === 0) {
    trace.ffn(0, `routing: useGPU=${useGPU}, isGPUBuffer=${hiddenStates instanceof GPUBuffer}, constructor=${hiddenStates?.constructor?.name}`);
  }

  // GPU-native path
  if (useGPU && hiddenStates instanceof GPUBuffer) {
    return processLayerGPU(layerIdx, hiddenStates, numTokens, isPrefill, numTokens * hiddenSize, context);
  }

  // CPU fallback path
  return processLayerCPU(layerIdx,  (hiddenStates), numTokens, isPrefill, context);
}

// ============================================================================
// GPU Layer Processing
// ============================================================================


export async function processLayerGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context) {
  // Debug entry (uses debug-utils)
  logLayer(layerIdx, 'enter', isPrefill, { numTokens });

  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const { config, weights, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  // Determine activation dtype from context (defaults to f32)
  
  const activationDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';

  // Wrap input buffer as Tensor for dtype-aware processing
  const inputTensor = createTensor(inputBuffer, activationDtype, [numTokens, hiddenSize], 'layer_input');

  const layerWeights =  (weights.get(`layer_${layerIdx}`));
  const sandwichNorm = detectSandwichNorm(layerWeights ?? null);
  const lastTokenIdx = Math.max(0, numTokens - 1);

  if (context.pipelinePlan) {
    return processLayerPlanGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context, layerWeights, sandwichNorm);
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    logKernelStep('layer', { layerIdx, label: `seqLen=${numTokens} prefill=${isPrefill}` });
    await dumpTokenVector(inputBuffer, 'layer_in', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: activationDtype,
    });
  }

  // 1. Self-attention
  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = layerType === 'sliding_attention';

  // Debug: log RoPE selection for first few layers
  if (context.debug && layerIdx < 3) {
    trace.attn(layerIdx, `RoPE selection: layerType=${layerType}, isLocal=${isLocalLayer}, hasLocalCos=${!!context.ropeLocalCos}, hasLocalSin=${!!context.ropeLocalSin}`);
  }

  
  const attnConfig = {
    layerIdx,
    numTokens,
    isPrefill,
    numHeads,
    numKVHeads,
    headDim,
    hiddenSize,
    rmsNormEps,
    currentSeqLen: context.currentSeqLen,
    activationDtype,
    slidingWindow: config.slidingWindow,
    layerType,
    residualTensor: (numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm))
      ? inputTensor
      : null,
    attnSoftcap: config.attnLogitSoftcapping ?? 0,
    queryPreAttnScalar: config.queryPreAttnScalar,
    queryKeyNorm: config.queryKeyNorm,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
  };

  
  const attnState = {
    ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
      ?  (context.ropeLocalCos)
      :  (ropeFreqsCos),
    ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
      ?  (context.ropeLocalSin)
      :  (ropeFreqsSin),
    kvCache:  ( (kvCache)),
  };

  const attnResult = await doAttention(
    inputTensor,
    layerWeights ?? null,
    attnConfig,
    attnState,
    context.debug,
    { debugLayers: context.debugLayers },
    (weight, label) => getWeightBuffer(weight, label),
    (weight, label) => getNormWeightBuffer(weight, label, weightConfig, debugFlags),
    context.debugCheckBuffer,
    recorder,
    context.lora
  );
  const attnOutput = attnResult.output;
  const residualFused = attnResult.residualFused;

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(attnOutput.buffer, 'attn_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: attnOutput.dtype,
    });
  }

  // Debug: trace attn output
  if (context.debug) {
    const stats = await getBufferStats(attnOutput.buffer);
    if (stats) logAttn(layerIdx, isPrefill, { numTokens, kvLen: context.currentSeqLen + (isPrefill ? numTokens : 1), maxAbsOut: stats.maxAbs });

    trace.attn(layerIdx, `attnOutput type check: isGPU=${attnOutput.buffer instanceof GPUBuffer}, type=${typeof attnOutput.buffer}, constructor=${attnOutput.buffer?.constructor?.name}, isPrefill=${isPrefill}`);
    if (shouldDebugLayerOutput(layerIdx, context.debugLayers) && attnOutput.buffer instanceof GPUBuffer && !recorder) {
      if (allowReadback(`layer.attn-out.${layerIdx}`)) {
        try {
          const sampleSize = Math.min(128, attnOutput.buffer.size);
          const staging = device.createBuffer({ size: sampleSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const enc = device.createCommandEncoder();
          enc.copyBufferToBuffer(attnOutput.buffer, 0, staging, 0, sampleSize);
          device.queue.submit([enc.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(staging.getMappedRange().slice(0));
          staging.unmap();
          staging.destroy();
          const maxAbs = Math.max(...Array.from(data).map(x => Math.abs(x)));
          const nonZero = Array.from(data).filter(x => x !== 0).length;
          trace.attn(layerIdx, `ATTN_OUT: maxAbs=${maxAbs.toFixed(4)}, nonZero=${nonZero}/${data.length}, sample=[${Array.from(data).slice(0, 5).map(x => x.toFixed(4)).join(', ')}]`);
        } catch (e) {
          trace.attn(layerIdx, `ATTN_OUT error: ${e}`);
        }
      }
    } else if (shouldDebugLayerOutput(layerIdx, context.debugLayers) && attnOutput.buffer instanceof GPUBuffer && recorder) {
      trace.attn(layerIdx, `ATTN_OUT: (skipped - using batched recorder, values not available until submit)`);
    }
  }
  await runProbes('attn_out', attnOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 2. Handle residual connection based on architecture
  
  let postAttn;
  if (residualFused) {
    postAttn = attnOutput;
    if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
      const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
      postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
        batchSize: numTokens,
        hiddenSize,
        label: `L${layerIdx}.post_attn_norm`,
        layerIdx,
        rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
      }, recorder);
      if (!(layerWeights.postAttentionNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
      if (recorder) {
        recorder.trackTemporaryBuffer(attnOutput.buffer);
      } else {
        releaseBuffer(attnOutput.buffer);
      }
    }
  } else if (sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm && layerWeights?.postAttentionNorm) {
    const normWeightBuf = getNormWeightBuffer(layerWeights.postAttentionNorm, 'post_attention_norm', weightConfig, debugFlags);
    postAttn = await doRMSNorm(attnOutput, normWeightBuf, rmsNormEps, {
      batchSize: numTokens,
      hiddenSize,
      residual: inputTensor,
      label: `L${layerIdx}.post_attn_norm`,
      layerIdx,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    }, recorder);

    if (!(layerWeights.postAttentionNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  } else {
    postAttn = await doResidualAdd(attnOutput, inputTensor, size, recorder, { label: `L${layerIdx}.post_attn_residual`, layerIdx });
    if (recorder) {
      recorder.trackTemporaryBuffer(attnOutput.buffer);
    } else {
      releaseBuffer(attnOutput.buffer);
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(postAttn.buffer, 'x_after_attn', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: postAttn.dtype,
    });
  }

  await runProbes('post_attn', postAttn.buffer, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  // 3. Feed-forward network
  
  let outputTensor;
  if (sandwichNorm.useSandwichNorm) {
    outputTensor = await processFFNWithSandwichNorm(layerIdx, postAttn, numTokens, size, context, layerWeights, sandwichNorm);
  } else {
    outputTensor = await processFFNStandard(layerIdx, postAttn, numTokens, size, context, layerWeights);
  }

  return outputTensor.buffer;
}

// ============================================================================
// Configurable Layer Pipeline (JSON-Driven)
// ============================================================================


function resolveNormWeightForPlan(weight, layerWeights) {
  if (!layerWeights) return null;
  switch (weight) {
    case 'input':
      return layerWeights.inputNorm;
    case 'post_attention':
      return layerWeights.postAttentionNorm ?? layerWeights.postAttnNorm ?? null;
    case 'post_attn':
      return layerWeights.postAttnNorm ?? layerWeights.postAttentionNorm ?? null;
    case 'pre_ffn':
      return layerWeights.preFeedforwardNorm ?? null;
    case 'post_ffn':
      return layerWeights.postFeedforwardNorm ?? null;
    default:
      return null;
  }
}


async function processLayerPlanGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context, layerWeights, sandwichNorm) {
  const { config, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  if (!context.pipelinePlan) {
    throw new Error('Layer pipeline plan missing from context');
  }

  const steps = getLayerPlanSteps(context.pipelinePlan, layerIdx);
  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = layerType === 'sliding_attention';
  
  const attnState = {
    ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
      ?  (context.ropeLocalCos)
      :  (ropeFreqsCos),
    ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
      ?  (context.ropeLocalSin)
      :  (ropeFreqsSin),
    kvCache:  ( (kvCache)),
  };

  const allowResidualFuse = numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm);

  
  const slots = new Map();
  
  const refCounts = new Map();
  const protectedBuffers = new Set([inputBuffer]);

  
  const addRef = (buf) => {
    refCounts.set(buf, (refCounts.get(buf) ?? 0) + 1);
  };
  
  const releaseRef = (buf) => {
    const next = (refCounts.get(buf) ?? 0) - 1;
    if (next > 0) {
      refCounts.set(buf, next);
      return;
    }
    refCounts.delete(buf);
    if (protectedBuffers.has(buf)) return;
    if (recorder) {
      recorder.trackTemporaryBuffer(buf);
    } else {
      releaseBuffer(buf);
    }
  };
  
  const getSlot = (name) => {
    const key = name.trim() || 'state';
    const buf = slots.get(key);
    if (!buf) {
      throw new Error(`Layer pipeline missing slot "${key}" at L${layerIdx}`);
    }
    return buf;
  };
  
  const setSlot = (name, buf) => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (prev && prev !== buf) {
      releaseRef(prev);
    }
    slots.set(key, buf);
    addRef(buf);
  };
  
  const clearSlot = (name) => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (!prev) return;
    slots.delete(key);
    releaseRef(prev);
  };

  setSlot('state', inputBuffer);

  const cleanupSlots = () => {
    for (const [name, buf] of slots) {
      if (name === 'state' || protectedBuffers.has(buf)) continue;
      const refs = refCounts.get(buf) ?? 0;
      if (refs > 0) {
        refCounts.delete(buf);
        if (recorder) {
          recorder.trackTemporaryBuffer(buf);
        } else {
          releaseBuffer(buf);
        }
      }
    }
  };

  try {
    for (const step of steps) {
      switch (step.op) {
        case 'save': {
          const src = getSlot(step.src);
          setSlot( (step.name), src);
          break;
        }
        case 'load': {
          const src = getSlot( (step.name));
          setSlot(step.dst, src);
          break;
        }
        case 'attention': {
          const srcBuf = getSlot(step.src);
          const residualBuf = step.residual ? getSlot(step.residual) : null;

          
          const activationDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_attn_src');
          const residualTensor = (residualBuf && allowResidualFuse)
            ? createTensor(residualBuf, activationDtype, [numTokens, hiddenSize], 'plan_attn_residual')
            : null;

          
          const attnConfig = {
            layerIdx,
            numTokens,
            isPrefill,
            numHeads,
            numKVHeads,
            headDim,
            hiddenSize,
            rmsNormEps,
            currentSeqLen: context.currentSeqLen,
            slidingWindow: config.slidingWindow,
            layerType,
            residualTensor,
            attnSoftcap: config.attnLogitSoftcapping ?? 0,
            queryPreAttnScalar: config.queryPreAttnScalar,
            queryKeyNorm: config.queryKeyNorm,
            rmsNormWeightOffset: config.rmsNormWeightOffset,
            skipInputNorm: step.skipInputNorm === true,
          };

          const result = await doAttention(
            srcTensor,
            layerWeights ?? null,
            attnConfig,
            attnState,
            context.debug,
            { debugLayers: context.debugLayers },
            (weight, label) => getWeightBuffer(weight, label),
            (weight, label) => getNormWeightBuffer(weight, label, weightConfig, debugFlags),
            context.debugCheckBuffer,
            recorder,
            context.lora
          );

          setSlot(step.dst, result.output.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, result.output.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'rmsnorm': {
          const srcBuf = getSlot(step.src);
          const weight = resolveNormWeightForPlan( (step.weight), layerWeights);
          if (!weight) {
            throw new Error(`Layer pipeline rmsnorm missing weights for "${step.weight}" at L${layerIdx}`);
          }
          const normWeightBuf = getNormWeightBuffer(weight, `rmsnorm_${step.weight}`, weightConfig, debugFlags);
          const residualBuf = step.residual ? getSlot(step.residual) : null;
          
          const activationDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_rmsnorm_src');
          const residualTensor = residualBuf ? createTensor(residualBuf, activationDtype, [numTokens, hiddenSize], 'plan_rmsnorm_residual') : null;
          const outputTensor = await doRMSNorm(srcTensor, normWeightBuf, rmsNormEps, {
            batchSize: numTokens,
            hiddenSize,
            residual: residualTensor,
            label: `L${layerIdx}.rmsnorm_${step.weight}`,
            layerIdx,
            rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
          }, recorder);
          if (!(weight instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'ffn': {
          const srcBuf = getSlot(step.src);
          
          const activationDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_ffn_src');
          
          let outputTensor;
          const { runMoEFFNGPU, runDenseFFNGPU } = await import('./ffn.js');

          const useMoe = step.variant === 'moe'
            || (step.variant === 'auto' && config.useMoE && isMoELayer(layerIdx, config, layerWeights));
          if (useMoe) {
            outputTensor = await runMoEFFNGPU(layerIdx, srcTensor, numTokens, context);
          } else {
            outputTensor = await runDenseFFNGPU(layerIdx, srcTensor, numTokens, context, layerWeights);
          }
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'residual_add': {
          const aBuf = getSlot(step.a ?? 'state');
          const bBuf = getSlot(step.b ?? 'residual');
          
          const activationDtype = context.activationDtype === 'f16' ? 'f16' : 'f32';
          const aTensor = createTensor(aBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_a');
          const bTensor = createTensor(bBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_b');
          const outputTensor = await doResidualAdd(aTensor, bTensor, size, recorder, {
            label: `L${layerIdx}.residual_add`,
            layerIdx,
          });
          setSlot(step.dst, outputTensor.buffer);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
            });
          }
          break;
        }
        case 'noop':
          break;
        default:
          throw new Error(`Unknown layer pipeline op "${step.op}" at L${layerIdx}`);
      }
    }

    // Normal cleanup
    for (const name of Array.from(slots.keys())) {
      if (name !== 'state') {
        clearSlot(name);
      }
    }
  } catch (err) {
    cleanupSlots();
    throw err;
  }

  const output = getSlot('state');
  await runProbes('layer_out', output, {
    layerIdx,
    numTokens,
    hiddenSize,
    probes: context.debugProbes,
    recorder,
  });

  return output;
}

// ============================================================================
// CPU Fallback
// ============================================================================


export async function processLayerCPU(layerIdx, hiddenStates, numTokens, isPrefill, context) {
  const { config } = context;
  const { hiddenSize } = config;

  log.warn('Layer', `L${layerIdx} CPU fallback - returning input unchanged`);
  return new Float32Array(hiddenStates);
}
