

import { getDevice } from '../../../gpu/device.js';
import { releaseBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import { doAttention, doRMSNorm, doResidualAdd, doConv, doCast, releaseOrTrack } from './ops.js';
import { getWeightBuffer, getNormWeightBuffer } from './weights.js';
import { runProbes } from './probes.js';
import { getLayerPlanSteps, filterLayerPlanStepsByPhase } from './layer-plan.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';
import { recordCheckFiniteness } from '../../../gpu/kernels/check-finiteness.js';
import { shouldRunFinitenessGuard } from './finiteness-policy.js';
import { isGpuBufferInstance, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import {
  isSlidingLayerType,
  resolveActivationDtype,
  resolveAttentionHeadDim,
  resolveAttentionRotaryDim,
  resolveAttentionKVSharing,
  getConvLayerState,
  isMoELayer,
  hasPerLayerInputBlock,
  applyPerLayerInputBlock,
} from './layer.js';

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


export async function processLayerPlanGPU(layerIdx, inputBuffer, numTokens, isPrefill, size, context, layerWeights, sandwichNorm) {
  const { config, weightConfig, debugFlags, kvCache, ropeFreqsCos, ropeFreqsSin, recorder } = context;
  const { hiddenSize, numHeads, numKVHeads, headDim, rmsNormEps } = config;

  if (!context.pipelinePlan) {
    throw new Error('Layer pipeline plan missing from context');
  }

  const planSteps = getLayerPlanSteps(context.pipelinePlan, layerIdx);
  const steps = filterLayerPlanStepsByPhase(planSteps, isPrefill);
  const device = getDevice();
  if (!device) throw new Error('No GPU device available');

  const layerType = config.layerTypes?.[layerIdx];
  const isLocalLayer = isSlidingLayerType(layerType);
  const activationDtype = resolveActivationDtype(context.activationDtype);

  const attnState = {
    ropeFreqsCos: (isLocalLayer && context.ropeLocalCos)
      ? (context.ropeLocalCos)
      : (ropeFreqsCos),
    ropeFreqsSin: (isLocalLayer && context.ropeLocalSin)
      ? (context.ropeLocalSin)
      : (ropeFreqsSin),
    sharedAttentionState: context.sharedAttentionState ?? null,
    kvCache: ((kvCache)),
    linearRuntime: context.linearAttentionRuntime ?? null,
    operatorDiagnostics: context.operatorDiagnostics,
    executionPolicies: context.executionPolicies ?? null,
  };

  const allowResidualFuse = numTokens === 1 && !(sandwichNorm.useSandwichNorm && sandwichNorm.hasPostAttentionNorm);


  const slots = new Map();
  const slotDtypes = new Map();

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

  const getSlotDtype = (name) => {
    const key = name.trim() || 'state';
    return slotDtypes.get(key) ?? null;
  };

  const resolveStepInputDtype = (step, slotName) => {
    const slotDtype = getSlotDtype(slotName) ?? resolveActivationDtype(context.activationDtype);
    if (!step.inputDtype) {
      return slotDtype;
    }
    const required = resolveActivationDtype(step.inputDtype);
    if (slotDtype !== required) {
      throw new Error(
        `Layer pipeline dtype mismatch at L${layerIdx} step "${step.op}": ` +
        `slot "${slotName}" is ${slotDtype} but step requires ${required}. ` +
        'Insert an explicit cast step.'
      );
    }
    return required;
  };

  const resolveStepOutputDtype = (step, actualOutputDtype) => {
    if (!step.outputDtype) {
      return actualOutputDtype;
    }
    const required = resolveActivationDtype(step.outputDtype);
    if (actualOutputDtype !== required) {
      throw new Error(
        `Layer pipeline output dtype mismatch at L${layerIdx} step "${step.op}": ` +
        `kernel produced ${actualOutputDtype} but step declares ${required}.`
      );
    }
    return required;
  };

  const setSlot = (name, buf, dtype) => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (prev && prev !== buf) {
      releaseRef(prev);
    }
    slots.set(key, buf);
    if (dtype) {
      slotDtypes.set(key, dtype);
    }
    addRef(buf);
  };

  const clearSlot = (name) => {
    const key = name.trim() || 'state';
    const prev = slots.get(key);
    if (!prev) return;
    slots.delete(key);
    slotDtypes.delete(key);
    releaseRef(prev);
  };

  setSlot('state', inputBuffer, activationDtype);

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
          const srcDtype = getSlotDtype(step.src) ?? resolveActivationDtype(context.activationDtype);
          setSlot((step.name), src, srcDtype);
          break;
        }
        case 'load': {
          const src = getSlot((step.name));
          const srcDtype = getSlotDtype(step.name) ?? resolveActivationDtype(context.activationDtype);
          setSlot(step.dst, src, srcDtype);
          break;
        }
        case 'attention': {
          const srcBuf = getSlot(step.src);
          const residualBuf = step.residual ? getSlot(step.residual) : null;


          const activationDtype = resolveStepInputDtype(step, step.src);
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
            headDim: resolveAttentionHeadDim(config, layerType),
            hiddenSize,
            rmsNormEps,
            currentSeqLen: context.currentSeqLen,
            slidingWindow: config.slidingWindow,
            layerType,
            residualTensor,
            attnSoftcap: config.attnLogitSoftcapping === null ? 0 : config.attnLogitSoftcapping,
            queryPreAttnScalar: config.queryPreAttnScalar,
            queryKeyNorm: config.queryKeyNorm,
            valueNorm: config.valueNorm,
            attentionOutputGate: config.attentionOutputGate,
            causalAttention: config.causalAttention,
            rmsNormWeightOffset: config.rmsNormWeightOffset,
            ropeRotaryDim: resolveAttentionRotaryDim(config, layerType),
            ropeInterleaved: config.ropeInterleaved,
            tokenIds: context.currentTokenIds ?? null,
            skipInputNorm: step.skipInputNorm === true,
            activationDtype,
            inputDtype: step.inputDtype ?? undefined,
            outputDtype: step.outputDtype ?? undefined,
            kvDtype: step.kvDtype ?? undefined,
            kernelPath: context.kernelPath ?? null,
            ...resolveAttentionKVSharing(config, layerIdx, layerType),
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

          const outputDtype = resolveStepOutputDtype(step, resolveActivationDtype(result.output.dtype));
          setSlot(step.dst, result.output.buffer, outputDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, result.output.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: outputDtype,
            });
          }
          break;
        }
        case 'conv': {
          const srcBuf = getSlot(step.src);
          const inputDtype = resolveStepInputDtype(step, step.src);
          const srcTensor = createTensor(srcBuf, inputDtype, [numTokens, hiddenSize], 'plan_conv_src');

          const convInProj = layerWeights?.convInProj ?? null;
          const convOutProj = layerWeights?.convOutProj ?? null;
          if (!convInProj || !convOutProj) {
            throw new Error(
              `Layer pipeline conv step missing conv weights at L${layerIdx}. ` +
              'Expected conv.in_proj.weight and conv.out_proj.weight.'
            );
          }
          const convKernel = layerWeights?.convKernel ?? null;

          const outputTensor = await doConv(
            srcTensor,
            getWeightBuffer(convInProj, `L${layerIdx}.plan_conv_in_proj`),
            convKernel ? getWeightBuffer(convKernel, `L${layerIdx}.plan_conv_kernel`) : null,
            getWeightBuffer(convOutProj, `L${layerIdx}.plan_conv_out_proj`),
            {
              numTokens,
              hiddenSize,
              layerIdx,
              label: `L${layerIdx}.plan_conv`,
              swigluLimit: config.swigluLimit,
              kernelPath: context.kernelPath ?? null,
              executionPolicies: context.executionPolicies ?? null,
              convState: getConvLayerState(context.convLayerStates, layerIdx),
            },
            recorder
          );
          const outputDtype = resolveStepOutputDtype(step, resolveActivationDtype(outputTensor.dtype));
          setSlot(step.dst, outputTensor.buffer, outputDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: outputDtype,
            });
          }
          break;
        }
        case 'rmsnorm': {
          const srcBuf = getSlot(step.src);
          const weight = resolveNormWeightForPlan((step.weight), layerWeights);
          if (!weight) {
            throw new Error(`Layer pipeline rmsnorm missing weights for "${step.weight}" at L${layerIdx}`);
          }
          const normWeightBuf = getNormWeightBuffer(weight, `rmsnorm_${step.weight}`, weightConfig, debugFlags);
          const residualBuf = step.residual ? getSlot(step.residual) : null;

          const activationDtype = resolveStepInputDtype(step, step.src);
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
          if (!isGpuBufferInstance(weight) && !isWeightBuffer(weight)) releaseOrTrack(recorder, normWeightBuf);
          const outputDtype = resolveStepOutputDtype(step, resolveActivationDtype(outputTensor.dtype));
          setSlot(step.dst, outputTensor.buffer, outputDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: outputDtype,
            });
          }
          break;
        }
        case 'ffn': {
          const srcBuf = getSlot(step.src);

          const activationDtype = resolveStepInputDtype(step, step.src);
          const srcTensor = createTensor(srcBuf, activationDtype, [numTokens, hiddenSize], 'plan_ffn_src');
          const ffnContext = step.inputDtype || step.outputDtype
            ? {
              ...context,
              ffnStepPrecision: {
                inputDtype: step.inputDtype ?? null,
                outputDtype: step.outputDtype ?? null,
              },
            }
            : context;

          let outputTensor;
          const { runMoEFFNGPU, runDenseFFNGPU } = await import('./ffn/index.js');

          const canAutoMoe = config.useMoE && isMoELayer(layerIdx, config);
          const useMoe = selectRuleValue(
            'inference',
            'layer',
            'ffnMode',
            { variant: step.variant, canAutoMoe }
          );
          if (useMoe) {
            outputTensor = await runMoEFFNGPU(layerIdx, srcTensor, numTokens, ffnContext);
          } else {
            outputTensor = await runDenseFFNGPU(layerIdx, srcTensor, numTokens, ffnContext, layerWeights);
          }
          const outputDtype = resolveStepOutputDtype(step, resolveActivationDtype(outputTensor.dtype));
          setSlot(step.dst, outputTensor.buffer, outputDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: outputDtype,
            });
          }
          break;
        }
        case 'residual_add': {
          const aBuf = getSlot(step.a ?? 'state');
          const bBuf = getSlot(step.b ?? 'residual');

          const activationDtype = resolveStepInputDtype(step, step.a ?? 'state');
          const aTensor = createTensor(aBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_a');
          const bTensor = createTensor(bBuf, activationDtype, [numTokens, hiddenSize], 'plan_residual_b');
          const outputTensor = await doResidualAdd(aTensor, bTensor, size, recorder, {
            label: `L${layerIdx}.residual_add`,
            layerIdx,
            executionPolicies: context.executionPolicies ?? null,
          });
          const outputDtype = resolveStepOutputDtype(step, resolveActivationDtype(outputTensor.dtype));
          setSlot(step.dst, outputTensor.buffer, outputDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: outputDtype,
            });
          }
          break;
        }
        case 'cast': {
          const srcBuf = getSlot(step.src);
          const inputDtype = resolveStepInputDtype(step, step.src);
          const srcTensor = createTensor(srcBuf, inputDtype, [numTokens, hiddenSize], 'plan_cast_src');
          if (step.fromDtype) {
            const expected = resolveActivationDtype(step.fromDtype);
            if (inputDtype !== expected) {
              throw new Error(
                `Layer pipeline cast mismatch at L${layerIdx}: fromDtype=${expected}, actual=${inputDtype}`
              );
            }
          }
          const toDtype = resolveActivationDtype(step.toDtype);
          const outputTensor = await doCast(srcTensor, toDtype, recorder);
          setSlot(step.dst, outputTensor.buffer, toDtype);
          if (step.probeStage) {
            await runProbes(step.probeStage, outputTensor.buffer, {
              layerIdx,
              numTokens,
              hiddenSize,
              probes: context.debugProbes,
              recorder,
              operatorDiagnostics: context.operatorDiagnostics,
              dtype: toDtype,
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

    if (hasPerLayerInputBlock(config)) {
      const stateBuffer = getSlot('state');
      const stateDtype = getSlotDtype('state') ?? activationDtype;
      const stateTensor = createTensor(stateBuffer, stateDtype, [numTokens, hiddenSize], 'plan_state');
      const outputTensor = await applyPerLayerInputBlock(
        layerIdx,
        stateTensor,
        numTokens,
        size,
        context,
        layerWeights
      );
      const outputDtype = resolveActivationDtype(outputTensor.dtype);
      setSlot('state', outputTensor.buffer, outputDtype);
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
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: getSlotDtype('state') ?? activationDtype,
  });

  const computeConfig = context.runtimeComputeConfig ?? null;
  const shouldCheckFiniteness = context.finitenessGuardEnabled !== undefined
    ? context.finitenessGuardEnabled
    : shouldRunFinitenessGuard(context.activationDtype, computeConfig);
  if (context.finitenessBuffer && context.activationDtype === 'f16' && shouldCheckFiniteness) {
    recordCheckFiniteness(
      recorder,
      output,
      size,
      context.finitenessBuffer,
      layerIdx,
      context.step,
      context.finitenessAbsThreshold
    );
  }

  return output;
}
