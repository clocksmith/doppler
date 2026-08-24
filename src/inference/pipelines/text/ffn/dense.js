

import {
  doBiasAdd, doMatmul, doSiLU, doGeLU, doSiLURowSplit, doMatmulRMSNormFused,
  releaseOrTrack
} from '../ops.js';
import { createTensor } from '../../../../gpu/tensor.js';
import {
  getWeightDtype,
  isGpuBufferInstance,
  isWeightBuffer,
  resolveWeightBufferMaterialization,
} from '../../../../gpu/weight-buffer.js';
import { getDevice, getKernelCapabilities } from '../../../../gpu/device.js';
import { getRuntimeConfig } from '../../../../config/runtime.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../../memory/buffer-pool.js';
import {
  runFusedFFN,
  recordFusedFFN,
  runFusedFFNFromRMSNormStats,
  recordFusedFFNFromRMSNormStats,
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
  isFusedQ4KDisabled
} from '../../../../gpu/kernel-selector.js';
import { log, trace, isTraceEnabled } from '../../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector, decodeReadback, getLogitsHealth, shouldDebugLayerOutput } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getWeightBuffer, getNormWeightBuffer, getVectorTensor } from '../weights.js';
import { runProbes } from '../probes.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import {
  getKernelPathMatmulConstants,
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
} from '../../../../config/kernel-path-loader.js';
import { resolveLayerIntermediateSize } from '../config.js';
import { assertImplicitDtypeTransitionAllowed } from '../dtype-contract.js';
import { canFuseSplitPrefillF16GateUpPath, hasQ4KMaterialization, resolveDenseFFNFusedPathDtypes, resolveDenseFFNMatmulStepDtype, resolveFusedGateUpPipelineConstants, resolveFusedGateUpVariant, resolveFusedGateUpWeights, resolveGateUpPathMode } from './dense-plan.js';
import { coerceTensorDtype, dispatchActivation, dispatchFusedGateUp, requireFusedWeightDtype, resolveActivationOp, resolveMatmulStepDtype, runDenseUngatedFFNGPU } from './dense-executor.js';
export { canUseNativeF16FusedGateUp, runDenseFFNWithFusedPostNormGPU } from './dense-executor.js';
export { canFuseSplitPrefillF16GateUpPath, resolveDenseFFNFusedPathDtypes, resolveDenseFFNMatmulStepDtype, resolveFusedGateUpPipelineConstants, resolveFusedGateUpVariant, resolveFusedGateUpWeights, resolveGateUpPathMode } from './dense-plan.js';

async function captureTrainingActivation(context, layerIdx, tensor, numTokens, hiddenSize) {
  const capture = context.trainingCapture;
  if (!capture || capture.layerIdx !== layerIdx || capture.stage !== 'ffn_act') return;
  await capture.capture({
    layerIdx,
    stage: 'ffn_act',
    tensor,
    numTokens,
    hiddenSize,
    recorder: context.recorder ?? null,
  });
}

function enqueueRecordedDenseHealth(context, layerIdx, label, tensor, elementCount) {
  const recorder = context.recorder ?? null;
  if (!recorder || !isTraceEnabled('logits') || !shouldDebugLayerOutput(layerIdx, context.debugLayers)) {
    return;
  }
  if (!tensor?.buffer || !Number.isFinite(elementCount) || elementCount <= 0) {
    return;
  }
  const dtype = tensor.dtype;
  const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype });
  recorder.enqueueCompletionTask(async () => {
    const data = await readBuffer(tensor.buffer, elementCount * bytesPerElement);
    trace.logits(`L${layerIdx}.${label}_HEALTH`, getLogitsHealth(decodeReadback(data, dtype)));
  });
}

function isWideTileQ4KPhaseEnabled(session, phase) {
  return phase === 'decode'
    ? session?.useWideTileQ4KDecode === true
    : session?.useWideTileQ4KPrefill === true;
}

function shouldMarkWideTileResidualFused(session, phase) {
  return getKernelCapabilities().hasF16 === true
    && session?.useWideTileResidualFusion === true
    && session?.retainQ4KMaterialization === true
    && isWideTileQ4KPhaseEnabled(session, phase);
}

export async function runDenseFFNGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights
) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, recorder } = context;
  const { hiddenSize, hiddenActivation, swigluLimit, useDoubleWideMlp } = config;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;
  const ffnStepPrecision = context.ffnStepPrecision ?? null;
  const kernelPath = context.kernelPath ?? null;
  const phase = context.phase ?? (numTokens === 1 ? 'decode' : 'prefill');
  const gateUpPathMode = resolveGateUpPathMode({ kernelPath, phase, layerIdx });

  if (config.gatedActivation === false) {
    return runDenseUngatedFFNGPU(layerIdx, inputTensor, numTokens, context, layerWeights);
  }

  if (layerWeights?.gateUp && layerWeights?.down) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

    const useF16 = inputTensor.dtype === 'f16';
    const defaultMatmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
      useF16,
      fallback: inputTensor.dtype,
    });
    const matmulOutputDtype = resolveMatmulStepDtype(
      'ffn_gate_up',
      phase,
      layerIdx,
      kernelPath,
      defaultMatmulOutputDtype,
      'outputDtype',
      ffnStepPrecision
    );
    const downOutputDtype = resolveMatmulStepDtype(
      'ffn_down',
      phase,
      layerIdx,
      kernelPath,
      'f32',
      'outputDtype',
      ffnStepPrecision
    );
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_gate_up`,
        layerIdx,
        kernelPath,
        outputDtype: matmulOutputDtype,
        role: 'ffn_gate_up',
        executionPolicies: context.executionPolicies ?? null,
      },
      recorder
    );

    const loraGateUp = getLoRAModule(lora, layerIdx, 'gate_up_proj');
    if (loraGateUp) {
      const combined = await applyLoRA(
        inputTensor,
        gateUpOutput,
        loraGateUp,
        { M: numTokens, N: intermediateSize * 2, K: hiddenSize },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== gateUpOutput.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(gateUpOutput.buffer);
        } else {
          releaseBuffer(gateUpOutput.buffer);
        }
        gateUpOutput = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(gateUpOutput.buffer, 'ffn_gate_up', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize * 2,
        dtype: gateUpOutput.dtype,
      });
    }
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_gate_up', gateUpOutput, numTokens * intermediateSize * 2);

    if (!isGpuBufferInstance(layerWeights.gateUp) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseOrTrack(recorder, isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    const activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation: resolveActivationOp(hiddenActivation),
      swigluLimit,
      label: `L${layerIdx}.ffn_activation`,
      layerIdx,
    }, recorder);

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: intermediateSize,
        dtype: activatedOutput.dtype,
      });
    }
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_act', activatedOutput, numTokens * intermediateSize);
    await captureTrainingActivation(context, layerIdx, activatedOutput, numTokens, intermediateSize);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }

    // Opt-in WideTile+residual fusion: if the caller (processFFNStandard)
    // staged a residual tensor on the context AND no LoRA is present on
    // down_proj (LoRA requires the pre-residual output for its add), route
    // this matmul to the q4_fused_widetile_residual variant which produces
    // (ffn_down_result + residual) in one dispatch. Tell the caller via a
    // context flag so processFFNStandard skips its downstream doResidualAdd.
    const pendingResidual = context.__pendingFfnResidualTensor;
    const downLoraProbe = getLoRAModule(lora, layerIdx, 'down_proj');
    const mergedSession = getRuntimeConfig().inference?.session;
    const tryFuseDownResidual = pendingResidual != null
      && !downLoraProbe
      && activatedOutput.dtype === 'f32'
      && downOutputDtype === 'f32'
      && pendingResidual.dtype === 'f32';
    let residualFusedHere = false;
    let output = await doMatmul(
      activatedOutput, downWeight,
      numTokens, hiddenSize, intermediateSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_down`,
        layerIdx,
        kernelPath,
        outputDtype: downOutputDtype,
        role: 'ffn_down',
        executionPolicies: context.executionPolicies ?? null,
        residualTensor: tryFuseDownResidual ? pendingResidual : null,
      },
      recorder
    );
    // Detect whether the fusion fired: if the selected variant for ffn_down
    // matmul was q4_fused_widetile_residual, then output IS post-residual.
    // We infer this cheaply by re-checking the conditions the selector uses.
    // (A cleaner signal would require a return-shape change across all
    // dense.js paths; this local signal is enough for correctness.)
    {
      if (tryFuseDownResidual
          && shouldMarkWideTileResidualFused(mergedSession, phase)
      ) {
        residualFusedHere = true;
        context.__ffnResidualFusedFired = true;
      }
    }

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        activatedOutput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== output.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(output.buffer);
        } else {
          releaseBuffer(output.buffer);
        }
        output = combined;
      }
    }

    if (isKernelDebugEnabled(layerIdx) && !recorder) {
      await dumpTokenVector(output.buffer, 'ffn_down_out', {
        layerIdx,
        tokenIdx: lastTokenIdx,
        rowSize: hiddenSize,
        dtype: output.dtype,
      });
    }
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_down', output, numTokens * hiddenSize);

    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }
    if (recorder) {
      recorder.trackTemporaryBuffer(activatedOutput.buffer);
    } else {
      releaseBuffer(activatedOutput.buffer);
    }

    return output;
  }

  const hasGate = Boolean(layerWeights?.gate);
  const hasUp = Boolean(layerWeights?.up);
  const hasDown = Boolean(layerWeights?.down);
  const hasFusedWeights = Boolean(layerWeights?.gateUp);
  const inputIsSupported = inputTensor.dtype === 'f32' || inputTensor.dtype === 'f16';
  const hasLoRA = Boolean(
    (hasGate ? getLoRAModule(lora, layerIdx, 'gate_proj') : null) ||
    (hasUp ? getLoRAModule(lora, layerIdx, 'up_proj') : null)
  );
  const hiddenSizeAligned32 = hiddenSize % 32 === 0;
  const activationDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: inputTensor.dtype === 'f16',
    fallback: inputTensor.dtype,
  });
  const defaultMatmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16: inputTensor.dtype === 'f16',
    fallback: inputTensor.dtype,
  });
  const fusedGateUpWeights = resolveFusedGateUpWeights(layerWeights, {
    activationDtype,
    hiddenSize,
    kernelPath,
    phase,
    layerIdx,
  });
  const gateDtype = hasGate
    ? requireFusedWeightDtype(fusedGateUpWeights.gateDtype, 'gate')
    : null;
  const upDtype = hasUp
    ? requireFusedWeightDtype(fusedGateUpWeights.upDtype, 'up')
    : null;
  const dtypeMatches = gateDtype != null && upDtype != null && gateDtype === upDtype;
  const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled({ kernelPath });
  const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
  const f16BatchSupported = getKernelCapabilities().hasF16;
  const useFusedGateUpByRule = selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    hasGate,
    hasUp,
    hasDown,
    hasFusedWeights,
    inputIsSupported,
    hasLoRA,
    dtypeMatches,
    dtypeSupported,
    weightDtype: gateDtype,
    hasQ4KMaterialization: fusedGateUpWeights.hasQ4KMaterialization,
    activationDtype,
    f16BatchSupported,
    useLargeBatchF16F32FusedGateUp: context.useLargeBatchF16F32FusedGateUp === true,
    batchSize: numTokens,
    hiddenSizeAligned32,
    useDoubleWideMlp: Boolean(useDoubleWideMlp),
  });
  const splitPrefillF16FusionAllowed = gateUpPathMode === 'split'
    && canFuseSplitPrefillF16GateUpPath({
      kernelPath,
      phase,
      layerIdx,
      gateDtype,
      upDtype,
    });
  const useFusedGateUp = gateUpPathMode === 'split' && !splitPrefillF16FusionAllowed
    ? false
    : useFusedGateUpByRule;
  trace.ffn(
    layerIdx,
    `useFusedGateUp=${useFusedGateUp} gateUpPathMode=${gateUpPathMode} splitPrefillF16FusionAllowed=${splitPrefillF16FusionAllowed} ` +
    `inputDtype=${inputTensor.dtype} activationDtype=${activationDtype} ` +
    `gateDtype=${gateDtype} upDtype=${upDtype} hasQ4KMaterialization=${fusedGateUpWeights.hasQ4KMaterialization} ` +
    `dtypeMatches=${dtypeMatches} dtypeSupported=${dtypeSupported} hiddenSizeAligned32=${hiddenSizeAligned32} ` +
    `largeBatchF16F32FusedGateUp=${context.useLargeBatchF16F32FusedGateUp === true} batchSize=${numTokens}`
  );

  if (useFusedGateUp) {
    const {
      fusedGateUpInputDtype,
      downInputDtype,
    } = resolveDenseFFNFusedPathDtypes({
      phase,
      layerIdx,
      kernelPath,
      ffnStepPrecision,
      fallbackInputDtype: inputTensor.dtype,
      fallbackOutputDtype: defaultMatmulOutputDtype,
    });
    const gateWeight = getWeightBuffer(fusedGateUpWeights.gate ?? layerWeights.gate, 'ffn_gate');
    const upWeight = getWeightBuffer(fusedGateUpWeights.up ?? layerWeights.up, 'ffn_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
    const fusedGateUpPipelineConstants = resolveFusedGateUpPipelineConstants({
      kernelPath,
      phase,
      layerIdx,
    });
    const fusedGateUpVariant = resolveFusedGateUpVariant({ phase });
    const fusedDownOutputDtype = resolveMatmulStepDtype(
      'ffn_down',
      phase,
      layerIdx,
      kernelPath,
      'f32',
      'outputDtype',
      ffnStepPrecision
    );
    let fusedInput = inputTensor;
    let fusedInputOwned = false;
    if (fusedGateUpInputDtype && fusedGateUpInputDtype !== inputTensor.dtype) {
      fusedInput = await coerceTensorDtype(inputTensor, fusedGateUpInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_gate_up_input',
        transitionDeclaredBy: 'step_precision',
      });
      fusedInputOwned = fusedInput !== inputTensor;
    }
    const fusedOutput = await dispatchFusedGateUp({
      inputTensor: fusedInput, gateWeight, upWeight, gateDtype,
      hiddenSize, intermediateSize, numTokens,
      hiddenActivation, swigluLimit, recorder,
      executionPolicies: context.executionPolicies ?? null,
      normStats: context.__pendingFfnInputNormStats ?? null,
      pipelineConstants: fusedGateUpPipelineConstants,
      variant: fusedGateUpVariant,
    });
    await runProbes('ffn_act', fusedOutput.buffer, {
      layerIdx,
      numTokens,
      hiddenSize: intermediateSize,
      probes: context.debugProbes,
      recorder,
      operatorDiagnostics: context.operatorDiagnostics,
      dtype: fusedOutput.dtype,
    });
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_fused_gate_up', fusedOutput, numTokens * intermediateSize);
    await captureTrainingActivation(context, layerIdx, fusedOutput, numTokens, intermediateSize);

    let downInput = fusedOutput;
    if (downInputDtype && fusedOutput.dtype !== downInputDtype) {
      downInput = await coerceTensorDtype(fusedOutput, downInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_down_input',
        transitionDeclaredBy: 'step_precision',
      });
      if (recorder) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
    }

    if (!isGpuBufferInstance(layerWeights.gate) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }
    if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }

    // Opt-in WideTile+residual fusion (fused-gate-up path).
    const pendingResidualFused = context.__pendingFfnResidualTensor;
    const downLoraProbeFused = getLoRAModule(lora, layerIdx, 'down_proj');
    const mergedSessionFused = getRuntimeConfig().inference?.session;
    const tryFuseDownResidualFused = pendingResidualFused != null
      && !downLoraProbeFused
      && downInput.dtype === 'f32'
      && fusedDownOutputDtype === 'f32'
      && pendingResidualFused.dtype === 'f32';
    let output = await doMatmul(
      downInput,
      downWeight,
      numTokens,
      hiddenSize,
      intermediateSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_down`,
        layerIdx,
        kernelPath,
        outputDtype: fusedDownOutputDtype,
        role: 'ffn_down',
        executionPolicies: context.executionPolicies ?? null,
        residualTensor: tryFuseDownResidualFused ? pendingResidualFused : null,
      },
      recorder
    );
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_down', output, numTokens * hiddenSize);
    {
      if (tryFuseDownResidualFused
          && shouldMarkWideTileResidualFused(mergedSessionFused, phase)
      ) {
        context.__ffnResidualFusedFired = true;
      }
    }

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        downInput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder,
        { kernelPath }
      );
      if (combined.buffer !== output.buffer) {
        if (recorder) {
          recorder.trackTemporaryBuffer(output.buffer);
        } else {
          releaseBuffer(output.buffer);
        }
        output = combined;
      }
    }

    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }

    if (recorder) {
      if (downInput !== fusedOutput) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
      if (fusedInputOwned) {
        recorder.trackTemporaryBuffer(fusedInput.buffer);
      }
      recorder.trackTemporaryBuffer(fusedOutput.buffer);
    } else {
      if (downInput !== fusedOutput) {
        releaseBuffer(downInput.buffer);
      }
      if (fusedInputOwned) {
        releaseBuffer(fusedInput.buffer);
      }
      releaseBuffer(fusedOutput.buffer);
    }

    return output;
  }

  if (context.__pendingFfnInputNormStats) {
    throw new Error(
      `Layer ${layerIdx} has precomputed FFN input norm stats, but fused gate/up was not selected.`
    );
  }

  if (!layerWeights?.gate || !layerWeights?.up || !layerWeights?.down) {
    log.warn('Layer', `L${layerIdx} FFN: no weights found`);
    const bytesPerElement = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: inputTensor.dtype });
    const byteSize = numTokens * hiddenSize * bytesPerElement;
    const outputBuffer = acquireBuffer(byteSize, undefined, 'ffn_output');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputTensor.buffer, 0, outputBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);
    return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'ffn_output_copy');
  }

  const gateInputDtype = resolveMatmulStepDtype(
    'ffn_gate',
    phase,
    layerIdx,
    kernelPath,
    inputTensor.dtype,
    'inputDtype',
    ffnStepPrecision
  );
  const gateOutputDtype = resolveMatmulStepDtype(
    'ffn_gate',
    phase,
    layerIdx,
    kernelPath,
    defaultMatmulOutputDtype,
    'outputDtype',
    ffnStepPrecision
  );
  const upInputDtype = resolveMatmulStepDtype(
    'ffn_up',
    phase,
    layerIdx,
    kernelPath,
    inputTensor.dtype,
    'inputDtype',
    ffnStepPrecision
  );
  const upOutputDtype = resolveMatmulStepDtype(
    'ffn_up',
    phase,
    layerIdx,
    kernelPath,
    defaultMatmulOutputDtype,
    'outputDtype',
    ffnStepPrecision
  );
  const downOutputDtype = resolveMatmulStepDtype(
    'ffn_down',
    phase,
    layerIdx,
    kernelPath,
    'f32',
    'outputDtype',
    ffnStepPrecision
  );
  const downInputDtype = resolveMatmulStepDtype(
    'ffn_down',
    phase,
    layerIdx,
    kernelPath,
    downOutputDtype,
    'inputDtype',
    ffnStepPrecision
  );
  const sharedInputDtype = gateInputDtype === upInputDtype ? gateInputDtype : null;
  let sharedInputTensor = inputTensor;
  let sharedInputOwned = false;
  if (sharedInputDtype && sharedInputDtype !== inputTensor.dtype) {
    sharedInputTensor = await coerceTensorDtype(inputTensor, sharedInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_shared_input',
      transitionDeclaredBy: 'step_precision',
    });
    sharedInputOwned = sharedInputTensor !== inputTensor;
  }
  // Opt-in fused gate + up + GeGLU path. Replaces 3 separate dispatches
  // (gate_proj + up_proj + gelu activation) with a single fused kernel when
  // preconditions match: prefill (M>1), f16-materialisable weights + f16
  // activations, gelu activation, no LoRA on gate/up. Gated by
  // runtime.inference.session.useFusedGateUpGelu (default false).
  const gateF16 = resolveWeightBufferMaterialization(layerWeights.gate, 'f16');
  const upF16 = resolveWeightBufferMaterialization(layerWeights.up, 'f16');
  const gateF16Dtype = getWeightDtype(gateF16);
  const upF16Dtype = getWeightDtype(upF16);
  const earlyLoraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  const earlyLoraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  const fusedGateUpGeluCandidate = context.useFusedGateUpGelu === true
    && numTokens > 1
    && hiddenActivation === 'gelu'
    && !earlyLoraGate && !earlyLoraUp
    && sharedInputTensor.dtype === 'f16'
    && gateF16Dtype === 'f16'
    && upF16Dtype === 'f16';
  if (fusedGateUpGeluCandidate) {
    const { runFusedGateUpGelu, recordFusedGateUpGelu } =
      await import('../../../../gpu/kernels/fused-gate-up-gelu.js');
    const fused = recorder
      ? await recordFusedGateUpGelu(recorder, sharedInputTensor, gateF16, upF16, {
        M: numTokens,
        hiddenSize,
        intermediateSize,
        transposeB: true,
      })
      : await runFusedGateUpGelu(sharedInputTensor, gateF16, upF16, {
        M: numTokens,
        hiddenSize,
        intermediateSize,
        transposeB: true,
      });
    if (sharedInputOwned) {
      if (recorder) { recorder.trackTemporaryBuffer(sharedInputTensor.buffer); }
      else { releaseBuffer(sharedInputTensor.buffer); }
    }
    // Proceed directly to down_proj with the fused activation output.
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
    let downInputTensor = fused;
    let downInputOwned = false;
    if (downInputDtype && fused.dtype !== downInputDtype) {
      downInputTensor = await coerceTensorDtype(fused, downInputDtype, recorder, {
        executionPolicies: context.executionPolicies ?? null,
        op: 'ffn_down_input',
        transitionDeclaredBy: 'step_precision',
      });
      downInputOwned = downInputTensor !== fused;
    }
    let outFused = await doMatmul(
      downInputTensor,
      downWeight,
      numTokens,
      hiddenSize,
      intermediateSize,
      {
        transposeB: 'auto',
        label: `L${layerIdx}.ffn_down`,
        layerIdx,
        kernelPath,
        outputDtype: downOutputDtype,
        role: 'ffn_down',
        executionPolicies: context.executionPolicies ?? null,
      },
      recorder
    );
    enqueueRecordedDenseHealth(context, layerIdx, 'ffn_down', outFused, numTokens * hiddenSize);
    if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }
    if (downInputOwned) {
      if (recorder) { recorder.trackTemporaryBuffer(downInputTensor.buffer); }
      else { releaseBuffer(downInputTensor.buffer); }
    }
    if (recorder) { recorder.trackTemporaryBuffer(fused.buffer); }
    else { releaseBuffer(fused.buffer); }
    return outFused;
  }
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(
    gateInputDtype === sharedInputTensor.dtype ? sharedInputTensor : inputTensor,
    gateWeight,
    numTokens,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_gate`,
      layerIdx,
      kernelPath,
      outputDtype: gateOutputDtype,
      role: 'ffn_gate',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  if (!isGpuBufferInstance(layerWeights.gate) && !isWeightBuffer(layerWeights.gate)) {
    releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
  }

  const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
  if (loraGate) {
    const combined = await applyLoRA(
      inputTensor,
      gateOutput,
      loraGate,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== gateOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput.buffer);
      } else {
        releaseBuffer(gateOutput.buffer);
      }
      gateOutput = combined;
    }
  }

  const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
  let upOutput = await doMatmul(
    upInputDtype === sharedInputTensor.dtype ? sharedInputTensor : inputTensor,
    upWeight,
    numTokens,
    intermediateSize,
    hiddenSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_up`,
      layerIdx,
      kernelPath,
      outputDtype: upOutputDtype,
      role: 'ffn_up',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  if (!isGpuBufferInstance(layerWeights.up) && !isWeightBuffer(layerWeights.up)) {
    releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
  }

  const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
  if (loraUp) {
    const combined = await applyLoRA(
      inputTensor,
      upOutput,
      loraUp,
      { M: numTokens, N: intermediateSize, K: hiddenSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== upOutput.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(upOutput.buffer);
      } else {
        releaseBuffer(upOutput.buffer);
      }
      upOutput = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(gateOutput.buffer, 'ffn_gate', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: gateOutput.dtype,
    });
    await dumpTokenVector(upOutput.buffer, 'ffn_up', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: upOutput.dtype,
    });
  }

  await runProbes('ffn_gate', gateOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: gateOutput.dtype,
  });
  await runProbes('ffn_up', upOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: upOutput.dtype,
  });
  enqueueRecordedDenseHealth(context, layerIdx, 'ffn_gate', gateOutput, numTokens * intermediateSize);
  enqueueRecordedDenseHealth(context, layerIdx, 'ffn_up', upOutput, numTokens * intermediateSize);

  const activatedOutput = await dispatchActivation(hiddenActivation, upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
    inputActivation: 'identity',
    swigluLimit,
    label: `L${layerIdx}.ffn_activation`,
    layerIdx,
  }, recorder);

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(activatedOutput.buffer, 'ffn_activated', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: intermediateSize,
      dtype: activatedOutput.dtype,
    });
  }

  await runProbes('ffn_act', activatedOutput.buffer, {
    layerIdx,
    numTokens,
    hiddenSize: intermediateSize,
    probes: context.debugProbes,
    recorder,
    operatorDiagnostics: context.operatorDiagnostics,
    dtype: activatedOutput.dtype,
  });
  enqueueRecordedDenseHealth(context, layerIdx, 'ffn_act', activatedOutput, numTokens * intermediateSize);
  await captureTrainingActivation(context, layerIdx, activatedOutput, numTokens, intermediateSize);

  if (recorder) {
    recorder.trackTemporaryBuffer(gateOutput.buffer);
    recorder.trackTemporaryBuffer(upOutput.buffer);
  } else {
    releaseBuffer(gateOutput.buffer);
    releaseBuffer(upOutput.buffer);
  }
  if (sharedInputOwned) {
    if (recorder) {
      recorder.trackTemporaryBuffer(sharedInputTensor.buffer);
    } else {
      releaseBuffer(sharedInputTensor.buffer);
    }
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let downInputTensor = activatedOutput;
  let downInputOwned = false;
  if (downInputDtype && activatedOutput.dtype !== downInputDtype) {
    downInputTensor = await coerceTensorDtype(activatedOutput, downInputDtype, recorder, {
      executionPolicies: context.executionPolicies ?? null,
      op: 'ffn_down_input',
      transitionDeclaredBy: 'step_precision',
    });
    downInputOwned = downInputTensor !== activatedOutput;
  }
  let output = await doMatmul(
    downInputTensor,
    downWeight,
    numTokens,
    hiddenSize,
    intermediateSize,
    {
      transposeB: 'auto',
      label: `L${layerIdx}.ffn_down`,
      layerIdx,
      kernelPath,
      outputDtype: downOutputDtype,
      role: 'ffn_down',
      executionPolicies: context.executionPolicies ?? null,
    },
    recorder
  );
  enqueueRecordedDenseHealth(context, layerIdx, 'ffn_down', output, numTokens * hiddenSize);

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    const combined = await applyLoRA(
      downInputTensor,
      output,
      loraDown,
      { M: numTokens, N: hiddenSize, K: intermediateSize },
      getWeightBuffer,
      recorder,
      { kernelPath }
    );
    if (combined.buffer !== output.buffer) {
      if (recorder) {
        recorder.trackTemporaryBuffer(output.buffer);
      } else {
        releaseBuffer(output.buffer);
      }
      output = combined;
    }
  }

  if (isKernelDebugEnabled(layerIdx) && !recorder) {
    await dumpTokenVector(output.buffer, 'ffn_down_out', {
      layerIdx,
      tokenIdx: lastTokenIdx,
      rowSize: hiddenSize,
      dtype: output.dtype,
    });
  }

  if (!isGpuBufferInstance(layerWeights.down) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (downInputOwned) {
    if (recorder) {
      recorder.trackTemporaryBuffer(downInputTensor.buffer);
    } else {
      releaseBuffer(downInputTensor.buffer);
    }
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

  return output;
}
