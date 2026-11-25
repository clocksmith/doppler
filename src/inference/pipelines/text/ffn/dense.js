import { runFusedGateUpFFNGPU } from "./fused-gate-up.js";
import { runPackedGateUpFFNGPU } from "./packed-gate-up.js";
import { doMatmul, releaseOrTrack } from "../execution/ops.js";
import { createTensor } from "../../../../gpu/tensor.js";
import { getWeightDtype, isGpuBufferInstance, isWeightBuffer, resolveWeightBufferMaterialization } from "../../../../gpu/weight-buffer.js";
import { getDevice, getKernelCapabilities } from "../../../../gpu/device.js";

import { acquireBuffer, releaseBuffer } from "../../../../memory/buffer-pool.js";

import { isFusedQ4KDisabled } from "../../../../gpu/kernels/matmul.js";
import { log, trace } from "../../../../debug/index.js";
import { isKernelDebugEnabled, dumpTokenVector } from "../debug-utils/tensor.js";

import { applyLoRA } from "../lora-apply.js";
import { getLoRAModule } from "../lora.js";
import { getWeightBuffer } from "../loading/weights.js";
import { runProbes } from "../observation/probes.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";

import { resolveLayerIntermediateSize } from "../config.js";

import { canFuseSplitPrefillF16GateUpPath, resolveFusedGateUpWeights, resolveGateUpPathMode } from "./dense-plan.js";
import { coerceTensorDtype, dispatchActivation, requireFusedWeightDtype, resolveMatmulStepDtype, runDenseUngatedFFNGPU } from "./dense-executor.js";
import { runPairedFFNGPU } from "./paired.js";
import { captureTrainingActivation, captureProjectionInput, enqueueRecordedDenseHealth } from "./activation-observation.js";

// Lazily loaded once: each dynamic import() makes the browser re-walk the module graph.
let fusedGateUpGeluModule = null;
export { canUseNativeF16FusedGateUp, runDenseFFNWithFusedPostNormGPU } from './dense-executor.js';
export { canFuseSplitPrefillF16GateUpPath, resolveDenseFFNFusedPathDtypes, resolveDenseFFNMatmulStepDtype, resolveFusedGateUpPipelineConstants, resolveFusedGateUpVariant, resolveFusedGateUpWeights, resolveGateUpPathMode } from './dense-plan.js';

export async function runDenseFFNGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights
) {
  const channelGain = context.channelGainRequest?.has(layerIdx) ? context.channelGainRequest : null;
  if (channelGain && (inputTensor.precision !== undefined || layerWeights?.gateUp)) {
    throw new Error('Channel gain requires split ordinary f32 SwiGLU.');
  }
  if (inputTensor.precision !== undefined) {
    return runPairedFFNGPU(layerIdx, inputTensor, numTokens, context, layerWeights);
  }
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, recorder } = context;
  const { hiddenSize, hiddenActivation, swigluLimit, useDoubleWideMlp } = config;
  const intermediateSize = resolveLayerIntermediateSize(config, layerIdx);
  channelGain?.validateLayer(layerIdx, context, numTokens, intermediateSize);
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
    return runPackedGateUpFFNGPU({
      layerWeights,
      inputTensor,
      phase,
      layerIdx,
      kernelPath,
      ffnStepPrecision,
      numTokens,
      intermediateSize,
      hiddenSize,
      context,
      recorder,
      lora,
      lastTokenIdx,
      hiddenActivation,
      swigluLimit
    });
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
  // Rotation's declared contract needs the post-SwiGLU tensor before down projection.
  const requiresRotationActivation = channelGain?.transformKind === 'rotation';
  const useFusedGateUp = requiresRotationActivation || (gateUpPathMode === 'split' && !splitPrefillF16FusionAllowed)
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

  if (channelGain && useFusedGateUp) throw new Error('Channel gain requires an explicit split gate/up execution path.');
  if (useFusedGateUp) {
    return runFusedGateUpFFNGPU({
      phase,
      layerIdx,
      kernelPath,
      ffnStepPrecision,
      inputTensor,
      defaultMatmulOutputDtype,
      fusedGateUpWeights,
      layerWeights,
      context,
      recorder,
      gateDtype,
      hiddenSize,
      intermediateSize,
      numTokens,
      hiddenActivation,
      swigluLimit,
      lora
    });
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
      transitionDeclaredBy: 'step-precision',
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
      await (fusedGateUpGeluModule ??= import('../../../../gpu/kernels/fused-gate-up-gelu.js'));
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
        transitionDeclaredBy: 'step-precision',
      });
      downInputOwned = downInputTensor !== fused;
    }
    await captureProjectionInput(context, layerIdx, downInputTensor, numTokens, intermediateSize);
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

  let activatedOutput = channelGain?.mode === 'apply' && channelGain.transformKind === 'gain'
    ? await channelGain.activate(layerIdx, upOutput, gateOutput, numTokens, intermediateSize, swigluLimit, recorder ?? null)
    : await dispatchActivation(hiddenActivation, upOutput, {
    kernelPath,
    phase,
    size: numTokens * intermediateSize,
    gate: gateOutput,
    inputActivation: 'identity',
    swigluLimit,
    label: `L${layerIdx}.ffn_activation`,
    layerIdx,
  }, recorder);

  if (channelGain?.mode === 'apply' && channelGain.transformKind === 'rotation') {
    const originalActivation = activatedOutput;
    try {
      activatedOutput = await channelGain.transform(layerIdx, originalActivation, numTokens, intermediateSize, recorder ?? null);
    } finally {
      releaseOrTrack(recorder, originalActivation.buffer, context.decodeBuffers);
    }
  }

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
  await channelGain?.capture(layerIdx, activatedOutput, numTokens, intermediateSize, recorder ?? null);
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
      transitionDeclaredBy: 'step-precision',
    });
    downInputOwned = downInputTensor !== activatedOutput;
  }
  await captureProjectionInput(context, layerIdx, downInputTensor, numTokens, intermediateSize);
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
