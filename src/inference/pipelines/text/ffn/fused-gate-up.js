import { shouldMarkWideTileResidualFused, resolveDenseFFNFusedPathDtypes, resolveFusedGateUpPipelineConstants, resolveFusedGateUpVariant } from "./dense-plan.js";
import { getWeightBuffer } from "../loading/weights.js";
import { coerceTensorDtype, dispatchFusedGateUp, resolveMatmulStepDtype } from "./dense-executor.js";
import { borrowDecodeFfnIntermediate } from "./decode-intermediate.js";
import { runProbes } from "../observation/probes.js";
import { captureTrainingActivation, captureProjectionInput, enqueueRecordedDenseHealth } from "./activation-observation.js";
import { isGpuBufferInstance, isWeightBuffer } from "../../../../gpu/weight-buffer.js";
import { doMatmul, releaseOrTrack } from "../execution/ops.js";
import { getLoRAModule } from "../lora.js";
import { getRuntimeConfig } from "../../../../config/runtime.js";
import { applyLoRA } from "../lora-apply.js";
import { releaseBuffer } from "../../../../memory/buffer-pool.js";

export async function runFusedGateUpFFNGPU({
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
}) {
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
  const fusedRuntimeSession = context.resolvedRuntimeSession?.runtime?.session;
  const fusedGateUpPipelineConstants = resolveFusedGateUpPipelineConstants({
    runtimeSession: fusedRuntimeSession,
    kernelPath,
    phase,
    layerIdx,
  });
  const fusedGateUpVariant = resolveFusedGateUpVariant({ phase, runtimeSession: fusedRuntimeSession });
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
      transitionDeclaredBy: 'step-precision',
    });
    fusedInputOwned = fusedInput !== inputTensor;
  }
  const fusedOutput = await dispatchFusedGateUp({
    inputTensor: fusedInput, gateWeight, upWeight, gateDtype,
    hiddenSize, intermediateSize, numTokens,
    hiddenActivation, swigluLimit, recorder, layerIdx,
    executionPolicies: context.executionPolicies ?? null,
    normStats: context.__pendingFfnInputNormStats ?? null,
    pipelineConstants: fusedGateUpPipelineConstants,
    variant: fusedGateUpVariant,
    outputBuffer: borrowDecodeFfnIntermediate(context, numTokens, intermediateSize, fusedInput.dtype),
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
      transitionDeclaredBy: 'step-precision',
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
  await captureProjectionInput(context, layerIdx, downInput, numTokens, intermediateSize);
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
    releaseOrTrack(recorder, fusedOutput.buffer, context.decodeBuffers);
  } else {
    if (downInput !== fusedOutput) {
      releaseBuffer(downInput.buffer);
    }
    if (fusedInputOwned) {
      releaseBuffer(fusedInput.buffer);
    }
    releaseOrTrack(null, fusedOutput.buffer, context.decodeBuffers);
  }

  return output;

}
