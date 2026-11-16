import { getWeightBuffer } from "../loading/weights.js";
import { selectRuleValue } from "../../../../rules/rule-registry.js";
import { resolveActivationOp, resolveMatmulStepDtype } from "./dense-executor.js";
import { doMatmul, doSiLURowSplit, releaseOrTrack } from "../execution/ops.js";
import { getLoRAModule } from "../lora.js";
import { applyLoRA } from "../lora-apply.js";
import { releaseBuffer } from "../../../../memory/buffer-pool.js";
import { isKernelDebugEnabled, dumpTokenVector } from "../debug-utils/tensor.js";
import { captureTrainingActivation, captureProjectionInput, enqueueRecordedDenseHealth } from "./activation-observation.js";
import { isGpuBufferInstance, isWeightBuffer } from "../../../../gpu/weight-buffer.js";
import { getRuntimeConfig } from "../../../../config/runtime.js";
import { shouldMarkWideTileResidualFused } from "./dense-plan.js";

export async function runPackedGateUpFFNGPU({
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
}) {
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
  await captureProjectionInput(context, layerIdx, activatedOutput, numTokens, intermediateSize);
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
