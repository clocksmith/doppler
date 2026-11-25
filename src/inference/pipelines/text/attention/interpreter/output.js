import { prepareAttentionProjectionInput } from "../output-projection.js";
import { assertAttentionDtypeTransitionAllowed } from "../interpreter/immediate.js";
import { recordCastF16ToF32, recordCastF32ToF16 } from "../../../../../gpu/kernels/cast.js";
import { getLoRAModule } from "../../lora.js";
import { isGpuBufferInstance, isWeightBuffer, getWeightDtype } from "../../../../../gpu/weight-buffer.js";
import { recordMatmulResidualFused } from "../../../../../gpu/kernels/fused_matmul_residual.js";
import { recordMatmul } from "../../../../../gpu/kernels/matmul.js";
import { applyLoRA } from "../../lora-apply.js";
import { getVectorTensor } from "../../loading/weights.js";
import { recordBiasAdd } from "../../../../../gpu/kernels/residual.js";
import { enqueueRecordedTensorHealth, shouldTraceRecordedHealth } from "../recorded-health.js";
import { finalizeRecordedAttentionSuccess } from "../interpreter/recorded.js";

export async function recordAttentionOutputProjection({
  attnForProjection,
  layerWeights,
  getWeightBuffer,
  oProjInputDtype,
  state,
  recorder,
  lora,
  layerIdx,
  attentionPlan,
  residualTensor,
  hiddenSize,
  numHeads,
  headDim,
  numTokens,
  kernelPath,
  oProjOutputDtype,
  resourceScope,
  debugFlags,
  kTensor,
  qGateTensor,
  qTensor,
  retainSharedKvBuffers,
  vTensor
}) {
  let output = null;
  let finalOutput = null;
  let oProjInputTemp = null;
  try {
    // 6. Output projection (with optional fused residual for decode)

    output = null;
    let residualFused = false;
    let oProjInput = attnForProjection;
    oProjInputTemp = null;
    if (layerWeights.oProj && getWeightBuffer) {
      ({ oProjInput, oProjInputTemp } = await prepareAttentionProjectionInput(
        attnForProjection,
        oProjInputDtype,
        (tensor) => {
          assertAttentionDtypeTransitionAllowed(
            state,
            tensor.dtype,
            oProjInputDtype,
            'Attention output projection would change activations implicitly.',
            'step-precision'
          );
          return oProjInputDtype === 'f16'
            ? recordCastF32ToF16(recorder, tensor)
            : recordCastF16ToF32(recorder, tensor);
        }
      ));
      const oProjBuf = getWeightBuffer(layerWeights.oProj, 'o_proj');

      // Use fused o_proj + residual for decode when possible
      // Note: dtype from WeightBuffer metadata (buffer-dtypes WeakMap removed)
      const oProjDtype = getWeightDtype(oProjBuf);
      const canUseFused = attentionPlan.outputProjection.residualMode === 'matmul-residual';
      const canUseWideTileResidual =
        attentionPlan.outputProjection.residualMode === 'wide-tile-epilogue';
      if (
        attentionPlan.outputProjection.weightDtype != null
        && oProjDtype !== attentionPlan.outputProjection.weightDtype
      ) {
        throw new Error(
          `Attention output projection dtype changed after planning at layer ${layerIdx}: ` +
          `plan=${attentionPlan.outputProjection.weightDtype}, bound=${oProjDtype}.`
        );
      }
      if (canUseFused && residualTensor) {
        // FUSED PATH: o_proj matmul + residual add in one dispatch
        output = await recordMatmulResidualFused(recorder, oProjInput, oProjBuf, residualTensor, {
          N: hiddenSize,
          K: numHeads * headDim,
        });
        residualFused = true;
      } else {
        // STANDARD PATH: o_proj matmul only unless Q4K WideTile fuses residual in the epilogue
        output = await recordMatmul(recorder, oProjInput, oProjBuf, numTokens, hiddenSize, numHeads * headDim, {
          transposeB: 'auto',
          role: 'o_proj',
          layerIdx,
          kernelPath,
          outputDtype: oProjOutputDtype,
          executionPolicies: state.executionPolicies ?? null,
          residualTensor: canUseWideTileResidual ? residualTensor : null,
        });
        residualFused = canUseWideTileResidual;
      }
      // Release temporary buffer if we created it (original was not already on GPU)
      if (!isGpuBufferInstance(layerWeights.oProj) && !isWeightBuffer(layerWeights.oProj)) {
        resourceScope.release(isWeightBuffer(oProjBuf) ? oProjBuf.buffer : oProjBuf);
      }
    } else {
      output = oProjInput;
    }

    // Apply LoRA to output projection if present (only if not using fused path)
    if (!residualFused) {
      const loraO = getLoRAModule(lora, layerIdx, 'o_proj');
      if (loraO && getWeightBuffer) {
        const combined = await applyLoRA(
          oProjInput,
          output,
          loraO,
          { M: numTokens, N: hiddenSize, K: numHeads * headDim },
          getWeightBuffer,
          recorder,
          { kernelPath }
        );
        if (combined.buffer !== output.buffer) {
          resourceScope.release(output.buffer);
          output = combined;
        }
      }
    }

    if (layerWeights.oProjBias) {
      const { tensor: oProjBiasTensor, owned } = getVectorTensor(
        layerWeights.oProjBias,
        `L${layerIdx}.o_proj_bias`,
        hiddenSize,
        {},
        debugFlags
      );
      output = await recordBiasAdd(
        recorder,
        output,
        oProjBiasTensor,
        numTokens,
        hiddenSize,
        {
          label: `L${layerIdx}.o_proj_bias`,
          layerIdx,
          executionPolicies: state.executionPolicies ?? null,
        }
      );
      if (owned) resourceScope.release(oProjBiasTensor.buffer);
    }

    finalOutput = output;
    if (shouldTraceRecordedHealth(layerIdx, debugFlags)) {
      enqueueRecordedTensorHealth(
        recorder,
        `L${layerIdx}.o_proj_HEALTH`,
        output,
        output.dtype,
        numTokens * hiddenSize
      );
    }

    const buffersToTrack = [];
    if (output.buffer !== attnForProjection.buffer) {
      buffersToTrack.push(attnForProjection.buffer);
    }
    if (oProjInputTemp && oProjInputTemp.buffer !== attnForProjection.buffer) {
      buffersToTrack.push(oProjInputTemp.buffer);
    }
    if (output.dtype !== oProjOutputDtype) {
      assertAttentionDtypeTransitionAllowed(state, output.dtype, oProjOutputDtype, 'Attention output would change implicitly before leaving the layer.');
      const coercedOutput = oProjOutputDtype === 'f16'
        ? await recordCastF32ToF16(recorder, output)
        : await recordCastF16ToF32(recorder, output);
      buffersToTrack.push(output.buffer);
      finalOutput = coercedOutput;
      resourceScope.register(finalOutput.buffer, 'coerced-attention-output', 'submitOwned');
    }

    finalizeRecordedAttentionSuccess({
      attentionPlan,
      buffersToTrack,
      finalOutput,
      kTensor,
      qGateTensor,
      qTensor,
      resourceScope,
      retainSharedKvBuffers,
      state,
      vTensor,
    });
    return { output: finalOutput, residualFused };

  } catch (error) {
    const release = buffer => resourceScope.release(buffer, 'attention-failure');
    if (finalOutput?.buffer && finalOutput.buffer !== output?.buffer) release(finalOutput.buffer);
    if (output?.buffer && output.buffer !== attnForProjection?.buffer) release(output.buffer);
    if (oProjInputTemp?.buffer) release(oProjInputTemp.buffer);
    throw error;
  }
}
