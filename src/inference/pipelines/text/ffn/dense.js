

import {
  doMatmul, doSiLU, doGeLU, doSiLURowSplit, doMatmulRMSNormFused,
  releaseOrTrack
} from '../ops.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { isWeightBuffer } from '../../../../gpu/weight-buffer.js';
import { getDevice, getKernelCapabilities } from '../../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../../memory/buffer-pool.js';
import {
  runFusedFFN,
  recordFusedFFN,
  castF16ToF32,
  castF32ToF16,
  recordCastF16ToF32,
  recordCastF32ToF16,
  isFusedQ4KDisabled
} from '../../../../gpu/kernel-selector.js';
import { log } from '../../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getWeightBuffer, getNormWeightBuffer } from '../weights.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';


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
  const { hiddenSize, intermediateSize, hiddenActivation, swigluLimit } = config;
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;

  if (layerWeights?.gateUp && layerWeights?.down) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

    const useF16 = inputTensor.dtype === 'f16';
    const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
      useF16,
      fallback: inputTensor.dtype,
    });
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_gate_up`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_gate_up' },
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
        recorder
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

    if (!(layerWeights.gateUp instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseOrTrack(recorder, isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    const activation = selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
    const activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
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

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }

    let output = await doMatmul(
      activatedOutput, downWeight,
      numTokens, hiddenSize, intermediateSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_down' },
      recorder
    );

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        activatedOutput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder
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

    if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
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
  const gateDtype = hasGate && isWeightBuffer(layerWeights.gate) ? layerWeights.gate.dtype : (hasGate ? 'f32' : null);
  const upDtype = hasUp && isWeightBuffer(layerWeights.up) ? layerWeights.up.dtype : (hasUp ? 'f32' : null);
  const dtypeMatches = gateDtype != null && upDtype != null && gateDtype === upDtype;
  const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled();
  const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
  const f16BatchSupported = gateDtype !== 'f16' || numTokens === 1;
  const useFusedGateUp = selectRuleValue('inference', 'ffn', 'useFusedGateUp', {
    hasGate,
    hasUp,
    hasDown,
    hasFusedWeights,
    inputIsSupported,
    hasLoRA,
    dtypeMatches,
    dtypeSupported,
    f16BatchSupported,
  });

  if (useFusedGateUp) {
    const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
    const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
    const useF16 = inputTensor.dtype === 'f16';
    const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
      useF16,
      fallback: inputTensor.dtype,
    });
    // f16_native variant accepts f16 input directly — skip the cast
    const useNativeF16Fused = inputTensor.dtype === 'f16' && gateDtype === 'f16' && getKernelCapabilities().hasF16;
    const fusedInput = (!useNativeF16Fused && inputTensor.dtype === 'f16')
      ? (recorder ? await recordCastF16ToF32(recorder, inputTensor) : await castF16ToF32(inputTensor))
      : inputTensor;

    if (recorder && fusedInput !== inputTensor) {
      recorder.trackTemporaryBuffer(fusedInput.buffer);
    }

    const activation = selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
    const fusedOutput = recorder
      ? await recordFusedFFN(
        recorder,
        fusedInput,
        gateWeight,
        upWeight,
        hiddenSize,
        intermediateSize,
        { batchSize: numTokens, activation, swigluLimit }
      )
      : await runFusedFFN(
        fusedInput,
        gateWeight,
        upWeight,
        hiddenSize,
        intermediateSize,
        { batchSize: numTokens, activation, swigluLimit }
      );

    if (!recorder && fusedInput !== inputTensor) {
      releaseBuffer(fusedInput.buffer);
    }

    let downInput = fusedOutput;
    if (matmulOutputDtype === 'f16' && fusedOutput.dtype !== 'f16') {
      downInput = recorder
        ? await recordCastF32ToF16(recorder, fusedOutput)
        : await castF32ToF16(fusedOutput);
      if (recorder) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
    }

    if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }
    if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }

    let output = await doMatmul(
      downInput,
      downWeight,
      numTokens,
      hiddenSize,
      intermediateSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_down' },
      recorder
    );

    const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
    if (loraDown) {
      const combined = await applyLoRA(
        downInput,
        output,
        loraDown,
        { M: numTokens, N: hiddenSize, K: intermediateSize },
        getWeightBuffer,
        recorder
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

    if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
      releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
    }

    if (recorder) {
      if (downInput !== fusedOutput) {
        recorder.trackTemporaryBuffer(downInput.buffer);
      }
      recorder.trackTemporaryBuffer(fusedOutput.buffer);
    } else {
      if (downInput !== fusedOutput) {
        releaseBuffer(downInput.buffer);
      }
      releaseBuffer(fusedOutput.buffer);
    }

    return output;
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

  const useF16 = inputTensor.dtype === 'f16';
  const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16,
    fallback: inputTensor.dtype,
  });
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(inputTensor, gateWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_gate`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_gate' }, recorder);
  if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
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
      recorder
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
  let upOutput = await doMatmul(inputTensor, upWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_up`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_up' }, recorder);
  if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
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
      recorder
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

  const activationFn = {
    gelu: doGeLU,
    silu: doSiLU,
  }[selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation })];
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
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

  if (recorder) {
    recorder.trackTemporaryBuffer(gateOutput.buffer);
    recorder.trackTemporaryBuffer(upOutput.buffer);
  } else {
    releaseBuffer(gateOutput.buffer);
    releaseBuffer(upOutput.buffer);
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  let output = await doMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: matmulOutputDtype, role: 'ffn_down' }, recorder);

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    const combined = await applyLoRA(
      activatedOutput,
      output,
      loraDown,
      { M: numTokens, N: hiddenSize, K: intermediateSize },
      getWeightBuffer,
      recorder
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

  if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

  return output;
}


export async function runDenseFFNWithFusedPostNormGPU(
  layerIdx,
  inputTensor,
  numTokens,
  context,
  layerWeights,
  residualTensor,
  eps,
  transposeB,
  outputBuffer
) {
  const device = getDevice();
  if (!device) throw new Error('No GPU device');

  const { config, weightConfig, debugFlags, recorder } = context;
  const { hiddenSize, intermediateSize, hiddenActivation, swigluLimit } = config;
  const lora = context.lora || null;

  if (!layerWeights.down || !layerWeights.postFeedforwardNorm) {
    throw new Error('Missing down or norm weights');
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

  
  let activatedOutput;
  const useF16 = inputTensor.dtype === 'f16';
  const matmulOutputDtype = selectRuleValue('shared', 'dtype', 'f16OrFallbackByFlag', {
    useF16,
    fallback: inputTensor.dtype,
  });

  if (layerWeights.gateUp) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      {
        transposeB: 'auto',
        outputDtype: matmulOutputDtype,
        role: 'ffn_gate_up',
        label: `L${layerIdx}.ffn_gate_up`,
        layerIdx,
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
        recorder
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

    if (!(layerWeights.gateUp instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gateUp)) {
      releaseOrTrack(recorder, isWeightBuffer(gateUpWeight) ? gateUpWeight.buffer : gateUpWeight);
    }

    const activation = selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
    activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
      swigluLimit,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }
  } else {
    const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
    const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
    const gateDtype = isWeightBuffer(layerWeights.gate) ? layerWeights.gate.dtype : 'f32';
    const upDtype = isWeightBuffer(layerWeights.up) ? layerWeights.up.dtype : 'f32';
    const hasLoRAGate = Boolean(getLoRAModule(lora, layerIdx, 'gate_proj'));
    const hasLoRAUp = Boolean(getLoRAModule(lora, layerIdx, 'up_proj'));
    const dtypeMatches = gateDtype != null && upDtype != null && gateDtype === upDtype;
    const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled();
    const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
    const canUseFusedGateUp = !hasLoRAGate && !hasLoRAUp && dtypeMatches && dtypeSupported;

    if (canUseFusedGateUp) {
      const useNativeF16Fused = inputTensor.dtype === 'f16' && gateDtype === 'f16' && getKernelCapabilities().hasF16;
      const fusedInput = (!useNativeF16Fused && inputTensor.dtype === 'f16')
        ? (recorder ? await recordCastF16ToF32(recorder, inputTensor) : await castF16ToF32(inputTensor))
        : inputTensor;

      if (recorder && fusedInput !== inputTensor) {
        recorder.trackTemporaryBuffer(fusedInput.buffer);
      }

      const activation = selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation });
      activatedOutput = recorder
        ? await recordFusedFFN(
          recorder,
          fusedInput,
          gateWeight,
          upWeight,
          hiddenSize,
          intermediateSize,
          { batchSize: numTokens, activation, swigluLimit }
        )
        : await runFusedFFN(
          fusedInput,
          gateWeight,
          upWeight,
          hiddenSize,
          intermediateSize,
          { batchSize: numTokens, activation, swigluLimit }
        );

      if (!recorder && fusedInput !== inputTensor) {
        releaseBuffer(fusedInput.buffer);
      }
    } else {
      const gateOutput = await doMatmul(
        inputTensor, gateWeight,
        numTokens, intermediateSize, hiddenSize,
        {
          transposeB: 'auto',
          outputDtype: matmulOutputDtype,
          role: 'ffn_gate',
          label: `L${layerIdx}.ffn_gate`,
          layerIdx,
        },
        recorder
      );

      const upOutput = await doMatmul(
        inputTensor, upWeight,
        numTokens, intermediateSize, hiddenSize,
        {
          transposeB: 'auto',
          outputDtype: matmulOutputDtype,
          role: 'ffn_up',
          label: `L${layerIdx}.ffn_up`,
          layerIdx,
        },
        recorder
      );

      const activationFn = {
        gelu: doGeLU,
        silu: doSiLU,
      }[selectRuleValue('inference', 'ffn', 'activationOp', { hiddenActivation })];
      activatedOutput = await activationFn(upOutput, {
        size: numTokens * intermediateSize,
        gate: gateOutput,
        swigluLimit,
      }, recorder);

      if (recorder) {
        recorder.trackTemporaryBuffer(gateOutput.buffer);
        recorder.trackTemporaryBuffer(upOutput.buffer);
      } else {
        releaseBuffer(gateOutput.buffer);
        releaseBuffer(upOutput.buffer);
      }
    }

    if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }
    if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }
  }

  const outputTensor = await doMatmulRMSNormFused(
    activatedOutput,
    downWeight,
    normWeightBuf,
    {
      N: hiddenSize,
      K: intermediateSize,
      eps,
      residual: residualTensor,
      outputBuffer,
      transposeB,
      label: `L${layerIdx}.ffn_down`,
      rmsNormWeightOffset: weightConfig.rmsNormWeightOffset,
    },
    recorder
  );

  const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
  if (loraDown) {
    log.warn('Layer', `L${layerIdx} LoRA down_proj with fused kernel not yet optimized`);
  }

  if (!(layerWeights.down instanceof GPUBuffer) && !isWeightBuffer(layerWeights.down)) {
    releaseOrTrack(recorder, isWeightBuffer(downWeight) ? downWeight.buffer : downWeight);
  }
  if (!(layerWeights.postFeedforwardNorm instanceof GPUBuffer)) releaseOrTrack(recorder, normWeightBuf);
  if (recorder) {
    recorder.trackTemporaryBuffer(activatedOutput.buffer);
  } else {
    releaseBuffer(activatedOutput.buffer);
  }

  return outputTensor;
}
