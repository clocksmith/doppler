/**
 * Dense FFN Operations
 *
 * Handles standard dense (non-MoE) FFN computations including:
 * - Gate/Up -> Activation -> Down projections
 * - Fused FFN variants
 * - Fused Down+Norm optimization
 *
 * @module inference/pipeline/ffn/dense
 */

import {
  doMatmul, doSiLU, doGeLU, doSiLURowSplit, doMatmulRMSNormFused,
  releaseOrTrack
} from '../ops.js';
import { createTensor } from '../../../gpu/tensor.js';
import { isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer } from '../../../gpu/buffer-pool.js';
import { runFusedFFN, recordFusedFFN, isFusedQ4KDisabled } from '../../../gpu/kernel-selector.js';
import { log } from '../../../debug/index.js';
import { isKernelDebugEnabled, dumpTokenVector } from '../debug-utils.js';
import { applyLoRA } from '../lora-apply.js';
import { getLoRAModule } from '../lora.js';
import { getWeightBuffer, getNormWeightBuffer } from '../weights.js';

/**
 * Run dense (non-MoE) FFN on GPU.
 * @param {number} layerIdx
 * @param {import('../../../gpu/tensor.js').Tensor} inputTensor
 * @param {number} numTokens
 * @param {import('../types.js').LayerContext} context
 * @param {import('../types.js').LayerWeights | undefined} layerWeights
 * @returns {Promise<import('../../../gpu/tensor.js').Tensor>}
 */
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
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const lastTokenIdx = Math.max(0, numTokens - 1);
  const lora = context.lora || null;

  if (layerWeights?.gateUp && layerWeights?.down) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

    const useF16 = inputTensor.dtype === 'f16';
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', label: `L${layerIdx}.ffn_gate_up`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate_up' },
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

    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    const activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
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
      { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_down' },
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

  if (layerWeights?.gate && layerWeights?.up && layerWeights?.down && !layerWeights.gateUp && inputTensor.dtype === 'f32') {
    const loraGate = getLoRAModule(lora, layerIdx, 'gate_proj');
    const loraUp = getLoRAModule(lora, layerIdx, 'up_proj');
    if (!loraGate && !loraUp) {
      const gateDtype = isWeightBuffer(layerWeights.gate) ? layerWeights.gate.dtype : 'f32';
      const upDtype = isWeightBuffer(layerWeights.up) ? layerWeights.up.dtype : 'f32';
      const dtypeMatches = gateDtype === upDtype;
      const q4kFusedAllowed = gateDtype !== 'q4k' || !isFusedQ4KDisabled();
      const dtypeSupported = gateDtype === 'f16' || gateDtype === 'f32' || (gateDtype === 'q4k' && q4kFusedAllowed);
      const f16BatchOk = gateDtype !== 'f16' || numTokens === 1;

      if (dtypeMatches && dtypeSupported && f16BatchOk) {
        const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
        const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');
        const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');

        const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
        const fusedOutput = recorder
          ? await recordFusedFFN(
            recorder,
            inputTensor,
            gateWeight,
            upWeight,
            hiddenSize,
            intermediateSize,
            { batchSize: numTokens, activation }
          )
          : await runFusedFFN(
            inputTensor,
            gateWeight,
            upWeight,
            hiddenSize,
            intermediateSize,
            { batchSize: numTokens, activation }
          );

        if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
          releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
        }
        if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
          releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
        }

        let output = await doMatmul(
          fusedOutput,
          downWeight,
          numTokens,
          hiddenSize,
          intermediateSize,
          { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, role: 'ffn_down' },
          recorder
        );

        const loraDown = getLoRAModule(lora, layerIdx, 'down_proj');
        if (loraDown) {
          const combined = await applyLoRA(
            fusedOutput,
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
          recorder.trackTemporaryBuffer(fusedOutput.buffer);
        } else {
          releaseBuffer(fusedOutput.buffer);
        }

        return output;
      }
    }
  }

  if (!layerWeights?.gate || !layerWeights?.up || !layerWeights?.down) {
    log.warn('Layer', `L${layerIdx} FFN: no weights found`);
    const bytesPerElement = inputTensor.dtype === 'f16' ? 2 : 4;
    const byteSize = numTokens * hiddenSize * bytesPerElement;
    const outputBuffer = acquireBuffer(byteSize, undefined, 'ffn_output');
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(inputTensor.buffer, 0, outputBuffer, 0, byteSize);
    device.queue.submit([encoder.finish()]);
    return createTensor(outputBuffer, inputTensor.dtype, [...inputTensor.shape], 'ffn_output_copy');
  }

  const useF16 = inputTensor.dtype === 'f16';
  const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
  let gateOutput = await doMatmul(inputTensor, gateWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_gate`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate' }, recorder);
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
  let upOutput = await doMatmul(inputTensor, upWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_up`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_up' }, recorder);
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

  const activationFn = hiddenActivation === 'gelu' ? doGeLU : doSiLU;
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
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
  let output = await doMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto', label: `L${layerIdx}.ffn_down`, layerIdx, outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_down' }, recorder);

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

/**
 * Run dense FFN with fused down projection + post-FFN norm.
 * Used for sandwich norm architectures when conditions allow fusion.
 * @param {number} layerIdx
 * @param {import('../../../gpu/tensor.js').Tensor} inputTensor
 * @param {number} numTokens
 * @param {import('../types.js').LayerContext} context
 * @param {import('../types.js').LayerWeights} layerWeights
 * @param {import('../../../gpu/tensor.js').Tensor} residualTensor
 * @param {number} eps
 * @param {boolean} transposeB
 * @param {GPUBuffer | null} [outputBuffer]
 * @returns {Promise<import('../../../gpu/tensor.js').Tensor>}
 */
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
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const lora = context.lora || null;

  if (!layerWeights.down || !layerWeights.postFeedforwardNorm) {
    throw new Error('Missing down or norm weights');
  }

  const downWeight = getWeightBuffer(layerWeights.down, 'ffn_down');
  const normWeightBuf = getNormWeightBuffer(layerWeights.postFeedforwardNorm, 'post_feedforward_norm', weightConfig, debugFlags);

  /** @type {import('../../../gpu/tensor.js').Tensor} */
  let activatedOutput;
  const useF16 = inputTensor.dtype === 'f16';

  if (layerWeights.gateUp) {
    const gateUpWeight = getWeightBuffer(layerWeights.gateUp, 'ffn_gate_up');
    let gateUpOutput = await doMatmul(
      inputTensor, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate_up' },
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

    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    activatedOutput = await doSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateUpOutput.buffer);
    } else {
      releaseBuffer(gateUpOutput.buffer);
    }
  } else {
    const gateWeight = getWeightBuffer(layerWeights.gate, 'ffn_gate');
    const upWeight = getWeightBuffer(layerWeights.up, 'ffn_up');

    const gateOutput = await doMatmul(
      inputTensor, gateWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_gate' },
      recorder
    );
    if (!(layerWeights.gate instanceof GPUBuffer) && !isWeightBuffer(layerWeights.gate)) {
      releaseOrTrack(recorder, isWeightBuffer(gateWeight) ? gateWeight.buffer : gateWeight);
    }

    const upOutput = await doMatmul(
      inputTensor, upWeight,
      numTokens, intermediateSize, hiddenSize,
      { transposeB: 'auto', outputDtype: useF16 ? 'f16' : undefined, role: 'ffn_up' },
      recorder
    );
    if (!(layerWeights.up instanceof GPUBuffer) && !isWeightBuffer(layerWeights.up)) {
      releaseOrTrack(recorder, isWeightBuffer(upWeight) ? upWeight.buffer : upWeight);
    }

    const activationFn = hiddenActivation === 'gelu' ? doGeLU : doSiLU;
    activatedOutput = await activationFn(upOutput, {
      size: numTokens * intermediateSize,
      gate: gateOutput,
    }, recorder);

    if (recorder) {
      recorder.trackTemporaryBuffer(gateOutput.buffer);
      recorder.trackTemporaryBuffer(upOutput.buffer);
    } else {
      releaseBuffer(gateOutput.buffer);
      releaseBuffer(upOutput.buffer);
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
