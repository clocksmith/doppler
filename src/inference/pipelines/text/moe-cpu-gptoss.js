import { getDevice } from '../../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../../memory/buffer-pool.js';
import { createTensor } from '../../../gpu/tensor.js';
import {
  runMatmul,
  dequantizeMXFP4Expert,
  runSwiGLURowsplitBias,
  runBiasAdd,
} from '../../../gpu/kernel-selector.js';

export async function runGptOssExpertCPU(layerIdx, expertIdx, input, config, expertWeights) {
  const device = getDevice();
  if (!device) {
    return new Float32Array(input.length);
  }

  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  const weights = expertWeights.get(key);
  if (!weights || weights.expertFormat !== 'gpt-oss') {
    return new Float32Array(input.length);
  }

  const { hiddenSize, intermediateSize, numExperts, swigluLimit } = config;
  const kernelPath = config.kernelPath ?? null;
  const numTokens = input.length / hiddenSize;
  const outDim = intermediateSize * 2;
  const gateUpGroups = hiddenSize / 32;
  const downGroups = intermediateSize / 32;

  const inputBuffer = acquireBuffer(input.byteLength, undefined, 'moe_cpu_gptoss_input');
  device.queue.writeBuffer(inputBuffer, 0, input);
  const inputTensor = createTensor(inputBuffer, 'f32', [numTokens, hiddenSize], 'moe_cpu_gptoss_input');

  const gateUpWeight = await dequantizeMXFP4Expert(
    weights.gateUpBlocks,
    weights.gateUpScales,
    expertIdx,
    numExperts,
    outDim,
    gateUpGroups,
    { outputDtype: 'f32', modelType: 'gpt-oss', groupSize: 32, dequantTileShape: 'scalar' }
  );

  const downWeight = await dequantizeMXFP4Expert(
    weights.downBlocks,
    weights.downScales,
    expertIdx,
    numExperts,
    hiddenSize,
    downGroups,
    { outputDtype: 'f32', modelType: 'gpt-oss', groupSize: 32, dequantTileShape: 'scalar' }
  );

  const gateUpOut = await runMatmul(
    inputTensor,
    gateUpWeight.buffer,
    numTokens,
    outDim,
    hiddenSize,
    {
      transposeB: 'auto',
      bDtype: 'f32',
      outputDtype: 'f32',
      role: 'moe_gate_up',
      kernelPath,
    }
  );

  const biasOffset = expertIdx * outDim * 4;
  const biasTensor = createTensor(weights.gateUpBias, 'f32', [numExperts * outDim], 'moe_cpu_gptoss_bias');
  const activated = await runSwiGLURowsplitBias(gateUpOut, biasTensor, numTokens, intermediateSize, {
    biasOffset,
    swigluLimit,
  });

  const downOut = await runMatmul(
    activated,
    downWeight.buffer,
    numTokens,
    hiddenSize,
    intermediateSize,
    {
      transposeB: 'auto',
      bDtype: 'f32',
      outputDtype: 'f32',
      role: 'moe_down',
      kernelPath,
    }
  );

  if (weights.downBias) {
    const downBiasOffset = expertIdx * hiddenSize * 4;
    const downBiasTensor = createTensor(weights.downBias, 'f32', [numExperts * hiddenSize], 'moe_cpu_gptoss_down_bias');
    await runBiasAdd(downOut, downBiasTensor, numTokens, hiddenSize, {
      dataOffset: 0,
      biasOffset: downBiasOffset,
    });
  }

  const outputData = await readBuffer(downOut.buffer, input.byteLength);

  releaseBuffer(inputBuffer);
  releaseBuffer(gateUpWeight.buffer);
  releaseBuffer(downWeight.buffer);
  releaseBuffer(gateUpOut.buffer);
  releaseBuffer(activated.buffer);
  releaseBuffer(downOut.buffer);

  return new Float32Array(outputData);
}
