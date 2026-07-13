import {
  runCrossEntropyLoss,
  runGather,
  runMatmul,
  runRMSNorm,
  runSoftmax,
} from '../../gpu/kernels/index.js';
import {
  runCrossEntropyBackward,
  runRmsNormBackward,
} from '../../gpu/kernels/backward/index.js';
import { runMatmulBackwardDx } from '../../gpu/kernels/backward/utils.js';
import { createTensor } from '../../gpu/tensor.js';
import {
  acquireBuffer,
  readBuffer,
  releaseBuffer,
  uploadData,
} from '../../memory/buffer-pool.js';
import {
  releaseQwenHybridDecoderCache,
  runQwenHybridDecoderBackward,
  runQwenHybridDecoderForward,
} from './qwen-hybrid-decoder-training-module.js';
import {
  releaseQwenCheckpointedHybridDecoderCache,
  runQwenCheckpointedHybridDecoderBackward,
  runQwenCheckpointedHybridDecoderForward,
} from './qwen-checkpointed-hybrid-decoder-training-module.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseLoraGradients(lora) {
  if (!lora) return;
  for (const gradients of Object.values(lora)) {
    releaseTensor(gradients?.A);
    releaseTensor(gradients?.B);
  }
}

function makeLossScaleTensor(numTokens, activeTokenCount) {
  const values = new Float32Array(numTokens).fill(1 / activeTokenCount);
  const buffer = acquireBuffer(values.byteLength, undefined, 'qwen_sft_loss_scale');
  uploadData(buffer, values);
  return createTensor(buffer, 'f32', [numTokens], 'qwen_sft_loss_scale');
}

function addAdapterPair(entries, prefix, adapter, gradients) {
  if (!adapter?.A || !adapter?.B || !gradients?.A || !gradients?.B) {
    throw new Error(`Qwen SFT microstep missing adapter or gradient for ${prefix}.`);
  }
  entries.push(
    { name: `${prefix}.lora_A`, parameter: adapter.A, gradient: gradients.A },
    { name: `${prefix}.lora_B`, parameter: adapter.B, gradient: gradients.B }
  );
}

function collectAdapterEntries(layers, backwardLayers) {
  if (layers.length !== backwardLayers.length) {
    throw new Error('Qwen SFT microstep layer-gradient count mismatch.');
  }
  const entries = [];
  for (let index = 0; index < layers.length; index += 1) {
    const layer = layers[index];
    const gradients = backwardLayers[index];
    if (layer.type === 'full_attention') {
      for (const projection of ['q', 'k', 'v', 'o']) {
        addAdapterPair(
          entries,
          `layers.${index}.self_attn.${projection}_proj`,
          layer.inputs.attention.lora?.[projection],
          gradients.lora?.[projection]
        );
      }
    }
    for (const projection of ['gate', 'up', 'down']) {
      addAdapterPair(
        entries,
        `layers.${index}.mlp.${projection}_proj`,
        layer.inputs.mlp.lora?.[projection],
        gradients.lora?.[projection]
      );
    }
  }
  return entries;
}

async function captureGradientSnapshots(entries) {
  const snapshots = {};
  for (const entry of entries) {
    const count = entry.gradient.shape.reduce((product, value) => product * value, 1);
    snapshots[entry.name] = new Float32Array(await readBuffer(entry.gradient.buffer, count * 4));
  }
  return snapshots;
}

export async function runQwenHybridSftMicrostep(inputs, options = {}) {
  const numTokens = positiveInteger(options.numTokens, 'numTokens');
  const hiddenSize = positiveInteger(options.hiddenSize, 'hiddenSize');
  const vocabSize = positiveInteger(options.vocabSize, 'vocabSize');
  const activeTokenCount = positiveInteger(options.activeTokenCount, 'activeTokenCount');
  if (activeTokenCount > numTokens) {
    throw new Error('activeTokenCount cannot exceed numTokens.');
  }
  const applyOptimizer = options.applyOptimizer !== false;
  const gradientAccumulator = options.gradientAccumulator ?? null;
  const checkpointInterval = options.layerCheckpointInterval == null
    ? null
    : positiveInteger(options.layerCheckpointInterval, 'layerCheckpointInterval');
  if (applyOptimizer && (!options.optimizer || typeof options.optimizer.step !== 'function')) {
    throw new Error('Qwen SFT microstep requires an optimizer.');
  }
  if (applyOptimizer && gradientAccumulator) {
    throw new Error('Qwen SFT microstep cannot apply an optimizer and accumulate gradients together.');
  }
  if (!applyOptimizer && (!gradientAccumulator
    || typeof gradientAccumulator.accumulate !== 'function')) {
    throw new Error('Qwen SFT microstep without an optimizer requires a gradient accumulator.');
  }
  if (!options.trainingConfig?.training?.optimizer) {
    throw new Error('Qwen SFT microstep requires trainingConfig.training.optimizer.');
  }

  let embedded = null;
  let hybrid = null;
  let finalNorm = null;
  let logits = null;
  let softmax = null;
  let losses = null;
  let lossScale = null;
  let gradLogits = null;
  let gradFinalNorm = null;
  let gradHybridOutput = null;
  let backward = null;
  let adapterEntries = null;
  try {
    embedded = await runGather(
      inputs.tokenIds,
      inputs.embeddingWeight,
      numTokens,
      hiddenSize,
      vocabSize,
      {
        embeddingDtype: inputs.embeddingWeight.dtype,
        outputDtype: 'f32',
        transpose: false,
      }
    );
    hybrid = checkpointInterval == null
      ? await runQwenHybridDecoderForward({ hidden: embedded, layers: inputs.layers })
      : await runQwenCheckpointedHybridDecoderForward(
          { hidden: embedded, layers: inputs.layers },
          { checkpointInterval }
        );
    finalNorm = await runRMSNorm(hybrid.output, inputs.finalNormWeight, options.rmsEps, {
      batchSize: numTokens,
      hiddenSize,
      rmsNormWeightOffset: true,
    });
    logits = await runMatmul(
      finalNorm,
      inputs.lmHeadWeight,
      numTokens,
      vocabSize,
      hiddenSize,
      { transposeB: true, outputDtype: 'f32' }
    );
    softmax = await runSoftmax(logits, -1, { batchSize: numTokens, size: vocabSize });
    losses = await runCrossEntropyLoss(softmax, inputs.targets, { numTokens, vocabSize });
    const lossValues = new Float32Array(await readBuffer(losses.buffer, numTokens * 4));
    const meanLoss = lossValues.reduce((sum, value) => sum + value, 0) / activeTokenCount;
    lossScale = makeLossScaleTensor(numTokens, activeTokenCount);
    gradLogits = await runCrossEntropyBackward(
      softmax,
      inputs.targets,
      lossScale,
      { numTokens, vocabSize }
    );
    gradFinalNorm = await runMatmulBackwardDx(
      gradLogits,
      inputs.lmHeadWeight,
      numTokens,
      hiddenSize,
      vocabSize,
      { transposeB: true }
    );
    gradHybridOutput = await runRmsNormBackward(
      hybrid.output,
      inputs.finalNormWeight,
      gradFinalNorm,
      {
        numTokens,
        hiddenSize,
        eps: options.rmsEps,
        rmsNormWeightOffset: true,
      }
    );
    backward = checkpointInterval == null
      ? await runQwenHybridDecoderBackward(gradHybridOutput, hybrid.cache)
      : await runQwenCheckpointedHybridDecoderBackward(gradHybridOutput, hybrid.cache);
    adapterEntries = collectAdapterEntries(inputs.layers, backward.layers);
    const gradientSnapshots = options.captureGradients === true
      ? await captureGradientSnapshots(adapterEntries)
      : null;
    let accumulationMetrics = null;
    let optimizerMetrics = null;
    if (gradientAccumulator) {
      accumulationMetrics = await gradientAccumulator.accumulate(adapterEntries);
    } else {
      const parameters = adapterEntries.map((entry) => entry.parameter);
      const gradients = new Map(
        adapterEntries.map((entry) => [entry.parameter, entry.gradient])
      );
      optimizerMetrics = await options.optimizer.step(
        parameters,
        gradients,
        options.trainingConfig
      );
    }
    return {
      meanLoss,
      activeTokenCount,
      parameterNames: adapterEntries.map((entry) => entry.name),
      gradientSnapshots,
      optimizerMetrics,
      accumulationMetrics,
    };
  } finally {
    releaseTensor(backward?.hidden);
    if (backward?.layers) {
      for (const layer of backward.layers) {
        releaseTensor(layer.initialState);
        releaseLoraGradients(layer.lora);
      }
    }
    releaseTensor(gradHybridOutput);
    releaseTensor(gradFinalNorm);
    releaseTensor(gradLogits);
    releaseTensor(lossScale);
    releaseTensor(losses);
    releaseTensor(softmax);
    releaseTensor(logits);
    releaseTensor(finalNorm);
    if (hybrid) {
      for (const item of hybrid.finalStates) releaseTensor(item.state);
      releaseTensor(hybrid.output);
      if (checkpointInterval == null) {
        releaseQwenHybridDecoderCache(hybrid.cache);
      } else {
        releaseQwenCheckpointedHybridDecoderCache(hybrid.cache);
      }
    }
    releaseTensor(embedded);
  }
}
