import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  releaseQwenHybridDecoderCache,
  runQwenHybridDecoderBackward,
  runQwenHybridDecoderForward,
} from './qwen-hybrid-decoder-training-module.js';

function positiveInteger(value, label) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseTensorsOnce(tensors) {
  const buffers = new Set();
  for (const tensor of tensors) {
    if (tensor?.buffer && !buffers.has(tensor.buffer)) {
      buffers.add(tensor.buffer);
      releaseBuffer(tensor.buffer);
    }
  }
}

function releaseLayerGradients(layer) {
  if (!layer) return;
  if (layer.lora) {
    for (const gradients of Object.values(layer.lora)) {
      releaseTensor(gradients?.A);
      releaseTensor(gradients?.B);
    }
  }
  releaseTensor(layer.initialState);
}

function releaseRecomputedForward(result) {
  if (!result) return;
  for (const item of result.finalStates ?? []) releaseTensor(item.state);
  releaseTensor(result.output);
  releaseQwenHybridDecoderCache(result.cache);
}

export async function runQwenCheckpointedHybridDecoderForward(inputs, options = {}) {
  if (!Array.isArray(inputs?.layers) || inputs.layers.length < 1) {
    throw new Error('Qwen checkpointed hybrid decoder requires at least one layer.');
  }
  const interval = positiveInteger(options.checkpointInterval, 'checkpointInterval');
  const checkpoints = [];
  const segmentTypes = [];
  let hidden = inputs.hidden;
  let ownsHidden = false;
  let completed = false;
  try {
    for (let startLayer = 0; startLayer < inputs.layers.length; startLayer += interval) {
      const endLayer = Math.min(inputs.layers.length, startLayer + interval);
      checkpoints.push({
        startLayer,
        endLayer,
        hidden,
        ownsHidden,
      });
      const segmentLayers = inputs.layers.slice(startLayer, endLayer);
      segmentTypes.push(segmentLayers.map((layer) => layer.type));
      const result = await runQwenHybridDecoderForward({
        hidden,
        layers: segmentLayers,
      });
      for (const item of result.finalStates) releaseTensor(item.state);
      releaseQwenHybridDecoderCache(result.cache);
      hidden = result.output;
      ownsHidden = true;
    }
    completed = true;
    return {
      output: hidden,
      finalStates: [],
      cache: {
        checkpointInterval: interval,
        layerCount: inputs.layers.length,
        layerTypes: inputs.layers.map((layer) => layer.type),
        segmentTypes,
        layers: inputs.layers,
        checkpoints,
      },
    };
  } finally {
    if (!completed) {
      releaseTensorsOnce([
        ...(checkpoints.filter((entry) => entry.ownsHidden).map((entry) => entry.hidden)),
        ...(ownsHidden ? [hidden] : []),
      ]);
    }
  }
}

export async function runQwenCheckpointedHybridDecoderBackward(
  gradOutput,
  cache
) {
  if (!Array.isArray(cache?.checkpoints) || cache.checkpoints.length < 1
    || !Array.isArray(cache.layers) || cache.layers.length !== cache.layerCount) {
    throw new Error('Qwen checkpointed hybrid backward requires a valid forward cache.');
  }
  const layers = new Array(cache.layerCount);
  let gradient = gradOutput;
  let ownsGradient = false;
  let completed = false;
  try {
    for (let segmentIndex = cache.checkpoints.length - 1; segmentIndex >= 0; segmentIndex -= 1) {
      const checkpoint = cache.checkpoints[segmentIndex];
      const segmentLayers = cache.layers.slice(checkpoint.startLayer, checkpoint.endLayer);
      const forward = await runQwenHybridDecoderForward({
        hidden: checkpoint.hidden,
        layers: segmentLayers,
      });
      let backward = null;
      try {
        backward = await runQwenHybridDecoderBackward(gradient, forward.cache);
        if (ownsGradient) releaseTensor(gradient);
        gradient = backward.hidden;
        ownsGradient = true;
        for (let localIndex = 0; localIndex < backward.layers.length; localIndex += 1) {
          layers[checkpoint.startLayer + localIndex] = backward.layers[localIndex];
        }
      } finally {
        releaseRecomputedForward(forward);
      }
    }
    completed = true;
    return { hidden: gradient, layers };
  } finally {
    if (!completed) {
      if (ownsGradient) releaseTensor(gradient);
      for (const layer of layers) releaseLayerGradients(layer);
    }
  }
}

export function releaseQwenCheckpointedHybridDecoderCache(cache) {
  if (!Array.isArray(cache?.checkpoints)) return;
  releaseTensorsOnce(
    cache.checkpoints
      .filter((entry) => entry.ownsHidden)
      .map((entry) => entry.hidden)
  );
}
