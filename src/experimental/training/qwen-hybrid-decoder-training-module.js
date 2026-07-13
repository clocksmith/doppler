import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  releaseQwenFullDecoderLayerCache,
  runQwenFullDecoderLayerBackward,
  runQwenFullDecoderLayerForward,
} from './qwen-full-decoder-training-module.js';
import {
  releaseQwenLinearDecoderLayerCache,
  runQwenLinearDecoderLayerBackward,
  runQwenLinearDecoderLayerForward,
} from './qwen-linear-decoder-training-module.js';

const LINEAR_ATTENTION = 'linear_attention';
const FULL_ATTENTION = 'full_attention';

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function requireLayerType(value, index) {
  if (value !== LINEAR_ATTENTION && value !== FULL_ATTENTION) {
    throw new Error(`Qwen hybrid layer ${index} has unsupported type "${String(value)}".`);
  }
  return value;
}

function releaseLayerCache(entry) {
  if (entry.type === LINEAR_ATTENTION) {
    releaseQwenLinearDecoderLayerCache(entry.cache);
  } else {
    releaseQwenFullDecoderLayerCache(entry.cache);
  }
}

function releaseLoraGradients(lora) {
  if (!lora) return;
  for (const gradients of Object.values(lora)) {
    releaseTensor(gradients?.A);
    releaseTensor(gradients?.B);
  }
}

function releaseLayerGradients(gradients) {
  if (!gradients) return;
  releaseLoraGradients(gradients.lora);
  releaseTensor(gradients.initialState);
}

export async function runQwenHybridDecoderForward(inputs) {
  if (!Array.isArray(inputs?.layers) || inputs.layers.length < 1) {
    throw new Error('Qwen hybrid decoder requires at least one layer.');
  }
  const entries = [];
  const finalStates = [];
  let hidden = inputs.hidden;
  let completed = false;
  try {
    for (let index = 0; index < inputs.layers.length; index += 1) {
      const layer = inputs.layers[index];
      const type = requireLayerType(layer?.type, index);
      const layerInputs = { ...layer.inputs, hidden };
      const result = type === LINEAR_ATTENTION
        ? await runQwenLinearDecoderLayerForward(layerInputs, layer.options)
        : await runQwenFullDecoderLayerForward(layerInputs, layer.options);
      entries.push({
        type,
        inputs: layerInputs,
        options: layer.options,
        output: result.output,
        cache: result.cache,
      });
      hidden = result.output;
      if (type === LINEAR_ATTENTION) {
        finalStates.push({ layerIndex: index, state: result.finalState });
      }
    }
    completed = true;
    return {
      output: hidden,
      finalStates,
      cache: {
        layerTypes: entries.map((entry) => entry.type),
        entries,
      },
    };
  } finally {
    if (!completed) {
      for (const entry of entries) {
        releaseTensor(entry.output);
        releaseLayerCache(entry);
      }
      for (const item of finalStates) releaseTensor(item.state);
    }
  }
}

export async function runQwenHybridDecoderBackward(gradOutput, cache) {
  if (!Array.isArray(cache?.entries) || cache.entries.length < 1) {
    throw new Error('Qwen hybrid decoder backward requires a forward cache.');
  }
  let gradient = gradOutput;
  let ownsGradient = false;
  const layerGradients = new Array(cache.entries.length);
  let completed = false;
  try {
    for (let index = cache.entries.length - 1; index >= 0; index -= 1) {
      const entry = cache.entries[index];
      const result = entry.type === LINEAR_ATTENTION
        ? await runQwenLinearDecoderLayerBackward(
            entry.inputs,
            gradient,
            entry.cache,
            entry.options
          )
        : await runQwenFullDecoderLayerBackward(
            entry.inputs,
            gradient,
            entry.cache,
            entry.options
          );
      if (ownsGradient) releaseTensor(gradient);
      gradient = result.hidden;
      ownsGradient = true;
      layerGradients[index] = {
        type: entry.type,
        lora: result.lora,
        initialState: result.initialState || null,
      };
    }
    completed = true;
    return { hidden: gradient, layers: layerGradients };
  } finally {
    if (!completed) {
      if (ownsGradient) releaseTensor(gradient);
      for (const result of layerGradients) releaseLayerGradients(result);
    }
  }
}

export function releaseQwenHybridDecoderCache(cache) {
  if (!Array.isArray(cache?.entries)) return;
  for (let index = 0; index < cache.entries.length; index += 1) {
    const entry = cache.entries[index];
    if (index < cache.entries.length - 1) releaseTensor(entry.output);
    releaseLayerCache(entry);
  }
}
