import { runResidualAdd, runSiLU } from '../../gpu/kernels/index.js';
import { runSiluGatedBackward } from '../../gpu/kernels/backward/index.js';
import { releaseBuffer } from '../../memory/buffer-pool.js';
import {
  runFrozenLoraProjectionBackward,
  runFrozenLoraProjectionForward,
} from './qwen-full-attention-training-module.js';

function positiveInteger(value, label) {
  const parsed = Math.floor(Number(value));
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${label} must be a positive integer.`);
  }
  return parsed;
}

function resolveDimensions(options) {
  return {
    numTokens: positiveInteger(options?.numTokens, 'numTokens'),
    hiddenSize: positiveInteger(options?.hiddenSize, 'hiddenSize'),
    intermediateSize: positiveInteger(options?.intermediateSize, 'intermediateSize'),
  };
}

function releaseTensor(tensor) {
  if (tensor?.buffer) releaseBuffer(tensor.buffer);
}

function releaseTensorMap(tensors) {
  if (!tensors) return;
  for (const tensor of Object.values(tensors)) releaseTensor(tensor);
}

function releaseProjectionGradients(gradients) {
  if (!gradients) return;
  releaseTensor(gradients.A);
  releaseTensor(gradients.B);
}

export async function runQwenDecoderMlpForward(inputs, options = {}) {
  const dims = resolveDimensions(options);
  let gate = null;
  let up = null;
  let activated = null;
  let down = null;
  let completed = false;
  try {
    gate = await runFrozenLoraProjectionForward(
      inputs.hidden,
      inputs.gateWeight,
      dims.numTokens,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.lora?.gate,
      'gate_proj'
    );
    up = await runFrozenLoraProjectionForward(
      inputs.hidden,
      inputs.upWeight,
      dims.numTokens,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.lora?.up,
      'up_proj'
    );
    activated = await runSiLU(up.output, {
      size: dims.numTokens * dims.intermediateSize,
      gate: gate.output,
      inputActivation: 'identity',
      swigluLimit: null,
    });
    down = await runFrozenLoraProjectionForward(
      activated,
      inputs.downWeight,
      dims.numTokens,
      dims.intermediateSize,
      dims.hiddenSize,
      inputs.lora?.down,
      'down_proj'
    );
    completed = true;
    return {
      output: down.output,
      cache: {
        dims,
        gate: gate.output,
        up: up.output,
        activated,
        projectionDowns: {
          gate: gate.down,
          up: up.down,
          down: down.down,
        },
      },
    };
  } finally {
    if (!completed) {
      releaseTensor(gate?.output);
      releaseTensor(up?.output);
      releaseTensor(activated);
      releaseTensor(down?.output);
      releaseTensor(gate?.down);
      releaseTensor(up?.down);
      releaseTensor(down?.down);
    }
  }
}

export async function runQwenDecoderMlpBackward(inputs, gradOutput, cache, options = {}) {
  const dims = resolveDimensions(options);
  for (const key of Object.keys(dims)) {
    if (cache?.dims?.[key] !== dims[key]) {
      throw new Error(`Qwen decoder MLP cache mismatch for ${key}.`);
    }
  }
  let downGradients = null;
  let activationGradients = null;
  let gateGradients = null;
  let upGradients = null;
  let hidden = null;
  let completed = false;
  try {
    downGradients = await runFrozenLoraProjectionBackward(
      cache.activated,
      inputs.downWeight,
      gradOutput,
      dims.numTokens,
      dims.intermediateSize,
      dims.hiddenSize,
      inputs.lora?.down,
      cache.projectionDowns?.down,
      'down_proj'
    );
    activationGradients = await runSiluGatedBackward(
      cache.gate,
      cache.up,
      downGradients.input,
      { count: dims.numTokens * dims.intermediateSize, swigluLimit: null }
    );
    gateGradients = await runFrozenLoraProjectionBackward(
      inputs.hidden,
      inputs.gateWeight,
      activationGradients.gate,
      dims.numTokens,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.lora?.gate,
      cache.projectionDowns?.gate,
      'gate_proj'
    );
    upGradients = await runFrozenLoraProjectionBackward(
      inputs.hidden,
      inputs.upWeight,
      activationGradients.up,
      dims.numTokens,
      dims.hiddenSize,
      dims.intermediateSize,
      inputs.lora?.up,
      cache.projectionDowns?.up,
      'up_proj'
    );
    hidden = await runResidualAdd(
      gateGradients.input,
      upGradients.input,
      dims.numTokens * dims.hiddenSize
    );
    completed = true;
    return {
      hidden,
      lora: {
        gate: { A: gateGradients.A, B: gateGradients.B },
        up: { A: upGradients.A, B: upGradients.B },
        down: { A: downGradients.A, B: downGradients.B },
      },
    };
  } finally {
    releaseTensor(downGradients?.input);
    releaseTensorMap(activationGradients);
    releaseTensor(gateGradients?.input);
    releaseTensor(upGradients?.input);
    if (!completed) {
      releaseTensor(hidden);
      releaseProjectionGradients(downGradients);
      releaseProjectionGradients(gateGradients);
      releaseProjectionGradients(upGradients);
    }
  }
}

export function releaseQwenDecoderMlpCache(cache) {
  releaseTensor(cache?.gate);
  releaseTensor(cache?.up);
  releaseTensor(cache?.activated);
  releaseTensorMap(cache?.projectionDowns);
}
