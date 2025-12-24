/**
 * Feed-forward network (FFN) operations for transformer layers.
 *
 * This module handles the dense FFN computation:
 * - Gate projection (W_gate @ x)
 * - Up projection (W_up @ x)
 * - Gated activation (SiLU/GELU with gate)
 * - Down projection (W_down @ activated)
 *
 * For MoE (Mixture of Experts) feed-forward, see moe-impl.ts.
 *
 * @module inference/pipeline/ffn
 */

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { runMatmul, runSiLU, runGeLU, runSiLURowSplit } from '../../gpu/kernel-selector.js';
import type { LayerWeights, MaybeGPUBuffer } from './types.js';

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for FFN computation.
 */
export interface FFNConfig {
  hiddenSize: number;
  intermediateSize: number;
  /** 'silu' | 'gelu' - activation function */
  hiddenActivation: string;
}

/**
 * FFN weights for a single layer.
 *
 * Supports two modes:
 * 1. Separate gate/up (3 matmul passes): gate, up, down
 * 2. Fused gate+up (2 matmul passes): gateUp, down
 */
export interface FFNWeights {
  /** Gate projection [intermediateSize, hiddenSize] - optional if gateUp provided */
  gate?: GPUBuffer | Float32Array;
  /** Up projection [intermediateSize, hiddenSize] - optional if gateUp provided */
  up?: GPUBuffer | Float32Array;
  /** Down projection [hiddenSize, intermediateSize] */
  down: GPUBuffer | Float32Array;
  /** Fused gate+up projection [intermediateSize*2, hiddenSize] - enables 2-pass FFN */
  gateUp?: GPUBuffer | Float32Array;
}

// ============================================================================
// GPU FFN Operations
// ============================================================================

/**
 * Run feed-forward network on GPU.
 *
 * Computes: output = down(activation(gate(x)) * up(x))
 * where activation is either SiLU or GELU depending on model.
 *
 * Supports two paths:
 * - Fused (2 matmuls): gateUp -> split+activate -> down
 * - Separate (3 matmuls): gate, up -> activate -> down
 *
 * @param inputBuffer - Input hidden states [numTokens, hiddenSize]
 * @param weights - FFN weight buffers (gate, up, down) or (gateUp, down)
 * @param config - FFN configuration
 * @param numTokens - Number of tokens in batch
 * @returns Output buffer [numTokens, hiddenSize]
 */
export async function runFFNGPU(
  inputBuffer: GPUBuffer,
  weights: FFNWeights,
  config: FFNConfig,
  numTokens: number
): Promise<GPUBuffer> {
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const downWeight = weights.down;

  if (!(downWeight instanceof GPUBuffer)) {
    throw new Error('FFN down weight must be GPUBuffer for GPU path');
  }

  // Check for fused gate+up path (2 matmuls instead of 3)
  if (weights.gateUp && weights.gateUp instanceof GPUBuffer) {
    // Fused path: single matmul for gate+up, then split+activate
    const gateUpWeight = weights.gateUp;

    // 1. Fused gate+up projection: [numTokens, hiddenSize] @ [intermediateSize*2, hiddenSize]^T -> [numTokens, intermediateSize*2]
    const gateUpOutput = await runMatmul(
      inputBuffer, gateUpWeight,
      numTokens, intermediateSize * 2, hiddenSize,
      { transposeB: 'auto' }
    );

    // 2. Split + Activation: output[i] = activation(gate[i]) * up[i]
    // Input is [numTokens, intermediateSize*2] where each row is [gate, up]
    const activation = hiddenActivation === 'gelu' ? 'gelu' : 'silu';
    const activatedOutput = await runSiLURowSplit(gateUpOutput, {
      numTokens,
      dim: intermediateSize,
      activation,
    });

    releaseBuffer(gateUpOutput);

    // 3. Down projection: [numTokens, intermediateSize] @ [hiddenSize, intermediateSize]^T -> [numTokens, hiddenSize]
    const output = await runMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto' });

    releaseBuffer(activatedOutput);

    return output;
  }

  // Separate path: 3 matmuls (original)
  const gateWeight = weights.gate;
  const upWeight = weights.up;

  if (!(gateWeight instanceof GPUBuffer) || !(upWeight instanceof GPUBuffer)) {
    throw new Error('FFN gate/up weights must be GPUBuffers for GPU path (or provide gateUp for fused path)');
  }

  // 1. Gate projection: gate = W_gate @ x (transposeB for SafeTensors layout)
  const gateOutput = await runMatmul(inputBuffer, gateWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 2. Up projection: up = W_up @ x (transposeB for SafeTensors layout)
  const upOutput = await runMatmul(inputBuffer, upWeight, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 3. Activation: activation(gate) * up
  // Use GELU for Gemma 3, SiLU for LLaMA/Mistral/Qwen
  const activationFn = hiddenActivation === 'gelu' ? runGeLU : runSiLU;
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
  });

  releaseBuffer(gateOutput);
  releaseBuffer(upOutput);

  // 4. Down projection: result = W_down @ activated (transposeB for SafeTensors layout)
  const output = await runMatmul(activatedOutput, downWeight, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto' });

  releaseBuffer(activatedOutput);

  return output;
}

/**
 * Run feed-forward network with CPU readback.
 *
 * Same as runFFNGPU but reads the result back to CPU.
 * Used when subsequent operations need CPU data.
 *
 * @param hiddenStates - Input hidden states (CPU Float32Array)
 * @param weights - FFN weights (will be uploaded to GPU)
 * @param config - FFN configuration
 * @returns Output as Float32Array [numTokens, hiddenSize]
 */
export async function runFFN(
  hiddenStates: Float32Array,
  weights: FFNWeights,
  config: FFNConfig
): Promise<Float32Array> {
  const device = getDevice();
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const numTokens = hiddenStates.length / hiddenSize;

  if (!device) {
    // CPU fallback - return zeros (proper CPU FFN would be expensive)
    console.warn('[FFN] No GPU device, returning zeros');
    return new Float32Array(hiddenStates.length);
  }

  // 1. Create input buffer
  const inputBuffer = acquireBuffer(hiddenStates.byteLength, undefined, 'ffn_input');
  device.queue.writeBuffer(inputBuffer, 0, hiddenStates as unknown as BufferSource);

  // 2. Upload weights if needed
  let gateWeightBuffer: GPUBuffer;
  let upWeightBuffer: GPUBuffer;
  let downWeightBuffer: GPUBuffer;
  let gateOwned = false, upOwned = false, downOwned = false;

  if (weights.gate instanceof GPUBuffer) {
    gateWeightBuffer = weights.gate;
  } else {
    gateWeightBuffer = acquireBuffer((weights.gate as Float32Array).byteLength, undefined, 'ffn_gate_w');
    device.queue.writeBuffer(gateWeightBuffer, 0, weights.gate as unknown as BufferSource);
    gateOwned = true;
  }

  if (weights.up instanceof GPUBuffer) {
    upWeightBuffer = weights.up;
  } else {
    upWeightBuffer = acquireBuffer((weights.up as Float32Array).byteLength, undefined, 'ffn_up_w');
    device.queue.writeBuffer(upWeightBuffer, 0, weights.up as unknown as BufferSource);
    upOwned = true;
  }

  if (weights.down instanceof GPUBuffer) {
    downWeightBuffer = weights.down;
  } else {
    downWeightBuffer = acquireBuffer((weights.down as Float32Array).byteLength, undefined, 'ffn_down_w');
    device.queue.writeBuffer(downWeightBuffer, 0, weights.down as unknown as BufferSource);
    downOwned = true;
  }

  // 3. Gate projection
  const gateOutput = await runMatmul(inputBuffer, gateWeightBuffer, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 4. Up projection
  const upOutput = await runMatmul(inputBuffer, upWeightBuffer, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 5. Activation
  const activationFn = hiddenActivation === 'gelu' ? runGeLU : runSiLU;
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
  });

  // 6. Down projection
  const output = await runMatmul(activatedOutput, downWeightBuffer, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto' });

  // 7. Read output back
  const outputData = await readBuffer(output, hiddenStates.byteLength);

  // Cleanup
  releaseBuffer(inputBuffer);
  if (gateOwned) releaseBuffer(gateWeightBuffer);
  if (upOwned) releaseBuffer(upWeightBuffer);
  if (downOwned) releaseBuffer(downWeightBuffer);
  releaseBuffer(gateOutput);
  releaseBuffer(upOutput);
  releaseBuffer(activatedOutput);
  releaseBuffer(output);

  return new Float32Array(outputData);
}

// ============================================================================
// CPU Fallback Operations
// ============================================================================

/**
 * Layer normalization (CPU fallback).
 *
 * Computes: output[i] = (x[i] - mean) / std
 *
 * @param x - Input tensor
 * @param eps - Epsilon for numerical stability
 * @returns Normalized tensor
 */
export function layerNormCPU(x: Float32Array, eps: number = 1e-5): Float32Array {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const diff = x[i] - mean;
    variance += diff * diff;
  }
  variance /= n;

  const std = Math.sqrt(variance + eps);
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    result[i] = (x[i] - mean) / std;
  }
  return result;
}

/**
 * Gather tokens by indices (CPU).
 *
 * Extracts specific tokens from hidden states by their indices.
 *
 * @param hiddenStates - Full hidden states [numTokens, hiddenSize]
 * @param indices - Token indices to gather
 * @param hiddenSize - Hidden dimension size
 * @returns Gathered tokens [indices.length, hiddenSize]
 */
export function gatherTokensCPU(
  hiddenStates: Float32Array,
  indices: number[],
  hiddenSize: number
): Float32Array {
  const gathered = new Float32Array(indices.length * hiddenSize);
  for (let i = 0; i < indices.length; i++) {
    const srcOffset = indices[i] * hiddenSize;
    gathered.set(
      hiddenStates.subarray(srcOffset, srcOffset + hiddenSize),
      i * hiddenSize
    );
  }
  return gathered;
}
