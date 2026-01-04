/**
 * Mixture of Experts (MoE) feed-forward implementation.
 *
 * This module handles:
 * - Token routing to experts via softmax + top-k
 * - Expert weight loading (on-demand)
 * - Parallel expert execution on GPU
 * - Scatter-add combination of expert outputs
 *
 * Supports multiple MoE architectures:
 * - Mixtral-style (gate/up/down per expert)
 * - GPT-OSS style (MXFP4 quantized fused gate_up + bias)
 *
 * @module inference/pipeline/moe-impl
 */

import { getDevice } from '../../gpu/device.js';
import { acquireBuffer, releaseBuffer, readBuffer } from '../../gpu/buffer-pool.js';
import { createTensor, type Tensor } from '../../gpu/tensor.js';
import {
  runMatmul,
  runSiLU,
  runGeLU,
  dequantizeMXFP4Expert,
  runBiasAdd,
  runSoftmaxTopK,
  runMoEGather,
  runScatterAddDynamic,
  runSwiGLURowsplitBias,
} from '../../gpu/kernel-selector.js';
import { MoERouter, createExpertExecutionPlan, combineExpertOutputs } from '../moe-router.js';
import { log } from '../../debug/index.js';
import type { ExpertWeights } from './types.js';
import { getRuntimeConfig } from '../../config/runtime.js';

// ============================================================================
// MXFP4 Dequantization Cache (avoids re-dequantizing same expert weights)
// ============================================================================

interface CachedExpertWeight {
  gateUp: GPUBuffer;
  down: GPUBuffer;
  lastUsed: number;
}

// Cache key: "layer_expert_type" -> dequantized weight
const dequantCache = new Map<string, CachedExpertWeight>();
// Use config value for max cache entries (default: 128 ~= 4 layers x 32 experts)
let dequantCacheMaxEntriesOverride: number | null = null;
let dequantCacheHits = 0;
let dequantCacheMisses = 0;

function getDequantCacheMaxEntries(): number {
  return dequantCacheMaxEntriesOverride ?? getRuntimeConfig().moe.cache.dequantCacheMaxEntries;
}

function getDequantCacheKey(layerIdx: number, expertIdx: number): string {
  return `${layerIdx}_${expertIdx}`;
}

function getCachedDequant(layerIdx: number, expertIdx: number): CachedExpertWeight | undefined {
  const key = getDequantCacheKey(layerIdx, expertIdx);
  const cached = dequantCache.get(key);
  if (cached) {
    cached.lastUsed = performance.now();
    dequantCacheHits++;
  }
  return cached;
}

function setCachedDequant(layerIdx: number, expertIdx: number, gateUp: GPUBuffer, down: GPUBuffer): void {
  const key = getDequantCacheKey(layerIdx, expertIdx);
  dequantCacheMisses++;

  // Evict oldest entries if cache is full
  if (dequantCache.size >= getDequantCacheMaxEntries()) {
    let oldestKey = '';
    let oldestTime = Infinity;
    for (const [k, v] of dequantCache.entries()) {
      if (v.lastUsed < oldestTime) {
        oldestTime = v.lastUsed;
        oldestKey = k;
      }
    }
    if (oldestKey) {
      const evicted = dequantCache.get(oldestKey);
      if (evicted) {
        evicted.gateUp.destroy();
        evicted.down.destroy();
      }
      dequantCache.delete(oldestKey);
    }
  }

  dequantCache.set(key, { gateUp, down, lastUsed: performance.now() });
}

/** Clear the dequantization cache (call on model unload). */
export function clearDequantCache(): void {
  for (const cached of dequantCache.values()) {
    cached.gateUp.destroy();
    cached.down.destroy();
  }
  dequantCache.clear();
  dequantCacheHits = 0;
  dequantCacheMisses = 0;
}

/** Get cache stats for debugging. */
export function getDequantCacheStats(): { hits: number; misses: number; size: number; maxEntries: number } {
  return {
    hits: dequantCacheHits,
    misses: dequantCacheMisses,
    size: dequantCache.size,
    maxEntries: getDequantCacheMaxEntries(),
  };
}

/** Configure dequant cache max entries at runtime. */
export function setDequantCacheMaxEntries(maxEntries: number): void {
  dequantCacheMaxEntriesOverride = maxEntries;
}

// ============================================================================
// Types
// ============================================================================

/**
 * Configuration for MoE feed-forward.
 */
export interface MoEConfig {
  hiddenSize: number;
  intermediateSize: number;
  numExperts: number;
  moeTopK: number;
  hiddenActivation: string;
}

/**
 * Expert weights with optional GPT-OSS quantized format.
 */
export interface MoEExpertWeights extends ExpertWeights {
  /** Flag for GPT-OSS quantized experts */
  isGptOss?: boolean;
  /** Number of experts in packed tensor (GPT-OSS) */
  numExperts?: number;
  /** Fused gate+up blocks (MXFP4, GPT-OSS) */
  gateUpBlocks?: GPUBuffer;
  /** Gate+up scales (MXFP4, GPT-OSS) */
  gateUpScales?: GPUBuffer;
  /** Gate+up bias (GPT-OSS) */
  gateUpBias?: GPUBuffer;
  /** Down blocks (MXFP4, GPT-OSS) */
  downBlocks?: GPUBuffer;
  /** Down scales (MXFP4, GPT-OSS) */
  downScales?: GPUBuffer;
  /** Down bias (GPT-OSS, optional) */
  downBias?: GPUBuffer;
}

/**
 * Layer router weights (for models with per-layer routers like GPT-OSS).
 */
export interface LayerRouterWeights {
  weight: Float32Array | GPUBuffer;
  bias: Float32Array | GPUBuffer | null;
}

/**
 * Expert weight loader interface.
 */
export interface ExpertLoader {
  loadExpert(layerIdx: number, expertIdx: number): Promise<MoEExpertWeights | null>;
}

// ============================================================================
// MoE Feed-Forward (CPU Routing Path)
// ============================================================================

/**
 * MoE feed-forward with CPU routing.
 *
 * This is the simpler path that routes on CPU, then executes experts on GPU.
 * Used when full GPU routing is not needed or for debugging.
 *
 * @param hiddenStates - Input hidden states (CPU Float32Array)
 * @param numTokens - Number of tokens
 * @param config - MoE configuration
 * @param moeRouter - MoE router instance
 * @param expertWeights - Map of loaded expert weights
 * @param expertLoader - Loader for on-demand expert weights
 * @param layerIdx - Current layer index
 * @returns Combined expert outputs
 */
export async function moeFeedForwardCPU(
  hiddenStates: Float32Array,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number
): Promise<Float32Array> {
  // 1. Route tokens to experts
  const selections = moeRouter.route(hiddenStates, numTokens);

  // 2. Create execution plan (group tokens by expert)
  const plan = createExpertExecutionPlan(selections, config.numExperts);

  // 3. Execute each active expert
  const expertOutputs = new Map<number, Float32Array>();

  for (const [expertIdx, data] of plan) {
    if (data.tokenIndices.length === 0) continue;

    // Load expert weights on demand
    await ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader);

    // Gather tokens for this expert
    const expertInput = gatherTokens(hiddenStates, data.tokenIndices, config.hiddenSize);

    // Run expert FFN
    const expertOutput = await runExpertCPU(
      layerIdx,
      expertIdx,
      expertInput,
      config,
      expertWeights
    );
    expertOutputs.set(expertIdx, expertOutput);
  }

  // 4. Combine expert outputs with routing weights
  const combined = combineExpertOutputs(
    expertOutputs,
    selections,
    numTokens,
    config.hiddenSize
  );

  return combined;
}

// ============================================================================
// MoE Feed-Forward (Full GPU Path)
// ============================================================================

/**
 * MoE feed-forward fully on GPU.
 *
 * This is the optimized path with GPU-native routing, gathering, and scatter-add.
 * All operations stay on GPU until the final output.
 *
 * @param inputBuffer - Input hidden states (GPU buffer)
 * @param numTokens - Number of tokens
 * @param config - MoE configuration
 * @param moeRouter - MoE router instance with GPU gate weights
 * @param expertWeights - Map of loaded expert weights
 * @param expertLoader - Loader for on-demand expert weights
 * @param layerIdx - Current layer index
 * @param layerRouterWeights - Optional per-layer router weights
 * @returns Output GPU buffer
 */
export async function moeFeedForwardGPU(
  inputBuffer: GPUBuffer,
  numTokens: number,
  config: MoEConfig,
  moeRouter: MoERouter,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader,
  layerIdx: number,
  layerRouterWeights?: Map<number, LayerRouterWeights>
): Promise<GPUBuffer> {
  const device = getDevice();
  if (!device) throw new Error('No GPU device for MoE');

  const { hiddenSize, numExperts, intermediateSize, moeTopK, hiddenActivation } = config;
  const topK = moeTopK || moeRouter.topK || 2;

  if (!moeRouter || !moeRouter.gateWeight) {
    throw new Error('MoE router not initialized');
  }

  // Load per-layer router if available
  const layerRouter = layerRouterWeights?.get(layerIdx) || null;
  if (layerRouter) {
    moeRouter.loadWeights(layerRouter.weight, layerRouter.bias || null);
  }

  // 1. Compute router logits on GPU: hidden_states @ gate_weight
  const logitsBuffer = await moeRouter.computeRouterLogitsGPU(inputBuffer, numTokens);

  // 2. Fused softmax + top-k selection on GPU
  const { indices: indicesBuffer, weights: weightsBuffer } = await runSoftmaxTopK(
    logitsBuffer,
    numTokens,
    numExperts,
    topK,
    { normalize: moeRouter.normalizeWeights }
  );

  // Clean up logits buffer
  releaseBuffer(logitsBuffer);

  // 3. Gather tokens by expert on GPU (sparse MoE execution)
  // Wrap inputBuffer in Tensor (MoE always uses F32)
  const inputTensor = createTensor(inputBuffer, 'f32', [numTokens, hiddenSize], 'moe_input');
  const { gathered, tokenCounts, tokenMap, maxTokensPerExpert } = await runMoEGather(
    inputTensor,
    indicesBuffer,
    numTokens,
    hiddenSize,
    numExperts,
    topK,
    { maxTokensPerExpert: numTokens }
  );

  // Allocate expert output buffer in gathered-slot order:
  // [numExperts, maxTokensPerExpert, hiddenSize]
  const expertOutputs = acquireBuffer(
    numExperts * maxTokensPerExpert * hiddenSize * 4,
    undefined,
    'moe_expert_outputs_gathered'
  );

  // Zero-initialize (covers empty slots and experts with no tokens)
  const zeroEncoder = device.createCommandEncoder({ label: 'zero_moe_expert_outputs' });
  zeroEncoder.clearBuffer(expertOutputs, 0, numExperts * maxTokensPerExpert * hiddenSize * 4);
  device.queue.submit([zeroEncoder.finish()]);

  // Read back tokenCounts and tokenMap to build tokenOffsets for dynamic scatter-add
  const countsData = await readBuffer(tokenCounts, numExperts * 4);
  const tokenCountsCPU = new Uint32Array(countsData);

  const tokenMapElems = numExperts * maxTokensPerExpert * 2;
  const tokenMapData = await readBuffer(tokenMap, tokenMapElems * 4);
  const tokenMapCPU = new Uint32Array(tokenMapData);

  // Build tokenOffsets for scatter-add
  const tokenOffsetsCPU = new Uint32Array(numTokens * topK);
  tokenOffsetsCPU.fill(0xFFFFFFFF);

  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCountsCPU[expertIdx] || 0;
    if (count > maxTokensPerExpert) {
      throw new Error(
        `[MoE] Gather overflow: expert ${expertIdx} count=${count} > maxTokensPerExpert=${maxTokensPerExpert}`
      );
    }
    for (let slotIdx = 0; slotIdx < count; slotIdx++) {
      const mapBase = (expertIdx * maxTokensPerExpert + slotIdx) * 2;
      const tokenIdx = tokenMapCPU[mapBase];
      const kIdx = tokenMapCPU[mapBase + 1];
      tokenOffsetsCPU[tokenIdx * topK + kIdx] = expertIdx * maxTokensPerExpert + slotIdx;
    }
  }

  // Validate all offsets are filled
  for (let i = 0; i < tokenOffsetsCPU.length; i++) {
    if (tokenOffsetsCPU[i] === 0xFFFFFFFF) {
      const tokenIdx = Math.floor(i / topK);
      const kIdx = i % topK;
      log.error('MoE', `Missing offset at i=${i} (token=${tokenIdx}, k=${kIdx})`);
      throw new Error(`[MoE] tokenOffsets incomplete at i=${i}`);
    }
  }

  const tokenOffsets = acquireBuffer(tokenOffsetsCPU.byteLength, undefined, 'moe_token_offsets');
  device.queue.writeBuffer(tokenOffsets, 0, tokenOffsetsCPU);

  // tokenCounts is a non-pooled GPUBuffer from runMoEGather
  tokenCounts.destroy();

  // 4. Execute only active experts (count > 0) on GPU
  const bytesPerToken = hiddenSize * 4;
  const expertStrideBytes = maxTokensPerExpert * bytesPerToken;

  for (let expertIdx = 0; expertIdx < numExperts; expertIdx++) {
    const count = tokenCountsCPU[expertIdx] || 0;
    if (count === 0) continue;

    await ensureExpertLoaded(layerIdx, expertIdx, expertWeights, expertLoader);
    const expertKey = `layer_${layerIdx}_expert_${expertIdx}`;
    const weights = expertWeights.get(expertKey);
    if (!weights) continue;

    const inputOffset = expertIdx * expertStrideBytes;
    const outputOffset = expertIdx * expertStrideBytes;

    if (weights.isGptOss) {
      // GPT-OSS experts are stored in MXFP4-packed tensors with a fused gate_up projection
      await runGptOssExpert(
        gathered,
        expertOutputs,
        weights,
        layerIdx,
        expertIdx,
        count,
        inputOffset,
        outputOffset,
        hiddenSize,
        intermediateSize,
        numExperts
      );
    } else if (weights.gate && weights.up && weights.down) {
      // Mixtral-style expert FFN: gate/up projections, activation, down projection
      await runMixtralExpert(
        gathered,
        expertOutputs,
        weights,
        count,
        inputOffset,
        outputOffset,
        hiddenSize,
        intermediateSize,
        hiddenActivation
      );
    }
  }

  // 5. Dynamic scatter-add: combine expert outputs weighted by routing probabilities
  // Wrap expertOutputs in Tensor (MoE uses F32)
  const expertOutputsTensor = createTensor(
    expertOutputs,
    'f32',
    [numExperts, maxTokensPerExpert, hiddenSize],
    'moe_expert_outputs'
  );
  const outputTensor = await runScatterAddDynamic(
    expertOutputsTensor,
    indicesBuffer,
    weightsBuffer,
    tokenOffsets,
    numTokens,
    hiddenSize,
    topK
  );

  // Cleanup
  releaseBuffer(gathered.buffer);
  releaseBuffer(tokenMap);
  releaseBuffer(expertOutputs);
  releaseBuffer(tokenOffsets);
  releaseBuffer(indicesBuffer);
  releaseBuffer(weightsBuffer);

  return outputTensor.buffer;
}

// ============================================================================
// Expert Execution Helpers
// ============================================================================

/**
 * Run GPT-OSS style expert (MXFP4 quantized).
 * Uses dequant cache to avoid re-dequantizing same expert weights.
 */
async function runGptOssExpert(
  gathered: Tensor,
  expertOutputs: GPUBuffer,
  weights: MoEExpertWeights,
  layerIdx: number,
  expertIdx: number,
  count: number,
  inputOffset: number,
  outputOffset: number,
  hiddenSize: number,
  intermediateSize: number,
  numExperts: number
): Promise<void> {
  const outDim = intermediateSize * 2;

  if (hiddenSize % 32 !== 0 || intermediateSize % 32 !== 0) {
    throw new Error(
      `[MoE] GPT-OSS MXFP4 expects hiddenSize and intermediateSize divisible by 32, got ` +
      `hiddenSize=${hiddenSize} intermediateSize=${intermediateSize}`
    );
  }

  const gateUpGroups = hiddenSize / 32;
  const downGroups = intermediateSize / 32;
  const totalExperts = weights.numExperts || numExperts;

  if (!weights.gateUpBlocks || !weights.gateUpScales || !weights.gateUpBias ||
      !weights.downBlocks || !weights.downScales) {
    log.warn('MoE', `GPT-OSS expert ${expertIdx} missing tensors, skipping`);
    return;
  }

  // Check dequant cache first
  let gateUpWeight: GPUBuffer;
  let downWeight: GPUBuffer;
  const cached = getCachedDequant(layerIdx, expertIdx);

  if (cached) {
    // Use cached dequantized weights
    gateUpWeight = cached.gateUp;
    downWeight = cached.down;
  } else {
    // Dequantize and cache (extract .buffer from Tensor)
    const gateUpTensor = await dequantizeMXFP4Expert(
      weights.gateUpBlocks,
      weights.gateUpScales,
      expertIdx,
      totalExperts,
      outDim,
      gateUpGroups
    );
    const downTensor = await dequantizeMXFP4Expert(
      weights.downBlocks,
      weights.downScales,
      expertIdx,
      totalExperts,
      hiddenSize,
      downGroups
    );
    gateUpWeight = gateUpTensor.buffer;
    downWeight = downTensor.buffer;
    setCachedDequant(layerIdx, expertIdx, gateUpWeight, downWeight);
  }

  // gate_up projection: [count, hiddenSize] x [hiddenSize, outDim]
  const gateUpOut = await runMatmul(
    gathered,
    gateUpWeight,
    count,
    outDim,
    hiddenSize,
    { transposeB: 'auto', aOffset: inputOffset }
  );
  // Don't release cached weights

  // SwiGLU with per-expert bias: output [count, intermediateSize]
  const biasOffset = expertIdx * outDim * 4;
  const biasTensor = createTensor(weights.gateUpBias, 'f32', [outDim], 'moe_gate_up_bias');
  const activated = await runSwiGLURowsplitBias(
    gateUpOut,
    biasTensor,
    count,
    intermediateSize,
    { biasOffset }
  );
  releaseBuffer(gateUpOut.buffer);

  // down projection to expertOutputs slice
  await runMatmul(
    activated,
    downWeight,
    count,
    hiddenSize,
    intermediateSize,
    { transposeB: 'auto', outputBuffer: expertOutputs, cOffset: outputOffset }
  );
  // Don't release cached weights
  releaseBuffer(activated.buffer);

  // Add down bias in-place (optional)
  if (weights.downBias) {
    const downBiasOffset = expertIdx * hiddenSize * 4;
    // Wrap expertOutputs and downBias in Tensor for runBiasAdd
    const expertOutputsTensor = createTensor(expertOutputs, 'f32', [count, hiddenSize], 'expert_outputs');
    const downBiasTensor = createTensor(weights.downBias, 'f32', [hiddenSize], 'down_bias');
    await runBiasAdd(expertOutputsTensor, downBiasTensor, count, hiddenSize, {
      dataOffset: outputOffset,
      biasOffset: downBiasOffset,
    });
  }
}

/**
 * Run Mixtral-style expert (gate/up/down).
 */
async function runMixtralExpert(
  gathered: Tensor,
  expertOutputs: GPUBuffer,
  weights: MoEExpertWeights,
  count: number,
  inputOffset: number,
  outputOffset: number,
  hiddenSize: number,
  intermediateSize: number,
  hiddenActivation: string
): Promise<void> {
  // GPU path - weights are always GPUBuffers here
  const gateOut = await runMatmul(
    gathered,
    weights.gate as GPUBuffer,
    count,
    intermediateSize,
    hiddenSize,
    { transposeB: 'auto', aOffset: inputOffset }
  );
  const upOut = await runMatmul(
    gathered,
    weights.up as GPUBuffer,
    count,
    intermediateSize,
    hiddenSize,
    { transposeB: 'auto', aOffset: inputOffset }
  );

  const activationFn = hiddenActivation === 'gelu' ? runGeLU : runSiLU;
  const activated = await activationFn(upOut, {
    size: count * intermediateSize,
    gate: gateOut,
  });
  releaseBuffer(gateOut.buffer);
  releaseBuffer(upOut.buffer);

  await runMatmul(
    activated,
    weights.down as GPUBuffer,
    count,
    hiddenSize,
    intermediateSize,
    { transposeB: 'auto', outputBuffer: expertOutputs, cOffset: outputOffset }
  );
  releaseBuffer(activated.buffer);
}

/**
 * Run expert FFN on GPU with CPU readback.
 */
async function runExpertCPU(
  layerIdx: number,
  expertIdx: number,
  input: Float32Array,
  config: MoEConfig,
  expertWeights: Map<string, MoEExpertWeights>
): Promise<Float32Array> {
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  const weights = expertWeights.get(key);

  if (!weights || !weights.gate || !weights.up || !weights.down) {
    log.warn('MoE', `Expert ${expertIdx} weights not available for layer ${layerIdx}`);
    return new Float32Array(input.length);
  }

  const device = getDevice();
  const { hiddenSize, intermediateSize, hiddenActivation } = config;
  const numTokens = input.length / hiddenSize;

  if (!device) {
    // CPU fallback
    return new Float32Array(input.length);
  }

  // 1. Create input buffer and wrap in Tensor
  const inputBuffer = acquireBuffer(input.byteLength, undefined, 'expert_input');
  device.queue.writeBuffer(inputBuffer, 0, input as unknown as BufferSource);
  const inputTensor = createTensor(inputBuffer, 'f32', [numTokens, hiddenSize], 'expert_input');

  // 2. Gate projection
  const gateOutput = await runMatmul(inputTensor, weights.gate as GPUBuffer, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 3. Up projection
  const upOutput = await runMatmul(inputTensor, weights.up as GPUBuffer, numTokens, intermediateSize, hiddenSize, { transposeB: 'auto' });

  // 4. Activation
  const activationFn = hiddenActivation === 'gelu' ? runGeLU : runSiLU;
  const activatedOutput = await activationFn(upOutput, {
    size: numTokens * intermediateSize,
    gate: gateOutput,
  });

  // 5. Down projection
  const output = await runMatmul(activatedOutput, weights.down as GPUBuffer, numTokens, hiddenSize, intermediateSize, { transposeB: 'auto' });

  // 6. Read output back
  const outputData = await readBuffer(output.buffer, input.byteLength);

  // Cleanup
  releaseBuffer(inputBuffer);
  releaseBuffer(gateOutput.buffer);
  releaseBuffer(upOutput.buffer);
  releaseBuffer(activatedOutput.buffer);
  releaseBuffer(output.buffer);

  return new Float32Array(outputData);
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Ensure expert weights are loaded.
 */
async function ensureExpertLoaded(
  layerIdx: number,
  expertIdx: number,
  expertWeights: Map<string, MoEExpertWeights>,
  expertLoader: ExpertLoader
): Promise<void> {
  const key = `layer_${layerIdx}_expert_${expertIdx}`;
  if (expertWeights.has(key)) return;

  const weights = await expertLoader.loadExpert(layerIdx, expertIdx);
  if (weights) {
    expertWeights.set(key, weights);
  }
}

/**
 * Gather tokens by indices (CPU helper).
 */
function gatherTokens(
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

/**
 * Check if layer is MoE layer (some models have dense layers too).
 *
 * @param layerIdx - Layer index
 * @returns True if layer uses MoE
 */
export function isMoELayer(_layerIdx: number): boolean {
  // For Mixtral/DeepSeek, all layers are MoE
  // Some models alternate between dense and MoE
  return true;
}
