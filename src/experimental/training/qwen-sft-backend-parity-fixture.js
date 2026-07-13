import { f16ToF32Array, f32ToF16Array } from '../../inference/kv-cache/types.js';
import { sha256Hex } from '../../utils/sha256.js';

function values(length, offset, scale) {
  return Float32Array.from(
    { length },
    (_, index) => Math.sin((index + offset) * 0.37) * scale
  );
}

function frozenTensor(shape, offset, scale) {
  const length = shape.reduce((product, value) => product * value, 1);
  const rounded = f16ToF32Array(f32ToF16Array(values(length, offset, scale)));
  return { shape, data: Array.from(rounded) };
}

function adapterPair(inputSize, outputSize, rank, alpha, offset) {
  return {
    rank,
    alpha,
    A: {
      shape: [inputSize, rank],
      data: Array.from(values(inputSize * rank, offset, 0.065)),
    },
    B: {
      shape: [rank, outputSize],
      data: Array.from(values(rank * outputSize, offset + 17, 0.05)),
    },
  };
}

export function createQwenSftBackendParityFixture(options = {}) {
  const rank = Number(options.rank ?? 32);
  const alpha = Number(options.alpha ?? 64);
  if (!Number.isInteger(rank) || rank < 1 || !Number.isFinite(alpha)) {
    throw new Error('Qwen SFT backend parity fixture requires positive rank and finite alpha.');
  }
  const model = {
    numTokens: 3,
    hiddenSize: 4,
    intermediateSize: 6,
    vocabSize: 11,
    activeTokenCount: 2,
    rmsEps: 1e-6,
  };
  const layer = {
    seqLen: model.numTokens,
    hiddenSize: model.hiddenSize,
    intermediateSize: model.intermediateSize,
    numHeads: 2,
    numKVHeads: 1,
    headDim: 16,
    rotaryDim: 4,
    pairSpanDim: 4,
    interleaved: false,
    startPos: 0,
    rmsEps: model.rmsEps,
  };
  const querySize = layer.numHeads * layer.headDim;
  const kvSize = layer.numKVHeads * layer.headDim;
  const cosine = Array.from(
    { length: model.numTokens * (layer.rotaryDim / 2) },
    (_, index) => Math.cos(index * 0.19)
  );
  const sine = Array.from(
    { length: cosine.length },
    (_, index) => Math.sin(index * 0.19)
  );
  const prefixRows = [
    { rowId: 'qwen-prefix-1', tokenIds: [1, 2, 3], targets: [-100, 4, 5] },
    { rowId: 'qwen-prefix-2', tokenIds: [2, 4, 6], targets: [-100, 5, 7] },
    { rowId: 'qwen-prefix-3', tokenIds: [3, 5, 7], targets: [-100, 6, 8] },
    { rowId: 'qwen-prefix-4', tokenIds: [4, 6, 8], targets: [-100, 7, 9] },
  ];
  const consumedPrefixSha256 = sha256Hex(
    JSON.stringify(prefixRows.map((row) => row.rowId))
  );
  return {
    artifactType: 'qwen_sft_backend_parity_fixture',
    schemaVersion: 1,
    precisionContract: {
      frozenWeights: 'f16_rounded_values_promoted_to_f32_compute',
      activations: 'f32',
      gradients: 'f32',
      adapterParameters: 'f32',
      optimizerState: 'f32',
      adapterDropout: 0,
    },
    architectureContract: {
      modelId: 'Qwen/Qwen3.5-9B',
      revision: 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
      transformersVersion: '5.13.1',
      transformersDecoderSourceSha256: 'cf085792cb59e5bdf9b88a3d20bd353892289d054662a9c2b662221b97caefba',
      decoderOrder: 'input_norm_token_mixer_residual_post_attention_norm_mlp_residual',
      ropePairing: 'split_half_within_partial_rotary_prefix',
      partialRotaryFactor: 0.25,
    },
    model,
    layer,
    tokenIds: [1, 2, 3],
    targets: [-100, 4, 5],
    unmaskedTargets: [6, 4, 5],
    prefixRows,
    prefixContract: {
      accumulationSteps: 2,
      microstepCount: prefixRows.length,
      optimizerStepCount: 2,
      checkpointAfterMicrostep: 2,
      checkpointAfterOptimizerStep: 1,
      consumedPrefixSha256,
    },
    frozen: {
      embedding: frozenTensor([model.vocabSize, model.hiddenSize], 3, 0.12),
      inputNorm: frozenTensor([model.hiddenSize], 59, 0.04),
      postAttentionNorm: frozenTensor([model.hiddenSize], 71, 0.04),
      qWeight: frozenTensor([querySize * 2, model.hiddenSize], 83, 0.1),
      kWeight: frozenTensor([kvSize, model.hiddenSize], 151, 0.09),
      vWeight: frozenTensor([kvSize, model.hiddenSize], 179, 0.09),
      oWeight: frozenTensor([model.hiddenSize, querySize], 211, 0.1),
      qNorm: frozenTensor([layer.headDim], 251, 0.04),
      kNorm: frozenTensor([layer.headDim], 263, 0.04),
      gateWeight: frozenTensor([model.intermediateSize, model.hiddenSize], 277, 0.11),
      upWeight: frozenTensor([model.intermediateSize, model.hiddenSize], 307, 0.1),
      downWeight: frozenTensor([model.hiddenSize, model.intermediateSize], 337, 0.1),
      finalNorm: frozenTensor([model.hiddenSize], 367, 0.04),
      lmHead: frozenTensor([model.vocabSize, model.hiddenSize], 379, 0.11),
      cosine: { shape: [model.numTokens, layer.rotaryDim / 2], data: cosine },
      sine: { shape: [model.numTokens, layer.rotaryDim / 2], data: sine },
    },
    adapters: {
      'layers.0.self_attn.q_proj': adapterPair(
        model.hiddenSize, querySize * 2, rank, alpha, 431
      ),
      'layers.0.self_attn.k_proj': adapterPair(
        model.hiddenSize, kvSize, rank, alpha, 461
      ),
      'layers.0.self_attn.v_proj': adapterPair(
        model.hiddenSize, kvSize, rank, alpha, 491
      ),
      'layers.0.self_attn.o_proj': adapterPair(
        querySize, model.hiddenSize, rank, alpha, 521
      ),
      'layers.0.mlp.gate_proj': adapterPair(
        model.hiddenSize, model.intermediateSize, rank, alpha, 557
      ),
      'layers.0.mlp.up_proj': adapterPair(
        model.hiddenSize, model.intermediateSize, rank, alpha, 593
      ),
      'layers.0.mlp.down_proj': adapterPair(
        model.intermediateSize, model.hiddenSize, rank, alpha, 631
      ),
    },
    optimizer: {
      type: 'adamw',
      lr: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.01,
    },
    claimBoundary: 'Deterministic tiny one-full-layer rank-32 fixture with the pinned Qwen 3.5 decoder order, pre-shifted completion targets, a four-row matched prefix, and zero adapter dropout; not production Qwen geometry or the PEFT default initialization distribution.',
  };
}
