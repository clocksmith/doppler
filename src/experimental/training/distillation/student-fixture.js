import { getDevice } from '../../../gpu/device.js';
import { createTrainingConfig } from '../../../config/training-defaults.js';
import {
  runAttention,
  castF16ToF32,
  runGather,
  runMatmul,
  runResidualAdd,
  runRMSNorm,
  runRoPE,
  runScale,
  runSiLU,
  runSiLURowSplit,
} from '../../../gpu/kernels/index.js';
import { createTensor } from '../../../gpu/tensor.js';
import { acquireBuffer, uploadData, releaseBuffer } from '../../../memory/buffer-pool.js';
import { getBufferDtype, getWeightDtype, isCpuWeightBuffer, isWeightBuffer } from '../../../gpu/weight-buffer.js';
import { OpType } from '../autograd.js';
import { LoraAdapter } from '../lora.js';
import { normalizeOptionalString } from './suite-data.js';
import { LORA_MODULE_ALIASES } from '../../../inference/pipelines/text/lora.js';
import { resolveEmbeddingScale } from '../../../inference/pipelines/text/embed.js';
import { isSlidingLayerType } from '../../../inference/pipelines/text/attention/dispatch-params.js';
import { createDistillStudentProjectionModelFixture, ensureTrainableTensor, normalizeTransformerLoraConfig, releaseTensor, resolvePhasePrompts, resolveTensorDtype } from './student-fixture/model.js';
import { DISTILL_STUDENT_GRAPH_PROJECTION, createRowSliceTensor, createTransformerLoraAdapters, disposeTransformerLoraAdapters, makeTensorFromUint32, normalizeDistillStudentGraphMode, recordTensorView, resolveLayerFfnIntermediateSize } from './student-fixture/execution.js';

async function ensureNormTensor(value, hiddenSize, label, ownedTrainables = null) {
  return ensureTrainableTensor(value, [hiddenSize], label, ownedTrainables);
}

function hasTensorPayload(value) {
  if (!value) return false;
  if (value instanceof GPUBuffer) return true;
  if (isWeightBuffer(value) || isCpuWeightBuffer(value)) return true;
  if (value?.buffer instanceof GPUBuffer) return true;
  if (ArrayBuffer.isView(value) || Array.isArray(value)) return true;
  return false;
}

async function createDistillStudentTransformerModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const studentPipeline = distillRuntime?.studentPipeline || null;
  if (!studentPipeline?.modelConfig || !(studentPipeline.weights instanceof Map)) {
    throw new Error('Distill full-graph student fixture requires loaded student pipeline weights.');
  }
  const modelConfig = studentPipeline.modelConfig;
  const hiddenSize = Math.max(1, Math.floor(Number(modelConfig.hiddenSize) || 0));
  const intermediateSize = Math.max(1, Math.floor(Number(modelConfig.intermediateSize) || 0));
  const numLayers = Math.max(1, Math.floor(Number(modelConfig.numLayers) || 0));
  const numHeads = Math.max(1, Math.floor(Number(modelConfig.numHeads) || 0));
  const numKVHeads = Math.max(1, Math.floor(Number(modelConfig.numKVHeads || numHeads) || 0));
  const headDim = Math.max(1, Math.floor(Number(modelConfig.headDim) || 0));
  const vocabSize = Math.max(1, Math.floor(Number(modelConfig.vocabSize) || 0));
  const rmsNormEps = Number.isFinite(modelConfig.rmsNormEps) ? modelConfig.rmsNormEps : 1e-6;
  const hiddenActivation = String(modelConfig.hiddenActivation || 'silu').toLowerCase();
  const swigluLimit = Number.isFinite(modelConfig.swigluLimit) ? modelConfig.swigluLimit : 0;
  const useEmbeddingTranspose = modelConfig.embeddingTranspose === true;
  const tieWordEmbeddings = modelConfig.useTiedEmbeddings === true;
  const embeddingScale = resolveEmbeddingScale(modelConfig, hiddenSize);
  const loraConfig = normalizeTransformerLoraConfig(options.loraAdapter || null);
  const freezeBaseGrad = Boolean(loraConfig);
  const frozenWeightOptions = { preserveF16: freezeBaseGrad };
  const stopBaseWeight = freezeBaseGrad ? { stopGradInputs: [1] } : {};
  const stopRopeWeights = freezeBaseGrad ? { stopGradInputs: [1, 2] } : {};

  const config = createTrainingConfig({
    ...overrides,
    training: {
      enabled: true,
      lossScaling: { enabled: false },
      gradient: { maxNorm: 0 },
      ...(overrides.training || {}),
    },
  });

  const ownedTrainables = new Set();
  const embeddingWeight = await ensureTrainableTensor(
    studentPipeline.weights.get('embed'),
    [vocabSize, hiddenSize],
    'embed',
    ownedTrainables,
    frozenWeightOptions
  );
  const lmHeadWeight = tieWordEmbeddings
    ? embeddingWeight
    : await ensureTrainableTensor(
      studentPipeline.weights.get('lm_head'),
      [vocabSize, hiddenSize],
      'lm_head',
      ownedTrainables,
      frozenWeightOptions
    );
  const finalNormWeight = await ensureNormTensor(
    studentPipeline.weights.get('final_norm'),
    hiddenSize,
    'final_norm',
    ownedTrainables
  );

  const ropeDim = Math.max(1, Math.floor(headDim / 2));
  const ropeRows = Math.max(1, Math.floor(Number(modelConfig.maxSeqLen) || 1));
  const ropeCos = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsCos, 'f32', [ropeRows, ropeDim], 'rope_cos'),
    [ropeRows, ropeDim],
    'rope_cos',
    ownedTrainables
  );
  const ropeSin = await ensureTrainableTensor(
    createTensor(studentPipeline.ropeFreqsSin, 'f32', [ropeRows, ropeDim], 'rope_sin'),
    [ropeRows, ropeDim],
    'rope_sin',
    ownedTrainables
  );
  const hasLocalAttention = Array.isArray(modelConfig.layerTypes)
    && modelConfig.layerTypes.some((layerType) => isSlidingLayerType(layerType));
  let ropeLocalCos = null;
  let ropeLocalSin = null;
  if (hasLocalAttention) {
    if (!(studentPipeline.ropeLocalCos instanceof GPUBuffer)
      || !(studentPipeline.ropeLocalSin instanceof GPUBuffer)) {
      throw new Error(
        'Distill full-graph student requires local RoPE tables for sliding-attention layers.'
      );
    }
    ropeLocalCos = await ensureTrainableTensor(
      createTensor(studentPipeline.ropeLocalCos, 'f32', [ropeRows, ropeDim], 'rope_local_cos'),
      [ropeRows, ropeDim],
      'rope_local_cos',
      ownedTrainables
    );
    ropeLocalSin = await ensureTrainableTensor(
      createTensor(studentPipeline.ropeLocalSin, 'f32', [ropeRows, ropeDim], 'rope_local_sin'),
      [ropeRows, ropeDim],
      'rope_local_sin',
      ownedTrainables
    );
  }

  const layerParams = [];
  const loraParams = [];
  const layers = [];
  const loraDims = {
    hiddenSize,
    intermediateSize,
    numHeads,
    numKVHeads,
    headDim,
    attentionSize: numHeads * headDim,
  };
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx += 1) {
    const layerWeights = studentPipeline.weights.get(`layer_${layerIdx}`);
    if (!layerWeights) {
      throw new Error(`Distill full-graph student missing layer_${layerIdx} weights.`);
    }
    const layerIntermediateSize = resolveLayerFfnIntermediateSize(layerIdx, layerWeights, intermediateSize);
    const gateUpWeight = layerWeights.gateUp || layerWeights.ffnGateUp || null;
    let layerGateUp = null;
    let layerGate = null;
    let layerUp = null;
    if (hasTensorPayload(gateUpWeight)) {
      layerGateUp = await ensureTrainableTensor(
        gateUpWeight,
        [layerIntermediateSize * 2, hiddenSize],
        `layer_${layerIdx}.ffn_gate_up`,
        ownedTrainables,
        frozenWeightOptions
      );
    } else {
      const gateWeight = layerWeights.gate || layerWeights.ffnGate || null;
      const upWeight = layerWeights.up || layerWeights.ffnUp || null;
      if (!hasTensorPayload(gateWeight) || !hasTensorPayload(upWeight)) {
        throw new Error(
          `Distill full-graph student missing gate/up projections on layer ${layerIdx}.`
        );
      }
      layerGate = await ensureTrainableTensor(
        gateWeight,
        [layerIntermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_gate`,
        ownedTrainables,
        frozenWeightOptions
      );
      layerUp = await ensureTrainableTensor(
        upWeight,
        [layerIntermediateSize, hiddenSize],
        `layer_${layerIdx}.ffn_up`,
        ownedTrainables,
        frozenWeightOptions
      );
    }
    const layer = {
      inputNorm: await ensureNormTensor(
        layerWeights.inputNorm,
        hiddenSize,
        `layer_${layerIdx}.input_norm`,
        ownedTrainables
      ),
      queryKeyNorm: modelConfig.queryKeyNorm === true
        && (!Array.isArray(modelConfig.queryKeyNormLayers)
          || modelConfig.queryKeyNormLayers.includes(layerIdx)),
      qNorm: null,
      kNorm: null,
      qProj: await ensureTrainableTensor(
        layerWeights.qProj,
        [numHeads * headDim, hiddenSize],
        `layer_${layerIdx}.q_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      kProj: await ensureTrainableTensor(
        layerWeights.kProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.k_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      vProj: await ensureTrainableTensor(
        layerWeights.vProj,
        [numKVHeads * headDim, hiddenSize],
        `layer_${layerIdx}.v_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      oProj: await ensureTrainableTensor(
        layerWeights.oProj,
        [hiddenSize, numHeads * headDim],
        `layer_${layerIdx}.o_proj`,
        ownedTrainables,
        frozenWeightOptions
      ),
      postAttentionNorm: layerWeights.postAttentionNorm
        ? await ensureNormTensor(
          layerWeights.postAttentionNorm,
          hiddenSize,
          `layer_${layerIdx}.post_attention_norm`,
          ownedTrainables
        )
        : null,
      preFeedforwardNorm: layerWeights.preFeedforwardNorm
        ? await ensureNormTensor(
          layerWeights.preFeedforwardNorm,
          hiddenSize,
          `layer_${layerIdx}.pre_feedforward_norm`,
          ownedTrainables
        )
        : null,
      postFeedforwardNorm: layerWeights.postFeedforwardNorm
        ? await ensureNormTensor(
          layerWeights.postFeedforwardNorm,
          hiddenSize,
          `layer_${layerIdx}.post_feedforward_norm`,
          ownedTrainables
        )
        : null,
      gateUp: layerGateUp,
      gate: layerGate,
      up: layerUp,
      down: await ensureTrainableTensor(
        layerWeights.down || layerWeights.ffnDown,
        [hiddenSize, layerIntermediateSize],
        `layer_${layerIdx}.ffn_down`,
        ownedTrainables,
        frozenWeightOptions
      ),
      intermediateSize: layerIntermediateSize,
      lora: createTransformerLoraAdapters(loraConfig, {
        ...loraDims,
        intermediateSize: layerIntermediateSize,
      }),
    };
    if (layer.gateUp && (layer.lora.gate_proj || layer.lora.up_proj)) {
      throw new Error(
        `Layer ${layerIdx} has fused gate/up weights but separate gate_proj or up_proj LoRA targets.`
      );
    }
    if (!layer.gateUp && layer.lora.gate_up_proj) {
      throw new Error(
        `Layer ${layerIdx} has separate gate/up weights but a fused gate_up_proj LoRA target.`
      );
    }
    if (layer.queryKeyNorm) {
      const weightedLayers = modelConfig.queryKeyNormWeightLayers;
      const expectsWeightedNorm = !Array.isArray(weightedLayers) || weightedLayers.includes(layerIdx);
      if (!expectsWeightedNorm) {
        throw new Error(
          `Distill full-graph student does not support unit-weight Q/K norm on layer ${layerIdx}.`
        );
      }
      layer.qNorm = await ensureNormTensor(
        layerWeights.qNorm,
        headDim,
        `layer_${layerIdx}.q_norm`,
        ownedTrainables
      );
      layer.kNorm = await ensureNormTensor(
        layerWeights.kNorm,
        headDim,
        `layer_${layerIdx}.k_norm`,
        ownedTrainables
      );
    }
    if (modelConfig.postAttentionNorm === true && !layer.postAttentionNorm) {
      throw new Error(`Distill full-graph student missing post-attention norm on layer ${layerIdx}.`);
    }
    if (modelConfig.preFeedforwardNorm === true && !layer.preFeedforwardNorm) {
      throw new Error(`Distill full-graph student missing pre-feedforward norm on layer ${layerIdx}.`);
    }
    if (modelConfig.postFeedforwardNorm === true && !layer.postFeedforwardNorm) {
      throw new Error(`Distill full-graph student missing post-feedforward norm on layer ${layerIdx}.`);
    }
    layers.push(layer);
    layerParams.push(
      layer.inputNorm,
      layer.qProj,
      layer.kProj,
      layer.vProj,
      layer.oProj,
      ...(layer.gateUp ? [layer.gateUp] : [layer.gate, layer.up]),
      layer.down
    );
    for (const adapter of Object.values(layer.lora)) {
      loraParams.push(adapter.A, adapter.B);
    }
    if (layer.postAttentionNorm) {
      layerParams.push(layer.postAttentionNorm);
    }
    if (layer.preFeedforwardNorm) {
      layerParams.push(layer.preFeedforwardNorm);
    }
    if (layer.postFeedforwardNorm) {
      layerParams.push(layer.postFeedforwardNorm);
    }
    if (layer.qNorm) {
      layerParams.push(layer.qNorm, layer.kNorm);
    }
  }

  const encoderParams = [embeddingWeight, ...layerParams];
  const decoderParams = [finalNormWeight, lmHeadWeight];
  const baseParams = [...encoderParams, ...decoderParams];
  const temporaryInputs = new Set();

  async function buildPromptTokens(prompt) {
    const normalized = String(prompt || '').trim();
    if (!normalized) {
      throw new Error('Distill full-graph student prompt is empty.');
    }
    const tokenIds = studentPipeline.tokenizer.encode(normalized);
    if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
      throw new Error('Distill full-graph student tokenizer produced no tokens.');
    }
    const tokenTensor = makeTensorFromUint32(
      tokenIds,
      [tokenIds.length],
      'distill_student_prompt_tokens'
    );
    temporaryInputs.add(tokenTensor);
    return { tokenTensor, seqLen: tokenIds.length };
  }

  async function runTransformerTokenTensor(tokenTensor, seqLen, tape, forwardOptions = {}) {
    const captureStage = async (stage, tensor, layerIdx = null) => {
      if (typeof forwardOptions.captureStage === 'function') {
        await forwardOptions.captureStage({
          stage,
          layerIdx,
          seqLen,
          tensor,
        });
      }
      return tensor;
    };
    let hidden = await tape.record(
      OpType.EMBED,
      (indices, embeddings) => runGather(
        indices,
        embeddings,
        seqLen,
        hiddenSize,
        vocabSize,
        {
          embeddingDtype: resolveTensorDtype(embeddingWeight, 'f32'),
          outputDtype: 'f32',
          transpose: useEmbeddingTranspose,
        }
      ),
      [tokenTensor, embeddingWeight],
      {
        numTokens: seqLen,
        hiddenSize,
        vocabSize,
        transpose: useEmbeddingTranspose,
        indexOffset: 0,
        ...stopBaseWeight,
      }
    );
    if (embeddingScale !== 1) {
      hidden = await tape.record(
        OpType.SCALE,
        (x) => runScale(x, embeddingScale, { count: seqLen * hiddenSize }),
        [hidden],
        { count: seqLen * hiddenSize, scale: embeddingScale }
      );
    }
    await captureStage('embed.out', hidden);

    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      const layerIntermediateSize = layer.intermediateSize || intermediateSize;
      const normed = await tape.record(
        OpType.RMSNORM,
        (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
          batchSize: seqLen,
          hiddenSize,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        }),
        [hidden, layer.inputNorm],
        {
          numTokens: seqLen,
          hiddenSize,
          eps: rmsNormEps,
          rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          ...stopBaseWeight,
        }
      );
      await captureStage('attn.post_input_norm', normed, layerIdx);

      let q2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.qProj],
        { M: seqLen, N: numHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      let k2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.kProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      let v2d = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, numKVHeads * headDim, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [normed, layer.vProj],
        { M: seqLen, N: numKVHeads * headDim, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      if (layer.lora.q_proj) {
        const delta = await layer.lora.q_proj.forward(normed, tape);
        q2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numHeads * headDim),
          [q2d, delta],
          { size: seqLen * numHeads * headDim }
        );
      }
      if (layer.lora.k_proj) {
        const delta = await layer.lora.k_proj.forward(normed, tape);
        k2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numKVHeads * headDim),
          [k2d, delta],
          { size: seqLen * numKVHeads * headDim }
        );
      }
      if (layer.lora.v_proj) {
        const delta = await layer.lora.v_proj.forward(normed, tape);
        v2d = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * numKVHeads * headDim),
          [v2d, delta],
          { size: seqLen * numKVHeads * headDim }
        );
      }
      await captureStage('attn.q_proj', q2d, layerIdx);
      await captureStage('attn.k_proj', k2d, layerIdx);
      await captureStage('attn.v_proj', v2d, layerIdx);

      if (layer.queryKeyNorm) {
        const qNormInput = await recordTensorView(
          tape,
          q2d,
          [seqLen * numHeads, headDim],
          `layer_${layerIdx}_q_norm_input`
        );
        const qNormed = await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen * numHeads,
            hiddenSize: headDim,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [qNormInput, layer.qNorm],
          {
            numTokens: seqLen * numHeads,
            hiddenSize: headDim,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        );
        q2d = await recordTensorView(
          tape,
          qNormed,
          [seqLen, numHeads * headDim],
          `layer_${layerIdx}_q_normed`
        );
        const kNormInput = await recordTensorView(
          tape,
          k2d,
          [seqLen * numKVHeads, headDim],
          `layer_${layerIdx}_k_norm_input`
        );
        const kNormed = await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen * numKVHeads,
            hiddenSize: headDim,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [kNormInput, layer.kNorm],
          {
            numTokens: seqLen * numKVHeads,
            hiddenSize: headDim,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        );
        k2d = await recordTensorView(
          tape,
          kNormed,
          [seqLen, numKVHeads * headDim],
          `layer_${layerIdx}_k_normed`
        );
        await captureStage('attn.q_norm', q2d, layerIdx);
        await captureStage('attn.k_norm', k2d, layerIdx);
      }

      const q3d = await recordTensorView(
        tape,
        q2d,
        [seqLen, numHeads, headDim],
        `layer_${layerIdx}_q`
      );
      const k3d = await recordTensorView(
        tape,
        k2d,
        [seqLen, numKVHeads, headDim],
        `layer_${layerIdx}_k`
      );
      const v3d = await recordTensorView(
        tape,
        v2d,
        [seqLen, numKVHeads, headDim],
        `layer_${layerIdx}_v`
      );
      const useLocalRope = isSlidingLayerType(modelConfig.layerTypes?.[layerIdx]);
      const layerRopeCos = useLocalRope ? ropeLocalCos : ropeCos;
      const layerRopeSin = useLocalRope ? ropeLocalSin : ropeSin;
      if (!layerRopeCos || !layerRopeSin) {
        throw new Error(`Distill full-graph student missing RoPE tensors on layer ${layerIdx}.`);
      }

      const qRope = await tape.record(
        OpType.ROPE,
        (q, cos, sin) => runRoPE(q, cos, sin, seqLen, { numHeads, headDim, startPos: 0 }),
        [q3d, layerRopeCos, layerRopeSin],
        { seqLen, numHeads, headDim, startPos: 0, ...stopRopeWeights }
      );
      const kRope = await tape.record(
        OpType.ROPE,
        (k, cos, sin) => runRoPE(k, cos, sin, seqLen, { numHeads: numKVHeads, headDim, startPos: 0 }),
        [k3d, layerRopeCos, layerRopeSin],
        { seqLen, numHeads: numKVHeads, headDim, startPos: 0, ...stopRopeWeights }
      );
      await captureStage('attn.q_rope', qRope, layerIdx);
      await captureStage('attn.k_rope', kRope, layerIdx);

      const attention = await tape.record(
        OpType.ATTENTION,
        (q, k, v) => runAttention(q, k, v, null, numHeads, headDim, {
          seqLen,
          kvLen: seqLen,
          numKVHeads,
          causal: true,
          startPos: 0,
          scale: 1 / Math.sqrt(headDim),
        }),
        [qRope, kRope, v3d],
        {
          seqLen,
          numHeads,
          numKVHeads,
          headDim,
          scale: 1 / Math.sqrt(headDim),
          causal: true,
          recomputeForward: true,
        }
      );
      await captureStage('attn.core_out', attention, layerIdx);
      const attention2d = await recordTensorView(
        tape,
        attention,
        [seqLen, numHeads * headDim],
        `layer_${layerIdx}_attn_2d`
      );

      let attentionOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, numHeads * headDim, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [attention2d, layer.oProj],
        {
          M: seqLen,
          N: hiddenSize,
          K: numHeads * headDim,
          transposeB: 'auto',
          ...stopBaseWeight,
        }
      );
      if (layer.lora.o_proj) {
        const delta = await layer.lora.o_proj.forward(attention2d, tape);
        attentionOutput = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
          [attentionOutput, delta],
          { size: seqLen * hiddenSize }
        );
      }
      await captureStage('attn.out', attentionOutput, layerIdx);
      const normalizedAttentionOutput = modelConfig.postAttentionNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [attentionOutput, layer.postAttentionNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : attentionOutput;
      const postAttention = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [normalizedAttentionOutput, hidden],
        { size: seqLen * hiddenSize }
      );
      await captureStage('attn.post_attn', postAttention, layerIdx);

      const ffnInput = modelConfig.preFeedforwardNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [postAttention, layer.preFeedforwardNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : postAttention;
      await captureStage('ffn.in', ffnInput, layerIdx);
      let activated = null;
      if (layer.gateUp) {
        let gateUp = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize * 2, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.gateUp],
          { M: seqLen, N: layerIntermediateSize * 2, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        if (layer.lora.gate_up_proj) {
          const delta = await layer.lora.gate_up_proj.forward(ffnInput, tape);
          gateUp = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize * 2),
            [gateUp, delta],
            { size: seqLen * layerIntermediateSize * 2 }
          );
        }
        activated = await tape.record(
          OpType.SILU_ROWSPLIT,
          (x) => runSiLURowSplit(x, {
            numTokens: seqLen,
            dim: layerIntermediateSize,
            activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
            swigluLimit: hiddenActivation === 'gelu' ? null : swigluLimit,
          }),
          [gateUp],
          {
            numTokens: seqLen,
            dim: layerIntermediateSize,
            activation: hiddenActivation === 'gelu' ? 'gelu' : 'silu',
            swigluLimit: hiddenActivation === 'gelu' ? 0 : swigluLimit,
          }
        );
      } else {
        if (hiddenActivation === 'gelu') {
          throw new Error('Split gate/up training currently requires SiLU activation.');
        }
        let gate = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.gate],
          { M: seqLen, N: layerIntermediateSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        let up = await tape.record(
          OpType.MATMUL,
          (x, w) => runMatmul(x, w, seqLen, layerIntermediateSize, hiddenSize, {
            transposeB: 'auto',
            outputDtype: 'f32',
          }),
          [ffnInput, layer.up],
          { M: seqLen, N: layerIntermediateSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
        );
        if (layer.lora.gate_proj) {
          const delta = await layer.lora.gate_proj.forward(ffnInput, tape);
          gate = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize),
            [gate, delta],
            { size: seqLen * layerIntermediateSize }
          );
        }
        if (layer.lora.up_proj) {
          const delta = await layer.lora.up_proj.forward(ffnInput, tape);
          up = await tape.record(
            OpType.RESIDUAL_ADD,
            (a, b) => runResidualAdd(a, b, seqLen * layerIntermediateSize),
            [up, delta],
            { size: seqLen * layerIntermediateSize }
          );
        }
        activated = await tape.record(
          OpType.SILU_GATED,
          (gateInput, upInput) => runSiLU(upInput, {
            size: seqLen * layerIntermediateSize,
            gate: gateInput,
            inputActivation: 'identity',
            swigluLimit,
          }),
          [gate, up],
          { count: seqLen * layerIntermediateSize, swigluLimit }
        );
      }
      await captureStage('ffn.act', activated, layerIdx);
      let ffnOutput = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, hiddenSize, layerIntermediateSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [activated, layer.down],
        { M: seqLen, N: hiddenSize, K: layerIntermediateSize, transposeB: 'auto', ...stopBaseWeight }
      );
      if (layer.lora.down_proj) {
        const delta = await layer.lora.down_proj.forward(activated, tape);
        ffnOutput = await tape.record(
          OpType.RESIDUAL_ADD,
          (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
          [ffnOutput, delta],
          { size: seqLen * hiddenSize }
        );
      }
      await captureStage('ffn.out', ffnOutput, layerIdx);
      const normalizedFfnOutput = modelConfig.postFeedforwardNorm === true
        ? await tape.record(
          OpType.RMSNORM,
          (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
            batchSize: seqLen,
            hiddenSize,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
          }),
          [ffnOutput, layer.postFeedforwardNorm],
          {
            numTokens: seqLen,
            hiddenSize,
            eps: rmsNormEps,
            rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
            ...stopBaseWeight,
          }
        )
        : ffnOutput;
      hidden = await tape.record(
        OpType.RESIDUAL_ADD,
        (a, b) => runResidualAdd(a, b, seqLen * hiddenSize),
        [normalizedFfnOutput, postAttention],
        { size: seqLen * hiddenSize }
      );
      await captureStage('layer.out', hidden, layerIdx);
    }

    await captureStage('final_norm.pre', hidden);
    const finalHidden = await tape.record(
      OpType.RMSNORM,
      (x, gamma) => runRMSNorm(x, gamma, rmsNormEps, {
        batchSize: seqLen,
        hiddenSize,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
      }),
      [hidden, finalNormWeight],
      {
        numTokens: seqLen,
        hiddenSize,
        eps: rmsNormEps,
        rmsNormWeightOffset: modelConfig.rmsNormWeightOffset === true,
        ...stopBaseWeight,
      }
    );
    await captureStage('final_norm.out', finalHidden);
    if (forwardOptions.logitsMode === 'all') {
      const logits = await tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, seqLen, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [finalHidden, lmHeadWeight],
        { M: seqLen, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
      await captureStage('logits.out', logits);
      return logits;
    }
    const lastHidden = await tape.record(
      OpType.ROW_SLICE,
      (x) => createRowSliceTensor(x, seqLen, hiddenSize, seqLen - 1, 'distill_last_hidden'),
      [finalHidden],
      { rows: seqLen, cols: hiddenSize, rowIndex: seqLen - 1 }
    );
    const logits = await tape.record(
      OpType.MATMUL,
      (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
        transposeB: 'auto',
        outputDtype: 'f32',
      }),
      [lastHidden, lmHeadWeight],
      { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
    );
    await captureStage('logits.out', logits);
    return logits;
  }

  async function runTransformerPrompt(prompt, tape, forwardOptions = {}) {
    const { tokenTensor, seqLen } = await buildPromptTokens(prompt);
    return runTransformerTokenTensor(tokenTensor, seqLen, tape, forwardOptions);
  }

  function collectLoraTensorEntries() {
    const entries = [];
    for (let layerIdx = 0; layerIdx < layers.length; layerIdx += 1) {
      const layer = layers[layerIdx];
      for (const moduleName of Object.keys(layer.lora).sort((left, right) => left.localeCompare(right))) {
        const adapter = layer.lora[moduleName];
        entries.push(
          { name: `layers.${layerIdx}.${moduleName}.lora_a`, tensor: adapter.A },
          { name: `layers.${layerIdx}.${moduleName}.lora_b`, tensor: adapter.B }
        );
      }
    }
    return entries;
  }

  const model = {
    async forward(inputTensor, tape) {
      if (Array.isArray(inputTensor?.shape) && inputTensor.shape.length === 1) {
        return runTransformerTokenTensor(inputTensor, inputTensor.shape[0], tape, { logitsMode: 'all' });
      }
      return tape.record(
        OpType.MATMUL,
        (x, w) => runMatmul(x, w, 1, vocabSize, hiddenSize, {
          transposeB: 'auto',
          outputDtype: 'f32',
        }),
        [inputTensor, lmHeadWeight],
        { M: 1, N: vocabSize, K: hiddenSize, transposeB: 'auto', ...stopBaseWeight }
      );
    },
    async forwardCausalLm(batch, tape) {
      const inputTensor = batch?.input || null;
      const seqLen = Array.isArray(inputTensor?.shape) && inputTensor.shape.length === 1
        ? inputTensor.shape[0]
        : 0;
      if (!Number.isInteger(seqLen) || seqLen < 1) {
        throw new Error('Transformer LoRA causal-LM batch requires input token tensor shape [seqLen].');
      }
      const logits = await runTransformerTokenTensor(inputTensor, seqLen, tape, { logitsMode: 'all' });
      return { logits };
    },
    async forwardDistill(batch, tape, forwardOptions = {}) {
      const requestedPhase = String(forwardOptions?.phase || 'anchor').trim();
      const phase = requestedPhase === 'positive'
        ? 'positive'
        : (requestedPhase === 'negative' ? 'negative' : 'anchor');
      const prompts = resolvePhasePrompts(batch, phase);
      if (prompts.length !== 1) {
        throw new Error(
          `Distill full-graph student currently requires batchSize=1, got ${prompts.length}.`
        );
      }
      const logits = await runTransformerPrompt(prompts[0], tape, forwardOptions);
      return { logits };
    },
    cleanupDistillStep() {
      for (const tensor of temporaryInputs) {
        releaseTensor(tensor);
      }
      temporaryInputs.clear();
    },
    loraParams() {
      return loraParams.length > 0 ? loraParams : decoderParams;
    },
    loraTensorEntries() {
      return collectLoraTensorEntries();
    },
    paramGroups() {
      if (loraParams.length > 0) {
        return {
          encoder: [],
          prior: [],
          decoder: [],
          base: baseParams,
          lora: loraParams,
        };
      }
      return {
        encoder: encoderParams,
        prior: [],
        decoder: decoderParams,
        base: baseParams,
        lora: loraParams,
      };
    },
  };

  return {
    config,
    model,
    outputDim: vocabSize,
    embeddingDim: hiddenSize,
    cleanup() {
      model.cleanupDistillStep();
      disposeTransformerLoraAdapters(layers);
      for (const tensor of ownedTrainables) {
        releaseTensor(tensor);
      }
      ownedTrainables.clear();
    },
  };
}

export async function createDistillStudentRuntimeModelFixture(overrides = {}, options = {}) {
  const distillRuntime = options.distillRuntime && typeof options.distillRuntime === 'object'
    ? options.distillRuntime
    : null;
  const graphMode = normalizeDistillStudentGraphMode(
    options.studentGraphMode
    ?? distillRuntime?.studentGraphMode
    ?? overrides?.training?.distill?.studentGraphMode
  );
  if (graphMode === DISTILL_STUDENT_GRAPH_PROJECTION) {
    return createDistillStudentProjectionModelFixture(overrides, options);
  }
  return createDistillStudentTransformerModelFixture(overrides, options);
}
