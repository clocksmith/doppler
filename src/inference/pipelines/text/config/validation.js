import { log } from '../../../../debug/index.js';
import { mergeConfig, dumpConfigSources } from '../../../../config/merge.js';
import { validateModelOverrides } from '../../../../config/param-validator.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';
import { appendHeterogeneousAttentionValidation, resolveHeterogeneousAttentionContract } from '../attention/heterogeneous-contract.js';
import { resolvePostNormContract } from '../normalization-contract.js';
import { resolveEmbeddingNormalization } from '../embedding-contract.js';
import {
  PER_LAYER_INPUT_MATERIALIZATION_MODES,
  PER_LAYER_INPUT_ROW_CACHE_MODES,
  PER_LAYER_INPUT_PREFETCH_MODES,
  PER_LAYER_INPUT_GPU_UPLOAD_MODES,
  PER_LAYER_INPUT_HOT_CACHE_MODES,
  PREFILL_CHUNK_SUBMIT_MODES,
} from '../../../../config/schema/execution-v1.schema.js';

export function normalizeFfnTensorShape(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
  if (rows <= 0 || cols <= 0) return null;
  return [Math.trunc(rows), Math.trunc(cols)];
}

export function getDenseFfnTensorShape(tensors, names) {
  if (!tensors || typeof tensors !== 'object') return null;
  for (const name of names) {
    const shape = normalizeFfnTensorShape(tensors[name]?.shape);
    if (shape) return shape;
  }
  return null;
}

export function assertDenseFfnTensorShape(modelId, layerIdx, label, actualShape, expectedShape) {
  if (!actualShape) return;
  if (actualShape[0] === expectedShape[0] && actualShape[1] === expectedShape[1]) {
    return;
  }
  throw new Error(
    `Manifest "${modelId}" layer ${layerIdx} ${label} shape [${actualShape.join(', ')}] does not match ` +
    `the resolved FFN contract [${expectedShape.join(', ')}]. Re-convert the model so manifest inference ` +
    'and FFN tensor shapes agree.'
  );
}

export function validateLayerIntermediateSizesAgainstManifest(manifest, hiddenSize, layerIntermediateSizes, modelId) {
  const tensors = manifest?.tensors;
  if (!tensors || typeof tensors !== 'object' || !Array.isArray(layerIntermediateSizes)) {
    return;
  }

  for (let layerIdx = 0; layerIdx < layerIntermediateSizes.length; layerIdx += 1) {
    const intermediateSize = Number(layerIntermediateSizes[layerIdx]);
    if (!Number.isFinite(intermediateSize) || intermediateSize <= 0) {
      throw new Error(
        `Manifest "${modelId}" resolved an invalid FFN intermediate size (${String(layerIntermediateSizes[layerIdx])}) ` +
        `for layer ${layerIdx}.`
      );
    }

    const expectedIntermediateSize = Math.trunc(intermediateSize);
    const languagePrefix = `model.language_model.layers.${layerIdx}`;
    const genericPrefix = `model.layers.${layerIdx}`;
    const gateShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.gate_proj.weight`,
      `${genericPrefix}.mlp.gate_proj.weight`,
      `${languagePrefix}.ffn.gate_proj.weight`,
      `${languagePrefix}.ffn_gate.weight`,
      `${languagePrefix}.feed_forward.w1.weight`,
      `${genericPrefix}.feed_forward.w1.weight`,
      `layers.${layerIdx}.feed_forward.w1.weight`,
    ]);
    const upShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.up_proj.weight`,
      `${genericPrefix}.mlp.up_proj.weight`,
      `${languagePrefix}.ffn.up_proj.weight`,
      `${languagePrefix}.ffn_up.weight`,
      `${languagePrefix}.feed_forward.w3.weight`,
      `${genericPrefix}.feed_forward.w3.weight`,
      `layers.${layerIdx}.feed_forward.w3.weight`,
    ]);
    const downShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.down_proj.weight`,
      `${genericPrefix}.mlp.down_proj.weight`,
      `${languagePrefix}.ffn.down_proj.weight`,
      `${languagePrefix}.ffn_down.weight`,
      `${languagePrefix}.feed_forward.w2.weight`,
      `${genericPrefix}.feed_forward.w2.weight`,
      `layers.${layerIdx}.feed_forward.w2.weight`,
    ]);
    const gateUpShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.gate_up_proj.weight`,
      `${genericPrefix}.mlp.gate_up_proj.weight`,
      `${languagePrefix}.ffn.gate_up_proj.weight`,
      `${languagePrefix}.ffn_gate_up.weight`,
      `${languagePrefix}.feed_forward.w1_w3.weight`,
      `${genericPrefix}.feed_forward.w1_w3.weight`,
      `layers.${layerIdx}.feed_forward.w1_w3.weight`,
    ]);

    assertDenseFfnTensorShape(
      modelId,
      layerIdx,
      'gate weight',
      gateShape,
      [expectedIntermediateSize, hiddenSize]
    );
    assertDenseFfnTensorShape(
      modelId,
      layerIdx,
      'up weight',
      upShape,
      [expectedIntermediateSize, hiddenSize]
    );
    assertDenseFfnTensorShape(
      modelId,
      layerIdx,
      'down weight',
      downShape,
      [hiddenSize, expectedIntermediateSize]
    );
    assertDenseFfnTensorShape(
      modelId,
      layerIdx,
      'gate_up weight',
      gateUpShape,
      [expectedIntermediateSize * 2, hiddenSize]
    );
  }
}

export function normalizeUnsupportedText(value) {
  if (typeof value !== 'string') return '';
  return value.trim();
}

export function assertSupportedManifestInference(manifest) {
  const modelId = manifest?.modelId ?? 'unknown';
  const unsupported = manifest?.inference?.unsupported;
  if (unsupported == null) {
    return;
  }
  if (typeof unsupported !== 'object' || Array.isArray(unsupported)) {
    throw new Error(
      `Manifest "${modelId}" has invalid inference.unsupported. ` +
      'Expected null or an object with code, message, and recommendation fields.'
    );
  }
  const code = normalizeUnsupportedText(unsupported.code) || 'unsupported-manifest-contract';
  const message = normalizeUnsupportedText(unsupported.message);
  const recommendation = normalizeUnsupportedText(unsupported.recommendation);
  throw new Error(
    `Manifest "${modelId}" is not supported by Doppler runtime (${code}).` +
    (message ? ` ${message}` : '') +
    (recommendation ? ` ${recommendation}` : '')
  );
}

export function validateRequiredInferenceFields(inf, modelId) {
  inf = inf ?? {};
  inf.attention = inf.attention ?? {};
  inf.normalization = inf.normalization ?? {};
  inf.ffn = inf.ffn ?? {};
  inf.rope = inf.rope ?? {};
  inf.output = inf.output ?? {};
  inf.layerPattern = inf.layerPattern ?? {};
  inf.chatTemplate = inf.chatTemplate ?? {};
  const errors = [];

  // Attention fields - non-nullable required
  if (inf.attention.queryPreAttnScalar == null) {
    errors.push('attention.queryPreAttnScalar is required');
  }
  appendHeterogeneousAttentionValidation(errors, inf);
  if (inf.attention.queryKeyNorm == null) {
    errors.push('attention.queryKeyNorm is required');
  }
  if (
    inf.attention.queryKeyNormType !== undefined
    && !['rmsnorm', 'layernorm'].includes(inf.attention.queryKeyNormType)
  ) {
    errors.push('attention.queryKeyNormType must be "rmsnorm" or "layernorm"');
  }
  if (
    inf.attention.queryKeyNormAxis !== undefined
    && !['head', 'projection'].includes(inf.attention.queryKeyNormAxis)
  ) {
    errors.push('attention.queryKeyNormAxis must be "head" or "projection"');
  }
  if (
    inf.attention.queryKeyNormLayers !== undefined
    && inf.attention.queryKeyNormLayers !== null
  ) {
    if (!Array.isArray(inf.attention.queryKeyNormLayers)) {
      errors.push('attention.queryKeyNormLayers must be null or an array of layer indices');
    } else {
      for (const layerIdx of inf.attention.queryKeyNormLayers) {
        if (!Number.isInteger(layerIdx) || layerIdx < 0) {
          errors.push('attention.queryKeyNormLayers must contain only non-negative integer layer indices');
          break;
        }
      }
    }
  }
  if (
    inf.attention.queryKeyNormWeightLayers !== undefined
    && inf.attention.queryKeyNormWeightLayers !== null
  ) {
    if (!Array.isArray(inf.attention.queryKeyNormWeightLayers)) {
      errors.push('attention.queryKeyNormWeightLayers must be null or an array of layer indices');
    } else {
      for (const layerIdx of inf.attention.queryKeyNormWeightLayers) {
        if (!Number.isInteger(layerIdx) || layerIdx < 0) {
          errors.push('attention.queryKeyNormWeightLayers must contain only non-negative integer layer indices');
          break;
        }
      }
    }
  }
  if (inf.attention.valueNorm == null) {
    errors.push('attention.valueNorm is required');
  }
  if (inf.attention.attentionBias == null) {
    errors.push('attention.attentionBias is required');
  }
  if (inf.attention.causal == null) {
    errors.push('attention.causal is required');
  }
  // Attention fields - nullable required (undefined = missing, null = disabled)
  if (inf.attention.slidingWindow === undefined) {
    errors.push('attention.slidingWindow must be explicitly set (null for no sliding window, or number)');
  }
  if (inf.attention.attnLogitSoftcapping === undefined) {
    errors.push('attention.attnLogitSoftcapping must be explicitly set (null for no softcapping, or number)');
  }

  // Normalization fields
  if (
    inf.normalization.type !== undefined
    && !['rmsnorm', 'layernorm'].includes(inf.normalization.type)
  ) {
    errors.push('normalization.type must be rmsnorm or layernorm');
  }
  if (inf.normalization.rmsNormWeightOffset == null) {
    errors.push('normalization.rmsNormWeightOffset is required');
  }
  if (inf.normalization.rmsNormEps == null) {
    errors.push('normalization.rmsNormEps is required');
  }
  if (inf.normalization.postAttentionNorm == null) {
    errors.push('normalization.postAttentionNorm is required');
  }
  if (inf.normalization.preFeedforwardNorm == null) {
    errors.push('normalization.preFeedforwardNorm is required');
  }
  if (inf.normalization.postFeedforwardNorm == null) {
    errors.push('normalization.postFeedforwardNorm is required');
  }
  if (
    inf.normalization.finalNormBiasTensor !== undefined
    &&
    inf.normalization.finalNormBiasTensor !== null
    && (
      typeof inf.normalization.finalNormBiasTensor !== 'string'
      || inf.normalization.finalNormBiasTensor.trim() === ''
    )
  ) {
    errors.push('normalization.finalNormBiasTensor must be null or a non-empty string');
  }
  if (
    (inf.normalization.type ?? 'rmsnorm') === 'rmsnorm'
    && (inf.normalization.finalNormBiasTensor ?? null) !== null
  ) {
    errors.push('normalization.finalNormBiasTensor requires normalization.type="layernorm"');
  }

  // FFN fields
  if (inf.ffn.activation == null) {
    errors.push('ffn.activation is required');
  }
  if (inf.ffn.gatedActivation == null) {
    errors.push('ffn.gatedActivation is required');
  }
  if (inf.ffn.branchMode == null) {
    errors.push('ffn.branchMode is required');
  } else {
    const normalizedBranchMode = typeof inf.ffn.branchMode === 'string'
      ? inf.ffn.branchMode.trim().toLowerCase()
      : '';
    const supportedBranchModes = new Set(['auto', 'dense', 'moe', 'dense_plus_moe']);
    if (!supportedBranchModes.has(normalizedBranchMode)) {
      errors.push('ffn.branchMode must be one of: auto, dense, moe, dense_plus_moe');
    }
  }
  if (inf.ffn.useDoubleWideMlp == null) {
    errors.push('ffn.useDoubleWideMlp is required');
  }
  if (inf.ffn.swigluLimit === undefined) {
    errors.push('ffn.swigluLimit must be explicitly set (null for no limit, or number)');
  } else {
    const limit = inf.ffn.swigluLimit;
    if (limit !== null && (typeof limit !== 'number' || Number.isNaN(limit) || limit <= 0)) {
      errors.push('ffn.swigluLimit must be a positive number or null');
    }
  }
  // RoPE fields - non-nullable required
  if (inf.rope.ropeTheta == null) {
    errors.push('rope.ropeTheta is required');
  }
  if (inf.rope.ropeScalingFactor == null) {
    errors.push('rope.ropeScalingFactor is required (use 1.0 for no scaling)');
  }
  // RoPE fields - nullable required (undefined = missing, null = disabled)
  if (inf.rope.ropeScalingType === undefined) {
    errors.push('rope.ropeScalingType must be explicitly set (null for no scaling, or scaling type string)');
  }
  if (inf.rope.ropeLocalTheta === undefined) {
    errors.push('rope.ropeLocalTheta must be explicitly set (null for no local theta, or number)');
  }
  if (
    inf.rope.ropeInterleaved !== undefined
    && inf.rope.ropeInterleaved != null
    && typeof inf.rope.ropeInterleaved !== 'boolean'
  ) {
    errors.push('rope.ropeInterleaved must be boolean when provided');
  }
  if (inf.rope.mropeInterleaved == null) {
    errors.push('rope.mropeInterleaved is required');
  }
  if (inf.rope.mropeSection === undefined) {
    errors.push('rope.mropeSection must be explicitly set (null when unused, or an array of positive integers)');
  }
  if (inf.rope.partialRotaryFactor === undefined) {
    errors.push('rope.partialRotaryFactor must be explicitly set (null when unused, or a number in (0, 1])');
  } else {
    const factor = inf.rope.partialRotaryFactor;
    if (factor !== null && (typeof factor !== 'number' || Number.isNaN(factor) || factor <= 0 || factor > 1)) {
      errors.push('rope.partialRotaryFactor must be a number in (0, 1] or null');
    }
  }
  if (inf.rope.ropeLocalPartialRotaryFactor === undefined) {
    errors.push('rope.ropeLocalPartialRotaryFactor must be explicitly set (null when unused, or a number in (0, 1])');
  } else {
    const factor = inf.rope.ropeLocalPartialRotaryFactor;
    if (factor !== null && (typeof factor !== 'number' || Number.isNaN(factor) || factor <= 0 || factor > 1)) {
      errors.push('rope.ropeLocalPartialRotaryFactor must be a number in (0, 1] or null');
    }
  }
  if (inf.rope.ropeFrequencyBaseDim === undefined) {
    errors.push('rope.ropeFrequencyBaseDim must be explicitly set (null when using rotary dim, or a positive even integer)');
  } else {
    const dim = inf.rope.ropeFrequencyBaseDim;
    if (dim !== null && (typeof dim !== 'number' || Number.isNaN(dim) || dim <= 0 || (Math.trunc(dim) % 2) !== 0)) {
      errors.push('rope.ropeFrequencyBaseDim must be a positive even integer or null');
    }
  }
  if (inf.rope.ropeLocalFrequencyBaseDim === undefined) {
    errors.push('rope.ropeLocalFrequencyBaseDim must be explicitly set (null when using local rotary dim, or a positive even integer)');
  } else {
    const dim = inf.rope.ropeLocalFrequencyBaseDim;
    if (dim !== null && (typeof dim !== 'number' || Number.isNaN(dim) || dim <= 0 || (Math.trunc(dim) % 2) !== 0)) {
      errors.push('rope.ropeLocalFrequencyBaseDim must be a positive even integer or null');
    }
  }
  // Output fields - non-nullable required
  if (inf.output.tieWordEmbeddings == null) {
    errors.push('output.tieWordEmbeddings is required');
  }
  if (inf.output.scaleEmbeddings == null) {
    errors.push('output.scaleEmbeddings is required');
  }
  if (inf.output.embeddingScale === undefined) {
    errors.push('output.embeddingScale must be explicitly set (null for scaleEmbeddings semantics, or number)');
  } else {
    const embeddingScale = inf.output.embeddingScale;
    if (embeddingScale !== null && (typeof embeddingScale !== 'number' || !Number.isFinite(embeddingScale) || embeddingScale <= 0)) {
      errors.push('output.embeddingScale must be a positive finite number or null');
    }
    if (embeddingScale !== null && inf.output.scaleEmbeddings === true) {
      errors.push('output.embeddingScale cannot be set when output.scaleEmbeddings is true');
    }
  }
  if (inf.output.logitInputScale == null) {
    errors.push('output.logitInputScale is required');
  } else if (
    typeof inf.output.logitInputScale !== 'number'
    || !Number.isFinite(inf.output.logitInputScale)
    || inf.output.logitInputScale <= 0
  ) {
    errors.push('output.logitInputScale must be a positive finite number');
  }
  if (inf.output.logitOutputScale !== undefined && (typeof inf.output.logitOutputScale !== 'number' || !Number.isFinite(inf.output.logitOutputScale) || inf.output.logitOutputScale <= 0)) errors.push('output.logitOutputScale must be a positive finite number');
  if (inf.output.embeddingTranspose == null) {
    errors.push('output.embeddingTranspose is required');
  }
  // Output fields - nullable required (undefined = missing, null = disabled)
  if (inf.output.finalLogitSoftcapping === undefined) {
    errors.push('output.finalLogitSoftcapping must be explicitly set (null for no softcapping, or number)');
  }
  if (inf.output.embeddingVocabSize === undefined) {
    errors.push('output.embeddingVocabSize must be explicitly set (null to use architecture.vocabSize, or number)');
  }
  if (inf.output.embeddingPostprocessor === undefined) {
    errors.push('output.embeddingPostprocessor must be explicitly set (null when unused, or an object)');
  } else if (inf.output.embeddingPostprocessor !== null) {
    const postprocessor = inf.output.embeddingPostprocessor;
    if (!postprocessor || typeof postprocessor !== 'object' || Array.isArray(postprocessor)) {
      errors.push('output.embeddingPostprocessor must be null or an object');
    } else {
      if (postprocessor.poolingMode !== 'mean' && postprocessor.poolingMode !== 'last') {
        errors.push('output.embeddingPostprocessor.poolingMode must be "mean" or "last"');
      }
      if (typeof postprocessor.includePrompt !== 'boolean') {
        errors.push('output.embeddingPostprocessor.includePrompt is required');
      }
      if (!Array.isArray(postprocessor.projections)) {
        errors.push('output.embeddingPostprocessor.projections must be an array');
      } else {
        for (let i = 0; i < postprocessor.projections.length; i++) {
          const projection = postprocessor.projections[i];
          const prefix = `output.embeddingPostprocessor.projections[${i}]`;
          if (typeof projection?.weightTensor !== 'string' || projection.weightTensor.trim() === '') {
            errors.push(`${prefix}.weightTensor is required`);
          }
          if (projection?.biasTensor === undefined) {
            errors.push(`${prefix}.biasTensor must be explicitly set (null when unused, or tensor name)`);
          } else if (projection.biasTensor !== null && (typeof projection.biasTensor !== 'string' || projection.biasTensor.trim() === '')) {
            errors.push(`${prefix}.biasTensor must be null or a non-empty string`);
          }
          if (!Number.isFinite(projection?.inputSize) || projection.inputSize <= 0) {
            errors.push(`${prefix}.inputSize must be a positive number`);
          }
          if (!Number.isFinite(projection?.outputSize) || projection.outputSize <= 0) {
            errors.push(`${prefix}.outputSize must be a positive number`);
          }
          if (projection?.activation !== 'identity') {
            errors.push(`${prefix}.activation must be "identity"`);
          }
        }
      }
      if (postprocessor.normalize === undefined) {
        errors.push('output.embeddingPostprocessor.normalize must be explicitly set (null when unused, or "l2")');
      } else if (postprocessor.normalize !== null && postprocessor.normalize !== 'l2') {
        errors.push('output.embeddingPostprocessor.normalize must be null or "l2"');
      }
    }
  }
  if (
    inf.output.lmHeadBiasTensor !== undefined
    && inf.output.lmHeadBiasTensor !== null
    && (typeof inf.output.lmHeadBiasTensor !== 'string' || inf.output.lmHeadBiasTensor.trim() === '')
  ) {
    errors.push('output.lmHeadBiasTensor must be null or a non-empty tensor name');
  }

  if (inf.supportsSequence !== undefined && typeof inf.supportsSequence !== 'boolean') {
    errors.push('supportsSequence must be boolean when provided');
  }
  if (inf.sequence !== undefined && inf.sequence !== null) {
    const sequence = inf.sequence;
    if (!sequence || typeof sequence !== 'object' || Array.isArray(sequence)) {
      errors.push('sequence must be null or an object');
    } else {
      if (sequence.alphabet !== 'amino_acid' && sequence.alphabet !== 'nucleotide') {
        errors.push('sequence.alphabet must be "amino_acid" or "nucleotide"');
      }
      if (typeof sequence.tokenEmbeddings !== 'boolean') {
        errors.push('sequence.tokenEmbeddings must be boolean');
      }
      if (typeof sequence.logits !== 'boolean') {
        errors.push('sequence.logits must be boolean');
      }
      if (sequence.pooledEmbedding !== null) {
        const pooling = sequence.pooledEmbedding;
        if (!pooling || typeof pooling !== 'object' || Array.isArray(pooling)) {
          errors.push('sequence.pooledEmbedding must be null or an object');
        } else {
          if (pooling.mode !== 'mean' && pooling.mode !== 'last') {
            errors.push('sequence.pooledEmbedding.mode must be "mean" or "last"');
          }
          if (!Array.isArray(pooling.excludeTokenIds) || pooling.excludeTokenIds.some(
            (tokenId) => !Number.isInteger(tokenId) || tokenId < 0
          )) {
            errors.push('sequence.pooledEmbedding.excludeTokenIds must be an array of non-negative integers');
          }
        }
      }
    }
  }
  if (inf.supportsSequence === true && (inf.sequence == null || typeof inf.sequence !== 'object')) {
    errors.push('sequence is required when supportsSequence=true');
  }
  if (inf.supportsSequence !== true && inf.sequence != null) {
    errors.push('supportsSequence must be true when sequence is configured');
  }

  // Layer pattern fields
  if (inf.layerPattern?.type == null) {
    errors.push('layerPattern.type is required');
  }
  if (inf.layerPattern?.globalPattern === undefined) {
    errors.push('layerPattern.globalPattern must be explicitly set (null if not applicable)');
  }
  if (inf.layerPattern?.period === undefined) {
    errors.push('layerPattern.period must be explicitly set (null if not applicable)');
  }
  if (inf.layerPattern?.offset === undefined) {
    errors.push('layerPattern.offset must be explicitly set (null if not applicable)');
  }
  if (inf.layerPattern?.residualBranchScale == null) {
    errors.push('layerPattern.residualBranchScale is required');
  } else if (
    typeof inf.layerPattern.residualBranchScale !== 'number'
    || !Number.isFinite(inf.layerPattern.residualBranchScale)
    || inf.layerPattern.residualBranchScale <= 0
  ) {
    errors.push('layerPattern.residualBranchScale must be a positive finite number');
  }
  if (inf.layerPattern?.type === 'custom' && inf.layerPattern?.layerTypes === undefined) {
    errors.push('layerPattern.layerTypes must be explicitly set for custom patterns');
  }

  // Chat template fields
  if (inf.chatTemplate?.type === undefined) {
    errors.push('chatTemplate.type must be explicitly set (null for no template)');
  }
  if (inf.chatTemplate?.enabled == null) {
    errors.push('chatTemplate.enabled is required');
  }

  // RoPE YARN fields
  if (inf.rope.yarnBetaFast === undefined) {
    errors.push('rope.yarnBetaFast must be explicitly set (null if not YARN)');
  }
  if (inf.rope.yarnBetaSlow === undefined) {
    errors.push('rope.yarnBetaSlow must be explicitly set (null if not YARN)');
  }
  if (inf.rope.yarnOriginalMaxPos === undefined) {
    errors.push('rope.yarnOriginalMaxPos must be explicitly set (null if not YARN)');
  }
  if (inf.rope.longropeShortFactor === undefined) {
    errors.push('rope.longropeShortFactor must be explicitly set (null if not LongRoPE)');
  }
  if (inf.rope.longropeLongFactor === undefined) {
    errors.push('rope.longropeLongFactor must be explicitly set (null if not LongRoPE)');
  }
  if (inf.rope.longropeOriginalMaxPos === undefined) {
    errors.push('rope.longropeOriginalMaxPos must be explicitly set (null if not LongRoPE)');
  }
  if (inf.rope.ropeScalingType === 'longrope') {
    if (!Array.isArray(inf.rope.longropeShortFactor) || inf.rope.longropeShortFactor.length === 0) {
      errors.push('rope.longropeShortFactor must be a non-empty number array for LongRoPE');
    }
    if (!Array.isArray(inf.rope.longropeLongFactor) || inf.rope.longropeLongFactor.length === 0) {
      errors.push('rope.longropeLongFactor must be a non-empty number array for LongRoPE');
    }
    if (
      typeof inf.rope.longropeOriginalMaxPos !== 'number'
      || !Number.isFinite(inf.rope.longropeOriginalMaxPos)
      || inf.rope.longropeOriginalMaxPos <= 0
    ) {
      errors.push('rope.longropeOriginalMaxPos must be a positive finite number for LongRoPE');
    }
  } else if (
    inf.rope.longropeShortFactor !== null
    || inf.rope.longropeLongFactor !== null
    || inf.rope.longropeOriginalMaxPos !== null
  ) {
    errors.push('rope LongRoPE fields must be null unless rope.ropeScalingType is "longrope"');
  }
  if (inf.rope.ropeLocalYarnBetaFast === undefined) {
    errors.push('rope.ropeLocalYarnBetaFast must be explicitly set (null if not local YARN)');
  }
  if (inf.rope.ropeLocalYarnBetaSlow === undefined) {
    errors.push('rope.ropeLocalYarnBetaSlow must be explicitly set (null if not local YARN)');
  }
  if (inf.rope.ropeLocalYarnOriginalMaxPos === undefined) {
    errors.push('rope.ropeLocalYarnOriginalMaxPos must be explicitly set (null if not local YARN)');
  }

  if (errors.length > 0) {
    throw new Error(
      `Manifest "${modelId}" has incomplete inference config. ` +
      `Missing required fields:\n  - ${errors.join('\n  - ')}\n` +
      `Re-convert the model using the latest converter.`
    );
  }
}
