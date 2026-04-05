import { log } from '../../../debug/index.js';
import { mergeConfig, dumpConfigSources } from '../../../config/merge.js';
import { selectRuleValue } from '../../../rules/rule-registry.js';

const UNSUPPORTED_RUNTIME_MODEL_TYPES = new Set(['mamba', 'rwkv']);

/**
 * Known chat template types that have registered formatters in chat-format.js.
 * Used for load-time validation so unknown types fail early instead of at
 * generation time.
 */
const KNOWN_CHAT_TEMPLATE_TYPES = new Set([
  'gemma',
  'gemma4',
  'llama3',
  'gpt-oss',
  'chatml',
  'qwen',
  'translategemma',
]);

/**
 * Validate that a chatTemplate.type value is either null (disabled) or a known
 * formatter type.
 *
 * @param {string | null} type - The chatTemplate.type from manifest inference.
 * @param {string} modelId - Model identifier for diagnostic messages.
 * @returns {boolean} True if the type is valid or null.
 */
export function validateChatTemplateType(type, modelId) {
  if (type === null || type === undefined) return true;
  if (KNOWN_CHAT_TEMPLATE_TYPES.has(type)) return true;
  throw new Error(
    `Manifest "${modelId}" declares chatTemplate.type="${type}" which is not a known formatter type. ` +
    `Known types: ${[...KNOWN_CHAT_TEMPLATE_TYPES].join(', ')}. Re-convert the model or fix the manifest.`
  );
}

// =============================================================================
// Model Detection Functions
// =============================================================================

function assertSupportedRuntimeModelType(manifest) {
  const modelType = typeof manifest?.modelType === 'string'
    ? manifest.modelType.trim().toLowerCase()
    : '';
  if (!modelType) return;
  if (!UNSUPPORTED_RUNTIME_MODEL_TYPES.has(modelType)) return;

  const modelId = manifest?.modelId ?? 'unknown';
  throw new Error(
    `Manifest "${modelId}" declares modelType "${modelType}", but that runtime family is not implemented yet.`
  );
}

function resolveRotaryDim(headDim, partialRotaryFactor, modelId) {
  if (partialRotaryFactor == null) {
    return headDim;
  }
  if (typeof partialRotaryFactor !== 'number' || Number.isNaN(partialRotaryFactor)) {
    throw new Error(`Manifest "${modelId}" has invalid rope.partialRotaryFactor.`);
  }
  if (partialRotaryFactor <= 0 || partialRotaryFactor > 1) {
    throw new Error(
      `Manifest "${modelId}" requires 0 < rope.partialRotaryFactor <= 1; got ${partialRotaryFactor}.`
    );
  }
  const rotaryDim = Math.trunc(headDim * partialRotaryFactor);
  if (rotaryDim <= 0 || (rotaryDim % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" resolves rope rotary dim ${rotaryDim} from headDim=${headDim} ` +
      `and partialRotaryFactor=${partialRotaryFactor}, but rotary dim must be a positive even integer.`
    );
  }
  return rotaryDim;
}

function resolveFrequencyBaseDim(headDim, rotaryDim, frequencyBaseDim, modelId, fieldName) {
  if (frequencyBaseDim == null) {
    return rotaryDim;
  }
  if (typeof frequencyBaseDim !== 'number' || Number.isNaN(frequencyBaseDim)) {
    throw new Error(`Manifest "${modelId}" has invalid ${fieldName}.`);
  }
  const resolved = Math.trunc(frequencyBaseDim);
  if (resolved <= 0 || (resolved % 2) !== 0) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} to be a positive even integer; got ${frequencyBaseDim}.`
    );
  }
  if (resolved < rotaryDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be >= rotary dim (${rotaryDim}).`
    );
  }
  if (resolved > headDim) {
    throw new Error(
      `Manifest "${modelId}" requires ${fieldName} (${resolved}) to be <= attention head dim (${headDim}).`
    );
  }
  return resolved;
}

export function getStopTokenIds(manifest) {
  const eosTokenId = manifest?.eos_token_id;
  if (Array.isArray(eosTokenId)) return eosTokenId;
  if (typeof eosTokenId === 'number') return [eosTokenId];
  const modelId = manifest?.modelId ?? 'unknown';
  if (eosTokenId == null) {
    throw new Error(
      `Manifest "${modelId}" is missing eos_token_id. Re-convert the model with tokenizer metadata.`
    );
  }
  throw new Error(
    `Manifest "${modelId}" has eos_token_id of unsupported type "${typeof eosTokenId}" (value: ${JSON.stringify(eosTokenId)}). ` +
    'Expected a number or array of numbers. Re-convert the model with tokenizer metadata.'
  );
}

function normalizeFfnTensorShape(value) {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const rows = Number(value[0]);
  const cols = Number(value[1]);
  if (!Number.isFinite(rows) || !Number.isFinite(cols)) return null;
  if (rows <= 0 || cols <= 0) return null;
  return [Math.trunc(rows), Math.trunc(cols)];
}

function isExpertTensorName(name) {
  const lower = String(name || '').toLowerCase();
  return lower.includes('.experts.') || lower.includes('.expert.') || lower.includes('block_sparse_moe');
}

function inferLfm2IntermediateSizeFromManifest(manifest) {
  const tensors = manifest?.tensors;
  if (!tensors || typeof tensors !== 'object') return null;
  const candidates = [];
  for (const [name, entry] of Object.entries(tensors)) {
    if (!name || isExpertTensorName(name)) continue;
    const shape = normalizeFfnTensorShape(entry?.shape);
    if (!shape) continue;
    const lower = name.toLowerCase();
    if (
      lower.endsWith('.feed_forward.w1.weight')
      || lower.endsWith('.feed_forward.w3.weight')
      || lower.endsWith('.ffn_gate.weight')
      || lower.endsWith('.ffn_up.weight')
      || lower.endsWith('.ffn.gate_proj.weight')
      || lower.endsWith('.ffn.up_proj.weight')
      || lower.endsWith('.mlp.gate_proj.weight')
      || lower.endsWith('.mlp.up_proj.weight')
    ) {
      candidates.push(shape[0]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w2.weight')
      || lower.endsWith('.ffn_down.weight')
      || lower.endsWith('.ffn.down_proj.weight')
      || lower.endsWith('.mlp.down_proj.weight')
    ) {
      candidates.push(shape[1]);
      continue;
    }
    if (
      lower.endsWith('.feed_forward.w1_w3.weight')
      || lower.endsWith('.ffn_gate_up.weight')
      || lower.endsWith('.ffn.gate_up_proj.weight')
      || lower.endsWith('.mlp.gate_up_proj.weight')
    ) {
      if (shape[0] % 2 === 0) {
        candidates.push(Math.trunc(shape[0] / 2));
      }
    }
  }
  if (candidates.length === 0) return null;
  const counts = new Map();
  for (const value of candidates) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  const result = [...counts.entries()]
    .sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0] - b[0];
    })[0]?.[0] ?? null;
  if (result != null && (!Number.isInteger(result) || result <= 0)) {
    log.warn(
      'Config',
      `inferLfm2IntermediateSizeFromManifest: inferred intermediateSize ${result} is not a positive integer, discarding`
    );
    return null;
  }
  return result;
}

function resolveIntermediateSizeForRuntime(manifest, inf, arch, modelId) {
  const fromArch = arch?.intermediateSize;
  if (typeof fromArch !== 'number' || !Number.isFinite(fromArch) || fromArch <= 0) {
    return fromArch;
  }
  const normalizedModelId = String(modelId ?? manifest?.modelId ?? '').toLowerCase();
  if (!normalizedModelId.includes('lfm2')) {
    return fromArch;
  }
  const inferred = inferLfm2IntermediateSizeFromManifest(manifest);
  if (inferred == null || inferred === fromArch) {
    return fromArch;
  }
  throw new Error(
    `Manifest "${modelId}" has intermediateSize=${fromArch}, but FFN tensors imply ${inferred}. ` +
    'Re-convert the model so manifest architecture matches the weights.'
  );
}

function buildPerLayerIntermediateSizes({
  baseIntermediateSize,
  numLayers,
  numKvSharedLayers,
  useDoubleWideMlp,
  modelId,
}) {
  if (!Number.isFinite(baseIntermediateSize) || baseIntermediateSize <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.intermediateSize (${String(baseIntermediateSize)}).`
    );
  }
  if (!Number.isFinite(numLayers) || numLayers <= 0) {
    throw new Error(
      `Manifest "${modelId}" has invalid architecture.numLayers (${String(numLayers)}).`
    );
  }

  const resolvedBaseIntermediateSize = Math.trunc(baseIntermediateSize);
  const resolvedNumLayers = Math.trunc(numLayers);
  const intermediateSizes = new Array(resolvedNumLayers).fill(resolvedBaseIntermediateSize);

  if (!useDoubleWideMlp) {
    return intermediateSizes;
  }

  if (!Number.isFinite(numKvSharedLayers) || numKvSharedLayers <= 0 || numKvSharedLayers >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but architecture.numKvSharedLayers=${String(numKvSharedLayers)} ` +
      `must be a positive integer smaller than numLayers=${resolvedNumLayers}.`
    );
  }

  const firstKvSharedLayerIdx = resolvedNumLayers - Math.trunc(numKvSharedLayers);
  if (firstKvSharedLayerIdx <= 0 || firstKvSharedLayerIdx >= resolvedNumLayers) {
    throw new Error(
      `Manifest "${modelId}" enables ffn.useDoubleWideMlp, but the derived first KV-shared layer index ` +
      `(${firstKvSharedLayerIdx}) is invalid for numLayers=${resolvedNumLayers}.`
    );
  }

  const widenedIntermediateSize = resolvedBaseIntermediateSize * 2;
  for (let layerIdx = firstKvSharedLayerIdx; layerIdx < resolvedNumLayers; layerIdx += 1) {
    intermediateSizes[layerIdx] = widenedIntermediateSize;
  }

  return intermediateSizes;
}

function getDenseFfnTensorShape(tensors, names) {
  if (!tensors || typeof tensors !== 'object') return null;
  for (const name of names) {
    const shape = normalizeFfnTensorShape(tensors[name]?.shape);
    if (shape) return shape;
  }
  return null;
}

function assertDenseFfnTensorShape(modelId, layerIdx, label, actualShape, expectedShape) {
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

function validateLayerIntermediateSizesAgainstManifest(manifest, hiddenSize, layerIntermediateSizes, modelId) {
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
      `layers.${layerIdx}.feed_forward.w1.weight`,
    ]);
    const upShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.up_proj.weight`,
      `${genericPrefix}.mlp.up_proj.weight`,
      `${languagePrefix}.ffn.up_proj.weight`,
      `${languagePrefix}.ffn_up.weight`,
      `layers.${layerIdx}.feed_forward.w3.weight`,
    ]);
    const downShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.down_proj.weight`,
      `${genericPrefix}.mlp.down_proj.weight`,
      `${languagePrefix}.ffn.down_proj.weight`,
      `${languagePrefix}.ffn_down.weight`,
      `layers.${layerIdx}.feed_forward.w2.weight`,
    ]);
    const gateUpShape = getDenseFfnTensorShape(tensors, [
      `${languagePrefix}.mlp.gate_up_proj.weight`,
      `${genericPrefix}.mlp.gate_up_proj.weight`,
      `${languagePrefix}.ffn.gate_up_proj.weight`,
      `${languagePrefix}.ffn_gate_up.weight`,
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

export function resolveLayerIntermediateSize(config, layerIdx) {
  const intermediateSizes = Array.isArray(config?.intermediateSizes) ? config.intermediateSizes : null;
  if (Number.isFinite(layerIdx) && intermediateSizes) {
    const resolved = intermediateSizes[Math.trunc(layerIdx)];
    if (Number.isFinite(resolved) && resolved > 0) {
      return Math.trunc(resolved);
    }
  }

  const fallback = Number(config?.intermediateSize);
  if (!Number.isFinite(fallback) || fallback <= 0) {
    throw new Error(`Invalid modelConfig.intermediateSize: ${String(config?.intermediateSize)}`);
  }
  return Math.trunc(fallback);
}

// =============================================================================
// Manifest-First Config Resolution (NEW)
// =============================================================================


export function hasManifestInference(manifest) {
  return 'inference' in manifest && manifest.inference != null;
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
  if (inf.attention.queryKeyNorm == null) {
    errors.push('attention.queryKeyNorm is required');
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

  // FFN fields
  if (inf.ffn.activation == null) {
    errors.push('ffn.activation is required');
  }
  if (inf.ffn.gatedActivation == null) {
    errors.push('ffn.gatedActivation is required');
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

function normalizeLayerTypeTag(value) {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (!normalized) return null;
  if (
    normalized === 'full_attention'
    || normalized === 'global_attention'
    || normalized === 'full'
    || normalized === 'global'
    || normalized === 'attention'
  ) {
    return 'full_attention';
  }
  if (
    normalized === 'sliding_attention'
    || normalized === 'local_attention'
    || normalized === 'local'
    || normalized === 'sliding'
  ) {
    return 'sliding_attention';
  }
  if (
    normalized === 'linear_attention'
    || normalized === 'linear'
    || normalized === 'gated_delta'
    || normalized === 'gated_delta_net'
  ) {
    return 'linear_attention';
  }
  if (
    normalized === 'conv'
    || normalized === 'convolution'
    || normalized === 'liv_conv'
    || normalized === 'liv_convolution'
  ) {
    return 'conv';
  }
  if (normalized === 'moe' || normalized === 'mamba' || normalized === 'rwkv') {
    return normalized;
  }
  return null;
}

function resolveVisionConfig(rawConfig, manifest) {
  const vc = rawConfig?.vision_config ?? manifest?.config?.vision_config;
  if (!vc || typeof vc !== 'object') {
    log.debug(
      'Config',
      `Vision config not present for model "${manifest?.modelId ?? 'unknown'}"; vision pipeline disabled.`
    );
    return null;
  }
  const rawVisionArchitecture = vc.vision_architecture ?? manifest?.visionArchitecture ?? null;
  const rawModelType = typeof vc.model_type === 'string' ? vc.model_type.trim().toLowerCase() : '';
  const visionArchitecture = rawVisionArchitecture
    ?? (rawModelType === 'gemma4_vision' ? 'gemma4' : null);
  const modelId = manifest?.modelId ?? 'unknown';
  const resolveRequiredVisionField = (keys, label) => {
    for (const key of keys) {
      if (vc[key] !== undefined) {
        return vc[key];
      }
    }
    throw new Error(
      `Manifest "${modelId}" is missing vision_config.${label}. ` +
      'Re-convert the model with explicit vision config metadata.'
    );
  };
  const resolveRequiredPositiveInteger = (keys, label) => {
    const value = resolveRequiredVisionField(keys, label);
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid vision_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredPositiveNumber = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
      throw new Error(
        `Manifest "${modelId}" has invalid ${label}=${JSON.stringify(value)}. ` +
        'Expected a positive number.'
      );
    }
    return number;
  };

  if (visionArchitecture === 'gemma4') {
    const hiddenSize = resolveRequiredPositiveInteger(['hidden_size'], 'hidden_size');
    const numHeads = resolveRequiredPositiveInteger(['num_heads', 'num_attention_heads'], 'num_attention_heads');
    const ropeParameters = vc.rope_parameters;
    if (!ropeParameters || typeof ropeParameters !== 'object') {
      throw new Error(
        `Manifest "${modelId}" is missing vision_config.rope_parameters. ` +
        'Re-convert the model with explicit Gemma 4 vision RoPE metadata.'
      );
    }
    const hiddenActivation = String(resolveRequiredVisionField(['hidden_activation'], 'hidden_activation')).trim();
    if (hiddenActivation !== 'gelu' && hiddenActivation !== 'gelu_pytorch_tanh') {
      throw new Error(
        `Manifest "${modelId}" has unsupported Gemma 4 vision hidden_activation="${hiddenActivation}". ` +
        'Supported values: "gelu", "gelu_pytorch_tanh".'
      );
    }
    if (vc.standardize === true) {
      throw new Error(
        `Manifest "${modelId}" enables vision_config.standardize, but Gemma 4 runtime preprocessing does not support it yet.`
      );
    }
    if (vc.use_clipped_linears !== true) {
      throw new Error(
        `Manifest "${modelId}" requires vision_config.use_clipped_linears=true for Gemma 4 vision weights.`
      );
    }

    return {
      depth: resolveRequiredPositiveInteger(['depth', 'num_hidden_layers'], 'num_hidden_layers'),
      hiddenSize,
      intermediateSize: resolveRequiredPositiveInteger(['intermediate_size'], 'intermediate_size'),
      numHeads,
      numKeyValueHeads: resolveRequiredPositiveInteger(['num_key_value_heads'], 'num_key_value_heads'),
      headDim: resolveRequiredPositiveInteger(['head_dim', 'global_head_dim'], 'head_dim'),
      outHiddenSize: vc.out_hidden_size ?? vc.output_proj_dims ?? null,
      patchSize: resolveRequiredPositiveInteger(['patch_size'], 'patch_size'),
      poolingKernelSize: resolveRequiredPositiveInteger(['pooling_kernel_size'], 'pooling_kernel_size'),
      spatialMergeSize: vc.spatial_merge_size ?? null,
      temporalPatchSize: vc.temporal_patch_size ?? null,
      positionEmbeddingSize: resolveRequiredPositiveInteger(['position_embedding_size'], 'position_embedding_size'),
      defaultOutputLength: resolveRequiredPositiveInteger(['default_output_length'], 'default_output_length'),
      ropeTheta: resolveRequiredPositiveNumber(ropeParameters.rope_theta, 'vision_config.rope_parameters.rope_theta'),
      eps: resolveRequiredPositiveNumber(
        resolveRequiredVisionField(['eps', 'rms_norm_eps'], 'rms_norm_eps'),
        'vision_config.rms_norm_eps'
      ),
      hiddenActivation,
      standardize: false,
      useClippedLinears: true,
      deepstackVisualIndexes: [],
      imageTokenId: rawConfig?.image_token_id ?? manifest?.image_token_id ?? null,
      visionArchitecture,
      softTokenBudgetTiers: Array.isArray(vc.soft_token_budget_tiers)
        ? vc.soft_token_budget_tiers.map(Number).filter((n) => Number.isFinite(n) && n > 0)
        : [70, 140, 280, 560, 1120],
    };
  }

  const hiddenSize = vc.hidden_size ?? 1024;
  const numHeads = vc.num_heads ?? vc.num_attention_heads ?? 16;
  return {
    depth: vc.depth ?? vc.num_hidden_layers ?? 24,
    hiddenSize,
    intermediateSize: vc.intermediate_size ?? 4096,
    numHeads,
    numKeyValueHeads: vc.num_key_value_heads ?? numHeads,
    headDim: vc.head_dim ?? vc.global_head_dim ?? Math.floor(hiddenSize / numHeads),
    outHiddenSize: vc.out_hidden_size ?? vc.output_proj_dims ?? hiddenSize,
    patchSize: vc.patch_size ?? 16,
    poolingKernelSize: vc.pooling_kernel_size ?? vc.spatial_merge_size ?? 2,
    spatialMergeSize: vc.spatial_merge_size ?? 2,
    temporalPatchSize: vc.temporal_patch_size ?? 2,
    positionEmbeddingSize: vc.position_embedding_size ?? null,
    defaultOutputLength: vc.default_output_length ?? null,
    ropeTheta: vc.rope_parameters?.rope_theta ?? null,
    eps: vc.eps ?? vc.rms_norm_eps ?? 1e-6,
    hiddenActivation: vc.hidden_activation ?? 'gelu',
    standardize: vc.standardize === true,
    useClippedLinears: vc.use_clipped_linears === true,
    deepstackVisualIndexes: Array.isArray(vc.deepstack_visual_indexes) ? vc.deepstack_visual_indexes : [],
    imageTokenId: rawConfig?.image_token_id ?? manifest?.image_token_id ?? null,
    visionArchitecture,
  };
}

function resolveAudioConfig(rawConfig, manifest) {
  const ac = rawConfig?.audio_config ?? manifest?.config?.audio_config;
  if (!ac || typeof ac !== 'object') {
    log.debug(
      'Config',
      `Audio config not present for model "${manifest?.modelId ?? 'unknown'}"; audio pipeline disabled.`
    );
    return null;
  }
  const modelId = manifest?.modelId ?? 'unknown';
  const resolveRequiredPositiveInteger = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0 || Math.floor(number) !== number) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive integer.'
      );
    }
    return Math.trunc(number);
  };
  const resolveRequiredPositiveNumber = (value, label) => {
    const number = Number(value);
    if (!Number.isFinite(number) || number <= 0) {
      throw new Error(
        `Manifest "${modelId}" has invalid audio_config.${label}=${JSON.stringify(value)}. ` +
        'Expected a positive number.'
      );
    }
    return number;
  };

  const rawModelType = typeof ac.model_type === 'string' ? ac.model_type.trim().toLowerCase() : '';
  const audioArchitecture = rawModelType === 'gemma4_audio' ? 'gemma4' : null;
  if (!audioArchitecture) {
    throw new Error(
      `Manifest "${modelId}" has unsupported audio_config.model_type="${ac.model_type}". ` +
      'Supported: "gemma4_audio".'
    );
  }

  const hiddenSize = resolveRequiredPositiveInteger(ac.hidden_size, 'hidden_size');
  const numAttentionHeads = resolveRequiredPositiveInteger(ac.num_attention_heads, 'num_attention_heads');
  const headDim = Math.trunc(hiddenSize / numAttentionHeads);

  if (!Array.isArray(ac.subsampling_conv_channels) || ac.subsampling_conv_channels.length < 1) {
    throw new Error(
      `Manifest "${modelId}" is missing audio_config.subsampling_conv_channels array.`
    );
  }

  return {
    audioArchitecture,
    depth: resolveRequiredPositiveInteger(ac.num_hidden_layers, 'num_hidden_layers'),
    hiddenSize,
    numAttentionHeads,
    headDim,
    convKernelSize: resolveRequiredPositiveInteger(ac.conv_kernel_size, 'conv_kernel_size'),
    subsamplingConvChannels: ac.subsampling_conv_channels.map(Number),
    outputProjDims: resolveRequiredPositiveInteger(ac.output_proj_dims, 'output_proj_dims'),
    attentionContextLeft: resolveRequiredPositiveInteger(ac.attention_context_left, 'attention_context_left'),
    attentionContextRight: Number(ac.attention_context_right ?? 0),
    attentionChunkSize: resolveRequiredPositiveInteger(ac.attention_chunk_size, 'attention_chunk_size'),
    attentionLogitCap: resolveRequiredPositiveNumber(ac.attention_logit_cap, 'attention_logit_cap'),
    attentionInvalidLogitsValue: Number(ac.attention_invalid_logits_value ?? -1e9),
    residualWeight: resolveRequiredPositiveNumber(ac.residual_weight, 'residual_weight'),
    rmsNormEps: resolveRequiredPositiveNumber(ac.rms_norm_eps ?? 1e-6, 'rms_norm_eps'),
    hiddenAct: String(ac.hidden_act ?? 'silu').trim(),
    useClippedLinears: ac.use_clipped_linears === true,
    audioTokenId: rawConfig?.audio_token_id ?? manifest?.audio_token_id ?? null,
  };
}

function parseCustomLayerTypes(layerTypes, numLayers, modelId) {
  if (!Array.isArray(layerTypes) || layerTypes.length === 0) {
    throw new Error(
      `Manifest "${modelId}" has layerPattern.type='custom' but layerPattern.layerTypes is missing or empty. ` +
      'Re-convert the model to include explicit layer types.'
    );
  }
  if (layerTypes.length !== numLayers) {
    throw new Error(
      `Manifest "${modelId}" has layerPattern.type='custom' with ${layerTypes.length} layer types, ` +
      `expected ${numLayers}. Re-convert the model to preserve full per-layer metadata.`
    );
  }
  return layerTypes.map((layerType, index) => {
    const normalized = normalizeLayerTypeTag(layerType);
    if (!normalized) {
      throw new Error(
        `Manifest "${modelId}" has unknown layerPattern.layerTypes[${index}]="${layerType}". ` +
        'Supported types: conv, full_attention, sliding_attention, linear_attention, moe, mamba, rwkv.'
      );
    }
    return normalized;
  });
}

function parseLinearNormMode(value, sharedFlag = null, modelId = 'unknown') {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === 'shared') return 'shared';
    if (normalized === 'per_head' || normalized === 'per-head' || normalized === 'perhead') {
      return 'per_head';
    }
    throw new Error(
      `Manifest "${modelId}" has unsupported linear_norm_mode="${value}". ` +
      'Supported values: "shared", "per_head".'
    );
  }
  if (typeof sharedFlag === 'boolean') {
    return sharedFlag ? 'shared' : 'per_head';
  }
  return null;
}


export function toParsedConfigFromMerged(merged, manifest) {
  const rawConfig = manifest.config ?? {};
  const config = rawConfig.text_config ?? rawConfig;
  const inf = merged.inference;

  // Validate required fields are present (fail fast on incomplete manifests)
  validateRequiredInferenceFields(inf, merged.modelId);
  if (manifest.quantization == null) {
    throw new Error(`Manifest "${merged.modelId}" is missing quantization.`);
  }

  // Get architecture dimensions
  const arch = (manifest.architecture && typeof manifest.architecture === 'object')
    ? manifest.architecture
    : null;
  if (!arch) {
    throw new Error(
      `Manifest "${merged.modelId}" is missing architecture config. ` +
      `Re-convert the model using the latest converter to add manifest.architecture.`
    );
  }
  const resolvedIntermediateSize = resolveIntermediateSizeForRuntime(manifest, inf, arch, merged.modelId);
  const archNumHeads = Number(arch.numAttentionHeads ?? arch.numHeads);
  const archNumKVHeads = Number(arch.numKeyValueHeads ?? arch.numKVHeads);
  const archHeadDim = Number(arch.headDim);
  const archGlobalHeadDimRaw = arch.globalHeadDim ?? null;
  const archGlobalHeadDim = (
    typeof archGlobalHeadDimRaw === 'number' && Number.isFinite(archGlobalHeadDimRaw) && archGlobalHeadDimRaw > 0
  )
    ? Math.trunc(archGlobalHeadDimRaw)
    : null;
  const archNumKvSharedLayersRaw = arch.numKvSharedLayers ?? 0;
  const archNumKvSharedLayers = (
    typeof archNumKvSharedLayersRaw === 'number'
      && Number.isFinite(archNumKvSharedLayersRaw)
      && archNumKvSharedLayersRaw >= 0
  )
    ? Math.trunc(archNumKvSharedLayersRaw)
    : 0;
  const intermediateSizes = buildPerLayerIntermediateSizes({
    baseIntermediateSize: resolvedIntermediateSize,
    numLayers: arch.numLayers,
    numKvSharedLayers: archNumKvSharedLayers,
    useDoubleWideMlp: inf.ffn.useDoubleWideMlp,
    modelId: merged.modelId,
  });
  const maxIntermediateSize = Math.max(...intermediateSizes);
  validateLayerIntermediateSizesAgainstManifest(
    manifest,
    arch.hiddenSize,
    intermediateSizes,
    merged.modelId
  );

  // Compute layer types from layerPattern
  
  let layerTypes = null;
  if (inf.layerPattern) {
    const numLayers = arch.numLayers;
    const patternType = inf.layerPattern.type;

    if (patternType === 'custom') {
      layerTypes = parseCustomLayerTypes(inf.layerPattern.layerTypes, numLayers, merged.modelId);
    } else {
      // Fail fast if alternating pattern lacks required globalPattern
      if (patternType === 'alternating' && inf.layerPattern.globalPattern == null) {
        throw new Error(
          `Manifest "${merged.modelId}" has layerPattern.type='alternating' but globalPattern is missing. ` +
          `Re-convert the model to include layerPattern.globalPattern.`
        );
      }

      // Fail fast if every_n pattern lacks required period
      if (patternType === 'every_n' && inf.layerPattern.period == null) {
        throw new Error(
          `Manifest "${merged.modelId}" has layerPattern.type='every_n' but period is missing. ` +
          `Re-convert the model to include layerPattern.period.`
        );
      }
      const period = inf.layerPattern.period;
      const rawOffset = inf.layerPattern.offset;
      const offset = (
        Number.isFinite(rawOffset) && period != null && period > 0
      )
        ? ((Math.trunc(rawOffset) % period) + period) % period
        : 0;
      const pattern = inf.layerPattern.globalPattern;
      const patternKind = selectRuleValue(
        'inference',
        'layerPattern',
        'patternKind',
        { patternType, globalPattern: pattern }
      );
      if (patternKind) {
        layerTypes = Array.from({ length: numLayers }, (_, i) => {
          const isEven = i % 2 === 0;
          // For every_n pattern: global at layer "offset" and every N thereafter.
          // e.g. period=6, offset=5 => indices 5,11,17,...
          const isStride = period == null ? false : (((i - offset) % period + period) % period) === 0;
          return selectRuleValue(
            'inference',
            'layerPattern',
            'layerType',
            { patternKind, isEven, isStride }
          );
        });
      }
    }
  }

  if (!Array.isArray(layerTypes) && Array.isArray(config.layer_types) && config.layer_types.length > 0) {
    layerTypes = parseCustomLayerTypes(config.layer_types, arch.numLayers, merged.modelId);
  }

  // Compute queryPreAttnScalar from manifest inference (NOT from family detection)
  // Manifest-first: queryPreAttnScalar is required in ManifestAttentionSchema
  const headDim = archHeadDim;
  const queryPreAttnScalar = inf.attention.queryPreAttnScalar;
  const causalAttention = inf.attention.causal;

  // Preserve the manifest scalar exactly. Gemma-family models legitimately use
  // queryPreAttnScalar=1, but sqrt(headDim) is still a known converter bug that
  // produces attnScale = 1/sqrt(sqrt(headDim)) instead of the intended value.
  if (queryPreAttnScalar != null && headDim != null
      && queryPreAttnScalar !== headDim
      && Math.abs(queryPreAttnScalar - Math.sqrt(headDim)) < 0.01) {
    throw new Error(
      `Model "${merged.modelId}": queryPreAttnScalar (${queryPreAttnScalar}) ` +
      `equals sqrt(headDim) instead of headDim (${headDim}). ` +
      `This is a known converter bug — the manifest must be regenerated ` +
      `with the corrected converter.`
    );
  }

  // Get stop token IDs (cast to Manifest for compatibility)
  const stopTokenIds = getStopTokenIds(manifest);
  const embeddingPostprocessor = inf.output.embeddingPostprocessor;
  if (embeddingPostprocessor) {
    if (embeddingPostprocessor.includePrompt !== true) {
      throw new Error(
        `Manifest "${merged.modelId}" requires output.embeddingPostprocessor.includePrompt=false, ` +
        'but prompt-token masking is not implemented for embedding extraction.'
      );
    }
    let expectedInputSize = arch.hiddenSize;
    for (let i = 0; i < embeddingPostprocessor.projections.length; i++) {
      const projection = embeddingPostprocessor.projections[i];
      if (projection.inputSize !== expectedInputSize) {
        throw new Error(
          `Manifest "${merged.modelId}" has output.embeddingPostprocessor.projections[${i}].inputSize=${projection.inputSize}, ` +
          `expected ${expectedInputSize}.`
        );
      }
      expectedInputSize = projection.outputSize;
    }
  }

  // Get MoE config
  const moeConfig = manifest.moeConfig ?? null;
  const useMoE = (moeConfig?.numExperts ?? 0) > 1;
  if (useMoE && (moeConfig?.numExperts == null || moeConfig?.numExpertsPerToken == null || !moeConfig?.expertFormat)) {
    throw new Error(`Manifest "${manifest.modelId}" is missing moeConfig fields for MoE inference.`);
  }
  const numExperts = useMoE ? moeConfig.numExperts : 0;
  const moeTopK = useMoE ? moeConfig.numExpertsPerToken : 0;
  const expertFormat = useMoE ? moeConfig.expertFormat : null;

  // RoPE scaling - use manifest inference as source of truth (not raw config)
  const ropeScale = inf.rope.ropeScalingFactor;
  const ropeScalingType = inf.rope.ropeScalingType;
  const ropeLocalScale = inf.rope.ropeLocalScalingFactor;
  const ropeLocalScalingType = inf.rope.ropeLocalScalingType;
  const partialRotaryFactor = inf.rope.partialRotaryFactor;
  const ropeLocalPartialRotaryFactor = inf.rope.ropeLocalPartialRotaryFactor;
  const mropeInterleaved = inf.rope.mropeInterleaved === true;
  const ropeInterleaved = inf.rope.ropeInterleaved === true;

  if (ropeLocalScale == null && (inf.rope.ropeLocalTheta != null || inf.rope.mropeSection != null)) {
    throw new Error(
      `Model "${merged.modelId}" uses hybrid/mRoPE but is missing rope.ropeLocalScalingFactor in manifest. ` +
      `Re-convert the model using the latest converter or update the manifest to include an explicit scale.`
    );
  }
  const mropeSection = Array.isArray(inf.rope.mropeSection)
    ? inf.rope.mropeSection.map((entry) => Math.trunc(Number(entry)))
    : null;
  const ropeRotaryDim = resolveRotaryDim(archGlobalHeadDim ?? archHeadDim, partialRotaryFactor, merged.modelId);
  const ropeLocalRotaryDim = resolveRotaryDim(archHeadDim, ropeLocalPartialRotaryFactor, merged.modelId);
  const ropeFrequencyBaseDim = resolveFrequencyBaseDim(
    archGlobalHeadDim ?? archHeadDim,
    ropeRotaryDim,
    inf.rope.ropeFrequencyBaseDim,
    merged.modelId,
    'rope.ropeFrequencyBaseDim'
  );
  const ropeLocalFrequencyBaseDim = resolveFrequencyBaseDim(
    archHeadDim,
    ropeLocalRotaryDim,
    inf.rope.ropeLocalFrequencyBaseDim,
    merged.modelId,
    'rope.ropeLocalFrequencyBaseDim'
  );
  if (mropeSection && mropeSection.some((entry) => !Number.isFinite(entry) || entry <= 0)) {
    throw new Error(
      `Manifest "${merged.modelId}" has invalid rope.mropeSection; expected positive integers.`
    );
  }
  if (mropeInterleaved && mropeSection) {
    const doubledMropeDim = mropeSection.reduce((sum, entry) => sum + entry, 0) * 2;
    if (doubledMropeDim !== ropeRotaryDim) {
      throw new Error(
        `Manifest "${merged.modelId}" declares rope.mropeSection=${JSON.stringify(mropeSection)}, ` +
        `which expands to rotary dim ${doubledMropeDim}, but the resolved rotary dim is ${ropeRotaryDim}.`
      );
    }
  }

  // Build ropeScaling object from manifest values if scaling is enabled
  // Include YARN params when present
  
  const ropeScaling = ropeScalingType ? {
    type: ropeScalingType,
    factor: ropeScale,
    ...(ropeScalingType === 'yarn' && inf.rope.yarnBetaFast != null && { beta_fast: inf.rope.yarnBetaFast }),
    ...(ropeScalingType === 'yarn' && inf.rope.yarnBetaSlow != null && { beta_slow: inf.rope.yarnBetaSlow }),
    ...(ropeScalingType === 'yarn' && inf.rope.yarnOriginalMaxPos != null && {
      original_max_position_embeddings: inf.rope.yarnOriginalMaxPos
    }),
  } : null;
  const ropeLocalScaling = ropeLocalScalingType ? {
    type: ropeLocalScalingType,
    factor: ropeLocalScale,
    ...(ropeLocalScalingType === 'yarn' && (inf.rope.ropeLocalYarnBetaFast ?? inf.rope.yarnBetaFast) != null && {
      beta_fast: inf.rope.ropeLocalYarnBetaFast ?? inf.rope.yarnBetaFast
    }),
    ...(ropeLocalScalingType === 'yarn' && (inf.rope.ropeLocalYarnBetaSlow ?? inf.rope.yarnBetaSlow) != null && {
      beta_slow: inf.rope.ropeLocalYarnBetaSlow ?? inf.rope.yarnBetaSlow
    }),
    ...(ropeLocalScalingType === 'yarn'
      && (inf.rope.ropeLocalYarnOriginalMaxPos ?? inf.rope.yarnOriginalMaxPos) != null && {
      original_max_position_embeddings:
        inf.rope.ropeLocalYarnOriginalMaxPos ?? inf.rope.yarnOriginalMaxPos
    }),
  } : null;

  // Activation type
  const activation = inf.ffn.activation;
  
  const hiddenActivation = selectRuleValue(
    'inference',
    'config',
    'hiddenActivation',
    { activation }
  );

  const chatTemplateType = inf.chatTemplate.type;
  validateChatTemplateType(chatTemplateType, merged.modelId);
  const chatTemplateEnabled = inf.chatTemplate.enabled;
  const chatTemplateThinking = inf.chatTemplate.thinking ?? null;
  const parsePositiveInt = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num) || num <= 0) return null;
    return Math.trunc(num);
  };

  const linearNumKeyHeads = parsePositiveInt(arch.linearNumKeyHeads ?? config.linear_num_key_heads);
  const linearNumValueHeads = parsePositiveInt(arch.linearNumValueHeads ?? config.linear_num_value_heads);
  const linearKeyHeadDim = parsePositiveInt(arch.linearKeyHeadDim ?? config.linear_key_head_dim);
  const linearValueHeadDim = parsePositiveInt(arch.linearValueHeadDim ?? config.linear_value_head_dim);
  const linearConvKernelDim = parsePositiveInt(arch.linearConvKernelDim ?? config.linear_conv_kernel_dim);
  const linearNormMode = parseLinearNormMode(
    arch.linearNormMode ?? config.linear_norm_mode,
    config.linear_norm_shared,
    merged.modelId
  );
  const hasMixedAttentionGeometry = archGlobalHeadDim != null && archGlobalHeadDim !== archHeadDim;
  const hasSharedKvLayers = archNumKvSharedLayers > 0;
  const hasExplicitLayerTypes = Array.isArray(layerTypes) && layerTypes.length === arch.numLayers;
  const decodeStrategy = (hasMixedAttentionGeometry || hasSharedKvLayers) && !hasExplicitLayerTypes
    ? 'replay_prefill'
    : 'incremental';

  return {
    numLayers: arch.numLayers,
    hiddenSize: arch.hiddenSize,
    intermediateSize: resolvedIntermediateSize,
    intermediateSizes,
    maxIntermediateSize,
    numHeads: archNumHeads,
    numKVHeads: archNumKVHeads,
    headDim: archHeadDim,
    globalHeadDim: archGlobalHeadDim,
    vocabSize: arch.vocabSize,
    hiddenSizePerLayerInput: arch.hiddenSizePerLayerInput ?? null,
    vocabSizePerLayerInput: arch.vocabSizePerLayerInput ?? null,
    numKvSharedLayers: archNumKvSharedLayers,
    maxSeqLen: arch.maxSeqLen,
    useMoE,
    numExperts,
    moeTopK,
    expertFormat,
    slidingWindow: inf.attention.slidingWindow,
    ropeTheta: inf.rope.ropeTheta,
    ropeLocalTheta: inf.rope.ropeLocalTheta,
    ropeRotaryDim,
    ropeLocalRotaryDim,
    ropeFrequencyBaseDim,
    ropeLocalFrequencyBaseDim,
    ropeInterleaved,
    mropeInterleaved,
    mropeSection,
    partialRotaryFactor,
    ropeLocalPartialRotaryFactor,
    ropeScale,
    ropeLocalScale,
    ropeScalingType,
    ropeLocalScalingType,
    ropeScaling,
    ropeLocalScaling,
    quantization: manifest.quantization,
    quantMethod: config.quantization_config?.quant_method ?? null,
    rmsNormEps: inf.normalization.rmsNormEps,
    rmsNormWeightOffset: inf.normalization.rmsNormWeightOffset,
    postAttentionNorm: inf.normalization.postAttentionNorm,
    preFeedforwardNorm: inf.normalization.preFeedforwardNorm,
    postFeedforwardNorm: inf.normalization.postFeedforwardNorm,
    scaleEmbeddings: inf.output.scaleEmbeddings,
    useTiedEmbeddings: inf.output.tieWordEmbeddings,
    embeddingTranspose: inf.output.embeddingTranspose,
    embeddingVocabSize: inf.output.embeddingVocabSize,
    embeddingPostprocessor,
    hiddenActivation,
    useDoubleWideMlp: inf.ffn.useDoubleWideMlp,
    swigluLimit: inf.ffn.swigluLimit,
    stopTokenIds,
    layerTypes,
    linearNumKeyHeads,
    linearNumValueHeads,
    linearKeyHeadDim,
    linearValueHeadDim,
    linearConvKernelDim,
    linearNormMode,
    attentionBias: inf.attention.attentionBias,
    causalAttention,
    finalLogitSoftcapping: inf.output.finalLogitSoftcapping,
    attnLogitSoftcapping: inf.attention.attnLogitSoftcapping,
    queryKeyNorm: inf.attention.queryKeyNorm,
    valueNorm: inf.attention.valueNorm,
    attentionOutputGate: inf.attention.attentionOutputGate === true,
    queryPreAttnScalar,
    layerPipeline: inf.pipeline ?? null,
    chatTemplateType,
    chatTemplateEnabled,
    chatTemplateThinking,
    decodeStrategy,
    kernelPath: null,
    visionConfig: resolveVisionConfig(config, manifest),
    audioConfig: resolveAudioConfig(config, manifest),
  };
}


export function parseModelConfigFromManifest(manifest, runtimeOverrides) {
  assertSupportedRuntimeModelType(manifest);

  // Merge manifest inference with runtime overrides
  const merged = mergeConfig(
    {
      modelId: manifest.modelId ?? 'unknown',
      inference: manifest.inference,
      architecture: manifest.architecture,
    },
    runtimeOverrides
  );

  // Log config source info
  const runtimeSources = Array.from(merged._sources.entries())
    .filter(([, src]) => src === 'runtime')
    .length;
  const totalSources = merged._sources.size;
  if (runtimeSources > 0) {
    log.info('Config', `Manifest-first config: ${totalSources - runtimeSources} from manifest, ${runtimeSources} from runtime`);
  } else {
    log.debug('Config', `Manifest-first config: ${totalSources} values from manifest`);
  }

  // Dump full field-to-source mapping at debug level for diagnostics
  const sourceDump = dumpConfigSources(merged);
  log.debug('Config', `Config source map: ${JSON.stringify(sourceDump)}`);

  // Convert to ParsedModelConfig
  return toParsedConfigFromMerged(merged, manifest);
}

// =============================================================================
// Main Entry Point
// =============================================================================


export function parseModelConfig(manifest, runtimeOverrides) {
  // Manifest-first architecture: inference config is required
  if (!hasManifestInference(manifest)) {
    const modelId = manifest.modelId ?? 'unknown';
    throw new Error(
      `Manifest "${modelId}" is missing inference config. ` +
      `Re-convert the model using the latest converter to add manifest.inference. ` +
      `Legacy family-registry resolution has been removed.`
    );
  }

  log.info('Config', 'Using manifest-first config (source of truth)');
  return parseModelConfigFromManifest(manifest, runtimeOverrides);
}
