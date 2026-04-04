import { validateRequiredInferenceFields } from '../inference/pipelines/text/config.js';

function cloneJson(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function setPath(root, path, value) {
  let current = root;
  for (let i = 0; i < path.length - 1; i += 1) {
    current = current[path[i]];
  }
  current[path[path.length - 1]] = value;
}

function deletePath(root, path) {
  let current = root;
  for (let i = 0; i < path.length - 1; i += 1) {
    current = current[path[i]];
  }
  delete current[path[path.length - 1]];
}

function createValidInferenceFixture() {
  return {
    attention: {
      queryPreAttnScalar: 256,
      queryKeyNorm: true,
      valueNorm: false,
      attentionBias: false,
      causal: true,
      slidingWindow: null,
      attnLogitSoftcapping: null,
    },
    normalization: {
      rmsNormWeightOffset: true,
      rmsNormEps: 1e-6,
      postAttentionNorm: true,
      preFeedforwardNorm: true,
      postFeedforwardNorm: true,
    },
    ffn: {
      activation: 'gelu',
      gatedActivation: true,
      useDoubleWideMlp: false,
      swigluLimit: null,
    },
    rope: {
      ropeTheta: 1000000,
      ropeScalingFactor: 1.0,
      ropeScalingType: null,
      ropeLocalTheta: null,
      mropeInterleaved: false,
      mropeSection: null,
      partialRotaryFactor: null,
      ropeLocalPartialRotaryFactor: null,
      ropeFrequencyBaseDim: null,
      ropeLocalFrequencyBaseDim: null,
      yarnBetaFast: null,
      yarnBetaSlow: null,
      yarnOriginalMaxPos: null,
      ropeLocalYarnBetaFast: null,
      ropeLocalYarnBetaSlow: null,
      ropeLocalYarnOriginalMaxPos: null,
    },
    output: {
      tieWordEmbeddings: true,
      scaleEmbeddings: true,
      embeddingTranspose: false,
      finalLogitSoftcapping: null,
      embeddingVocabSize: null,
      embeddingPostprocessor: null,
    },
    layerPattern: {
      type: 'every_n',
      globalPattern: null,
      period: 6,
      offset: 0,
    },
    chatTemplate: {
      type: null,
      enabled: true,
    },
  };
}

const FIELD_CASES = Object.freeze([
  { kind: 'nonNullable', path: ['attention', 'queryPreAttnScalar'], message: 'attention.queryPreAttnScalar is required' },
  { kind: 'nonNullable', path: ['attention', 'queryKeyNorm'], message: 'attention.queryKeyNorm is required' },
  { kind: 'nonNullable', path: ['attention', 'valueNorm'], message: 'attention.valueNorm is required' },
  { kind: 'nonNullable', path: ['attention', 'attentionBias'], message: 'attention.attentionBias is required' },
  { kind: 'nonNullable', path: ['attention', 'causal'], message: 'attention.causal is required' },
  { kind: 'nullable', path: ['attention', 'slidingWindow'], message: 'attention.slidingWindow must be explicitly set' },
  { kind: 'nullable', path: ['attention', 'attnLogitSoftcapping'], message: 'attention.attnLogitSoftcapping must be explicitly set' },
  { kind: 'nonNullable', path: ['normalization', 'rmsNormWeightOffset'], message: 'normalization.rmsNormWeightOffset is required' },
  { kind: 'nonNullable', path: ['normalization', 'rmsNormEps'], message: 'normalization.rmsNormEps is required' },
  { kind: 'nonNullable', path: ['normalization', 'postAttentionNorm'], message: 'normalization.postAttentionNorm is required' },
  { kind: 'nonNullable', path: ['normalization', 'preFeedforwardNorm'], message: 'normalization.preFeedforwardNorm is required' },
  { kind: 'nonNullable', path: ['normalization', 'postFeedforwardNorm'], message: 'normalization.postFeedforwardNorm is required' },
  { kind: 'nonNullable', path: ['ffn', 'activation'], message: 'ffn.activation is required' },
  { kind: 'nonNullable', path: ['ffn', 'gatedActivation'], message: 'ffn.gatedActivation is required' },
  { kind: 'nonNullable', path: ['ffn', 'useDoubleWideMlp'], message: 'ffn.useDoubleWideMlp is required' },
  { kind: 'nullable', path: ['ffn', 'swigluLimit'], message: 'ffn.swigluLimit must be explicitly set' },
  { kind: 'nonNullable', path: ['rope', 'ropeTheta'], message: 'rope.ropeTheta is required' },
  { kind: 'nonNullable', path: ['rope', 'ropeScalingFactor'], message: 'rope.ropeScalingFactor is required' },
  { kind: 'nullable', path: ['rope', 'ropeScalingType'], message: 'rope.ropeScalingType must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalTheta'], message: 'rope.ropeLocalTheta must be explicitly set' },
  { kind: 'nonNullable', path: ['rope', 'mropeInterleaved'], message: 'rope.mropeInterleaved is required' },
  { kind: 'nullable', path: ['rope', 'mropeSection'], message: 'rope.mropeSection must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'partialRotaryFactor'], message: 'rope.partialRotaryFactor must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalPartialRotaryFactor'], message: 'rope.ropeLocalPartialRotaryFactor must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeFrequencyBaseDim'], message: 'rope.ropeFrequencyBaseDim must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalFrequencyBaseDim'], message: 'rope.ropeLocalFrequencyBaseDim must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'yarnBetaFast'], message: 'rope.yarnBetaFast must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'yarnBetaSlow'], message: 'rope.yarnBetaSlow must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'yarnOriginalMaxPos'], message: 'rope.yarnOriginalMaxPos must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalYarnBetaFast'], message: 'rope.ropeLocalYarnBetaFast must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalYarnBetaSlow'], message: 'rope.ropeLocalYarnBetaSlow must be explicitly set' },
  { kind: 'nullable', path: ['rope', 'ropeLocalYarnOriginalMaxPos'], message: 'rope.ropeLocalYarnOriginalMaxPos must be explicitly set' },
  { kind: 'nonNullable', path: ['output', 'tieWordEmbeddings'], message: 'output.tieWordEmbeddings is required' },
  { kind: 'nonNullable', path: ['output', 'scaleEmbeddings'], message: 'output.scaleEmbeddings is required' },
  { kind: 'nonNullable', path: ['output', 'embeddingTranspose'], message: 'output.embeddingTranspose is required' },
  { kind: 'nullable', path: ['output', 'finalLogitSoftcapping'], message: 'output.finalLogitSoftcapping must be explicitly set' },
  { kind: 'nullable', path: ['output', 'embeddingVocabSize'], message: 'output.embeddingVocabSize must be explicitly set' },
  { kind: 'nullable', path: ['output', 'embeddingPostprocessor'], message: 'output.embeddingPostprocessor must be explicitly set' },
  { kind: 'nonNullable', path: ['layerPattern', 'type'], message: 'layerPattern.type is required' },
  { kind: 'nullable', path: ['layerPattern', 'globalPattern'], message: 'layerPattern.globalPattern must be explicitly set' },
  { kind: 'nullable', path: ['layerPattern', 'period'], message: 'layerPattern.period must be explicitly set' },
  { kind: 'nullable', path: ['layerPattern', 'offset'], message: 'layerPattern.offset must be explicitly set' },
  { kind: 'nullable', path: ['chatTemplate', 'type'], message: 'chatTemplate.type must be explicitly set' },
  { kind: 'nonNullable', path: ['chatTemplate', 'enabled'], message: 'chatTemplate.enabled is required' },
]);

export function buildRequiredInferenceFieldsContractArtifact() {
  const errors = [];
  const checks = [];
  const baseInference = createValidInferenceFixture();

  try {
    validateRequiredInferenceFields(cloneJson(baseInference), 'required-fields-contract');
    checks.push({ id: 'requiredInferenceFields.validFixture', ok: true });
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
    checks.push({ id: 'requiredInferenceFields.validFixture', ok: false });
  }

  for (const field of FIELD_CASES) {
    const missingInference = cloneJson(baseInference);
    if (field.kind === 'nonNullable') {
      setPath(missingInference, field.path, null);
    } else {
      deletePath(missingInference, field.path);
    }
    let rejectsAsExpected = false;
    try {
      validateRequiredInferenceFields(missingInference, 'required-fields-contract');
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      rejectsAsExpected = message.includes(field.message);
    }
    if (!rejectsAsExpected) {
      errors.push(
        `[RequiredInferenceFieldsContract] ${field.path.join('.')} did not produce the expected required-field failure.`
      );
    }
    checks.push({
      id: `requiredInferenceFields.${field.path.join('.')}.rejectsInvalid`,
      ok: rejectsAsExpected,
    });

    if (field.kind === 'nullable') {
      const nullableInference = cloneJson(baseInference);
      setPath(nullableInference, field.path, null);
      let nullableAccepted = true;
      try {
        validateRequiredInferenceFields(nullableInference, 'required-fields-contract');
      } catch {
        nullableAccepted = false;
      }
      if (!nullableAccepted) {
        errors.push(
          `[RequiredInferenceFieldsContract] ${field.path.join('.')} should allow explicit null but was rejected.`
        );
      }
      checks.push({
        id: `requiredInferenceFields.${field.path.join('.')}.acceptsNull`,
        ok: nullableAccepted,
      });
    }
  }

  const customInference = cloneJson(baseInference);
  customInference.layerPattern.type = 'custom';
  delete customInference.layerPattern.layerTypes;
  let customLayerTypesRejected = false;
  try {
    validateRequiredInferenceFields(customInference, 'required-fields-contract');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    customLayerTypesRejected = message.includes('layerPattern.layerTypes must be explicitly set for custom patterns');
  }
  if (!customLayerTypesRejected) {
    errors.push('[RequiredInferenceFieldsContract] custom layerPattern without layerTypes was not rejected.');
  }
  checks.push({
    id: 'requiredInferenceFields.layerPattern.layerTypes.rejectsMissingForCustom',
    ok: customLayerTypesRejected,
  });

  return {
    schemaVersion: 1,
    source: 'doppler',
    ok: errors.length === 0,
    checks,
    errors,
    stats: {
      fieldCases: FIELD_CASES.length,
      nullableCases: FIELD_CASES.filter((field) => field.kind === 'nullable').length,
      nonNullableCases: FIELD_CASES.filter((field) => field.kind === 'nonNullable').length,
    },
  };
}

export function buildManifestRequiredInferenceFieldsArtifact(inference, label = 'manifest.inference') {
  const errors = [];
  const checks = [];
  let ok = true;
  try {
    validateRequiredInferenceFields(cloneJson(inference), label);
  } catch (error) {
    ok = false;
    errors.push(error instanceof Error ? error.message : String(error));
  }
  checks.push({
    id: `${label}.requiredInferenceFields`,
    ok,
  });
  return {
    schemaVersion: 1,
    source: 'doppler',
    scope: 'manifest',
    label,
    ok,
    checks,
    errors,
    stats: {
      fieldCases: FIELD_CASES.length,
      nullableCases: FIELD_CASES.filter((field) => field.kind === 'nullable').length,
      nonNullableCases: FIELD_CASES.filter((field) => field.kind === 'nonNullable').length,
    },
  };
}
