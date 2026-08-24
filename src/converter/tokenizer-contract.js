import {
  // Constants
  SHARD_SIZE as SCHEMA_SHARD_SIZE,
  RDRR_VERSION as SCHEMA_RDRR_VERSION,
  ConversionStage as SchemaConversionStage,
  DEFAULT_MANIFEST_INFERENCE,
  formatBytes,
} from '../config/schema/index.js';
import {
  classifyTensorRole,
  generateShardFilename,
  resolveTensorGroup,
  resolveTensorRole,
} from '../formats/rdrr/index.js';
import { log } from '../debug/index.js';
import {
  getInferenceLayerPatternContractArtifact,
  selectRuleValue,
} from '../rules/rule-registry.js';
import {
  createConverterConfig,
} from '../config/index.js';
import { buildExecutionContractArtifact } from '../config/execution-contract-check.js';
import { buildManifestRequiredInferenceFieldsArtifact } from '../config/required-inference-fields-contract-check.js';
import { resolveEosTokenId } from './tokenizer-special-tokens.js';
import { inferBundledTokenizerBehaviorFlags } from '../inference/tokenizers/behavior-flags.js';
import {
  normalizeQ4KLayout,
  resolveManifestQuantization,
  resolveEffectiveQuantizationInfo,
} from './quantization-info.js';
import {
  float16ToFloat32,
  float32ToFloat16,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  quantizeToQ4KMColumnWise,
  quantizeToInt4PerRowSymmetric,
} from './quantizer.js';
import { cloneJsonValue } from '../utils/clone-json.js';
import { SOURCE_PACKED_QUANT_DTYPES, isGemma4PerLayerEmbedTensor, normalizeModulesToNotConvert, normalizeStorageQuant, resolveQuantizeEmbeddings, transformTensorBytes } from './tensor-transform.js';
import { buildArtifactIdentity, extractArchitecture, getNestedTextConfig, sanitizeModelId } from './artifact-identity.js';
import { RDRR_VERSION, createManifest, modelHasMoETensors } from './manifest-builder.js';

export const COMPRESSED_TENSORS_W4A16_SUFFIXES = {
  packed: '.weight_packed',
  scale: '.weight_scale',
  shape: '.weight_shape',
};

export function normalizeTensorName(tensor) {
  const name = tensor?.name;
  return typeof name === 'string' ? name : '';
}

export function normalizePositiveIntegerShape(value, tensorName) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`Compressed-tensors W4A16 tensor "${tensorName}" is missing a logical shape.`);
  }
  return value.map((entry, index) => {
    const number = Number(entry);
    if (!Number.isInteger(number) || number <= 0) {
      throw new Error(
        `Compressed-tensors W4A16 tensor "${tensorName}" has invalid shape[${index}]=${JSON.stringify(entry)}.`
      );
    }
    return number;
  });
}

export function resolveCompressedTensorBaseName(name, suffix) {
  return name.endsWith(suffix)
    ? name.slice(0, -suffix.length)
    : null;
}

export function collectCompressedTensorsW4A16(tensors) {
  const groups = new Map();
  const groupFor = (baseName) => {
    const existing = groups.get(baseName);
    if (existing) return existing;
    const created = {
      baseName,
      packed: null,
      scale: null,
      shape: null,
    };
    groups.set(baseName, created);
    return created;
  };

  for (const tensor of tensors) {
    const name = normalizeTensorName(tensor);
    const packedBase = resolveCompressedTensorBaseName(name, COMPRESSED_TENSORS_W4A16_SUFFIXES.packed);
    if (packedBase) {
      groupFor(packedBase).packed = tensor;
      continue;
    }
    const scaleBase = resolveCompressedTensorBaseName(name, COMPRESSED_TENSORS_W4A16_SUFFIXES.scale);
    if (scaleBase) {
      groupFor(scaleBase).scale = tensor;
      continue;
    }
    const shapeBase = resolveCompressedTensorBaseName(name, COMPRESSED_TENSORS_W4A16_SUFFIXES.shape);
    if (shapeBase) {
      groupFor(shapeBase).shape = tensor;
    }
  }

  return groups;
}

export function assertCompressedTensorsW4A16Group(group) {
  const missing = [];
  if (!group.packed) missing.push('weight_packed');
  if (!group.scale) missing.push('weight_scale');
  if (!group.shape) missing.push('weight_shape');
  if (missing.length > 0) {
    throw new Error(
      `Compressed-tensors W4A16 tensor "${group.baseName}.weight" is missing companion tensors: ${missing.join(', ')}.`
    );
  }
  const packedShape = normalizePositiveIntegerShape(group.packed.shape, group.packed.name);
  if (packedShape.length !== 2) {
    throw new Error(
      `Compressed-tensors W4A16 tensor "${group.packed.name}" must be 2D; got shape ${JSON.stringify(packedShape)}.`
    );
  }
  const scaleShape = normalizePositiveIntegerShape(group.scale.shape, group.scale.name);
  if (scaleShape.length !== 2) {
    throw new Error(
      `Compressed-tensors W4A16 scale tensor "${group.scale.name}" must be 2D; got shape ${JSON.stringify(scaleShape)}.`
    );
  }
  normalizePositiveIntegerShape(group.shape.shape, group.shape.name);
  const shapeDtype = String(group.shape.dtype || '').toUpperCase();
  if (shapeDtype !== 'I64' && shapeDtype !== 'I32' && shapeDtype !== 'U32') {
    throw new Error(
      `Compressed-tensors W4A16 shape tensor "${group.shape.name}" must use I64, I32, or U32; got "${group.shape.dtype}".`
    );
  }
}

export const COMPRESSED_TENSORS_W4A16_PACKED_VALUES_PER_ELEMENT = {
  U8: 2,
  I8: 2,
  U16: 4,
  I16: 4,
  U32: 8,
  I32: 8,
};

export function inferCompressedTensorsW4A16LogicalShape(group) {
  const packedShape = normalizePositiveIntegerShape(group.packed.shape, group.packed.name);
  const packedDtype = String(group.packed.dtype || '').toUpperCase();
  const valuesPerElement = COMPRESSED_TENSORS_W4A16_PACKED_VALUES_PER_ELEMENT[packedDtype];
  if (!valuesPerElement) {
    throw new Error(
      `Compressed-tensors W4A16 tensor "${group.packed.name}" has unsupported packed dtype "${group.packed.dtype}".`
    );
  }
  return [packedShape[0], packedShape[1] * valuesPerElement];
}

export function shouldNormalizeCompressedTensorsW4A16(converterConfig) {
  return normalizeStorageQuant(converterConfig?.quantization?.weights) === 'w4a16'
    || normalizeStorageQuant(converterConfig?.quantization?.sourceQuantizationTarget) === 'w4a16';
}

export function sortTensorsForConversion(tensors) {
  return [...tensors].sort((left, right) => normalizeTensorName(left).localeCompare(normalizeTensorName(right)));
}

export function normalizeCompressedTensorsW4A16(tensors, converterConfig) {
  const groups = collectCompressedTensorsW4A16(tensors);
  if (groups.size === 0) {
    return tensors;
  }
  if (!shouldNormalizeCompressedTensorsW4A16(converterConfig)) {
    throw new Error(
      'Compressed-tensors W4A16 tensors were detected, but converter.quantization.weights or ' +
      'converter.quantization.sourceQuantizationTarget is not "w4a16".'
    );
  }

  const byName = new Map(tensors.map((tensor) => [normalizeTensorName(tensor), tensor]));
  const consumed = new Set();
  const companionByName = new Map();
  const synthetic = [];
  const sortedGroups = [...groups.values()].sort((left, right) => left.baseName.localeCompare(right.baseName));
  for (const group of sortedGroups) {
    assertCompressedTensorsW4A16Group(group);
    const logicalName = `${group.baseName}.weight`;
    if (byName.has(logicalName)) {
      throw new Error(
        `Compressed-tensors W4A16 logical tensor "${logicalName}" conflicts with an existing source tensor.`
      );
    }
    consumed.add(group.packed.name);
    companionByName.set(group.scale.name, {
      ...group.scale,
      compressedTensorsW4A16Companion: {
        role: 'scales',
        primary: logicalName,
      },
    });
    companionByName.set(group.shape.name, {
      ...group.shape,
      compressedTensorsW4A16Companion: {
        role: 'shape',
        primary: logicalName,
      },
    });
    synthetic.push({
      ...group.packed,
      name: logicalName,
      dtype: 'W4A16',
      shape: inferCompressedTensorsW4A16LogicalShape(group),
      packedSourceName: group.packed.name,
      compressedTensorsW4A16: {
        packed: group.packed.name,
        scales: group.scale.name,
        shape: group.shape.name,
      },
      storage: {
        packing: 'w4a16',
        blockShape: [32],
        blockBytes: 16,
        companions: [
          { role: 'scales', tensorId: group.scale.name },
          { role: 'shape', tensorId: group.shape.name },
        ],
      },
    });
  }

  return sortTensorsForConversion([
    ...tensors
      .filter((tensor) => !consumed.has(normalizeTensorName(tensor)))
      .map((tensor) => companionByName.get(normalizeTensorName(tensor)) ?? tensor),
    ...synthetic,
  ]);
}
