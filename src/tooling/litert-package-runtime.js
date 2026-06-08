import { parseTFLiteFromSource, TFLITE_FILE_IDENTIFIER } from '../formats/tflite/types.js';
import {
  LITERT_TASK_DEFAULT_METADATA_ENTRY,
  LITERT_TASK_DEFAULT_TFLITE_ENTRY,
  LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY,
  findLiteRTLMSectionByType,
  findLiteRTLMMetadataSection,
  findLiteRTLMSentencePieceTokenizerSection,
  findLiteRTLMTFLiteModelSection,
  findLiteRTLMTFLiteWeightsSection,
  parseLiteRTLMFromSource,
  parseLiteRTTaskFromSource,
} from '../formats/litert/types.js';
import { resolveDirectSourcePackageProfile } from './source-package-profiles.js';
import { cloneJsonValue } from '../utils/clone-json.js';

export const LITERT_PACKAGE_SOURCE_KIND_TASK = 'litert-task';
export const LITERT_PACKAGE_SOURCE_KIND_LITERTLM = 'litertlm';

function normalizeText(value) {
  return String(value || '').trim();
}

function normalizeLiteRTStorageEncoding(value, fieldName) {
  const storageEncoding = normalizeText(value).toLowerCase();
  if (storageEncoding !== 'signed' && storageEncoding !== 'offset_binary') {
    throw new Error(
      `direct-source runtime: ${fieldName} must be "signed" or "offset_binary", got "${value ?? 'missing'}".`
    );
  }
  return storageEncoding;
}

function normalizeLayerIndexList(value, numLayers, fieldName) {
  if (!Array.isArray(value)) {
    throw new Error(`direct-source runtime: ${fieldName} must be an array of layer indices.`);
  }
  const indexes = [];
  let previous = -1;
  for (const rawIndex of value) {
    const layerIndex = Number(rawIndex);
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= numLayers) {
      throw new Error(
        `direct-source runtime: ${fieldName} contains invalid layer index "${String(rawIndex)}".`
      );
    }
    if (layerIndex <= previous) {
      throw new Error(`direct-source runtime: ${fieldName} must be sorted and unique.`);
    }
    indexes.push(layerIndex);
    previous = layerIndex;
  }
  return indexes;
}

function createVirtualFile(path, offset, size, kind, options = {}) {
  return {
    path,
    offset,
    size,
    kind,
    externalPath: normalizeText(options.externalPath) || null,
  };
}

function createSectionSource(source, entry) {
  return {
    name: entry.path,
    size: entry.size,
    async readRange(offset, length) {
      if (entry.externalPath) {
        return source.readRange(offset, length);
      }
      return source.readRange(entry.offset + offset, length);
    },
  };
}

function resolveRequiredProfile(sourceKind, packageBasename) {
  const profile = resolveDirectSourcePackageProfile({
    sourceKind,
    packageBasename,
  });
  if (!profile) {
    throw new Error(
      `direct-source runtime: no package profile matches ${sourceKind} artifact "${packageBasename}". ` +
      'Add an explicit profile under src/config/source-packages/.'
    );
  }
  return profile;
}

function resolvePackageTokenizerConfig(sourceKind, runtimeProfile) {
  const packageTokenizer = runtimeProfile?.tokenizer;
  if (!packageTokenizer || typeof packageTokenizer !== 'object') {
    return null;
  }
  if (sourceKind === LITERT_PACKAGE_SOURCE_KIND_TASK) {
    return cloneJsonValue(packageTokenizer.task ?? null);
  }
  if (sourceKind === LITERT_PACKAGE_SOURCE_KIND_LITERTLM) {
    return cloneJsonValue(packageTokenizer.litertlm ?? null);
  }
  return null;
}

function throwIfUnsupportedPackageProfile(sourceKind, packageBasename, profile, packageConfig) {
  const unsupported = packageConfig?.unsupported;
  if (!unsupported || typeof unsupported !== 'object') {
    return;
  }
  const code = normalizeText(unsupported.code) || 'unsupported-package-contract';
  const message = normalizeText(unsupported.message);
  const recommendation = normalizeText(unsupported.recommendation);
  throw new Error(
    `direct-source runtime: ${sourceKind} artifact "${packageBasename}" matches package profile ` +
    `"${profile.id}" but is not supported by Doppler direct-source import (${code}).` +
    (message ? ` ${message}` : '') +
    (recommendation ? ` ${recommendation}` : '')
  );
}

function findTFLiteMetadataEntry(parsedTFLite, name) {
  const target = normalizeText(name);
  if (!target) {
    return null;
  }
  const metadataEntries = Array.isArray(parsedTFLite?.metadataEntries)
    ? parsedTFLite.metadataEntries
    : [];
  return metadataEntries.find((entry) => normalizeText(entry?.name) === target) ?? null;
}

function computePackedByteSize(shape, sourceDtype, tensorName) {
  if (!Array.isArray(shape) || shape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${tensorName}" requires an explicit 2D expected shape.`
    );
  }
  const rows = Number(shape[0]);
  const cols = Number(shape[1]);
  if (!Number.isInteger(rows) || rows <= 0 || !Number.isInteger(cols) || cols <= 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${tensorName}" has invalid expected shape ${JSON.stringify(shape)}.`
    );
  }
  const elementCount = rows * cols;
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    return elementCount;
  }
  if (sourceDtype === 'INT4') {
    return Math.ceil(elementCount / 2);
  }
  if (sourceDtype === 'INT2') {
    return Math.ceil(elementCount / 4);
  }
  throw new Error(
    `direct-source runtime: unsupported packed source dtype "${sourceDtype}" for "${tensorName}".`
  );
}

function resolveLiteRTScaleContract(sourceDtype, tensorName, options = {}) {
  const explicitScaleSemantics = normalizeText(options.scaleSemantics).toLowerCase();
  if (explicitScaleSemantics) {
    if (explicitScaleSemantics === 'step') {
      return {
        scaleSemantics: 'step',
      };
    }
    if (explicitScaleSemantics === 'qmax_abs') {
      const explicitScaleDivisor = Number(options.scaleDivisor);
      if (Number.isFinite(explicitScaleDivisor) && explicitScaleDivisor > 0) {
        return {
          scaleSemantics: 'qmax_abs',
          scaleDivisor: explicitScaleDivisor,
        };
      }
      if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
        return {
          scaleSemantics: 'qmax_abs',
          scaleDivisor: 128,
        };
      }
      if (sourceDtype === 'INT4') {
        return {
          scaleSemantics: 'qmax_abs',
          scaleDivisor: 8,
        };
      }
      if (sourceDtype === 'INT2') {
        return {
          scaleSemantics: 'qmax_abs',
          scaleDivisor: 2,
        };
      }
      throw new Error(
        `direct-source runtime: unsupported LiteRT scale contract source dtype "${sourceDtype}" for "${tensorName}".`
      );
    }
    throw new Error(
      `direct-source runtime: unsupported LiteRT scaleSemantics "${options.scaleSemantics}" for "${tensorName}".`
    );
  }
  if (options.hasSumCompanion === true) {
    return {
      scaleSemantics: 'step',
    };
  }
  if (sourceDtype === 'INT8' || sourceDtype === 'UINT8') {
    return {
      scaleSemantics: 'qmax_abs',
      scaleDivisor: 128,
    };
  }
  if (sourceDtype === 'INT4') {
    return {
      scaleSemantics: 'qmax_abs',
      scaleDivisor: 8,
    };
  }
  if (sourceDtype === 'INT2') {
    return {
      scaleSemantics: 'qmax_abs',
      scaleDivisor: 2,
    };
  }
  throw new Error(
    `direct-source runtime: unsupported LiteRT scale contract source dtype "${sourceDtype}" for "${tensorName}".`
  );
}

function computeBlockedAxisPackedByteSize(storageShape, storageBlockSize, sourceDtype, tensorName) {
  if (!Array.isArray(storageShape) || storageShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${tensorName}" requires an explicit blocked storageShape=[rows, cols].`
    );
  }
  const storageRows = Number(storageShape[0]);
  const storageCols = Number(storageShape[1]);
  const blockSize = Number(storageBlockSize);
  if (
    !Number.isInteger(storageRows)
    || storageRows <= 0
    || !Number.isInteger(storageCols)
    || storageCols <= 0
    || !Number.isInteger(blockSize)
    || blockSize <= 0
  ) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${tensorName}" has invalid blocked storage shape ` +
      `${JSON.stringify({ storageShape, storageBlockSize })}.`
    );
  }
  return computePackedByteSize([storageRows, storageCols * blockSize], sourceDtype, tensorName);
}

function inferLiteRTBlockedAxisLayout(
  rawTensor,
  storageShape,
  storageBlockSize,
  tensorName = rawTensor?.name ?? 'unknown',
  options = {}
) {
  const dtypeId = Number(rawTensor?.dtypeId);
  const candidates = [];
  if (dtypeId === 17) {
    candidates.push('INT4');
  } else if (dtypeId === 9) {
    candidates.push('INT8', 'INT4', 'INT2');
  } else if (dtypeId === 3) {
    candidates.push('UINT8', 'INT4', 'INT2');
  } else {
    throw new Error(
      `direct-source runtime: unsupported LiteRT blocked tensor dtype for "${tensorName}" (dtypeId=${dtypeId}).`
    );
  }

  for (const sourceDtype of candidates) {
    if (rawTensor.size === computeBlockedAxisPackedByteSize(storageShape, storageBlockSize, sourceDtype, tensorName)) {
      const preferSignedPacked = options.preferSignedPacked === true;
      return {
        sourceDtype,
        storageEncoding: sourceDtype === 'INT8' || sourceDtype === 'UINT8'
          ? 'signed'
          : (preferSignedPacked ? 'signed' : 'offset_binary'),
      };
    }
  }

  throw new Error(
    `direct-source runtime: LiteRT tensor "${tensorName}" size ${rawTensor?.size} does not match any supported ` +
    `blocked packed layout for storage shape ${JSON.stringify(storageShape)} and blockSize=${storageBlockSize}.`
  );
}

export function inferLiteRTRowwiseLayout(
  rawTensor,
  expectedShape,
  tensorName = rawTensor?.name ?? 'unknown',
  options = {}
) {
  const dtypeId = Number(rawTensor?.dtypeId);
  const candidates = [];
  if (dtypeId === 17) {
    candidates.push('INT4');
  } else if (dtypeId === 9) {
    candidates.push('INT8', 'INT4', 'INT2');
  } else if (dtypeId === 3) {
    candidates.push('UINT8', 'INT4', 'INT2');
  } else {
    throw new Error(
      `direct-source runtime: unsupported LiteRT packed tensor dtype for "${tensorName}" (dtypeId=${dtypeId}).`
    );
  }

  for (const sourceDtype of candidates) {
    if (rawTensor.size === computePackedByteSize(expectedShape, sourceDtype, tensorName)) {
      const preferSignedPacked = options.preferSignedPacked === true;
      return {
        sourceDtype,
        storageEncoding: sourceDtype === 'INT8' || sourceDtype === 'UINT8'
          ? 'signed'
          : (preferSignedPacked ? 'signed' : 'offset_binary'),
      };
    }
  }

  throw new Error(
    `direct-source runtime: LiteRT tensor "${tensorName}" size ${rawTensor?.size} does not match any supported ` +
    `packed layout for expected shape ${JSON.stringify(expectedShape)}.`
  );
}

function isGemma4GlobalLayer(runtimeProfile, layerIndex) {
  const layerPattern = runtimeProfile?.manifestInference?.layerPattern ?? null;
  if (!layerPattern || layerPattern.type !== 'every_n') {
    return false;
  }
  const period = Number(layerPattern.period);
  const rawOffset = Number(layerPattern.offset ?? 0);
  if (!Number.isInteger(period) || period <= 0) {
    return false;
  }
  const offset = ((rawOffset % period) + period) % period;
  return (((layerIndex - offset) % period) + period) % period === 0;
}

export function resolveGemma4AttentionHeadDim(runtimeProfile, layerIndex) {
  const headDim = Number(runtimeProfile?.architecture?.headDim ?? 0);
  const globalHeadDim = Number(runtimeProfile?.architecture?.globalHeadDim ?? headDim);
  if (!Number.isInteger(headDim) || headDim <= 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT profile is missing architecture.headDim.');
  }
  if (!Number.isInteger(globalHeadDim) || globalHeadDim <= 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT profile is missing architecture.globalHeadDim.');
  }
  return isGemma4GlobalLayer(runtimeProfile, layerIndex) ? globalHeadDim : headDim;
}

function resolveGemma4IntermediateSize(runtimeProfile, layerIndex) {
  const arch = runtimeProfile?.architecture ?? {};
  const numLayers = Number(arch.numLayers ?? 0);
  const intermediateSize = Number(arch.intermediateSize ?? 0);
  const numKvSharedLayers = Number(arch.numKvSharedLayers ?? 0);
  const useDoubleWideMlp = runtimeProfile?.manifestInference?.ffn?.useDoubleWideMlp === true;
  if (!Number.isInteger(intermediateSize) || intermediateSize <= 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT profile is missing architecture.intermediateSize.');
  }
  if (
    useDoubleWideMlp
    && Number.isInteger(numLayers)
    && numLayers > 0
    && Number.isInteger(numKvSharedLayers)
    && numKvSharedLayers > 0
    && layerIndex >= numLayers - numKvSharedLayers
  ) {
    return intermediateSize * 2;
  }
  return intermediateSize;
}

function createLiteRTFloatTensor(rawTensor, sourcePath, canonicalName, role, group = null) {
  if (!rawTensor || typeof rawTensor !== 'object') {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  if (rawTensor.size % 4 !== 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" must have a float32 byte size.`
    );
  }
  return {
    name: canonicalName,
    shape: [rawTensor.size / 4],
    dtype: 'F32',
    offset: rawTensor.offset,
    size: rawTensor.size,
    sourcePath,
    role,
    ...(group ? { group } : {}),
  };
}

function shapeEquals(actualShape, expectedShape) {
  if (!Array.isArray(actualShape) || !Array.isArray(expectedShape)) {
    return false;
  }
  if (actualShape.length !== expectedShape.length) {
    return false;
  }
  for (let index = 0; index < expectedShape.length; index += 1) {
    if (Number(actualShape[index]) !== Number(expectedShape[index])) {
      return false;
    }
  }
  return true;
}

function isLiteRTTensorDtype(rawTensor, sourceDtype) {
  const dtypeId = Number(rawTensor?.dtypeId);
  if (sourceDtype === 'F32') {
    return dtypeId === 0 || normalizeText(rawTensor?.dtype).toUpperCase() === 'F32';
  }
  if (sourceDtype === 'INT4') {
    return dtypeId === 17 || normalizeText(rawTensor?.sourceDtype).toUpperCase() === 'INT4';
  }
  return false;
}

function createGemma412BSplitCursor(parsedTFLite, sourcePath) {
  return {
    tensors: Array.isArray(parsedTFLite?.tensors) ? parsedTFLite.tensors : [],
    sourcePath,
    index: 0,
    lastHiddenNorm: null,
  };
}

function takeNextLiteRTTensor(cursor, expectedShape, sourceDtype, label, options = {}) {
  const optional = options.optional === true;
  for (let tensorIndex = cursor.index; tensorIndex < cursor.tensors.length; tensorIndex += 1) {
    const tensor = cursor.tensors[tensorIndex];
    if (!isLiteRTTensorDtype(tensor, sourceDtype)) {
      continue;
    }
    if (!shapeEquals(tensor.shape, expectedShape)) {
      continue;
    }
    cursor.index = tensorIndex + 1;
    return tensor;
  }
  if (optional) {
    return null;
  }
  throw new Error(
    `direct-source runtime: Gemma 4 12B split-section adapter could not find ${sourceDtype} tensor ` +
    `${label} with shape ${JSON.stringify(expectedShape)} after tensor index ${cursor.index}.`
  );
}

function takeOptionalConsecutiveLiteRTTensor(cursor, expectedShape, sourceDtype) {
  const tensor = cursor.tensors[cursor.index] ?? null;
  if (!isLiteRTTensorDtype(tensor, sourceDtype) || !shapeEquals(tensor.shape, expectedShape)) {
    return null;
  }
  cursor.index += 1;
  return tensor;
}

function collectGemma412BLayerScalarTensors(prefillTFLite, numLayers, packageConfig) {
  const expectedLayers = normalizeLayerIndexList(
    packageConfig?.layerScalarLayers,
    numLayers,
    'package.litertlm.layerScalarLayers'
  );
  if (expectedLayers.length !== numLayers) {
    const missingValue = Number(packageConfig?.missingLayerScalarValue);
    if (!Number.isFinite(missingValue) || missingValue !== 1) {
      throw new Error(
        'direct-source runtime: package.litertlm.missingLayerScalarValue must be 1 when ' +
        'package.litertlm.layerScalarLayers does not cover every layer.'
      );
    }
  }

  const expected = new Set(expectedLayers);
  const layerScalars = new Map();
  for (const tensor of prefillTFLite.tensors ?? []) {
    if (!isLiteRTTensorDtype(tensor, 'F32') || !shapeEquals(tensor.shape, [])) {
      continue;
    }
    const match = String(tensor.name ?? '').match(/Gemma4UnifiedTextDecoderLayer_(\d+);?$/);
    if (!match) {
      continue;
    }
    const layerIndex = Number(match[1]);
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= numLayers) {
      throw new Error(
        `direct-source runtime: Gemma 4 12B layer_scalar tensor "${tensor.name}" ` +
        `has out-of-range layer index ${String(match[1])}.`
      );
    }
    if (!expected.has(layerIndex)) {
      throw new Error(
        `direct-source runtime: Gemma 4 12B layer_scalar tensor for layer ${layerIndex} ` +
        'is present in the LiteRT graph but missing from package.litertlm.layerScalarLayers.'
      );
    }
    if (layerScalars.has(layerIndex)) {
      throw new Error(
        `direct-source runtime: Gemma 4 12B LiteRT graph has duplicate layer_scalar tensors for layer ${layerIndex}.`
      );
    }
    layerScalars.set(layerIndex, tensor);
  }

  for (const layerIndex of expectedLayers) {
    if (!layerScalars.has(layerIndex)) {
      throw new Error(
        `direct-source runtime: Gemma 4 12B split-section adapter expected layer_scalar for layer ${layerIndex}, ` +
        'but the LiteRT graph did not contain a matching Gemma4UnifiedTextDecoderLayer scalar tensor.'
      );
    }
  }
  return layerScalars;
}

function createLiteRTFixedAffineInt4Tensor(
  rawTensor,
  sourcePath,
  canonicalName,
  role,
  group,
  logicalShape,
  fixedScale,
  storageEncoding
) {
  if (!rawTensor || typeof rawTensor !== 'object') {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  if (!Array.isArray(logicalShape) || logicalShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires an explicit 2D logical shape.`
    );
  }
  if (!shapeEquals(rawTensor.shape, logicalShape)) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" shape ${JSON.stringify(rawTensor.shape)} ` +
      `does not match expected split-section logical shape ${JSON.stringify(logicalShape)} for "${canonicalName}".`
    );
  }
  const expectedSize = computePackedByteSize(logicalShape, 'INT4', canonicalName);
  if (Number(rawTensor.size) !== expectedSize) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" size ${rawTensor.size} does not match ` +
      `INT4 packed shape ${JSON.stringify(logicalShape)} for "${canonicalName}".`
    );
  }
  if (!Number.isFinite(fixedScale) || fixedScale <= 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires package.litertlm.fixedInt4Scale > 0.`
    );
  }
  return {
    name: canonicalName,
    shape: [Number(logicalShape[0]), Number(logicalShape[1])],
    dtype: 'F16',
    offset: rawTensor.offset,
    size: rawTensor.size,
    sourcePath,
    role,
    ...(group ? { group } : {}),
    sourceTransform: {
      kind: 'affine_dequant',
      scheme: 'per_tensor_affine',
      sourceDtype: 'INT4',
      targetDtype: 'F16',
      storageEncoding,
      scale: fixedScale,
      zeroPoint: 0,
    },
  };
}

function resolveGemma412BLayerTypes(runtimeProfile) {
  const explicitLayerTypes = runtimeProfile?.manifestInference?.layerPattern?.layerTypes;
  if (Array.isArray(explicitLayerTypes) && explicitLayerTypes.length > 0) {
    return explicitLayerTypes.map((value) => normalizeText(value));
  }
  const numLayers = Number(runtimeProfile?.architecture?.numLayers ?? 0);
  if (!Number.isInteger(numLayers) || numLayers <= 0) {
    throw new Error('direct-source runtime: Gemma 4 12B split-section adapter is missing architecture.numLayers.');
  }
  const layerTypes = [];
  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    layerTypes.push(isGemma4GlobalLayer(runtimeProfile, layerIndex) ? 'full_attention' : 'sliding_attention');
  }
  return layerTypes;
}

function addFloatToNormalized(normalized, rawTensor, sourcePath, canonicalName, role, group = null) {
  if (!rawTensor) {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  normalized.push(createLiteRTFloatTensor(rawTensor, sourcePath, canonicalName, role, group));
}

function createLiteRTRowwiseTensor(
  rawTensor,
  scaleTensor,
  sumTensor,
  sourcePath,
  canonicalName,
  role,
  group = null,
  expectedShape = null
) {
  if (!rawTensor || typeof rawTensor !== 'object') {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  if (!scaleTensor || typeof scaleTensor !== 'object') {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" is missing row-scale companion "${rawTensor.name}_quantized_scale".`
    );
  }
  if (scaleTensor.size % 4 !== 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid row-scale size ${scaleTensor.size}.`
    );
  }

  const rowsFromScale = scaleTensor.size / 4;
  if (!Number.isInteger(rowsFromScale) || rowsFromScale <= 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid row count ${rowsFromScale}.`
    );
  }
  if (sumTensor && typeof sumTensor === 'object') {
    if (sumTensor.size % 4 !== 0) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid row-sum size ${sumTensor.size}.`
      );
    }
    const rowsFromSum = sumTensor.size / 4;
    if (rowsFromSum !== rowsFromScale) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" row-sum count ${rowsFromSum} ` +
        `does not match row-scale count ${rowsFromScale}.`
      );
    }
  }
  const resolvedShape = Array.isArray(expectedShape) && expectedShape.length === 2
    ? expectedShape
    : null;
  const rows = resolvedShape ? Number(resolvedShape[0]) : rowsFromScale;
  const cols = resolvedShape ? Number(resolvedShape[1]) : null;
  if (!Number.isInteger(rows) || rows <= 0 || (resolvedShape && rows !== rowsFromScale)) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" row-scale count ${rowsFromScale} ` +
      `does not match expected rows ${resolvedShape?.[0] ?? rows}.`
    );
  }
  const layout = resolvedShape
    ? inferLiteRTRowwiseLayout(rawTensor, resolvedShape, canonicalName, {
      preferSignedPacked: !(sumTensor && typeof sumTensor === 'object'),
    })
    : inferLiteRTRowwiseLayout(rawTensor, [rowsFromScale, rawTensor.size / rowsFromScale], canonicalName, {
      preferSignedPacked: !(sumTensor && typeof sumTensor === 'object'),
    });
  const resolvedCols = resolvedShape ? cols : Math.floor(rawTensor.size / rowsFromScale);
  const scaleContract = resolveLiteRTScaleContract(layout.sourceDtype, canonicalName, {
    hasSumCompanion: Boolean(sumTensor && typeof sumTensor === 'object'),
  });
  return {
    name: canonicalName,
    shape: [rows, resolvedCols],
    dtype: 'F16',
    offset: rawTensor.offset,
    size: rawTensor.size,
    sourcePath,
    role,
    ...(group ? { group } : {}),
    sourceTransform: {
      kind: 'litert_rowwise_dequant',
      scheme: 'per_row_affine',
      sourceDtype: layout.sourceDtype,
      targetDtype: 'F16',
      storageEncoding: layout.storageEncoding,
      scaleSemantics: scaleContract.scaleSemantics,
      scaleDivisor: scaleContract.scaleDivisor,
      scaleSourcePath: sourcePath,
      scaleOffset: scaleTensor.offset,
      scaleSize: scaleTensor.size,
      ...(sumTensor && typeof sumTensor === 'object'
        ? {
          rowSumSourcePath: sourcePath,
          rowSumOffset: sumTensor.offset,
          rowSumSize: sumTensor.size,
        }
        : {}),
    },
  };
}

function createLiteRTAxisTensor(
  rawTensor,
  scaleTensor,
  sumTensor,
  sourcePath,
  canonicalName,
  role,
  group = null,
  logicalShape = null,
  storageShape = null,
  quantAxis = 1,
  scaleContractOptions = null
) {
  if (!rawTensor || typeof rawTensor !== 'object') {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  if (!scaleTensor || typeof scaleTensor !== 'object') {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" is missing scale companion ` +
      `"${rawTensor.name}_quantized_scale".`
    );
  }
  if (!Array.isArray(logicalShape) || logicalShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires an explicit 2D logical shape.`
    );
  }
  if (!Array.isArray(storageShape) || storageShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires an explicit 2D storage shape.`
    );
  }
  if (quantAxis !== 0 && quantAxis !== 1) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" has unsupported quantAxis ${quantAxis}.`
    );
  }
  if (scaleTensor.size % 4 !== 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid scale size ${scaleTensor.size}.`
    );
  }

  const scaleTensorSourceTransform = scaleTensor.sourceTransform;
  const scaleCompanionDtype = String(scaleTensor.sourceDtype || '').toUpperCase();
  const hasUint8ScaleCompanion = scaleCompanionDtype === 'UINT8';

  // When the scale companion is UINT8 without affine_dequant metadata, the
  // UINT8 container holds packed F32 row-scales (4 bytes per F32 value).
  // This is the MediaPipe symmetric quantization convention used by LiteRT-LM
  // .task weight bags (e.g. Gemma 4 E2B per_layer_embeddings): weight bytes
  // are semantically INT8 (signed, zero_point=0), scale = max(|row|) / 127.
  // The companion bytes are native F32 — no affine dequant needed.
  const isPackedF32ScaleCompanion = hasUint8ScaleCompanion
    && (!scaleTensorSourceTransform || scaleTensorSourceTransform.kind !== 'affine_dequant');

  const scaleCompanionDequant = hasUint8ScaleCompanion && !isPackedF32ScaleCompanion
    ? {
      scale: Number(scaleTensorSourceTransform?.scale),
      zeroPoint: Number(scaleTensorSourceTransform?.zeroPoint),
    }
    : null;
  if (hasUint8ScaleCompanion && !isPackedF32ScaleCompanion) {
    if (!Number.isFinite(scaleCompanionDequant.scale) || scaleCompanionDequant.scale <= 0) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid scale companion affine_dequant scale ${scaleTensorSourceTransform.scale}.`
      );
    }
    if (!Number.isSafeInteger(scaleCompanionDequant.zeroPoint)) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid scale companion affine_dequant zeroPoint ${scaleTensorSourceTransform.zeroPoint}.`
      );
    }
  }

  const logicalRows = Number(logicalShape[0]);
  const logicalCols = Number(logicalShape[1]);
  const storageRows = Number(storageShape[0]);
  const storageCols = Number(storageShape[1]);
  if (
    !Number.isInteger(logicalRows)
    || logicalRows <= 0
    || !Number.isInteger(logicalCols)
    || logicalCols <= 0
    || !Number.isInteger(storageRows)
    || storageRows <= 0
    || !Number.isInteger(storageCols)
    || storageCols <= 0
  ) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" has invalid logical/storage shapes ` +
      `${JSON.stringify({ logicalShape, storageShape })}.`
    );
  }

  const layout = inferLiteRTRowwiseLayout(rawTensor, storageShape, canonicalName, {
    preferSignedPacked: !(sumTensor && typeof sumTensor === 'object'),
  });
  const scaleContract = resolveLiteRTScaleContract(layout.sourceDtype, canonicalName, {
    hasSumCompanion: Boolean(sumTensor && typeof sumTensor === 'object'),
    ...(scaleContractOptions && typeof scaleContractOptions === 'object'
      ? scaleContractOptions
      : {}),
  });
  const expectedScaleCount = quantAxis === 0 ? storageCols : storageRows;
  const scaleCount = scaleTensor.size / 4;
  if (scaleCount !== expectedScaleCount || scaleCount !== logicalRows) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" scale count ${scaleCount} ` +
      `does not match logical rows ${logicalRows} and expected storage-axis count ${expectedScaleCount}.`
    );
  }

  if (sumTensor && typeof sumTensor === 'object') {
    if (sumTensor.size % 4 !== 0) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid sum size ${sumTensor.size}.`
      );
    }
    const sumCount = sumTensor.size / 4;
    if (sumCount !== logicalRows) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" sum count ${sumCount} ` +
        `does not match logical rows ${logicalRows}.`
      );
    }
  }

  return {
    name: canonicalName,
    shape: [logicalRows, logicalCols],
    dtype: 'F16',
    offset: rawTensor.offset,
    size: rawTensor.size,
    sourcePath,
    role,
    ...(group ? { group } : {}),
    sourceTransform: {
      kind: 'litert_axis_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: isPackedF32ScaleCompanion && layout.sourceDtype === 'UINT8'
        ? 'INT8'
        : layout.sourceDtype,
      targetDtype: 'F16',
      storageEncoding: layout.storageEncoding,
      scaleSemantics: scaleContract.scaleSemantics,
      scaleDivisor: scaleContract.scaleDivisor,
      storageShape: [storageRows, storageCols],
      quantAxis,
      scaleSourcePath: sourcePath,
      scaleOffset: scaleTensor.offset,
      scaleSize: scaleTensor.size,
      ...(hasUint8ScaleCompanion && !isPackedF32ScaleCompanion
        ? {
          scaleCompanionDtype,
          scaleCompanionDequant,
        }
        : {}),
      ...(sumTensor && typeof sumTensor === 'object'
        ? {
          sumSourcePath: sourcePath,
          sumOffset: sumTensor.offset,
          sumSize: sumTensor.size,
        }
        : {}),
    },
  };
}

function createLiteRTBlockedAxisTensor(
  rawTensor,
  scaleTensor,
  sumTensor,
  sourcePath,
  canonicalName,
  role,
  group = null,
  logicalShape = null,
  storageShape = null,
  quantAxis = 0,
  storageBlockSize = 4,
  storageLaneOrder = null,
  scaleContractOptions = null
) {
  if (!rawTensor || typeof rawTensor !== 'object') {
    throw new Error(`direct-source runtime: missing LiteRT tensor "${canonicalName}".`);
  }
  if (!scaleTensor || typeof scaleTensor !== 'object') {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" is missing scale companion ` +
      `"${rawTensor.name}_quantized_scale".`
    );
  }
  if (!Array.isArray(logicalShape) || logicalShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires an explicit 2D logical shape.`
    );
  }
  if (!Array.isArray(storageShape) || storageShape.length !== 2) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" requires an explicit 2D blocked storage shape.`
    );
  }
  if (quantAxis !== 0) {
    throw new Error(
      `direct-source runtime: LiteRT blocked tensor "${canonicalName}" only supports quantAxis=0.`
    );
  }
  if (scaleTensor.size % 4 !== 0) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid blocked scale size ${scaleTensor.size}.`
    );
  }

  const logicalRows = Number(logicalShape[0]);
  const logicalCols = Number(logicalShape[1]);
  const storageRows = Number(storageShape[0]);
  const storageCols = Number(storageShape[1]);
  const blockSize = Number(storageBlockSize);
  if (
    !Number.isInteger(logicalRows)
    || logicalRows <= 0
    || !Number.isInteger(logicalCols)
    || logicalCols <= 0
    || !Number.isInteger(storageRows)
    || storageRows <= 0
    || !Number.isInteger(storageCols)
    || storageCols <= 0
    || !Number.isInteger(blockSize)
    || blockSize <= 0
  ) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${canonicalName}" has invalid logical/blocked storage shapes ` +
      `${JSON.stringify({ logicalShape, storageShape, storageBlockSize })}.`
    );
  }
  if (storageCols !== logicalRows || storageRows * blockSize !== logicalCols) {
    throw new Error(
      `direct-source runtime: LiteRT blocked tensor "${canonicalName}" expects storageShape ` +
      `[${logicalCols / blockSize}, ${logicalRows}] for logical shape [${logicalRows}, ${logicalCols}] and blockSize=${blockSize}. ` +
      `Got [${storageRows}, ${storageCols}].`
    );
  }

  const resolvedLaneOrder = Array.isArray(storageLaneOrder) && storageLaneOrder.length > 0
    ? storageLaneOrder.map((value) => Number(value))
    : Array.from({ length: blockSize }, (_value, index) => index);
  if (
    resolvedLaneOrder.length !== blockSize
    || resolvedLaneOrder.some((value) => !Number.isInteger(value) || value < 0 || value >= blockSize)
    || new Set(resolvedLaneOrder).size !== blockSize
  ) {
    throw new Error(
      `direct-source runtime: LiteRT blocked tensor "${canonicalName}" has invalid storageLaneOrder ${JSON.stringify(storageLaneOrder)}.`
    );
  }

  const layout = inferLiteRTBlockedAxisLayout(rawTensor, storageShape, blockSize, canonicalName, {
    preferSignedPacked: !(sumTensor && typeof sumTensor === 'object'),
  });
  const scaleContract = resolveLiteRTScaleContract(layout.sourceDtype, canonicalName, {
    hasSumCompanion: Boolean(sumTensor && typeof sumTensor === 'object'),
    ...(scaleContractOptions && typeof scaleContractOptions === 'object'
      ? scaleContractOptions
      : {}),
  });
  const scaleCount = scaleTensor.size / 4;
  if (scaleCount !== logicalRows) {
    throw new Error(
      `direct-source runtime: LiteRT tensor "${rawTensor.name}" blocked scale count ${scaleCount} ` +
      `does not match logical rows ${logicalRows}.`
    );
  }

  if (sumTensor && typeof sumTensor === 'object') {
    if (sumTensor.size % 4 !== 0) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" has invalid blocked sum size ${sumTensor.size}.`
      );
    }
    const sumCount = sumTensor.size / 4;
    if (sumCount !== logicalRows) {
      throw new Error(
        `direct-source runtime: LiteRT tensor "${rawTensor.name}" blocked sum count ${sumCount} ` +
        `does not match logical rows ${logicalRows}.`
      );
    }
  }

  return {
    name: canonicalName,
    shape: [logicalRows, logicalCols],
    dtype: 'F16',
    offset: rawTensor.offset,
    size: rawTensor.size,
    sourcePath,
    role,
    ...(group ? { group } : {}),
    sourceTransform: {
      kind: 'litert_axis_blocked_dequant',
      scheme: 'per_axis_affine',
      sourceDtype: layout.sourceDtype,
      targetDtype: 'F16',
      storageEncoding: layout.storageEncoding,
      scaleSemantics: scaleContract.scaleSemantics,
      scaleDivisor: scaleContract.scaleDivisor,
      storageShape: [storageRows, storageCols],
      quantAxis,
      storageBlockSize: blockSize,
      storageLaneOrder: resolvedLaneOrder,
      scaleSourcePath: sourcePath,
      scaleOffset: scaleTensor.offset,
      scaleSize: scaleTensor.size,
      ...(sumTensor && typeof sumTensor === 'object'
        ? {
          sumSourcePath: sourcePath,
          sumOffset: sumTensor.offset,
          sumSize: sumTensor.size,
        }
        : {}),
    },
  };
}

function normalizeGemma412BSplitLiteRTTensors(parsedTFLitesByPath, virtualFiles, runtimeProfile, packageConfig) {
  const embedderModelType = normalizeText(packageConfig?.embedderTFLiteModelType);
  const prefillModelType = normalizeText(packageConfig?.tfliteModelType);
  const embedderSourceFile = virtualFiles.find((entry) => entry.kind === 'tflite_embedder_model') ?? null;
  const prefillSourceFile = virtualFiles.find((entry) => entry.kind === 'tflite_model') ?? null;
  if (!embedderModelType || !embedderSourceFile) {
    throw new Error(
      'direct-source runtime: Gemma 4 12B split-section adapter requires package.litertlm.embedderTFLiteModelType.'
    );
  }
  if (!prefillModelType || !prefillSourceFile) {
    throw new Error(
      'direct-source runtime: Gemma 4 12B split-section adapter requires package.litertlm.tfliteModelType.'
    );
  }
  const embedderTFLite = parsedTFLitesByPath?.get(embedderSourceFile.path) ?? null;
  const prefillTFLite = parsedTFLitesByPath?.get(prefillSourceFile.path) ?? null;
  if (!embedderTFLite || !prefillTFLite) {
    throw new Error('direct-source runtime: Gemma 4 12B split-section adapter did not receive both parsed TFLite sections.');
  }

  const arch = runtimeProfile?.architecture ?? {};
  const hiddenSize = Number(arch.hiddenSize ?? 0);
  const vocabSize = Number(arch.vocabSize ?? 0);
  const intermediateSize = Number(arch.intermediateSize ?? 0);
  const numAttentionHeads = Number(arch.numAttentionHeads ?? 0);
  const numKeyValueHeads = Number(arch.numKeyValueHeads ?? 0);
  const numGlobalKeyValueHeads = Number(arch.numGlobalKeyValueHeads ?? numKeyValueHeads);
  const headDim = Number(arch.headDim ?? 0);
  const globalHeadDim = Number(arch.globalHeadDim ?? headDim);
  const fixedInt4Scale = Number(packageConfig?.fixedInt4Scale);
  const fixedInt4StorageEncoding = normalizeLiteRTStorageEncoding(
    packageConfig?.fixedInt4StorageEncoding,
    'package.litertlm.fixedInt4StorageEncoding'
  );
  if (
    !Number.isInteger(hiddenSize)
    || hiddenSize <= 0
    || !Number.isInteger(vocabSize)
    || vocabSize <= 0
    || !Number.isInteger(intermediateSize)
    || intermediateSize <= 0
    || !Number.isInteger(numAttentionHeads)
    || numAttentionHeads <= 0
    || !Number.isInteger(numKeyValueHeads)
    || numKeyValueHeads <= 0
    || !Number.isInteger(numGlobalKeyValueHeads)
    || numGlobalKeyValueHeads <= 0
    || !Number.isInteger(headDim)
    || headDim <= 0
    || !Number.isInteger(globalHeadDim)
    || globalHeadDim <= 0
    || !Number.isFinite(fixedInt4Scale)
    || fixedInt4Scale <= 0
  ) {
    throw new Error(
      'direct-source runtime: Gemma 4 12B split-section adapter requires explicit architecture sizes and fixedInt4Scale.'
    );
  }

  const normalized = [];
  const embedderTensor = embedderTFLite.tensors.find((tensor) => (
    isLiteRTTensorDtype(tensor, 'INT4') && shapeEquals(tensor.shape, [vocabSize, hiddenSize])
  )) ?? null;
  normalized.push(
    createLiteRTFixedAffineInt4Tensor(
      embedderTensor,
      embedderSourceFile.path,
      'model.language_model.embed_tokens.weight',
      'embedding',
      'embed',
      [vocabSize, hiddenSize],
      fixedInt4Scale,
      fixedInt4StorageEncoding
    )
  );

  const cursor = createGemma412BSplitCursor(prefillTFLite, prefillSourceFile.path);
  const leadingFinalNorm = takeNextLiteRTTensor(cursor, [hiddenSize], 'F32', 'model.language_model.norm.weight', {
    optional: true,
  });
  if (leadingFinalNorm) {
    cursor.lastHiddenNorm = leadingFinalNorm;
    takeOptionalConsecutiveLiteRTTensor(cursor, [numGlobalKeyValueHeads * globalHeadDim, hiddenSize], 'INT4');
    takeOptionalConsecutiveLiteRTTensor(cursor, [globalHeadDim], 'F32');
  }
  const layerTypes = resolveGemma412BLayerTypes(runtimeProfile);
  const layerScalarTensors = collectGemma412BLayerScalarTensors(
    prefillTFLite,
    layerTypes.length,
    packageConfig
  );
  for (let layerIndex = 0; layerIndex < layerTypes.length; layerIndex += 1) {
    const canonicalLayerPrefix = `model.language_model.layers.${layerIndex}`;
    const globalLayer = normalizeText(layerTypes[layerIndex]) === 'full_attention';
    const attentionHeadDim = globalLayer ? globalHeadDim : headDim;
    const kvHeads = globalLayer ? numGlobalKeyValueHeads : numKeyValueHeads;
    const qRows = numAttentionHeads * attentionHeadDim;
    const kvRows = kvHeads * attentionHeadDim;

    const addHiddenNorm = (canonicalName) => {
      const rawTensor = takeNextLiteRTTensor(cursor, [hiddenSize], 'F32', canonicalName);
      cursor.lastHiddenNorm = rawTensor;
      normalized.push(createLiteRTFloatTensor(rawTensor, prefillSourceFile.path, canonicalName, 'norm'));
    };
    const addFloat = (rawTensor, canonicalName, role = 'norm') => {
      if (!rawTensor) return;
      normalized.push(createLiteRTFloatTensor(rawTensor, prefillSourceFile.path, canonicalName, role));
    };
    const addInt4 = (rawTensor, canonicalName, role, logicalShape) => {
      normalized.push(
        createLiteRTFixedAffineInt4Tensor(
          rawTensor,
          prefillSourceFile.path,
          canonicalName,
          role,
          null,
          logicalShape,
          fixedInt4Scale,
          fixedInt4StorageEncoding
        )
      );
    };

    addFloat(layerScalarTensors.get(layerIndex) ?? null, `${canonicalLayerPrefix}.layer_scalar`, 'other');
    addHiddenNorm(`${canonicalLayerPrefix}.input_layernorm.weight`);
    addHiddenNorm(`${canonicalLayerPrefix}.post_attention_layernorm.weight`);
    addHiddenNorm(`${canonicalLayerPrefix}.pre_feedforward_layernorm.weight`);
    addHiddenNorm(`${canonicalLayerPrefix}.post_feedforward_layernorm.weight`);

    addInt4(
      takeNextLiteRTTensor(cursor, [hiddenSize, intermediateSize], 'INT4', `${canonicalLayerPrefix}.mlp.down_proj.weight`),
      `${canonicalLayerPrefix}.mlp.down_proj.weight`,
      'matmul',
      [hiddenSize, intermediateSize]
    );
    addInt4(
      takeNextLiteRTTensor(cursor, [intermediateSize, hiddenSize], 'INT4', `${canonicalLayerPrefix}.mlp.gate_proj.weight`),
      `${canonicalLayerPrefix}.mlp.gate_proj.weight`,
      'matmul',
      [intermediateSize, hiddenSize]
    );
    addInt4(
      takeNextLiteRTTensor(cursor, [intermediateSize, hiddenSize], 'INT4', `${canonicalLayerPrefix}.mlp.up_proj.weight`),
      `${canonicalLayerPrefix}.mlp.up_proj.weight`,
      'matmul',
      [intermediateSize, hiddenSize]
    );
    addInt4(
      takeNextLiteRTTensor(cursor, [hiddenSize, qRows], 'INT4', `${canonicalLayerPrefix}.self_attn.o_proj.weight`),
      `${canonicalLayerPrefix}.self_attn.o_proj.weight`,
      'matmul',
      [hiddenSize, qRows]
    );
    const kProj = takeNextLiteRTTensor(cursor, [kvRows, hiddenSize], 'INT4', `${canonicalLayerPrefix}.self_attn.k_proj.weight`);
    const vProj = takeOptionalConsecutiveLiteRTTensor(cursor, [kvRows, hiddenSize], 'INT4') ?? kProj;
    addInt4(kProj, `${canonicalLayerPrefix}.self_attn.k_proj.weight`, 'matmul', [kvRows, hiddenSize]);
    addInt4(vProj, `${canonicalLayerPrefix}.self_attn.v_proj.weight`, 'matmul', [kvRows, hiddenSize]);

    const firstNorm = takeOptionalConsecutiveLiteRTTensor(cursor, [attentionHeadDim], 'F32');
    const secondNorm = takeOptionalConsecutiveLiteRTTensor(cursor, [attentionHeadDim], 'F32');
    if (firstNorm) {
      addFloat(firstNorm, `${canonicalLayerPrefix}.self_attn.q_norm.weight`);
      addFloat(secondNorm ?? firstNorm, `${canonicalLayerPrefix}.self_attn.k_norm.weight`);
    }

    addInt4(
      takeNextLiteRTTensor(cursor, [qRows, hiddenSize], 'INT4', `${canonicalLayerPrefix}.self_attn.q_proj.weight`),
      `${canonicalLayerPrefix}.self_attn.q_proj.weight`,
      'matmul',
      [qRows, hiddenSize]
    );
  }

  const finalNorm = takeNextLiteRTTensor(cursor, [hiddenSize], 'F32', 'model.language_model.norm.weight', {
    optional: true,
  }) ?? leadingFinalNorm ?? cursor.lastHiddenNorm;
  addFloatToNormalized(normalized, finalNorm, prefillSourceFile.path, 'model.language_model.norm.weight', 'norm', 'head');

  if (normalized.length === 0) {
    throw new Error('direct-source runtime: Gemma 4 12B split-section adapter did not produce any normalized tensors.');
  }
  return normalized;
}

function normalizeGemma4LiteRTTensors(parsedTFLite, sourcePath, runtimeProfile) {
  const rawByName = new Map();
  for (const tensor of parsedTFLite.tensors) {
    rawByName.set(tensor.name, tensor);
  }
  const numLayers = Number(runtimeProfile?.architecture?.numLayers ?? 0);
  if (!Number.isInteger(numLayers) || numLayers <= 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT profile is missing architecture.numLayers.');
  }
  const hiddenSize = Number(runtimeProfile?.architecture?.hiddenSize ?? 0);
  const hiddenSizePerLayerInput = Number(runtimeProfile?.architecture?.hiddenSizePerLayerInput ?? 0);
  const vocabSize = Number(runtimeProfile?.architecture?.vocabSize ?? 0);
  const vocabSizePerLayerInput = Number(runtimeProfile?.architecture?.vocabSizePerLayerInput ?? 0);
  const numAttentionHeads = Number(runtimeProfile?.architecture?.numAttentionHeads ?? 0);
  const numKeyValueHeads = Number(runtimeProfile?.architecture?.numKeyValueHeads ?? 0);

  const normalized = [];
  const addFloat = (rawName, canonicalName, role, group = null) => {
    const rawTensor = rawByName.get(rawName) ?? null;
    if (!rawTensor) return;
    normalized.push(createLiteRTFloatTensor(rawTensor, sourcePath, canonicalName, role, group));
  };
  const addAxisQuantized = (
    rawName,
    canonicalName,
    role,
    group = null,
    logicalShape = null,
    options = {}
  ) => {
    const rawTensor = rawByName.get(rawName) ?? null;
    if (!rawTensor) return;
    const scaleTensor = rawByName.get(`${rawName}_quantized_scale`) ?? null;
    const sumTensor = rawByName.get(`${rawName}.sum_i`) ?? null;
    const resolvedLogicalShape = Array.isArray(logicalShape) && logicalShape.length === 2
      ? logicalShape
      : null;
    const transposeStorage = options.transposeStorage === true;
    const resolvedStorageShape = Array.isArray(options.storageShape) && options.storageShape.length === 2
      ? options.storageShape
      : (
        resolvedLogicalShape && transposeStorage
          ? [resolvedLogicalShape[1], resolvedLogicalShape[0]]
          : resolvedLogicalShape
      );
    const resolvedQuantAxis = options.quantAxis === 0 ? 0 : 1;
    normalized.push(
      createLiteRTAxisTensor(
        rawTensor,
        scaleTensor,
        sumTensor,
        sourcePath,
        canonicalName,
        role,
        group,
        resolvedLogicalShape,
        resolvedStorageShape,
        resolvedQuantAxis,
        {
          scaleSemantics: options.scaleSemantics || 'step',
        }
      )
    );
  };

  normalized.push(
      createLiteRTBlockedAxisTensor(
        rawByName.get('transformer.embedder.input_embedding.w') ?? null,
        rawByName.get('transformer.embedder.input_embedding.w_quantized_scale') ?? null,
        rawByName.get('transformer.embedder.input_embedding.w.sum_i') ?? null,
      sourcePath,
      'model.language_model.embed_tokens.weight',
      'embedding',
      'embed',
      [vocabSize, hiddenSize],
      [hiddenSize / 4, vocabSize],
      0,
      4,
      [0, 1, 2, 3],
      {
        scaleSemantics: 'step',
      }
    )
  );
  addAxisQuantized(
    'transformer.embedder.per_layer_model_projection.w',
    'model.language_model.per_layer_model_projection.weight',
    'matmul',
    null,
    [numLayers * hiddenSizePerLayerInput, hiddenSize],
    {
      transposeStorage: true,
      quantAxis: 0,
    }
  );
  addFloat(
    'transformer.embedder.per_layer_model_projection.input_activation_static_scale',
    'model.language_model.per_layer_model_projection.input_activation_static_scale',
    'other'
  );
  addFloat(
    'transformer.embedder.per_layer_model_projection.output_activation_static_scale',
    'model.language_model.per_layer_model_projection.output_activation_static_scale',
    'other'
  );
  addFloat(
    'transformer.embedder.per_layer_projection_norm.scale',
    'model.language_model.per_layer_projection_norm.weight',
    'norm'
  );
  addFloat(
    'transformer.final_norm.scale',
    'model.language_model.norm.weight',
    'norm',
    'head'
  );

  for (let layerIndex = 0; layerIndex < numLayers; layerIndex += 1) {
    const rawLayerPrefix = `transformer.layer_${layerIndex}`;
    const canonicalLayerPrefix = `model.language_model.layers.${layerIndex}`;
    const attentionHeadDim = resolveGemma4AttentionHeadDim(runtimeProfile, layerIndex);
    const kvHeadDim = attentionHeadDim;
    const intermediateSize = resolveGemma4IntermediateSize(runtimeProfile, layerIndex);

    addFloat(`${rawLayerPrefix}.skip.scale`, `${canonicalLayerPrefix}.layer_scalar`, 'other');
    addFloat(`${rawLayerPrefix}.pre_attention_norm.scale`, `${canonicalLayerPrefix}.input_layernorm.weight`, 'norm');
    addAxisQuantized(
      `${rawLayerPrefix}.attn.q.w`,
      `${canonicalLayerPrefix}.self_attn.q_proj.weight`,
      'matmul',
      null,
      [numAttentionHeads * attentionHeadDim, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addFloat(`${rawLayerPrefix}.attn.q_norm.scale`, `${canonicalLayerPrefix}.self_attn.q_norm.weight`, 'norm');
    addAxisQuantized(
      `${rawLayerPrefix}.attn.k.w`,
      `${canonicalLayerPrefix}.self_attn.k_proj.weight`,
      'matmul',
      null,
      [numKeyValueHeads * kvHeadDim, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.attn.v.w`,
      `${canonicalLayerPrefix}.self_attn.v_proj.weight`,
      'matmul',
      null,
      [numKeyValueHeads * kvHeadDim, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addFloat(`${rawLayerPrefix}.attn.k_norm.scale`, `${canonicalLayerPrefix}.self_attn.k_norm.weight`, 'norm');
    addAxisQuantized(
      `${rawLayerPrefix}.attn.attn_vec_einsum.w`,
      `${canonicalLayerPrefix}.self_attn.o_proj.weight`,
      'matmul',
      null,
      [hiddenSize, numAttentionHeads * attentionHeadDim],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addFloat(`${rawLayerPrefix}.post_attention_norm.scale`, `${canonicalLayerPrefix}.post_attention_layernorm.weight`, 'norm');
    addFloat(`${rawLayerPrefix}.pre_ffw_norm.scale`, `${canonicalLayerPrefix}.pre_feedforward_layernorm.weight`, 'norm');
    addFloat(`${rawLayerPrefix}.post_ffw_norm.scale`, `${canonicalLayerPrefix}.post_feedforward_layernorm.weight`, 'norm');
    addFloat(`${rawLayerPrefix}.post_per_layer_input_norm.scale`, `${canonicalLayerPrefix}.post_per_layer_input_norm.weight`, 'norm');
    addAxisQuantized(
      `${rawLayerPrefix}.mlp.ff_gate.w`,
      `${canonicalLayerPrefix}.mlp.gate_proj.weight`,
      'matmul',
      null,
      [intermediateSize, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.mlp.ff1.w`,
      `${canonicalLayerPrefix}.mlp.up_proj.weight`,
      'matmul',
      null,
      [intermediateSize, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.mlp.linear.w`,
      `${canonicalLayerPrefix}.mlp.down_proj.weight`,
      'matmul',
      null,
      [hiddenSize, intermediateSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.per_layer_embedding_gate.w`,
      `${canonicalLayerPrefix}.per_layer_input_gate.weight`,
      'matmul',
      null,
      [hiddenSizePerLayerInput, hiddenSize],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.per_layer_embedding_projection.w`,
      `${canonicalLayerPrefix}.per_layer_projection.weight`,
      'matmul',
      null,
      [hiddenSize, hiddenSizePerLayerInput],
      {
        transposeStorage: true,
        quantAxis: 0,
      }
    );
    addAxisQuantized(
      `${rawLayerPrefix}.per_layer_embeddings.w`,
      `${canonicalLayerPrefix}.embed_tokens_per_layer.weight`,
      'embedding',
      'per_layer_input',
      [vocabSizePerLayerInput, hiddenSizePerLayerInput],
      {
        quantAxis: 1,
        scaleSemantics: 'step',
      }
    );
  }

  if (normalized.length === 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT package did not produce any normalized tensors.');
  }

  return normalized;
}

function buildPackageParsedArtifact(
  sourceKind,
  sourcePathForModelId,
  runtimeProfile,
  parsedTFLite,
  virtualFiles,
  packageConfig = null,
  parsedTFLitesByPath = null
) {
  const tfliteSourceFiles = virtualFiles.filter((entry) => (
    entry.kind === 'tflite_model' || entry.kind === 'tflite_embedder_model'
  ));
  const tfliteSourceFile = virtualFiles.find((entry) => entry.kind === 'tflite_model') ?? null;
  if (!tfliteSourceFile) {
    throw new Error('direct-source runtime: LiteRT package is missing a TFLite model entry.');
  }
  const tokenizerJsonFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_json') ?? null;
  const tokenizerConfigFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_config') ?? null;
  const tokenizerModelFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_model') ?? null;
  const metadataFile = virtualFiles.find((entry) => entry.kind === 'litert_metadata') ?? null;
  const config = cloneJsonValue(runtimeProfile.rawConfig ?? {});
  const graphAdapter = normalizeText(packageConfig?.graphAdapter);
  const tensors = graphAdapter === 'gemma4_unified_12b_split_int4'
    ? normalizeGemma412BSplitLiteRTTensors(parsedTFLitesByPath, virtualFiles, runtimeProfile, packageConfig)
    : (normalizeText(runtimeProfile.modelType) === 'gemma4'
      ? normalizeGemma4LiteRTTensors(parsedTFLite, tfliteSourceFile.path, runtimeProfile)
      : parsedTFLite.tensors.map((tensor) => ({
      ...tensor,
      sourcePath: tfliteSourceFile.path,
      })));
  const manifestInference = cloneJsonValue(runtimeProfile.manifestInference ?? {});
  if (graphAdapter === 'gemma4_unified_12b_split_int4') {
    const qNormLayers = new Set();
    const kNormLayers = new Set();
    for (const tensor of tensors) {
      const qMatch = String(tensor?.name ?? '').match(/\.layers\.(\d+)\.self_attn\.q_norm\.weight$/);
      if (qMatch) {
        qNormLayers.add(Number(qMatch[1]));
      }
      const kMatch = String(tensor?.name ?? '').match(/\.layers\.(\d+)\.self_attn\.k_norm\.weight$/);
      if (kMatch) {
        kNormLayers.add(Number(kMatch[1]));
      }
    }
    const layers = [...qNormLayers]
      .filter((layerIdx) => kNormLayers.has(layerIdx))
      .sort((left, right) => left - right);
    manifestInference.attention = {
      ...(manifestInference.attention ?? {}),
      queryKeyNorm: manifestInference.attention?.queryKeyNorm === true || layers.length > 0,
      queryKeyNormLayers: null,
      queryKeyNormWeightLayers: layers,
    };
  }
  return {
    sourceKind,
    modelType: normalizeText(runtimeProfile.modelType),
    config,
    manifestConfig: cloneJsonValue(runtimeProfile.manifestConfig ?? {}),
    manifestInference,
    architectureHint: normalizeText(runtimeProfile.modelType) || 'transformer',
    embeddingPostprocessor: null,
    architecture: cloneJsonValue(runtimeProfile.architecture ?? null),
    sourceQuantization: parsedTFLite.sourceQuantization,
    tokenizerJson: null,
    tokenizerConfig: resolvePackageTokenizerConfig(sourceKind, runtimeProfile),
    tokenizerModelName: tokenizerModelFile ? tokenizerModelFile.path : null,
    tokenizerJsonPath: tokenizerJsonFile ? tokenizerJsonFile.path : null,
    tokenizerConfigPath: tokenizerConfigFile ? tokenizerConfigFile.path : null,
    tokenizerModelPath: tokenizerModelFile ? tokenizerModelFile.path : null,
    sourceFiles: tfliteSourceFiles.map((entry) => ({
      path: entry.path,
      size: entry.size,
    })),
    auxiliaryFiles: [
      ...(tokenizerJsonFile
        ? [{
          path: tokenizerJsonFile.path,
          size: tokenizerJsonFile.size,
          kind: 'tokenizer_json',
        }]
        : []),
      ...(tokenizerConfigFile
        ? [{
          path: tokenizerConfigFile.path,
          size: tokenizerConfigFile.size,
          kind: 'tokenizer_config',
        }]
        : []),
      ...(tokenizerModelFile
        ? [{
          path: tokenizerModelFile.path,
          size: tokenizerModelFile.size,
          kind: 'tokenizer_model',
        }]
        : []),
      ...(metadataFile
        ? [{
          path: metadataFile.path,
          size: metadataFile.size,
          kind: metadataFile.kind,
        }]
        : []),
    ],
    sourcePathForModelId,
    tensors,
  };
}

async function isRawTFLiteTaskSource(source) {
  const header = await source.readRange(0, 8);
  const bytes = header instanceof Uint8Array ? header : new Uint8Array(header);
  if (bytes.byteLength < 8) {
    return false;
  }
  const identifier = Array.from(bytes.subarray(4, 8), (value) => String.fromCharCode(value)).join('');
  return identifier === TFLITE_FILE_IDENTIFIER;
}

async function parseLiteRTTaskPackage(source, sourcePathForModelId) {
  const packageBasename = normalizeText(source?.name);
  const profile = resolveRequiredProfile(LITERT_PACKAGE_SOURCE_KIND_TASK, packageBasename);
  const taskConfig = profile.package?.task ?? {};
  throwIfUnsupportedPackageProfile(
    LITERT_PACKAGE_SOURCE_KIND_TASK,
    packageBasename,
    profile,
    taskConfig
  );
  const runtimeProfile = profile.runtime ?? null;
  if (!runtimeProfile) {
    throw new Error(`direct-source runtime: package profile "${profile.id}" is missing runtime data.`);
  }

  const rawTFLiteTask = await isRawTFLiteTaskSource(source);
  const virtualFiles = [];
  let parsedTFLite = null;

  if (rawTFLiteTask) {
    virtualFiles.push(createVirtualFile(packageBasename, 0, Number(source.size) || 0, 'tflite_model'));
    parsedTFLite = await parseTFLiteFromSource(source, {
      allowPackedQuantization: true,
    });
    const tokenizerMetadataEntry = findTFLiteMetadataEntry(parsedTFLite, 'spm_vocab_model');
    if (tokenizerMetadataEntry) {
      virtualFiles.push(
        createVirtualFile(
          'TOKENIZER_MODEL',
          tokenizerMetadataEntry.offset,
          tokenizerMetadataEntry.size,
          'tokenizer_model'
        )
      );
    }
    const llmParametersEntry = findTFLiteMetadataEntry(parsedTFLite, 'odml.infra.proto.LlmParameters');
    if (llmParametersEntry) {
      virtualFiles.push(
        createVirtualFile(
          'METADATA',
          llmParametersEntry.offset,
          llmParametersEntry.size,
          'litert_metadata'
        )
      );
    }
  } else {
    const parsedTask = await parseLiteRTTaskFromSource(source);
    const tfliteEntryName = normalizeText(taskConfig.tfliteEntry) || LITERT_TASK_DEFAULT_TFLITE_ENTRY;
    const tokenizerEntryName = normalizeText(taskConfig.tokenizerModelEntry) || LITERT_TASK_DEFAULT_TOKENIZER_MODEL_ENTRY;
    const metadataEntryName = normalizeText(taskConfig.metadataEntry) || LITERT_TASK_DEFAULT_METADATA_ENTRY;
    const tfliteEntry = parsedTask.entryMap.get(tfliteEntryName) ?? null;
    if (!tfliteEntry) {
      throw new Error(
        `direct-source runtime: LiteRT task "${packageBasename}" is missing the required TFLite entry "${tfliteEntryName}".`
      );
    }

    virtualFiles.push(createVirtualFile(tfliteEntryName, tfliteEntry.offset, tfliteEntry.size, 'tflite_model'));
    const tokenizerEntry = parsedTask.entryMap.get(tokenizerEntryName) ?? null;
    if (tokenizerEntry) {
      virtualFiles.push(createVirtualFile('TOKENIZER_MODEL', tokenizerEntry.offset, tokenizerEntry.size, 'tokenizer_model'));
    }
    const metadataEntry = parsedTask.entryMap.get(metadataEntryName) ?? null;
    if (metadataEntry) {
      virtualFiles.push(createVirtualFile('METADATA', metadataEntry.offset, metadataEntry.size, 'litert_metadata'));
    }

    parsedTFLite = await parseTFLiteFromSource(createSectionSource(source, virtualFiles[0]), {
      allowPackedQuantization: true,
    });
  }
  return {
    parsedArtifact: buildPackageParsedArtifact(
      LITERT_PACKAGE_SOURCE_KIND_TASK,
      sourcePathForModelId,
      runtimeProfile,
      parsedTFLite,
      virtualFiles,
      taskConfig
    ),
    virtualFiles,
    packageProfile: profile,
  };
}

async function parseLiteRTLMPackage(source, sourcePathForModelId) {
  const packageBasename = normalizeText(source?.name);
  const profile = resolveRequiredProfile(LITERT_PACKAGE_SOURCE_KIND_LITERTLM, packageBasename);
  const litertConfig = profile.package?.litertlm ?? {};
  throwIfUnsupportedPackageProfile(
    LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
    packageBasename,
    profile,
    litertConfig
  );
  const runtimeProfile = profile.runtime ?? null;
  if (!runtimeProfile) {
    throw new Error(`direct-source runtime: package profile "${profile.id}" is missing runtime data.`);
  }

  const parsedLiteRTLM = await parseLiteRTLMFromSource(source);
  const graphAdapter = normalizeText(litertConfig.graphAdapter);
  const tfliteModelType = normalizeText(litertConfig.tfliteModelType) || LITERT_TASK_DEFAULT_TFLITE_ENTRY;
  const weightsSection = findLiteRTLMTFLiteWeightsSection(parsedLiteRTLM, tfliteModelType);
  if (weightsSection) {
    throw new Error(
      `direct-source runtime: LiteRT-LM "${packageBasename}" uses external TFLiteWeights sections. ` +
      'External-weight LiteRT-LM packages are not supported yet.'
    );
  }
  const preferredModelSection = findLiteRTLMTFLiteModelSection(parsedLiteRTLM, tfliteModelType);
  const fallbackModelSections = findLiteRTLMSectionByType(parsedLiteRTLM, 'TFLiteModel');
  const modelSections = preferredModelSection
    ? [preferredModelSection, ...fallbackModelSections.filter((section) => section !== preferredModelSection)]
    : fallbackModelSections;
  if (modelSections.length === 0) {
    throw new Error(
      `direct-source runtime: LiteRT-LM "${packageBasename}" is missing a TFLiteModel section for "${tfliteModelType}".`
    );
  }

  const tokenizerSection = findLiteRTLMSentencePieceTokenizerSection(parsedLiteRTLM);
  const metadataSection = findLiteRTLMMetadataSection(parsedLiteRTLM);
  if (graphAdapter === 'gemma4_unified_12b_split_int4') {
    const embedderTFLiteModelType = normalizeText(litertConfig.embedderTFLiteModelType);
    if (!embedderTFLiteModelType) {
      throw new Error(
        `direct-source runtime: LiteRT-LM "${packageBasename}" split-section graph adapter requires ` +
        'package.litertlm.embedderTFLiteModelType.'
      );
    }
    const prefillSection = findLiteRTLMTFLiteModelSection(parsedLiteRTLM, tfliteModelType);
    const embedderSection = findLiteRTLMTFLiteModelSection(parsedLiteRTLM, embedderTFLiteModelType);
    if (!prefillSection || !embedderSection) {
      throw new Error(
        `direct-source runtime: LiteRT-LM "${packageBasename}" split-section graph adapter requires ` +
        `TFLiteModel sections "${embedderTFLiteModelType}" and "${tfliteModelType}".`
      );
    }
    const virtualFiles = [
      createVirtualFile(embedderTFLiteModelType, embedderSection.beginOffset, embedderSection.size, 'tflite_embedder_model'),
      createVirtualFile(tfliteModelType, prefillSection.beginOffset, prefillSection.size, 'tflite_model'),
    ];
    if (tokenizerSection) {
      virtualFiles.push(createVirtualFile('TOKENIZER_MODEL', tokenizerSection.beginOffset, tokenizerSection.size, 'tokenizer_model'));
    }
    if (metadataSection) {
      virtualFiles.push(createVirtualFile('METADATA', metadataSection.beginOffset, metadataSection.size, 'litert_metadata'));
    }
    const parsedTFLitesByPath = new Map();
    for (const virtualFile of virtualFiles.filter((entry) => entry.kind === 'tflite_model' || entry.kind === 'tflite_embedder_model')) {
      parsedTFLitesByPath.set(
        virtualFile.path,
        await parseTFLiteFromSource(createSectionSource(source, virtualFile), {
          allowPackedQuantization: true,
        })
      );
    }
    return {
      parsedArtifact: buildPackageParsedArtifact(
        LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
        sourcePathForModelId,
        runtimeProfile,
        parsedTFLitesByPath.get(tfliteModelType),
        virtualFiles,
        litertConfig,
        parsedTFLitesByPath
      ),
      virtualFiles,
      packageProfile: profile,
    };
  }
  const errors = [];

  for (let candidateIndex = 0; candidateIndex < modelSections.length; candidateIndex += 1) {
    const modelSection = modelSections[candidateIndex];
    const modelPath = modelSections.length === 1
      ? tfliteModelType
      : `${tfliteModelType}_${candidateIndex}`;
    const virtualFiles = [
      createVirtualFile(modelPath, modelSection.beginOffset, modelSection.size, 'tflite_model'),
    ];
    if (tokenizerSection) {
      virtualFiles.push(createVirtualFile('TOKENIZER_MODEL', tokenizerSection.beginOffset, tokenizerSection.size, 'tokenizer_model'));
    }
    if (metadataSection) {
      virtualFiles.push(createVirtualFile('METADATA', metadataSection.beginOffset, metadataSection.size, 'litert_metadata'));
    }

    try {
      const parsedTFLite = await parseTFLiteFromSource(createSectionSource(source, virtualFiles[0]), {
        allowPackedQuantization: true,
      });
      return {
        parsedArtifact: buildPackageParsedArtifact(
          LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
          sourcePathForModelId,
          runtimeProfile,
          parsedTFLite,
          virtualFiles,
          litertConfig
        ),
        virtualFiles,
        packageProfile: profile,
      };
    } catch (error) {
      errors.push(
        `candidate ${candidateIndex}: ${String(error?.message || error)}`
      );
    }
  }

  throw new Error(
    `direct-source runtime: LiteRT-LM "${packageBasename}" did not expose a supported text TFLiteModel section. ` +
    errors.join(' | ')
  );
}

export async function resolveLiteRTPackageParsedArtifact(options = {}) {
  const source = options.source;
  if (!source || typeof source.readRange !== 'function') {
    throw new Error('direct-source runtime: LiteRT package source.readRange(offset, length) is required.');
  }
  const sourceKind = normalizeText(options.sourceKind).toLowerCase();
  const sourcePathForModelId = normalizeText(options.sourcePathForModelId) || normalizeText(source.name);
  if (sourceKind === LITERT_PACKAGE_SOURCE_KIND_TASK) {
    return parseLiteRTTaskPackage(source, sourcePathForModelId);
  }
  if (sourceKind === LITERT_PACKAGE_SOURCE_KIND_LITERTLM) {
    return parseLiteRTLMPackage(source, sourcePathForModelId);
  }
  throw new Error(`direct-source runtime: unsupported LiteRT package sourceKind "${options.sourceKind}".`);
}

export function appendLiteRTPackageVirtualFiles(virtualFiles, entries = []) {
  if (!Array.isArray(entries) || entries.length === 0) {
    return Array.isArray(virtualFiles) ? [...virtualFiles] : [];
  }
  const merged = Array.isArray(virtualFiles) ? [...virtualFiles] : [];
  const seenPaths = new Set(merged.map((entry) => normalizeText(entry?.path)).filter(Boolean));
  for (const entry of entries) {
    const path = normalizeText(entry?.path);
    if (!path || seenPaths.has(path)) {
      continue;
    }
    merged.push(createVirtualFile(
      path,
      Number.isFinite(entry?.offset) ? Number(entry.offset) : 0,
      Number.isFinite(entry?.size) ? Number(entry.size) : 0,
      normalizeText(entry?.kind) || 'unknown',
      { externalPath: entry?.externalPath ?? null }
    ));
    seenPaths.add(path);
  }
  return merged;
}
