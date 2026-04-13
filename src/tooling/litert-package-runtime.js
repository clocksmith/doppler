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

export const LITERT_PACKAGE_SOURCE_KIND_TASK = 'litert-task';
export const LITERT_PACKAGE_SOURCE_KIND_LITERTLM = 'litertlm';

function normalizeText(value) {
  return String(value || '').trim();
}

function cloneJsonValue(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
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
  quantAxis = 1
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
      sourceDtype: layout.sourceDtype,
      targetDtype: 'F16',
      storageEncoding: layout.storageEncoding,
      storageShape: [storageRows, storageCols],
      quantAxis,
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
        resolvedQuantAxis
      )
    );
  };

  addAxisQuantized(
    'transformer.embedder.input_embedding.w',
    'model.language_model.embed_tokens.weight',
    'embedding',
    'embed',
    [vocabSize, hiddenSize],
    {
      transposeStorage: true,
      quantAxis: 0,
    }
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
        transposeStorage: true,
        quantAxis: 0,
      }
    );
  }

  if (normalized.length === 0) {
    throw new Error('direct-source runtime: Gemma 4 LiteRT package did not produce any normalized tensors.');
  }

  return normalized;
}

function buildPackageParsedArtifact(sourceKind, sourcePathForModelId, runtimeProfile, parsedTFLite, virtualFiles) {
  const tfliteSourceFile = virtualFiles.find((entry) => entry.kind === 'tflite_model') ?? null;
  if (!tfliteSourceFile) {
    throw new Error('direct-source runtime: LiteRT package is missing a TFLite model entry.');
  }
  const tokenizerJsonFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_json') ?? null;
  const tokenizerConfigFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_config') ?? null;
  const tokenizerModelFile = virtualFiles.find((entry) => entry.kind === 'tokenizer_model') ?? null;
  const metadataFile = virtualFiles.find((entry) => entry.kind === 'litert_metadata') ?? null;
  const config = cloneJsonValue(runtimeProfile.rawConfig ?? {});
  const tensors = normalizeText(runtimeProfile.modelType) === 'gemma4'
    ? normalizeGemma4LiteRTTensors(parsedTFLite, tfliteSourceFile.path, runtimeProfile)
    : parsedTFLite.tensors.map((tensor) => ({
      ...tensor,
      sourcePath: tfliteSourceFile.path,
    }));
  return {
    sourceKind,
    modelType: normalizeText(runtimeProfile.modelType),
    config,
    manifestConfig: cloneJsonValue(runtimeProfile.manifestConfig ?? {}),
    manifestInference: cloneJsonValue(runtimeProfile.manifestInference ?? {}),
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
    sourceFiles: [
      {
        path: tfliteSourceFile.path,
        size: tfliteSourceFile.size,
      },
    ],
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
  const runtimeProfile = profile.runtime ?? null;
  if (!runtimeProfile) {
    throw new Error(`direct-source runtime: package profile "${profile.id}" is missing runtime data.`);
  }

  const taskConfig = profile.package?.task ?? {};
  const rawTFLiteTask = await isRawTFLiteTaskSource(source);
  const virtualFiles = [];
  let parsedTFLite = null;

  if (rawTFLiteTask) {
    virtualFiles.push(createVirtualFile(packageBasename, 0, Number(source.size) || 0, 'tflite_model'));
    parsedTFLite = await parseTFLiteFromSource(source, {
      allowPackedQuantization: true,
    });
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
      virtualFiles
    ),
    virtualFiles,
    packageProfile: profile,
  };
}

async function parseLiteRTLMPackage(source, sourcePathForModelId) {
  const packageBasename = normalizeText(source?.name);
  const profile = resolveRequiredProfile(LITERT_PACKAGE_SOURCE_KIND_LITERTLM, packageBasename);
  const runtimeProfile = profile.runtime ?? null;
  if (!runtimeProfile) {
    throw new Error(`direct-source runtime: package profile "${profile.id}" is missing runtime data.`);
  }

  const parsedLiteRTLM = await parseLiteRTLMFromSource(source);
  const litertConfig = profile.package?.litertlm ?? {};
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
          virtualFiles
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
