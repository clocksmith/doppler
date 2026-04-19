function normalizeText(value) {
  return String(value || '').trim();
}

function normalizeNonNegativeInteger(value, label) {
  const normalized = Number(value);
  if (!Number.isFinite(normalized) || normalized < 0) {
    throw new Error(`${label} must be a non-negative number.`);
  }
  return Math.floor(normalized);
}

function normalizeOptionalObject(value, label) {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object when provided.`);
  }
  return value;
}

function normalizeShards(shards) {
  if (!Array.isArray(shards) || shards.length === 0) {
    throw new Error('runtime model contract requires shards[].');
  }
  return shards.map((shard, index) => {
    const filename = normalizeText(shard?.filename);
    if (!filename) {
      throw new Error(`runtime model shard[${index}] is missing filename.`);
    }
    return {
      ...shard,
      index: Number.isInteger(shard?.index)
        ? shard.index
        : index,
      filename,
      size: normalizeNonNegativeInteger(shard?.size, `runtime model shard[${index}] size`),
      offset: normalizeNonNegativeInteger(shard?.offset ?? 0, `runtime model shard[${index}] offset`),
      hash: shard?.hash == null ? '' : String(shard.hash),
      ...(shard?.hashAlgorithm != null ? { hashAlgorithm: String(shard.hashAlgorithm) } : {}),
    };
  });
}

function inferTotalSize(shards) {
  return shards.reduce((sum, shard) => sum + shard.size, 0);
}

export function createRuntimeModelContract(options = {}) {
  const modelId = normalizeText(options.modelId);
  if (!modelId) {
    throw new Error('runtime model contract requires modelId.');
  }
  const modelType = normalizeText(options.modelType);
  if (!modelType) {
    throw new Error(`runtime model contract "${modelId}" requires modelType.`);
  }
  const quantization = normalizeText(options.quantization);
  if (!quantization) {
    throw new Error(`runtime model contract "${modelId}" requires quantization.`);
  }
  const hashAlgorithm = normalizeText(options.hashAlgorithm);
  if (!hashAlgorithm) {
    throw new Error(`runtime model contract "${modelId}" requires hashAlgorithm.`);
  }
  const architecture = options.architecture;
  if (
    architecture == null
    || (typeof architecture !== 'string'
      && (typeof architecture !== 'object' || Array.isArray(architecture)))
  ) {
    throw new Error(`runtime model contract "${modelId}" requires architecture.`);
  }
  const inference = normalizeOptionalObject(
    options.inference,
    `runtime model contract "${modelId}" inference`
  );
  if (!inference) {
    throw new Error(`runtime model contract "${modelId}" requires inference.`);
  }

  const shards = normalizeShards(options.shards);
  const totalSize = options.totalSize == null
    ? inferTotalSize(shards)
    : normalizeNonNegativeInteger(options.totalSize, `runtime model contract "${modelId}" totalSize`);

  return {
    kind: 'runtime-model',
    sourceFormat: normalizeText(options.sourceFormat).toLowerCase() || null,
    ...(options.version !== undefined ? { version: options.version } : {}),
    modelId,
    modelType,
    quantization,
    hashAlgorithm,
    architecture,
    inference,
    shards,
    totalSize,
    eos_token_id: options.eos_token_id ?? null,
    ...(options.quantizationInfo ? { quantizationInfo: options.quantizationInfo } : {}),
    ...(options.image_token_id !== undefined ? { image_token_id: options.image_token_id } : {}),
    ...(options.audio_token_id !== undefined ? { audio_token_id: options.audio_token_id } : {}),
    ...(options.video_token_id !== undefined ? { video_token_id: options.video_token_id } : {}),
    ...(options.groups ? { groups: options.groups } : {}),
    ...(options.tensorsFile ? { tensorsFile: options.tensorsFile } : {}),
    ...(options.tensorCount !== undefined ? { tensorCount: options.tensorCount } : {}),
    ...(options.tokenizer ? { tokenizer: options.tokenizer } : {}),
    ...(options.moeConfig ? { moeConfig: options.moeConfig } : {}),
    ...(options.optimizations ? { optimizations: options.optimizations } : {}),
    ...(options.runtime ? { runtime: options.runtime } : {}),
    ...(options.config ? { config: options.config } : {}),
    ...(options.quantization_config ? { quantization_config: options.quantization_config } : {}),
    ...(options.conversion ? { conversion: options.conversion } : {}),
    ...(options.blake3Full ? { blake3Full: options.blake3Full } : {}),
    ...(options.defaultWeightLayout ? { defaultWeightLayout: options.defaultWeightLayout } : {}),
    ...(options.metadata ? { metadata: options.metadata } : {}),
    ...(options.adapterType ? { adapterType: options.adapterType } : {}),
    ...(options.baseCompatibility ? { baseCompatibility: options.baseCompatibility } : {}),
    ...(options.mergedAdapter ? { mergedAdapter: options.mergedAdapter } : {}),
    ...(options.adapterConfig ? { adapterConfig: options.adapterConfig } : {}),
    ...(options.provenance ? { provenance: options.provenance } : {}),
    ...(options.baseModel ? { baseModel: options.baseModel } : {}),
    ...(options.loraConfig ? { loraConfig: options.loraConfig } : {}),
    ...(options.draftModel ? { draftModel: options.draftModel } : {}),
    ...(options.tensors ? { tensors: options.tensors } : {}),
  };
}

