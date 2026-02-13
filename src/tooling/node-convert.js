import fs from 'node:fs/promises';
import path from 'node:path';
import { installNodeFileFetchShim } from './node-file-fetch.js';


function generateShardFilename(index) {
  return `shard_${String(index).padStart(5, '0')}.bin`;
}

function assertPath(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`node convert: ${label} is required.`);
  }
  return path.resolve(value);
}

function parseModelId(value, outputDir) {
  if (typeof value === 'string' && value.trim()) {
    return value.trim();
  }
  return path.basename(outputDir);
}

function isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function normalizeConverterConfigOverride(value) {
  if (value == null) return null;
  if (!isPlainObject(value)) {
    throw new Error('node convert: converterConfig must be an object when provided.');
  }
  return value;
}

async function readOptionalJson(filePath) {
  try {
    const text = await fs.readFile(filePath, 'utf8');
    return JSON.parse(text);
  } catch {
    return null;
  }
}

async function fileExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readSafetensorsHeader(filePath, parseSafetensorsHeader) {
  const fd = await fs.open(filePath, 'r');
  try {
    const sizeBuf = Buffer.allocUnsafe(8);
    await fd.read(sizeBuf, 0, 8, 0);
    const headerSize = Number(sizeBuf.readBigUInt64LE(0));
    const fullHeader = Buffer.allocUnsafe(8 + headerSize);
    await fd.read(fullHeader, 0, fullHeader.length, 0);
    return parseSafetensorsHeader(
      fullHeader.buffer.slice(fullHeader.byteOffset, fullHeader.byteOffset + fullHeader.byteLength)
    );
  } finally {
    await fd.close();
  }
}

async function loadTensorHeaders(inputDir, parseSafetensorsHeader) {
  const indexPath = path.join(inputDir, 'model.safetensors.index.json');
  const singlePath = path.join(inputDir, 'model.safetensors');
  const tensors = [];

  let hasIndex = false;
  try {
    await fs.access(indexPath);
    hasIndex = true;
  } catch {
    hasIndex = false;
  }

  if (hasIndex) {
    const indexJson = JSON.parse(await fs.readFile(indexPath, 'utf8'));
    const shardFiles = [...new Set(Object.values(indexJson.weight_map || {}))];
    for (const shardFile of shardFiles) {
      const shardPath = path.join(inputDir, shardFile);
      const parsed = await readSafetensorsHeader(shardPath, parseSafetensorsHeader);
      for (const tensor of parsed.tensors) {
        tensors.push({ ...tensor, sourcePath: shardPath });
      }
    }
  } else {
    const parsed = await readSafetensorsHeader(singlePath, parseSafetensorsHeader);
    for (const tensor of parsed.tensors) {
      tensors.push({ ...tensor, sourcePath: singlePath });
    }
  }

  tensors.sort((a, b) => a.name.localeCompare(b.name));
  return tensors;
}

function inferSourceWeightQuantization(tensors) {
  if (!Array.isArray(tensors) || tensors.length === 0) {
    return 'f16';
  }
  const dtypes = new Set(
    tensors
      .filter((tensor) => typeof tensor?.name === 'string' && tensor.name.includes('.weight'))
      .map((tensor) => String(tensor?.dtype || '').trim().toUpperCase())
      .filter((dtype) => dtype.length > 0)
  );
  if (dtypes.size === 0) return 'f16';
  if (dtypes.size === 1) {
    const only = [...dtypes][0];
    if (only === 'F32') return 'f32';
    if (only === 'F16') return 'f16';
    if (only === 'BF16') return 'f16';
  }
  if (dtypes.has('F32')) return 'f32';
  if (dtypes.has('BF16')) return 'f16';
  return 'f16';
}

function createNodeConvertIO(outputDir, options) {
  const hashAlgorithm = options?.hashAlgorithm;
  const computeHash = options?.computeHash;
  if (!hashAlgorithm || typeof hashAlgorithm !== 'string') {
    throw new Error('node convert: hashAlgorithm is required.');
  }
  if (typeof computeHash !== 'function') {
    throw new Error('node convert: computeHash(data, algorithm) is required.');
  }
  return {
    async readTensorData(tensor) {
      const fd = await fs.open(tensor.sourcePath, 'r');
      try {
        const out = Buffer.allocUnsafe(tensor.size);
        await fd.read(out, 0, tensor.size, tensor.offset);
        return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
      } finally {
        await fd.close();
      }
    },
    async writeShard(index, data) {
      const filename = generateShardFilename(index);
      await fs.writeFile(path.join(outputDir, filename), data);
      return computeHash(data, hashAlgorithm);
    },
    async writeManifest(manifest) {
      await fs.writeFile(
        path.join(outputDir, 'manifest.json'),
        JSON.stringify(manifest, null, 2),
        'utf8'
      );
    },
  };
}

function toNodeProgress(update) {
  if (!update) return null;
  return {
    stage: update.stage ?? null,
    current: Number.isFinite(update.current) ? update.current : null,
    total: Number.isFinite(update.total) ? update.total : null,
    message: typeof update.message === 'string' ? update.message : null,
  };
}

function normalizeTokenizerManifest(manifest) {
  if (!manifest?.tokenizer) return manifest;
  const tokenizer = manifest.tokenizer;
  if (tokenizer.type === 'bundled' || tokenizer.type === 'huggingface') {
    tokenizer.file = tokenizer.file ?? 'tokenizer.json';
  }
  if (tokenizer.type === 'sentencepiece') {
    tokenizer.sentencepieceModel = tokenizer.sentencepieceModel ?? 'tokenizer.model';
  }
  return manifest;
}

export async function convertSafetensorsDirectory(options) {
  const inputDir = assertPath(options?.inputDir, 'inputDir');
  const outputDir = assertPath(options?.outputDir, 'outputDir');
  const modelId = parseModelId(options?.modelId, outputDir);
  const converterConfigOverride = normalizeConverterConfigOverride(options?.converterConfig);
  const onProgress = typeof options?.onProgress === 'function' ? options.onProgress : null;

  installNodeFileFetchShim();

  const [
    { parseSafetensorsHeader },
    { convertModel, extractArchitecture },
    { resolveManifestQuantization },
    { createConverterConfig },
    { detectPreset, resolvePreset },
    { computeHash },
  ] = await Promise.all([
    import('../formats/safetensors/types.js'),
    import('../converter/core.js'),
    import('../converter/quantization-info.js'),
    import('../config/schema/converter.schema.js'),
    import('../config/loader.js'),
    import('../storage/shard-manager.js'),
  ]);

  await fs.mkdir(outputDir, { recursive: true });

  const configPath = path.join(inputDir, 'config.json');
  const config = JSON.parse(await fs.readFile(configPath, 'utf8'));
  const tensors = await loadTensorHeaders(inputDir, parseSafetensorsHeader);

  const architectureHint = config.architectures?.[0] ?? config.model_type ?? '';
  const architecture = extractArchitecture(config, null);
  const presetId = detectPreset(config, architectureHint);
  const preset = resolvePreset(presetId);
  const resolvedModelType = preset.modelType ?? 'transformer';
  const sourceQuantization = inferSourceWeightQuantization(tensors);
  const converterConfig = createConverterConfig(converterConfigOverride ?? undefined);
  const weightOverride = converterConfig.quantization?.weights ?? null;
  const targetQuantization = resolveManifestQuantization(weightOverride, sourceQuantization);

  const tokenizerJsonPath = path.join(inputDir, 'tokenizer.json');
  const tokenizerModelPath = path.join(inputDir, 'tokenizer.model');
  const tokenizerConfigPath = path.join(inputDir, 'tokenizer_config.json');

  const tokenizerJson = await readOptionalJson(tokenizerJsonPath);
  const tokenizerConfig = await readOptionalJson(tokenizerConfigPath);
  const hasTokenizerModel = await fileExists(tokenizerModelPath);

  const model = {
    name: path.basename(inputDir),
    modelId,
    tensors: tensors.map((tensor) => ({
      name: tensor.name,
      shape: tensor.shape,
      dtype: tensor.dtype,
      size: tensor.size,
      offset: tensor.offset,
      sourcePath: tensor.sourcePath,
    })),
    config,
    architecture: architectureHint || 'unknown',
    quantization: targetQuantization,
    tokenizerJson,
    tokenizerConfig,
    tokenizerModel: hasTokenizerModel ? 'tokenizer.model' : null,
  };

  const io = createNodeConvertIO(outputDir, {
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    computeHash,
  });
  const result = await convertModel(model, io, {
    modelId,
    modelType: resolvedModelType,
    quantization: targetQuantization,
    architecture,
    converterConfig,
    onProgress(update) {
      onProgress?.(toNodeProgress(update));
    },
  });

  if (tokenizerJson) {
    await fs.writeFile(path.join(outputDir, 'tokenizer.json'), JSON.stringify(tokenizerJson), 'utf8');
  }
  if (hasTokenizerModel) {
    await fs.copyFile(tokenizerModelPath, path.join(outputDir, 'tokenizer.model'));
  }

  normalizeTokenizerManifest(result.manifest);
  await io.writeManifest(result.manifest);

  return {
    manifest: result.manifest,
    shardCount: result.shardCount,
    tensorCount: result.tensorCount,
    presetId,
    modelType: resolvedModelType,
    outputDir,
  };
}
