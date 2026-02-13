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

async function listRelativeFiles(rootDir, relDir = '', out = []) {
  const currentDir = relDir ? path.join(rootDir, relDir) : rootDir;
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  for (const entry of entries) {
    const relPath = relDir ? `${relDir}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      await listRelativeFiles(rootDir, relPath, out);
      continue;
    }
    out.push(relPath.replace(/\\/g, '/'));
  }
  return out;
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
    { resolveConversionPlan, inferSourceWeightQuantization },
    { parseDiffusionModel },
    { parseTransformerModel },
    { createConverterConfig },
    { computeHash },
  ] = await Promise.all([
    import('../formats/safetensors/types.js'),
    import('../converter/core.js'),
    import('../converter/conversion-plan.js'),
    import('../converter/parsers/diffusion.js'),
    import('../converter/parsers/transformer.js'),
    import('../config/schema/converter.schema.js'),
    import('../storage/shard-manager.js'),
  ]);

  await fs.mkdir(outputDir, { recursive: true });

  const converterConfig = createConverterConfig(converterConfigOverride ?? undefined);
  const diffusionIndexPath = path.join(inputDir, 'model_index.json');
  const isDiffusionInput = await fileExists(diffusionIndexPath);

  let config = null;
  let tensors = [];
  let architectureHint = '';
  let architecture = null;
  let modelKind = 'transformer';
  let tokenizerJson = null;
  let tokenizerConfig = null;
  let hasTokenizerModel = false;
  let tokenizerModelPath = null;
  let diffusionAuxFiles = [];

  if (isDiffusionInput) {
    const relativeFiles = await listRelativeFiles(inputDir);
    const fileSet = new Set(relativeFiles);
    const toArrayBuffer = (buffer) => (
      buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
    );
    const parsedDiffusion = await parseDiffusionModel({
      onProgress,
      findExistingSuffix(suffixes) {
        for (const suffix of suffixes || []) {
          if (fileSet.has(suffix)) return suffix;
        }
        return null;
      },
      async readJson(suffix, label = 'json') {
        if (!fileSet.has(suffix)) {
          throw new Error(`Missing ${label} (${suffix})`);
        }
        return JSON.parse(await fs.readFile(path.join(inputDir, suffix), 'utf8'));
      },
      async readText(suffix, label = 'text') {
        if (!fileSet.has(suffix)) {
          throw new Error(`Missing ${label} (${suffix})`);
        }
        return fs.readFile(path.join(inputDir, suffix), 'utf8');
      },
      async readBinary(suffix, label = 'binary') {
        if (!fileSet.has(suffix)) {
          throw new Error(`Missing ${label} (${suffix})`);
        }
        const bytes = await fs.readFile(path.join(inputDir, suffix));
        return toArrayBuffer(bytes);
      },
      async parseSingleSafetensors(suffix) {
        if (!fileSet.has(suffix)) {
          throw new Error(`Missing safetensors file (${suffix})`);
        }
        const fullPath = path.join(inputDir, suffix);
        const parsed = await readSafetensorsHeader(fullPath, parseSafetensorsHeader);
        return {
          tensors: parsed.tensors.map((tensor) => ({
            ...tensor,
            sourcePath: fullPath,
          })),
        };
      },
      async parseShardedSafetensors(indexSuffix, indexJson, componentId) {
        const weightMap = indexJson?.weight_map || {};
        const shardNames = Array.from(new Set(Object.values(weightMap)));
        if (shardNames.length === 0) {
          throw new Error(`No shards listed in ${componentId} index file`);
        }
        const baseDir = indexSuffix.includes('/')
          ? indexSuffix.split('/').slice(0, -1).join('/')
          : '';
        const shardSuffixes = shardNames.map((name) => (baseDir ? `${baseDir}/${name}` : name));
        const missing = shardSuffixes.filter((suffix) => !fileSet.has(suffix));
        if (missing.length > 0) {
          throw new Error(
            `Missing shard files for ${componentId} (${shardSuffixes.length - missing.length}/${shardSuffixes.length} found)`
          );
        }
        const tensorsOut = [];
        for (const shardSuffix of shardSuffixes) {
          const fullPath = path.join(inputDir, shardSuffix);
          const parsed = await readSafetensorsHeader(fullPath, parseSafetensorsHeader);
          for (const tensor of parsed.tensors) {
            tensorsOut.push({
              ...tensor,
              sourcePath: fullPath,
            });
          }
        }
        return { tensors: tensorsOut };
      },
    });
    config = parsedDiffusion.config;
    tensors = parsedDiffusion.tensors;
    architectureHint = 'diffusion';
    modelKind = 'diffusion';
    diffusionAuxFiles = parsedDiffusion.auxFiles ?? [];
  } else {
    const parsedTransformer = await parseTransformerModel({
      async readJson(suffix, label = 'json') {
        const filePath = path.join(inputDir, suffix);
        try {
          return JSON.parse(await fs.readFile(filePath, 'utf8'));
        } catch (error) {
          throw new Error(`Missing ${label} (${suffix})`);
        }
      },
      async fileExists(suffix) {
        return fileExists(path.join(inputDir, suffix));
      },
      async loadSingleSafetensors(suffix) {
        const filePath = path.join(inputDir, suffix);
        const parsed = await readSafetensorsHeader(filePath, parseSafetensorsHeader);
        return parsed.tensors.map((tensor) => ({
          ...tensor,
          sourcePath: filePath,
        }));
      },
      async loadShardedSafetensors(indexJson) {
        const shardFiles = [...new Set(Object.values(indexJson.weight_map || {}))];
        const tensorsOut = [];
        for (const shardFile of shardFiles) {
          const shardPath = path.join(inputDir, shardFile);
          const parsed = await readSafetensorsHeader(shardPath, parseSafetensorsHeader);
          for (const tensor of parsed.tensors) {
            tensorsOut.push({ ...tensor, sourcePath: shardPath });
          }
        }
        return tensorsOut;
      },
    });
    config = parsedTransformer.config;
    tensors = parsedTransformer.tensors;
    architectureHint = parsedTransformer.architectureHint;
    architecture = extractArchitecture(config, null);
    const tokenizerJsonPath = path.join(inputDir, 'tokenizer.json');
    tokenizerModelPath = path.join(inputDir, 'tokenizer.model');
    const tokenizerConfigPath = path.join(inputDir, 'tokenizer_config.json');
    tokenizerJson = await readOptionalJson(tokenizerJsonPath);
    tokenizerConfig = await readOptionalJson(tokenizerConfigPath);
    hasTokenizerModel = await fileExists(tokenizerModelPath);
  }

  const sourceQuantization = inferSourceWeightQuantization(tensors);
  const plan = resolveConversionPlan({
    rawConfig: config,
    tensors,
    converterConfig,
    sourceQuantization,
    modelKind,
    architectureHint,
    architectureConfig: architecture,
    includePresetOverrideHint: modelKind === 'transformer',
  });
  const resolvedModelType = plan.modelType;
  const targetQuantization = plan.manifestQuantization;
  const quantizationInfo = plan.quantizationInfo;
  const inference = plan.manifestInference;
  const presetId = plan.presetId;

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
  const manifestArchitecture = modelKind === 'diffusion' ? 'diffusion' : architecture;
  const result = await convertModel(model, io, {
    modelId,
    modelType: resolvedModelType,
    quantization: targetQuantization,
    quantizationInfo,
    architecture: manifestArchitecture,
    inference,
    converterConfig,
    onProgress(update) {
      onProgress?.(toNodeProgress(update));
    },
  });

  if (tokenizerJson) {
    await fs.writeFile(path.join(outputDir, 'tokenizer.json'), JSON.stringify(tokenizerJson), 'utf8');
  }
  if (hasTokenizerModel && tokenizerModelPath) {
    await fs.copyFile(tokenizerModelPath, path.join(outputDir, 'tokenizer.model'));
  }
  if (diffusionAuxFiles.length > 0) {
    for (const asset of diffusionAuxFiles) {
      const outPath = path.join(outputDir, asset.name);
      if (typeof asset.data === 'string') {
        await fs.writeFile(outPath, asset.data, 'utf8');
      } else {
        await fs.writeFile(outPath, Buffer.from(asset.data));
      }
    }
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
