import fs from 'node:fs/promises';
import path from 'node:path';
import {
  HEADER_READ_SIZE,
  createConverterConfig,
} from '../config/schema/index.js';
import { extractArchitecture } from '../converter/core.js';
import {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} from '../converter/conversion-plan.js';
import { parseGGUFModel } from '../converter/parsers/gguf.js';
import { parseTransformerModel } from '../converter/parsers/transformer.js';
import { parseGGUFHeader } from '../formats/gguf/types.js';
import { parseSafetensorsHeader } from '../formats/safetensors/types.js';
import { log } from '../debug/index.js';
import {
  buildSourceRuntimeBundle,
  createSourceStorageContext,
} from './source-runtime-bundle.js';

const SUPPORTED_SOURCE_DTYPES = new Set([
  'F32',
  'F16',
  'BF16',
  'Q4_K',
  'Q4_K_M',
  'Q6_K',
]);

const SOURCE_RUNTIME_EXECUTION_OVERRIDE = {
  steps: [
    {
      id: 'cast.identity',
      op: 'cast',
      phase: 'both',
      section: 'layer',
      src: 'attn_q',
      dst: 'attn_q',
      layers: 'all',
      toDtype: 'f16',
    },
  ],
};

function toArrayBuffer(value, label) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (value instanceof Uint8Array) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new Error(`${label} must return ArrayBuffer or Uint8Array.`);
}

function normalizePath(value) {
  return String(value || '').trim();
}

function isGgufPath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.gguf');
}

function resolveModelIdHint(requestedModelId, plan, sourceKind, sourcePath) {
  const explicit = String(requestedModelId || '').trim();
  if (explicit) {
    return resolveConvertedModelId({
      explicitModelId: explicit,
      converterConfig: null,
      detectedModelId: explicit,
      quantizationInfo: plan.quantizationInfo,
    }) || explicit;
  }
  const basename = path.basename(sourcePath, path.extname(sourcePath)) || sourceKind;
  return resolveConvertedModelId({
    explicitModelId: basename,
    converterConfig: null,
    detectedModelId: basename,
    quantizationInfo: plan.quantizationInfo,
  }) || basename;
}

async function getPathStats(targetPath, label) {
  try {
    return await fs.stat(targetPath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`node source runtime: ${label} does not exist: ${targetPath}`);
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`node source runtime: failed to stat ${label} "${targetPath}": ${message}`);
  }
}

async function fileExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

async function readJson(filePath, label) {
  const text = await fs.readFile(filePath, 'utf8');
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('JSON root must be an object');
    }
    return parsed;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON in ${label}: ${message}`);
  }
}

async function readRange(filePath, offset, length) {
  if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
    return new ArrayBuffer(0);
  }
  const handle = await fs.open(filePath, 'r');
  try {
    const stats = await handle.stat();
    const start = Math.max(0, Math.floor(offset));
    const end = Math.min(Number(stats.size), start + Math.floor(length));
    if (end <= start) {
      return new ArrayBuffer(0);
    }
    const out = Buffer.allocUnsafe(end - start);
    await handle.read(out, 0, out.length, start);
    return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
  } finally {
    await handle.close();
  }
}

async function readSafetensorsHeaderFromFile(filePath) {
  const headerPrefixBuffer = await readRange(filePath, 0, 8);
  const prefixBytes = new Uint8Array(headerPrefixBuffer);
  if (prefixBytes.byteLength < 8) {
    throw new Error(`Invalid safetensors header prefix for "${filePath}"`);
  }
  const headerSize = Number(new DataView(headerPrefixBuffer).getBigUint64(0, true));
  const headerBuffer = await readRange(filePath, 8, headerSize);
  const fullHeader = new Uint8Array(8 + headerSize);
  fullHeader.set(prefixBytes, 0);
  fullHeader.set(new Uint8Array(headerBuffer), 8);
  return parseSafetensorsHeader(
    fullHeader.buffer.slice(fullHeader.byteOffset, fullHeader.byteOffset + fullHeader.byteLength)
  );
}

async function parseSafetensorsInput(inputDir) {
  const configPath = path.join(inputDir, 'config.json');
  if (!(await fileExists(configPath))) {
    return null;
  }
  const hasSingle = await fileExists(path.join(inputDir, 'model.safetensors'));
  const hasIndex = await fileExists(path.join(inputDir, 'model.safetensors.index.json'));
  if (!hasSingle && !hasIndex) {
    return null;
  }

  const parsedTransformer = await parseTransformerModel({
    async readJson(suffix, label = 'json') {
      return readJson(path.join(inputDir, suffix), `${label} (${suffix})`);
    },
    async fileExists(suffix) {
      return fileExists(path.join(inputDir, suffix));
    },
    async loadSingleSafetensors(suffix) {
      const filePath = path.join(inputDir, suffix);
      const parsed = await readSafetensorsHeaderFromFile(filePath);
      return parsed.tensors.map((tensor) => ({
        ...tensor,
        sourcePath: filePath,
      }));
    },
    async loadShardedSafetensors(indexJson) {
      const shardFiles = [...new Set(Object.values(indexJson.weight_map || {}))];
      const tensors = [];
      for (const shardFile of shardFiles) {
        const shardPath = path.join(inputDir, shardFile);
        const parsed = await readSafetensorsHeaderFromFile(shardPath);
        for (const tensor of parsed.tensors) {
          tensors.push({
            ...tensor,
            sourcePath: shardPath,
          });
        }
      }
      return tensors;
    },
  });

  const config = parsedTransformer.config;
  const tensors = parsedTransformer.tensors;
  const architectureHint = parsedTransformer.architectureHint;
  const architecture = extractArchitecture(config, null);

  const tokenizerJsonPath = path.join(inputDir, 'tokenizer.json');
  const tokenizerConfigPath = path.join(inputDir, 'tokenizer_config.json');
  const tokenizerModelPath = path.join(inputDir, 'tokenizer.model');
  const tokenizerJson = await fileExists(tokenizerJsonPath)
    ? await readJson(tokenizerJsonPath, 'tokenizer.json')
    : null;
  const tokenizerConfig = await fileExists(tokenizerConfigPath)
    ? await readJson(tokenizerConfigPath, 'tokenizer_config.json')
    : null;
  const hasTokenizerModel = await fileExists(tokenizerModelPath);

  const sourceFiles = [];
  const uniquePaths = new Set(tensors.map((tensor) => normalizePath(tensor.sourcePath)));
  for (const sourcePath of uniquePaths) {
    const stats = await getPathStats(sourcePath, `source shard (${sourcePath})`);
    sourceFiles.push({ path: sourcePath, size: Number(stats.size) });
  }

  return {
    sourceKind: 'safetensors',
    sourceRoot: inputDir,
    sourcePathForModelId: inputDir,
    config,
    tensors,
    architectureHint,
    architecture,
    sourceQuantization: inferSourceWeightQuantization(tensors),
    tokenizerJson,
    tokenizerConfig,
    tokenizerModelName: hasTokenizerModel ? 'tokenizer.model' : null,
    tokenizerJsonPath: tokenizerJsonPath,
    tokenizerModelPath: hasTokenizerModel ? tokenizerModelPath : null,
    sourceFiles,
  };
}

async function parseGgufInput(ggufPath) {
  const ggufStats = await getPathStats(ggufPath, 'GGUF file');
  const fileSize = Number(ggufStats.size);
  const ggufSource = {
    sourceType: 'node-file',
    name: path.basename(ggufPath),
    size: fileSize,
    file: {
      name: path.basename(ggufPath),
      size: fileSize,
    },
    async readRange(offset, length) {
      return readRange(ggufPath, offset, length);
    },
  };

  const parseGGUFHeaderFromSource = async (source) => {
    const resolved = source && typeof source.readRange === 'function' ? source : ggufSource;
    const readSize = Math.min(resolved.size, HEADER_READ_SIZE);
    const header = await resolved.readRange(0, readSize);
    const info = parseGGUFHeader(toArrayBuffer(header, `gguf header (${ggufPath})`));
    return {
      ...info,
      fileSize: resolved.size,
    };
  };

  const parsed = await parseGGUFModel({
    file: ggufSource,
    parseGGUFHeaderFromSource,
    normalizeTensorSource(source) {
      if (source && typeof source.readRange === 'function' && Number.isFinite(source.size)) {
        return source;
      }
      return ggufSource;
    },
    onProgress() {},
    signal: null,
  });

  const tensors = parsed.tensors.map((tensor) => ({
    ...tensor,
    sourcePath: ggufPath,
  }));

  return {
    sourceKind: 'gguf',
    sourceRoot: path.dirname(ggufPath),
    sourcePathForModelId: ggufPath,
    config: parsed.config,
    tensors,
    architectureHint: parsed.architecture,
    architecture: extractArchitecture({}, parsed.config || {}),
    sourceQuantization: parsed.quantization ?? inferSourceWeightQuantization(tensors),
    tokenizerJson: null,
    tokenizerConfig: null,
    tokenizerModelName: null,
    tokenizerJsonPath: null,
    tokenizerModelPath: null,
    sourceFiles: [{ path: ggufPath, size: fileSize }],
  };
}

function assertSupportedSourceDtypes(tensors, sourceKind) {
  const unsupported = new Set();
  for (const tensor of tensors) {
    const dtype = String(tensor?.dtype || '').trim().toUpperCase();
    if (!dtype) {
      unsupported.add('(empty)');
      continue;
    }
    if (!SUPPORTED_SOURCE_DTYPES.has(dtype)) {
      unsupported.add(dtype);
    }
  }
  if (unsupported.size > 0) {
    throw new Error(
      `Unsupported ${sourceKind} tensor dtypes for direct-source runtime: ` +
      `${Array.from(unsupported).sort((a, b) => a.localeCompare(b)).join(', ')}. ` +
      'Convert to RDRR first for this model.'
    );
  }
}

function buildNodeFileReaders() {
  const readRangeFromFile = async (filePath, offset, length) => readRange(filePath, offset, length);
  const readText = async (filePath) => {
    try {
      return await fs.readFile(filePath, 'utf8');
    } catch (error) {
      if (error?.code === 'ENOENT') {
        return null;
      }
      throw error;
    }
  };
  const readBinary = async (filePath) => {
    const bytes = await fs.readFile(filePath);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  };
  return {
    readRange: readRangeFromFile,
    readText,
    readBinary,
  };
}

export async function resolveNodeSourceRuntimeBundle(options = {}) {
  const inputPath = normalizePath(options.inputPath);
  if (!inputPath) {
    throw new Error('node source runtime: inputPath is required.');
  }
  const resolvedInputPath = path.resolve(inputPath);
  const stats = await getPathStats(resolvedInputPath, 'inputPath');

  let parsed = null;
  if (stats.isFile()) {
    if (!isGgufPath(resolvedInputPath)) {
      return null;
    }
    parsed = await parseGgufInput(resolvedInputPath);
  } else if (stats.isDirectory()) {
    if (await fileExists(path.join(resolvedInputPath, 'manifest.json'))) {
      return null;
    }
    parsed = await parseSafetensorsInput(resolvedInputPath);
    if (!parsed) {
      const entries = await fs.readdir(resolvedInputPath, { withFileTypes: true });
      const ggufFiles = entries
        .filter((entry) => entry.isFile() && isGgufPath(entry.name))
        .map((entry) => entry.name)
        .sort((left, right) => left.localeCompare(right));
      if (ggufFiles.length === 1) {
        parsed = await parseGgufInput(path.join(resolvedInputPath, ggufFiles[0]));
      } else if (ggufFiles.length > 1) {
        throw new Error(
          `node source runtime: multiple GGUF files found in "${resolvedInputPath}": ${ggufFiles.join(', ')}.`
        );
      }
    }
  } else {
    return null;
  }

  if (!parsed) {
    return null;
  }

  assertSupportedSourceDtypes(parsed.tensors, parsed.sourceKind);

  const converterConfig = createConverterConfig({
    output: {
      modelBaseId: options.modelId || null,
    },
    inference: {
      execution: SOURCE_RUNTIME_EXECUTION_OVERRIDE,
    },
  });
  const plan = resolveConversionPlan({
    rawConfig: parsed.config,
    tensors: parsed.tensors,
    converterConfig,
    sourceQuantization: parsed.sourceQuantization,
    modelKind: 'transformer',
    architectureHint: parsed.architectureHint,
    architectureConfig: parsed.architecture,
    includePresetOverrideHint: true,
  });

  const modelId = resolveModelIdHint(
    options.modelId || null,
    plan,
    parsed.sourceKind,
    parsed.sourcePathForModelId
  );
  const { manifest, shardSources } = await buildSourceRuntimeBundle({
    modelId,
    modelName: modelId,
    modelType: plan.modelType,
    architecture: parsed.architecture,
    architectureHint: parsed.architectureHint,
    rawConfig: parsed.config,
    inference: plan.manifestInference,
    tensors: parsed.tensors,
    sourceFiles: parsed.sourceFiles,
    sourceQuantization: parsed.sourceQuantization,
    quantizationInfo: plan.quantizationInfo,
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    tokenizerJson: parsed.tokenizerJson,
    tokenizerConfig: parsed.tokenizerConfig,
    tokenizerModelName: parsed.tokenizerModelName,
  });

  const readers = buildNodeFileReaders();
  const storageContext = createSourceStorageContext({
    manifest,
    shardSources,
    readRange: readers.readRange,
    readText: readers.readText,
    readBinary: readers.readBinary,
    tokenizerJsonPath: parsed.tokenizerJsonPath,
    tokenizerModelPath: parsed.tokenizerModelPath,
    verifyHashes: false,
  });

  log.info(
    'NodeSourceRuntime',
    `Source runtime ready: ${manifest.modelId} (${parsed.sourceKind}, ${parsed.tensors.length} tensors)`
  );

  return {
    manifest,
    storageContext,
    sourceKind: parsed.sourceKind,
    sourceRoot: parsed.sourceRoot,
  };
}

