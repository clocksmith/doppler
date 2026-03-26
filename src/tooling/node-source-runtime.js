import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import {
  HEADER_READ_SIZE,
  createConverterConfig,
  DEFAULT_EXECUTION_V1_SESSION,
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
  steps: [],
};

function cloneExecutionSession() {
  if (typeof structuredClone === 'function') {
    return structuredClone(DEFAULT_EXECUTION_V1_SESSION);
  }
  return JSON.parse(JSON.stringify(DEFAULT_EXECUTION_V1_SESSION));
}

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

async function readFileBytes(filePath, label) {
  try {
    const bytes = await fs.readFile(filePath);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to read ${label} "${filePath}": ${message}`);
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
    let pos = 0;
    while (pos < out.length) {
      const { bytesRead } = await handle.read(out, pos, out.length - pos, start + pos);
      if (bytesRead === 0) break;
      pos += bytesRead;
    }
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
  const embeddingPostprocessor = parsedTransformer.embeddingPostprocessor ?? null;
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
  const auxiliaryFiles = [
    { path: configPath, size: Number((await getPathStats(configPath, 'config.json')).size), kind: 'config' },
    ...(hasIndex
      ? [{
        path: path.join(inputDir, 'model.safetensors.index.json'),
        size: Number((await getPathStats(path.join(inputDir, 'model.safetensors.index.json'), 'model.safetensors.index.json')).size),
        kind: 'safetensors_index',
      }]
      : []),
    ...(tokenizerJson
      ? [{
        path: tokenizerJsonPath,
        size: Number((await getPathStats(tokenizerJsonPath, 'tokenizer.json')).size),
        kind: 'tokenizer_json',
      }]
      : []),
    ...(tokenizerConfig
      ? [{
        path: tokenizerConfigPath,
        size: Number((await getPathStats(tokenizerConfigPath, 'tokenizer_config.json')).size),
        kind: 'tokenizer_config',
      }]
      : []),
    ...(hasTokenizerModel
      ? [{
        path: tokenizerModelPath,
        size: Number((await getPathStats(tokenizerModelPath, 'tokenizer.model')).size),
        kind: 'tokenizer_model',
      }]
      : []),
  ];

  return {
    sourceKind: 'safetensors',
    sourceRoot: inputDir,
    sourcePathForModelId: inputDir,
    config,
    tensors,
    architectureHint,
    embeddingPostprocessor,
    architecture,
    sourceQuantization: inferSourceQuantizationForSourceRuntime(tensors, 'safetensors'),
    tokenizerJson,
    tokenizerConfig,
    tokenizerModelName: hasTokenizerModel ? 'tokenizer.model' : null,
    tokenizerJsonPath: tokenizerJsonPath,
    tokenizerConfigPath: tokenizerConfigPath,
    tokenizerModelPath: hasTokenizerModel ? tokenizerModelPath : null,
    sourceFiles,
    auxiliaryFiles,
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
    sourceQuantization: parsed.quantization ?? inferSourceQuantizationForSourceRuntime(tensors, 'gguf'),
    tokenizerJson: null,
    tokenizerConfig: null,
    tokenizerModelName: null,
    tokenizerJsonPath: null,
    tokenizerConfigPath: null,
    tokenizerModelPath: null,
    sourceFiles: [{ path: ggufPath, size: fileSize }],
    auxiliaryFiles: [],
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

function inferSourceQuantizationForSourceRuntime(tensors, sourceKind) {
  try {
    return inferSourceWeightQuantization(tensors);
  } catch (error) {
    const dtypes = new Set();
    for (const tensor of tensors) {
      const dtype = String(tensor?.dtype || '').trim().toUpperCase();
      if (dtype) dtypes.add(dtype);
    }
    const hasLowPrecision = dtypes.has('F16') || dtypes.has('BF16');
    const onlyLowAndF32 = dtypes.size > 0 && Array.from(dtypes).every(
      (dtype) => dtype === 'F16' || dtype === 'BF16' || dtype === 'F32'
    );
    if (hasLowPrecision && onlyLowAndF32) {
      log.warn(
        'NodeSourceRuntime',
        `Mixed ${sourceKind} tensor dtypes detected (${Array.from(dtypes).sort((a, b) => a.localeCompare(b)).join(', ')}). ` +
        'Using F32 source quantization for direct-source parity.'
      );
      return 'F32';
    }
    throw error;
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

// Source dtype → compute precision mapping for source-runtime inference.
// BF16/F32 sources require f32 compute (BF16 has no native WebGPU support).
// Quantized formats require f32 compute for dequantization accuracy.
// F16 sources can use f16 compute directly.
const SOURCE_QUANT_COMPUTE_MAP = {
  'F16': 'f16',
  'BF16': 'f32',
  'F32': 'f32',
  'Q4_K': 'f32',
  'Q4_K_M': 'f32',
  'Q6_K': 'f32',
};
const SOURCE_COMPUTE_DEFAULT = 'f16';

function resolveSourceRuntimeComputePrecision(tensors, sourceQuantization) {
  const dtypes = new Set();
  for (const tensor of Array.isArray(tensors) ? tensors : []) {
    const dtype = String(tensor?.dtype || '').trim().toUpperCase();
    if (dtype) {
      dtypes.add(dtype);
    }
  }
  // If any tensor requires f32 compute, use f32 for all.
  for (const dtype of dtypes) {
    if (SOURCE_QUANT_COMPUTE_MAP[dtype] === 'f32') {
      return 'f32';
    }
  }

  const normalized = String(sourceQuantization || '').trim().toUpperCase();
  return SOURCE_QUANT_COMPUTE_MAP[normalized] ?? SOURCE_COMPUTE_DEFAULT;
}

async function addHashesToFileEntries(entries, hashAlgorithm) {
  const normalized = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const filePath = normalizePath(entry?.path);
    if (!filePath) continue;
    const stats = await getPathStats(filePath, `source asset (${filePath})`);
    normalized.push({
      ...entry,
      path: filePath,
      size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : Number(stats.size),
      hash: await computeFileHash(filePath, hashAlgorithm),
      hashAlgorithm,
    });
  }
  return normalized;
}

async function computeFileHash(filePath, hashAlgorithm) {
  return new Promise((resolve, reject) => {
    const hash = createHash(hashAlgorithm);
    const stream = createReadStream(filePath);

    stream.on('data', (chunk) => {
      hash.update(chunk);
    });
    stream.on('end', () => {
      resolve(hash.digest('hex'));
    });
    stream.on('error', (error) => {
      const message = error instanceof Error ? error.message : String(error);
      reject(new Error(`Failed to stream source asset "${filePath}" for hashing: ${message}`));
    });
  });
}

export async function resolveNodeSourceRuntimeBundle(options = {}) {
  const inputPath = normalizePath(options.inputPath);
  if (!inputPath) {
    throw new Error('node source runtime: inputPath is required.');
  }
  const verifyHashes = options.verifyHashes === true;
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
    quantization: {
      computePrecision: resolveSourceRuntimeComputePrecision(parsed.tensors, parsed.sourceQuantization),
    },
    output: {
      modelBaseId: options.modelId || null,
    },
    session: cloneExecutionSession(),
    execution: SOURCE_RUNTIME_EXECUTION_OVERRIDE,
  });
  const plan = resolveConversionPlan({
    rawConfig: parsed.config,
    tensors: parsed.tensors,
    converterConfig,
    sourceQuantization: parsed.sourceQuantization,
    modelKind: 'transformer',
    architectureHint: parsed.architectureHint,
    architectureConfig: parsed.architecture,
  });

  const modelId = resolveModelIdHint(
    options.modelId || null,
    plan,
    parsed.sourceKind,
    parsed.sourcePathForModelId
  );
  const hashAlgorithm = converterConfig.manifest.hashAlgorithm;
  const sourceFiles = await addHashesToFileEntries(parsed.sourceFiles, hashAlgorithm);
  const auxiliaryFiles = await addHashesToFileEntries(parsed.auxiliaryFiles, hashAlgorithm);
  const { manifest, shardSources } = await buildSourceRuntimeBundle({
    modelId,
    modelName: modelId,
    modelType: plan.modelType,
    sourceKind: parsed.sourceKind,
    architecture: parsed.architecture,
    architectureHint: parsed.architectureHint,
    rawConfig: parsed.config,
    inference: plan.manifestInference,
    tensors: parsed.tensors,
    embeddingPostprocessor: parsed.embeddingPostprocessor ?? null,
    sourceFiles,
    auxiliaryFiles,
    sourceQuantization: parsed.sourceQuantization,
    quantizationInfo: plan.quantizationInfo,
    hashAlgorithm,
    tokenizerJson: parsed.tokenizerJson,
    tokenizerConfig: parsed.tokenizerConfig,
    tokenizerModelName: parsed.tokenizerModelName,
    tokenizerJsonPath: parsed.tokenizerJsonPath,
    tokenizerConfigPath: parsed.tokenizerConfigPath,
    tokenizerModelPath: parsed.tokenizerModelPath,
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
    verifyHashes,
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
