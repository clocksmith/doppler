import {
  createConverterConfig,
  HEADER_READ_SIZE,
} from '../../config/schema/index.js';
import { DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS } from '../../config/schema/execution-v1.schema.js';
import { extractArchitecture } from '../../converter/core.js';
import {
  inferSourceWeightQuantization,
  resolveConversionPlan,
  resolveConvertedModelId,
} from '../../converter/conversion-plan.js';
import { parseGGUFModel } from '../../converter/parsers/gguf.js';
import { parseTransformerModel } from '../../converter/parsers/transformer.js';
import { parseGGUFHeader } from '../../formats/gguf/types.js';
import { parseSafetensorsHeader } from '../../formats/safetensors/types.js';
import { log } from '../../debug/index.js';
import { computeHash } from '../../storage/shard-manager.js';
import {
  buildSourceRuntimeBundle,
  createSourceStorageContext,
  getSourceRuntimeMetadata,
} from '../../tooling/source-runtime-bundle.js';

const SUPPORTED_SOURCE_DTYPES = new Set([
  'F32',
  'F16',
  'BF16',
  'Q4_K',
  'Q4_K_M',
  'Q6_K',
]);

const ZERO_DIGEST = 'sha256:' + '0'.repeat(64);

const SOURCE_RUNTIME_SESSION_DEFAULTS = {
  compute: {
    defaults: { ...DEFAULT_EXECUTION_V1_COMPUTE_DEFAULTS },
  },
  kvcache: null,
  decodeLoop: null,
};

const SOURCE_RUNTIME_EXECUTION = {
  kernels: {
    embed: { kernel: 'gather_f16.wgsl', entry: 'main', digest: ZERO_DIGEST },
  },
  preLayer: [['embed', 'embed', 'embed_tokens']],
  decode: [],
  prefill: [],
  postLayer: [],
  policies: {
    unsupportedPrecision: 'error',
    dtypeTransition: 'require_cast_step',
    unresolvedKernel: 'error',
  },
};

const SOURCE_RUNTIME_INFERENCE = {
  attention: {
    slidingWindow: null,
    attnLogitSoftcapping: null,
    queryKeyNorm: false,
    attentionOutputGate: false,
    causal: true,
    attentionBias: false,
  },
  normalization: {
    rmsNormWeightOffset: false,
    rmsNormEps: 1e-6,
  },
  ffn: {
    activation: 'gelu',
    gatedActivation: true,
    swigluLimit: null,
  },
  rope: {
    ropeTheta: 10000,
    partialRotaryFactor: 1.0,
    ropeInterleaved: false,
  },
  output: {
    scaleEmbeddings: false,
    tieWordEmbeddings: false,
    embeddingTranspose: false,
    embeddingVocabSize: null,
    finalLogitSoftcapping: null,
  },
  chatTemplate: { type: 'none' },
  layerPattern: { type: 'uniform' },
};

function normalizeRelativePath(value) {
  return String(value || '')
    .replace(/\\/g, '/')
    .replace(/^\.\/+/, '')
    .replace(/^\/+/, '')
    .trim();
}

function joinPath(base, relativePath) {
  const root = String(base || '').replace(/\/+$/, '');
  const rel = normalizeRelativePath(relativePath);
  return rel ? `${root}/${rel}` : root;
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

function toUint8Array(value) {
  return value instanceof Uint8Array ? value : new Uint8Array(value);
}

function decodeText(value) {
  return new TextDecoder().decode(value);
}

function ensureJsonObject(raw, label) {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return raw;
}

async function listBridgeFilesRecursive(bridgeClient, rootPath) {
  const files = [];
  const base = String(rootPath || '').replace(/\/+$/, '');

  async function walk(relativePath = '') {
    const absolutePath = relativePath ? joinPath(base, relativePath) : base;
    const entries = await bridgeClient.list(absolutePath);
    for (const entry of entries) {
      const name = String(entry?.name || '');
      if (!name) continue;
      const childRelative = relativePath
        ? `${normalizeRelativePath(relativePath)}/${name}`
        : name;
      if (entry.isDir) {
        await walk(childRelative);
      } else {
        const normalized = normalizeRelativePath(childRelative);
        files.push({
          relativePath: normalized,
          absolutePath: joinPath(base, normalized),
          size: Number(entry?.size) || 0,
        });
      }
    }
  }

  await walk('');
  files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
  return files;
}

function indexBridgeFiles(files) {
  const map = new Map();
  for (const file of files) {
    map.set(file.relativePath, file);
  }
  return map;
}

function hasModelManifest(fileIndex) {
  return fileIndex.has('manifest.json');
}

function detectBridgeSourceFormat(fileIndex) {
  const relativePaths = Array.from(fileIndex.keys());
  const ggufFiles = relativePaths.filter((path) => path.toLowerCase().endsWith('.gguf'));
  if (ggufFiles.length === 1) {
    return { kind: 'gguf', ggufPath: ggufFiles[0] };
  }

  const hasConfig = fileIndex.has('config.json');
  const hasSafetensors = relativePaths.some((path) => path.toLowerCase().endsWith('.safetensors'));
  const hasSafetensorsIndex = fileIndex.has('model.safetensors.index.json');
  if (hasConfig && (hasSafetensors || hasSafetensorsIndex)) {
    return { kind: 'safetensors', ggufPath: null };
  }

  return null;
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

async function readBridgeRange(bridgeClient, fileEntry, offset, length) {
  return bridgeClient.read(fileEntry.absolutePath, offset, length);
}

async function readBridgeAllBytes(bridgeClient, fileEntry, label) {
  const size = Number(fileEntry?.size) || 0;
  if (size < 0) {
    throw new Error(`Invalid bridge file size for ${label}.`);
  }
  return readBridgeRange(bridgeClient, fileEntry, 0, size);
}

async function readBridgeTextFile(bridgeClient, fileEntry, label) {
  const size = Number(fileEntry?.size) || 0;
  if (size <= 0) {
    throw new Error(`Bridge file "${label}" is empty.`);
  }
  const bytes = await readBridgeRange(bridgeClient, fileEntry, 0, size);
  return decodeText(bytes);
}

async function readBridgeJsonFile(bridgeClient, fileEntry, label) {
  const text = await readBridgeTextFile(bridgeClient, fileEntry, label);
  try {
    return ensureJsonObject(JSON.parse(text), label);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON in ${label}: ${message}`);
  }
}

async function readSafetensorsHeaderFromBridge(bridgeClient, fileEntry) {
  const prefix = await readBridgeRange(bridgeClient, fileEntry, 0, 8);
  const prefixBytes = toUint8Array(prefix);
  if (prefixBytes.byteLength < 8) {
    throw new Error(`Invalid safetensors header prefix for "${fileEntry.relativePath}"`);
  }
  const headerSize = Number(new DataView(toArrayBuffer(prefixBytes, 'safetensors prefix')).getBigUint64(0, true));
  const header = await readBridgeRange(bridgeClient, fileEntry, 8, headerSize);
  const full = new Uint8Array(8 + headerSize);
  full.set(prefixBytes, 0);
  full.set(toUint8Array(header), 8);
  return parseSafetensorsHeader(full.buffer);
}

async function parseBridgeSafetensorsModel(bridgeClient, fileIndex) {
  const parsedTransformer = await parseTransformerModel({
    async readJson(path, label = 'json') {
      const entry = fileIndex.get(normalizeRelativePath(path));
      if (!entry) {
        throw new Error(`Missing ${label} (${path})`);
      }
      return readBridgeJsonFile(bridgeClient, entry, `${label} (${path})`);
    },
    async fileExists(path) {
      return fileIndex.has(normalizeRelativePath(path));
    },
    async loadSingleSafetensors(path) {
      const normalized = normalizeRelativePath(path);
      const entry = fileIndex.get(normalized);
      if (!entry) {
        throw new Error(`Missing safetensors file (${path})`);
      }
      const parsed = await readSafetensorsHeaderFromBridge(bridgeClient, entry);
      return parsed.tensors.map((tensor) => ({
        ...tensor,
        sourcePath: normalized,
      }));
    },
    async loadShardedSafetensors(indexJson) {
      const shardFiles = [...new Set(Object.values(indexJson.weight_map || {}))]
        .map((path) => normalizeRelativePath(path));
      const tensors = [];
      for (const shardPath of shardFiles) {
        const entry = fileIndex.get(shardPath);
        if (!entry) {
          throw new Error(`Missing safetensors shard (${shardPath})`);
        }
        const parsed = await readSafetensorsHeaderFromBridge(bridgeClient, entry);
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
  const tokenizerJson = fileIndex.has('tokenizer.json')
    ? await readBridgeJsonFile(bridgeClient, fileIndex.get('tokenizer.json'), 'tokenizer.json')
    : null;
  const tokenizerConfig = fileIndex.has('tokenizer_config.json')
    ? await readBridgeJsonFile(bridgeClient, fileIndex.get('tokenizer_config.json'), 'tokenizer_config.json')
    : null;
  const tokenizerModelName = fileIndex.has('tokenizer.model') ? 'tokenizer.model' : null;

  return {
    sourceKind: 'safetensors',
    config,
    tensors,
    architectureHint,
    embeddingPostprocessor,
    architecture,
    sourceQuantization: inferSourceWeightQuantization(tensors),
    tokenizerJson,
    tokenizerConfig,
    tokenizerModelName,
    sourceFiles: Array.from(new Set(tensors.map((tensor) => normalizeRelativePath(tensor.sourcePath))))
      .map((path) => {
        const entry = fileIndex.get(path);
        if (!entry) {
          throw new Error(`Missing source file entry for "${path}"`);
        }
        return { path, size: entry.size };
      }),
    auxiliaryFiles: [
      { path: 'config.json', size: Number(fileIndex.get('config.json')?.size || 0), kind: 'config' },
      ...(fileIndex.has('model.safetensors.index.json')
        ? [{
          path: 'model.safetensors.index.json',
          size: Number(fileIndex.get('model.safetensors.index.json')?.size || 0),
          kind: 'safetensors_index',
        }]
        : []),
      ...(fileIndex.has('tokenizer.json')
        ? [{
          path: 'tokenizer.json',
          size: Number(fileIndex.get('tokenizer.json')?.size || 0),
          kind: 'tokenizer_json',
        }]
        : []),
      ...(fileIndex.has('tokenizer_config.json')
        ? [{
          path: 'tokenizer_config.json',
          size: Number(fileIndex.get('tokenizer_config.json')?.size || 0),
          kind: 'tokenizer_config',
        }]
        : []),
      ...(fileIndex.has('tokenizer.model')
        ? [{
          path: 'tokenizer.model',
          size: Number(fileIndex.get('tokenizer.model')?.size || 0),
          kind: 'tokenizer_model',
        }]
        : []),
    ],
    tokenizerJsonPath: fileIndex.has('tokenizer.json') ? 'tokenizer.json' : null,
    tokenizerConfigPath: fileIndex.has('tokenizer_config.json') ? 'tokenizer_config.json' : null,
    tokenizerModelPath: fileIndex.has('tokenizer.model') ? 'tokenizer.model' : null,
  };
}

async function parseBridgeGGUFModel(bridgeClient, fileIndex, ggufRelativePath) {
  const ggufEntry = fileIndex.get(ggufRelativePath);
  if (!ggufEntry) {
    throw new Error(`Missing GGUF file (${ggufRelativePath})`);
  }

  const ggufSource = {
    sourceType: 'bridge-file',
    name: ggufRelativePath,
    size: ggufEntry.size,
    file: {
      name: ggufRelativePath,
      size: ggufEntry.size,
    },
    async readRange(offset, length) {
      return readBridgeRange(bridgeClient, ggufEntry, offset, length);
    },
  };

  const parseGGUFHeaderFromSource = async (source) => {
    const resolved = source && typeof source.readRange === 'function' ? source : ggufSource;
    const readSize = Math.min(resolved.size, HEADER_READ_SIZE);
    const buffer = await resolved.readRange(0, readSize);
    const info = parseGGUFHeader(toArrayBuffer(buffer, `gguf header (${ggufRelativePath})`));
    return {
      ...info,
      fileSize: resolved.size,
    };
  };

  const parsedGGUF = await parseGGUFModel({
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

  const tensors = parsedGGUF.tensors.map((tensor) => ({
    ...tensor,
    sourcePath: ggufRelativePath,
  }));
  const architecture = extractArchitecture({}, parsedGGUF.config || {});

  return {
    sourceKind: 'gguf',
    config: parsedGGUF.config,
    tensors,
    architectureHint: parsedGGUF.architecture,
    architecture,
    sourceQuantization: parsedGGUF.quantization ?? inferSourceWeightQuantization(tensors),
    tokenizerJson: null,
    tokenizerConfig: null,
    tokenizerModelName: null,
    sourceFiles: [{ path: ggufRelativePath, size: ggufEntry.size }],
    auxiliaryFiles: [],
    tokenizerJsonPath: null,
    tokenizerConfigPath: null,
    tokenizerModelPath: null,
  };
}

async function parseBridgeSourceModel(bridgeClient, localPath) {
  const files = await listBridgeFilesRecursive(bridgeClient, localPath);
  const fileIndex = indexBridgeFiles(files);
  if (hasModelManifest(fileIndex)) {
    return null;
  }

  const detected = detectBridgeSourceFormat(fileIndex);
  if (!detected) {
    return null;
  }

  if (detected.kind === 'gguf') {
    return parseBridgeGGUFModel(bridgeClient, fileIndex, detected.ggufPath);
  }

  return parseBridgeSafetensorsModel(bridgeClient, fileIndex);
}

function resolveModelIdHint(requestedModelId, plan, sourceKind) {
  const explicit = String(requestedModelId || '').trim();
  if (explicit) {
    return resolveConvertedModelId({
      explicitModelId: explicit,
      converterConfig: null,
      detectedModelId: explicit,
      quantizationInfo: plan.quantizationInfo,
    }) || explicit;
  }
  const sourcePrefix = sourceKind === 'gguf' ? 'gguf' : 'safetensors';
  return resolveConvertedModelId({
    explicitModelId: `${sourcePrefix}-runtime`,
    converterConfig: null,
    detectedModelId: `${sourcePrefix}-runtime`,
    quantizationInfo: plan.quantizationInfo,
  }) || `${sourcePrefix}-runtime`;
}

function createBridgeFileReaders(bridgeClient, fileMap, rootPath) {
  const map = fileMap;

  const resolveEntry = (pathHint) => {
    const hint = normalizeRelativePath(pathHint);
    if (!hint) {
      return null;
    }
    const direct = map.get(hint);
    return direct || null;
  };

  const readRange = async (relativePath, offset, length) => {
    const entry = resolveEntry(relativePath);
    if (!entry) {
      throw new Error(`Missing source shard file: ${relativePath}`);
    }
    return bridgeClient.read(entry.absolutePath, offset, length);
  };

  const readText = async (pathHint) => {
    const entry = resolveEntry(pathHint);
    if (!entry) return null;
    const bytes = await bridgeClient.read(entry.absolutePath, 0, entry.size);
    return decodeText(bytes);
  };

  const readBinary = async (pathHint) => {
    const entry = resolveEntry(pathHint);
    if (!entry) {
      throw new Error(`Missing source binary file: ${pathHint}`);
    }
    return bridgeClient.read(entry.absolutePath, 0, entry.size);
  };

  return {
    rootPath,
    readRange,
    readText,
    readBinary,
  };
}

async function addHashesToBridgeFiles(bridgeClient, fileIndex, entries, hashAlgorithm) {
  const hashedEntries = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const relativePath = normalizeRelativePath(entry?.path);
    if (!relativePath) continue;
    const fileEntry = fileIndex.get(relativePath);
    if (!fileEntry) {
      throw new Error(`Missing bridge file entry for "${relativePath}"`);
    }
    const bytes = await readBridgeAllBytes(bridgeClient, fileEntry, `bridge source asset (${relativePath})`);
    hashedEntries.push({
      ...entry,
      path: relativePath,
      size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : fileEntry.size,
      hash: await computeHash(toUint8Array(bytes), hashAlgorithm),
      hashAlgorithm,
    });
  }
  return hashedEntries;
}

async function resolveBridgeStorageContext(options = {}) {
  const bridgeClient = options.bridgeClient;
  const localPath = options.localPath;
  const manifest = options.manifest;
  const sourceRuntime = getSourceRuntimeMetadata(manifest);
  if (!sourceRuntime) {
    return null;
  }
  const files = await listBridgeFilesRecursive(bridgeClient, localPath);
  const fileMap = indexBridgeFiles(files);
  const readers = createBridgeFileReaders(bridgeClient, fileMap, localPath);
  return createSourceStorageContext({
    manifest,
    readRange: readers.readRange,
    readText: readers.readText,
    readBinary: readers.readBinary,
    verifyHashes: options.verifyHashes !== false,
  });
}

export async function resolveBridgeSourceRuntimeBundle(options = {}) {
  const bridgeClient = options.bridgeClient;
  const localPath = options.localPath;
  const requestedModelId = options.modelId || null;
  const verifyHashes = options.verifyHashes !== false;
  const existingManifest = options.manifest ?? null;

  if (!bridgeClient || typeof bridgeClient.read !== 'function' || typeof bridgeClient.list !== 'function') {
    throw new Error('Bridge source runtime requires a connected bridge client with read/list support.');
  }
  if (!localPath || typeof localPath !== 'string') {
    throw new Error('Bridge source runtime requires localPath.');
  }

  if (existingManifest && getSourceRuntimeMetadata(existingManifest)) {
    const storageContext = await resolveBridgeStorageContext({
      bridgeClient,
      localPath,
      manifest: existingManifest,
      verifyHashes,
    });
    return {
      manifest: existingManifest,
      storageContext,
      sourceKind: getSourceRuntimeMetadata(existingManifest)?.sourceKind ?? 'safetensors',
      sourceRoot: localPath,
    };
  }

  options.onProgress?.({
    stage: 'source-discovery',
    message: 'Scanning source files via bridge...',
  });

  const parsed = await parseBridgeSourceModel(bridgeClient, localPath);
  if (!parsed) {
    return null;
  }

  assertSupportedSourceDtypes(parsed.tensors, parsed.sourceKind);

  const converterConfig = createConverterConfig({
    output: {
      modelBaseId: requestedModelId || null,
    },
    inference: SOURCE_RUNTIME_INFERENCE,
    sessionDefaults: SOURCE_RUNTIME_SESSION_DEFAULTS,
    execution: SOURCE_RUNTIME_EXECUTION,
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

  const modelId = resolveModelIdHint(requestedModelId, plan, parsed.sourceKind);
  const hashAlgorithm = converterConfig.manifest.hashAlgorithm;
  const files = await listBridgeFilesRecursive(bridgeClient, localPath);
  const fileMap = indexBridgeFiles(files);
  const sourceFiles = await addHashesToBridgeFiles(bridgeClient, fileMap, parsed.sourceFiles, hashAlgorithm);
  const auxiliaryFiles = await addHashesToBridgeFiles(
    bridgeClient,
    fileMap,
    parsed.auxiliaryFiles,
    hashAlgorithm
  );
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

  const readers = createBridgeFileReaders(bridgeClient, fileMap, localPath);
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
    'DopplerProvider',
    `Bridge source runtime ready: ${manifest.modelId} (${parsed.sourceKind}, ${parsed.tensors.length} tensors)`
  );

  return {
    manifest,
    storageContext,
    sourceKind: parsed.sourceKind,
    sourceRoot: localPath,
  };
}
