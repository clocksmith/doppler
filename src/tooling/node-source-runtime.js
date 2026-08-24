import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { totalmem } from 'node:os';
import path from 'node:path';
import {
  HEADER_READ_SIZE,
} from '../config/schema/index.js';
import { getRuntimeConfig } from '../config/runtime.js';
import { extractArchitecture } from '../converter/core.js';
import { parseGGUFModel } from '../converter/parsers/gguf.js';
import { parseTransformerModel } from '../converter/parsers/transformer.js';
import { parseGGUFHeader } from '../formats/gguf/types.js';
import { parseTFLiteFromSource } from '../formats/tflite/types.js';
import { log } from '../debug/index.js';
import { formatBytes } from '../storage/quota.js';
import { toArrayBuffer } from '../utils/array-buffer.js';
import {
  createSourceStorageContext,
} from './source-runtime-bundle.js';
import {
  inferSourceQuantizationForSourceRuntime,
  resolveSourceRuntimeBundleFromParsedArtifact,
} from './source-artifact-adapter.js';
import {
  LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
  LITERT_PACKAGE_SOURCE_KIND_TASK,
  appendLiteRTPackageVirtualFiles,
  resolveLiteRTPackageParsedArtifact,
} from './litert-package-runtime.js';
import {
  MAX_NODE_READ_BYTES,
  addHashesToFileEntries,
  buildNodeFileReaders,
  createNodeFileAccess,
  fileExists,
  getPathStats,
  isGgufPath,
  isLiteRTLMPath,
  isLiteRTTaskPath,
  isTflitePath,
  normalizePath,
  readJson,
  readSafetensorsHeaderFromFile,
} from './source-runtime/node-loader.js';

async function collectTokenizerAssets(inputDir) {
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
  return {
    tokenizerJson,
    tokenizerConfig,
    tokenizerModelName: hasTokenizerModel ? 'tokenizer.model' : null,
    tokenizerJsonPath,
    tokenizerConfigPath,
    tokenizerModelPath: hasTokenizerModel ? tokenizerModelPath : null,
  };
}

async function buildTokenizerAuxiliaryFiles(tokenizerAssets) {
  const auxiliaryFiles = [];
  if (tokenizerAssets.tokenizerJson) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerJsonPath,
      size: Number((await getPathStats(tokenizerAssets.tokenizerJsonPath, 'tokenizer.json')).size),
      kind: 'tokenizer_json',
    });
  }
  if (tokenizerAssets.tokenizerConfig) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerConfigPath,
      size: Number((await getPathStats(tokenizerAssets.tokenizerConfigPath, 'tokenizer_config.json')).size),
      kind: 'tokenizer_config',
    });
  }
  if (tokenizerAssets.tokenizerModelPath) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerModelPath,
      size: Number((await getPathStats(tokenizerAssets.tokenizerModelPath, 'tokenizer.model')).size),
      kind: 'tokenizer_model',
    });
  }
  return auxiliaryFiles;
}

function buildPackageTokenizerVirtualFiles(tokenizerAssets) {
  const virtualFiles = [];
  if (tokenizerAssets.tokenizerJson) {
    virtualFiles.push({
      path: 'tokenizer.json',
      offset: 0,
      size: 0,
      kind: 'tokenizer_json',
      externalPath: tokenizerAssets.tokenizerJsonPath,
    });
  }
  if (tokenizerAssets.tokenizerConfig) {
    virtualFiles.push({
      path: 'tokenizer_config.json',
      offset: 0,
      size: 0,
      kind: 'tokenizer_config',
      externalPath: tokenizerAssets.tokenizerConfigPath,
    });
  }
  if (tokenizerAssets.tokenizerModelPath) {
    virtualFiles.push({
      path: 'TOKENIZER_MODEL',
      offset: 0,
      size: 0,
      kind: 'tokenizer_model',
      externalPath: tokenizerAssets.tokenizerModelPath,
    });
  }
  return virtualFiles;
}

function applyPackageTokenizerAssets(parsedArtifact, tokenizerAssets) {
  const next = {
    ...parsedArtifact,
  };
  const hasPackageTokenizer = next.tokenizerJson != null
    || next.tokenizerJsonPath != null
    || next.tokenizerModelName != null
    || next.tokenizerModelPath != null;
  if (tokenizerAssets.tokenizerJson && !hasPackageTokenizer) {
    next.tokenizerJson = tokenizerAssets.tokenizerJson;
    next.tokenizerJsonPath = 'tokenizer.json';
  }
  if (tokenizerAssets.tokenizerConfig && next.tokenizerConfigPath == null) {
    next.tokenizerConfigPath = 'tokenizer_config.json';
  }
  if (tokenizerAssets.tokenizerConfig && next.tokenizerConfig == null) {
    next.tokenizerConfig = tokenizerAssets.tokenizerConfig;
  }
  if (tokenizerAssets.tokenizerModelPath && next.tokenizerModelPath == null) {
    next.tokenizerModelName = 'TOKENIZER_MODEL';
    next.tokenizerModelPath = 'TOKENIZER_MODEL';
  }
  return next;
}

async function parseSafetensorsInput(inputDir, fileAccess) {
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
      const parsed = await readSafetensorsHeaderFromFile(filePath, fileAccess);
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
        const parsed = await readSafetensorsHeaderFromFile(shardPath, fileAccess);
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
  const tokenizerAssets = await collectTokenizerAssets(inputDir);

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
    ...await buildTokenizerAuxiliaryFiles(tokenizerAssets),
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
    sourceQuantization: inferSourceQuantizationForSourceRuntime(tensors, 'safetensors', {
      logCategory: 'NodeSourceRuntime',
    }),
    tokenizerJson: tokenizerAssets.tokenizerJson,
    tokenizerConfig: tokenizerAssets.tokenizerConfig,
    tokenizerModelName: tokenizerAssets.tokenizerModelName,
    tokenizerJsonPath: tokenizerAssets.tokenizerJsonPath,
    tokenizerConfigPath: tokenizerAssets.tokenizerConfigPath,
    tokenizerModelPath: tokenizerAssets.tokenizerModelPath,
    sourceFiles,
    auxiliaryFiles,
  };
}

async function parseTfliteInput(tflitePath, fileAccess) {
  const tfliteStats = await getPathStats(tflitePath, 'TFLite file');
  const inputDir = path.dirname(tflitePath);
  const configPath = path.join(inputDir, 'config.json');
  if (!(await fileExists(configPath))) {
    throw new Error(
      `node source runtime: config.json is required next to TFLite source "${tflitePath}".`
    );
  }

  const config = await readJson(configPath, 'config.json');
  const parsedTFLite = await parseTFLiteFromSource({
    name: path.basename(tflitePath),
    size: Number(tfliteStats.size),
    async readRange(offset, length) {
      return fileAccess.readRange(tflitePath, offset, length);
    },
  });
  const tokenizerAssets = await collectTokenizerAssets(inputDir);
  const tensors = parsedTFLite.tensors.map((tensor) => ({
    ...tensor,
    sourcePath: tflitePath,
  }));
  const architecture = extractArchitecture(config, null);
  const architectureHint = config.architectures?.[0] ?? config.model_type ?? '';
  const auxiliaryFiles = [
    { path: configPath, size: Number((await getPathStats(configPath, 'config.json')).size), kind: 'config' },
    ...await buildTokenizerAuxiliaryFiles(tokenizerAssets),
  ];

  return {
    sourceKind: 'tflite',
    sourceRoot: inputDir,
    sourcePathForModelId: tflitePath,
    config,
    tensors,
    architectureHint,
    embeddingPostprocessor: null,
    architecture,
    sourceQuantization: parsedTFLite.sourceQuantization,
    tokenizerJson: tokenizerAssets.tokenizerJson,
    tokenizerConfig: tokenizerAssets.tokenizerConfig,
    tokenizerModelName: tokenizerAssets.tokenizerModelName,
    tokenizerJsonPath: tokenizerAssets.tokenizerJsonPath,
    tokenizerConfigPath: tokenizerAssets.tokenizerConfigPath,
    tokenizerModelPath: tokenizerAssets.tokenizerModelPath,
    sourceFiles: [{ path: tflitePath, size: Number(tfliteStats.size) }],
    auxiliaryFiles,
  };
}

function createNodeVirtualFileReaders(packagePath, virtualFiles, fileAccess) {
  const virtualFileMap = new Map(
    (Array.isArray(virtualFiles) ? virtualFiles : []).map((entry) => [entry.path, entry])
  );

  const resolveVirtualFile = (virtualPath) => {
    const normalized = normalizePath(virtualPath);
    const entry = virtualFileMap.get(normalized);
    if (!entry) {
      throw new Error(`node source runtime: missing package asset "${virtualPath}".`);
    }
    return entry;
  };

  const readRangeFromVirtualFile = async (virtualPath, offset, length) => {
    const entry = resolveVirtualFile(virtualPath);
    if (entry.externalPath) {
      return fileAccess.readRange(entry.externalPath, offset, length);
    }
    const start = Math.max(0, Math.floor(Number(offset) || 0));
    const requested = Math.max(0, Math.floor(Number(length) || 0));
    const available = Math.max(0, entry.size - start);
    const readLength = Math.min(available, requested);
    if (readLength <= 0) {
      return new ArrayBuffer(0);
    }
    return fileAccess.readRange(packagePath, entry.offset + start, readLength);
  };

  const streamRange = async function* (virtualPath, offset, length, options = {}) {
    const entry = resolveVirtualFile(virtualPath);
    if (entry.externalPath) {
      const stats = await getPathStats(entry.externalPath, `package sidecar (${entry.externalPath})`);
      const start = Math.max(0, Math.floor(Number(offset) || 0));
      const requested = Math.max(0, Math.floor(Number(length) || 0));
      const available = Math.max(0, Number(stats.size) - start);
      const readLength = Math.min(available, requested);
      if (readLength <= 0) {
        return;
      }
      const chunkBytesRaw = Number(options?.chunkBytes);
      const highWaterMark = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
        ? Math.floor(chunkBytesRaw)
        : MAX_NODE_READ_BYTES;
      const stream = createReadStream(entry.externalPath, {
        start,
        end: start + readLength - 1,
        highWaterMark,
      });
      for await (const chunk of stream) {
        yield chunk;
      }
      return;
    }
    const start = Math.max(0, Math.floor(Number(offset) || 0));
    const requested = Math.max(0, Math.floor(Number(length) || 0));
    const available = Math.max(0, entry.size - start);
    const readLength = Math.min(available, requested);
    if (readLength <= 0) {
      return;
    }
    const chunkBytesRaw = Number(options?.chunkBytes);
    const highWaterMark = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
      ? Math.floor(chunkBytesRaw)
      : MAX_NODE_READ_BYTES;
    const stream = createReadStream(packagePath, {
      start: entry.offset + start,
      end: entry.offset + start + readLength - 1,
      highWaterMark,
    });
    for await (const chunk of stream) {
      yield chunk;
    }
  };

  const readText = async (virtualPath) => {
    const entry = resolveVirtualFile(virtualPath);
    const bytes = entry.externalPath
      ? await fs.readFile(entry.externalPath)
      : await fileAccess.readRange(packagePath, entry.offset, entry.size);
    return new TextDecoder().decode(bytes);
  };

  const readBinary = async (virtualPath) => {
    const entry = resolveVirtualFile(virtualPath);
    return entry.externalPath
      ? fs.readFile(entry.externalPath)
      : fileAccess.readRange(packagePath, entry.offset, entry.size);
  };

  return {
    virtualFileMap,
    readRange: readRangeFromVirtualFile,
    streamRange,
    readText,
    readBinary,
    close: fileAccess.close,
  };
}

async function addHashesToVirtualEntries(packagePath, virtualFiles, fileAccess, entries, hashAlgorithm) {
  const readers = createNodeVirtualFileReaders(packagePath, virtualFiles, fileAccess);
  const hashedEntries = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const virtualPath = normalizePath(entry?.path);
    if (!virtualPath) {
      continue;
    }
    const descriptor = readers.virtualFileMap.get(virtualPath);
    if (!descriptor) {
      throw new Error(`node source runtime: missing virtual package asset "${virtualPath}" for hashing.`);
    }
    const hash = createHash(hashAlgorithm);
    for await (const chunk of readers.streamRange(virtualPath, 0, descriptor.size)) {
      hash.update(chunk);
    }
    hashedEntries.push({
      ...entry,
      path: virtualPath,
      size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : descriptor.size,
      hash: hash.digest('hex'),
      hashAlgorithm,
    });
  }
  return hashedEntries;
}

async function parseLiteRTPackageInput(packagePath, sourceKind, fileAccess) {
  const stats = await getPathStats(packagePath, `LiteRT package (${sourceKind})`);
  const resolved = await resolveLiteRTPackageParsedArtifact({
    sourceKind,
    sourcePathForModelId: packagePath,
    source: {
      name: path.basename(packagePath),
      size: Number(stats.size),
      async readRange(offset, length) {
        return fileAccess.readRange(packagePath, offset, length);
      },
    },
  });
  const tokenizerAssets = await collectTokenizerAssets(path.dirname(packagePath));
  const virtualFiles = appendLiteRTPackageVirtualFiles(
    resolved.virtualFiles,
    buildPackageTokenizerVirtualFiles(tokenizerAssets)
  );
  const parsedArtifact = applyPackageTokenizerAssets(resolved.parsedArtifact, tokenizerAssets);
  return {
    ...parsedArtifact,
    sourceRoot: packagePath,
    storageReaders: createNodeVirtualFileReaders(packagePath, virtualFiles, fileAccess),
    async hashFileEntries(entries, hashAlgorithm) {
      return addHashesToVirtualEntries(packagePath, virtualFiles, fileAccess, entries, hashAlgorithm);
    },
  };
}

async function parseGgufInput(ggufPath, fileAccess) {
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
      return fileAccess.readRange(ggufPath, offset, length);
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
    sourceQuantization: parsed.quantization ?? inferSourceQuantizationForSourceRuntime(tensors, 'gguf', {
      logCategory: 'NodeSourceRuntime',
    }),
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

function resolveLoadingConfig(runtimeConfig) {
  const loadingConfig = runtimeConfig?.loading ?? getRuntimeConfig().loading;
  if (!loadingConfig || typeof loadingConfig !== 'object') {
    throw new Error('node source runtime: runtime.loading is required.');
  }
  return loadingConfig;
}

function resolveMemoryBudgetConfig(loadingConfig) {
  const budget = loadingConfig?.memoryManagement?.budget;
  if (!budget || typeof budget !== 'object') {
    throw new Error('node source runtime: runtime.loading.memoryManagement.budget is required.');
  }
  return budget;
}

function resolveResidentBudgetBytes(loadingConfig) {
  const budget = resolveMemoryBudgetConfig(loadingConfig);
  if (budget.enabled !== true) {
    return null;
  }

  const explicitMaxResidentBytes = budget.maxResidentBytes;
  if (explicitMaxResidentBytes != null) {
    const normalized = Number(explicitMaxResidentBytes);
    if (!Number.isFinite(normalized) || normalized <= 0) {
      throw new Error(
        'node source runtime: runtime.loading.memoryManagement.budget.maxResidentBytes ' +
        'must be a positive number or null.'
      );
    }
    return Math.floor(normalized);
  }

  const systemMemoryFraction = Number(budget.systemMemoryFraction);
  const reserveBytes = Number(budget.reserveBytes);
  const minimumBudgetBytes = Number(budget.minimumBudgetBytes);
  if (!Number.isFinite(systemMemoryFraction) || systemMemoryFraction <= 0 || systemMemoryFraction > 1) {
    throw new Error(
      'node source runtime: runtime.loading.memoryManagement.budget.systemMemoryFraction ' +
      'must be within (0, 1].'
    );
  }
  if (!Number.isFinite(reserveBytes) || reserveBytes < 0) {
    throw new Error(
      'node source runtime: runtime.loading.memoryManagement.budget.reserveBytes ' +
      'must be a non-negative number.'
    );
  }
  if (!Number.isFinite(minimumBudgetBytes) || minimumBudgetBytes <= 0) {
    throw new Error(
      'node source runtime: runtime.loading.memoryManagement.budget.minimumBudgetBytes ' +
      'must be a positive number.'
    );
  }

  const derived = Math.floor(totalmem() * systemMemoryFraction) - Math.floor(reserveBytes);
  return Math.max(Math.floor(minimumBudgetBytes), derived);
}

function estimateSourceRuntimeTransientBytes(parsed, loadingConfig) {
  const maxTensorBytes = (Array.isArray(parsed?.tensors) ? parsed.tensors : []).reduce((maxBytes, tensor) => {
    const size = Number(tensor?.size);
    return Number.isFinite(size) && size > maxBytes ? Math.floor(size) : maxBytes;
  }, 0);
  const streamConfig = loadingConfig?.storage?.backend?.streaming ?? {};
  const readChunkBytes = Number(streamConfig.readChunkBytes);
  const maxInFlightBytes = Number(streamConfig.maxInFlightBytes);
  const streamWindowBytes = Math.max(
    Number.isFinite(readChunkBytes) && readChunkBytes > 0 ? Math.floor(readChunkBytes) : 0,
    Number.isFinite(maxInFlightBytes) && maxInFlightBytes > 0 ? Math.floor(maxInFlightBytes) : 0,
    MAX_NODE_READ_BYTES
  );
  return Math.max(maxTensorBytes, streamWindowBytes);
}

function assertSourceRuntimeFitsResidentBudget(parsed, loadingConfig, residentBudgetBytes) {
  if (!Number.isFinite(residentBudgetBytes) || residentBudgetBytes <= 0) {
    return;
  }

  const currentRssBytes = typeof process !== 'undefined' && typeof process.memoryUsage === 'function'
    ? process.memoryUsage().rss
    : 0;
  const estimatedTransientBytes = estimateSourceRuntimeTransientBytes(parsed, loadingConfig);
  const projectedResidentBytes = currentRssBytes + estimatedTransientBytes;
  if (projectedResidentBytes <= residentBudgetBytes) {
    return;
  }

  const sourceFiles = Array.isArray(parsed?.sourceFiles) ? parsed.sourceFiles : [];
  const totalSourceBytes = sourceFiles.reduce((totalBytes, entry) => {
    const size = Number(entry?.size);
    return Number.isFinite(size) ? totalBytes + Math.max(0, Math.floor(size)) : totalBytes;
  }, 0);
  const largestSourceBytes = sourceFiles.reduce((maxBytes, entry) => {
    const size = Number(entry?.size);
    return Number.isFinite(size) && size > maxBytes ? Math.floor(size) : maxBytes;
  }, 0);

  throw new Error(
    'node source runtime: direct-source load exceeds resident memory budget. ' +
    `rss=${formatBytes(currentRssBytes)}, transient=${formatBytes(estimatedTransientBytes)}, ` +
    `projected=${formatBytes(projectedResidentBytes)}, budget=${formatBytes(residentBudgetBytes)}, ` +
    `largestSource=${formatBytes(largestSourceBytes)}, totalSource=${formatBytes(totalSourceBytes)}. ` +
    'Lower the model working set or adjust runtime.loading.memoryManagement.budget.'
  );
}

export async function resolveNodeSourceRuntimeBundle(options = {}) {
  const inputPath = normalizePath(options.inputPath);
  if (!inputPath) {
    throw new Error('node source runtime: inputPath is required.');
  }
  const verifyHashes = options.verifyHashes === true;
  const resolvedInputPath = path.resolve(inputPath);
  const stats = await getPathStats(resolvedInputPath, 'inputPath');
  const fileAccess = createNodeFileAccess();

  try {
    let parsed = null;
    if (stats.isFile()) {
      if (isTflitePath(resolvedInputPath)) {
        parsed = await parseTfliteInput(resolvedInputPath, fileAccess);
      }
      if (!parsed && isLiteRTTaskPath(resolvedInputPath)) {
        parsed = await parseLiteRTPackageInput(
          resolvedInputPath,
          LITERT_PACKAGE_SOURCE_KIND_TASK,
          fileAccess
        );
      }
      if (!parsed && isLiteRTLMPath(resolvedInputPath)) {
        parsed = await parseLiteRTPackageInput(
          resolvedInputPath,
          LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
          fileAccess
        );
      }
      if (!parsed && !isGgufPath(resolvedInputPath)) {
        await fileAccess.close();
        return null;
      }
      if (!parsed) {
        parsed = await parseGgufInput(resolvedInputPath, fileAccess);
      }
    } else if (stats.isDirectory()) {
      if (await fileExists(path.join(resolvedInputPath, 'manifest.json'))) {
        await fileAccess.close();
        return null;
      }
      parsed = await parseSafetensorsInput(resolvedInputPath, fileAccess);
      if (!parsed) {
        const entries = await fs.readdir(resolvedInputPath, { withFileTypes: true });
        const ggufFiles = entries
          .filter((entry) => entry.isFile() && isGgufPath(entry.name))
          .map((entry) => entry.name)
          .sort((left, right) => left.localeCompare(right));
        if (ggufFiles.length === 1) {
          parsed = await parseGgufInput(path.join(resolvedInputPath, ggufFiles[0]), fileAccess);
        } else if (ggufFiles.length > 1) {
          throw new Error(
            `node source runtime: multiple GGUF files found in "${resolvedInputPath}": ${ggufFiles.join(', ')}.`
          );
        }
        if (!parsed) {
          const tfliteFiles = entries
            .filter((entry) => entry.isFile() && isTflitePath(entry.name))
            .map((entry) => entry.name)
            .sort((left, right) => left.localeCompare(right));
          if (tfliteFiles.length === 1) {
            parsed = await parseTfliteInput(path.join(resolvedInputPath, tfliteFiles[0]), fileAccess);
          } else if (tfliteFiles.length > 1) {
            throw new Error(
              `node source runtime: multiple TFLite files found in "${resolvedInputPath}": ${tfliteFiles.join(', ')}.`
            );
          }
        }
        if (!parsed) {
          const taskFiles = entries
            .filter((entry) => entry.isFile() && isLiteRTTaskPath(entry.name))
            .map((entry) => entry.name)
            .sort((left, right) => left.localeCompare(right));
          if (taskFiles.length === 1) {
            parsed = await parseLiteRTPackageInput(
              path.join(resolvedInputPath, taskFiles[0]),
              LITERT_PACKAGE_SOURCE_KIND_TASK,
              fileAccess
            );
          } else if (taskFiles.length > 1) {
            throw new Error(
              `node source runtime: multiple LiteRT task files found in "${resolvedInputPath}": ${taskFiles.join(', ')}.`
            );
          }
        }
        if (!parsed) {
          const litertLmFiles = entries
            .filter((entry) => entry.isFile() && isLiteRTLMPath(entry.name))
            .map((entry) => entry.name)
            .sort((left, right) => left.localeCompare(right));
          if (litertLmFiles.length === 1) {
            parsed = await parseLiteRTPackageInput(
              path.join(resolvedInputPath, litertLmFiles[0]),
              LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
              fileAccess
            );
          } else if (litertLmFiles.length > 1) {
            throw new Error(
              `node source runtime: multiple LiteRT-LM files found in "${resolvedInputPath}": ${litertLmFiles.join(', ')}.`
            );
          }
        }
      }
    } else {
      await fileAccess.close();
      return null;
    }

    if (!parsed) {
      await fileAccess.close();
      return null;
    }

    const loadingConfig = resolveLoadingConfig(options.runtimeConfig ?? null);
    const resolvedMemoryBudgetBytes = resolveResidentBudgetBytes(loadingConfig);
    assertSourceRuntimeFitsResidentBudget(parsed, loadingConfig, resolvedMemoryBudgetBytes);
    const {
      model,
      shardSources,
      sourceKind,
    } = await resolveSourceRuntimeBundleFromParsedArtifact({
      parsedArtifact: parsed,
      requestedModelId: options.modelId || null,
      runtimeLabel: 'node source runtime',
      logCategory: 'NodeSourceRuntime',
      hashFileEntries: typeof parsed?.hashFileEntries === 'function'
        ? (entries, hashAlgorithm) => parsed.hashFileEntries(entries, hashAlgorithm)
        : addHashesToFileEntries,
    });

    const readers = parsed?.storageReaders ?? buildNodeFileReaders(fileAccess);
    const storageContext = createSourceStorageContext({
      model,
      shardSources,
      readRange: readers.readRange,
      streamRange: readers.streamRange,
      readText: readers.readText,
      readBinary: readers.readBinary,
      close: readers.close,
      tokenizerJsonPath: parsed.tokenizerJsonPath,
      tokenizerModelPath: parsed.tokenizerModelPath,
      verifyHashes,
      sourceHashesTrusted: true,
    });

    log.info(
      'NodeSourceRuntime',
      `Source runtime ready: ${model.modelId} (${sourceKind}, ${parsed.tensors.length} tensors)`
    );

    return {
      model,
      manifest: model,
      storageContext,
      sourceKind,
      sourceRoot: parsed.sourceRoot,
      resolvedMemoryBudgetBytes,
    };
  } catch (error) {
    await fileAccess.close();
    throw error;
  }
}
