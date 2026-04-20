import {
  HEADER_READ_SIZE,
} from '../../config/schema/index.js';
import { extractArchitecture } from '../../converter/core.js';
import { parseGGUFModel } from '../../converter/parsers/gguf.js';
import { parseTransformerModel } from '../../converter/parsers/transformer.js';
import { parseGGUFHeader } from '../../formats/gguf/types.js';
import { parseSafetensorsHeader } from '../../formats/safetensors/types.js';
import { parseTFLiteFromSource } from '../../formats/tflite/types.js';
import { log } from '../../debug/index.js';
import { computeHash } from '../../storage/shard-manager.js';
import { toArrayBuffer } from '../../utils/array-buffer.js';
import {
  createSourceStorageContext,
  getSourceRuntimeMetadata,
} from '../../tooling/source-runtime-bundle.js';
import {
  inferSourceQuantizationForSourceRuntime,
  resolveSourceRuntimeBundleFromParsedArtifact,
} from '../../tooling/source-artifact-adapter.js';
import {
  LITERT_PACKAGE_SOURCE_KIND_LITERTLM,
  LITERT_PACKAGE_SOURCE_KIND_TASK,
  appendLiteRTPackageVirtualFiles,
  resolveLiteRTPackageParsedArtifact,
} from '../../tooling/litert-package-runtime.js';

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

function isLiteRTTaskPath(value) {
  return String(value || '').toLowerCase().endsWith('.task');
}

function isLiteRTLMPath(value) {
  return String(value || '').toLowerCase().endsWith('.litertlm');
}

function splitBridgeFilePath(targetPath) {
  const normalized = String(targetPath || '').replace(/\\/g, '/').replace(/\/+$/, '');
  const lastSlash = normalized.lastIndexOf('/');
  if (lastSlash < 0) {
    return { parentPath: '.', basename: normalized };
  }
  if (lastSlash === 0) {
    return {
      parentPath: '/',
      basename: normalized.slice(1),
    };
  }
  return {
    parentPath: normalized.slice(0, lastSlash),
    basename: normalized.slice(lastSlash + 1),
  };
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

  const tfliteFiles = relativePaths.filter((path) => path.toLowerCase().endsWith('.tflite'));
  if (tfliteFiles.length === 1) {
    return { kind: 'tflite', ggufPath: null };
  }

  const taskFiles = relativePaths.filter((path) => path.toLowerCase().endsWith('.task'));
  if (taskFiles.length === 1) {
    return { kind: LITERT_PACKAGE_SOURCE_KIND_TASK, ggufPath: null };
  }

  const litertLmFiles = relativePaths.filter((path) => path.toLowerCase().endsWith('.litertlm'));
  if (litertLmFiles.length === 1) {
    return { kind: LITERT_PACKAGE_SOURCE_KIND_LITERTLM, ggufPath: null };
  }

  return null;
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

async function collectBridgeTokenizerAssets(bridgeClient, fileIndex) {
  return {
    tokenizerJson: fileIndex.has('tokenizer.json')
      ? await readBridgeJsonFile(bridgeClient, fileIndex.get('tokenizer.json'), 'tokenizer.json')
      : null,
    tokenizerConfig: fileIndex.has('tokenizer_config.json')
      ? await readBridgeJsonFile(bridgeClient, fileIndex.get('tokenizer_config.json'), 'tokenizer_config.json')
      : null,
    tokenizerModelName: fileIndex.has('tokenizer.model') ? 'tokenizer.model' : null,
    tokenizerJsonPath: fileIndex.has('tokenizer.json') ? 'tokenizer.json' : null,
    tokenizerConfigPath: fileIndex.has('tokenizer_config.json') ? 'tokenizer_config.json' : null,
    tokenizerModelPath: fileIndex.has('tokenizer.model') ? 'tokenizer.model' : null,
  };
}

function buildBridgeTokenizerAuxiliaryFiles(fileIndex, tokenizerAssets) {
  const auxiliaryFiles = [];
  if (tokenizerAssets.tokenizerJsonPath) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerJsonPath,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerJsonPath)?.size || 0),
      kind: 'tokenizer_json',
    });
  }
  if (tokenizerAssets.tokenizerConfigPath) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerConfigPath,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerConfigPath)?.size || 0),
      kind: 'tokenizer_config',
    });
  }
  if (tokenizerAssets.tokenizerModelPath) {
    auxiliaryFiles.push({
      path: tokenizerAssets.tokenizerModelPath,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerModelPath)?.size || 0),
      kind: 'tokenizer_model',
    });
  }
  return auxiliaryFiles;
}

function buildBridgePackageTokenizerVirtualFiles(fileIndex, tokenizerAssets, rootAbsolutePath) {
  const virtualFiles = [];
  if (tokenizerAssets.tokenizerJsonPath) {
    virtualFiles.push({
      path: 'tokenizer.json',
      offset: 0,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerJsonPath)?.size || 0),
      kind: 'tokenizer_json',
      externalPath: joinPath(rootAbsolutePath, tokenizerAssets.tokenizerJsonPath),
    });
  }
  if (tokenizerAssets.tokenizerConfigPath) {
    virtualFiles.push({
      path: 'tokenizer_config.json',
      offset: 0,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerConfigPath)?.size || 0),
      kind: 'tokenizer_config',
      externalPath: joinPath(rootAbsolutePath, tokenizerAssets.tokenizerConfigPath),
    });
  }
  if (tokenizerAssets.tokenizerModelPath) {
    virtualFiles.push({
      path: 'TOKENIZER_MODEL',
      offset: 0,
      size: Number(fileIndex.get(tokenizerAssets.tokenizerModelPath)?.size || 0),
      kind: 'tokenizer_model',
      externalPath: joinPath(rootAbsolutePath, tokenizerAssets.tokenizerModelPath),
    });
  }
  return virtualFiles;
}

function applyBridgePackageTokenizerAssets(parsedArtifact, tokenizerAssets) {
  const next = {
    ...parsedArtifact,
  };
  if (tokenizerAssets.tokenizerJson && next.tokenizerJson == null) {
    next.tokenizerJson = tokenizerAssets.tokenizerJson;
    next.tokenizerJsonPath = 'tokenizer.json';
  }
  if (tokenizerAssets.tokenizerConfig && next.tokenizerConfig == null) {
    next.tokenizerConfig = tokenizerAssets.tokenizerConfig;
  }
  if (tokenizerAssets.tokenizerConfigPath && next.tokenizerConfigPath == null) {
    next.tokenizerConfigPath = 'tokenizer_config.json';
  }
  if (tokenizerAssets.tokenizerModelPath && next.tokenizerModelPath == null) {
    next.tokenizerModelName = 'TOKENIZER_MODEL';
    next.tokenizerModelPath = 'TOKENIZER_MODEL';
  }
  return next;
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
  const tokenizerAssets = await collectBridgeTokenizerAssets(bridgeClient, fileIndex);

  return {
    sourceKind: 'safetensors',
    config,
    tensors,
    architectureHint,
    embeddingPostprocessor,
    architecture,
    sourceQuantization: inferSourceQuantizationForSourceRuntime(tensors, 'safetensors', {
      logCategory: 'DopplerProvider',
    }),
    tokenizerJson: tokenizerAssets.tokenizerJson,
    tokenizerConfig: tokenizerAssets.tokenizerConfig,
    tokenizerModelName: tokenizerAssets.tokenizerModelName,
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
      ...buildBridgeTokenizerAuxiliaryFiles(fileIndex, tokenizerAssets),
    ],
    tokenizerJsonPath: tokenizerAssets.tokenizerJsonPath,
    tokenizerConfigPath: tokenizerAssets.tokenizerConfigPath,
    tokenizerModelPath: tokenizerAssets.tokenizerModelPath,
  };
}

async function parseBridgeTfliteModel(bridgeClient, fileIndex, tfliteRelativePath) {
  const tfliteEntry = fileIndex.get(tfliteRelativePath);
  if (!tfliteEntry) {
    throw new Error(`Missing TFLite file (${tfliteRelativePath})`);
  }
  if (!fileIndex.has('config.json')) {
    throw new Error(
      `Bridge source runtime: config.json is required next to TFLite source "${tfliteRelativePath}".`
    );
  }

  const parsedTFLite = await parseTFLiteFromSource({
    name: tfliteRelativePath,
    size: tfliteEntry.size,
    async readRange(offset, length) {
      return readBridgeRange(bridgeClient, tfliteEntry, offset, length);
    },
  });
  const config = await readBridgeJsonFile(bridgeClient, fileIndex.get('config.json'), 'config.json');
  const tokenizerAssets = await collectBridgeTokenizerAssets(bridgeClient, fileIndex);
  const tensors = parsedTFLite.tensors.map((tensor) => ({
    ...tensor,
    sourcePath: tfliteRelativePath,
  }));
  const architecture = extractArchitecture(config, null);
  const architectureHint = config.architectures?.[0] ?? config.model_type ?? '';

  return {
    sourceKind: 'tflite',
    config,
    tensors,
    architectureHint,
    embeddingPostprocessor: null,
    architecture,
    sourceQuantization: parsedTFLite.sourceQuantization,
    tokenizerJson: tokenizerAssets.tokenizerJson,
    tokenizerConfig: tokenizerAssets.tokenizerConfig,
    tokenizerModelName: tokenizerAssets.tokenizerModelName,
    sourceFiles: [{ path: tfliteRelativePath, size: tfliteEntry.size }],
    auxiliaryFiles: [
      { path: 'config.json', size: Number(fileIndex.get('config.json')?.size || 0), kind: 'config' },
      ...buildBridgeTokenizerAuxiliaryFiles(fileIndex, tokenizerAssets),
    ],
    tokenizerJsonPath: tokenizerAssets.tokenizerJsonPath,
    tokenizerConfigPath: tokenizerAssets.tokenizerConfigPath,
    tokenizerModelPath: tokenizerAssets.tokenizerModelPath,
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
    sourceQuantization: parsedGGUF.quantization ?? inferSourceQuantizationForSourceRuntime(tensors, 'gguf', {
      logCategory: 'DopplerProvider',
    }),
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

async function resolveBridgeFileEntry(bridgeClient, absolutePath) {
  const { parentPath, basename } = splitBridgeFilePath(absolutePath);
  if (!basename) {
    throw new Error(`Bridge source runtime: invalid file path "${absolutePath}".`);
  }
  const entries = await bridgeClient.list(parentPath);
  const match = entries.find((entry) => entry?.isDir !== true && String(entry?.name || '') === basename) ?? null;
  if (!match) {
    throw new Error(`Bridge source runtime: file "${absolutePath}" does not exist.`);
  }
  return {
    relativePath: basename,
    absolutePath,
    size: Number(match?.size) || 0,
  };
}

function createBridgeVirtualFileReaders(bridgeClient, packageFileEntry, virtualFiles, rootPath) {
  const virtualFileMap = new Map(
    (Array.isArray(virtualFiles) ? virtualFiles : []).map((entry) => [entry.path, entry])
  );

  const resolveVirtualFile = (pathHint) => {
    const hint = normalizeRelativePath(pathHint);
    if (!hint) {
      return null;
    }
    return virtualFileMap.get(hint) ?? null;
  };

  const readRange = async (virtualPath, offset, length) => {
    const entry = resolveVirtualFile(virtualPath);
    if (!entry) {
      throw new Error(`Missing source shard file: ${virtualPath}`);
    }
    if (entry.externalPath) {
      return bridgeClient.read(entry.externalPath, offset, length);
    }
    const start = Math.max(0, Math.floor(Number(offset) || 0));
    const requested = Math.max(0, Math.floor(Number(length) || 0));
    const available = Math.max(0, entry.size - start);
    const readLength = Math.min(available, requested);
    return bridgeClient.read(packageFileEntry.absolutePath, entry.offset + start, readLength);
  };

  const readText = async (virtualPath) => {
    const entry = resolveVirtualFile(virtualPath);
    if (!entry) {
      return null;
    }
    const bytes = entry.externalPath
      ? await bridgeClient.read(entry.externalPath, 0, entry.size)
      : await bridgeClient.read(packageFileEntry.absolutePath, entry.offset, entry.size);
    return decodeText(bytes);
  };

  const readBinary = async (virtualPath) => {
    const entry = resolveVirtualFile(virtualPath);
    if (!entry) {
      throw new Error(`Missing source binary file: ${virtualPath}`);
    }
    if (entry.externalPath) {
      return bridgeClient.read(entry.externalPath, 0, entry.size);
    }
    return bridgeClient.read(packageFileEntry.absolutePath, entry.offset, entry.size);
  };

  return {
    rootPath,
    virtualFileMap,
    readRange,
    readText,
    readBinary,
  };
}

async function addHashesToBridgeVirtualFiles(bridgeClient, packageFileEntry, virtualFiles, entries, hashAlgorithm) {
  const readers = createBridgeVirtualFileReaders(bridgeClient, packageFileEntry, virtualFiles, packageFileEntry.absolutePath);
  const hashedEntries = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const virtualPath = normalizeRelativePath(entry?.path);
    if (!virtualPath) {
      continue;
    }
    const descriptor = readers.virtualFileMap.get(virtualPath);
    if (!descriptor) {
      throw new Error(`Missing bridge package file entry for "${virtualPath}"`);
    }
    const bytes = descriptor.externalPath
      ? await bridgeClient.read(descriptor.externalPath, 0, descriptor.size)
      : await bridgeClient.read(packageFileEntry.absolutePath, descriptor.offset, descriptor.size);
    hashedEntries.push({
      ...entry,
      path: virtualPath,
      size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : descriptor.size,
      hash: await computeHash(toUint8Array(bytes), hashAlgorithm),
      hashAlgorithm,
    });
  }
  return hashedEntries;
}

async function parseBridgeLiteRTPackageModel(bridgeClient, packageFileEntry, sourceKind) {
  const resolved = await resolveLiteRTPackageParsedArtifact({
    sourceKind,
    sourcePathForModelId: packageFileEntry.absolutePath,
    source: {
      name: packageFileEntry.relativePath,
      size: packageFileEntry.size,
      async readRange(offset, length) {
        return readBridgeRange(bridgeClient, packageFileEntry, offset, length);
      },
    },
  });
  const rootAbsolutePath = splitBridgeFilePath(packageFileEntry.absolutePath).parentPath;
  const siblingFileIndex = indexBridgeFiles(await listBridgeFilesRecursive(bridgeClient, rootAbsolutePath));
  const tokenizerAssets = await collectBridgeTokenizerAssets(bridgeClient, siblingFileIndex);
  const virtualFiles = appendLiteRTPackageVirtualFiles(
    resolved.virtualFiles,
    buildBridgePackageTokenizerVirtualFiles(siblingFileIndex, tokenizerAssets, rootAbsolutePath)
  );
  const parsedArtifact = applyBridgePackageTokenizerAssets(resolved.parsedArtifact, tokenizerAssets);
  return {
    ...parsedArtifact,
    storageReaders: createBridgeVirtualFileReaders(
      bridgeClient,
      packageFileEntry,
      virtualFiles,
      packageFileEntry.absolutePath
    ),
    async hashFileEntries(entries, hashAlgorithm) {
      return addHashesToBridgeVirtualFiles(
        bridgeClient,
        packageFileEntry,
        virtualFiles,
        entries,
        hashAlgorithm
      );
    },
  };
}

async function parseBridgeSourceModel(bridgeClient, localPath) {
  if (isLiteRTTaskPath(localPath)) {
    const packageFileEntry = await resolveBridgeFileEntry(bridgeClient, localPath);
    return parseBridgeLiteRTPackageModel(bridgeClient, packageFileEntry, LITERT_PACKAGE_SOURCE_KIND_TASK);
  }
  if (isLiteRTLMPath(localPath)) {
    const packageFileEntry = await resolveBridgeFileEntry(bridgeClient, localPath);
    return parseBridgeLiteRTPackageModel(bridgeClient, packageFileEntry, LITERT_PACKAGE_SOURCE_KIND_LITERTLM);
  }

  const files = await listBridgeFilesRecursive(bridgeClient, localPath);
  const fileIndex = indexBridgeFiles(files);
  if (hasModelManifest(fileIndex)) {
    return null;
  }

  const detected = detectBridgeSourceFormat(fileIndex);
  if (!detected) {
    return null;
  }

  if (detected.kind === 'tflite') {
    const tflitePath = Array.from(fileIndex.keys()).find((path) => path.toLowerCase().endsWith('.tflite')) || null;
    if (!tflitePath) {
      throw new Error('Bridge source runtime: failed to resolve the detected TFLite file.');
    }
    return parseBridgeTfliteModel(bridgeClient, fileIndex, tflitePath);
  }
  if (detected.kind === LITERT_PACKAGE_SOURCE_KIND_TASK) {
    const taskPath = Array.from(fileIndex.keys()).find((entryPath) => entryPath.toLowerCase().endsWith('.task')) || null;
    if (!taskPath) {
      throw new Error('Bridge source runtime: failed to resolve the detected LiteRT task file.');
    }
    const packageFileEntry = fileIndex.get(taskPath);
    if (!packageFileEntry) {
      throw new Error(`Bridge source runtime: missing bridge file entry for "${taskPath}".`);
    }
    return parseBridgeLiteRTPackageModel(bridgeClient, packageFileEntry, LITERT_PACKAGE_SOURCE_KIND_TASK);
  }
  if (detected.kind === LITERT_PACKAGE_SOURCE_KIND_LITERTLM) {
    const litertLmPath = Array.from(fileIndex.keys()).find((entryPath) => entryPath.toLowerCase().endsWith('.litertlm')) || null;
    if (!litertLmPath) {
      throw new Error('Bridge source runtime: failed to resolve the detected LiteRT-LM file.');
    }
    const packageFileEntry = fileIndex.get(litertLmPath);
    if (!packageFileEntry) {
      throw new Error(`Bridge source runtime: missing bridge file entry for "${litertLmPath}".`);
    }
    return parseBridgeLiteRTPackageModel(bridgeClient, packageFileEntry, LITERT_PACKAGE_SOURCE_KIND_LITERTLM);
  }
  if (detected.kind === 'gguf') {
    return parseBridgeGGUFModel(bridgeClient, fileIndex, detected.ggufPath);
  }

  return parseBridgeSafetensorsModel(bridgeClient, fileIndex);
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
  const model = options.model ?? options.manifest;
  const sourceRuntime = getSourceRuntimeMetadata(model);
  if (!sourceRuntime) {
    return null;
  }
  if (
    sourceRuntime.sourceKind === LITERT_PACKAGE_SOURCE_KIND_TASK
    || sourceRuntime.sourceKind === LITERT_PACKAGE_SOURCE_KIND_LITERTLM
  ) {
    const packageFileEntry = await resolveBridgeFileEntry(bridgeClient, localPath);
    const resolved = await resolveLiteRTPackageParsedArtifact({
      sourceKind: sourceRuntime.sourceKind,
      sourcePathForModelId: localPath,
      source: {
        name: packageFileEntry.relativePath,
        size: packageFileEntry.size,
        async readRange(offset, length) {
          return readBridgeRange(bridgeClient, packageFileEntry, offset, length);
        },
      },
    });
    const rootAbsolutePath = splitBridgeFilePath(packageFileEntry.absolutePath).parentPath;
    const siblingFileIndex = indexBridgeFiles(await listBridgeFilesRecursive(bridgeClient, rootAbsolutePath));
    const tokenizerAssets = await collectBridgeTokenizerAssets(bridgeClient, siblingFileIndex);
    const virtualFiles = appendLiteRTPackageVirtualFiles(
      resolved.virtualFiles,
      buildBridgePackageTokenizerVirtualFiles(siblingFileIndex, tokenizerAssets, rootAbsolutePath)
    );
    const readers = createBridgeVirtualFileReaders(
      bridgeClient,
      packageFileEntry,
      virtualFiles,
      localPath
    );
    return createSourceStorageContext({
      model,
      readRange: readers.readRange,
      readText: readers.readText,
      readBinary: readers.readBinary,
      verifyHashes: options.verifyHashes !== false,
    });
  }
  const files = await listBridgeFilesRecursive(bridgeClient, localPath);
  const fileMap = indexBridgeFiles(files);
  const readers = createBridgeFileReaders(bridgeClient, fileMap, localPath);
  return createSourceStorageContext({
    model,
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
  const existingModel = options.model ?? options.manifest ?? null;

  if (!bridgeClient || typeof bridgeClient.read !== 'function' || typeof bridgeClient.list !== 'function') {
    throw new Error('Bridge source runtime requires a connected bridge client with read/list support.');
  }
  if (!localPath || typeof localPath !== 'string') {
    throw new Error('Bridge source runtime requires localPath.');
  }

  if (existingModel && getSourceRuntimeMetadata(existingModel)) {
    const storageContext = await resolveBridgeStorageContext({
      bridgeClient,
      localPath,
      model: existingModel,
      verifyHashes,
    });
    return {
      model: existingModel,
      manifest: existingModel,
      storageContext,
      sourceKind: getSourceRuntimeMetadata(existingModel)?.sourceKind ?? 'safetensors',
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
  const {
    model,
    shardSources,
    sourceKind,
  } = await resolveSourceRuntimeBundleFromParsedArtifact({
    parsedArtifact: parsed,
    requestedModelId,
    runtimeLabel: 'bridge source runtime',
    logCategory: 'DopplerProvider',
    hashFileEntries: typeof parsed?.hashFileEntries === 'function'
      ? (entries, hashAlgorithm) => parsed.hashFileEntries(entries, hashAlgorithm)
      : async (entries, hashAlgorithm) => {
        const files = await listBridgeFilesRecursive(bridgeClient, localPath);
        const fileMap = indexBridgeFiles(files);
        return addHashesToBridgeFiles(bridgeClient, fileMap, entries, hashAlgorithm);
      },
  });

  const readers = parsed?.storageReaders ?? createBridgeFileReaders(
    bridgeClient,
    indexBridgeFiles(await listBridgeFilesRecursive(bridgeClient, localPath)),
    localPath
  );
  const storageContext = createSourceStorageContext({
    model,
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
    `Bridge source runtime ready: ${model.modelId} (${sourceKind}, ${parsed.tensors.length} tensors)`
  );

  return {
    model,
    manifest: model,
    storageContext,
    sourceKind,
    sourceRoot: localPath,
  };
}
