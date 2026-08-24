import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { NodeConvertWorkerPool } from './node-convert-worker-pool.js';
import { createConversionRunTiming } from './conversion-run-timing.js';
import { createRowChunks, mapOrderedChunkBatches } from './node-converter-chunk-batches.js';
import { bootstrapNodeWebGPU } from './node-webgpu.js';
import { buildManifestIntegrityFromModelDir } from './rdrr-integrity-refresh.js';
import { applySourceTensorRules } from '../converter/source-tensor-rules.js';
import {
  buildSourceTokenizerJson,
  validateSourceTokenizerPolicy,
} from '../converter/source-tokenizer.js';
import { isPlainObject } from '../formats/plain-object.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { log, trace } from '../debug/index.js';
import { saveReport } from '../storage/reports.js';
import {
  CONVERSION_REPORT_SCHEMA_VERSION,
  validateConversionReport,
} from '../config/schema/conversion-report.schema.js';
import { createNodeGpuTensorTransformer, createNodeLargeTensorTransformer, createNodeTensorTransformer } from './node-converter/input.js';
import { buildConvertReport, createFileRangeReader, createNodeConvertIO, normalizeExecutionConfig } from './node-converter/output.js';

function resolveHostParallelism() {
  if (typeof os.availableParallelism === 'function') {
    const value = os.availableParallelism();
    if (Number.isInteger(value) && value > 0) return value;
  }
  const cpus = typeof os.cpus === 'function' ? os.cpus() : null;
  return Array.isArray(cpus) && cpus.length > 0 ? cpus.length : 1;
}

function resolveExecutionPlan(executionConfig) {
  const requestedWorkers = executionConfig.workers;
  const availableWorkers = resolveHostParallelism();
  if (executionConfig.workerCountPolicy === 'error' && requestedWorkers > availableWorkers) {
    throw new Error(
      `node convert: requested workers (${requestedWorkers}) exceed available CPU parallelism (${availableWorkers}).`
    );
  }

  const effectiveWorkers = executionConfig.workerCountPolicy === 'cap'
    ? Math.min(requestedWorkers, availableWorkers)
    : requestedWorkers;

  return {
    ...executionConfig,
    requestedWorkers,
    availableWorkers,
    effectiveWorkers: Math.max(1, effectiveWorkers),
  };
}

function createStageTimer(label) {
  const start = performance.now();
  return {
    stop(extra = '', data = null) {
      const elapsed = performance.now() - start;
      const suffix = extra ? ` - ${extra}` : '';
      log.verbose('NodeConvert', `${label}: ${elapsed.toFixed(0)}ms${suffix}`);
      trace.perf(`NodeConvert ${label}`, {
        ms: elapsed,
        ...(data && typeof data === 'object' ? data : {}),
      });
      return elapsed;
    },
  };
}

function compareNullableStrings(a, b) {
  const left = typeof a === 'string' ? a : '';
  const right = typeof b === 'string' ? b : '';
  return left.localeCompare(right);
}

function sortTensorsByDeterministicLocality(tensors) {
  if (!Array.isArray(tensors) || tensors.length <= 1) {
    return tensors;
  }
  tensors.sort((left, right) => {
    const sourcePathCmp = compareNullableStrings(left?.sourcePath, right?.sourcePath);
    if (sourcePathCmp !== 0) return sourcePathCmp;
    const leftOffset = Number.isFinite(left?.offset) ? Number(left.offset) : 0;
    const rightOffset = Number.isFinite(right?.offset) ? Number(right.offset) : 0;
    if (leftOffset !== rightOffset) {
      return leftOffset - rightOffset;
    }
    return compareNullableStrings(left?.name, right?.name);
  });
  return tensors;
}

let gpuCastRuntimePromise = null;

async function loadNodeGpuCastRuntime() {
  if (!gpuCastRuntimePromise) {
    gpuCastRuntimePromise = (async () => {
      await bootstrapNodeWebGPU();
      const [
        { initDevice, getDevice, destroyDevice, resetDeviceState },
        { castF32ToF16, runBF16ToF16 },
        { createTensor },
        { acquireBuffer, releaseBuffer, getBufferPool, destroyBufferPool },
      ] = await Promise.all([
        import('../gpu/device.js'),
        import('../gpu/kernel-selector.js'),
        import('../gpu/tensor.js'),
        import('../memory/buffer-pool.js'),
      ]);
      const device = await initDevice();
      if (!device || !getDevice()) {
        throw new Error(
          'node convert: execution.useGpuCast requires a WebGPU-capable Node runtime.'
        );
      }
      return {
        getDevice,
        destroyDevice,
        resetDeviceState,
        castF32ToF16,
        runBF16ToF16,
        createTensor,
        acquireBuffer,
        releaseBuffer,
        getBufferPool,
        destroyBufferPool,
      };
    })();
  }
  try {
    return await gpuCastRuntimePromise;
  } catch (error) {
    gpuCastRuntimePromise = null;
    throw error;
  }
}

function assertPath(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`node convert: ${label} is required.`);
  }
  return path.resolve(value);
}

function readOptionalNonEmptyString(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  return trimmed || null;
}

function toRepoRelativePath(filePath) {
  const normalized = readOptionalNonEmptyString(filePath);
  if (!normalized) return null;
  const relative = path.relative(process.cwd(), path.resolve(normalized)).replace(/\\/g, '/');
  return relative && !relative.startsWith('..') ? relative : path.resolve(normalized);
}

function resolveConfiguredModelId(explicitModelId, converterConfig) {
  return (
    readOptionalNonEmptyString(explicitModelId)
    ?? readOptionalNonEmptyString(converterConfig?.output?.modelBaseId)
  );
}

function resolveOutputDir(outputDirOverride, converterConfig, modelId) {
  const override = readOptionalNonEmptyString(outputDirOverride);
  if (override) {
    return path.resolve(override);
  }

  const configuredDir = readOptionalNonEmptyString(converterConfig?.output?.dir);
  if (configuredDir) {
    return path.resolve(configuredDir);
  }

  const configuredBaseDir = readOptionalNonEmptyString(converterConfig?.output?.baseDir);
  if (configuredBaseDir) {
    if (!modelId) {
      throw new Error(
        'node convert: converterConfig.output.baseDir requires modelId. ' +
        'Set converterConfig.output.modelBaseId or pass modelId.'
      );
    }
    return path.resolve(configuredBaseDir, modelId);
  }

  throw new Error(
    'node convert: outputDir is required. ' +
    'Provide --output-dir, converterConfig.output.dir, or converterConfig.output.baseDir.'
  );
}

function normalizeConverterConfigOverride(value) {
  if (value == null) return null;
  if (!isPlainObject(value)) {
    throw new Error('node convert: converterConfig must be an object when provided.');
  }
  return value;
}

function isGgufPath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.gguf');
}

async function getPathStats(targetPath, label) {
  try {
    return await fs.stat(targetPath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`node convert: ${label} does not exist: ${targetPath}`);
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`node convert: failed to stat ${label} "${targetPath}": ${message}`);
  }
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

async function resolveGgufPathFromDirectory(inputDir) {
  const entries = await fs.readdir(inputDir, { withFileTypes: true });
  const ggufFiles = entries
    .filter((entry) => entry.isFile() && isGgufPath(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  if (ggufFiles.length === 0) {
    return null;
  }
  if (ggufFiles.length > 1) {
    throw new Error(
      `node convert: multiple GGUF files found in "${inputDir}": ${ggufFiles.join(', ')}. ` +
      'Pass a .gguf file path directly.'
    );
  }
  return path.join(inputDir, ggufFiles[0]);
}

async function readSafetensorsHeader(filePath, parseSafetensorsHeader, readRange) {
  const headerPrefixBuffer = await readRange(filePath, 0, 8);
  const headerPrefixBytes = new Uint8Array(headerPrefixBuffer);
  if (headerPrefixBytes.byteLength < 8) {
    throw new Error(`Invalid safetensors header prefix for "${filePath}"`);
  }
  const headerSize = Number(new DataView(headerPrefixBuffer).getBigUint64(0, true));
  const headerBuffer = await readRange(filePath, 8, headerSize);
  const fullHeader = new Uint8Array(8 + headerSize);
  fullHeader.set(headerPrefixBytes, 0);
  fullHeader.set(new Uint8Array(headerBuffer), 8);
  return parseSafetensorsHeader(
    fullHeader.buffer.slice(fullHeader.byteOffset, fullHeader.byteOffset + fullHeader.byteLength)
  );
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

async function clearExistingConversionOutputs(outputDir) {
  let entries;
  try {
    entries = await fs.readdir(outputDir, { withFileTypes: true });
  } catch {
    return;
  }
  const artifactFiles = entries
    .filter((entry) => (
      entry.isFile()
      && (
        /^shard_\d{5}\.bin$/i.test(entry.name)
        || entry.name === 'manifest.json'
      )
    ))
    .map((entry) => path.join(outputDir, entry.name));
  if (artifactFiles.length === 0) return;
  await Promise.all(artifactFiles.map((filePath) => fs.unlink(filePath)));
}

function toNodeProgress(update) {
  if (!update) return null;
  return {
    stage: update.stage ?? null,
    current: Number.isFinite(update.current) ? update.current : null,
    total: Number.isFinite(update.total) ? update.total : null,
    message: typeof update.message === 'string' ? update.message : null,
    tensorName: typeof update.tensorName === 'string' ? update.tensorName : null,
    tensorBytesCurrent: Number.isFinite(update.tensorBytesCurrent)
      ? update.tensorBytesCurrent
      : null,
    tensorBytesTotal: Number.isFinite(update.tensorBytesTotal)
      ? update.tensorBytesTotal
      : null,
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
  const conversionRunTiming = createConversionRunTiming();
  const inputDir = assertPath(options?.inputDir, 'inputDir');
  const outputDirOverride = readOptionalNonEmptyString(options?.outputDir);
  const converterConfigOverride = normalizeConverterConfigOverride(options?.converterConfig);
  const onProgress = typeof options?.onProgress === 'function' ? options.onProgress : null;
  const inputStats = await getPathStats(inputDir, 'inputDir');
  const isInputDirectory = inputStats.isDirectory();
  const inputGgufPath = (
    inputStats.isFile() && isGgufPath(inputDir)
      ? inputDir
      : (isInputDirectory ? await resolveGgufPathFromDirectory(inputDir) : null)
  );
  const isInputGgufFile = Boolean(inputGgufPath);

  installNodeFileFetchShim();
  const fileRangeReader = createFileRangeReader();
  let gpuRuntimeForCleanup = null;
  try {

  const [
    { parseSafetensorsHeader },
    { parseGGUFHeader },
    {
      convertModel,
      extractArchitecture,
      transformTensorBytes,
      resolveTensorTargetQuant,
      normalizeStorageQuant,
      shouldQuantize,
    },
    { parseGGUFModel },
    { resolveConversionPlan, inferSourceWeightQuantization, resolveConvertedModelId },
    { parseDiffusionModel },
    { parseTransformerModel },
    { createConverterConfig, HEADER_READ_SIZE, DEFAULT_CONVERTER_EXECUTION_CONFIG },
    { computeHash },
  ] = await Promise.all([
    import('../formats/safetensors/types.js'),
    import('../formats/gguf/types.js'),
    import('../converter/core.js'),
    import('../converter/parsers/gguf.js'),
    import('../converter/conversion-plan.js'),
    import('../converter/parsers/diffusion.js'),
    import('../converter/parsers/transformer.js'),
    import('../config/schema/index.js'),
    import('../storage/shard-manager.js'),
  ]);

  const hashStringSha256 = async (value) => (
    computeHash(new TextEncoder().encode(String(value)), 'sha256')
  );
  const converterConfig = createConverterConfig(converterConfigOverride ?? undefined);
  const executionConfig = normalizeExecutionConfig(
    options?.execution,
    DEFAULT_CONVERTER_EXECUTION_CONFIG
  );
  const executionPlan = resolveExecutionPlan(executionConfig);
  const diffusionIndexPath = isInputDirectory ? path.join(inputDir, 'model_index.json') : null;
  const isDiffusionInput = isInputDirectory && diffusionIndexPath ? await fileExists(diffusionIndexPath) : false;

  let config = null;
  let tensors = [];
  let architectureHint = '';
  let architecture = null;
  let embeddingPostprocessor = null;
  let modelKind = 'transformer';
  let sourceQuantization = null;
  let tokenizerJson = null;
  let tokenizerConfig = null;
  let generationConfig = null;
  let rerankScoring = null;
  let hasTokenizerModel = false;
  let tokenizerModelPath = null;
  let diffusionAuxFiles = [];
  const parseTimer = createStageTimer('Parse input');

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
        const text = await fs.readFile(path.join(inputDir, suffix), 'utf8');
        try {
          return JSON.parse(text);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Invalid JSON in ${label} (${suffix}): ${message}`);
        }
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
        const parsed = await readSafetensorsHeader(
          fullPath,
          parseSafetensorsHeader,
          fileRangeReader.readRange
        );
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
        const parsedShards = await Promise.all(
          shardSuffixes.map(async (shardSuffix) => {
            const fullPath = path.join(inputDir, shardSuffix);
            const parsed = await readSafetensorsHeader(
              fullPath,
              parseSafetensorsHeader,
              fileRangeReader.readRange
            );
            return {
              fullPath,
              tensors: parsed.tensors,
            };
          })
        );
        const tensorsOut = [];
        for (const parsedShard of parsedShards) {
          for (const tensor of parsedShard.tensors) {
            tensorsOut.push({
              ...tensor,
              sourcePath: parsedShard.fullPath,
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
  } else if (isInputGgufFile) {
    const ggufPath = inputGgufPath;
    const ggufStats = await getPathStats(ggufPath, 'GGUF file');
    const ggufSource = {
      sourceType: 'node-file',
      name: path.basename(ggufPath),
      size: ggufStats.size,
      file: {
        name: path.basename(ggufPath),
        size: ggufStats.size,
      },
      async readRange(offset, length) {
        return fileRangeReader.readRange(ggufPath, offset, length);
      },
    };
    const normalizeTensorSource = (input) => {
      if (input && typeof input.readRange === 'function' && Number.isFinite(input.size)) {
        return input;
      }
      return ggufSource;
    };
    const parseGGUFHeaderFromSource = async (source) => {
      const resolved = normalizeTensorSource(source);
      const readSize = Math.min(resolved.size, HEADER_READ_SIZE);
      const buffer = await resolved.readRange(0, readSize);
      const info = parseGGUFHeader(buffer);
      return {
        ...info,
        fileSize: resolved.size,
      };
    };
    const parsedGGUF = await parseGGUFModel({
      file: ggufSource,
      parseGGUFHeaderFromSource,
      normalizeTensorSource,
      onProgress(update) {
        onProgress?.(toNodeProgress({
          stage: update?.stage ?? 'parsing',
          message: update?.message ?? null,
        }));
      },
      signal: null,
    });
    config = parsedGGUF.config;
    tensors = parsedGGUF.tensors.map((tensor) => ({
      ...tensor,
      sourcePath: ggufPath,
    }));
    architectureHint = parsedGGUF.architecture;
    sourceQuantization = parsedGGUF.quantization ?? null;
    architecture = extractArchitecture({}, parsedGGUF.config || {});
  } else {
    if (!isInputDirectory) {
      throw new Error(
        'node convert: inputDir must be a directory containing safetensors files or a .gguf file path.'
      );
    }
    const parsedTransformer = await parseTransformerModel({
      async readJson(suffix, label = 'json') {
        const filePath = path.join(inputDir, suffix);
        let text;
        try {
          text = await fs.readFile(filePath, 'utf8');
        } catch (error) {
          if (error?.code === 'ENOENT') {
            throw new Error(`Missing ${label} (${suffix})`);
          }
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Failed to read ${label} (${suffix}): ${message}`);
        }
        try {
          return JSON.parse(text);
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          throw new Error(`Invalid JSON in ${label} (${suffix}): ${message}`);
        }
      },
      async fileExists(suffix) {
        return fileExists(path.join(inputDir, suffix));
      },
      async loadSingleSafetensors(suffix) {
        const filePath = path.join(inputDir, suffix);
        const parsed = await readSafetensorsHeader(
          filePath,
          parseSafetensorsHeader,
          fileRangeReader.readRange
        );
        return parsed.tensors.map((tensor) => ({
          ...tensor,
          sourcePath: filePath,
        }));
      },
      async loadShardedSafetensors(indexJson) {
        const shardFiles = [...new Set(Object.values(indexJson.weight_map || {}))];
        const parsedShards = await Promise.all(
          shardFiles.map(async (shardFile) => {
            const shardPath = path.join(inputDir, shardFile);
            const parsed = await readSafetensorsHeader(
              shardPath,
              parseSafetensorsHeader,
              fileRangeReader.readRange
            );
            return {
              shardPath,
              tensors: parsed.tensors,
            };
          })
        );
        const tensorsOut = [];
        for (const parsedShard of parsedShards) {
          for (const tensor of parsedShard.tensors) {
            tensorsOut.push({ ...tensor, sourcePath: parsedShard.shardPath });
          }
        }
        return tensorsOut;
      },
    });
    config = parsedTransformer.config;
    generationConfig = parsedTransformer.generationConfig ?? null;
    tensors = parsedTransformer.tensors;
    architectureHint = parsedTransformer.architectureHint;
    embeddingPostprocessor = parsedTransformer.embeddingPostprocessor ?? null;
    rerankScoring = parsedTransformer.rerankScoring ?? null;
    tensors = applySourceTensorRules(tensors, converterConfig.sourceTensors);
    architecture = converterConfig.architecture ?? extractArchitecture(config, null);
    const tokenizerJsonPath = path.join(inputDir, 'tokenizer.json');
    tokenizerModelPath = path.join(inputDir, 'tokenizer.model');
    const tokenizerConfigPath = path.join(inputDir, 'tokenizer_config.json');
    tokenizerJson = await readOptionalJson(tokenizerJsonPath);
    tokenizerConfig = await readOptionalJson(tokenizerConfigPath);
    hasTokenizerModel = await fileExists(tokenizerModelPath);
    const sourceTokenizerPolicy = validateSourceTokenizerPolicy(converterConfig.sourceTokenizer);
    if (sourceTokenizerPolicy) {
      if (tokenizerJson) {
        throw new Error(
          'node convert: sourceTokenizer cannot be combined with a source tokenizer.json. ' +
          'Remove sourceTokenizer or make the source tokenizer identity explicit in one place.'
        );
      }
      if (hasTokenizerModel) {
        throw new Error(
          'node convert: sourceTokenizer cannot be combined with a source tokenizer.model.'
        );
      }
      const vocabPath = path.join(inputDir, sourceTokenizerPolicy.vocabFile);
      let vocabText;
      try {
        vocabText = await fs.readFile(vocabPath, 'utf8');
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `node convert: unable to read sourceTokenizer.vocabFile ` +
          `"${sourceTokenizerPolicy.vocabFile}": ${message}`
        );
      }
      tokenizerJson = buildSourceTokenizerJson(sourceTokenizerPolicy, vocabText);
    }
  }
  parseTimer.stop(`${modelKind} tensors=${tensors.length}`);

  sortTensorsByDeterministicLocality(tensors);

  const weightOverride = converterConfig.quantization?.weights ?? null;
  sourceQuantization = sourceQuantization || weightOverride || inferSourceWeightQuantization(tensors);
  const plan = resolveConversionPlan({
    rawConfig: config,
    tensors,
    converterConfig,
    sourceQuantization,
    modelKind,
    architectureHint,
    architectureConfig: architecture,
  });
  const resolvedModelType = plan.modelType;
  const targetQuantization = plan.manifestQuantization;
  const quantizationInfo = plan.quantizationInfo;
  const inference = plan.manifestInference;
  if (rerankScoring) {
    const rerankConfig = inference?.rerank;
    if (!rerankConfig || typeof rerankConfig !== 'object' || Array.isArray(rerankConfig)) {
      throw new Error(
        'node convert: sentence-transformers LogitScore module requires explicit inference.rerank config.'
      );
    }
    if (Number(rerankConfig.trueTokenId) !== rerankScoring.trueTokenId) {
      throw new Error(
        `node convert: inference.rerank.trueTokenId=${rerankConfig.trueTokenId} does not match LogitScore true_token_id=${rerankScoring.trueTokenId}.`
      );
    }
    if (Number(rerankConfig.falseTokenId) !== rerankScoring.falseTokenId) {
      throw new Error(
        `node convert: inference.rerank.falseTokenId=${rerankConfig.falseTokenId} does not match LogitScore false_token_id=${rerankScoring.falseTokenId}.`
      );
    }
  }
  const explicitModelId = resolveConfiguredModelId(options?.modelId, converterConfig);
  if (!explicitModelId) {
    throw new Error(
      'node convert: modelId is required. ' +
      'Set converterConfig.output.modelBaseId.'
    );
  }
  const modelId = resolveConvertedModelId({
    explicitModelId,
    converterConfig,
    detectedModelId: explicitModelId,
    quantizationInfo,
  });
  if (!modelId) {
    throw new Error('node convert: failed to resolve modelId from converterConfig.output.modelBaseId.');
  }
  const outputDir = resolveOutputDir(outputDirOverride, converterConfig, modelId);

  await fs.mkdir(outputDir, { recursive: true });
  await clearExistingConversionOutputs(outputDir);

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
      role: tensor.role,
      group: tensor.group ?? null,
    })),
    config,
    architecture: architectureHint || 'unknown',
    quantization: targetQuantization,
    tokenizerJson,
    tokenizerConfig,
    // GGUF inputs carry the tokenizer (eos/bos/pad IDs, chat template, etc.)
    // inside config.tokenizer; lift it to model.tokenizer so resolveEosTokenId
    // and other manifest helpers find it without a GGUF-specific branch.
    tokenizer: tokenizerConfig ?? config?.tokenizer ?? null,
    generationConfig,
    tokenizerModel: hasTokenizerModel ? 'tokenizer.model' : null,
    embeddingPostprocessor,
  };

  const io = createNodeConvertIO(outputDir, {
    hashAlgorithm: converterConfig.manifest.hashAlgorithm,
    computeHash,
    readRange: fileRangeReader.readRange,
  });
  const deferredManifestState = {
    manifest: null,
  };
  const convertIo = {
    ...io,
    async writeManifest(manifest) {
      deferredManifestState.manifest = manifest;
    },
  };
  const manifestArchitecture = modelKind === 'diffusion' ? 'diffusion' : architecture;
  let workerPool = null;
  let workerTensorTransformer = null;
  let gpuTensorTransformer = null;
  let tensorTransformer = null;
  let largeTensorTransformer = null;
  let result = null;
  try {
    if (executionPlan.useGpuCast) {
      try {
        gpuRuntimeForCleanup = await loadNodeGpuCastRuntime();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `node convert: execution.useGpuCast requires a WebGPU-capable Node runtime. ${message}`
        );
      }
      gpuTensorTransformer = createNodeGpuTensorTransformer({
        runtime: gpuRuntimeForCleanup,
        gpuCastMinTensorBytes: executionPlan.gpuCastMinTensorBytes,
        requireGpuCast: executionPlan.gpuCastRequestedExplicitly === true,
        resolveTensorTargetQuant,
      });
    }
    if (executionPlan.effectiveWorkers > 1) {
      workerPool = new NodeConvertWorkerPool({ size: executionPlan.effectiveWorkers });
      workerTensorTransformer = createNodeTensorTransformer({
        pool: workerPool,
        execution: executionPlan,
        transformTensorBytes,
        resolveTensorTargetQuant,
        normalizeStorageQuant,
        shouldQuantize,
      });
    }
    const chunkTransformPool = workerPool ?? {
      async transformTensor(tensor, tensorData, transformContext) {
        return transformTensorBytes(tensor, tensorData, transformContext);
      },
    };
    largeTensorTransformer = createNodeLargeTensorTransformer({
      pool: chunkTransformPool,
      execution: executionPlan,
      readRange: fileRangeReader.readRange,
      resolveTensorTargetQuant,
      normalizeStorageQuant,
      shouldQuantize,
    });
    if (gpuTensorTransformer || workerTensorTransformer) {
      tensorTransformer = async (input) => {
        if (gpuTensorTransformer) {
          const gpuResult = await gpuTensorTransformer(input);
          if (gpuResult) {
            return gpuResult;
          }
        }
        if (workerTensorTransformer) {
          return workerTensorTransformer(input);
        }
        const tensor = input?.tensor;
        const tensorData = input?.tensorData;
        if (!tensor || !(tensorData instanceof Uint8Array)) {
          throw new Error('node convert: invalid tensor transform input.');
        }
        return transformTensorBytes(tensor, tensorData, input?.transformContext ?? {});
      };
    }
    onProgress?.(toNodeProgress({
      stage: 'writing',
      message: (
        `Convert execution workers: requested=${executionPlan.requestedWorkers}, ` +
        `effective=${executionPlan.effectiveWorkers}, available=${executionPlan.availableWorkers}, ` +
        `gpuCast=${executionPlan.useGpuCast ? 'on' : 'off'}`
      ),
    }));

    const convertTimer = createStageTimer('Convert tensors');
    result = await convertModel(model, convertIo, {
      modelId,
      modelType: resolvedModelType,
      quantization: targetQuantization,
      quantizationInfo,
      architecture: manifestArchitecture,
      inference,
      converterConfig,
      source: pathToFileURL(inputDir).href,
      sourcePath: inputDir,
      sourceFormat: isInputGgufFile ? 'gguf' : 'safetensors',
      conversionConfigPath: toRepoRelativePath(options?.configPath),
      conversionConfig: converterConfigOverride ?? null,
      hashString: hashStringSha256,
      tensorTransformer,
      largeTensorTransformer,
      onProgress(update) {
        onProgress?.(toNodeProgress(update));
      },
    });
    convertTimer.stop(`tensors=${result.tensorCount}, shards=${result.shardCount}`);
  } finally {
    if (workerPool) {
      await workerPool.close();
    }
  }

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
  if (!deferredManifestState.manifest) {
    throw new Error('node convert: convert core did not produce a manifest.');
  }
  const builtIntegrity = await buildManifestIntegrityFromModelDir(result.manifest, {
    modelDir: outputDir,
    tensorMap: result.manifest.tensors ?? undefined,
    readRange: fileRangeReader.readRange,
  });
  result.manifest = {
    ...result.manifest,
    integrityExtensions: builtIntegrity.integrityExtensions,
  };
  deferredManifestState.manifest = result.manifest;
  await io.writeManifest(result.manifest);

  const report = buildConvertReport(result, {
    physicalTiming: conversionRunTiming.complete(),
    modelType: resolvedModelType,
    outputDir,
    modelId: result.manifest?.modelId ?? modelId,
  });
  const reportInfo = await saveReport(report.modelId, report, {
    timestamp: report.timestamp,
  });

  return {
    manifest: result.manifest,
    shardCount: result.shardCount,
    tensorCount: result.tensorCount,
    executionContractArtifact: result.executionContractArtifact ?? null,
    layerPatternContractArtifact: result.layerPatternContractArtifact ?? null,
    requiredInferenceFieldsArtifact: result.requiredInferenceFieldsArtifact ?? null,
    report,
    reportInfo,
    modelType: resolvedModelType,
    outputDir,
  };
  } finally {
    await fileRangeReader.closeAll();
    if (gpuRuntimeForCleanup) {
      try {
        gpuRuntimeForCleanup.destroyBufferPool();
      } finally {
        try {
          gpuRuntimeForCleanup.destroyDevice();
        } finally {
          gpuRuntimeForCleanup.resetDeviceState();
          gpuCastRuntimePromise = null;
        }
      }
    }
  }
}
