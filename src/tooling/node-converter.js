import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { installNodeFileFetchShim } from './node-file-fetch.js';
import { NodeConvertWorkerPool } from './node-convert-worker-pool.js';
import { bootstrapNodeWebGPU } from './node-webgpu.js';
import { buildManifestIntegrityFromModelDir } from './rdrr-integrity-refresh.js';
import { isPlainObject } from '../utils/plain-object.js';
import { selectRuleValue } from '../rules/rule-registry.js';
import { log, trace } from '../debug/index.js';
import { saveReport } from '../storage/reports.js';
import {
  CONVERSION_REPORT_SCHEMA_VERSION,
  validateConversionReport,
} from '../config/schema/conversion-report.schema.js';

function asPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`node convert: ${label} must be a positive integer.`);
  }
  return value;
}

function normalizeExecutionConfig(value, defaults) {
  if (!isPlainObject(defaults)) {
    throw new Error('node convert: execution defaults must be an object.');
  }

  if (value == null) {
    return { ...defaults };
  }
  if (!isPlainObject(value)) {
    throw new Error('node convert: execution must be an object when provided.');
  }
  const workers = value.workers == null
    ? defaults.workers
    : asPositiveInteger(Number(value.workers), 'execution.workers');
  const workerCountPolicyRaw = value.workerCountPolicy == null
    ? defaults.workerCountPolicy
    : String(value.workerCountPolicy).trim().toLowerCase();
  if (workerCountPolicyRaw !== 'cap' && workerCountPolicyRaw !== 'error') {
    throw new Error('node convert: execution.workerCountPolicy must be "cap" or "error".');
  }
  const rowChunkRows = value.rowChunkRows == null
    ? defaults.rowChunkRows
    : asPositiveInteger(Number(value.rowChunkRows), 'execution.rowChunkRows');
  const rowChunkMinTensorBytes = value.rowChunkMinTensorBytes == null
    ? defaults.rowChunkMinTensorBytes
    : asPositiveInteger(Number(value.rowChunkMinTensorBytes), 'execution.rowChunkMinTensorBytes');
  const maxInFlightJobs = value.maxInFlightJobs == null
    ? defaults.maxInFlightJobs
    : asPositiveInteger(Number(value.maxInFlightJobs), 'execution.maxInFlightJobs');
  const useGpuCast = value.useGpuCast == null
    ? defaults.useGpuCast === true
    : value.useGpuCast === true;
  const gpuCastRequestedExplicitly = value.useGpuCast === true;
  if (value.useGpuCast != null && typeof value.useGpuCast !== 'boolean') {
    throw new Error('node convert: execution.useGpuCast must be a boolean when provided.');
  }
  const gpuCastMinTensorBytes = value.gpuCastMinTensorBytes == null
    ? asPositiveInteger(
      Number(defaults.gpuCastMinTensorBytes ?? defaults.rowChunkMinTensorBytes ?? (32 * 1024 * 1024)),
      'execution.gpuCastMinTensorBytes'
    )
    : asPositiveInteger(Number(value.gpuCastMinTensorBytes), 'execution.gpuCastMinTensorBytes');

  return {
    workers,
    workerCountPolicy: workerCountPolicyRaw,
    rowChunkRows,
    rowChunkMinTensorBytes,
    maxInFlightJobs,
    useGpuCast,
    gpuCastRequestedExplicitly,
    gpuCastMinTensorBytes,
  };
}

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

function getDtypeBytes(dtype) {
  const upper = String(dtype || '').toUpperCase();
  if (upper === 'F32') return 4;
  if (upper === 'F16' || upper === 'BF16') return 2;
  return null;
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

function normalizeWorkerTransformResult(result, tensor) {
  if (!result || !(result.tensorData instanceof Uint8Array)) {
    throw new Error(`node convert: worker transform returned invalid bytes for ${tensor.name}.`);
  }
  return {
    tensorData: result.tensorData,
    outDtype: result.outDtype ?? tensor.dtype,
    outLayout: result.outLayout ?? null,
    ...(result.companionData instanceof Uint8Array
      ? { companionData: result.companionData }
      : {}),
    ...(result.sourceTransform ? { sourceTransform: result.sourceTransform } : {}),
  };
}

const MAX_NODE_CONVERT_BUFFER_BYTES = Math.min(Buffer.kMaxLength, 0x7fff_ffff);

function resolveRowChunkTransformPlan(input) {
  const tensor = input?.tensor;
  const execution = input?.execution;
  const transformContext = input?.transformContext ?? {};
  const resolveTensorTargetQuant = input?.resolveTensorTargetQuant;
  const normalizeStorageQuant = input?.normalizeStorageQuant;
  const shouldQuantize = input?.shouldQuantize;
  const tensorByteLength = Number(input?.tensorByteLength ?? 0);

  if (!tensor || !execution) {
    throw new Error('node convert: row chunk transform plan requires tensor and execution.');
  }
  if (typeof resolveTensorTargetQuant !== 'function' || typeof normalizeStorageQuant !== 'function') {
    throw new Error('node convert: row chunk transform plan requires quantization helpers.');
  }
  if (typeof shouldQuantize !== 'function') {
    throw new Error('node convert: row chunk transform plan requires shouldQuantize().');
  }

  const sourceDtype = String(tensor.dtype || '').toUpperCase();
  const sourceQuant = normalizeStorageQuant(sourceDtype);
  const tensorTargetQuant = resolveTensorTargetQuant(
    tensor.name,
    transformContext.targetQuant,
    transformContext.quantizationInfo ?? null
  );
  const is2D = Array.isArray(tensor.shape) && tensor.shape.length === 2;
  const rows = is2D ? tensor.shape[0] : 0;
  const cols = is2D ? tensor.shape[1] : 0;
  const sourceBytesPerElement = getDtypeBytes(sourceDtype);
  const q4kLayout = String(transformContext.q4kLayout || 'row').trim().toLowerCase() === 'col'
    ? 'col'
    : 'row';
  const canChunkRows = (
    is2D
    && rows > 0
    && cols > 0
    && sourceBytesPerElement != null
    && sourceQuant !== 'q4k'
    && tensorByteLength >= execution.rowChunkMinTensorBytes
    && !(tensorTargetQuant === 'q4k' && q4kLayout === 'col')
  );
  const jobMode = selectRuleValue('converter', 'execution', 'jobMode', {
    workers: execution.effectiveWorkers,
    canChunkRows,
  });
  if (jobMode !== 'row_chunks' || !canChunkRows) {
    return null;
  }

  const rowChunkRows = execution.rowChunkRows
    ?? selectRuleValue('converter', 'execution', 'rowChunkRows', {
      workers: execution.effectiveWorkers,
      canChunkRows,
    });
  if (!Number.isInteger(rowChunkRows) || rowChunkRows < 1) {
    return null;
  }

  const rowSourceBytes = cols * sourceBytesPerElement;
  if (!Number.isInteger(rowSourceBytes) || rowSourceBytes < 1) {
    return null;
  }

  const forceQuantizeDecision = tensorTargetQuant === 'q4k'
    ? shouldQuantize(tensor.name, tensor.shape, {
      quantizeEmbeddings: Boolean(transformContext.quantizeEmbeddings),
      modulesToNotConvert: transformContext.modulesToNotConvert ?? null,
    })
    : null;

  return {
    rows,
    cols,
    rowChunkRows,
    rowSourceBytes,
    forceQuantizeDecision,
  };
}

async function mapWithConcurrency(items, concurrency, mapper) {
  if (!Array.isArray(items) || items.length === 0) return [];
  const workerCount = Math.max(1, Math.min(concurrency, items.length));
  const results = new Array(items.length);
  let nextIndex = 0;

  const runners = Array.from({ length: workerCount }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await mapper(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

let gpuCastRuntimePromise = null;

async function loadNodeGpuCastRuntime() {
  if (!gpuCastRuntimePromise) {
    gpuCastRuntimePromise = (async () => {
      await bootstrapNodeWebGPU();
      const [
        { initDevice, getDevice },
        { castF32ToF16, runBF16ToF16 },
        { createTensor },
        { acquireBuffer, releaseBuffer, getBufferPool },
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
        castF32ToF16,
        runBF16ToF16,
        createTensor,
        acquireBuffer,
        releaseBuffer,
        getBufferPool,
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

function createNodeGpuTensorTransformer(options) {
  const {
    runtime,
    gpuCastMinTensorBytes,
    requireGpuCast,
    resolveTensorTargetQuant,
  } = options;
  const {
    getDevice,
    castF32ToF16,
    runBF16ToF16,
    createTensor,
    acquireBuffer,
    releaseBuffer,
    getBufferPool,
  } = runtime;
  const minTensorBytes = Math.max(1, Number(gpuCastMinTensorBytes) || 1);
  let warnedFallback = false;

  return async function maybeTransformWithGPU(input) {
    const tensor = input?.tensor;
    const tensorData = input?.tensorData;
    const transformContext = input?.transformContext ?? {};
    const reportProgress = typeof input?.reportProgress === 'function'
      ? input.reportProgress
      : null;
    if (!tensor || !(tensorData instanceof Uint8Array)) {
      return null;
    }

    const sourceDtype = String(tensor.dtype || '').toUpperCase();
    if (sourceDtype !== 'F32' && sourceDtype !== 'BF16') {
      return null;
    }

    const targetQuant = resolveTensorTargetQuant(
      tensor.name,
      transformContext.targetQuant,
      transformContext.quantizationInfo ?? null
    );
    if (targetQuant !== 'f16') {
      return null;
    }
    if (tensorData.byteLength < minTensorBytes) {
      return null;
    }

    const elementBytes = sourceDtype === 'F32' ? 4 : 2;
    if (tensorData.byteLength % elementBytes !== 0) {
      return null;
    }
    const numElements = tensorData.byteLength / elementBytes;
    const outputBytes = numElements * 2;

    let inputBuffer = null;
    let outputBuffer = null;
    try {
      const device = getDevice();
      if (!device) {
        if (requireGpuCast) {
          throw new Error(
            `node convert: execution.useGpuCast failed for tensor "${tensor.name}": GPU device is unavailable.`
          );
        }
        return null;
      }
      inputBuffer = acquireBuffer(tensorData.byteLength, undefined, `convert_gpu_cast_in_${tensor.name}`);
      device.queue.writeBuffer(inputBuffer, 0, tensorData, tensorData.byteOffset, tensorData.byteLength);

      if (sourceDtype === 'F32') {
        const inputTensor = createTensor(inputBuffer, 'f32', [numElements], `${tensor.name}_f32`);
        const converted = await castF32ToF16(inputTensor);
        outputBuffer = converted.buffer;
      } else {
        const converted = await runBF16ToF16(inputBuffer, [numElements], `${tensor.name}_f16`);
        outputBuffer = converted.buffer;
      }

      const readback = await getBufferPool().readBuffer(outputBuffer, outputBytes);
      if (!(readback instanceof ArrayBuffer) || readback.byteLength !== outputBytes) {
        if (requireGpuCast) {
          throw new Error(
            `node convert: execution.useGpuCast failed for tensor "${tensor.name}": invalid GPU readback.`
          );
        }
        return null;
      }
      reportProgress?.(tensorData.byteLength, tensorData.byteLength);
      return {
        tensorData: new Uint8Array(readback),
        outDtype: 'F16',
        outLayout: null,
      };
    } catch (error) {
      if (requireGpuCast) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`node convert: execution.useGpuCast failed for tensor "${tensor.name}": ${message}`);
      }
      if (!warnedFallback) {
        warnedFallback = true;
        const message = error instanceof Error ? error.message : String(error);
        log.warn('NodeConvert', `GPU cast fallback to CPU: ${message}`);
      }
      return null;
    } finally {
      if (outputBuffer && outputBuffer !== inputBuffer) {
        releaseBuffer(outputBuffer);
      }
      if (inputBuffer) {
        releaseBuffer(inputBuffer);
      }
    }
  };
}


function generateShardFilename(index) {
  return `shard_${String(index).padStart(5, '0')}.bin`;
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

function createFileRangeReader() {
  const handleMap = new Map();
  const maxNodeFdReadInt32 = 0x7fff_ffff;

  async function getHandleEntry(filePath) {
    const existingPromise = handleMap.get(filePath);
    if (existingPromise) {
      return existingPromise;
    }
    const openPromise = (async () => {
      const fd = await fs.open(filePath, 'r');
      try {
        const stats = await fd.stat();
        return {
          fd,
          size: Number(stats.size),
        };
      } catch (error) {
        await fd.close().catch(() => {});
        throw error;
      }
    })();
    handleMap.set(filePath, openPromise);
    try {
      return await openPromise;
    } catch (error) {
      if (handleMap.get(filePath) === openPromise) {
        handleMap.delete(filePath);
      }
      throw error;
    }
  }

  return {
    async readRange(filePath, offset, length) {
      if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
        return new ArrayBuffer(0);
      }

      const entry = await getHandleEntry(filePath);
      const start = Math.max(0, Math.floor(offset));
      const end = Math.min(entry.size, start + Math.floor(length));
      if (end <= start) {
        return new ArrayBuffer(0);
      }

      if (start > maxNodeFdReadInt32 || (end - start) > maxNodeFdReadInt32) {
        const chunks = [];
        let totalBytes = 0;
        await new Promise((resolve, reject) => {
          const stream = createReadStream(filePath, {
            start,
            end: end - 1,
          });
          stream.on('data', (chunk) => {
            chunks.push(chunk);
            totalBytes += chunk.byteLength;
          });
          stream.on('end', resolve);
          stream.on('error', reject);
        });
        const out = Buffer.concat(chunks, totalBytes);
        return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
      }

      const out = Buffer.allocUnsafe(end - start);
      await entry.fd.read(out, 0, out.length, start);
      return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
    },
    async closeAll() {
      const closes = [];
      for (const entryPromise of handleMap.values()) {
        closes.push(
          Promise.resolve(entryPromise).then((entry) => entry.fd.close())
        );
      }
      handleMap.clear();
      await Promise.allSettled(closes);
    },
  };
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

function createNodeConvertIO(outputDir, options) {
  const hashAlgorithm = options?.hashAlgorithm;
  const computeHash = options?.computeHash;
  const readRange = options?.readRange;
  if (!hashAlgorithm || typeof hashAlgorithm !== 'string') {
    throw new Error('node convert: hashAlgorithm is required.');
  }
  if (typeof computeHash !== 'function') {
    throw new Error('node convert: computeHash(data, algorithm) is required.');
  }
  if (typeof readRange !== 'function') {
    throw new Error('node convert: readRange(filePath, offset, length) is required.');
  }
  return {
    async readTensorData(tensor) {
      return readRange(tensor.sourcePath, tensor.offset, tensor.size);
    },
    async readShardRange(index, offset, length) {
      const filename = generateShardFilename(index);
      return readRange(path.join(outputDir, filename), offset, length);
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

function buildConvertReport(result, context) {
  const manifest = result?.manifest ?? null;
  const inference = manifest?.inference && typeof manifest.inference === 'object'
    ? manifest.inference
    : null;
  return validateConversionReport({
    schemaVersion: CONVERSION_REPORT_SCHEMA_VERSION,
    suite: 'convert',
    command: 'convert',
    modelId: manifest?.modelId ?? context.modelId ?? 'unknown',
    timestamp: manifest?.metadata?.convertedAt ?? new Date().toISOString(),
    source: 'doppler',
    result: {
      modelType: context.modelType ?? null,
      outputDir: context.outputDir ?? null,
      shardCount: result?.shardCount ?? null,
      tensorCount: result?.tensorCount ?? null,
      totalSize: result?.totalSize ?? null,
    },
    manifest: manifest
        ? {
          quantization: manifest.quantization ?? null,
          quantizationInfo: manifest.quantizationInfo ?? null,
          inference: {
            schema: inference?.schema ?? null,
          },
        }
      : null,
    executionContractArtifact: result?.executionContractArtifact ?? null,
    layerPatternContractArtifact: result?.layerPatternContractArtifact ?? null,
    requiredInferenceFieldsArtifact: result?.requiredInferenceFieldsArtifact ?? null,
  });
}

function createNodeTensorTransformer(options) {
  const pool = options?.pool;
  const execution = options?.execution;
  const transformTensorBytes = options?.transformTensorBytes;
  const resolveTensorTargetQuant = options?.resolveTensorTargetQuant;
  const normalizeStorageQuant = options?.normalizeStorageQuant;
  const shouldQuantize = options?.shouldQuantize;

  if (!pool || !execution || typeof transformTensorBytes !== 'function') {
    throw new Error('node convert: invalid worker tensor transformer setup.');
  }

  return async function tensorTransformer(input) {
    const tensor = input?.tensor;
    const tensorData = input?.tensorData;
    const transformContext = input?.transformContext ?? {};
    const reportProgress = typeof input?.reportProgress === 'function'
      ? input.reportProgress
      : null;

    if (!tensor || !(tensorData instanceof Uint8Array)) {
      throw new Error('node convert: invalid tensor transform input.');
    }
    const chunkPlan = resolveRowChunkTransformPlan({
      tensor,
      tensorByteLength: tensorData.byteLength,
      execution,
      transformContext,
      resolveTensorTargetQuant,
      normalizeStorageQuant,
      shouldQuantize,
    });

    if (!chunkPlan) {
      const transformed = await pool.transformTensor(tensor, tensorData, transformContext);
      const normalized = normalizeWorkerTransformResult(transformed, tensor);
      reportProgress?.(tensorData.byteLength, tensorData.byteLength);
      return normalized;
    }

    const chunks = [];
    for (let rowStart = 0; rowStart < chunkPlan.rows; rowStart += chunkPlan.rowChunkRows) {
      const rowCount = Math.min(chunkPlan.rowChunkRows, chunkPlan.rows - rowStart);
      const start = rowStart * chunkPlan.rowSourceBytes;
      const end = start + (rowCount * chunkPlan.rowSourceBytes);
      chunks.push({ rowStart, rowCount, start, end });
    }

    const maxInFlightJobs = execution.maxInFlightJobs
      ?? selectRuleValue('converter', 'execution', 'maxInFlightJobs', {
        workers: execution.effectiveWorkers,
      });
    const concurrency = Number.isInteger(maxInFlightJobs) && maxInFlightJobs > 0
      ? maxInFlightJobs
      : execution.effectiveWorkers;

    let processedBytes = 0;
    const chunkResults = await mapWithConcurrency(chunks, concurrency, async (chunk) => {
      const chunkTensorData = tensorData.subarray(chunk.start, chunk.end);
      const chunkTensor = {
        ...tensor,
        shape: [chunk.rowCount, chunkPlan.cols],
      };
      const transformed = await pool.transformTensor(chunkTensor, chunkTensorData, {
        ...transformContext,
        forceQuantizeDecision: chunkPlan.forceQuantizeDecision,
        originalTensorShape: tensor.shape,
      });
      const normalized = normalizeWorkerTransformResult(transformed, chunkTensor);
      processedBytes += chunkTensorData.byteLength;
      reportProgress?.(
        Math.min(processedBytes, tensorData.byteLength),
        tensorData.byteLength
      );
      return normalized;
    });

    if (chunkResults.length === 0) {
      return transformTensorBytes(tensor, tensorData, transformContext);
    }

    const outDtype = chunkResults[0].outDtype ?? tensor.dtype;
    const outLayout = chunkResults[0].outLayout ?? null;
    for (const chunkResult of chunkResults) {
      if ((chunkResult.outDtype ?? tensor.dtype) !== outDtype) {
        throw new Error(`node convert: inconsistent chunk dtype for ${tensor.name}.`);
      }
      if ((chunkResult.outLayout ?? null) !== outLayout) {
        throw new Error(`node convert: inconsistent chunk layout for ${tensor.name}.`);
      }
    }

    const totalOutputBytes = chunkResults.reduce((sum, chunkResult) => (
      sum + chunkResult.tensorData.byteLength
    ), 0);
    const combined = new Uint8Array(totalOutputBytes);
    let outputOffset = 0;
    for (const chunkResult of chunkResults) {
      combined.set(chunkResult.tensorData, outputOffset);
      outputOffset += chunkResult.tensorData.byteLength;
    }

    const companionResults = chunkResults.filter((chunkResult) => (
      chunkResult.companionData instanceof Uint8Array
    ));
    let companionData = null;
    let sourceTransform = null;
    if (companionResults.length > 0) {
      if (companionResults.length !== chunkResults.length) {
        throw new Error(`node convert: inconsistent chunk companion data for ${tensor.name}.`);
      }
      sourceTransform = chunkResults[0].sourceTransform ?? null;
      if (!sourceTransform) {
        throw new Error(`node convert: chunk companion data is missing sourceTransform for ${tensor.name}.`);
      }
      const totalCompanionBytes = companionResults.reduce((sum, chunkResult) => (
        sum + chunkResult.companionData.byteLength
      ), 0);
      companionData = new Uint8Array(totalCompanionBytes);
      let companionOffset = 0;
      for (const chunkResult of companionResults) {
        companionData.set(chunkResult.companionData, companionOffset);
        companionOffset += chunkResult.companionData.byteLength;
      }
    }

    return {
      tensorData: combined,
      outDtype,
      outLayout,
      ...(companionData ? { companionData } : {}),
      ...(sourceTransform ? { sourceTransform } : {}),
    };
  };
}

function createNodeLargeTensorTransformer(options) {
  const pool = options?.pool;
  const execution = options?.execution;
  const readRange = options?.readRange;
  const resolveTensorTargetQuant = options?.resolveTensorTargetQuant;
  const normalizeStorageQuant = options?.normalizeStorageQuant;
  const shouldQuantize = options?.shouldQuantize;

  if (!pool || typeof pool.transformTensor !== 'function' || !execution || typeof readRange !== 'function') {
    throw new Error('node convert: invalid large tensor transformer setup.');
  }

  return async function largeTensorTransformer(input) {
    const tensor = input?.tensor;
    const transformContext = input?.transformContext ?? {};
    const reportProgress = typeof input?.reportProgress === 'function'
      ? input.reportProgress
      : null;
    const writeChunk = typeof input?.writeChunk === 'function'
      ? input.writeChunk
      : null;

    if (!tensor || typeof tensor !== 'object') {
      throw new Error('node convert: invalid large tensor transform input.');
    }
    if (!writeChunk) {
      throw new Error('node convert: large tensor transform requires writeChunk().');
    }

    const tensorByteLength = Number(tensor?.size ?? 0);
    const chunkPlan = resolveRowChunkTransformPlan({
      tensor,
      tensorByteLength,
      execution,
      transformContext,
      resolveTensorTargetQuant,
      normalizeStorageQuant,
      shouldQuantize,
    });
    if (!chunkPlan) {
      throw new Error(
        `node convert: tensor "${tensor.name}" is ${tensorByteLength} bytes and exceeds the single-buffer limit, ` +
        'but it is not eligible for row-chunked conversion.'
      );
    }
    if (chunkPlan.rowSourceBytes > MAX_NODE_CONVERT_BUFFER_BYTES) {
      throw new Error(
        `node convert: tensor "${tensor.name}" cannot be row-chunked because each source row is ` +
        `${chunkPlan.rowSourceBytes} bytes, above the single-buffer limit ${MAX_NODE_CONVERT_BUFFER_BYTES}.`
      );
    }
    const maxRowsPerRead = Math.floor(MAX_NODE_CONVERT_BUFFER_BYTES / chunkPlan.rowSourceBytes);
    if (maxRowsPerRead < 1) {
      throw new Error(
        `node convert: tensor "${tensor.name}" cannot be row-chunked under the current single-buffer limit.`
      );
    }
    if (chunkPlan.rowChunkRows > maxRowsPerRead) {
      throw new Error(
        `node convert: execution.rowChunkRows=${chunkPlan.rowChunkRows} is too large for tensor "${tensor.name}". ` +
        `Use ${maxRowsPerRead} rows or fewer for streamed conversion.`
      );
    }

    let processedBytes = 0;
    let outDtype = null;
    let outLayout = null;

    for (let rowStart = 0; rowStart < chunkPlan.rows; rowStart += chunkPlan.rowChunkRows) {
      const rowCount = Math.min(chunkPlan.rowChunkRows, chunkPlan.rows - rowStart);
      const chunkOffset = rowStart * chunkPlan.rowSourceBytes;
      const chunkLength = rowCount * chunkPlan.rowSourceBytes;
      const chunkTensor = {
        ...tensor,
        shape: [rowCount, chunkPlan.cols],
        size: chunkLength,
      };
      const rawChunk = await readRange(
        tensor.sourcePath,
        tensor.offset + chunkOffset,
        chunkLength
      );
      const chunkTensorData = new Uint8Array(rawChunk);
      const transformed = await pool.transformTensor(chunkTensor, chunkTensorData, {
        ...transformContext,
        forceQuantizeDecision: chunkPlan.forceQuantizeDecision,
        originalTensorShape: tensor.shape,
      });
      const normalized = normalizeWorkerTransformResult(transformed, chunkTensor);
      if (outDtype == null) {
        outDtype = normalized.outDtype ?? tensor.dtype;
        outLayout = normalized.outLayout ?? null;
      } else {
        if ((normalized.outDtype ?? tensor.dtype) !== outDtype) {
          throw new Error(`node convert: inconsistent streamed chunk dtype for ${tensor.name}.`);
        }
        if ((normalized.outLayout ?? null) !== outLayout) {
          throw new Error(`node convert: inconsistent streamed chunk layout for ${tensor.name}.`);
        }
      }
      await writeChunk(normalized);
      processedBytes += chunkLength;
      reportProgress?.(
        Math.min(processedBytes, tensorByteLength),
        tensorByteLength
      );
    }

    return {
      outDtype: outDtype ?? tensor.dtype,
      outLayout: outLayout ?? null,
    };
  };
}

export async function convertSafetensorsDirectory(options) {
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
  const totalTimer = createStageTimer('Total');
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
    architecture = extractArchitecture(config, null);
    const tokenizerJsonPath = path.join(inputDir, 'tokenizer.json');
    tokenizerModelPath = path.join(inputDir, 'tokenizer.model');
    const tokenizerConfigPath = path.join(inputDir, 'tokenizer_config.json');
    tokenizerJson = await readOptionalJson(tokenizerJsonPath);
    tokenizerConfig = await readOptionalJson(tokenizerConfigPath);
    hasTokenizerModel = await fileExists(tokenizerModelPath);
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
      let gpuRuntime;
      try {
        gpuRuntime = await loadNodeGpuCastRuntime();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `node convert: execution.useGpuCast requires a WebGPU-capable Node runtime. ${message}`
        );
      }
      gpuTensorTransformer = createNodeGpuTensorTransformer({
        runtime: gpuRuntime,
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
    totalTimer.stop();
  }
}
