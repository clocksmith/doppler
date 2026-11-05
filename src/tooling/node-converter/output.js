import { createHash, randomUUID } from 'node:crypto';
import { DEFAULT_CONVERTER_EXECUTION_CONFIG } from '../../config/schema/converter.schema.js';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { installNodeFileFetchShim } from '../node-file-fetch.js';
import { NodeConvertWorkerPool } from '../node-convert-worker-pool.js';
import { createConversionRunTiming } from '../conversion-run-timing.js';
import { createRowChunks, mapOrderedChunkBatches } from '../node-converter-chunk-batches.js';
import { bootstrapNodeWebGPU } from '../node-webgpu.js';
import { buildManifestIntegrityFromModelDir } from '../rdrr-integrity-refresh.js';
import { applySourceTensorRules } from '../../converter/source-tensor-rules.js';
import {
  buildSourceTokenizerJson,
  validateSourceTokenizerPolicy,
} from '../../converter/source-tokenizer.js';
import { isPlainObject } from '../../formats/plain-object.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { log, trace } from '../../debug/index.js';
import { saveReport } from '../../storage/reports.js';
import {
  CONVERSION_REPORT_SCHEMA_VERSION,
  validateConversionReport,
} from '../../config/schema/conversion-report.schema.js';
import { createNodeGpuTensorTransformer, createNodeLargeTensorTransformer, createNodeTensorTransformer } from './input.js';

export function asPositiveInteger(value, label) {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`node convert: ${label} must be a positive integer.`);
  }
  return value;
}

export function normalizeExecutionConfig(value, defaults) {
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
  const retainedSourceFiles = value.retainedSourceFiles ?? defaults.retainedSourceFiles;
  if (retainedSourceFiles !== 'link' && retainedSourceFiles !== 'copy') {
    throw new Error('node convert: execution.retainedSourceFiles must be "link" or "copy".');
  }

  return {
    workers,
    workerCountPolicy: workerCountPolicyRaw,
    rowChunkRows,
    rowChunkMinTensorBytes,
    maxInFlightJobs,
    useGpuCast,
    gpuCastRequestedExplicitly,
    gpuCastMinTensorBytes,
    retainedSourceFiles,
  };
}

export function generateShardFilename(index) {
  return `shard_${String(index).padStart(5, '0')}.bin`;
}

export function createFileRangeReader() {
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

export function createNodeConvertIO(outputDir, options) {
  const hashAlgorithm = options?.hashAlgorithm;
  const computeHash = options?.computeHash;
  const readRange = options?.readRange;
  const retainedSourceFiles = options?.retainedSourceFiles ?? DEFAULT_CONVERTER_EXECUTION_CONFIG.retainedSourceFiles;
  if (retainedSourceFiles !== 'link' && retainedSourceFiles !== 'copy') throw new Error('node convert: retainedSourceFiles must be link or copy.');
  if (!hashAlgorithm || typeof hashAlgorithm !== 'string') {
    throw new Error('node convert: hashAlgorithm is required.');
  }
  if (typeof computeHash !== 'function') {
    throw new Error('node convert: computeHash(data, algorithm) is required.');
  }
  if (typeof readRange !== 'function') {
    throw new Error('node convert: readRange(filePath, offset, length) is required.');
  }
  const retainedNames = new Map();
  const existingRelativeSource = (sourcePath) => {
    const relative = path.relative(path.resolve(outputDir), path.resolve(sourcePath));
    return relative && !path.isAbsolute(relative) && relative !== '..' && !relative.startsWith(`..${path.sep}`)
      ? relative.split(path.sep).join('/') : null;
  };
  const linkSource = async (sourcePath, target) => {
    try { await fs.link(sourcePath, target); }
    catch (error) {
      if (['EXDEV', 'EPERM', 'ENOTSUP', 'EOPNOTSUPP'].includes(error?.code)) {
        throw new Error(`node convert: cannot retain ${sourcePath} without copying on this filesystem. Put the source directory inside the artifact output root, or use a filesystem supporting hard links.`, { cause: error });
      }
      throw error;
    }
  };
  return {
    async preflightRetainedSources(sourcePaths) {
      for (const sourcePath of sourcePaths) {
        if (retainedSourceFiles === 'copy' || existingRelativeSource(sourcePath)) { await fs.access(sourcePath); continue; }
        const probe = path.join(outputDir, `.doppler-retention-${randomUUID()}`);
        await linkSource(sourcePath, probe);
        await fs.unlink(probe);
      }
    },
    async readTensorData(tensor) {
      return readRange(tensor.sourcePath, tensor.offset, tensor.size);
    },
    async readShardRange(index, offset, length) {
      const filename = retainedNames.get(index) ?? generateShardFilename(index);
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
    async retainSourceFile(sourcePath, index, sha256) {
      const existing = retainedSourceFiles === 'link' ? existingRelativeSource(sourcePath) : null;
      const filename = existing ?? generateShardFilename(index);
      const target = path.join(outputDir, filename);
      let ownsTarget = false;
      try {
        const hasher = createHash('sha256');
        let size = 0;
        if (retainedSourceFiles === 'copy') {
          const output = await fs.open(target, 'wx');
          ownsTarget = true;
          try {
            for await (const chunk of createReadStream(sourcePath, { highWaterMark: 8 << 20 })) {
              let written = 0;
              while (written < chunk.length) {
                const result = await output.write(chunk, written, chunk.length - written);
                if (result.bytesWritten <= 0) throw new Error('Retained source copy made no write progress.');
                written += result.bytesWritten;
              }
              hasher.update(chunk);
              size += chunk.length;
            }
          } finally { await output.close(); }
        } else {
          if (!existing) { await linkSource(sourcePath, target); ownsTarget = true; }
          for await (const chunk of createReadStream(target, { highWaterMark: 8 << 20 })) {
            hasher.update(chunk);
            size += chunk.length;
          }
        }
        const hash = hasher.digest('hex');
        if (hash !== sha256) throw new Error(`node convert: ${sourcePath} SHA-256 ${hash} differs from its pin ${sha256}.`);
        retainedNames.set(index, filename);
        return { size, hash, ...(existing ? { filename } : {}) };
      } catch (error) {
        if (ownsTarget) {
          try { await fs.unlink(target); }
          catch (cleanupError) { throw new AggregateError([error, cleanupError], 'Retained source verification and cleanup failed.', { cause: error }); }
        }
        throw error;
      }
    },
  };
}

export function buildConvertReport(result, context) {
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
    ...context.physicalTiming,
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
