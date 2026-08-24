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
import { isPlainObject } from '../../utils/plain-object.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { log, trace } from '../../debug/index.js';
import { saveReport } from '../../storage/reports.js';
import {
  CONVERSION_REPORT_SCHEMA_VERSION,
  validateConversionReport,
} from '../../config/schema/conversion-report.schema.js';

export function getDtypeBytes(dtype) {
  const upper = String(dtype || '').toUpperCase();
  if (upper === 'F32') return 4;
  if (upper === 'F16' || upper === 'BF16') return 2;
  return null;
}

export function normalizeWorkerTransformResult(result, tensor) {
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
    ...(result.storage ? { storage: result.storage } : {}),
  };
}

export const MAX_NODE_CONVERT_BUFFER_BYTES = Math.min(Buffer.kMaxLength, 0x7fff_ffff);

export function resolveRowChunkTransformPlan(input) {
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

export async function mapWithConcurrency(items, concurrency, mapper) {
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

export function createNodeGpuTensorTransformer(options) {
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

export function createNodeTensorTransformer(options) {
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
    const storage = chunkResults[0].storage ?? null;
    for (const chunkResult of chunkResults) {
      if ((chunkResult.outDtype ?? tensor.dtype) !== outDtype) {
        throw new Error(`node convert: inconsistent chunk dtype for ${tensor.name}.`);
      }
      if ((chunkResult.outLayout ?? null) !== outLayout) {
        throw new Error(`node convert: inconsistent chunk layout for ${tensor.name}.`);
      }
      if (JSON.stringify(chunkResult.storage ?? null) !== JSON.stringify(storage)) {
        throw new Error(`node convert: inconsistent chunk storage descriptor for ${tensor.name}.`);
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
      ...(storage ? { storage } : {}),
      ...(companionData ? { companionData } : {}),
      ...(sourceTransform ? { sourceTransform } : {}),
    };
  };
}

export function createNodeLargeTensorTransformer(options) {
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

    const maxInFlightJobs = execution.maxInFlightJobs ?? selectRuleValue(
      'converter', 'execution', 'maxInFlightJobs', { workers: execution.effectiveWorkers }
    );
    const batchSize = Number.isInteger(maxInFlightJobs) && maxInFlightJobs > 0
      ? maxInFlightJobs : execution.effectiveWorkers;
    const chunks = createRowChunks(chunkPlan);

    let processedBytes = 0;
    let outDtype = null;
    let outLayout = null;
    let storage = null;

    await mapOrderedChunkBatches({
      chunks, batchSize,
      async transform(chunk) {
        const chunkTensor = { ...tensor, shape: [chunk.rowCount, chunkPlan.cols], size: chunk.length };
        const rawChunk = await readRange(tensor.sourcePath, tensor.offset + chunk.offset, chunk.length);
        const chunkTensorData = new Uint8Array(rawChunk);
        const transformed = await pool.transformTensor(chunkTensor, chunkTensorData, {
          ...transformContext,
          forceQuantizeDecision: chunkPlan.forceQuantizeDecision,
          originalTensorShape: tensor.shape,
        });
        const normalized = normalizeWorkerTransformResult(transformed, chunkTensor);
        processedBytes += chunk.length;
        reportProgress?.(Math.min(processedBytes, tensorByteLength), tensorByteLength);
        return normalized;
      },
      async consume(normalized) {
        if (outDtype == null) {
          outDtype = normalized.outDtype ?? tensor.dtype;
          outLayout = normalized.outLayout ?? null;
          storage = normalized.storage ?? null;
        } else {
          if ((normalized.outDtype ?? tensor.dtype) !== outDtype) {
            throw new Error(`node convert: inconsistent streamed chunk dtype for ${tensor.name}.`);
          }
          if ((normalized.outLayout ?? null) !== outLayout) {
            throw new Error(`node convert: inconsistent streamed chunk layout for ${tensor.name}.`);
          }
          if (JSON.stringify(normalized.storage ?? null) !== JSON.stringify(storage)) {
            throw new Error(`node convert: inconsistent streamed chunk storage descriptor for ${tensor.name}.`);
          }
        }
        await writeChunk(normalized);
      },
    });

    return {
      outDtype: outDtype ?? tensor.dtype,
      outLayout: outLayout ?? null,
      ...(storage ? { storage } : {}),
    };
  };
}
