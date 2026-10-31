import { ggufBlockFormat, ggufBlockBytes, isGgufBlockDtype } from '../../formats/gguf/block-storage.js';
import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { selectRuleValue } from '../../rules/rule-registry.js';
import { getKernelConfig } from './kernel-configs.js';
import { unifiedKernelWrapper, withKernelOutput } from './kernel-execution.js';

/** @param {Record<string, unknown> | undefined} overrides @param {string} key */
function positiveOverride(overrides,key) {
  const value=overrides?.[key];
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1) throw new Error(`Q4_0 invalid ${key} override.`);
  return value;
}

/** @type {typeof import('./q4_0.js').q40Dispatch} */
export function q40Dispatch(device, count) {
  const limit = device?.limits?.maxComputeWorkgroupsPerDimension;
  if (!Number.isSafeInteger(count) || count < 1 || typeof limit !== 'number' || !Number.isSafeInteger(limit) || limit < 1) {
    throw new Error('Q4_0 dispatch requires explicit positive geometry and device limits.');
  }
  const x = Math.min(count, limit), y = Math.ceil(count / x);
  if (y > limit) throw new Error('Q4_0 dispatch exceeds device axis limits.');
  return [x, y, 1];
}

/** @type {typeof import('./q4_0.js').assertQ40Buffer} */
export function assertQ40Buffer(device, buffer, bytes, usage, label) {
  if (!Number.isSafeInteger(bytes) || bytes < 1 || bytes > 0xffffffff
    || !buffer || buffer.size < Math.ceil(bytes / 4) * 4 || !(buffer.usage & usage)
    || buffer.size > device.limits.maxStorageBufferBindingSize) {
    throw new Error(`Q4_0 ${label} storage is undersized, incompatible or exceeds binding limits.`);
  }
}

/** @type {import('./q4_0.js').Q40Execute} */
async function execute(recorder, input, weights, M, N, K, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('Q4_0 requires an initialized GPU device.');
  options.signal?.throwIfAborted();
  if (![M, N, K].every(value => Number.isSafeInteger(value) && value > 0 && value <= 0xffffff)
    || M*K > 0xffffffff || M*N > 0xffffffff) throw new Error('Q4_0 requires bounded explicit M/N/K.');
  if (input.dtype !== 'f32' || input.precision != null || options.outputDtype !== 'f32'
    || (options.outputPrecision != null && options.outputPrecision !== 'f32')
    || (options.transposeB != null && options.transposeB !== true && options.transposeB !== 'auto')
    || (options.bDtype != null && options.bDtype !== weights.dtype)) {
    throw new Error('Q4_0 requires declared scalar F32 activation/output and whole row-packed weights.');
  }
  if (!isGgufBlockDtype(weights.dtype) || ![1,2,3].includes(input.shape.length)
    || (input.shape.length > 1 && input.shape.at(-1) !== K) || ![2,3].includes(weights.shape.length)
    || weights.shape.at(-2) !== N || weights.shape.at(-1) !== K) {
    throw new Error(`Q4_0 operand shapes ${input.shape}/${weights.shape} differ from explicit ${M}/${N}/${K}.`);
  }
  const aOffset = options.aOffset ?? 0, bOffset = options.bOffset ?? 0, cOffset = options.cOffset ?? 0;
  const format = ggufBlockFormat(weights.dtype);
  const rowBytes = K/format.values*format.bytes;
  if (![aOffset,bOffset,cOffset].every(value => Number.isSafeInteger(value) && value >= 0)
    || aOffset%4 || cOffset%4 || bOffset%rowBytes
    || aOffset/4+M*K > input.shape.reduce((a,b)=>a*b,1)) throw new Error('Q4_0 offsets exceed declared tensor ranges.');
  if (weights.layout !== 'row') throw new Error('Packed GGUF projections require row-packed weights.');
  const bytes = ggufBlockBytes(weights.shape,weights.dtype);
  if (bOffset + N*rowBytes > bytes) throw new Error('Q4_0 weight offset exceeds bank storage.');
  assertQ40Buffer(device, input.buffer, aOffset+M*K*4, GPUBufferUsage.STORAGE, 'input');
  assertQ40Buffer(device, weights.buffer, bytes, GPUBufferUsage.STORAGE, 'weights');
  const variant = options.variant ?? selectRuleValue('kernels', 'matmul', 'q40Variant', { M, weightDtype: weights.dtype });
  if (typeof variant !== 'string' || !variant) throw new Error('Q4_0 requires an explicit registered variant.');
  const config = getKernelConfig('matmul', variant);
  if (config.weightDtype !== weights.dtype || config.outputDtype !== 'f32'
    || config.wgslOverrides?.QUANT_FORMAT !== format.code || config.wgslOverrides?.BLOCK_BYTES !== format.bytes
    || config.wgslOverrides?.BLOCK_VALUES !== format.values) throw new Error('Q4_0 declared kernel has incompatible storage.');
  if (options.constants && Object.entries(options.constants).some(([key,value]) => config.wgslOverrides?.[key] !== value)) {
    throw new Error('Q4_0 constants differ from the declared registry program.');
  }
  const alpha = options.alpha ?? 1;
  if (!Number.isFinite(alpha) || !Number.isFinite(Math.fround(alpha))) throw new Error('Q4_0 alpha must be finite F32.');
  const geometry = config.variantMetadata?.q40Geometry;
  const columns = positiveOverride(config.wgslOverrides,'COLS_PER_WG');
  const threads = positiveOverride(config.wgslOverrides,'THREADS_PER_COL_GEMV');
  const width = positiveOverride(config.wgslOverrides,'WORKGROUP_SIZE');
  const tile = positiveOverride(config.wgslOverrides,'TILE_SIZE');
  let count;
  if (geometry === 'gemv') {
    if (width !== columns*threads
      || (threads & (threads-1)) !== 0) throw new Error('Q4_0 invalid reduction geometry.');
    count = Math.ceil(N / columns)*M;
  } else if (geometry === 'tiled') {
    count = Math.ceil(N / tile)*Math.ceil(M / tile);
  } else { throw new Error('Q4_0 missing declared dispatch geometry.'); }
  const dispatch = q40Dispatch(device, count);
  const supplied = options.outputBuffer ?? null;
  if (supplied) {
    assertQ40Buffer(device, supplied, cOffset+M*N*4, GPUBufferUsage.STORAGE, 'output');
    if (supplied === input.buffer || supplied === weights.buffer) throw new Error('Q4_0 output must not alias input weights or activations.');
  }
  if (cOffset+M*N*4 > device.limits.maxStorageBufferBindingSize) throw new Error('Q4_0 output exceeds binding limits.');
  return withKernelOutput(recorder, supplied, cOffset+M*N*4, 'q4_0_output', async output => {
    await unifiedKernelWrapper('matmul', recorder, variant, [input.buffer, weights.buffer, output], {
      M, N, K, alpha, blocks: K/format.values, a_offset: aOffset/4, b_row: bOffset/rowBytes, c_offset: cOffset/4,
    }, dispatch, null, null, null, options.signal ?? null);
    return createTensor(output, 'f32', [M,N], 'q4_0_output');
  });
}
/** @type {typeof import('./q4_0.js').runQ40Matmul} */
export function runQ40Matmul(input, weights, M, N, K, options) { return execute(null,input,weights,M,N,K,options); }
/** @type {typeof import('./q4_0.js').recordQ40Matmul} */
export function recordQ40Matmul(recorder,input,weights,M,N,K,options) { return execute(recorder,input,weights,M,N,K,options); }

/** @type {typeof import('./q4_0.js').executeQ40Gather} */
export async function executeQ40Gather(recorder, indices, weights, numTokens, hiddenSize, vocabSize, options) {
  const device = recorder?.device ?? getDevice();
  if (!device) throw new Error('Q4_0 gather requires an initialized device.');
  if (![numTokens,hiddenSize,vocabSize].every(value => Number.isSafeInteger(value) && value > 0)
    || options.outputDtype !== 'f32' || options.transpose === true
    || (options.inputHiddenSize != null && options.inputHiddenSize !== hiddenSize)
    || (options.hiddenOffset != null && options.hiddenOffset !== 0) || options.indirectBuffer) {
    throw new Error('Q4_0 gather requires explicit F32 output and complete row-packed embedding rows.');
  }
  const offset = options.indexOffset ?? 0;
  if (!Number.isSafeInteger(offset) || offset < 0) throw new Error('Q4_0 gather indexOffset must be nonnegative.');
  const dtype = options.embeddingDtype;
  if (typeof dtype !== 'string') throw new Error('Packed gather requires explicit storage dtype.');
  const bytes = ggufBlockBytes([vocabSize,hiddenSize],dtype);
  assertQ40Buffer(device, weights, bytes, GPUBufferUsage.STORAGE, 'embeddings');
  assertQ40Buffer(device, indices, (offset+numTokens)*4, GPUBufferUsage.STORAGE, 'indices');
  const variant = selectRuleValue('kernels','gather','q40Variant',{ outputDtype: options.outputDtype, weightDtype: dtype });
  const config = getKernelConfig('gather',variant);
  const groups = q40Dispatch(device,Math.ceil(numTokens*hiddenSize/config.workgroupSize[0]));
  const supplied = options.outputBuffer ?? null;
  if (supplied) {
    assertQ40Buffer(device,supplied,numTokens*hiddenSize*4,GPUBufferUsage.STORAGE,'gather output');
    if (supplied === weights || supplied === indices) throw new Error('Q4_0 gather output aliases an input.');
  }
  if (numTokens*hiddenSize*4 > device.limits.maxStorageBufferBindingSize) throw new Error('Q4_0 gather exceeds binding limits.');
  return withKernelOutput(recorder,supplied,numTokens*hiddenSize*4,'q4_0_gather',async output => {
    await unifiedKernelWrapper('gather',recorder,variant,[indices,weights,output], {
      num_tokens:numTokens,hidden_size:hiddenSize,vocab_size:vocabSize,index_offset:offset,
    },groups);
    return createTensor(output,'f32',[numTokens,hiddenSize],'q4_0_gather');
  });
}
