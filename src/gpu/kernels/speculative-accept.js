import { recordGather } from './gather.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { isGpuBufferInstance } from '../weight-buffer.js';

/** @type {WeakMap<import('./speculative-accept.js').SpeculativeAcceptance, {device: GPUDevice; buffer: GPUBuffer; maxTokens: number}>} */
const recordedAcceptances = new WeakMap();

/** @param {GPUBuffer} buffer @param {number} size @param {number} usage @param {string} label */
function requireBuffer(buffer, size, usage, label) {
  if (!isGpuBufferInstance(buffer) || buffer.size < size || (buffer.usage & usage) !== usage) {
    throw new Error(`[Speculative acceptance] ${label} requires ${size} bytes and the declared GPU usages.`);
  }
}

/** @type {import('./speculative-accept.js').recordSpeculativeAcceptance} */
export async function recordSpeculativeAcceptance(recorder, drafts, targets, options) {
  const { draftCount, outputLimit, vocabSize, outputBuffer } = options;
  const maxTokens = draftCount + 1;
  if (!Number.isSafeInteger(draftCount) || draftCount < 1 || draftCount >= 0xffffffff
    || !Number.isSafeInteger(outputLimit) || outputLimit < 1 || outputLimit > maxTokens
    || !Number.isSafeInteger(vocabSize) || vocabSize < 1 || vocabSize > 0xffffffff) {
    throw new Error('[Speculative acceptance] Explicit positive draft, output and vocabulary bounds are required.');
  }
  requireBuffer(drafts, draftCount * 4, GPUBufferUsage.STORAGE, 'draft IDs');
  requireBuffer(targets, maxTokens * 4, GPUBufferUsage.STORAGE, 'target IDs');
  requireBuffer(outputBuffer, (maxTokens + 5) * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'result');
  if (outputBuffer === drafts || outputBuffer === targets) {
    throw new Error('[Speculative acceptance] Result must not alias either input.');
  }
  await unifiedKernelWrapper('speculative_accept', recorder, 'greedy', [drafts, targets, outputBuffer], {
    draft_count: draftCount, output_limit: outputLimit, vocab_size: vocabSize, _pad: 0,
  }, [1, 1, 1]);
  const result = Object.freeze({ buffer: outputBuffer, maxTokens });
  recordedAcceptances.set(result, { device: recorder.device, buffer: outputBuffer, maxTokens });
  return result;
}

/**
 * Private mutable parameter frame: GPU-selected count never rewrites the content-addressed uniform cache.
 * The count is bounded by the acceptance kernel. An invalid token produces zero updates and a failed status.
 * The batch owner must check that status before publishing tokens or continuing the attempt.
 * @type {import('./speculative-accept.js').createSpeculativeRecoveryUniform}
 */
export function createSpeculativeRecoveryUniform(recorder, acceptance, payload, maxTokens) {
  const source = recordedAcceptances.get(acceptance);
  if (!source || source.device !== recorder.device || source.maxTokens !== maxTokens) {
    throw new Error('[Speculative recovery] A same-device recorded acceptance with matching token bounds is required.');
  }
  if (!(payload instanceof ArrayBuffer) || payload.byteLength < 4 || payload.byteLength % 4 !== 0) {
    throw new Error('[Speculative recovery] An aligned, private parameter payload is required.');
  }
  const params = recorder.createTempBuffer(payload.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'speculative_recovery_params');
  recorder.device.queue.writeBuffer(params, 0, payload);
  recorder.getEncoder().copyBufferToBuffer(source.buffer, 0, params, 0, 4);
  return params;
}

/** @type {import('./speculative-accept.js').recordSpeculativeCarry} */
export async function recordSpeculativeCarry(recorder, acceptance, hidden, hiddenSize, carriedHidden, carriedToken) {
  const source = recordedAcceptances.get(acceptance);
  if (!source || source.device !== recorder.device || !Number.isSafeInteger(hiddenSize) || hiddenSize < 1
      || hidden.dtype !== 'f32' || hidden.precision != null
      || hidden.shape.length !== 2 || hidden.shape[0] !== source.maxTokens || hidden.shape[1] !== hiddenSize) {
    throw new Error('[Speculative carry] Same-device acceptance and explicit F32 verification rows are required.');
  }
  requireBuffer(hidden.buffer, source.maxTokens * hiddenSize * 4, GPUBufferUsage.STORAGE, 'verification hidden');
  requireBuffer(carriedHidden, hiddenSize * 4, GPUBufferUsage.STORAGE, 'carried hidden');
  requireBuffer(carriedToken, 4, GPUBufferUsage.COPY_DST, 'carried token');
  if (new Set([source.buffer, hidden.buffer, carriedHidden, carriedToken]).size !== 4) {
    throw new Error('[Speculative carry] Input, output and control buffers must be distinct.');
  }
  await recordGather(recorder, source.buffer, hidden.buffer, 1, hiddenSize, source.maxTokens, {
    indexOffset: 3, transpose: false, inputHiddenSize: hiddenSize, hiddenOffset: 0,
    embeddingDtype: 'f32', outputDtype: 'f32', outputBuffer: carriedHidden,
  });
  recorder.getEncoder().copyBufferToBuffer(source.buffer, 16, carriedToken, 0, 4);
}
