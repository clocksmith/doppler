import { getDevice } from '../device.js';
import { createTensor } from '../tensor.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

const WORKGROUP_SIZE = 256;

export async function runEmbeddingPool(input, options) {
  const device = getDevice();
  if (!device) throw new Error('[EmbeddingPool] GPU device is required.');
  const rowCount = Number(options?.rowCount);
  const hiddenSize = Number(options?.hiddenSize);
  const mode = options?.mode;
  const includedCount = Number(options?.includedCount ?? rowCount);
  const lastIndex = Number(options?.lastIndex ?? (rowCount - 1));
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error('[EmbeddingPool] rowCount must be a positive integer.');
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new Error('[EmbeddingPool] hiddenSize must be a positive integer.');
  }
  if (mode !== 'mean' && mode !== 'last') {
    throw new Error('[EmbeddingPool] mode must be "mean" or "last".');
  }
  if (!Number.isInteger(includedCount) || includedCount < 1 || includedCount > rowCount) {
    throw new Error('[EmbeddingPool] includedCount must be within the input row range.');
  }
  if (!Number.isInteger(lastIndex) || lastIndex < 0 || lastIndex >= rowCount) {
    throw new Error('[EmbeddingPool] lastIndex must identify an input row.');
  }
  if (input?.dtype !== 'f16' && input?.dtype !== 'f32') {
    throw new Error(`[EmbeddingPool] unsupported input dtype "${input?.dtype ?? 'unknown'}".`);
  }
  const maskValues = options?.mask ?? null;
  if (maskValues != null && maskValues.length !== rowCount) {
    throw new Error('[EmbeddingPool] mask length must equal rowCount.');
  }
  const mask = acquireBuffer(
    Math.max(4, rowCount * Uint32Array.BYTES_PER_ELEMENT),
    undefined,
    'embedding_pool_mask'
  );
  const output = acquireBuffer(
    hiddenSize * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    'embedding_pool_output'
  );
  device.queue.writeBuffer(mask, 0, maskValues ?? new Uint32Array(rowCount));
  try {
    await unifiedKernelWrapper(
      'embedding_pool',
      null,
      input.dtype,
      [input, mask, output],
      {
        row_count: rowCount,
        hidden_size: hiddenSize,
        mode: mode === 'last' ? 1 : 0,
        has_mask: maskValues == null ? 0 : 1,
        included_count: includedCount,
        last_index: lastIndex,
      },
      Math.ceil(hiddenSize / WORKGROUP_SIZE)
    );
    return createTensor(output, 'f32', [1, hiddenSize], 'embedding_pool_output');
  } catch (error) {
    releaseBuffer(output);
    throw error;
  } finally {
    releaseBuffer(mask);
  }
}
