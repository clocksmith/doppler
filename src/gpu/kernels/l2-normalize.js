import { createTensor } from '../tensor.js';
import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

export async function runL2Normalize(input, options) {
  const rowCount = Number(options?.rowCount);
  const hiddenSize = Number(options?.hiddenSize);
  if (input?.dtype !== 'f32') {
    throw new Error(`[L2Normalize] input must be f32, got "${input?.dtype ?? 'unknown'}".`);
  }
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error('[L2Normalize] rowCount must be a positive integer.');
  }
  if (!Number.isInteger(hiddenSize) || hiddenSize < 1) {
    throw new Error('[L2Normalize] hiddenSize must be a positive integer.');
  }
  const output = acquireBuffer(
    rowCount * hiddenSize * Float32Array.BYTES_PER_ELEMENT,
    undefined,
    'l2_normalize_output'
  );
  try {
    await unifiedKernelWrapper(
      'l2_normalize',
      null,
      'f32',
      [input, output],
      { row_count: rowCount, hidden_size: hiddenSize },
      rowCount
    );
    return createTensor(output, 'f32', [rowCount, hiddenSize], 'l2_normalize_output');
  } catch (error) {
    releaseBuffer(output);
    throw error;
  }
}
