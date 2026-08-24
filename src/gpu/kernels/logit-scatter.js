import { createTensor } from '../tensor.js';
import { unifiedKernelWrapper } from './kernel-execution.js';

const WORKGROUP_SIZE = 256;

export async function runLogitScatter(input, outputBuffer, options) {
  const rowCount = Number(options?.rowCount);
  const chunkColumns = Number(options?.chunkColumns);
  const targetColumns = Number(options?.targetColumns);
  const columnOffset = Number(options?.columnOffset);
  if (!Number.isInteger(rowCount) || rowCount < 1) {
    throw new Error('[LogitScatter] rowCount must be a positive integer.');
  }
  if (!Number.isInteger(chunkColumns) || chunkColumns < 1) {
    throw new Error('[LogitScatter] chunkColumns must be a positive integer.');
  }
  if (!Number.isInteger(targetColumns) || targetColumns < chunkColumns) {
    throw new Error('[LogitScatter] targetColumns must be an integer not smaller than chunkColumns.');
  }
  if (!Number.isInteger(columnOffset) || columnOffset < 0 || columnOffset + chunkColumns > targetColumns) {
    throw new Error('[LogitScatter] columnOffset places the chunk outside the target tensor.');
  }
  if (input?.dtype !== 'f16' && input?.dtype !== 'f32') {
    throw new Error(`[LogitScatter] unsupported input dtype "${input?.dtype ?? 'unknown'}".`);
  }
  await unifiedKernelWrapper(
    'logit_scatter',
    null,
    input.dtype,
    [input, outputBuffer],
    {
      row_count: rowCount,
      chunk_columns: chunkColumns,
      target_columns: targetColumns,
      column_offset: columnOffset,
    },
    Math.ceil((rowCount * chunkColumns) / WORKGROUP_SIZE)
  );
  return createTensor(outputBuffer, 'f32', [rowCount, targetColumns], 'logit_scatter_output');
}
