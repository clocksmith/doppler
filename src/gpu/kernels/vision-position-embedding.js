import { acquireBuffer, releaseBuffer } from '../../memory/buffer-pool.js';
import { createTensor, dtypeBytes } from '../tensor.js';
import { WORKGROUP_SIZES } from './constants.js';
import { unifiedKernelWrapper } from './kernel-execution.js';
import { selectRuleValue } from './rule-registry.js';

function requirePositiveInteger(value, label) {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`Vision position embedding requires ${label} to be a positive integer.`);
  }
}

export async function runVisionPositionEmbedding(table, geometry) {
  const { gridHeight, gridWidth, positionEmbeddingSize, hiddenSize } = geometry;
  for (const [label, value] of Object.entries({
    gridHeight,
    gridWidth,
    positionEmbeddingSize,
    hiddenSize,
  })) {
    requirePositiveInteger(value, label);
  }
  if (gridHeight > positionEmbeddingSize || gridWidth > positionEmbeddingSize) {
    throw new Error(
      `Vision grid ${gridWidth}x${gridHeight} exceeds positionEmbeddingSize=${positionEmbeddingSize}.`
    );
  }
  if (!table?.buffer || !Array.isArray(table.shape)) {
    throw new Error('Vision position embedding requires a typed GPU table tensor.');
  }
  if (table.dtype !== 'f32' && table.dtype !== 'f16') {
    throw new Error(`Vision position embedding requires f32 or f16 table data, got ${table.dtype}.`);
  }
  const tableElements = table.shape.reduce((product, value) => product * value, 1);
  const expectedTableElements = 2 * positionEmbeddingSize * hiddenSize;
  if (tableElements !== expectedTableElements) {
    throw new Error(
      `Vision position embedding table has ${tableElements} elements, expected ${expectedTableElements}.`
    );
  }
  if (table.buffer.size < tableElements * dtypeBytes(table.dtype)) {
    throw new Error('Vision position embedding table buffer is smaller than its declared shape.');
  }

  const outputElements = gridHeight * gridWidth * hiddenSize;
  const outputBuffer = acquireBuffer(outputElements * Float32Array.BYTES_PER_ELEMENT, undefined, 'vision_position_embedding_output');
  const variant = selectRuleValue('visionPositionEmbedding', 'variant', { tableDtype: table.dtype });
  let succeeded = false;
  try {
    await unifiedKernelWrapper(
      'vision_position_embedding',
      null,
      variant,
      [table, outputBuffer],
      {
        grid_height: gridHeight,
        grid_width: gridWidth,
        position_embedding_size: positionEmbeddingSize,
        hidden_size: hiddenSize,
        output_elements: outputElements,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
      },
      Math.ceil(outputElements / WORKGROUP_SIZES.DEFAULT)
    );
    succeeded = true;
    return createTensor(
      outputBuffer,
      'f32',
      [gridHeight * gridWidth, hiddenSize],
      'vision_position_embedding_output'
    );
  } finally {
    if (!succeeded) {
      releaseBuffer(outputBuffer);
    }
  }
}
