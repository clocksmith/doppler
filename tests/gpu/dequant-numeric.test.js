import assert from 'node:assert/strict';

import {
  QK_K,
  QK4_K_BLOCK_SIZE,
  quantizeQ4KBlock,
  quantizeToQ4KM,
  quantizeToQ4KMRowWise,
  dequantizeQ4KM,
  dequantizeQ4KMRowWise,
  calculateQuantizationError,
} from '../../src/converter/quantizer.js';

// === Flat dequant round-trip with known values ===
{
  const shape = [1, QK_K];
  const data = new Float32Array(QK_K);
  for (let i = 0; i < QK_K; i++) {
    data[i] = Math.sin(i * 0.05) * 2.0;
  }

  const { quantized, numBlocks } = quantizeToQ4KM(data, shape);
  assert.equal(numBlocks, 1);
  assert.equal(quantized.length, QK4_K_BLOCK_SIZE);

  const dequantized = dequantizeQ4KM(quantized, numBlocks, shape);
  assert.equal(dequantized.length, QK_K);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(err.maxError < 0.5, `single-block max error too high: ${err.maxError}`);
  assert.ok(err.snr > 15, `single-block SNR too low: ${err.snr}`);
}

// === Multi-block flat dequant ===
{
  const numElements = QK_K * 3;
  const shape = [1, numElements];
  const data = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    data[i] = ((i % 37) - 18) / 9;
  }

  const { quantized, numBlocks } = quantizeToQ4KM(data, shape);
  assert.equal(numBlocks, 3);
  assert.equal(quantized.length, 3 * QK4_K_BLOCK_SIZE);

  const dequantized = dequantizeQ4KM(quantized, numBlocks, shape);
  assert.equal(dequantized.length, numElements);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(Number.isFinite(err.mse));
  assert.ok(err.maxError < 0.5, `multi-block max error too high: ${err.maxError}`);
}

// === Rowwise dequant with K % 256 !== 0 (Gemma 270M/1B path) ===
// K=300 means blocksPerRow=2, padded stride=512 vs actual K=300.
// The rowwise path must strip the padded tail and produce exactly rows*K values.
{
  const rows = 4;
  const K = 300;
  const shape = [rows, K];
  const data = new Float32Array(rows * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i % 23) - 11) / 5.5;
  }

  const { quantized, numBlocks } = quantizeToQ4KMRowWise(data, shape);
  const blocksPerRow = Math.ceil(K / QK_K);
  assert.equal(blocksPerRow, 2);
  assert.equal(numBlocks, rows * blocksPerRow);

  const dequantized = dequantizeQ4KMRowWise(quantized, shape);
  assert.equal(dequantized.length, rows * K);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(err.maxError < 0.5, `rowwise K=300 max error too high: ${err.maxError}`);

  // Verify per-row reconstruction does not leak values from adjacent rows
  for (let row = 0; row < rows; row++) {
    const rowStart = row * K;
    const rowSlice = dequantized.slice(rowStart, rowStart + K);
    for (let i = 0; i < K; i++) {
      const diff = Math.abs(rowSlice[i] - data[rowStart + i]);
      assert.ok(diff < 0.5, `row ${row} element ${i}: diff=${diff}`);
    }
  }
}

// === Rowwise dequant with K=192 (another non-256-aligned dimension) ===
{
  const rows = 2;
  const K = 192;
  const shape = [rows, K];
  const data = new Float32Array(rows * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.cos(i * 0.1) * 3.0;
  }

  const { quantized } = quantizeToQ4KMRowWise(data, shape);
  const blocksPerRow = Math.ceil(K / QK_K);
  assert.equal(blocksPerRow, 1);

  const dequantized = dequantizeQ4KMRowWise(quantized, shape);
  assert.equal(dequantized.length, rows * K);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(err.maxError < 0.5, `rowwise K=192 max error too high: ${err.maxError}`);
}

// === Rowwise dequant: tail block padding must not corrupt values ===
// When K % 256 !== 0 the last block is only partially filled; the padded zeros
// must not pull non-zero tail values toward zero.
{
  const rows = 1;
  const K = 300;
  const shape = [rows, K];
  const data = new Float32Array(K);
  // First 256 elements: moderate range
  for (let i = 0; i < 256; i++) {
    data[i] = ((i % 17) - 8) / 4;
  }
  // Last 44 elements: strongly negative to detect zero-padding corruption
  for (let i = 256; i < K; i++) {
    data[i] = -10 + (i - 256) * 0.01;
  }

  const { quantized } = quantizeToQ4KMRowWise(data, shape);
  const dequantized = dequantizeQ4KMRowWise(quantized, shape);

  assert.ok(
    Math.max(...dequantized.slice(256)) < -9.5,
    'rowwise tail block should not be pulled toward zero by padded zeros'
  );
}

// === Rowwise dequant: K exactly 256 (block-aligned edge case) ===
{
  const rows = 3;
  const K = QK_K;
  const shape = [rows, K];
  const data = new Float32Array(rows * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i % 41) - 20) / 10;
  }

  const { quantized, numBlocks } = quantizeToQ4KMRowWise(data, shape);
  assert.equal(numBlocks, rows);

  const dequantized = dequantizeQ4KMRowWise(quantized, shape);
  assert.equal(dequantized.length, rows * K);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(err.maxError < 0.5, `rowwise K=256 max error too high: ${err.maxError}`);
}

// === Rowwise vs flat dequant shape comparison ===
// For K that is a multiple of 256, flat and rowwise should produce identical results.
{
  const rows = 2;
  const K = QK_K * 2;
  const shape = [rows, K];
  const data = new Float32Array(rows * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i % 19) - 9) / 4.5;
  }

  const flat = quantizeToQ4KM(data, shape);
  const row = quantizeToQ4KMRowWise(data, shape);

  const flatDeq = dequantizeQ4KM(flat.quantized, flat.numBlocks, shape);
  const rowDeq = dequantizeQ4KMRowWise(row.quantized, shape);

  assert.equal(flatDeq.length, rowDeq.length);
  for (let i = 0; i < flatDeq.length; i++) {
    assert.equal(flatDeq[i], rowDeq[i], `flat vs rowwise mismatch at index ${i}`);
  }
}

// === Quantize block byte layout invariant ===
{
  const blockInput = new Float32Array(QK_K);
  for (let i = 0; i < QK_K; i++) {
    blockInput[i] = Math.sin(i / 16) * 5;
  }
  const block = quantizeQ4KBlock(blockInput, 0);
  assert.equal(block.byteLength, QK4_K_BLOCK_SIZE);
  assert.equal(QK4_K_BLOCK_SIZE, 144);
}

// === Rowwise dequant stride contract ===
// When K % 256 !== 0, the quantized representation uses a padded stride
// (blocksPerRow * 256) but dequantized output must have exactly rows * K elements.
{
  const rows = 5;
  const K = 384;
  const shape = [rows, K];
  const data = new Float32Array(rows * K);
  for (let i = 0; i < data.length; i++) {
    data[i] = ((i % 31) - 15) / 7.5;
  }

  const { quantized } = quantizeToQ4KMRowWise(data, shape);
  const blocksPerRow = Math.ceil(K / QK_K);
  assert.equal(blocksPerRow, 2);

  // Quantized byte size = rows * blocksPerRow * blockBytes
  assert.equal(quantized.length, rows * blocksPerRow * QK4_K_BLOCK_SIZE);

  const dequantized = dequantizeQ4KMRowWise(quantized, shape);
  // Output must be exactly rows * K, not rows * blocksPerRow * 256
  assert.equal(dequantized.length, rows * K);

  const err = calculateQuantizationError(data, dequantized);
  assert.ok(err.maxError < 0.5, `rowwise K=384 max error too high: ${err.maxError}`);
}

console.log('dequant-numeric.test: ok');
