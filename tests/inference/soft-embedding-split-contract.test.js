import assert from 'node:assert/strict';

const { setDevice } = await import('../../src/gpu/device.js');
const { createSplitWeightBuffer } = await import('../../src/gpu/weight-buffer.js');
const { runSoftEmbeddingSplitF16 } = await import('../../src/gpu/kernels/soft-embedding.js');

class FakeBuffer {
  constructor(size = 16) {
    this.size = size;
    this.usage = 0x80;
    this.destroyed = false;
  }

  destroy() {
    this.destroyed = true;
  }
}

const ORIGINAL_GPU_BUFFER = globalThis.GPUBuffer;
globalThis.GPUBuffer = FakeBuffer;

function createSoftmaxTensor(rows, cols) {
  return {
    buffer: new FakeBuffer(rows * cols * Float32Array.BYTES_PER_ELEMENT),
    dtype: 'f32',
    shape: [rows, cols],
    label: 'softmax_rows',
  };
}

function createSplit(sections, dtype = 'f16', shape = [4, 2]) {
  return createSplitWeightBuffer(
    sections.map((section) => ({
      buffer: new FakeBuffer(section.rowCount * shape[1] * 2),
      rowStart: section.rowStart,
      rowCount: section.rowCount,
    })),
    dtype,
    'row',
    shape,
    'embed_tokens'
  );
}

try {
  setDevice(null);

  await assert.rejects(
    () => runSoftEmbeddingSplitF16(
      createSoftmaxTensor(1, 4),
      createSplit([{ rowStart: 1, rowCount: 3 }]),
      1,
      2,
      4
    ),
    /not contiguous from row 0/
  );

  await assert.rejects(
    () => runSoftEmbeddingSplitF16(
      createSoftmaxTensor(1, 4),
      createSplit([{ rowStart: 0, rowCount: 2 }]),
      1,
      2,
      4
    ),
    /exposes 2 rows but vocabSize=4/
  );

  await assert.rejects(
    () => runSoftEmbeddingSplitF16(
      createSoftmaxTensor(1, 4),
      createSplit([{ rowStart: 0, rowCount: 4 }], 'f32'),
      1,
      2,
      4
    ),
    /row-major f16 embeddings only/
  );

  await assert.rejects(
    () => runSoftEmbeddingSplitF16(
      createSoftmaxTensor(1, 4),
      createSplit([{ rowStart: 0, rowCount: 2 }, { rowStart: 2, rowCount: 2 }]),
      1,
      2,
      4
    ),
    /GPU device not available/
  );
} finally {
  setDevice(null);
  if (ORIGINAL_GPU_BUFFER === undefined) {
    delete globalThis.GPUBuffer;
  } else {
    globalThis.GPUBuffer = ORIGINAL_GPU_BUFFER;
  }
}

console.log('soft-embedding-split-contract.test: ok');
