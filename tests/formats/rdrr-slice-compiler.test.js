import assert from 'node:assert/strict';

import { compileTensorSlice } from '../../src/formats/rdrr/slice-compiler.js';

{
  const tensorMap = {
    dense: {
      shape: [4, 8],
      dtype: 'F16',
      shardIndex: 0,
      offset: 0,
      size: 64,
    },
  };
  const slice = compileTensorSlice({
    tensorMap,
    tensorId: 'dense',
    axis: 0,
    rangeStart: 1,
    rangeEnd: 3,
  });
  assert.deepEqual(slice.byteRanges, [{
    shardIndex: 0,
    byteStart: 16,
    byteEnd: 48,
  }]);
}

{
  const tensorMap = {
    q4k: {
      shape: [2, 256],
      dtype: 'Q4_K_M',
      shardIndex: 0,
      offset: 0,
      size: 288,
      storage: {
        packing: 'q4k',
        blockShape: [256],
      },
    },
  };
  const slice = compileTensorSlice({
    tensorMap,
    tensorId: 'q4k',
    axis: 0,
    rangeStart: 1,
    rangeEnd: 2,
  });
  assert.deepEqual(slice.byteRanges, [{
    shardIndex: 0,
    byteStart: 144,
    byteEnd: 288,
  }]);
}

{
  const tensorMap = {
    q4_0: {
      shape: [2, 64],
      dtype: 'INT4',
      shardIndex: 0,
      offset: 0,
      size: 72,
      storage: {
        packing: 'q4_0',
        blockShape: [32],
      },
    },
  };
  const slice = compileTensorSlice({
    tensorMap,
    tensorId: 'q4_0',
    axis: 1,
    rangeStart: 16,
    rangeEnd: 48,
  });
  assert.equal(slice.byteRanges.length, 2);
  assert.deepEqual(slice.byteRanges[0], {
    shardIndex: 0,
    byteStart: 0,
    byteEnd: 36,
  });
}

{
  const tensorMap = {
    gguf: {
      shape: [2, 64],
      dtype: 'INT4',
      shardIndex: 0,
      offset: 0,
      size: 128,
      storage: {
        packing: 'gguf-block-v2',
        blockShape: [16],
        blockBytes: 16,
      },
    },
  };
  const slice = compileTensorSlice({
    tensorMap,
    tensorId: 'gguf',
    axis: 1,
    rangeStart: 8,
    rangeEnd: 24,
  });
  assert.equal(slice.byteRanges.length, 2);
  assert.deepEqual(slice.byteRanges[0], {
    shardIndex: 0,
    byteStart: 0,
    byteEnd: 32,
  });
}

{
  const tensorMap = {
    primary: {
      shape: [2, 8],
      dtype: 'F16',
      shardIndex: 0,
      offset: 0,
      size: 32,
      storage: {
        packing: 'dense',
        companions: [{ role: 'scales', tensorId: 'scales' }],
      },
    },
  };
  assert.throws(
    () => compileTensorSlice({
      tensorMap,
      tensorId: 'primary',
      axis: 0,
      rangeStart: 0,
      rangeEnd: 1,
    }),
    /missing companion tensor/,
  );
}

console.log('rdrr-slice-compiler.test: ok');

