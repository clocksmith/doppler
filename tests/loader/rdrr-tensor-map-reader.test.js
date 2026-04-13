import assert from 'node:assert/strict';

const { buildTensorLocations } = await import('../../src/loader/shard-resolver.js');
const { assembleShardData } = await import('../../src/loader/tensors/tensor-reader.js');

{
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({
      ok: true,
      async text() {
        return JSON.stringify({
          'layers.0.weight': {
            shardIndex: 3,
            offset: 16,
            size: 8,
            shape: [2, 2],
            dtype: 'F16',
            role: 'matmul',
            spans: [
              { shard: 3, offset: 16, size: 4 },
              { shardIndex: 4, offset: 0, size: 4 },
            ],
            sourceTransform: {
              kind: 'litert_rowwise_dequant',
              scheme: 'per_row_affine',
              sourceDtype: 'INT8',
              targetDtype: 'F16',
              storageEncoding: 'signed',
              scaleSource: {
                shard: 5,
                offset: 32,
                size: 8,
              },
            },
          },
        });
      },
    });

    const locations = await buildTensorLocations(
      { tensorsFile: 'tensors.json' },
      {
        hasCustomLoader: true,
        tensorsJsonUrl: 'https://example.com/tensors.json',
      }
    );

    const location = locations.get('layers.0.weight');
    assert.ok(location);
    assert.equal(location.shardIndex, 3);
  assert.deepEqual(
    location.spans,
    [
      { shardIndex: 3, offset: 16, size: 4 },
      { shardIndex: 4, offset: 0, size: 4 },
    ]
  );
  assert.deepEqual(location.sourceTransform, {
    kind: 'litert_rowwise_dequant',
    scheme: 'per_row_affine',
    sourceDtype: 'INT8',
    targetDtype: 'F16',
    storageEncoding: 'signed',
    scaleSource: {
      shard: 5,
      offset: 32,
      size: 8,
    },
  });
  } finally {
    globalThis.fetch = originalFetch;
  }
}

{
  const locations = await buildTensorLocations({
    tensors: {
      'layers.1.weight': {
        shard: 2,
        offset: 8,
        size: 8,
        shape: [2, 2],
        dtype: 'F16',
        role: 'matmul',
        spans: [
          { shard: 2, offset: 8, size: 4 },
          { shardIndex: 3, offset: 0, size: 4 },
        ],
        sourceTransform: {
          kind: 'litert_rowwise_dequant',
          scheme: 'per_row_affine',
          sourceDtype: 'INT8',
          targetDtype: 'F16',
          storageEncoding: 'signed',
          scaleSource: {
            shard: 4,
            offset: 24,
            size: 8,
          },
        },
      },
    },
  });

  const location = locations.get('layers.1.weight');
  assert.ok(location);
  assert.equal(location.shardIndex, 2);
  assert.deepEqual(
    location.spans,
    [
      { shardIndex: 2, offset: 8, size: 4 },
      { shardIndex: 3, offset: 0, size: 4 },
    ]
  );
  assert.deepEqual(location.sourceTransform, {
    kind: 'litert_rowwise_dequant',
    scheme: 'per_row_affine',
    sourceDtype: 'INT8',
    targetDtype: 'F16',
    storageEncoding: 'signed',
    scaleSource: {
      shard: 4,
      offset: 24,
      size: 8,
    },
  });
}

{
  const data = await assembleShardData(
    {
      shardIndex: 0,
      offset: 1,
      size: 3,
      spans: [],
    },
    'single-shard-fallback',
    async () => new Uint8Array([0, 1, 2, 3, 4]).buffer
  );

  assert.deepEqual(Array.from(data), [1, 2, 3]);
}

{
  const data = await assembleShardData(
    {
      shard: 0,
      offset: 2,
      size: 2,
    },
    'legacy-shard-alias',
    async () => new Uint8Array([0, 1, 2, 3, 4]).buffer
  );

  assert.deepEqual(Array.from(data), [2, 3]);
}

{
  await assert.rejects(
    () => assembleShardData(
      {
        shardIndex: 0,
        offset: 0,
        size: 8,
        spans: [
          { shardIndex: 0, offset: 0, size: 4 },
          { shardIndex: 1, offset: 0, size: 3 },
        ],
      },
      'mismatched-spans',
      async (shardIndex) => new Uint8Array(
        shardIndex === 0 ? [1, 2, 3, 4] : [5, 6, 7]
      ).buffer
    ),
    /expected 8/
  );
}

console.log('rdrr-tensor-map-reader.test: ok');
