import assert from 'node:assert/strict';

const { persistDownloadedShardIfNeeded } = await import('../../src/storage/downloader.js');

{
  const writes = [];
  const persisted = await persistDownloadedShardIfNeeded(
    {
      source: 'p2p',
      wrote: false,
      buffer: new Uint8Array([1, 2, 3, 4]).buffer,
    },
    3,
    {
      writeShardFn: async (idx, buffer) => {
        writes.push({ idx, bytes: buffer.byteLength });
      },
    }
  );
  assert.equal(persisted, true);
  assert.deepEqual(writes, [{ idx: 3, bytes: 4 }]);
}

{
  const writes = [];
  const persisted = await persistDownloadedShardIfNeeded(
    {
      source: 'cache',
      wrote: false,
      buffer: new Uint8Array([1]).buffer,
    },
    1,
    {
      writeShardFn: async () => {
        writes.push('write');
      },
    }
  );
  assert.equal(persisted, false);
  assert.equal(writes.length, 0);
}

{
  await assert.rejects(
    () => persistDownloadedShardIfNeeded(
      {
        source: 'p2p',
        wrote: false,
        buffer: null,
      },
      7,
      {
        writeShardFn: async () => {},
      }
    ),
    /non-persisted data without buffer/
  );
}

console.log('downloader-persistence-policy.test: ok');
