import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createNodeFileAccess as storageAccess } from '../../src/storage/node-file-access.js';
import { createNodeFileAccess as sourceAccess } from '../../src/tooling/source-runtime/node-loader.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-range-owner-'));
try {
  for (const createAccess of [storageAccess, sourceAccess]) {
    const file = path.join(root, 'source.bin');
    const source = Uint8Array.from({ length: 1024 * 1024 }, (_, i) => i % 251);
    await fs.writeFile(file, source);
    const reader = createAccess();
    try {
      let allocated;
      const original = Buffer.allocUnsafe;
      Buffer.allocUnsafe = size => {
        const buffer = original(size);
        if (size === source.length) allocated = buffer;
        return buffer;
      };
      let full;
      try { full = await reader.readRange(file, 0, source.length); }
      finally { Buffer.allocUnsafe = original; }
      assert.equal(allocated.byteOffset, 0);
      assert.equal(full, allocated.buffer, 'exact owned large range must not be copied');
      assert.deepEqual(new Uint8Array(full), source);
      const small = await reader.readRange(file, 1, 3);
      assert.equal(small.byteLength, 3, 'never expose a pooled slab');
      assert.deepEqual([...new Uint8Array(small)], [1, 2, 3]);
      new Uint8Array(full).fill(0);
      assert.deepEqual([...new Uint8Array(await reader.readRange(file, 1, 3))], [1, 2, 3], 'returned ownership must not alias later reads or the file');
      // The reader has cached original size: truncation exercises actual EOF,
      // not just the declared range clamp. Never expose allocUnsafe tail bytes.
      await fs.truncate(file, 7);
      assert.equal((await reader.readRange(file, 0, source.length)).byteLength, 7);
    } finally { await reader.close(); }
  }
  console.log('node file ranges: exact ownership, compact slabs, independent reads, EOF and handle cleanup passed');
} finally { await fs.rm(root, { recursive: true, force: true }); }
