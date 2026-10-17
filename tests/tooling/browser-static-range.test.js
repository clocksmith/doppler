import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createStaticFileServer } from '../../src/tooling/node-browser/static-server.js';

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-range-'));
await fs.writeFile(path.join(root, 'shard.bin'), Buffer.from('0123456789'));
await fs.writeFile(path.join(root, 'empty.bin'), '');
const server = await createStaticFileServer({ rootDir: root, host: '127.0.0.1', port: 0 });
try {
  for (const [range, expected, contentRange] of [
    ['bytes=2-5', '2345', 'bytes 2-5/10'],
    ['bytes=7-', '789', 'bytes 7-9/10'],
    ['bytes=-3', '789', 'bytes 7-9/10'],
    ['bytes=8-99', '89', 'bytes 8-9/10'],
    ['bytes=-20', '0123456789', 'bytes 0-9/10'],
  ]) {
    const response = await fetch(server.baseUrl + '/shard.bin', { headers: { Range: range } });
    assert.equal(response.status, 206);
    assert.equal(response.headers.get('content-range'), contentRange);
    assert.equal(Number(response.headers.get('content-length')), expected.length);
    assert.equal(await response.text(), expected);
  }
  for (const range of ['bytes=10-', 'bytes=5-2', 'bytes=-0', 'bytes=-', 'bytes=9007199254740992-']) {
    const response = await fetch(server.baseUrl + '/shard.bin', { headers: { Range: range } });
    assert.equal(response.status, 416); assert.equal(response.headers.get('content-range'), 'bytes */10');
    await response.arrayBuffer();
  }
  const empty = await fetch(server.baseUrl + '/empty.bin', { headers: { Range: 'bytes=0-0' } });
  assert.equal(empty.status, 416); await empty.arrayBuffer();
  const full = await fetch(server.baseUrl + '/shard.bin');
  assert.equal(full.status, 200); assert.equal(await full.text(), '0123456789');
  const head = await fetch(server.baseUrl + '/shard.bin', { method: 'HEAD', headers: { Range: 'bytes=2-5' } });
  assert.equal(head.status, 200); assert.equal(head.headers.get('content-length'), '10');
  assert.equal(await head.text(), '');
} finally { await server.close(); await fs.rm(root, { recursive: true, force: true }); }
console.log('browser-static-range.test: ok (actual HTTP responses and bytes)');
