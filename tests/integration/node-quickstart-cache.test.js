import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';

import { resolveNodeQuickstartCachedSource } from '../../src/client/runtime/node-quickstart-cache.js';

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function createServer(files) {
  const server = http.createServer((request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname.replace(/^\/+/, '');
    const bytes = files.get(pathname);
    if (!bytes) {
      response.writeHead(404);
      response.end('not found');
      return;
    }
    response.writeHead(200, {
      'content-length': bytes.byteLength,
    });
    response.end(bytes);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return {
    server,
    baseUrl: `http://127.0.0.1:${server.address().port}/`,
  };
}

const previousCacheDir = process.env.DOPPLER_QUICKSTART_CACHE_DIR;
const previousCache = process.env.DOPPLER_QUICKSTART_CACHE;
const tempRoot = mkdtempSync(path.join(os.tmpdir(), 'doppler-node-cache-'));
process.env.DOPPLER_QUICKSTART_CACHE_DIR = tempRoot;
delete process.env.DOPPLER_QUICKSTART_CACHE;

try {
  const shardBytes = Buffer.from('tiny shard payload');
  const tokenizerBytes = Buffer.from('{"version":"test"}');
  const manifest = {
    modelId: 'tiny-test-model',
    hashAlgorithm: 'sha256',
    shards: [
      {
        filename: 'shard_00000.bin',
        size: shardBytes.byteLength,
        hash: sha256Hex(shardBytes),
      },
    ],
    tokenizer: {
      type: 'bundled',
      file: 'tokenizer.json',
    },
  };
  const manifestText = JSON.stringify(manifest);
  const files = new Map([
    ['shard_00000.bin', shardBytes],
    ['tokenizer.json', tokenizerBytes],
  ]);
  const { server, baseUrl } = await createServer(files);
  const resolved = {
    modelId: manifest.modelId,
    baseUrl,
    trace: [
      {
        source: 'quickstart-registry',
        id: manifest.modelId,
        outcome: 'resolved',
      },
    ],
  };
  try {
    const imported = await resolveNodeQuickstartCachedSource(resolved, {
      text: manifestText,
      manifest,
    });
    assert.equal(imported.cache.state, 'imported');
    assert.ok(imported.baseUrl.startsWith('file://'));
    assert.ok(existsSync(path.join(tempRoot, manifest.modelId, 'manifest.json')));
    assert.equal(
      readFileSync(path.join(tempRoot, manifest.modelId, 'shard_00000.bin')).toString(),
      shardBytes.toString()
    );
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }

  const hit = await resolveNodeQuickstartCachedSource(resolved, {
    text: manifestText,
    manifest,
  });
  assert.equal(hit.cache.state, 'hit');
  assert.ok(hit.baseUrl.startsWith('file://'));
} finally {
  if (previousCacheDir == null) {
    delete process.env.DOPPLER_QUICKSTART_CACHE_DIR;
  } else {
    process.env.DOPPLER_QUICKSTART_CACHE_DIR = previousCacheDir;
  }
  if (previousCache == null) {
    delete process.env.DOPPLER_QUICKSTART_CACHE;
  } else {
    process.env.DOPPLER_QUICKSTART_CACHE = previousCache;
  }
  rmSync(tempRoot, { recursive: true, force: true });
}

console.log('node-quickstart-cache.test: ok');
