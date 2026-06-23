import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { fetchManifestPayloadFromBaseUrl } from '../../src/client/runtime/model-source.js';

function sha256Hex(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-model-source-file-url-'));

try {
  const shardBytes = Buffer.from('tiny shard payload');
  const manifest = {
    version: 1,
    modelId: 'tiny-file-url-model',
    modelType: 'embedding',
    quantization: 'q4k',
    hashAlgorithm: 'sha256',
    inference: {},
    totalSize: shardBytes.byteLength,
    tensors: {
      'model.embed_tokens.weight': {
        role: 'embedding',
      },
    },
    shards: [
      {
        index: 0,
        filename: 'shard_00000.bin',
        size: shardBytes.byteLength,
        hash: sha256Hex(shardBytes),
        offset: 0,
      },
    ],
  };
  await fs.writeFile(path.join(tempRoot, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  await fs.writeFile(path.join(tempRoot, 'shard_00000.bin'), shardBytes);

  const payload = await fetchManifestPayloadFromBaseUrl(pathToFileURL(tempRoot).href);
  assert.equal(payload.manifest.modelId, manifest.modelId);
  assert.equal(payload.manifest.shards[0].filename, 'shard_00000.bin');
  assert.match(payload.text, /tiny-file-url-model/);
} finally {
  await fs.rm(tempRoot, { recursive: true, force: true });
}

console.log('runtime-model-source-file-url.test: ok');
