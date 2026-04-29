import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';

import { validateRemoteRegistry } from '../../tools/check-hf-registry.js';

const tempDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-hf-registry-check-'));

const artifactIdentity = {
  sourceCheckpointId: 'unit/toy-model',
  weightPackId: 'toy-model-wp-catalog-v1',
  manifestVariantId: 'toy-model-mv-exec-v1',
};
const catalogIdentity = {
  ...artifactIdentity,
  artifactCompleteness: 'complete',
  runtimePromotionState: 'manifest-owned',
  weightsRefAllowed: false,
};

const manifest = {
  modelId: 'toy-model',
  artifactIdentity,
  totalSize: 4,
  shards: [
    { index: 0, filename: 'shard_00000.bin', size: 4, offset: 0, hash: 'abcd' },
  ],
};
const sharedWeightsManifest = {
  modelId: 'toy-model-weight-pack',
  artifactIdentity: {
    sourceCheckpointId: 'unit/toy-model',
    weightPackId: 'toy-model-wp-catalog-v1',
    manifestVariantId: 'toy-model-weight-pack-mv-exec-v1',
    shardSetHash: 'sha256:toy-shards',
  },
  totalSize: 4,
  shards: [
    { index: 0, filename: 'shard_00000.bin', size: 4, offset: 0, hash: 'abcd' },
  ],
};
const sharedWeightsManifestText = JSON.stringify(sharedWeightsManifest);
const sharedWeightsManifestDigest = createHash('sha256').update(sharedWeightsManifestText).digest('hex');
const weightsRefManifest = {
  modelId: 'toy-model-variant',
  artifactIdentity: {
    sourceCheckpointId: 'unit/toy-model',
    weightPackId: 'toy-model-wp-catalog-v1',
    manifestVariantId: 'toy-model-variant-mv-exec-v1',
  },
  weightsRef: {
    weightPackId: 'toy-model-wp-catalog-v1',
    artifactRoot: '../toy-model-weight-pack',
    manifestDigest: `sha256:${sharedWeightsManifestDigest}`,
    shardSetHash: 'sha256:toy-shards',
  },
};

const registryPayload = {
  models: [
    {
      modelId: 'toy-model',
      ...catalogIdentity,
      modes: ['run'],
      baseUrl: '/models/toy-model',
    },
  ],
};

const localCatalog = {
  version: 1,
  lifecycleSchemaVersion: 1,
  updatedAt: '2026-03-06',
  models: [
    {
      modelId: 'toy-model',
      family: 'gemma3',
      ...catalogIdentity,
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'abc123',
        path: 'models/toy-model',
      },
      baseUrl: null,
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'active',
          tested: 'verified',
        },
      },
    },
    {
      modelId: 'failing-qwen',
      family: 'qwen3',
      hf: {
        repoId: 'Clocksmith/rdrr',
        revision: 'def456',
        path: 'models/failing-qwen',
      },
      lifecycle: {
        availability: {
          hf: true,
        },
        status: {
          runtime: 'active',
          tested: 'failing',
        },
      },
    },
  ],
};

const catalogFile = path.join(tempDir, 'catalog.json');
await writeFile(catalogFile, `${JSON.stringify(localCatalog, null, 2)}\n`, 'utf8');

const server = createServer((req, res) => {
  if (!req.url) {
    res.statusCode = 404;
    res.end('missing url');
    return;
  }
  const url = new URL(req.url, 'http://127.0.0.1');
  if (url.pathname === '/registry/catalog.json') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(registryPayload));
    return;
  }
  if (url.pathname === '/models/toy-model/manifest.json') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(manifest));
    return;
  }
  if (url.pathname === '/models/toy-model/shard_00000.bin') {
    res.setHeader('content-type', 'application/octet-stream');
    res.end('abcd');
    return;
  }
  if (url.pathname === '/models/toy-model-variant/manifest.json') {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(weightsRefManifest));
    return;
  }
  if (url.pathname === '/models/toy-model-weight-pack/manifest.json') {
    res.setHeader('content-type', 'application/json');
    res.end(sharedWeightsManifestText);
    return;
  }
  if (url.pathname === '/models/toy-model-weight-pack/shard_00000.bin') {
    res.setHeader('content-type', 'application/octet-stream');
    res.end('abcd');
    return;
  }
  res.statusCode = 404;
  res.end('not found');
});
server.keepAliveTimeout = 0;

await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const address = server.address();
const baseUrl = `http://127.0.0.1:${address.port}`;

try {
  const remoteValidation = await validateRemoteRegistry({
    models: [
      {
        modelId: 'toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
      },
    ],
  }, `${baseUrl}/registry/catalog.json`, {
    models: [
      {
        modelId: 'toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'verified',
          },
        },
      },
      {
        modelId: 'failing-qwen',
        baseUrl: `${baseUrl}/models/failing-qwen`,
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'failing',
          },
        },
      },
    ],
  });
  assert.deepEqual(remoteValidation.errors, []);

  // weights-ref variant must coexist with its primary lane in the same payload.
  const weightsRefValidation = await validateRemoteRegistry({
    models: [
      {
        modelId: 'toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
      },
      {
        modelId: 'toy-model-variant',
        sourceCheckpointId: 'unit/toy-model',
        weightPackId: 'toy-model-wp-catalog-v1',
        manifestVariantId: 'toy-model-variant-mv-exec-v1',
        artifactCompleteness: 'weights-ref',
        runtimePromotionState: 'manifest-owned',
        weightsRefAllowed: true,
        baseUrl: `${baseUrl}/models/toy-model-variant`,
      },
    ],
  }, `${baseUrl}/registry/catalog.json`, {
    models: [
      {
        modelId: 'toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'verified',
          },
        },
      },
      {
        modelId: 'toy-model-variant',
        sourceCheckpointId: 'unit/toy-model',
        weightPackId: 'toy-model-wp-catalog-v1',
        manifestVariantId: 'toy-model-variant-mv-exec-v1',
        artifactCompleteness: 'weights-ref',
        runtimePromotionState: 'manifest-owned',
        weightsRefAllowed: true,
        baseUrl: `${baseUrl}/models/toy-model-variant`,
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'verified',
          },
        },
      },
    ],
  });
  assert.deepEqual(weightsRefValidation.errors, []);

  const mismatchedManifestValidation = await validateRemoteRegistry({
    models: [
      {
        modelId: 'approved-toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
      },
    ],
  }, `${baseUrl}/registry/catalog.json`, {
    models: [
      {
        modelId: 'approved-toy-model',
        ...catalogIdentity,
        baseUrl: `${baseUrl}/models/toy-model`,
        lifecycle: {
          availability: {
            hf: true,
          },
          status: {
            runtime: 'active',
            tested: 'verified',
          },
        },
      },
    ],
  });
  assert.deepEqual(mismatchedManifestValidation.errors, [
    'approved-toy-model: demo-visible registry entry is not fetchable (approved-toy-model: manifest modelId "toy-model" does not match the approved support entry modelId)',
  ]);

  const result = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [
      'tools/check-hf-registry.js',
      '--catalog-file',
      catalogFile,
      '--remote-only',
      '--registry-url',
      `${baseUrl}/registry/catalog.json`,
    ], {
      cwd: process.cwd(),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    child.stdout.setEncoding('utf8');
    child.stderr.setEncoding('utf8');
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('error', reject);
    child.on('close', (code, signal) => {
      resolve({
        status: code,
        signal,
        stdout,
        stderr,
      });
    });
  });

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /remote demo-visible entries verified: 1/);
} finally {
  server.closeIdleConnections?.();
  server.closeAllConnections?.();
  await new Promise((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
  await rm(tempDir, { recursive: true, force: true });
}

console.log('check-hf-registry-cli.test: ok');
