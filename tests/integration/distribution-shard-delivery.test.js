import assert from 'node:assert/strict';

const {
  downloadShard,
  resolveShardDeliveryPlan,
  getInFlightShardDeliveryCount,
} = await import('../../src/distribution/shard-delivery.js');

const sha256Data = {
  p2p: 'e1e853684a206f162ee800a54b695c9cc1a8d1d554a47fcb13fe51229c17773f',
  http: '9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a',
};

const originalFetch = globalThis.fetch;

const httpData = new Uint8Array([1, 2, 3, 4]);
const p2pData = new Uint8Array([9, 10, 11, 12]);
const manifestVersionSet = 'manifest:v1:sha256:abc123';

function createResponse(payload) {
  return new Response(payload, {
    status: 200,
    headers: {
      'content-encoding': 'gzip',
    },
  });
}

function toHex(value) {
  return Array.from(value)
    .map((byte) => byte.toString(16).padStart(2, '0'))
    .join('');
}

try {
  const deterministicPlanA = resolveShardDeliveryPlan({
    sourceOrder: ['cache', 'p2p', 'http'],
    enableSourceCache: false,
    p2pEnabled: true,
    p2pTransportAvailable: false,
    httpEnabled: true,
  });
  const deterministicPlanB = resolveShardDeliveryPlan({
    sourceOrder: ['cache', 'p2p', 'http'],
    enableSourceCache: false,
    p2pEnabled: true,
    p2pTransportAvailable: false,
    httpEnabled: true,
  });
  assert.deepEqual(deterministicPlanA, deterministicPlanB);

  const shardInfo = {
    filename: 'shard_0.bin',
    size: 4,
    hash: sha256Data.http,
  };

  globalThis.fetch = async () => createResponse(httpData);

  const directHttp = await downloadShard('https://example.com/models', 0, shardInfo, {
    algorithm: 'sha256',
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['http'],
      p2p: {
        enabled: false,
        timeoutMs: 3000,
        maxRetries: 0,
        transport: null,
      },
      requiredContentEncoding: 'gzip',
    },
    writeToStore: false,
  });

  assert.equal(directHttp.source, 'http');
  assert.equal(toHex(new Uint8Array(directHttp.buffer)), '01020304');

  let p2pCalls = 0;
  let fetchCalls = 0;
  globalThis.fetch = async () => {
    fetchCalls += 1;
    return createResponse(httpData);
  };

  const p2pFirst = await downloadShard('https://example.com/models', 1, shardInfo, {
    algorithm: 'sha256',
    expectedHash: sha256Data.p2p,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        transport: async () => {
          p2pCalls += 1;
          return p2pData;
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });
  assert.equal(p2pFirst.source, 'p2p');
  assert.equal(p2pCalls, 1);
  assert.equal(fetchCalls, 0);
  assert.equal(toHex(new Uint8Array(p2pFirst.buffer)), '090a0b0c');

  p2pCalls = 0;
  fetchCalls = 0;
  const p2pEnvelope = await downloadShard('https://example.com/models', 11, shardInfo, {
    algorithm: 'sha256',
    expectedHash: sha256Data.p2p,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        transport: async () => {
          p2pCalls += 1;
          return {
            data: p2pData,
            manifestVersionSet,
          };
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });
  assert.equal(p2pEnvelope.source, 'p2p');
  assert.equal(p2pCalls, 1);
  assert.equal(fetchCalls, 0);
  assert.equal(toHex(new Uint8Array(p2pEnvelope.buffer)), '090a0b0c');

  p2pCalls = 0;
  fetchCalls = 0;
  const p2pTimeoutRetrySuccess = await downloadShard('https://example.com/models', 14, shardInfo, {
    algorithm: 'sha256',
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 2,
        retryDelayMs: 0,
        transport: async () => {
          p2pCalls += 1;
          if (p2pCalls < 3) {
            const timeoutError = new Error('timed out');
            timeoutError.name = 'TimeoutError';
            throw timeoutError;
          }
          return {
            data: httpData,
            manifestVersionSet,
          };
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });
  assert.equal(p2pTimeoutRetrySuccess.source, 'p2p');
  assert.equal(p2pCalls, 3);
  assert.equal(fetchCalls, 0);
  assert.equal(toHex(new Uint8Array(p2pTimeoutRetrySuccess.buffer)), '01020304');

  p2pCalls = 0;
  fetchCalls = 0;
  const p2pVersionMismatchFallsBackToHttp = await downloadShard('https://example.com/models', 12, shardInfo, {
    algorithm: 'sha256',
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        transport: async () => {
          p2pCalls += 1;
          return {
            data: p2pData,
            manifestVersionSet: 'manifest:v0:sha256:deadbeef',
          };
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });

  assert.equal(p2pVersionMismatchFallsBackToHttp.source, 'http');
  assert.equal(p2pCalls, 1);
  assert.equal(fetchCalls, 1);
  assert.equal(toHex(new Uint8Array(p2pVersionMismatchFallsBackToHttp.buffer)), '01020304');

  await assert.rejects(
    () => downloadShard('https://example.com/models', 13, shardInfo, {
      algorithm: 'sha256',
      expectedManifestVersionSet: manifestVersionSet,
      distributionConfig: {
        sourceOrder: ['p2p'],
        antiRollback: {
          enabled: true,
          requireExpectedHash: true,
          requireExpectedSize: false,
          requireManifestVersionSet: true,
        },
        p2p: {
          enabled: true,
          timeoutMs: 1000,
          maxRetries: 0,
          transport: async () => ({
            data: p2pData,
            manifestVersionSet: 'manifest:v0:sha256:deadbeef',
          }),
        },
      },
      writeToStore: false,
    }),
    /manifestVersionSet mismatch/
  );

  p2pCalls = 0;
  fetchCalls = 0;
  const p2pFailsThenHttp = await downloadShard('https://example.com/models', 2, shardInfo, {
    algorithm: 'sha256',
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        transport: async () => {
          p2pCalls += 1;
          if (p2pCalls === 1) {
            throw new Error('peer miss');
          }
          return p2pData;
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });

  assert.equal(p2pFailsThenHttp.source, 'http');
  assert.equal(p2pCalls, 1);
  assert.equal(fetchCalls, 1);
  assert.equal(toHex(new Uint8Array(p2pFailsThenHttp.buffer)), '01020304');

  p2pCalls = 0;
  fetchCalls = 0;
  const p2pMismatchFallsBackToHttp = await downloadShard('https://example.com/models', 3, shardInfo, {
    algorithm: 'sha256',
    expectedHash: sha256Data.http,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        transport: async () => {
          p2pCalls += 1;
          return p2pData;
        },
      },
      requiredContentEncoding: null,
    },
    writeToStore: false,
  });

  assert.equal(p2pMismatchFallsBackToHttp.source, 'http');
  assert.equal(p2pCalls, 1);
  assert.equal(fetchCalls, 1);
  assert.equal(toHex(new Uint8Array(p2pMismatchFallsBackToHttp.buffer)), '01020304');

  let delayedFetchCalls = 0;
  globalThis.fetch = async () => {
    delayedFetchCalls += 1;
    await new Promise((resolve) => setTimeout(resolve, 20));
    return createResponse(httpData);
  };
  const deduped = await Promise.all([
    downloadShard('https://example.com/models', 4, shardInfo, {
      algorithm: 'sha256',
      expectedManifestVersionSet: manifestVersionSet,
      distributionConfig: {
        sourceOrder: ['http'],
        requiredContentEncoding: null,
      },
      writeToStore: false,
    }),
    downloadShard('https://example.com/models', 4, shardInfo, {
      algorithm: 'sha256',
      expectedManifestVersionSet: manifestVersionSet,
      distributionConfig: {
        sourceOrder: ['http'],
        requiredContentEncoding: null,
      },
      writeToStore: false,
    }),
  ]);
  assert.equal(delayedFetchCalls, 1);
  assert.equal(deduped[0].hash, deduped[1].hash);
  assert.equal(getInFlightShardDeliveryCount(), 0);

  const withTrace = await downloadShard('https://example.com/models', 5, shardInfo, {
    algorithm: 'sha256',
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['cache', 'p2p', 'http'],
      sourceDecision: {
        deterministic: true,
        trace: {
          enabled: true,
          includeSkippedSources: true,
        },
      },
      p2p: {
        enabled: false,
        timeoutMs: 1000,
        maxRetries: 0,
        contractVersion: 1,
      },
      requiredContentEncoding: null,
    },
    enableSourceCache: false,
    writeToStore: false,
  });
  assert.equal(withTrace.source, 'http');
  assert.equal(withTrace.decisionTrace?.schemaVersion, 1);
  assert.equal(withTrace.decisionTrace?.deterministic, true);
  assert.equal(withTrace.decisionTrace?.expectedManifestVersionSet, manifestVersionSet);
  assert.equal(withTrace.decisionTrace?.attempts?.some((attempt) => attempt.source === 'cache' && attempt.status === 'skipped'), true);
  assert.equal(withTrace.decisionTrace?.attempts?.some((attempt) => attempt.source === 'http' && attempt.status === 'success'), true);
  assert.equal(withTrace.decisionTrace?.attempts?.some((attempt) => attempt.source === 'http' && attempt.manifestVersionSet === manifestVersionSet), true);

  await assert.rejects(
    () => downloadShard('https://example.com/models', 6, shardInfo, {
      algorithm: 'sha256',
      expectedManifestVersionSet: manifestVersionSet,
      distributionConfig: {
        sourceOrder: ['p2p', 'http'],
        p2p: {
          enabled: true,
          timeoutMs: 1000,
          maxRetries: 0,
          contractVersion: 99,
          transport: async () => p2pData,
        },
      },
      writeToStore: false,
    }),
    /contractVersion/
  );

  await assert.rejects(
    () => downloadShard('https://example.com/models', 7, { filename: 'shard_7.bin', size: 4 }, {
      algorithm: 'sha256',
      expectedManifestVersionSet: manifestVersionSet,
      distributionConfig: {
        sourceOrder: ['http'],
        antiRollback: {
          enabled: true,
          requireExpectedHash: true,
          requireExpectedSize: true,
        },
      },
      writeToStore: false,
    }),
    /Missing expected hash/
  );

  await assert.rejects(
    () => downloadShard('https://example.com/models', 8, shardInfo, {
      algorithm: 'sha256',
      distributionConfig: {
        sourceOrder: ['http'],
        antiRollback: {
          enabled: true,
          requireExpectedHash: true,
          requireExpectedSize: false,
          requireManifestVersionSet: true,
        },
      },
      writeToStore: false,
    }),
    /Missing expected manifestVersionSet/
  );

  assert.equal(directHttp.hash.length, 64);
} finally {
  globalThis.fetch = originalFetch;
}

console.log('distribution-shard-delivery.test: ok');
