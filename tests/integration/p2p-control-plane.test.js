import assert from 'node:assert/strict';

const {
  downloadShard,
} = await import('../../src/distribution/shard-delivery.js');

const originalFetch = globalThis.fetch;

const httpData = new Uint8Array([1, 2, 3, 4]);
const p2pData = new Uint8Array([9, 10, 11, 12]);
const hashHttp = '9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a';
const hashP2P = 'e1e853684a206f162ee800a54b695c9cc1a8d1d554a47fcb13fe51229c17773f';
const manifestVersionSet = 'manifest:v1:sha256:control-plane';

try {
  globalThis.fetch = async () => new Response(httpData, { status: 200 });

  let tokenProviderCalls = 0;
  let p2pCalls = 0;
  const issuedTokenResult = await downloadShard('https://example.com/model', 0, {
    filename: 'shard_0.bin',
    size: 4,
    hash: hashP2P,
  }, {
    algorithm: 'sha256',
    expectedHash: hashP2P,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        security: {
          requireSessionToken: true,
        },
        controlPlane: {
          enabled: true,
          tokenProvider: async () => {
            tokenProviderCalls += 1;
            return {
              sessionToken: 'session-token-issued',
              tokenExpiresAtMs: Date.now() + 60000,
            };
          },
        },
        transport: async () => {
          p2pCalls += 1;
          return {
            data: p2pData,
            manifestVersionSet,
          };
        },
      },
    },
    writeToStore: false,
  });

  assert.equal(issuedTokenResult.source, 'p2p');
  assert.equal(tokenProviderCalls, 1);
  assert.equal(p2pCalls, 1);

  let deniedPolicyCalls = 0;
  p2pCalls = 0;
  const policyDeniedFallback = await downloadShard('https://example.com/model', 1, {
    filename: 'shard_1.bin',
    size: 4,
    hash: hashHttp,
  }, {
    algorithm: 'sha256',
    expectedHash: hashHttp,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p', 'http'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 0,
        controlPlane: {
          enabled: true,
          policyEvaluator: async () => {
            deniedPolicyCalls += 1;
            return {
              allow: false,
              reason: 'peer_not_allowed',
            };
          },
        },
        transport: async () => {
          p2pCalls += 1;
          return {
            data: p2pData,
            manifestVersionSet,
          };
        },
      },
    },
    writeToStore: false,
  });

  assert.equal(policyDeniedFallback.source, 'http');
  assert.equal(deniedPolicyCalls, 1);
  assert.equal(p2pCalls, 0);

  let refreshCalls = 0;
  p2pCalls = 0;
  const refreshedTokenResult = await downloadShard('https://example.com/model', 2, {
    filename: 'shard_2.bin',
    size: 4,
    hash: hashP2P,
  }, {
    algorithm: 'sha256',
    expectedHash: hashP2P,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['p2p'],
      p2p: {
        enabled: true,
        timeoutMs: 1000,
        maxRetries: 1,
        retryDelayMs: 0,
        security: {
          requireSessionToken: true,
        },
        controlPlane: {
          enabled: true,
          tokenProvider: async () => {
            refreshCalls += 1;
            return {
              sessionToken: `session-token-${refreshCalls}`,
              tokenExpiresAtMs: Date.now() + (refreshCalls === 1 ? 1 : 30000),
            };
          },
        },
        transport: async (context) => {
          p2pCalls += 1;
          if (context.attempt === 0) {
            const timeoutError = new Error('timed out');
            timeoutError.name = 'TimeoutError';
            throw timeoutError;
          }
          return {
            data: p2pData,
            manifestVersionSet,
          };
        },
      },
    },
    writeToStore: false,
  });

  assert.equal(refreshedTokenResult.source, 'p2p');
  assert.equal(p2pCalls, 2);
  assert.equal(refreshCalls, 2);

  let metricsHookEvent = null;
  await downloadShard('https://example.com/model', 3, {
    filename: 'shard_3.bin',
    size: 4,
    hash: hashHttp,
  }, {
    algorithm: 'sha256',
    expectedHash: hashHttp,
    expectedManifestVersionSet: manifestVersionSet,
    distributionConfig: {
      sourceOrder: ['http'],
    },
    onDeliveryMetrics: async (event) => {
      metricsHookEvent = event;
    },
    writeToStore: false,
  });

  assert.ok(metricsHookEvent);
  assert.equal(metricsHookEvent.schemaVersion, 1);
  assert.equal(metricsHookEvent.shardIndex, 3);
  assert.ok(metricsHookEvent.deliveryMetrics && typeof metricsHookEvent.deliveryMetrics === 'object');

  console.log('p2p-control-plane.test: ok');
} finally {
  globalThis.fetch = originalFetch;
}
