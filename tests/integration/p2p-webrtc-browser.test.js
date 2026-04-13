import assert from 'node:assert/strict';

const {
  createBrowserWebRTCDataPlaneTransport,
  isBrowserWebRTCAvailable,
} = await import('../../src/experimental/distribution/p2p-webrtc-browser.js');

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer))
    .map((entry) => entry.toString(16).padStart(2, '0'))
    .join('');
}

function createMockChannel(onSend) {
  const listeners = {
    message: new Set(),
    error: new Set(),
    close: new Set(),
  };

  return {
    readyState: 'open',
    addEventListener(type, handler) {
      listeners[type]?.add(handler);
    },
    removeEventListener(type, handler) {
      listeners[type]?.delete(handler);
    },
    send(payload) {
      onSend(payload, {
        emit(type, data) {
          for (const handler of listeners[type] ?? []) {
            handler({ type, data });
          }
        },
      });
    },
  };
}

const originalRTCPeerConnection = globalThis.RTCPeerConnection;

try {
  assert.equal(createBrowserWebRTCDataPlaneTransport({ enabled: false }), null);

  globalThis.RTCPeerConnection = undefined;
  assert.equal(isBrowserWebRTCAvailable(), false);
  assert.throws(
    () => createBrowserWebRTCDataPlaneTransport({
      enabled: true,
      getDataChannel: () => null,
    }),
    /RTCPeerConnection is unavailable/
  );

  globalThis.RTCPeerConnection = function MockRTCPeerConnection() {};

  let selectedPeer = null;
  const transport = createBrowserWebRTCDataPlaneTransport({
    enabled: true,
    requestTimeoutMs: 1000,
    maxPayloadBytes: 1024,
    selectPeer: async () => ({ peerId: 'peer-a' }),
    getDataChannel: async ({ peerId }) => {
      selectedPeer = peerId;
      return createMockChannel((payload, channelCtx) => {
        const request = JSON.parse(payload);
        const payloadBase64 = Buffer.from(new Uint8Array([5, 6, 7, 8])).toString('base64');
        queueMicrotask(() => {
          channelCtx.emit('message', JSON.stringify({
            schemaVersion: 1,
            type: 'doppler_p2p_shard_response',
            requestId: request.requestId,
            payloadBase64,
            manifestVersionSet: 'manifest:v1:peer',
            rangeStart: 0,
            totalSize: 4,
          }));
        });
      });
    },
  });

  const result = await transport({
    shardIndex: 3,
    shardInfo: { filename: 'shard_3.bin' },
    source: 'p2p',
    timeoutMs: 1500,
    contractVersion: 1,
    attempt: 0,
    maxRetries: 1,
    expectedHash: null,
    expectedSize: 4,
    expectedManifestVersionSet: 'manifest:v1:peer',
  });

  assert.equal(selectedPeer, 'peer-a');
  assert.equal(result.manifestVersionSet, 'manifest:v1:peer');
  assert.equal(result.totalSize, 4);
  assert.equal(toHex(result.data), '05060708');

  const missTransport = createBrowserWebRTCDataPlaneTransport({
    enabled: true,
    peerId: 'peer-b',
    getDataChannel: async () => createMockChannel((payload, channelCtx) => {
      const request = JSON.parse(payload);
      queueMicrotask(() => {
        channelCtx.emit('message', JSON.stringify({
          schemaVersion: 1,
          type: 'doppler_p2p_shard_response',
          requestId: request.requestId,
          miss: true,
        }));
      });
    }),
  });

  const missResult = await missTransport({
    shardIndex: 9,
    shardInfo: { filename: 'shard_9.bin' },
    source: 'p2p',
    timeoutMs: 1500,
    contractVersion: 1,
    attempt: 0,
    maxRetries: 0,
  });
  assert.equal(missResult.miss, true);

  await assert.rejects(
    () => transport({
      shardIndex: 4,
      shardInfo: { filename: 'shard_4.bin' },
      source: 'p2p',
      timeoutMs: 1500,
      contractVersion: 99,
      attempt: 0,
      maxRetries: 0,
    }),
    /Unsupported p2p\.webrtc contractVersion "99"/
  );

  console.log('p2p-webrtc-browser.test: ok');
} finally {
  globalThis.RTCPeerConnection = originalRTCPeerConnection;
}
