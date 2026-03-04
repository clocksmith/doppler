import {
  P2P_TRANSPORT_ERROR_CODES,
  createP2PTransportError,
  normalizeP2PTransportError,
} from './p2p-transport-contract.js';

export const P2P_WEBRTC_DATA_PLANE_CONTRACT_VERSION = 1;
const P2P_WEBRTC_MESSAGE_SCHEMA_VERSION = 1;
const DEFAULT_REQUEST_TIMEOUT_MS = 2500;
const DEFAULT_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024;

function asNonNegativeInteger(value, fallback) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    return fallback;
  }
  return parsed;
}

function asOptionalString(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const normalized = String(value).trim();
  return normalized || null;
}

function toArrayBuffer(value, label) {
  if (value instanceof ArrayBuffer) {
    return value;
  }
  if (value instanceof Uint8Array) {
    if (value.byteOffset === 0 && value.byteLength === value.buffer.byteLength) {
      return value.buffer;
    }
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
    `${label} must be ArrayBuffer or Uint8Array.`,
    { label }
  );
}

function decodeBase64Bytes(value, label) {
  if (typeof value !== 'string' || !value.trim()) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a non-empty base64 string.`,
      { label }
    );
  }

  const source = value.trim();
  if (typeof globalThis.atob === 'function') {
    const binary = globalThis.atob(source);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return toArrayBuffer(bytes, label);
  }

  if (typeof globalThis.Buffer === 'function') {
    const bytes = globalThis.Buffer.from(source, 'base64');
    return toArrayBuffer(bytes, label);
  }

  throw createP2PTransportError(
    P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
    `${label} base64 decoding is unavailable in this runtime.`,
    { label }
  );
}

function createRequestId(prefix = 'p2p-webrtc') {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {
    return `${prefix}-${globalThis.crypto.randomUUID()}`;
  }
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function safeParseJsonMessage(value) {
  if (typeof value !== 'string') {
    return null;
  }
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function normalizeOptionalEnvelopeInteger(value, label) {
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `${label} must be a non-negative integer when provided.`,
      { label }
    );
  }
  return parsed;
}

function normalizeResponsePayload(message, requestId, maxPayloadBytes) {
  const parsed = typeof message === 'string'
    ? safeParseJsonMessage(message)
    : (message && typeof message === 'object' ? message : null);

  if (!parsed || typeof parsed !== 'object') {
    return null;
  }

  const responseRequestId = asOptionalString(parsed.requestId);
  if (responseRequestId && responseRequestId !== requestId) {
    return null;
  }

  if (parsed.type !== 'doppler_p2p_shard_response') {
    return null;
  }

  if (parsed.schemaVersion !== P2P_WEBRTC_MESSAGE_SCHEMA_VERSION) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `Unexpected WebRTC response schemaVersion "${parsed.schemaVersion}".`,
      {
        expectedSchemaVersion: P2P_WEBRTC_MESSAGE_SCHEMA_VERSION,
        actualSchemaVersion: parsed.schemaVersion,
      }
    );
  }

  if (parsed.miss === true || parsed.notFound === true) {
    return {
      miss: true,
      notFound: true,
    };
  }

  if (parsed.error != null) {
    throw normalizeP2PTransportError(parsed.error, {
      requestId,
      channel: 'webrtc',
    });
  }

  const payloadBase64 = asOptionalString(parsed.payloadBase64);
  const payloadData = payloadBase64
    ? decodeBase64Bytes(payloadBase64, 'p2p.webrtc.payloadBase64')
    : toArrayBuffer(parsed.data ?? parsed.payload, 'p2p.webrtc.data');
  if (payloadData.byteLength > maxPayloadBytes) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `WebRTC payload exceeded maxPayloadBytes (${payloadData.byteLength} > ${maxPayloadBytes}).`,
      {
        payloadBytes: payloadData.byteLength,
        maxPayloadBytes,
      }
    );
  }

  return {
    data: payloadData,
    manifestVersionSet: asOptionalString(parsed.manifestVersionSet),
    rangeStart: normalizeOptionalEnvelopeInteger(parsed.rangeStart, 'p2p.webrtc.rangeStart'),
    totalSize: normalizeOptionalEnvelopeInteger(parsed.totalSize, 'p2p.webrtc.totalSize'),
  };
}

function assertOpenDataChannel(channel, peerId) {
  if (!channel || typeof channel !== 'object') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unavailable,
      `WebRTC peer "${peerId}" did not return a data channel.`
    );
  }
  if (typeof channel.send !== 'function') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `WebRTC peer "${peerId}" data channel is missing send().`
    );
  }
  if (typeof channel.addEventListener !== 'function' || typeof channel.removeEventListener !== 'function') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `WebRTC peer "${peerId}" data channel must support addEventListener/removeEventListener.`
    );
  }
  if (typeof channel.readyState === 'string' && channel.readyState !== 'open') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unavailable,
      `WebRTC peer "${peerId}" data channel is not open (state="${channel.readyState}").`
    );
  }
}

function toRequestMessage(requestId, context) {
  return {
    schemaVersion: P2P_WEBRTC_MESSAGE_SCHEMA_VERSION,
    contractVersion: P2P_WEBRTC_DATA_PLANE_CONTRACT_VERSION,
    type: 'doppler_p2p_shard_request',
    requestId,
    shardIndex: context.shardIndex,
    attempt: context.attempt,
    maxRetries: context.maxRetries,
    resumeOffset: context.resumeOffset ?? 0,
    expectedHash: context.expectedHash ?? null,
    expectedSize: context.expectedSize ?? null,
    expectedManifestVersionSet: context.expectedManifestVersionSet ?? null,
  };
}

function sendRequestOnDataChannel(channel, request) {
  const serialized = JSON.stringify(request);
  try {
    channel.send(serialized);
  } catch (error) {
    throw normalizeP2PTransportError(error, {
      requestId: request.requestId,
      channel: 'webrtc',
    });
  }
}

function waitForShardResponse(channel, requestId, signal, timeoutMs, maxPayloadBytes) {
  return new Promise((resolve, reject) => {
    let timer = null;

    const cleanup = () => {
      if (timer != null) {
        clearTimeout(timer);
      }
      channel.removeEventListener('message', onMessage);
      channel.removeEventListener('error', onError);
      channel.removeEventListener('close', onClose);
      if (signal) {
        signal.removeEventListener('abort', onAbort);
      }
    };

    const fail = (error) => {
      cleanup();
      reject(error);
    };

    const finish = (value) => {
      cleanup();
      resolve(value);
    };

    const onAbort = () => {
      fail(createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.aborted,
        'WebRTC shard request aborted.',
        { requestId },
        false
      ));
    };

    const onError = (event) => {
      fail(createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.internal,
        `WebRTC data channel error for request ${requestId}.`,
        {
          requestId,
          eventType: event?.type ?? null,
        },
        true
      ));
    };

    const onClose = () => {
      fail(createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.unavailable,
        `WebRTC data channel closed while waiting for request ${requestId}.`,
        { requestId },
        false
      ));
    };

    const onMessage = (event) => {
      try {
        const payload = normalizeResponsePayload(event?.data, requestId, maxPayloadBytes);
        if (!payload) {
          return;
        }
        finish(payload);
      } catch (error) {
        fail(error);
      }
    };

    channel.addEventListener('message', onMessage);
    channel.addEventListener('error', onError);
    channel.addEventListener('close', onClose);
    if (signal) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }

    timer = setTimeout(() => {
      fail(createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.timeout,
        `WebRTC data channel timed out for request ${requestId}.`,
        {
          requestId,
          timeoutMs,
        },
        true
      ));
    }, timeoutMs);
  });
}

function normalizePeerSelectionResult(value) {
  if (typeof value === 'string') {
    const peerId = asOptionalString(value);
    return {
      peerId,
      metadata: null,
    };
  }

  if (!value || typeof value !== 'object') {
    return {
      peerId: null,
      metadata: null,
    };
  }

  const peerId = asOptionalString(value.peerId);
  const metadata = value.metadata && typeof value.metadata === 'object'
    ? value.metadata
    : null;
  return {
    peerId,
    metadata,
  };
}

export function isBrowserWebRTCAvailable() {
  return typeof globalThis.RTCPeerConnection === 'function';
}

export function createBrowserWebRTCDataPlaneTransport(config = {}) {
  if (config?.enabled !== true) {
    return null;
  }

  if (!isBrowserWebRTCAvailable()) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unconfigured,
      'Browser WebRTC data plane is enabled but RTCPeerConnection is unavailable.'
    );
  }

  if (typeof config.getDataChannel !== 'function') {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.unconfigured,
      'Browser WebRTC data plane requires getDataChannel(peerContext).'
    );
  }

  const selectPeer = typeof config.selectPeer === 'function'
    ? config.selectPeer
    : null;
  const staticPeerId = asOptionalString(config.peerId);
  const requestTimeoutMs = Math.max(1, asNonNegativeInteger(config.requestTimeoutMs, DEFAULT_REQUEST_TIMEOUT_MS));
  const maxPayloadBytes = Math.max(1, asNonNegativeInteger(config.maxPayloadBytes, DEFAULT_MAX_PAYLOAD_BYTES));

  return async function webRtcDataPlaneTransport(context) {
    const selection = normalizePeerSelectionResult(
      selectPeer ? await selectPeer(context) : { peerId: staticPeerId }
    );
    const peerId = selection.peerId ?? staticPeerId;
    if (!peerId) {
      throw createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.unavailable,
        `No WebRTC peer available for shard ${context?.shardIndex}.`,
        {
          shardIndex: context?.shardIndex ?? null,
          policyReason: 'peer_unavailable',
        }
      );
    }

    const channel = await config.getDataChannel({
      peerId,
      shardIndex: context.shardIndex,
      context,
      peerSelection: selection.metadata,
    });
    assertOpenDataChannel(channel, peerId);

    const requestId = createRequestId('doppler-p2p-webrtc');
    const request = toRequestMessage(requestId, context);
    const effectiveTimeoutMs = Math.max(
      1,
      Math.min(requestTimeoutMs, asNonNegativeInteger(context?.timeoutMs, requestTimeoutMs) || requestTimeoutMs)
    );

    const responsePromise = waitForShardResponse(
      channel,
      requestId,
      context.signal,
      effectiveTimeoutMs,
      maxPayloadBytes
    );
    sendRequestOnDataChannel(channel, request);
    const response = await responsePromise;

    if (response.miss === true || response.notFound === true) {
      return {
        miss: true,
        notFound: true,
      };
    }

    return {
      data: response.data,
      manifestVersionSet: response.manifestVersionSet ?? null,
      rangeStart: response.rangeStart ?? null,
      totalSize: response.totalSize ?? null,
    };
  };
}
