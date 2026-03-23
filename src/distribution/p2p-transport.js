import { log } from '../debug/index.js';
import {
  P2P_TRANSPORT_ERROR_CODES,
  createP2PTransportError,
} from './p2p-transport-contract.js';

const REQUIRED_MESSAGE_FIELDS = ['type', 'schemaVersion'];

const KNOWN_MESSAGE_TYPES = new Set([
  'doppler_p2p_shard_request',
  'doppler_p2p_shard_response',
  'doppler_p2p_heartbeat',
  'doppler_p2p_announce',
]);

export function validatePeerMessage(message) {
  if (!message || typeof message !== 'object') {
    log.warn('P2P', 'Peer message is not a valid object');
    return false;
  }

  for (const field of REQUIRED_MESSAGE_FIELDS) {
    if (message[field] === undefined || message[field] === null) {
      log.warn('P2P', `Peer message missing required field: ${field}`);
      return false;
    }
  }

  if (typeof message.type !== 'string' || !message.type.trim()) {
    log.warn('P2P', 'Peer message type must be a non-empty string');
    return false;
  }

  if (!KNOWN_MESSAGE_TYPES.has(message.type)) {
    log.warn('P2P', `Peer message has unknown type: ${message.type}`);
    return false;
  }

  if (typeof message.schemaVersion !== 'number' || !Number.isInteger(message.schemaVersion)) {
    log.warn('P2P', `Peer message schemaVersion must be an integer, got: ${message.schemaVersion}`);
    return false;
  }

  return true;
}

export function deserializeAndValidate(raw) {
  let parsed = null;
  if (typeof raw === 'string') {
    try {
      parsed = JSON.parse(raw);
    } catch {
      throw createP2PTransportError(
        P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
        'Peer message is not valid JSON.'
      );
    }
  } else if (raw && typeof raw === 'object') {
    parsed = raw;
  } else {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      'Peer message must be a string or object.'
    );
  }

  if (!validatePeerMessage(parsed)) {
    throw createP2PTransportError(
      P2P_TRANSPORT_ERROR_CODES.payloadInvalid,
      `Peer message failed schema validation (type="${parsed?.type}").`
    );
  }

  return parsed;
}
