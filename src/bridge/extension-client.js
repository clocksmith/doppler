/**
 * Extension Bridge Client
 * Phase 3: Communication with Native Host via Chrome Extension
 *
 * This module handles:
 * - Connection to background script
 * - Binary message passing with transferables
 * - Backpressure handling
 * - Request/response correlation
 *
 * @module bridge/extension-client
 */

import {
  Command,
  Flag,
  HEADER_SIZE,
  encodeMessage,
  decodeHeader,
  createReadRequest,
  createListRequest,
  parseReadResponse,
  parseListResponse,
  parseErrorResponse,
} from './protocol.js';
import { log } from '../debug/index.js';
import { DEFAULT_BRIDGE_TIMEOUT_CONFIG } from '../config/schema/index.js';

// ============================================================================
// Types and Interfaces
// ============================================================================

/**
 * Bridge status values
 */
export const BridgeStatus = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error',
};

// ============================================================================
// Extension Bridge Client Class
// ============================================================================

/**
 * Extension Bridge Client
 */
export class ExtensionBridgeClient {
  /** @type {object|null} */
  #port = null;
  /** @type {string} */
  #status = BridgeStatus.DISCONNECTED;
  /** @type {number} */
  #nextReqId = 1;
  /** @type {Map<number, object>} */
  #pendingRequests = new Map();
  /** @type {string|null} */
  #extensionId = null;

  /** Status change event handler */
  onStatusChange = null;
  /** Error event handler */
  onError = null;

  /**
   * Check if the DOPPLER extension is installed
   */
  static isExtensionAvailable() {
    return (
      typeof chrome !== 'undefined' &&
      typeof chrome.runtime !== 'undefined' &&
      typeof chrome.runtime.connect === 'function'
    );
  }

  /**
   * Connect to the DOPPLER extension
   * @param {string|null} extensionId - Extension ID (optional, uses known ID)
   */
  async connect(extensionId = null) {
    if (!ExtensionBridgeClient.isExtensionAvailable()) {
      throw new Error('Chrome extension API not available');
    }

    this.#extensionId = extensionId;
    this.#status = BridgeStatus.CONNECTING;
    this.#notifyStatusChange();

    return new Promise((resolve, reject) => {
      try {
        // Connect to extension's background script
        const connectInfo = { name: 'doppler-bridge' };

        if (extensionId) {
          this.#port = chrome.runtime.connect(extensionId, connectInfo);
        } else {
          // Try to connect to the extension (requires externally_connectable)
          this.#port = chrome.runtime.connect(connectInfo);
        }

        // Set up message handler
        this.#port.onMessage.addListener((message) => {
          this.#handleMessage(message);
        });

        // Set up disconnect handler
        this.#port.onDisconnect.addListener(() => {
          this.#handleDisconnect();
        });

        // Send ping to verify connection
        const pingReqId = this.#getNextReqId();
        const pingPromise = this.#createPendingRequest(pingReqId, DEFAULT_BRIDGE_TIMEOUT_CONFIG.pingTimeoutMs);

        this.#port.postMessage({
          type: 'binary',
          data: Array.from(new Uint8Array(encodeMessage(Command.PING, pingReqId))),
        });

        pingPromise
          .then(() => {
            this.#status = BridgeStatus.CONNECTED;
            this.#notifyStatusChange();
            resolve();
          })
          .catch((err) => {
            this.#status = BridgeStatus.ERROR;
            this.#notifyStatusChange();
            reject(err);
          });
      } catch (err) {
        this.#status = BridgeStatus.ERROR;
        this.#notifyStatusChange();
        reject(new Error(`Failed to connect to extension: ${err.message}`));
      }
    });
  }

  /**
   * Disconnect from the extension
   */
  disconnect() {
    if (this.#port) {
      this.#port.disconnect();
      this.#port = null;
    }

    // Reject all pending requests
    for (const [, pending] of this.#pendingRequests) {
      pending.reject(new Error('Connection closed'));
    }
    this.#pendingRequests.clear();

    this.#status = BridgeStatus.DISCONNECTED;
    this.#notifyStatusChange();
  }

  /**
   * Read data from a file via native host
   * @param {string} path - File path
   * @param {number} offset - Byte offset
   * @param {number} length - Bytes to read
   * @param {Function|null} onChunk - Callback for each chunk (for streaming)
   */
  async read(path, offset, length, onChunk = null) {
    if (this.#status !== BridgeStatus.CONNECTED) {
      throw new Error('Not connected to extension');
    }

    const reqId = this.#getNextReqId();
    const request = createReadRequest(reqId, path, offset, length);

    // Create pending request with chunk accumulator
    const pending = this.#createPendingRequest(reqId, DEFAULT_BRIDGE_TIMEOUT_CONFIG.readTimeoutMs, onChunk);

    // Send request (convert to array for postMessage compatibility)
    this.#port.postMessage({
      type: 'binary',
      data: Array.from(new Uint8Array(request)),
    });

    return pending;
  }

  /**
   * List directory contents via native host
   * @param {string} path - Directory path
   */
  async list(path) {
    if (this.#status !== BridgeStatus.CONNECTED) {
      throw new Error('Not connected to extension');
    }

    const reqId = this.#getNextReqId();
    const request = createListRequest(reqId, path);

    // Create pending request
    const pending = this.#createPendingRequest(reqId, DEFAULT_BRIDGE_TIMEOUT_CONFIG.listTimeoutMs);

    // Send request
    this.#port.postMessage({
      type: 'binary',
      data: Array.from(new Uint8Array(request)),
    });

    return pending;
  }

  /**
   * Get next request ID
   */
  #getNextReqId() {
    // Wrap at 32-bit unsigned max to avoid overflow
    const current = this.#nextReqId;
    this.#nextReqId = (this.#nextReqId + 1) >>> 0;
    if (this.#nextReqId === 0) {
      this.#nextReqId = 1;
    }
    return current;
  }

  /**
   * Create a pending request
   */
  #createPendingRequest(reqId, timeoutMs = DEFAULT_BRIDGE_TIMEOUT_CONFIG.defaultTimeoutMs, onChunk = null) {
    return new Promise((resolve, reject) => {
      const pending = {
        resolve,
        reject,
        chunks: [],
        totalReceived: 0,
        onChunk,
        timeout: setTimeout(() => {
          this.#pendingRequests.delete(reqId);
          reject(new Error(`Request ${reqId} timed out`));
        }, timeoutMs),
      };

      this.#pendingRequests.set(reqId, pending);
    });
  }

  /**
   * Handle incoming message from extension
   */
  #handleMessage(message) {
    if (message.type !== 'binary' || !message.data) {
      log.warn('ExtensionBridge', `Unexpected message type: ${message.type}`);
      return;
    }

    // Convert array back to Uint8Array
    const data = new Uint8Array(message.data);

    if (data.length < HEADER_SIZE) {
      log.error('ExtensionBridge', 'Message too short');
      return;
    }

    const header = decodeHeader(data.buffer);
    if (!header) {
      log.error('ExtensionBridge', 'Invalid message header');
      return;
    }

    const payload = data.slice(HEADER_SIZE, HEADER_SIZE + header.payloadLen);
    const pending = this.#pendingRequests.get(header.reqId);

    switch (header.cmd) {
      case Command.PONG:
        if (pending) {
          clearTimeout(pending.timeout);
          this.#pendingRequests.delete(header.reqId);
          pending.resolve(undefined);
        }
        break;

      case Command.READ_RESPONSE:
        if (pending) {
          const { data: chunkData } = parseReadResponse(payload);

          // Accumulate chunk
          pending.chunks.push(chunkData);
          pending.totalReceived += chunkData.length;

          // Notify chunk callback
          if (pending.onChunk) {
            pending.onChunk(chunkData, pending.totalReceived);
          }

          // Send ACK for backpressure
          this.#sendAck(header.reqId);

          // Check if this is the last chunk
          if (header.flags & Flag.LAST_CHUNK) {
            clearTimeout(pending.timeout);
            this.#pendingRequests.delete(header.reqId);

            // Combine chunks
            const totalSize = pending.chunks.reduce((s, c) => s + c.length, 0);
            const result = new Uint8Array(totalSize);
            let pos = 0;
            for (const chunk of pending.chunks) {
              result.set(chunk, pos);
              pos += chunk.length;
            }

            pending.resolve(result);
          }
        }
        break;

      case Command.LIST_RESPONSE:
        if (pending) {
          clearTimeout(pending.timeout);
          this.#pendingRequests.delete(header.reqId);
          const entries = parseListResponse(payload);
          pending.resolve(entries);
        }
        break;

      case Command.ERROR:
        if (pending) {
          clearTimeout(pending.timeout);
          this.#pendingRequests.delete(header.reqId);
          const error = parseErrorResponse(payload);
          pending.reject(new Error(`Native host error ${error.code}: ${error.message}`));
        }
        break;

      default:
        log.warn('ExtensionBridge', `Unknown command: ${header.cmd}`);
    }
  }

  /**
   * Send ACK for backpressure
   */
  #sendAck(reqId) {
    if (this.#port) {
      this.#port.postMessage({
        type: 'ack',
        reqId,
      });
    }
  }

  /**
   * Handle disconnection
   */
  #handleDisconnect() {
    const error = chrome.runtime?.lastError;
    log.warn('ExtensionBridge', `Disconnected: ${error?.message || 'unknown'}`);

    this.#port = null;
    this.#status = BridgeStatus.DISCONNECTED;
    this.#notifyStatusChange();

    // Reject pending requests
    for (const [, pending] of this.#pendingRequests) {
      pending.reject(new Error('Connection lost'));
    }
    this.#pendingRequests.clear();

    if (this.onError) {
      this.onError(new Error(error?.message || 'Connection lost'));
    }
  }

  /**
   * Notify status change
   */
  #notifyStatusChange() {
    if (this.onStatusChange) {
      this.onStatusChange(this.#status);
    }
  }

  /**
   * Get current status
   */
  getStatus() {
    return this.#status;
  }

  /**
   * Check if connected
   */
  isConnected() {
    return this.#status === BridgeStatus.CONNECTED;
  }
}

// ============================================================================
// Module-level functions
// ============================================================================

/** Global client instance */
let globalClient = null;

/**
 * Get global bridge client
 */
export function getBridgeClient() {
  if (!globalClient) {
    globalClient = new ExtensionBridgeClient();
  }
  return globalClient;
}

/**
 * Check if native bridge is available
 */
export function isBridgeAvailable() {
  return ExtensionBridgeClient.isExtensionAvailable();
}

export default ExtensionBridgeClient;
