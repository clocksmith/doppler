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
  type CommandType,
  type FlagType,
  type ListEntry,
} from './protocol.js';

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
} as const;

export type BridgeStatusType = (typeof BridgeStatus)[keyof typeof BridgeStatus];

/**
 * Status change callback
 */
export type StatusChangeCallback = (status: BridgeStatusType) => void;

/**
 * Error callback
 */
export type ErrorCallback = (error: Error) => void;

/**
 * Chunk callback for streaming reads
 */
export type ChunkCallback = (chunk: Uint8Array, totalReceived: number) => void;

/**
 * Pending request info
 */
interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  chunks: Uint8Array[];
  totalReceived: number;
  onChunk: ChunkCallback | null;
  timeout: ReturnType<typeof setTimeout>;
}

/**
 * Chrome runtime types (minimal for what we need)
 */
interface ChromePort {
  name: string;
  postMessage: (message: unknown) => void;
  disconnect: () => void;
  onMessage: {
    addListener: (callback: (message: unknown) => void) => void;
  };
  onDisconnect: {
    addListener: (callback: () => void) => void;
  };
}

interface ChromeRuntime {
  connect: (extensionIdOrInfo?: string | { name: string }, connectInfo?: { name: string }) => ChromePort;
  lastError?: { message: string };
}

declare const chrome: {
  runtime?: ChromeRuntime;
};

/**
 * Binary message format from extension
 */
interface BinaryMessage {
  type: 'binary' | 'ack';
  data?: number[];
  reqId?: number;
}

// ============================================================================
// Extension Bridge Client Class
// ============================================================================

/**
 * Extension Bridge Client
 */
export class ExtensionBridgeClient {
  private port: ChromePort | null = null;
  private status: BridgeStatusType = BridgeStatus.DISCONNECTED;
  private nextReqId = 1;
  private pendingRequests = new Map<number, PendingRequest>();
  private extensionId: string | null = null;

  /** Status change event handler */
  public onStatusChange: StatusChangeCallback | null = null;
  /** Error event handler */
  public onError: ErrorCallback | null = null;

  /**
   * Check if the DOPPLER extension is installed
   */
  static isExtensionAvailable(): boolean {
    return (
      typeof chrome !== 'undefined' &&
      typeof chrome.runtime !== 'undefined' &&
      typeof chrome.runtime.connect === 'function'
    );
  }

  /**
   * Connect to the DOPPLER extension
   * @param extensionId - Extension ID (optional, uses known ID)
   */
  async connect(extensionId: string | null = null): Promise<void> {
    if (!ExtensionBridgeClient.isExtensionAvailable()) {
      throw new Error('Chrome extension API not available');
    }

    this.extensionId = extensionId;
    this.status = BridgeStatus.CONNECTING;
    this._notifyStatusChange();

    return new Promise((resolve, reject) => {
      try {
        // Connect to extension's background script
        const connectInfo = { name: 'doppler-bridge' };

        if (extensionId) {
          this.port = chrome.runtime!.connect(extensionId, connectInfo);
        } else {
          // Try to connect to the extension (requires externally_connectable)
          this.port = chrome.runtime!.connect(connectInfo);
        }

        // Set up message handler
        this.port.onMessage.addListener((message: unknown) => {
          this._handleMessage(message as BinaryMessage);
        });

        // Set up disconnect handler
        this.port.onDisconnect.addListener(() => {
          this._handleDisconnect();
        });

        // Send ping to verify connection
        const pingReqId = this._getNextReqId();
        const pingPromise = this._createPendingRequest(pingReqId, 5000);

        this.port.postMessage({
          type: 'binary',
          data: Array.from(new Uint8Array(encodeMessage(Command.PING, pingReqId))),
        });

        pingPromise
          .then(() => {
            this.status = BridgeStatus.CONNECTED;
            this._notifyStatusChange();
            resolve();
          })
          .catch((err: Error) => {
            this.status = BridgeStatus.ERROR;
            this._notifyStatusChange();
            reject(err);
          });
      } catch (err) {
        this.status = BridgeStatus.ERROR;
        this._notifyStatusChange();
        reject(new Error(`Failed to connect to extension: ${(err as Error).message}`));
      }
    });
  }

  /**
   * Disconnect from the extension
   */
  disconnect(): void {
    if (this.port) {
      this.port.disconnect();
      this.port = null;
    }

    // Reject all pending requests
    for (const [, pending] of this.pendingRequests) {
      pending.reject(new Error('Connection closed'));
    }
    this.pendingRequests.clear();

    this.status = BridgeStatus.DISCONNECTED;
    this._notifyStatusChange();
  }

  /**
   * Read data from a file via native host
   * @param path - File path
   * @param offset - Byte offset
   * @param length - Bytes to read
   * @param onChunk - Callback for each chunk (for streaming)
   */
  async read(
    path: string,
    offset: number,
    length: number,
    onChunk: ChunkCallback | null = null
  ): Promise<Uint8Array> {
    if (this.status !== BridgeStatus.CONNECTED) {
      throw new Error('Not connected to extension');
    }

    const reqId = this._getNextReqId();
    const request = createReadRequest(reqId, path, offset, length);

    // Create pending request with chunk accumulator
    const pending = this._createPendingRequest(reqId, 60000, onChunk);

    // Send request (convert to array for postMessage compatibility)
    this.port!.postMessage({
      type: 'binary',
      data: Array.from(new Uint8Array(request)),
    });

    return pending as Promise<Uint8Array>;
  }

  /**
   * List directory contents via native host
   * @param path - Directory path
   */
  async list(path: string): Promise<ListEntry[]> {
    if (this.status !== BridgeStatus.CONNECTED) {
      throw new Error('Not connected to extension');
    }

    const reqId = this._getNextReqId();
    const request = createListRequest(reqId, path);

    // Create pending request
    const pending = this._createPendingRequest(reqId, 30000);

    // Send request
    this.port!.postMessage({
      type: 'binary',
      data: Array.from(new Uint8Array(request)),
    });

    return pending as Promise<ListEntry[]>;
  }

  /**
   * Get next request ID
   */
  private _getNextReqId(): number {
    // Wrap at 32-bit unsigned max to avoid overflow
    const current = this.nextReqId;
    this.nextReqId = (this.nextReqId + 1) >>> 0;
    if (this.nextReqId === 0) {
      this.nextReqId = 1;
    }
    return current;
  }

  /**
   * Create a pending request
   */
  private _createPendingRequest(
    reqId: number,
    timeoutMs = 30000,
    onChunk: ChunkCallback | null = null
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      const pending: PendingRequest = {
        resolve,
        reject,
        chunks: [],
        totalReceived: 0,
        onChunk,
        timeout: setTimeout(() => {
          this.pendingRequests.delete(reqId);
          reject(new Error(`Request ${reqId} timed out`));
        }, timeoutMs),
      };

      this.pendingRequests.set(reqId, pending);
    });
  }

  /**
   * Handle incoming message from extension
   */
  private _handleMessage(message: BinaryMessage): void {
    if (message.type !== 'binary' || !message.data) {
      console.warn('[ExtensionBridge] Unexpected message type:', message.type);
      return;
    }

    // Convert array back to Uint8Array
    const data = new Uint8Array(message.data);

    if (data.length < HEADER_SIZE) {
      console.error('[ExtensionBridge] Message too short');
      return;
    }

    const header = decodeHeader(data.buffer);
    if (!header) {
      console.error('[ExtensionBridge] Invalid message header');
      return;
    }

    const payload = data.slice(HEADER_SIZE, HEADER_SIZE + header.payloadLen);
    const pending = this.pendingRequests.get(header.reqId);

    switch (header.cmd) {
      case Command.PONG:
        if (pending) {
          clearTimeout(pending.timeout);
          this.pendingRequests.delete(header.reqId);
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
          this._sendAck(header.reqId);

          // Check if this is the last chunk
          if (header.flags & Flag.LAST_CHUNK) {
            clearTimeout(pending.timeout);
            this.pendingRequests.delete(header.reqId);

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
          this.pendingRequests.delete(header.reqId);
          const entries = parseListResponse(payload);
          pending.resolve(entries);
        }
        break;

      case Command.ERROR:
        if (pending) {
          clearTimeout(pending.timeout);
          this.pendingRequests.delete(header.reqId);
          const error = parseErrorResponse(payload);
          pending.reject(new Error(`Native host error ${error.code}: ${error.message}`));
        }
        break;

      default:
        console.warn('[ExtensionBridge] Unknown command:', header.cmd);
    }
  }

  /**
   * Send ACK for backpressure
   */
  private _sendAck(reqId: number): void {
    if (this.port) {
      this.port.postMessage({
        type: 'ack',
        reqId,
      });
    }
  }

  /**
   * Handle disconnection
   */
  private _handleDisconnect(): void {
    const error = chrome.runtime?.lastError;
    console.warn('[ExtensionBridge] Disconnected:', error?.message || 'unknown');

    this.port = null;
    this.status = BridgeStatus.DISCONNECTED;
    this._notifyStatusChange();

    // Reject pending requests
    for (const [, pending] of this.pendingRequests) {
      pending.reject(new Error('Connection lost'));
    }
    this.pendingRequests.clear();

    if (this.onError) {
      this.onError(new Error(error?.message || 'Connection lost'));
    }
  }

  /**
   * Notify status change
   */
  private _notifyStatusChange(): void {
    if (this.onStatusChange) {
      this.onStatusChange(this.status);
    }
  }

  /**
   * Get current status
   */
  getStatus(): BridgeStatusType {
    return this.status;
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.status === BridgeStatus.CONNECTED;
  }
}

// ============================================================================
// Module-level functions
// ============================================================================

/** Global client instance */
let globalClient: ExtensionBridgeClient | null = null;

/**
 * Get global bridge client
 */
export function getBridgeClient(): ExtensionBridgeClient {
  if (!globalClient) {
    globalClient = new ExtensionBridgeClient();
  }
  return globalClient;
}

/**
 * Check if native bridge is available
 */
export function isBridgeAvailable(): boolean {
  return ExtensionBridgeClient.isExtensionAvailable();
}

export default ExtensionBridgeClient;
