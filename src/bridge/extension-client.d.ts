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

import type { ListEntry } from './protocol.js';

/**
 * Bridge status values
 */
export declare const BridgeStatus: {
  readonly DISCONNECTED: 'disconnected';
  readonly CONNECTING: 'connecting';
  readonly CONNECTED: 'connected';
  readonly ERROR: 'error';
};

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
 * Extension Bridge Client
 */
export declare class ExtensionBridgeClient {
  /** Status change event handler */
  onStatusChange: StatusChangeCallback | null;
  /** Error event handler */
  onError: ErrorCallback | null;

  /**
   * Check if the DOPPLER extension is installed
   */
  static isExtensionAvailable(): boolean;

  /**
   * Connect to the DOPPLER extension
   * @param extensionId - Extension ID (optional, uses known ID)
   */
  connect(extensionId?: string | null): Promise<void>;

  /**
   * Disconnect from the extension
   */
  disconnect(): void;

  /**
   * Read data from a file via native host
   * @param path - File path
   * @param offset - Byte offset
   * @param length - Bytes to read
   * @param onChunk - Callback for each chunk (for streaming)
   */
  read(
    path: string,
    offset: number,
    length: number,
    onChunk?: ChunkCallback | null
  ): Promise<Uint8Array>;

  /**
   * List directory contents via native host
   * @param path - Directory path
   */
  list(path: string): Promise<ListEntry[]>;

  /**
   * Get current status
   */
  getStatus(): BridgeStatusType;

  /**
   * Check if connected
   */
  isConnected(): boolean;
}

/**
 * Get global bridge client
 */
export declare function getBridgeClient(): ExtensionBridgeClient;

/**
 * Check if native bridge is available
 */
export declare function isBridgeAvailable(): boolean;

export default ExtensionBridgeClient;
