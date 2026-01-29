/**
 * DOPPLER Native Bridge Module
 * Phase 3: Native Host Communication
 *
 * @module bridge
 */

export {
  MAGIC,
  HEADER_SIZE,
  MAX_CHUNK_SIZE,
  Command,
  Flag,
  ErrorCode,
  CMD,
  FLAGS,
  ERROR_CODES,
  encodeMessage,
  decodeHeader,
  createReadRequest,
  createAck,
  parseReadResponse,
  parseErrorResponse,
} from './protocol.js';

export type {
  CommandType,
  FlagType,
  ErrorCodeType,
  MessageHeader,
  ReadResponse,
  ErrorResponse,
  ListEntry,
} from './protocol.js';

export {
  ExtensionBridgeClient,
  BridgeStatus,
  getBridgeClient,
  isBridgeAvailable,
} from './extension-client.js';

export type {
  BridgeStatusType,
  StatusChangeCallback,
  ErrorCallback,
  ChunkCallback,
} from './extension-client.js';

/**
 * Create and connect a bridge client
 * @param extensionId - Optional extension ID
 */
export declare function createBridgeClient(
  extensionId?: string | null
): Promise<import('./extension-client.js').ExtensionBridgeClient>;

/**
 * Read file via native bridge
 * @param path - File path
 * @param offset - Byte offset (default: 0)
 * @param length - Bytes to read (default: entire file)
 */
export declare function readFileNative(
  path: string,
  offset?: number,
  length?: number
): Promise<Uint8Array>;
