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
export async function createBridgeClient(
  extensionId: string | null = null
): Promise<import('./extension-client.js').ExtensionBridgeClient> {
  const { ExtensionBridgeClient, isBridgeAvailable } = await import('./extension-client.js');

  if (!isBridgeAvailable()) {
    throw new Error('Native bridge not available - Chrome extension API required');
  }

  const client = new ExtensionBridgeClient();
  await client.connect(extensionId);
  return client;
}

/**
 * Read file via native bridge
 * @param path - File path
 * @param offset - Byte offset (default: 0)
 * @param length - Bytes to read (default: entire file)
 */
export async function readFileNative(
  path: string,
  offset = 0,
  length = 0
): Promise<Uint8Array> {
  const { getBridgeClient } = await import('./extension-client.js');
  const client = getBridgeClient();

  if (!client.isConnected()) {
    await client.connect();
  }

  return client.read(path, offset, length);
}
