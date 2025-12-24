/**
 * DOPPLER Native Bridge Protocol
 * Binary protocol for Extension <-> Native Host communication.
 *
 * Header format (16 bytes, little-endian):
 * | Offset | Field   | Type | Description           |
 * |--------|---------|------|-----------------------|
 * | 0x00   | MAGIC   | u32  | 0x5245504C ("REPL")   |
 * | 0x04   | CMD     | u8   | Command type          |
 * | 0x05   | FLAGS   | u8   | Flags (compressed)    |
 * | 0x06   | PADDING | u16  | Reserved              |
 * | 0x08   | REQ_ID  | u32  | Request correlation   |
 * | 0x0C   | LEN     | u32  | Payload length        |
 * | 0x10   | PAYLOAD | []   | Variable-length       |
 */

export const MAGIC = 0x5245504c;
export const HEADER_SIZE = 16;
export const MAX_CHUNK_SIZE = 8 * 1024 * 1024;
export const ACK_SIZE = 4;

export const Command = {
  PING: 0x00,
  PONG: 0x01,
  READ: 0x02,
  READ_RESPONSE: 0x03,
  WRITE: 0x04,
  WRITE_ACK: 0x05,
  LIST: 0x06,
  LIST_RESPONSE: 0x07,
  ERROR: 0xff,
} as const;

export type CommandType = (typeof Command)[keyof typeof Command];

export const Flag = {
  NONE: 0x00,
  COMPRESSED: 0x01,
  LAST_CHUNK: 0x02,
} as const;

export type FlagType = (typeof Flag)[keyof typeof Flag];

export const ErrorCode = {
  OK: 0,
  NOT_FOUND: 1,
  PERMISSION_DENIED: 2,
  IO_ERROR: 3,
  INVALID_REQUEST: 4,
  QUOTA_EXCEEDED: 5,
} as const;

export type ErrorCodeType = (typeof ErrorCode)[keyof typeof ErrorCode];

export interface MessageHeader {
  cmd: CommandType;
  flags: FlagType;
  reqId: number;
  payloadLen: number;
}

export interface ReadResponse {
  offset: number;
  data: Uint8Array;
}

export interface ErrorResponse {
  code: ErrorCodeType;
  message: string;
}

export interface ListEntry {
  name: string;
  isDir: boolean;
  size: number;
}

export function encodeMessage(
  cmd: CommandType,
  reqId: number,
  payload: Uint8Array | null = null,
  flags: FlagType = Flag.NONE
): ArrayBuffer {
  const payloadLen = payload?.length ?? 0;
  const buffer = new ArrayBuffer(HEADER_SIZE + payloadLen);
  const view = new DataView(buffer);

  view.setUint32(0, MAGIC, true);
  view.setUint8(4, cmd);
  view.setUint8(5, flags);
  view.setUint16(6, 0, true);
  view.setUint32(8, reqId, true);
  view.setUint32(12, payloadLen, true);

  if (payload && payloadLen > 0) {
    new Uint8Array(buffer, HEADER_SIZE).set(payload);
  }

  return buffer;
}

export function decodeHeader(buffer: ArrayBuffer): MessageHeader | null {
  if (buffer.byteLength < HEADER_SIZE) return null;

  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) return null;

  return {
    cmd: view.getUint8(4) as CommandType,
    flags: view.getUint8(5) as FlagType,
    reqId: view.getUint32(8, true),
    payloadLen: view.getUint32(12, true),
  };
}

export function createReadRequest(
  reqId: number,
  path: string,
  offset: number,
  length: number
): ArrayBuffer {
  const pathBytes = new TextEncoder().encode(path);
  const payload = new Uint8Array(16 + pathBytes.length);
  const view = new DataView(payload.buffer);

  view.setUint32(0, offset & 0xffffffff, true);
  view.setUint32(4, Math.floor(offset / 0x100000000), true);
  view.setUint32(8, length & 0xffffffff, true);
  view.setUint32(12, Math.floor(length / 0x100000000), true);
  payload.set(pathBytes, 16);

  return encodeMessage(Command.READ, reqId, payload);
}

export function createListRequest(reqId: number, path: string): ArrayBuffer {
  const pathBytes = new TextEncoder().encode(path);
  return encodeMessage(Command.LIST, reqId, pathBytes);
}

export function createAck(reqId: number): ArrayBuffer {
  const buffer = new ArrayBuffer(ACK_SIZE);
  new DataView(buffer).setUint32(0, reqId, true);
  return buffer;
}

export function parseReadResponse(payload: Uint8Array): ReadResponse {
  const view = new DataView(payload.buffer, payload.byteOffset);
  const offsetLow = view.getUint32(0, true);
  const offsetHigh = view.getUint32(4, true);

  return {
    offset: offsetLow + offsetHigh * 0x100000000,
    data: payload.slice(8),
  };
}

export function parseListResponse(payload: Uint8Array): ListEntry[] {
  return JSON.parse(new TextDecoder().decode(payload));
}

export function parseErrorResponse(payload: Uint8Array): ErrorResponse {
  const view = new DataView(payload.buffer, payload.byteOffset);
  return {
    code: view.getUint32(0, true) as ErrorCodeType,
    message: new TextDecoder().decode(payload.slice(4)),
  };
}

// Legacy aliases for backwards compatibility
export const CMD = Command;
export const FLAGS = Flag;
export const ERROR_CODES = ErrorCode;
