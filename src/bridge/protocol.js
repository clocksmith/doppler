
import { MB } from '../config/schema/index.js';

export const MAGIC = 0x5245504c;
export const HEADER_SIZE = 16;
export const MAX_CHUNK_SIZE = 8 * MB;
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
};

export const Flag = {
  NONE: 0x00,
  COMPRESSED: 0x01,
  LAST_CHUNK: 0x02,
};

export const ErrorCode = {
  OK: 0,
  NOT_FOUND: 1,
  PERMISSION_DENIED: 2,
  IO_ERROR: 3,
  INVALID_REQUEST: 4,
  QUOTA_EXCEEDED: 5,
};

export function encodeMessage(
  cmd,
  reqId,
  payload = null,
  flags = Flag.NONE
) {
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

export function decodeHeader(buffer) {
  if (buffer.byteLength < HEADER_SIZE) return null;

  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== MAGIC) return null;

  return {
    cmd: view.getUint8(4),
    flags: view.getUint8(5),
    reqId: view.getUint32(8, true),
    payloadLen: view.getUint32(12, true),
  };
}

export function createReadRequest(
  reqId,
  path,
  offset,
  length
) {
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

export function createListRequest(reqId, path) {
  const pathBytes = new TextEncoder().encode(path);
  return encodeMessage(Command.LIST, reqId, pathBytes);
}

export function createAck(reqId) {
  const buffer = new ArrayBuffer(ACK_SIZE);
  new DataView(buffer).setUint32(0, reqId, true);
  return buffer;
}

export function parseReadResponse(payload) {
  const view = new DataView(payload.buffer, payload.byteOffset);
  const offsetLow = view.getUint32(0, true);
  const offsetHigh = view.getUint32(4, true);

  return {
    offset: offsetLow + offsetHigh * 0x100000000,
    data: payload.slice(8),
  };
}

export function parseListResponse(payload) {
  return JSON.parse(new TextDecoder().decode(payload));
}

export function parseErrorResponse(payload) {
  const view = new DataView(payload.buffer, payload.byteOffset);
  return {
    code: view.getUint32(0, true),
    message: new TextDecoder().decode(payload.slice(4)),
  };
}

// Legacy aliases for backwards compatibility
export const CMD = Command;
export const FLAGS = Flag;
export const ERROR_CODES = ErrorCode;
