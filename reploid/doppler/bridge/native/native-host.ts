#!/usr/bin/env node
/**
 * DOPPLER Native Bridge Host - Native messaging protocol for Chrome extension.
 * Provides file access to bypass browser storage limits.
 */

import { statSync, readdirSync } from 'fs';
import { promises as fsPromises } from 'fs';
import { resolve, normalize, join } from 'path';

const MAGIC = 0x5245504C; // "REPL"
const HEADER_SIZE = 16;
const MAX_CHUNK_SIZE = 8 * 1024 * 1024; // 8MB
const MAX_READ_SIZE = 100 * 1024 * 1024; // 100MB max per request to prevent OOM

const CMD_PING = 0x00;
const CMD_PONG = 0x01;
const CMD_READ = 0x02;
const CMD_READ_RESPONSE = 0x03;
const CMD_LIST = 0x06;
const CMD_LIST_RESPONSE = 0x07;
const CMD_ERROR = 0xFF;

const FLAG_LAST_CHUNK = 0x02;

const ERR_NOT_FOUND = 1;
const ERR_PERMISSION_DENIED = 2;
const ERR_IO_ERROR = 3;
const ERR_INVALID_REQUEST = 4;

const DEFAULT_ALLOWED_DIRS = '/Users:/home:/tmp:/var/tmp';
const ALLOWED_DIRS = (process.env.DOPPLER_ALLOWED_DIRS || DEFAULT_ALLOWED_DIRS).split(':');

export interface BinaryMessage {
  type: 'binary';
  data: number[];
}

export interface AckMessage {
  type: 'ack';
}

export type NativeMessage = BinaryMessage | AckMessage;

export interface ListEntry {
  name: string;
  isDir: boolean;
  size: number;
}

interface NodeError extends Error {
  code?: string;
}

function isPathAllowed(filePath: string): boolean {
  try {
    const resolved = resolve(filePath);
    const normalized = normalize(resolved);

    if (!normalized.startsWith('/')) return false;

    for (const dir of ALLOWED_DIRS) {
      // Ensure proper directory boundary check to prevent /home/user matching /home/username
      // Path must either equal the allowed dir or be inside it (with a path separator)
      if (normalized === dir || normalized.startsWith(dir + '/')) {
        return true;
      }
    }
    return false;
  } catch {
    return false;
  }
}

function createHeader(cmd: number, flags: number, reqId: number, payloadLen: number): Buffer {
  const header = Buffer.alloc(HEADER_SIZE);
  header.writeUInt32LE(MAGIC, 0);
  header.writeUInt8(cmd, 4);
  header.writeUInt8(flags, 5);
  header.writeUInt32LE(reqId, 8);
  header.writeUInt32LE(payloadLen, 12);
  return header;
}

function createPongResponse(reqId: number): BinaryMessage {
  const header = createHeader(CMD_PONG, 0, reqId, 0);
  return { type: 'binary', data: Array.from(header) };
}

function createReadResponse(reqId: number, offset: number, data: Buffer, isLast: boolean): BinaryMessage {
  const payloadLen = 8 + data.length;
  const message = Buffer.alloc(HEADER_SIZE + payloadLen);

  message.writeUInt32LE(MAGIC, 0);
  message.writeUInt8(CMD_READ_RESPONSE, 4);
  message.writeUInt8(isLast ? FLAG_LAST_CHUNK : 0, 5);
  message.writeUInt32LE(reqId, 8);
  message.writeUInt32LE(payloadLen, 12);

  const offsetBigInt = BigInt(offset);
  message.writeUInt32LE(Number(offsetBigInt & 0xFFFFFFFFn), 16);
  message.writeUInt32LE(Number(offsetBigInt >> 32n), 20);

  data.copy(message, 24);

  return { type: 'binary', data: Array.from(message) };
}

function createErrorResponse(reqId: number, code: number, errorMsg: string): BinaryMessage {
  const msgBytes = Buffer.from(errorMsg, 'utf8');
  const payloadLen = 4 + msgBytes.length;
  const message = Buffer.alloc(HEADER_SIZE + payloadLen);

  message.writeUInt32LE(MAGIC, 0);
  message.writeUInt8(CMD_ERROR, 4);
  message.writeUInt8(0, 5);
  message.writeUInt32LE(reqId, 8);
  message.writeUInt32LE(payloadLen, 12);

  message.writeUInt32LE(code, 16);
  msgBytes.copy(message, 20);

  return { type: 'binary', data: Array.from(message) };
}

function createListResponse(reqId: number, entries: ListEntry[]): BinaryMessage {
  const jsonStr = JSON.stringify(entries);
  const jsonBytes = Buffer.from(jsonStr, 'utf8');
  const payloadLen = jsonBytes.length;
  const message = Buffer.alloc(HEADER_SIZE + payloadLen);

  message.writeUInt32LE(MAGIC, 0);
  message.writeUInt8(CMD_LIST_RESPONSE, 4);
  message.writeUInt8(FLAG_LAST_CHUNK, 5);
  message.writeUInt32LE(reqId, 8);
  message.writeUInt32LE(payloadLen, 12);

  jsonBytes.copy(message, HEADER_SIZE);

  return { type: 'binary', data: Array.from(message) };
}

async function handleListRequest(reqId: number, payload: Buffer): Promise<BinaryMessage[]> {
  const dirPath = payload.toString('utf8');

  if (!isPathAllowed(dirPath)) {
    return [createErrorResponse(reqId, ERR_PERMISSION_DENIED, 'Path not in allowed directory')];
  }

  try {
    const stats = statSync(dirPath);
    if (!stats.isDirectory()) {
      return [createErrorResponse(reqId, ERR_INVALID_REQUEST, 'Path is not a directory')];
    }

    const dirents = readdirSync(dirPath, { withFileTypes: true });
    const entries: ListEntry[] = [];

    for (const dirent of dirents) {
      const entryPath = join(dirPath, dirent.name);
      try {
        const entryStat = statSync(entryPath);
        entries.push({
          name: dirent.name,
          isDir: dirent.isDirectory(),
          size: entryStat.size,
        });
      } catch {
        entries.push({
          name: dirent.name,
          isDir: dirent.isDirectory(),
          size: 0,
        });
      }
    }

    entries.sort((a, b) => {
      if (a.isDir !== b.isDir) return (b.isDir ? 1 : 0) - (a.isDir ? 1 : 0);
      return a.name.localeCompare(b.name);
    });

    return [createListResponse(reqId, entries)];
  } catch (err) {
    const nodeErr = err as NodeError;
    const code = nodeErr.code === 'ENOENT' ? ERR_NOT_FOUND :
                 nodeErr.code === 'EACCES' ? ERR_PERMISSION_DENIED :
                 ERR_IO_ERROR;
    return [createErrorResponse(reqId, code, nodeErr.message)];
  }
}

async function handleReadRequest(reqId: number, payload: Buffer): Promise<BinaryMessage[]> {
  if (payload.length < 16) {
    return [createErrorResponse(reqId, ERR_INVALID_REQUEST, 'Payload too short')];
  }

  const offsetLow = payload.readUInt32LE(0);
  const offsetHigh = payload.readUInt32LE(4);
  const offset = BigInt(offsetLow) + (BigInt(offsetHigh) << 32n);

  const lengthLow = payload.readUInt32LE(8);
  const lengthHigh = payload.readUInt32LE(12);
  const length = BigInt(lengthLow) + (BigInt(lengthHigh) << 32n);

  const filePath = payload.slice(16).toString('utf8');

  if (!isPathAllowed(filePath)) {
    return [createErrorResponse(reqId, ERR_PERMISSION_DENIED, 'Path not in allowed directory')];
  }

  try {
    const stats = statSync(filePath);
    const fileSize = BigInt(stats.size);

    if (offset >= fileSize) {
      return [createErrorResponse(reqId, ERR_INVALID_REQUEST, 'Offset beyond file end')];
    }

    // Cap read size to prevent OOM on large requests
    const requestedLength = Number(length < (fileSize - offset) ? length : (fileSize - offset));
    const actualLength = Math.min(requestedLength, MAX_READ_SIZE);

    const responses: BinaryMessage[] = [];
    let pos = 0;

    // Stream chunks directly from file descriptor to avoid loading full buffer
    const fd = await fsPromises.open(filePath, 'r');
    try {
      const chunkBuffer = Buffer.alloc(MAX_CHUNK_SIZE);

      while (pos < actualLength) {
        const chunkSize = Math.min(MAX_CHUNK_SIZE, actualLength - pos);
        const { bytesRead } = await fd.read(chunkBuffer, 0, chunkSize, Number(offset) + pos);

        if (bytesRead === 0) break;

        const chunk = chunkBuffer.slice(0, bytesRead);
        const isLast = pos + bytesRead >= actualLength;

        responses.push(createReadResponse(reqId, Number(offset) + pos, chunk, isLast));
        pos += bytesRead;
      }
    } finally {
      await fd.close();
    }

    return responses;
  } catch (err) {
    const nodeErr = err as NodeError;
    const code = nodeErr.code === 'ENOENT' ? ERR_NOT_FOUND :
                 nodeErr.code === 'EACCES' ? ERR_PERMISSION_DENIED :
                 ERR_IO_ERROR;
    return [createErrorResponse(reqId, code, nodeErr.message)];
  }
}

async function handleMessage(msg: NativeMessage): Promise<BinaryMessage[]> {
  if (msg.type === 'binary') {
    const bytes = Buffer.from(msg.data);

    if (bytes.length < HEADER_SIZE) {
      return [createErrorResponse(0, ERR_INVALID_REQUEST, 'Message too short')];
    }

    const magic = bytes.readUInt32LE(0);
    if (magic !== MAGIC) {
      return [createErrorResponse(0, ERR_INVALID_REQUEST, 'Invalid magic')];
    }

    const cmd = bytes.readUInt8(4);
    const reqId = bytes.readUInt32LE(8);
    const payloadLen = bytes.readUInt32LE(12);

    const payload = payloadLen > 0 ? bytes.slice(HEADER_SIZE, HEADER_SIZE + payloadLen) : Buffer.alloc(0);

    switch (cmd) {
      case CMD_PING:
        return [createPongResponse(reqId)];
      case CMD_READ:
        return await handleReadRequest(reqId, payload);
      case CMD_LIST:
        return await handleListRequest(reqId, payload);
      default:
        return [createErrorResponse(reqId, ERR_INVALID_REQUEST, 'Unknown command')];
    }
  } else if (msg.type === 'ack') {
    return [];
  }

  return [];
}

function writeResponse(response: BinaryMessage): void {
  const json = JSON.stringify(response);
  const buffer = Buffer.from(json, 'utf8');
  const lenBuffer = Buffer.alloc(4);
  lenBuffer.writeUInt32LE(buffer.length, 0);

  process.stdout.write(lenBuffer);
  process.stdout.write(buffer);
}

async function main(): Promise<void> {
  let buffer = Buffer.alloc(0);

  process.stdin.on('readable', async () => {
    let chunk: Buffer | string | null;
    while ((chunk = process.stdin.read()) !== null) {
      if (typeof chunk === 'string') {
        chunk = Buffer.from(chunk, 'binary');
      }
      buffer = Buffer.concat([buffer, chunk as Buffer]);

      while (buffer.length >= 4) {
        const msgLen = buffer.readUInt32LE(0);

        if (msgLen === 0 || msgLen > 1024 * 1024) {
          console.error('[DopplerBridge] Invalid message length:', msgLen);
          buffer = buffer.slice(4);
          continue;
        }

        if (buffer.length < 4 + msgLen) {
          break;
        }

        const msgData = buffer.slice(4, 4 + msgLen);
        buffer = buffer.slice(4 + msgLen);

        try {
          const msg = JSON.parse(msgData.toString('utf8')) as NativeMessage;
          const responses = await handleMessage(msg);

          for (const response of responses) {
            writeResponse(response);
          }
        } catch (err) {
          const error = err as Error;
          console.error('[DopplerBridge] Error handling message:', error.message);
        }
      }
    }
  });

  process.stdin.on('end', () => {
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[DopplerBridge] Fatal error:', err);
  process.exit(1);
});

export { isPathAllowed, handleMessage, handleReadRequest, handleListRequest };
