import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseSafetensorsHeader } from '../../formats/safetensors/types.js';

export const MAX_NODE_READ_BYTES = 64 * 1024 * 1024;

export function normalizePath(value) {
  return String(value || '').trim();
}

export function isGgufPath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.gguf');
}

export function isTflitePath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.tflite');
}

export function isLiteRTTaskPath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.task');
}

export function isLiteRTLMPath(filePath) {
  return String(filePath || '').toLowerCase().endsWith('.litertlm');
}

export async function getPathStats(targetPath, label) {
  try {
    return await fs.stat(targetPath);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      throw new Error(`node source runtime: ${label} does not exist: ${targetPath}`);
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`node source runtime: failed to stat ${label} "${targetPath}": ${message}`);
  }
}

export async function fileExists(targetPath) {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

export async function readJson(filePath, label) {
  const text = await fs.readFile(filePath, 'utf8');
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      throw new Error('JSON root must be an object');
    }
    return parsed;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Invalid JSON in ${label}: ${message}`);
  }
}

export function createNodeFileAccess() {
  const readers = new Map();

  const getReader = (filePath) => {
    const normalizedPath = normalizePath(filePath);
    if (!normalizedPath) {
      throw new Error('node source runtime: filePath is required.');
    }
    let reader = readers.get(normalizedPath);
    if (reader) {
      return reader;
    }
    let handlePromise = null;
    let sizePromise = null;
    let closed = false;
    const ensureHandle = async () => {
      if (closed) {
        throw new Error(`node source runtime: file reader already closed for "${normalizedPath}".`);
      }
      if (!handlePromise) {
        handlePromise = fs.open(normalizedPath, 'r').catch((error) => {
          handlePromise = null;
          throw error;
        });
      }
      return handlePromise;
    };
    const getSize = async () => {
      if (!sizePromise) {
        sizePromise = (async () => {
          const handle = await ensureHandle();
          const stats = await handle.stat();
          return Number(stats.size);
        })().catch((error) => {
          sizePromise = null;
          throw error;
        });
      }
      return sizePromise;
    };
    reader = {
      async readRange(offset, length) {
        if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
          return new ArrayBuffer(0);
        }
        const start = Math.max(0, Math.floor(offset));
        const fileSize = await getSize();
        const end = Math.min(fileSize, start + Math.floor(length));
        if (end <= start) {
          return new ArrayBuffer(0);
        }
        const handle = await ensureHandle();
        const out = Buffer.allocUnsafe(end - start);
        let position = 0;
        while (position < out.length) {
          const nextChunkBytes = Math.min(out.length - position, MAX_NODE_READ_BYTES);
          const { bytesRead } = await handle.read(
            out,
            position,
            nextChunkBytes,
            start + position
          );
          if (bytesRead === 0) break;
          position += bytesRead;
        }
        return out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);
      },
      async getSize() {
        return getSize();
      },
      async close() {
        closed = true;
        const handle = await handlePromise;
        handlePromise = null;
        if (handle) {
          await handle.close();
        }
      },
    };
    readers.set(normalizedPath, reader);
    return reader;
  };

  return {
    async readRange(filePath, offset, length) {
      return getReader(filePath).readRange(offset, length);
    },
    async getSize(filePath) {
      return getReader(filePath).getSize();
    },
    async close() {
      const pending = Array.from(readers.values(), (reader) => reader.close());
      readers.clear();
      await Promise.all(pending);
    },
  };
}

export async function readSafetensorsHeaderFromFile(filePath, fileAccess) {
  const headerPrefixBuffer = await fileAccess.readRange(filePath, 0, 8);
  const prefixBytes = new Uint8Array(headerPrefixBuffer);
  if (prefixBytes.byteLength < 8) {
    throw new Error(`Invalid safetensors header prefix for "${filePath}"`);
  }
  const headerSize = Number(new DataView(headerPrefixBuffer).getBigUint64(0, true));
  const headerBuffer = await fileAccess.readRange(filePath, 8, headerSize);
  const fullHeader = new Uint8Array(8 + headerSize);
  fullHeader.set(prefixBytes, 0);
  fullHeader.set(new Uint8Array(headerBuffer), 8);
  return parseSafetensorsHeader(
    fullHeader.buffer.slice(fullHeader.byteOffset, fullHeader.byteOffset + fullHeader.byteLength)
  );
}

export function buildNodeFileReaders(fileAccess) {
  const readRange = async (filePath, offset, length) => fileAccess.readRange(filePath, offset, length);
  const streamRange = async function* (filePath, offset, length, options = {}) {
    if (!Number.isFinite(offset) || !Number.isFinite(length) || length <= 0) {
      return;
    }
    const fileSize = await fileAccess.getSize(filePath);
    const start = Math.max(0, Math.floor(offset));
    const end = Math.min(fileSize, start + Math.floor(length));
    if (end <= start) {
      return;
    }
    const chunkBytesRaw = Number(options?.chunkBytes);
    const highWaterMark = Number.isFinite(chunkBytesRaw) && chunkBytesRaw > 0
      ? Math.floor(chunkBytesRaw)
      : MAX_NODE_READ_BYTES;
    const stream = createReadStream(filePath, {
      start,
      end: end - 1,
      highWaterMark,
    });
    for await (const chunk of stream) {
      yield chunk;
    }
  };
  const readText = async (filePath) => {
    try {
      return await fs.readFile(filePath, 'utf8');
    } catch (error) {
      if (error?.code === 'ENOENT') {
        return null;
      }
      throw error;
    }
  };
  const readBinary = async (filePath) => {
    const bytes = await fs.readFile(filePath);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  };
  return {
    readRange,
    streamRange,
    readText,
    readBinary,
    close: fileAccess.close,
  };
}

async function computeFileHash(filePath, hashAlgorithm) {
  return new Promise((resolve, reject) => {
    const hash = createHash(hashAlgorithm);
    const stream = createReadStream(filePath);

    stream.on('data', (chunk) => {
      hash.update(chunk);
    });
    stream.on('end', () => {
      resolve(hash.digest('hex'));
    });
    stream.on('error', (error) => {
      const message = error instanceof Error ? error.message : String(error);
      reject(new Error(`Failed to stream source asset "${filePath}" for hashing: ${message}`));
    });
  });
}

export async function addHashesToFileEntries(entries, hashAlgorithm) {
  const normalized = [];
  for (const entry of Array.isArray(entries) ? entries : []) {
    const filePath = normalizePath(entry?.path);
    if (!filePath) continue;
    const stats = await getPathStats(filePath, `source asset (${filePath})`);
    normalized.push({
      ...entry,
      path: filePath,
      size: Number.isFinite(entry?.size) ? Math.max(0, Math.floor(Number(entry.size))) : Number(stats.size),
      hash: await computeFileHash(filePath, hashAlgorithm),
      hashAlgorithm,
    });
  }
  return normalized;
}
