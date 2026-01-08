/**
 * GGUF Model Format Parser (Node wrapper).
 * Supports large files (>2GB) via streaming reads.
 */

import { open, stat } from 'fs/promises';
import { parseGGUF } from './types.js';

export * from './types.js';

/**
 * Parse GGUF file using streaming for large file support.
 * Only reads header + metadata + tensor info (not the tensor data itself).
 */
export async function parseGGUFFile(filePath) {
  const fileStats = await stat(filePath);
  const fileSize = fileStats.size;

  // For files > 2GB, we need to read in chunks
  // GGUF header + metadata typically fits in first 100MB even for huge models
  const HEADER_READ_SIZE = Math.min(fileSize, 100 * 1024 * 1024);

  const fileHandle = await open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(HEADER_READ_SIZE);
    await fileHandle.read(buffer, 0, HEADER_READ_SIZE, 0);

    const arrayBuffer = buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength
    );

    const result = parseGGUF(arrayBuffer);

    // Store file path for later tensor data reading
    result.filePath = filePath;
    result.fileSize = fileSize;

    return result;
  } finally {
    await fileHandle.close();
  }
}
