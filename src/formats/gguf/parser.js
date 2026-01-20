

import { open, stat } from 'fs/promises';
import { parseGGUF } from './types.js';
import { MAX_HEADER_SIZE } from '../../config/schema/index.js';

export * from './types.js';


export async function parseGGUFFile(filePath) {
  const fileStats = await stat(filePath);
  const fileSize = fileStats.size;

  // For files > 2GB, we need to read in chunks
  // GGUF header + metadata typically fits in first 100MB even for huge models
  const headerReadSize = Math.min(fileSize, MAX_HEADER_SIZE);

  const fileHandle = await open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(headerReadSize);
    await fileHandle.read(buffer, 0, headerReadSize, 0);

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
