
import { parseGGUFHeader } from '../formats/gguf/types.js';
import { HEADER_READ_SIZE } from '../config/schema/index.js';
import { normalizeTensorSource } from './tensor-source-file.js';

export async function parseGGUFHeaderFromSource(source) {
  const resolved = normalizeTensorSource(source);
  const readSize = Math.min(resolved.size, HEADER_READ_SIZE);
  const buffer = await resolved.readRange(0, readSize);
  const info = parseGGUFHeader(buffer);
  return {
    ...info,
    fileSize: resolved.size,
  };
}

export * from '../formats/gguf/types.js';
