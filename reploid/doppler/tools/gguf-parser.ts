/**
 * GGUF Model Format Parser (Node wrapper).
 */

import { readFile } from 'fs/promises';
import { parseGGUF, type GGUFParseResult } from '../formats/gguf.js';

export * from '../formats/gguf.js';

export async function parseGGUFFile(filePath: string): Promise<GGUFParseResult> {
  const buffer = await readFile(filePath);
  const arrayBuffer = buffer.buffer.slice(
    buffer.byteOffset,
    buffer.byteOffset + buffer.byteLength
  );
  return parseGGUF(arrayBuffer);
}
