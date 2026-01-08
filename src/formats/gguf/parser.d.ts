/**
 * GGUF Model Format Parser (Node wrapper).
 * Supports large files (>2GB) via streaming reads.
 */

import type { GGUFParseResult } from './types.js';

export * from './types.js';

/**
 * Parse GGUF file using streaming for large file support.
 * Only reads header + metadata + tensor info (not the tensor data itself).
 */
export declare function parseGGUFFile(filePath: string): Promise<GGUFParseResult>;
