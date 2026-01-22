/**
 * gguf-parser-browser.ts - Browser-safe GGUF Parser
 *
 * Re-exports the shared GGUF parser for browser usage.
 *
 * @module browser/gguf-parser-browser
 */

import type { GGUFParseResult } from '../formats/gguf.js';
import type { TensorSource } from './tensor-source-file.js';

export declare function parseGGUFHeaderFromSource(source: TensorSource | File): Promise<GGUFParseResult & { fileSize: number }>;

export * from '../formats/gguf.js';
