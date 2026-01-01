/**
 * GGUF format module - types and parsing utilities.
 *
 * Browser code should import from './types.js' directly.
 * Node code can use './parser.js' for file system operations.
 */

// Re-export all types and browser-safe parsing
export * from './types.js';

// Re-export Node.js file parser
export { parseGGUFFile } from './parser.js';
