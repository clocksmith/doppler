#!/usr/bin/env node
/**
 * Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.
 *
 * This is a re-export facade that maintains backward compatibility.
 * The actual implementation is in the node-converter/ directory.
 *
 * Usage:
 *   npx tsx converter/node-converter.ts <input> <output> [options]
 *
 * @module converter/node-converter
 */

// Re-export everything from the modular implementation
export * from './node-converter/index.js';

// Import and run main for CLI usage
import { main } from './node-converter/index.js';

// Execute main when this file is run directly as CLI entry point
main().catch((err) => {
  console.error(err);
  process.exit(1);
});
