#!/usr/bin/env node


// Re-export everything from the modular implementation
export * from './node-converter/index.js';

// Import and run main for CLI usage
import { main } from './node-converter/index.js';

// Execute main when this file is run directly as CLI entry point
main().catch((err) => {
  console.error(err);
  process.exit(1);
});
