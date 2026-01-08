/**
 * Progress reporting and logging utilities for the Node.js Model Converter.
 *
 * Provides verbose logging helpers and progress display utilities.
 *
 * @module converter/node-converter/progress
 */

import { log as debugLog } from '../../debug/index.js';

/**
 * Create a verbose logger that only logs when verbose mode is enabled.
 */
export function createVerboseLogger(verbose, category) {
  return (msg) => {
    if (verbose) {
      debugLog.verbose(category, msg);
    }
  };
}

/**
 * Log conversion start information.
 */
export function logConversionStart(format, inputPath, outputPath, weightQuant, embedQuant) {
  console.log(`Converting ${format.toUpperCase()} model...`);
  console.log(`  Input: ${inputPath}`);
  console.log(`  Output: ${outputPath}`);
  if (weightQuant) {
    console.log(`  Weight quantization: ${weightQuant}`);
  }
  if (embedQuant) {
    console.log(`  Embed quantization: ${embedQuant}`);
  }
}

/**
 * Log conversion completion information.
 */
export function logConversionComplete(manifestPath, shardCount, tensorCount, totalSize) {
  console.log(`\nConversion complete:`);
  console.log(`  Manifest: ${manifestPath}`);
  console.log(`  Shards: ${shardCount}`);
  console.log(`  Tensors: ${tensorCount}`);
  console.log(`  Size: ${(totalSize / (1024 * 1024)).toFixed(1)} MB`);
}

/**
 * Log test fixture creation information.
 */
export function logTestFixtureComplete(manifestPath, shardCount, tensorCount, totalSize) {
  console.log(`\nTest fixture created:`);
  console.log(`  Manifest: ${manifestPath}`);
  console.log(`  Shards: ${shardCount}`);
  console.log(`  Tensors: ${tensorCount}`);
  console.log(`  Size: ${(totalSize / 1024).toFixed(1)} KB`);
}

/**
 * Log conversion error.
 */
export function logConversionError(error, verbose) {
  console.error(`\nConversion failed: ${error.message}`);
  if (verbose) {
    console.error(error.stack);
  }
}
