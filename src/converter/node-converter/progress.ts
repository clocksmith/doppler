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
 *
 * @param verbose - Whether verbose logging is enabled
 * @param category - Log category (e.g., 'Convert', 'Safetensors')
 * @returns Logger function
 */
export function createVerboseLogger(
  verbose: boolean,
  category: string
): (msg: string) => void {
  return (msg: string) => {
    if (verbose) {
      debugLog.verbose(category, msg);
    }
  };
}

/**
 * Log conversion start information.
 *
 * @param format - Detected input format
 * @param inputPath - Input path
 * @param outputPath - Output path
 * @param weightQuant - Weight quantization option
 * @param embedQuant - Embedding quantization option
 */
export function logConversionStart(
  format: string,
  inputPath: string,
  outputPath: string,
  weightQuant: string | null,
  embedQuant: string | null
): void {
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
 *
 * @param manifestPath - Path to generated manifest
 * @param shardCount - Number of shards created
 * @param tensorCount - Number of tensors written
 * @param totalSize - Total size in bytes
 */
export function logConversionComplete(
  manifestPath: string,
  shardCount: number,
  tensorCount: number,
  totalSize: number
): void {
  console.log(`\nConversion complete:`);
  console.log(`  Manifest: ${manifestPath}`);
  console.log(`  Shards: ${shardCount}`);
  console.log(`  Tensors: ${tensorCount}`);
  console.log(`  Size: ${(totalSize / (1024 * 1024)).toFixed(1)} MB`);
}

/**
 * Log test fixture creation information.
 *
 * @param manifestPath - Path to generated manifest
 * @param shardCount - Number of shards created
 * @param tensorCount - Number of tensors written
 * @param totalSize - Total size in bytes
 */
export function logTestFixtureComplete(
  manifestPath: string,
  shardCount: number,
  tensorCount: number,
  totalSize: number
): void {
  console.log(`\nTest fixture created:`);
  console.log(`  Manifest: ${manifestPath}`);
  console.log(`  Shards: ${shardCount}`);
  console.log(`  Tensors: ${tensorCount}`);
  console.log(`  Size: ${(totalSize / 1024).toFixed(1)} KB`);
}

/**
 * Log conversion error.
 *
 * @param error - Error that occurred
 * @param verbose - Whether to show stack trace
 */
export function logConversionError(error: Error, verbose: boolean): void {
  console.error(`\nConversion failed: ${error.message}`);
  if (verbose) {
    console.error(error.stack);
  }
}
