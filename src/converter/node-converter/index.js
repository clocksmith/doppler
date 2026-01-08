#!/usr/bin/env node
/**
 * Node.js Model Converter - Convert HuggingFace/GGUF models to RDRR format.
 *
 * Uses config-as-code pattern: model families detected via JSON presets.
 *
 * Usage:
 *   npx tsx converter/node-converter.ts <input> <output> [options]
 *
 * Examples:
 *   npx tsx converter/node-converter.ts ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/HASH/  models/gemma-1b
 *   npx tsx converter/node-converter.ts model.gguf models/my-model --quantize q4_k_m
 *   npx tsx converter/node-converter.ts --test ./test-model  # Create tiny test fixture
 *
 * @module converter/node-converter
 */

import { resolve } from 'path';
import { stat } from 'fs/promises';
import { parseArgs, printHelp } from './cli.js';
import { detectInputFormat } from './detection.js';
import { convertSafetensors, convertGGUF } from './converter.js';
import {
  logConversionStart,
  logConversionComplete,
  logTestFixtureComplete,
  logConversionError,
} from './progress.js';
import { createTestModel } from '../writer.js';

// Re-export types and utilities for programmatic use
export { parseArgs, printHelp } from './cli.js';
export { detectInputFormat, detectModelTypeFromPreset } from './detection.js';
export { convertSafetensors, convertGGUF } from './converter.js';
export {
  normalizeQuantTag,
  validateQuantType,
  resolveManifestQuantization,
  buildVariantTag,
  buildQuantizationInfo,
  resolveModelId,
  toWebGPUDtype,
} from './quantization.js';
export {
  createVerboseLogger,
  logConversionStart,
  logConversionComplete,
  logTestFixtureComplete,
  logConversionError,
} from './progress.js';

/**
 * Main entry point for the CLI.
 *
 * Parses arguments, detects input format, and runs conversion.
 */
export async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  // Handle --test mode
  if (opts.test) {
    const outputPath = opts.output || resolve(process.cwd(), 'test-model');
    console.log(`Creating test model fixture at: ${outputPath}`);
    const result = await createTestModel(outputPath);
    logTestFixtureComplete(
      result.manifestPath,
      result.shardCount,
      result.tensorCount,
      result.totalSize
    );
    process.exit(0);
  }

  if (!opts.input || !opts.output) {
    console.error('Error: <input> and <output> are required');
    printHelp();
    process.exit(1);
  }

  const inputPath = resolve(opts.input);
  const outputPath = resolve(opts.output);

  // Validate input exists
  try {
    await stat(inputPath);
  } catch {
    console.error(`Error: Input path does not exist: ${inputPath}`);
    process.exit(1);
  }

  // Detect format
  const format = await detectInputFormat(inputPath);
  if (format === 'unknown') {
    console.error(`Error: Could not detect model format for: ${inputPath}`);
    console.error('Expected: HuggingFace directory or .gguf file');
    process.exit(1);
  }

  logConversionStart(format, inputPath, outputPath, opts.weightQuant, opts.embedQuant);

  try {
    const result = format === 'gguf'
      ? await convertGGUF(inputPath, outputPath, opts)
      : await convertSafetensors(inputPath, outputPath, opts);

    logConversionComplete(
      result.manifestPath,
      result.shardCount,
      result.tensorCount,
      result.totalSize
    );
  } catch (error) {
    logConversionError(error, opts.verbose);
    process.exit(1);
  }
}

// Note: main() is NOT auto-executed here.
// The parent node-converter.ts facade handles CLI execution.
// This allows index.ts to be imported programmatically without side effects.
