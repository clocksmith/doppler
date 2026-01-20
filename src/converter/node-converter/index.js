#!/usr/bin/env node


import { resolve } from 'path';
import { readFileSync } from 'fs';
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
import { createConverterConfig } from '../../config/index.js';

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

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function assertString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function assertBoolean(value, label) {
  if (typeof value !== 'boolean') {
    throw new Error(`${label} must be a boolean`);
  }
}

function assertOptionalObject(value, label) {
  if (value === undefined || value === null) return;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function loadRawConfig(ref) {
  if (!ref) {
    throw new Error('--config is required');
  }
  const trimmed = ref.trim();
  if (trimmed.startsWith('{')) {
    return JSON.parse(trimmed);
  }
  const fullPath = resolve(process.cwd(), ref);
  const content = readFileSync(fullPath, 'utf8');
  return JSON.parse(content);
}

function parseConverterConfig(raw) {
  assertObject(raw, 'config');
  assertObject(raw.converter, 'converter');
  const converter = raw.converter;

  if (converter.test !== undefined) {
    assertBoolean(converter.test, 'converter.test');
  }
  if (converter.verbose !== undefined) {
    assertBoolean(converter.verbose, 'converter.verbose');
  }

  assertObject(converter.paths, 'converter.paths');
  const paths = converter.paths;
  assertString(paths.output, 'converter.paths.output');
  if (!converter.test) {
    assertString(paths.input, 'converter.paths.input');
  }

  assertOptionalObject(converter.quantization, 'converter.quantization');
  assertOptionalObject(converter.sharding, 'converter.sharding');
  assertOptionalObject(converter.weightLayout, 'converter.weightLayout');
  assertOptionalObject(converter.manifest, 'converter.manifest');
  assertOptionalObject(converter.output, 'converter.output');
  assertOptionalObject(converter.presets, 'converter.presets');

  const converterConfig = createConverterConfig({
    quantization: converter.quantization,
    sharding: converter.sharding,
    weightLayout: converter.weightLayout,
    manifest: converter.manifest,
    output: converter.output,
    presets: converter.presets,
  });

  return {
    converterConfig,
    inputPath: paths.input ? resolve(paths.input) : null,
    outputPath: resolve(paths.output),
    test: converter.test === true,
    verbose: converter.verbose === true,
  };
}

export async function main() {
  const opts = parseArgs(process.argv.slice(2));

  if (opts.help) {
    printHelp();
    process.exit(0);
  }

  if (!opts.config) {
    console.error('Error: --config is required');
    printHelp();
    process.exit(1);
  }

  let parsedConfig;
  try {
    const raw = loadRawConfig(opts.config);
    parsedConfig = parseConverterConfig(raw);
  } catch (error) {
    console.error(`Failed to load config "${opts.config}": ${error.message}`);
    process.exit(1);
  }

  const {
    converterConfig,
    inputPath,
    outputPath,
    test,
    verbose,
  } = parsedConfig;

  if (test) {
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

  if (!inputPath) {
    console.error('Error: converter.paths.input is required');
    process.exit(1);
  }

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

  logConversionStart(
    format,
    inputPath,
    outputPath,
    converterConfig.quantization.weights,
    converterConfig.quantization.embeddings
  );

  try {
    const result = format === 'gguf'
      ? await convertGGUF(inputPath, outputPath, { converterConfig, verbose })
      : await convertSafetensors(inputPath, outputPath, { converterConfig, verbose });

    logConversionComplete(
      result.manifestPath,
      result.shardCount,
      result.tensorCount,
      result.totalSize
    );
  } catch (error) {
    logConversionError(error, verbose);
    process.exit(1);
  }
}

// Note: main() is NOT auto-executed here.
// The parent node-converter.ts facade handles CLI execution.
// This allows index.ts to be imported programmatically without side effects.
