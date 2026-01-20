#!/usr/bin/env node


import { createTestModel } from '../src/converter/test-model.js';
import { loadConfig } from '../cli/config/index.js';
import { resolve } from 'path';

function parseArgs(argv) {
  const opts = { config: null, help: false };
  let i = 0;
  while (i < argv.length) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      opts.help = true;
      i += 1;
      continue;
    }
    if (arg === '--config' || arg === '-c') {
      opts.config = argv[i + 1] || null;
      i += 2;
      continue;
    }
    if (!arg.startsWith('-') && !opts.config) {
      opts.config = arg;
      i += 1;
      continue;
    }
    console.error(`Unknown argument: ${arg}`);
    opts.help = true;
    break;
  }
  return opts;
}

function printHelp() {
  console.log(`
Generate a tiny test model fixture.

Usage:
  doppler --config <ref>

Config requirements:
  tools.generateFixture.outputDir (string, required)
`);
}

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


async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help) {
    printHelp();
    process.exit(0);
  }
  if (!parsed.config) {
    console.error('Error: --config is required');
    printHelp();
    process.exit(1);
  }

  const loaded = await loadConfig(parsed.config);
  const raw = loaded.raw ?? {};
  assertObject(raw.tools, 'tools');
  const toolConfig = raw.tools?.generateFixture;
  if (!toolConfig || typeof toolConfig !== 'object') {
    throw new Error('tools.generateFixture is required in config');
  }

  assertString(toolConfig.outputDir, 'tools.generateFixture.outputDir');
  const outputDir = resolve(toolConfig.outputDir);

  console.log(`Generating test model fixture...`);
  console.log(`Output: ${outputDir}`);

  try {
    const result = await createTestModel(outputDir);

    console.log(`\nFixture created successfully:`);
    console.log(`  Manifest: ${result.manifestPath}`);
    console.log(`  Shards: ${result.shardCount}`);
    console.log(`  Tensors: ${result.tensorCount}`);
    console.log(`  Total size: ${(result.totalSize / 1024).toFixed(1)} KB`);

    console.log(`\nModel config:`);
    console.log(`  vocab_size: 1000`);
    console.log(`  hidden_size: 64`);
    console.log(`  num_layers: 2`);
    console.log(`  num_heads: 2`);
    console.log(`  context_length: 128`);

    console.log(`\nReady for Agent-B testing!`);

  } catch (error) {
    const err =  (error);
    console.error(`Error generating fixture: ${err.message}`);
    process.exit(1);
  }
}

main();
