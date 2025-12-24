#!/usr/bin/env node
/**
 * Generate Test Model Fixture - Creates a tiny .rdrr model for testing.
 */

import { createTestModel } from './rdrr-writer.js';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main(): Promise<void> {
  const outputDir = process.argv[2] || resolve(__dirname, '../tests/fixtures/tiny-model');

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
    const err = error as Error;
    console.error(`Error generating fixture: ${err.message}`);
    process.exit(1);
  }
}

main();
