#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { PROGRAM_BUNDLE_JSON_SCHEMA } from '../src/config/schema/program-bundle-json-schema.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outputPath = path.join(repoRoot, 'src/config/schema/program-bundle.schema.json');
const expected = `${JSON.stringify(PROGRAM_BUNDLE_JSON_SCHEMA, null, 2)}\n`;
const check = process.argv.includes('--check');

if (check) {
  const actual = await fs.readFile(outputPath, 'utf8').catch(() => null);
  if (actual !== expected) {
    throw new Error(`Program Bundle schema artifact is stale: ${outputPath}`);
  }
  console.log(`Program Bundle schema artifact is current: ${outputPath}`);
} else {
  await fs.writeFile(outputPath, expected, 'utf8');
  console.log(`Wrote Program Bundle schema artifact: ${outputPath}`);
}
