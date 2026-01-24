#!/usr/bin/env node
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const RESULTS_DIR = path.join(__dirname, 'results');
const SCHEMA_PATH = path.join(__dirname, '..', 'docs', 'BENCHMARK_SCHEMA.json');

const REQUIRED_TYPE_MAP = {
  schemaVersion: 'number',
  timestamp: 'string',
  suite: 'string'
};

async function loadSchema() {
  const text = await fs.readFile(SCHEMA_PATH, 'utf8');
  return JSON.parse(text);
}

async function listResultFiles() {
  const entries = await fs.readdir(RESULTS_DIR, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => path.join(RESULTS_DIR, entry.name));
}

function validateRequiredFields(data, required) {
  const missing = [];
  for (const key of required) {
    if (!(key in data)) missing.push(key);
  }
  return missing;
}

function validateTypes(data) {
  const mismatches = [];
  for (const [key, type] of Object.entries(REQUIRED_TYPE_MAP)) {
    if (key in data && typeof data[key] !== type) {
      mismatches.push(`${key} (${typeof data[key]} != ${type})`);
    }
  }
  return mismatches;
}

async function main() {
  const schema = await loadSchema();
  const required = Array.isArray(schema.required) ? schema.required : [];
  const files = await listResultFiles();

  if (files.length === 0) {
    console.error('[bench-validate] No JSON results found in tests/results.');
    process.exitCode = 1;
    return;
  }

  let hasErrors = false;
  for (const file of files) {
    let data;
    try {
      data = JSON.parse(await fs.readFile(file, 'utf8'));
    } catch (err) {
      console.error(`[bench-validate] Invalid JSON: ${file}`);
      console.error(`  ${err.message}`);
      hasErrors = true;
      continue;
    }

    const missing = validateRequiredFields(data, required);
    const mismatches = validateTypes(data);

    if (missing.length || mismatches.length) {
      console.error(`[bench-validate] Schema mismatch: ${file}`);
      if (missing.length) {
        console.error(`  Missing required fields: ${missing.join(', ')}`);
      }
      if (mismatches.length) {
        console.error(`  Type mismatches: ${mismatches.join(', ')}`);
      }
      hasErrors = true;
    }
  }

  if (hasErrors) {
    process.exitCode = 1;
    return;
  }

  console.log(`[bench-validate] OK (${files.length} files)`);
}

main().catch((err) => {
  console.error('[bench-validate] Failed to validate benchmark results');
  console.error(err);
  process.exitCode = 1;
});
