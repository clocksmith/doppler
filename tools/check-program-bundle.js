#!/usr/bin/env node

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkProgramBundleFile } from '../src/tooling/program-bundle.js';

function usage() {
  return [
    'Usage:',
    '  node tools/check-program-bundle.js <bundle.json> [...bundle.json]',
  ].join('\n');
}

async function main() {
  const paths = process.argv.slice(2).filter((arg) => arg !== '--help' && arg !== '-h');
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log(usage());
    return;
  }
  if (paths.length === 0) {
    throw new Error('At least one Program Bundle path is required.');
  }
  const results = [];
  for (const bundlePath of paths) {
    results.push(await checkProgramBundleFile(bundlePath));
  }
  console.log(JSON.stringify({ ok: true, results }, null, 2));
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[program-bundle:check] ${error.message}`);
    process.exit(1);
  });
}
