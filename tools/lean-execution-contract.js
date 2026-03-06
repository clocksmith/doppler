#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  runLeanExecutionContractForManifest,
  writeExecutionContractLeanModuleForManifest,
} from '../src/tooling/lean-execution-contract-runner.js';

function usage() {
  console.error(
    'Usage: node tools/lean-execution-contract.js --manifest <manifest.json> ' +
    '[--module-name <Name>] [--emit <output.lean>] [--no-check]'
  );
}

function parseArgs(argv) {
  const args = {
    manifestPath: null,
    moduleName: null,
    emitPath: null,
    check: true,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--manifest') {
      args.manifestPath = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === '--module-name') {
      args.moduleName = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === '--emit') {
      args.emitPath = argv[index + 1] ?? null;
      index += 1;
      continue;
    }
    if (arg === '--no-check') {
      args.check = false;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.manifestPath) {
    throw new Error('--manifest is required.');
  }
  return args;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

try {
  const args = parseArgs(process.argv.slice(2));
  const manifestText = fs.readFileSync(path.resolve(rootDir, args.manifestPath), 'utf8');
  const manifest = JSON.parse(manifestText);
  if (args.check) {
    const result = runLeanExecutionContractForManifest(manifest, {
      rootDir,
      moduleName: args.moduleName,
      emitPath: args.emitPath,
      check: true,
    });
    if (!result.ok) {
      process.exitCode = 1;
      console.error(`lean-execution-contract: contract failed (${result.toolchainRef})`);
    } else {
      console.log(`lean-execution-contract: ok (${result.toolchainRef})`);
    }
    console.log(`lean-execution-contract: wrote ${path.relative(rootDir, result.generatedPath)}`);
  } else {
    const generated = writeExecutionContractLeanModuleForManifest(manifest, {
      rootDir,
      moduleName: args.moduleName,
      emitPath: args.emitPath,
    });
    console.log(`lean-execution-contract: wrote ${path.relative(rootDir, generated.generatedPath)}`);
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.startsWith('Unknown argument:') || message === '--manifest is required.') {
    usage();
  }
  console.error(message);
  process.exitCode = 1;
}
