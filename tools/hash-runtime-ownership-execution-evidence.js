#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';

import {
  validateRuntimeOwnershipExecutionEvidence,
} from './lib/runtime-ownership-execution-evidence.js';

export function parseArgs(argv) {
  const args = { receiptPath: '', json: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--receipt') args.receiptPath = argv[++index] || '';
    else if (token === '--json') args.json = true;
    else throw new Error(`Unknown argument: ${token}`);
  }
  if (!args.receiptPath) throw new Error('--receipt is required.');
  return args;
}

export async function hashRuntimeOwnershipExecutionEvidence(receiptPath) {
  const resolvedPath = path.resolve(receiptPath);
  let receipt;
  try {
    receipt = JSON.parse(await fs.readFile(resolvedPath, 'utf8'));
  } catch (error) {
    throw new Error(`Execution evidence is not readable JSON: ${error.message}`);
  }
  const result = validateRuntimeOwnershipExecutionEvidence(receipt);
  if (result.errors.length > 0) {
    throw new Error(`Execution evidence is invalid: ${result.errors.join('; ')}`);
  }
  return {
    path: resolvedPath,
    executionId: result.evidenceId,
    status: result.status,
    reasons: result.reasons,
  };
}

export async function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const result = await hashRuntimeOwnershipExecutionEvidence(args.receiptPath);
  if (args.json) console.log(JSON.stringify(result, null, 2));
  else console.log(result.executionId);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error?.stack || error?.message || String(error));
    process.exitCode = 1;
  });
}
