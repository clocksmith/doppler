#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { buildRuntimeOptimizationResultsIndex } from '../src/tooling/runtime-optimization-index.js';

function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--check') {
      flags.check = true;
      continue;
    }
    if (!token.startsWith('--') || argv[index + 1] === undefined) {
      throw new Error(`runtime optimization index: invalid argument "${token}"`);
    }
    flags[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  if (!flags.receipts || !flags.out) {
    throw new Error('Usage: build-runtime-optimization-index --receipts <file|dir> --out <file> [--check]');
  }
  return flags;
}

async function readReceipts(target) {
  const resolved = path.resolve(target);
  const stat = await fs.stat(resolved);
  const files = stat.isDirectory()
    ? (await fs.readdir(resolved))
      .filter((filename) => filename.endsWith('.json'))
      .sort()
      .map((filename) => path.join(resolved, filename))
    : [resolved];
  const receipts = [];
  for (const file of files) {
    const value = JSON.parse(await fs.readFile(file, 'utf8'));
    if (value?.schema === 'doppler.runtime-optimization-receipt/v1') {
      receipts.push(value);
    }
  }
  return receipts;
}

const flags = parseArgs(process.argv.slice(2));
const index = buildRuntimeOptimizationResultsIndex(await readReceipts(flags.receipts));
const serialized = `${JSON.stringify(index, null, 2)}\n`;
const outputPath = path.resolve(flags.out);
if (flags.check) {
  const current = await fs.readFile(outputPath, 'utf8').catch(() => null);
  if (current !== serialized) {
    throw new Error(`runtime optimization index: ${outputPath} is stale`);
  }
  console.log(`[ok] runtime optimization index is current (${index.receiptCount} receipts)`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, serialized, 'utf8');
  console.log(`[ok] wrote ${outputPath} (${index.negativeResultCount} negative results)`);
}
