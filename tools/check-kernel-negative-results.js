#!/usr/bin/env node

import registry from '../benchmarks/kernels/negative-results.json' with { type: 'json' };
import {
  findKernelNegativeResults,
  validateKernelNegativeResults,
} from '../src/tooling/kernel-negative-results.js';

const parsed = validateKernelNegativeResults(registry);
const filters = {};
for (let index = 2; index < process.argv.length; index += 1) {
  const token = process.argv[index];
  const value = process.argv[index + 1];
  if (!['--model', '--candidate', '--adapter', '--phase'].includes(token) || !value) {
    throw new Error(
      'Usage: node tools/check-kernel-negative-results.js ' +
      '[--model <id>] [--candidate <id>] [--adapter <digest>] [--phase <phase>]'
    );
  }
  const field = {
    '--model': 'modelId',
    '--candidate': 'candidate',
    '--adapter': 'adapterDigest',
    '--phase': 'phase',
  }[token];
  filters[field] = value;
  index += 1;
}
const matches = findKernelNegativeResults(parsed, filters);
console.log(`kernel negative results: ${parsed.entries.length} valid, ${matches.length} matched`);
for (const entry of matches) {
  console.log(
    `${entry.id}: ${entry.scope.modelId} ${entry.scope.candidate} ` +
    `${entry.scope.adapter.architecture} ${entry.measurement.throughputDeltaPercent.toFixed(2)}%`
  );
}
