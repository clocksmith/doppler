#!/usr/bin/env node

import { spawnSync } from 'node:child_process';

// Composed CI validates reference and conversion digests together in check:all.
// Standalone kernels:check retains its reference-digest validation.
const args = process.argv.slice(2);
if (args.some((arg) => arg !== '--defer-digests')) {
  console.error('Usage: node tools/check-kernels.js [--defer-digests]');
  process.exit(1);
}
const deferDigests = args.includes('--defer-digests');

const checks = [
  { name: 'compensated-state-helpers', command: ['node', 'tools/generate-linear-state-kernels.js', '--check'] },
  {
    name: 'uniform-interfaces',
    command: ['node', 'tools/generate-kernel-uniforms.js', '--check'],
  },
  {
    name: 'codegen',
    command: ['node', 'tools/generate-wgsl.js', '--check'],
  },
  {
    name: 'registry',
    command: ['node', 'tools/sync-kernel-reachability.js', '--check'],
  },
  {
    name: 'digests',
    command: ['node', 'tools/sync-kernel-ref-digests.js', '--check'],
  },
  {
    name: 'language-contract',
    command: ['node', 'tests/config/kernel-language-contract.test.js'],
  },
];

const failures = [];

for (const check of checks) {
  if (deferDigests && check.name === 'digests') continue;
  console.log(`[kernels:check] ${check.name}`);
  const result = spawnSync(check.command[0], check.command.slice(1), {
    stdio: 'inherit',
    shell: false,
  });
  if (result.status !== 0) {
    failures.push(check.name);
  }
}

if (failures.length > 0) {
  console.error(`[kernels:check] failed: ${failures.join(', ')}`);
  process.exit(1);
}

console.log('[kernels:check] all checks passed');
