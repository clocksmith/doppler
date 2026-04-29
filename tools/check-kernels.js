#!/usr/bin/env node

import { spawnSync } from 'node:child_process';

const checks = [
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
];

const failures = [];

for (const check of checks) {
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
