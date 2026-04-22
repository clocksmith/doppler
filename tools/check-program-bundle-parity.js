#!/usr/bin/env node

import { checkProgramBundleParity } from '../src/tooling/program-bundle-parity.js';

function parseArgs(argv) {
  const args = {
    bundlePath: null,
    providers: null,
    mode: 'contract',
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--bundle') {
      args.bundlePath = argv[++index] ?? null;
    } else if (arg === '--providers') {
      args.providers = String(argv[++index] ?? '')
        .split(',')
        .map((entry) => entry.trim())
        .filter(Boolean);
    } else if (arg === '--mode') {
      args.mode = argv[++index] ?? 'contract';
    } else if (!arg.startsWith('--') && !args.bundlePath) {
      args.bundlePath = arg;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.bundlePath) {
    throw new Error('Usage: node tools/check-program-bundle-parity.js --bundle <bundle.json> [--providers browser-webgpu,node:webgpu,node:doe-gpu] [--mode contract|execute]');
  }
  return args;
}

try {
  const options = parseArgs(process.argv.slice(2));
  const result = await checkProgramBundleParity(options);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) {
    process.exitCode = 1;
  }
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
