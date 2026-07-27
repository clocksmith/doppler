#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { KERNEL_REF_CONTENT_DIGESTS } from '../src/config/kernels/kernel-ref-digests.js';
import kernelRegistry from '../src/config/kernels/registry.json' with { type: 'json' };
import { runBrowserCommandInNode } from '../src/tooling/node-browser-command-runner.js';
import { runNodeCommand } from '../src/tooling/node-command-runner.js';
import {
  runRegisteredVariantCalibrationJob,
} from '../src/tooling/registered-variant-calibration-job.js';

function parseArgs(argv) {
  const options = { jobPath: null, outputPath: null };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    const value = argv[index + 1];
    if ((token === '--job' || token === '--out') && (!value || value.startsWith('--'))) {
      throw new Error(`${token} requires a path`);
    }
    if (token === '--job') {
      options.jobPath = resolve(value);
      index += 1;
    } else if (token === '--out') {
      options.outputPath = resolve(value);
      index += 1;
    } else {
      throw new Error(`Unsupported argument: ${token}`);
    }
  }
  if (!options.jobPath || !options.outputPath) {
    throw new Error(
      'Usage: node tools/calibrate-registered-variants.js --job <job.json> --out <receipt.json>'
    );
  }
  return options;
}

const options = parseArgs(process.argv.slice(2));
const job = JSON.parse(await readFile(options.jobPath, 'utf8'));
const runCommand = job.surface === 'browser'
  ? runBrowserCommandInNode
  : runNodeCommand;
const executionEngine = job.surface === 'browser'
  ? `playwright:${job.commandOptions?.browser?.channel ?? 'configured-default'}`
  : typeof globalThis.Bun?.version === 'string'
    ? `bun:${globalThis.Bun.version}`
    : `node:${process.versions.node}`;
const receipt = await runRegisteredVariantCalibrationJob(job, {
  registry: kernelRegistry,
  kernelDigests: KERNEL_REF_CONTENT_DIGESTS,
  runCommand,
  executionEngine,
  onEvent(event) {
    if (event.type === 'candidate:start' || event.type === 'candidate:complete') {
      console.log(JSON.stringify(event));
    }
  },
});
await mkdir(dirname(options.outputPath), { recursive: true });
await writeFile(options.outputPath, `${JSON.stringify(receipt, null, 2)}\n`);
console.log(
  `registered calibration: wrote ${options.outputPath}; ` +
  `${receipt.proposedSelections.length} selection(s) require manual promotion`
);
