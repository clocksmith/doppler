#!/usr/bin/env node

import { execFile } from 'node:child_process';

function parseArgs(argv) {
  const parsed = {
    report: null,
    checkpoint: null,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--report') {
      parsed.report = argv[i + 1] || null;
      i += 1;
      continue;
    }
    if (arg === '--checkpoint') {
      parsed.checkpoint = argv[i + 1] || null;
      i += 1;
      continue;
    }
    throw new Error(`Unknown flag: ${arg}`);
  }
  if (!parsed.report) {
    throw new Error('Usage: node tools/distill-studio-diagnostics.mjs --report <report.json> [--checkpoint <checkpoint.json>]');
  }
  return parsed;
}

function runNodeScript(args) {
  return new Promise((resolve, reject) => {
    execFile('node', args, { cwd: process.cwd() }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`${error.message}\n${stderr || stdout}`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

async function main() {
  const args = parseArgs(process.argv);
  const checks = [];

  const provenanceArgs = ['tools/verify-training-provenance.mjs', '--report', args.report];
  if (args.checkpoint) {
    provenanceArgs.push('--checkpoint', args.checkpoint);
  }
  await runNodeScript(provenanceArgs);
  checks.push({ name: 'provenance-report', ok: true });

  console.log(JSON.stringify({
    ok: true,
    checks,
  }, null, 2));
}

await main();
