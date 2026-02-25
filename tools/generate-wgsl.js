#!/usr/bin/env node

import path from 'node:path';
import process from 'node:process';
import { generateWgslVariants } from './wgsl-variant-generator.js';

function usage() {
  return [
    'Usage:',
    '  node tools/generate-wgsl.js [--check] [--json] [--root <dir>]',
    '',
    'Options:',
    '  --check       Verify generated WGSL files are up-to-date without writing.',
    '  --json        Emit machine-readable report.',
    '  --root <dir>  Repository root (default: current working directory).',
  ].join('\n');
}

function parseArgs(argv) {
  const flags = {
    check: false,
    json: false,
    root: process.cwd(),
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = String(argv[index] ?? '');
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--check') {
      flags.check = true;
      continue;
    }
    if (token === '--json') {
      flags.json = true;
      continue;
    }
    if (token === '--root') {
      const value = argv[index + 1];
      if (!value) {
        throw new Error('--root requires a value');
      }
      flags.root = path.resolve(process.cwd(), String(value));
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return flags;
}

function printHumanReport(report) {
  const mode = report.checkOnly ? 'check' : 'generate';
  console.log(`[wgsl:${mode}] variants=${report.variantCount} changed=${report.changedCount} unchanged=${report.unchangedCount}`);
  if (report.changedTargets.length) {
    console.log('[wgsl] changed targets:');
    for (const target of report.changedTargets) {
      console.log(`- ${target}`);
    }
  }
  if (report.errors.length) {
    console.log('[wgsl] errors:');
    for (const message of report.errors) {
      console.log(`- ${message}`);
    }
  }
}

async function main() {
  const flags = parseArgs(process.argv.slice(2));
  if (flags.help) {
    console.log(usage());
    process.exit(0);
  }

  const report = await generateWgslVariants({
    rootDir: flags.root,
    checkOnly: flags.check,
  });

  if (flags.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    printHumanReport(report);
  }

  const failed = report.errors.length > 0 || (flags.check && report.changedCount > 0);
  process.exit(failed ? 1 : 0);
}

main().catch((error) => {
  console.error(`[wgsl] ${error.message}`);
  process.exit(1);
});
