#!/usr/bin/env node

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import {
  extractExecutionContractFacts,
  renderExecutionContractLeanModule,
  sanitizeLeanModuleName,
} from '../src/tooling/lean-execution-contract.js';

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

function resolveLeanBinary() {
  const elanLean = path.join(os.homedir(), '.elan', 'bin', 'lean');
  if (fs.existsSync(elanLean)) {
    return elanLean;
  }
  const probe = spawnSync('bash', ['-lc', 'command -v lean'], { encoding: 'utf8' });
  if (probe.status === 0) {
    const resolved = probe.stdout.trim();
    if (resolved) {
      return resolved;
    }
  }
  throw new Error('lean binary not found. Install Lean with elan first.');
}

function runLeanCommand({ leanBin, toolchainRef, buildDir, rootDir, sourcePath, outputPath }) {
  const result = spawnSync(
    leanBin,
    [`+${toolchainRef}`, '-o', outputPath, sourcePath],
    {
      cwd: rootDir,
      encoding: 'utf8',
      env: {
        ...process.env,
        LEAN_PATH: `${buildDir}:${path.join(rootDir, 'lean')}`,
      },
    }
  );
  if (result.stdout) {
    process.stdout.write(result.stdout);
  }
  if (result.stderr) {
    process.stderr.write(result.stderr);
  }
  if (result.status !== 0) {
    throw new Error(`lean exited with status ${result.status}`);
  }
}

function runLeanCheck({ sourcePath, rootDir }) {
  const toolchainVersion = process.env.DOPPLER_LEAN_VERSION ?? '4.16.0';
  const toolchainRef = toolchainVersion.startsWith('v')
    ? `leanprover/lean4:${toolchainVersion}`
    : `leanprover/lean4:v${toolchainVersion}`;
  const leanBin = resolveLeanBinary();
  const buildDir = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-lean-execution-contract-'));
  try {
    fs.mkdirSync(path.join(buildDir, 'Doppler'), { recursive: true });
    runLeanCommand({
      leanBin,
      toolchainRef,
      buildDir,
      rootDir,
      sourcePath: path.join(rootDir, 'lean', 'Doppler', 'Model.lean'),
      outputPath: path.join(buildDir, 'Doppler', 'Model.olean'),
    });
    runLeanCommand({
      leanBin,
      toolchainRef,
      buildDir,
      rootDir,
      sourcePath: path.join(rootDir, 'lean', 'Doppler', 'ExecutionContract.lean'),
      outputPath: path.join(buildDir, 'Doppler', 'ExecutionContract.olean'),
    });
    const generatedOutput = path.join(buildDir, 'GeneratedExecutionContractCheck.olean');
    const result = spawnSync(
      leanBin,
      [`+${toolchainRef}`, '-o', generatedOutput, sourcePath],
      {
        cwd: rootDir,
        encoding: 'utf8',
        env: {
          ...process.env,
          LEAN_PATH: `${buildDir}:${path.join(rootDir, 'lean')}`,
        },
      }
    );
    if (result.stdout) {
      process.stdout.write(result.stdout);
    }
    if (result.stderr) {
      process.stderr.write(result.stderr);
    }
    if (result.status !== 0) {
      throw new Error(`lean exited with status ${result.status}`);
    }
    const overallMatch = result.stdout.match(/executionContractOverall:(pass|fail)/);
    if (!overallMatch) {
      throw new Error('unable to parse executionContractOverall from Lean output.');
    }
    return { ok: overallMatch[1] === 'pass', toolchainRef };
  } finally {
    fs.rmSync(buildDir, { recursive: true, force: true });
  }
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

try {
  const args = parseArgs(process.argv.slice(2));
  const manifestText = fs.readFileSync(path.resolve(rootDir, args.manifestPath), 'utf8');
  const manifest = JSON.parse(manifestText);
  const facts = extractExecutionContractFacts(manifest);
  const moduleName = sanitizeLeanModuleName(args.moduleName ?? `${facts.modelId}_ExecutionContractCheck`);
  const source = renderExecutionContractLeanModule(facts, { moduleName });

  const tempDir = args.emitPath
    ? null
    : fs.mkdtempSync(path.join(rootDir, 'lean', '.generated-'));
  const generatedPath = args.emitPath
    ? path.resolve(rootDir, args.emitPath)
    : path.join(tempDir, `${moduleName}.lean`);

  fs.mkdirSync(path.dirname(generatedPath), { recursive: true });
  fs.writeFileSync(generatedPath, source);

  console.log(`lean-execution-contract: wrote ${path.relative(rootDir, generatedPath)}`);

  try {
    if (args.check) {
      const result = runLeanCheck({ sourcePath: generatedPath, rootDir });
      if (!result.ok) {
        process.exitCode = 1;
        console.error(`lean-execution-contract: contract failed (${result.toolchainRef})`);
      } else {
        console.log(`lean-execution-contract: ok (${result.toolchainRef})`);
      }
    }
  } finally {
    if (!args.emitPath) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.startsWith('Unknown argument:') || message === '--manifest is required.') {
    usage();
  }
  console.error(message);
  process.exitCode = 1;
}
