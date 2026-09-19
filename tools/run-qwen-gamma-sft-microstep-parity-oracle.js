#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  createQwenSftBackendParityFixture,
} from '../src/experimental/training/qwen-sft-backend-parity-fixture.js';
import { runGammaWgslRequest } from './trainers/gamma-wgsl-trainer.js';
import { runBrowserOracle } from './lib/run-browser-oracle.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_OUTPUT = 'reports/training/native-parity/qwen-gamma-sft-microstep-parity-oracle.json';
const RUN_ROOT = 'reports/training/native-parity/qwen-gamma-sft-microstep-parity';
const GAMMA_SCRIPT = 'projects/distillation/wgsl/training/train_wgsl.py';

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

function git(cwd, args, options = {}) {
  try {
    return execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  } catch (error) {
    if (options.allowFailure) return null;
    throw error;
  }
}

function hasRelevantGammaDiff(gammaRoot) {
  const status = execFileSync(
    'git',
    ['status', '--porcelain', '--', GAMMA_SCRIPT, 'tests/test_wgsl_training_protocol.py'],
    { cwd: gammaRoot, encoding: 'utf8' }
  ).trim();
  return status.length > 0;
}

function resolveGammaRoot() {
  if (process.env.GAMMA_ROOT) return path.resolve(process.env.GAMMA_ROOT);
  const commonGitDirectory = path.resolve(git(ROOT, ['rev-parse', '--git-common-dir']));
  const configuredWorktree = git(ROOT, ['config', '--get', 'core.worktree']);
  const dopplerRoot = configuredWorktree
    ? path.resolve(commonGitDirectory, configuredWorktree)
    : ROOT;
  return path.join(path.dirname(dopplerRoot), 'gamma');
}

async function main() {
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log([
      'Usage: node tools/run-qwen-gamma-sft-microstep-parity-oracle.js [--output <path>]',
      '',
      'Runs the frozen rank-32 microstep through Gamma PyTorch/ROCm and',
      'Doppler WebGPU/Vulkan, then compares loss, gradients, AdamW state, and updates.',
    ].join('\n'));
    return;
  }
  const runRoot = path.resolve(ROOT, RUN_ROOT);
  const gammaRoot = resolveGammaRoot();
  process.env.GAMMA_ROOT = gammaRoot;
  const fixturePath = path.join(runRoot, 'fixture.json');
  const gammaRunRoot = path.join(runRoot, 'gamma');
  await mkdir(runRoot, { recursive: true });
  const fixture = createQwenSftBackendParityFixture({ rank: 32, alpha: 64 });
  const fixtureBytes = new TextEncoder().encode(`${JSON.stringify(fixture, null, 2)}\n`);
  await writeFile(fixturePath, fixtureBytes);
  const request = {
    protocol: 'gamma_wgsl_trainer_json_v1',
    action: 'parity_microstep',
    runId: 'qwen35-9b-rank32-backend-parity',
    outputRoot: gammaRunRoot,
    fixturePath,
    training: { dtype: 'float32' },
  };
  const gamma = await runGammaWgslRequest(request, {
    runRoot: gammaRunRoot,
    prefix: 'parity-microstep',
  });
  const fixtureSha256 = sha256(fixtureBytes);
  if (gamma.response.result.fixtureSha256 !== fixtureSha256) {
    throw new Error('Gamma response fixture hash does not match the Doppler fixture bytes.');
  }

  const gammaScriptBytes = await readFile(path.join(gammaRoot, GAMMA_SCRIPT));
  const gammaIdentity = {
    repository: 'gamma',
    sourceRevision: git(gammaRoot, ['rev-parse', 'HEAD']),
    worktreeDirty: git(gammaRoot, ['status', '--porcelain']).length > 0,
    relevantFilesDirty: hasRelevantGammaDiff(gammaRoot),
    sourcePath: GAMMA_SCRIPT,
    sourceSha256: sha256(gammaScriptBytes),
    requestHash: gamma.response.requestHash,
    responsePath: path.relative(ROOT, gamma.paths.responsePath),
    runtime: gamma.response.runtime,
  };
  if (gammaIdentity.relevantFilesDirty) {
    throw new Error('Gamma parity protocol files are dirty; commit them before sealing evidence.');
  }

  const browserArgs = process.argv.slice(2);
  await runBrowserOracle({
    argv: browserArgs,
    root: ROOT,
    defaultOutput: DEFAULT_OUTPUT,
    modulePath: 'tests/training/browser/qwen-gamma-sft-microstep-parity-oracle.js',
    exportName: 'runQwenGammaSftMicrostepParityOracle',
    oracleArgs: {
      gammaReference: gamma.response.result,
      gammaIdentity,
    },
    sourcePaths: {
      microstep: 'src/experimental/training/qwen-hybrid-sft-microstep.js',
      fixture: 'src/experimental/training/qwen-sft-backend-parity-fixture.js',
      parityOracle: 'tests/training/browser/qwen-gamma-sft-microstep-parity-oracle.js',
      runner: 'tools/run-qwen-gamma-sft-microstep-parity-oracle.js',
      gammaBridge: 'tools/trainers/gamma-wgsl-trainer.js',
    },
  });
}

main().catch((error) => {
  console.error(error?.stack || error);
  process.exitCode = 1;
});
