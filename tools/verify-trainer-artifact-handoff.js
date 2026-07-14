#!/usr/bin/env node

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  buildTrainerArtifactImportPlan,
  buildTrainerArtifactParityTemplate,
  verifyTrainerArtifactParityEvidence,
} from '../src/experimental/bridge/trainer-artifact-bridge.js';
import {
  importTrainerArtifactHandoff,
  verifyTrainerArtifactHandoff,
} from '../src/tooling/trainer-artifact-handoff.js';

const COMMANDS = new Set(['verify', 'plan', 'import', 'parity-template', 'parity-check']);

function parseRepositoryRoot(value) {
  const text = String(value || '').trim();
  const delimiter = text.indexOf('=');
  if (delimiter < 1 || delimiter === text.length - 1) {
    throw new Error('--repo-root must use <repository>=<path>.');
  }
  return [text.slice(0, delimiter), text.slice(delimiter + 1)];
}

function parseArgs(argv) {
  const args = {
    command: 'verify',
    contractPath: '',
    repositoryRoots: {},
    evidencePath: '',
    outPath: '',
    verifiedAt: undefined,
    help: false,
  };
  let index = 0;
  if (COMMANDS.has(argv[0])) {
    args.command = argv[0];
    index = 1;
  }
  for (; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--contract') {
      args.contractPath = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--repo-root') {
      const [repository, root] = parseRepositoryRoot(argv[index + 1]);
      args.repositoryRoots[repository] = root;
      index += 1;
      continue;
    }
    if (token === '--evidence') {
      args.evidencePath = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--out') {
      args.outPath = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--verified-at') {
      args.verifiedAt = String(argv[index + 1] || '').trim();
      index += 1;
      continue;
    }
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

function usage() {
  return [
    'Usage:',
    '  node tools/verify-trainer-artifact-handoff.js <verify|plan|import|parity-template|parity-check> \\',
    '    --contract <handoff.json> --repo-root <repository>=<path> [--repo-root ...] \\',
    '    [--verified-at <ISO-8601>] [--evidence <parity-evidence.json>] [--out <result.json>]',
  ].join('\n');
}

async function readJson(filePath, label) {
  if (!filePath) throw new Error(`${label} is required.`);
  return JSON.parse(await fs.readFile(path.resolve(filePath), 'utf8'));
}

async function emit(result, outPath) {
  const json = `${JSON.stringify(result, null, 2)}\n`;
  if (outPath) {
    const resolved = path.resolve(outPath);
    await fs.mkdir(path.dirname(resolved), { recursive: true });
    await fs.writeFile(resolved, json, 'utf8');
  }
  process.stdout.write(json);
}

export async function runTrainerArtifactHandoffTool(args) {
  const common = {
    contractPath: args.contractPath,
    repositoryRoots: args.repositoryRoots,
    ...(args.verifiedAt ? { verifiedAt: args.verifiedAt } : {}),
  };
  if (args.command === 'import') {
    const result = await importTrainerArtifactHandoff(common);
    return {
      descriptor: result.descriptor,
      verification: result.verification,
      plan: result.plan,
      importReceipt: result.receipt,
    };
  }
  const verification = await verifyTrainerArtifactHandoff(common);
  if (args.command === 'verify') {
    return { descriptor: verification.descriptor, verification: verification.receipt };
  }
  if (!verification.receipt.ok) {
    throw new Error('trainer artifact handoff: identity verification failed.');
  }
  if (args.command === 'plan') {
    return {
      descriptor: verification.descriptor,
      verification: verification.receipt,
      plan: buildTrainerArtifactImportPlan(verification.descriptor, verification.receipt),
    };
  }
  if (args.command === 'parity-template') {
    return {
      descriptor: verification.descriptor,
      verification: verification.receipt,
      evidence: buildTrainerArtifactParityTemplate(verification.descriptor, verification.receipt),
    };
  }
  if (args.command === 'parity-check') {
    const evidence = await readJson(args.evidencePath, '--evidence');
    return {
      descriptor: verification.descriptor,
      verification: verification.receipt,
      parity: verifyTrainerArtifactParityEvidence(
        verification.descriptor,
        verification.receipt,
        evidence
      ),
    };
  }
  throw new Error(`Unsupported command: ${args.command}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return;
  }
  if (!args.contractPath) throw new Error('--contract is required.');
  if (Object.keys(args.repositoryRoots).length === 0) {
    throw new Error('At least one --repo-root is required.');
  }
  const result = await runTrainerArtifactHandoffTool(args);
  await emit(result, args.outPath);
  const decision = result.verification?.ok === false || result.parity?.decision === 'block';
  if (decision) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
