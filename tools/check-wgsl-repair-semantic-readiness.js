#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { evaluateWgslSemanticReadiness } from '../src/tooling/wgsl-repair-semantic-gate.js';

const DEFAULT_POLICY = 'tools/policies/wgsl-repair-v13-semantic-policy.json';

function parseArgs(argv) {
  const args = { policyPath: DEFAULT_POLICY, evidencePath: '', allowBlocked: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') args.policyPath = argv[++index] || '';
    else if (token === '--evidence') args.evidencePath = argv[++index] || '';
    else if (token === '--allow-blocked') args.allowBlocked = true;
    else throw new Error(`Unknown argument: ${token}`);
  }
  return args;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(path.resolve(filePath), 'utf8'));
}

async function sha256File(filePath) {
  const bytes = await fs.readFile(path.resolve(filePath));
  return createHash('sha256').update(bytes).digest('hex');
}

export async function runWgslSemanticReadinessGate(args) {
  const policy = await readJson(args.policyPath);
  const predecessorChecks = await Promise.all([
    sha256File(policy.predecessor.resultPath),
    sha256File(policy.predecessor.preservationVerificationPath),
  ]);
  const predecessorVerified = predecessorChecks[0] === policy.predecessor.resultSha256
    && predecessorChecks[1] === policy.predecessor.preservationVerificationSha256;
  const preservationReceipt = await readJson(policy.predecessor.preservationVerificationPath);
  const taskEvidence = args.evidencePath
    ? (await readJson(args.evidencePath)).tasks
    : [];
  return evaluateWgslSemanticReadiness({
    policy,
    predecessorVerified,
    preservationReceipt,
    taskEvidence,
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const receipt = await runWgslSemanticReadinessGate(args);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
  if (receipt.decision === 'blocked' && !args.allowBlocked) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
