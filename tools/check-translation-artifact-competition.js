#!/usr/bin/env node

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { evaluateTranslationArtifactCompetition } from '../src/tooling/translation-artifact-competition.js';

function parseArgs(argv) {
  const args = {
    policyPath: 'tools/policies/translation-artifact-competition.json',
    handoffPath: '',
    verificationPath: '',
    allowBlocked: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') args.policyPath = argv[++index] || '';
    else if (token === '--handoff') args.handoffPath = argv[++index] || '';
    else if (token === '--verification') args.verificationPath = argv[++index] || '';
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

export async function runTranslationArtifactCompetitionGate(args) {
  const policy = await readJson(args.policyPath);
  const handoff = args.handoffPath ? await readJson(args.handoffPath) : null;
  const verificationReceipt = args.verificationPath
    ? await readJson(args.verificationPath)
    : null;
  return evaluateTranslationArtifactCompetition({
    policy,
    handoff,
    handoffSha256: args.handoffPath ? await sha256File(args.handoffPath) : null,
    verificationReceipt,
  });
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const receipt = await runTranslationArtifactCompetitionGate(args);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
  if (receipt.decision === 'blocked' && !args.allowBlocked) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
