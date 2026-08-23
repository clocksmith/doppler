#!/usr/bin/env node

/** Materialize a deterministic Release-to-JavaScript receipt from workspace bytes. */

import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { createReleaseToJavaScriptReceipt } from '../src/converter/release-to-javascript-receipt.js';
import { stableSortObject } from '../src/utils/stable-sort-object.js';

export function usage() {
  return [
    'Doppler Forge: Release-to-JavaScript receipt materializer',
    '',
    'Usage:',
    '  node tools/create-release-to-javascript-receipt.js --spec <path> --out <path>',
    '',
    'The spec declares acceptedCode.files as workspace-relative paths, qualification.packPath,',
    'and evidence entries as { kind, path }. The tool hashes every referenced byte itself.',
  ].join('\n');
}

export function parseArgs(argv) {
  const flags = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      flags.help = true;
      continue;
    }
    if (token === '--json') {
      flags.json = true;
      continue;
    }
    if (token !== '--spec' && token !== '--out') throw new Error(`Unsupported argument "${token}".`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${token}.`);
    flags[token.slice(2)] = value;
    index += 1;
  }
  return flags;
}

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value;
}

function workspaceFile(repoRoot, filePath, label) {
  if (typeof filePath !== 'string' || !filePath.trim() || path.isAbsolute(filePath)) {
    throw new Error(`${label} must be a workspace-relative path.`);
  }
  const normalized = path.normalize(filePath);
  if (normalized === '..' || normalized.startsWith(`..${path.sep}`)) {
    throw new Error(`${label} must remain inside the workspace.`);
  }
  return { path: filePath, absolutePath: path.resolve(repoRoot, normalized) };
}

async function sha256File(filePath) {
  const bytes = await fs.readFile(filePath);
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

function canonicalDigest(value) {
  const bytes = JSON.stringify(stableSortObject(value));
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

async function fileEvidence(repoRoot, filePaths, label) {
  if (!Array.isArray(filePaths) || filePaths.length < 1) {
    throw new Error(`${label} must be a non-empty array of workspace-relative paths.`);
  }
  const uniquePaths = new Set(filePaths);
  if (uniquePaths.size !== filePaths.length) throw new Error(`${label} paths must be unique.`);
  return Promise.all([...filePaths].sort((left, right) => left.localeCompare(right)).map(async (filePath) => {
    const file = workspaceFile(repoRoot, filePath, label);
    return { path: file.path, digest: await sha256File(file.absolutePath) };
  }));
}

async function writeJsonAtomic(outputPath, value) {
  const resolved = path.resolve(outputPath);
  await fs.mkdir(path.dirname(resolved), { recursive: true });
  const temporary = `${resolved}.${process.pid}.tmp`;
  await fs.writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await fs.rename(temporary, resolved);
  return resolved;
}

export async function materializeReleaseToJavaScriptReceipt(spec, options = {}) {
  requireObject(spec, 'Release-to-JavaScript spec');
  const repoRoot = path.resolve(options.repoRoot ?? process.cwd());
  const acceptedCodeSpec = requireObject(spec.acceptedCode, 'acceptedCode');
  const acceptedFiles = await fileEvidence(repoRoot, acceptedCodeSpec.files, 'acceptedCode.files');
  const acceptedCode = {
    revision: acceptedCodeSpec.revision,
    files: acceptedFiles,
  };
  acceptedCode.digest = canonicalDigest(acceptedCode);

  const qualificationSpec = requireObject(spec.qualification, 'qualification');
  const packFile = workspaceFile(repoRoot, qualificationSpec.packPath, 'qualification.packPath');
  const evidenceSpec = spec.evidence;
  if (!Array.isArray(evidenceSpec) || evidenceSpec.length < 1) {
    throw new Error('evidence must be a non-empty array.');
  }
  const evidence = await Promise.all(evidenceSpec.map(async (entry, index) => {
    requireObject(entry, `evidence[${index}]`);
    if (typeof entry.kind !== 'string' || !entry.kind.trim()) {
      throw new Error(`evidence[${index}].kind must be a non-empty string.`);
    }
    const file = workspaceFile(repoRoot, entry.path, `evidence[${index}].path`);
    return { kind: entry.kind, path: file.path, digest: await sha256File(file.absolutePath) };
  }));

  return createReleaseToJavaScriptReceipt({
    ...structuredClone(spec),
    acceptedCode,
    qualification: {
      status: qualificationSpec.status,
      packId: qualificationSpec.packId,
      packDigest: await sha256File(packFile.absolutePath),
    },
    evidence,
  });
}

export async function createReceiptFromFiles(options) {
  if (!options?.specPath) throw new Error('Release receipt materialization requires --spec.');
  if (!options?.outputPath) throw new Error('Release receipt materialization requires --out.');
  const specPath = path.resolve(options.specPath);
  const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
  const receipt = await materializeReleaseToJavaScriptReceipt(spec, options);
  const outputPath = await writeJsonAtomic(options.outputPath, receipt);
  return { receipt, outputPath };
}

export async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  if (flags.help) {
    console.log(usage());
    return;
  }
  const result = await createReceiptFromFiles({
    specPath: flags.spec,
    outputPath: flags.out,
  });
  if (flags.json) {
    console.log(JSON.stringify({
      ok: true,
      campaignId: result.receipt.campaignId,
      receiptDigest: result.receipt.receiptDigest,
      outputPath: path.relative(process.cwd(), result.outputPath),
    }, null, 2));
    return;
  }
  console.log('✔ Release-to-JavaScript receipt materialized');
  console.log(`  Campaign: ${result.receipt.campaignId}`);
  console.log(`  Digest:   ${result.receipt.receiptDigest}`);
  console.log(`  Output:   ${path.relative(process.cwd(), result.outputPath)}`);
}

function isMainModule(metaUrl) {
  const entryPath = process.argv[1];
  return entryPath && path.resolve(fileURLToPath(metaUrl)) === path.resolve(entryPath);
}

if (isMainModule(import.meta.url)) {
  main().catch((error) => {
    console.error(`[doppler-release-receipt] ${error.message}`);
    process.exit(1);
  });
}
