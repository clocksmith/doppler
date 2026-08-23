#!/usr/bin/env node

/** Materialize a deterministic semantic lowerability audit without loading model weights. */

import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import { auditEntryPointLowerability } from '../src/converter/execution-candidate-forge.js';
import { sha256Hex } from '../src/utils/sha256.js';
import { stableSortObject } from '../src/utils/stable-sort-object.js';

function parseArgs(argv) {
  const options = { check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--check') {
      options.check = true;
      continue;
    }
    if (!['--model-ir-receipt', '--vocabulary', '--entry-point', '--out'].includes(token)) {
      throw new Error(`Unknown argument "${token}".`);
    }
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${token}.`);
    options[token.slice(2)] = value;
    index += 1;
  }
  for (const field of ['model-ir-receipt', 'vocabulary', 'entry-point', 'out']) {
    if (!options[field]) throw new Error(`Lowerability audit requires --${field}.`);
  }
  return options;
}

function digestBytes(bytes) {
  return `sha256:${sha256Hex(bytes)}`;
}

function digestValue(value) {
  return digestBytes(JSON.stringify(stableSortObject(value)));
}

async function readJson(filePath, label) {
  const resolved = path.resolve(filePath);
  const raw = await fs.readFile(resolved, 'utf8');
  const json = JSON.parse(raw);
  if (!json || typeof json !== 'object' || Array.isArray(json)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return { resolved, raw, json };
}

function relativePath(filePath) {
  const relative = path.relative(process.cwd(), filePath);
  return relative || '.';
}

export async function createLowerabilityAuditReceipt(options) {
  const modelIRFile = await readJson(options.modelIRReceiptPath, 'ModelIR receipt');
  const vocabularyFile = await readJson(options.vocabularyPath, 'lowering vocabulary');
  const modelIR = modelIRFile.json.modelIR;
  if (!modelIR || typeof modelIR !== 'object' || Array.isArray(modelIR)) {
    throw new Error('ModelIR receipt must contain modelIR.');
  }
  const audit = auditEntryPointLowerability({
    modelIR,
    entryPointId: options.entryPointId,
    vocabulary: vocabularyFile.json,
  });
  const core = {
    schema: 'doppler.lowerability-audit-receipt/v1',
    modelIREvidence: {
      path: relativePath(modelIRFile.resolved),
      digest: digestBytes(modelIRFile.raw),
    },
    vocabularyEvidence: {
      path: relativePath(vocabularyFile.resolved),
      digest: digestBytes(vocabularyFile.raw),
      vocabularyId: vocabularyFile.json.vocabularyId ?? null,
      scope: vocabularyFile.json.scope ?? null,
    },
    audit,
  };
  return { ...core, receiptDigest: digestValue(core) };
}

export async function materializeLowerabilityAudit(options) {
  const receipt = await createLowerabilityAuditReceipt(options);
  const outputPath = path.resolve(options.outputPath);
  const encoded = `${JSON.stringify(receipt, null, 2)}\n`;
  if (options.check) {
    const observed = await fs.readFile(outputPath, 'utf8');
    if (observed !== encoded) throw new Error(`Lowerability audit drifted: ${relativePath(outputPath)}.`);
    return { receipt, outputPath, checked: true };
  }
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, encoded, 'utf8');
  return { receipt, outputPath, checked: false };
}

export async function main(argv = process.argv.slice(2)) {
  const flags = parseArgs(argv);
  const result = await materializeLowerabilityAudit({
    modelIRReceiptPath: flags['model-ir-receipt'],
    vocabularyPath: flags.vocabulary,
    entryPointId: flags['entry-point'],
    outputPath: flags.out,
    check: flags.check,
  });
  console.log(JSON.stringify({
    ok: true,
    checked: result.checked,
    lowerable: result.receipt.audit.lowerable,
    receiptDigest: result.receipt.receiptDigest,
    outputPath: relativePath(result.outputPath),
  }, null, 2));
}

const entryPath = process.argv[1];
if (entryPath && path.resolve(fileURLToPath(import.meta.url)) === path.resolve(entryPath)) {
  main().catch((error) => {
    console.error(`[doppler-lowerability-audit] ${error.message}`);
    process.exit(1);
  });
}
