#!/usr/bin/env node
// Evaluate a routed-MoE run against doppler.routing-aware-acceptance/v1.
//
// Usage: node tools/evaluate-routing-aware-acceptance.js --evidence EVIDENCE.json [--out VERDICT.json]
//        [--policy tools/policies/routing-aware-acceptance.json]
//        node tools/evaluate-routing-aware-acceptance.js --policy-hash [POLICY.json]
//
// Fail-closed: an unreadable, hash-mismatched or absent input becomes a reason and the
// verdict is 'fail'. Exit code 0 only for 'pass'. Never runs a GPU.
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { isPlainObject } from '../src/formats/plain-object.js';
import { EVIDENCE_SCHEMA, evaluateRoutingAwareAcceptance, hashPolicy } from './lib/routing-aware-acceptance.js';

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const DEFAULT_POLICY = path.join(REPO_ROOT, 'tools', 'policies', 'routing-aware-acceptance.json');
const FIXTURE_SCHEMA = 'doppler.flash-next-routing-selection-fixture/v1';

const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

async function readHashed(baseDir, ref, label) {
  if (!isPlainObject(ref) || typeof ref.file !== 'string' || typeof ref.sha256 !== 'string') {
    throw new Error(`${label}: {file, sha256} reference required`);
  }
  const bytes = await fs.readFile(path.resolve(baseDir, ref.file));
  if (sha256(bytes) !== ref.sha256) throw new Error(`${label}: SHA-256 mismatch`);
  return bytes;
}

function typed(bytes, Type, length, label) {
  if (bytes.byteLength !== length * Type.BYTES_PER_ELEMENT) throw new Error(`${label}: size mismatch`);
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return new Type(copy.buffer);
}

async function loadCpuFixture(baseDir, item) {
  const fixturePath = path.resolve(baseDir, item.cpu.fixture);
  const fixture = JSON.parse(await fs.readFile(fixturePath, 'utf8'));
  if (fixture.schema !== FIXTURE_SCHEMA) throw new Error(`CPU fixture schema must be ${FIXTURE_SCHEMA}`);
  for (const [key, field] of [['numLayers', 'layers'], ['numTokens', 'numTokens'], ['numExperts', 'numExperts'], ['topK', 'topK']]) {
    if (fixture[field] !== item[key]) throw new Error(`CPU fixture ${field} differs from the case`);
  }
  if (fixture.sequenceId !== item.caseId) throw new Error('CPU fixture names a different case');
  const dir = path.dirname(fixturePath);
  const experts = [], logits = [];
  for (const [layer, entry] of fixture.files.entries()) {
    if (entry.layer !== layer) throw new Error('CPU fixture layers out of order');
    experts.push(typed(await readHashed(dir, entry.experts, `cpu experts L${layer}`), Uint32Array, item.numTokens * item.topK, `cpu experts L${layer}`));
    logits.push(typed(await readHashed(dir, entry.logits, `cpu logits L${layer}`), Float32Array, item.numTokens * item.numExperts, `cpu logits L${layer}`));
  }
  if (!logits.every(row => row.every(Number.isFinite))) throw new Error('CPU router logits are not finite');
  return { producerSha256: fixture.producerSha256, artifactManifestSha256: fixture.manifestSha256, experts, logits };
}

async function loadUnforced(baseDir, item) {
  const { layers, ...rest } = item.unforced;
  if (!Array.isArray(layers) || layers.length !== item.numLayers) throw new Error('unforced layers must cover every layer');
  const experts = [];
  for (const [layer, entry] of layers.entries()) {
    if (entry?.layer !== layer) throw new Error('unforced layers out of order');
    experts.push(typed(await readHashed(baseDir, entry.experts, `gpu experts L${layer}`), Uint32Array, item.numTokens * item.topK, `gpu experts L${layer}`));
  }
  return { ...rest, experts };
}

// Returns the in-memory bundle and load failures; a failed part becomes null so the
// evaluator reports it as missing.
export async function loadEvidenceBundle(evidencePath) {
  const loadFailures = [];
  let evidence;
  try {
    evidence = JSON.parse(await fs.readFile(evidencePath, 'utf8'));
  } catch (error) {
    return { evidence: null, loadFailures: [{ code: 'EVIDENCE_UNREADABLE', message: error.message }] };
  }
  if (evidence?.schema !== EVIDENCE_SCHEMA) return { evidence, loadFailures };
  const baseDir = path.dirname(evidencePath);
  const cases = [];
  for (const item of Array.isArray(evidence.cases) ? evidence.cases : []) {
    const loaded = { ...item };
    for (const [key, load] of [['cpu', loadCpuFixture], ['unforced', loadUnforced]]) {
      if (!isPlainObject(item?.[key])) continue;
      try {
        loaded[key] = await load(baseDir, item);
      } catch (error) {
        loaded[key] = null;
        loadFailures.push({ code: 'INPUT_UNREADABLE', caseId: item.caseId ?? null, message: `${key}: ${error.message}` });
      }
    }
    cases.push(loaded);
  }
  let taskQuality = null;
  if (evidence.taskQuality !== null && evidence.taskQuality !== undefined) {
    try {
      taskQuality = JSON.parse((await readHashed(baseDir, evidence.taskQuality, 'task-quality results')).toString('utf8'));
    } catch (error) {
      loadFailures.push({ code: 'INPUT_UNREADABLE', message: `taskQuality: ${error.message}` });
    }
  }
  return { evidence: { ...evidence, cases, taskQuality }, loadFailures };
}

// repoRoot resolves the policy's prompt-set and tolerance paths (tests pass a temporary root).
export async function runEvaluation({ evidencePath, policyPath = DEFAULT_POLICY, repoRoot = REPO_ROOT }) {
  const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
  const readRepoJson = async relative => {
    try {
      return JSON.parse(await fs.readFile(path.join(repoRoot, relative), 'utf8'));
    } catch {
      return null;
    }
  };
  const promptSet = await readRepoJson(policy.taskQuality?.promptSet?.path ?? '');
  const tolerance = await readRepoJson(policy.numericalTolerance?.policyPath ?? '');
  const { evidence, loadFailures } = await loadEvidenceBundle(evidencePath);
  const verdict = evaluateRoutingAwareAcceptance({ policy, promptSet,
    tolerancePolicies: tolerance?.tolerancePolicies ?? null, evidence });
  if (loadFailures.length) {
    verdict.reasons.unshift(...loadFailures);
    verdict.verdict = 'fail';
    verdict.qualifiedTuple = null;
  }
  return verdict;
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 2) {
    const [flag, value] = [argv[i], argv[i + 1]];
    if (!['--evidence', '--out', '--policy'].includes(flag) || !value) {
      throw new Error('Usage: evaluate-routing-aware-acceptance.js --evidence EVIDENCE.json [--out VERDICT.json] [--policy POLICY.json]');
    }
    args[flag.slice(2)] = path.resolve(value);
  }
  if (!args.evidence) throw new Error('--evidence is required');
  return args;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href && process.argv[2] === '--policy-hash') {
  // Prints the canonical policy hash that evidence and task-quality results must bind.
  const policyPath = path.resolve(process.argv[3] ?? DEFAULT_POLICY);
  console.log(hashPolicy(JSON.parse(await fs.readFile(policyPath, 'utf8'))));
} else if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = parseArgs(process.argv.slice(2));
  const verdict = await runEvaluation({ evidencePath: args.evidence, policyPath: args.policy ?? DEFAULT_POLICY });
  const text = `${JSON.stringify(verdict, null, 2)}\n`;
  if (args.out) await fs.writeFile(args.out, text, { flag: 'wx' });
  console.log(JSON.stringify({ verdict: verdict.verdict, reasons: verdict.reasons.length,
    codes: [...new Set(verdict.reasons.map(entry => entry.code))] }));
  process.exitCode = verdict.verdict === 'pass' ? 0 : 1;
}
