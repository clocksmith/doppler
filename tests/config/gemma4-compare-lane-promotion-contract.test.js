import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'compare-engines.config.json');
const EVIDENCE_PATH = path.join(
  REPO_ROOT,
  'benchmarks',
  'vendors',
  'fixtures',
  'gemma4-e2b-sky-t032-chat.compare.json'
);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const compareConfig = readJson(COMPARE_CONFIG_PATH);
const profile = (Array.isArray(compareConfig.modelProfiles) ? compareConfig.modelProfiles : [])
  .find((entry) => entry?.dopplerModelId === 'gemma-4-e2b-it-q4k-ehf16-af32') || null;

assert.ok(profile, 'compare config must include gemma-4-e2b-it-q4k-ehf16-af32');
assert.equal(profile.compareLane, 'performance_comparable');
assert.match(
  String(profile.compareLaneReason ?? ''),
  /not correctness-parity claims/,
  'Gemma 4 E2B compare lane must keep its non-correctness claim reason'
);
assert.equal(fs.existsSync(EVIDENCE_PATH), true, 'Gemma 4 E2B compare promotion requires a committed compare fixture');

const evidence = readJson(EVIDENCE_PATH);
assert.equal(evidence.compareLane?.declared, 'performance_comparable');
assert.equal(evidence.compareLane?.reason, profile.compareLaneReason);
assert.equal(evidence.dopplerModelId, profile.dopplerModelId);
assert.equal(evidence.tjsModelId, profile.defaultTjsModelId);
assert.equal(evidence.dopplerSurface, 'browser');
assert.equal(evidence.decodeProfile, 'parity');
assert.equal(evidence.methodology?.promptParity?.dopplerChatTemplateEnabled, true);
assert.equal(evidence.correctness?.status, 'mismatch');
assert.ok(
  Number.isInteger(evidence.correctness?.tokenMatch?.matchingPrefixTokens)
  && evidence.correctness.tokenMatch.matchingPrefixTokens >= 4,
  'Gemma 4 E2B compare fixture must preserve a stable shared prefix before divergence'
);
assert.ok(
  Number(evidence.sections?.compute?.parity?.doppler?.result?.timing?.decodeTokensPerSec) >
    Number(evidence.sections?.compute?.parity?.transformersjs?.timing?.decodeTokensPerSec),
  'Gemma 4 E2B parity evidence must show Doppler ahead on decode throughput'
);

console.log('gemma4-compare-lane-promotion-contract.test: ok');
