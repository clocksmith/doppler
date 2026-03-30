import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const COMPARE_CONFIG_PATH = path.join(REPO_ROOT, 'benchmarks', 'vendors', 'compare-engines.config.json');
const QWEN_08_EVIDENCE_PATH = path.join(
  REPO_ROOT,
  'benchmarks',
  'vendors',
  'fixtures',
  'qwen3-5-0-8b-p064-d064-t0-k1.compare.json'
);

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const compareConfig = readJson(COMPARE_CONFIG_PATH);
const qwenProfiles = (Array.isArray(compareConfig.modelProfiles) ? compareConfig.modelProfiles : [])
  .filter((entry) => String(entry?.dopplerModelId ?? '').startsWith('qwen-3-5-'));

assert.ok(qwenProfiles.length >= 2, 'compare config must include the Qwen 3.5 compare profiles');

const evidenceByModelId = new Map([
  ['qwen-3-5-0-8b-q4k-ehaf16', QWEN_08_EVIDENCE_PATH],
]);

for (const profile of qwenProfiles) {
  const modelId = String(profile?.dopplerModelId ?? '');
  if (profile.compareLane !== 'performance_comparable') {
    continue;
  }
  const evidencePath = evidenceByModelId.get(modelId) ?? null;
  assert.ok(
    evidencePath,
    `${modelId}: Qwen performance_comparable lanes require a committed correctness-clean compare fixture`
  );
  assert.equal(fs.existsSync(evidencePath), true, `${modelId}: compare fixture must exist`);
}

{
  const profile = qwenProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-0-8b-q4k-ehaf16') || null;
  assert.ok(profile, 'compare config must include qwen-3-5-0-8b-q4k-ehaf16');
  assert.equal(profile.compareLane, 'performance_comparable');
  assert.equal(profile.compareLaneReason, null);

  const evidence = readJson(QWEN_08_EVIDENCE_PATH);
  assert.equal(evidence.compareLane?.declared, 'performance_comparable');
  assert.equal(evidence.dopplerModelId, profile.dopplerModelId);
  assert.equal(evidence.compareLane?.reason ?? null, null);
  assert.equal(evidence.workload?.id, 'p064-d064-t0-k1');
  assert.equal(evidence.dopplerSurface, 'browser');
  assert.equal(evidence.decodeProfile, 'parity');
  assert.equal(evidence.mode, 'all');
  assert.equal(evidence.dopplerModelSource?.source, 'quickstart-registry');
  assert.equal(evidence.correctness?.status, 'match');
  assert.equal(evidence.correctness?.exactMatch, true);
  assert.equal(evidence.correctness?.normalizedMatch, true);
  assert.equal(evidence.correctness?.tokenMatch?.firstMismatchTokenIndex, -1);
  assert.equal(evidence.methodology?.dopplerDecodeCadence?.disableMultiTokenDecode, true);
  assert.equal(evidence.methodology?.dopplerDecodeCadence?.speculationMode, 'none');
  assert.equal(typeof evidence.reproductionCommand, 'string');
  assert.ok(
    evidence.reproductionCommand.includes('tools/compare-engines.js'),
    'Qwen 0.8B compare fixture must preserve the compare command'
  );
}

{
  const profile = qwenProfiles.find((entry) => entry?.dopplerModelId === 'qwen-3-5-2b-q4k-ehaf16') || null;
  assert.ok(profile, 'compare config must include qwen-3-5-2b-q4k-ehaf16');
  assert.equal(profile.compareLane, 'capability_only');
  assert.match(
    String(profile.compareLaneReason ?? ''),
    /claimable compare lane/i,
    'Qwen 2B promotion must keep an explicit non-claimable reason until compare evidence exists'
  );
}

console.log('qwen-compare-lane-promotion-contract.test: ok');
