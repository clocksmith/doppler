// Promotion rule for Qwen 3.5 compare lanes: a profile may declare
// performance_comparable only with a committed compare fixture for the same
// model that is correctness-clean on the parity decode profile and whose
// fairness verdict is consistent with its own measurements. Values are checked
// against the compare config, the claim matrix and the fixture's own fields,
// not copied literals.
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const FIXTURE_DIR = 'benchmarks/vendors/fixtures';
// Observations bind the frozen pre-capsule compare config, not the current one.
const FROZEN_COMPARE_CONFIG = 'tests/fixtures/pre-capsule/benchmarks/vendors/compare-engines.config.json';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

const compareConfig = readJson('benchmarks/vendors/compare-engines.config.json');
const claimMatrix = readJson('benchmarks/vendors/local-inference-claim-matrix.json');
const runContract = claimMatrix.sharedRunContract;

const fixtures = fs.readdirSync(FIXTURE_DIR)
  .filter((name) => name.endsWith('.compare.json'))
  .map((name) => ({ name, evidence: readJson(path.join(FIXTURE_DIR, name)) }));

function promotionViolations(profile, evidence) {
  const problems = [];
  const id = profile.dopplerModelId;
  if (evidence.dopplerModelId !== id) problems.push('model id mismatch');
  if (evidence.compareLane?.declared !== profile.compareLane) problems.push('declared lane differs from profile');
  if ((evidence.compareLane?.reason ?? null) !== (profile.compareLaneReason ?? null)) problems.push('lane reason differs from profile');
  if (evidence.decodeProfile !== 'parity') problems.push('promotion evidence must use the parity decode profile');
  if (evidence.dopplerSurface !== profile.defaultDopplerSurface) problems.push('surface differs from profile default');
  if (evidence.dopplerModelSource?.source !== profile.defaultDopplerSource) problems.push('model source differs from profile default');
  if (!runContract.workloads.includes(evidence.workload?.id)) problems.push('workload not in claim-matrix run contract');
  const correctness = evidence.correctness ?? {};
  if (correctness.status !== 'match' || correctness.exactMatch !== true || correctness.normalizedMatch !== true
    || correctness.tokenMatch?.firstMismatchTokenIndex !== -1) {
    problems.push('correctness is not clean');
  }
  if (correctness.doppler !== undefined && correctness.transformersjs !== undefined
    && correctness.exactMatch === true && correctness.doppler !== correctness.transformersjs) {
    problems.push('exactMatch claimed for differing outputs');
  }
  const fairness = evidence.fairness;
  if (fairness?.claimGrade === true) {
    if (!(evidence.runs >= runContract.minTimedSamplesForPercentiles)) problems.push('claim-grade evidence below claim-matrix sample count');
    if (fairness.sections?.[fairness.primarySection]?.claimGrade !== true) problems.push('primary section is not claim grade');
    if (fairness.releaseClaimable !== (evidence.dopplerModelSource?.source === 'quickstart-registry')) {
      problems.push('releaseClaimable must mean hosted quickstart-registry artifact');
    }
    if (fairness.localComparable !== !fairness.releaseClaimable) problems.push('localComparable must be the non-hosted claim-grade case');
  } else if (fairness && (fairness.releaseClaimable || fairness.localComparable)) {
    problems.push('non-claim-grade evidence cannot be claimable');
  }
  const cadenceGate = evidence.sections?.compute?.throughputCadenceGate;
  if (cadenceGate && cadenceGate.ok !== (cadenceGate.invalidReasons?.length === 0)) problems.push('cadence gate verdict disagrees with its reasons');
  if (evidence.reproductionCommand !== undefined && !String(evidence.reproductionCommand).includes('tools/compare-engines.js')) {
    problems.push('reproduction command must invoke tools/compare-engines.js');
  }
  return problems;
}

const qwenProfiles = compareConfig.modelProfiles
  .filter((profile) => String(profile?.dopplerModelId ?? '').startsWith('qwen-3-5-'));
assert.ok(qwenProfiles.length > 0, 'compare config must include Qwen 3.5 profiles');

let promoted = 0;
for (const profile of qwenProfiles) {
  if (profile.compareLane !== 'performance_comparable') continue;
  promoted += 1;
  assert.equal(profile.compareLaneReason, null, `${profile.dopplerModelId}: performance_comparable lanes carry no blocking reason`);
  const candidates = fixtures.filter(({ evidence }) => evidence.dopplerModelId === profile.dopplerModelId);
  assert.ok(candidates.length > 0, `${profile.dopplerModelId}: performance_comparable requires a committed compare fixture`);
  const clean = candidates.filter(({ evidence }) => promotionViolations(profile, evidence).length === 0);
  assert.ok(
    clean.length > 0,
    `${profile.dopplerModelId}: no fixture satisfies the promotion rule: ${
      candidates.map(({ name, evidence }) => `${name}: ${promotionViolations(profile, evidence).join(', ')}`).join('; ')}`
  );
}
assert.ok(promoted > 0, 'at least one Qwen lane is promoted; otherwise this contract checks nothing');

// Committed observations stay bound to the frozen compare config they measured.
{
  const frozenSha = crypto.createHash('sha256').update(fs.readFileSync(FROZEN_COMPARE_CONFIG)).digest('hex');
  const qwenEvidence = fixtures.filter(({ evidence }) => String(evidence.dopplerModelId ?? '').startsWith('qwen-3-5-'));
  for (const { name, evidence } of qwenEvidence) {
    assert.match(String(evidence.compareConfig?.sourceSha256), /^[0-9a-f]{64}$/, `${name}: compare config hash must be recorded`);
  }
  assert.ok(
    qwenEvidence.some(({ evidence }) => evidence.compareConfig.sourceSha256 === frozenSha),
    'the frozen pre-capsule compare config must still hash to a recorded Qwen observation'
  );
}

// The rule rejects the failure modes it exists to catch.
{
  const profile = qwenProfiles.find((entry) => entry.compareLane === 'performance_comparable');
  const { evidence } = fixtures.find(({ evidence: e }) => e.dopplerModelId === profile.dopplerModelId
    && promotionViolations(profile, e).length === 0);
  const mismatch = structuredClone(evidence);
  mismatch.correctness.status = 'mismatch';
  mismatch.correctness.tokenMatch.firstMismatchTokenIndex = 3;
  assert.ok(promotionViolations(profile, mismatch).includes('correctness is not clean'));
  const throughputOnly = structuredClone(evidence);
  throughputOnly.decodeProfile = 'throughput';
  assert.ok(promotionViolations(profile, throughputOnly).includes('promotion evidence must use the parity decode profile'));
  const otherModel = structuredClone(evidence);
  otherModel.dopplerModelId = 'some-other-model';
  assert.ok(promotionViolations(profile, otherModel).includes('model id mismatch'));
}

console.log('qwen-compare-lane-promotion-contract.test: ok');
