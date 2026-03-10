import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const TEST_COVERAGE_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'test-coverage-policy.json');
const CLAIM_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'release-claim-policy.json');

const coveragePolicy = JSON.parse(fs.readFileSync(TEST_COVERAGE_POLICY_PATH, 'utf8'));
const claimPolicy = JSON.parse(fs.readFileSync(CLAIM_POLICY_PATH, 'utf8'));

const claimedModelIds = new Set(
  (Array.isArray(claimPolicy.claims) ? claimPolicy.claims : [])
    .map((entry) => String(entry?.modelId ?? '').trim())
    .filter(Boolean)
);

const excludedTests = Array.isArray(coveragePolicy.excludeTests) ? coveragePolicy.excludeTests : [];

for (const relativeTestPath of excludedTests) {
  if (!String(relativeTestPath).startsWith('tests/inference/')) {
    continue;
  }

  const absoluteTestPath = path.join(REPO_ROOT, relativeTestPath);
  assert.equal(fs.existsSync(absoluteTestPath), true, `Excluded inference test must exist: ${relativeTestPath}`);

  const source = fs.readFileSync(absoluteTestPath, 'utf8');
  const modelIdMatch = source.match(/\bconst\s+modelId\s*=\s*['"]([^'"]+)['"]/);
  if (!modelIdMatch) {
    continue;
  }

  const modelId = String(modelIdMatch[1] ?? '').trim();
  assert.ok(
    !claimedModelIds.has(modelId),
    `${relativeTestPath} targets claimed model "${modelId}" but is excluded from the broad suite. ` +
    'Claimed-model inference smoke must not be hidden behind test-coverage exclusions.'
  );
}

console.log('excluded-inference-smoke-contract.test: ok');
