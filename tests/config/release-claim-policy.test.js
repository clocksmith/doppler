import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const REPO_ROOT = process.cwd();
const CATALOG_PATH = path.join(REPO_ROOT, 'models', 'catalog.json');
const CLAIM_POLICY_PATH = path.join(REPO_ROOT, 'tools', 'policies', 'release-claim-policy.json');

const catalog = JSON.parse(fs.readFileSync(CATALOG_PATH, 'utf8'));
const claimPolicy = JSON.parse(fs.readFileSync(CLAIM_POLICY_PATH, 'utf8'));

assert.equal(claimPolicy.schemaVersion, 1, 'release-claim-policy.json schemaVersion must be 1');
assert.equal(Array.isArray(claimPolicy.claims), true, 'release-claim-policy.json must define claims[]');

const verifiedCatalogModels = (Array.isArray(catalog.models) ? catalog.models : [])
  .filter((entry) => entry?.lifecycle?.status?.runtime === 'active')
  .filter((entry) => entry?.lifecycle?.status?.tested === 'verified')
  .map((entry) => ({
    modelId: entry.modelId,
    modes: Array.isArray(entry.modes) ? entry.modes : [],
    surface: entry?.lifecycle?.tested?.surface ?? null,
    source: entry?.lifecycle?.tested?.source ?? null,
    lastVerifiedAt: entry?.lifecycle?.tested?.lastVerifiedAt ?? null,
    artifactFormat: entry?.artifact?.format ?? null,
    artifactSchema: entry?.artifact?.sourceRuntimeSchema ?? null,
  }))
  .sort((left, right) => left.modelId.localeCompare(right.modelId));

const claimEntries = claimPolicy.claims
  .map((entry) => ({
    modelId: entry.modelId,
    mode: entry.mode,
    surface: entry.surface,
    source: entry.verificationSource,
    lastVerifiedAt: entry.lastVerifiedAt,
    artifactFormat: entry.artifactFormat,
    artifactSchema: entry.artifactSchema ?? null,
    evidence: entry.evidence,
  }))
  .sort((left, right) => left.modelId.localeCompare(right.modelId));

assert.deepEqual(
  claimEntries.map((entry) => entry.modelId),
  verifiedCatalogModels.map((entry) => entry.modelId),
  'Every verified active catalog model must have an explicit release claim policy entry, and vice versa.'
);

for (const claim of claimEntries) {
  const catalogEntry = verifiedCatalogModels.find((entry) => entry.modelId === claim.modelId);
  assert.ok(catalogEntry, `${claim.modelId}: missing verified catalog entry`);
  assert.equal(typeof claim.mode, 'string', `${claim.modelId}: claim mode must be a string`);
  assert.ok(catalogEntry.modes.includes(claim.mode), `${claim.modelId}: claim mode "${claim.mode}" must exist in catalog modes`);
  assert.equal(claim.surface, catalogEntry.surface, `${claim.modelId}: claim surface must match models/catalog.json`);
  assert.equal(claim.source, catalogEntry.source, `${claim.modelId}: claim verificationSource must match models/catalog.json`);
  assert.equal(claim.lastVerifiedAt, catalogEntry.lastVerifiedAt, `${claim.modelId}: claim lastVerifiedAt must match models/catalog.json`);
  assert.equal(
    claim.artifactFormat,
    catalogEntry.artifactFormat,
    `${claim.modelId}: claim artifactFormat must match models/catalog.json`
  );
  assert.equal(
    claim.artifactSchema,
    catalogEntry.artifactSchema,
    `${claim.modelId}: claim artifactSchema must match models/catalog.json`
  );
  assert.ok(claim.evidence && typeof claim.evidence === 'object', `${claim.modelId}: claim evidence must be an object`);
  assert.equal(typeof claim.evidence.kind, 'string', `${claim.modelId}: claim evidence.kind must be a string`);

  if (claim.source === 'manual-review') {
    assert.equal(
      claim.evidence.kind,
      'manual-report',
      `${claim.modelId}: manual-review claims must point to a committed report artifact`
    );
    const reportPath = path.join(REPO_ROOT, String(claim.evidence.reportPath ?? ''));
    assert.ok(String(claim.evidence.reportPath ?? '').trim(), `${claim.modelId}: manual-review claim must define evidence.reportPath`);
    assert.equal(fs.existsSync(reportPath), true, `${claim.modelId}: manual-review report path must exist (${claim.evidence.reportPath})`);
  }
}

console.log('release-claim-policy.test: ok');
