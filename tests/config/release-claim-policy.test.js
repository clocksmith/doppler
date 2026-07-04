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

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function artifactPaths(value) {
  const paths = [];
  if (typeof value.reportPath === 'string' && value.reportPath.trim()) {
    paths.push(value.reportPath);
  }
  if (typeof value.receiptPath === 'string' && value.receiptPath.trim()) {
    paths.push(value.receiptPath);
  }
  if (Array.isArray(value.receiptPaths)) {
    for (const receiptPath of value.receiptPaths) {
      if (typeof receiptPath === 'string' && receiptPath.trim()) {
        paths.push(receiptPath);
      }
    }
  }
  return Array.from(new Set(paths));
}

function readArtifact(relativePath, context) {
  assert.equal(path.isAbsolute(relativePath), false, `${context}: artifact path must be repository-relative`);
  const absolutePath = path.join(REPO_ROOT, relativePath);
  assert.equal(fs.existsSync(absolutePath), true, `${context}: artifact path must exist (${relativePath})`);
  return JSON.parse(fs.readFileSync(absolutePath, 'utf8'));
}

function getPath(value, pathExpression) {
  return String(pathExpression).split('.').reduce((current, key) => {
    if (!current || typeof current !== 'object') {
      return undefined;
    }
    return current[key];
  }, value);
}

function artifactModelIds(artifact) {
  return [
    artifact.modelId,
    artifact.model,
    artifact.dopplerModelId,
    artifact?.sections?.compute?.parity?.dopplerModelId,
    artifact?.sections?.compute?.throughput?.dopplerModelId,
  ].filter((value) => typeof value === 'string' && value.trim());
}

function assertArtifactModelMatches(claim, artifact, context) {
  const modelIds = artifactModelIds(artifact);
  assert.ok(modelIds.length > 0, `${context}: artifact must identify the Doppler model`);
  assert.ok(
    modelIds.includes(claim.modelId),
    `${context}: artifact model id must match ${claim.modelId}; saw ${modelIds.join(', ')}`
  );
}

function assertArtifactResultsPassed(artifact, context) {
  if (!Array.isArray(artifact.results)) {
    return;
  }
  assert.ok(artifact.results.length > 0, `${context}: run report results[] must not be empty`);
  for (const result of artifact.results) {
    assert.equal(result?.passed, true, `${context}: run report result "${result?.name ?? '<unnamed>'}" must pass`);
  }
}

function assertEvidenceArtifacts(claim, evidence, fieldName) {
  assert.ok(isObject(evidence), `${claim.modelId}: ${fieldName} must be an object`);
  assert.equal(typeof evidence.kind, 'string', `${claim.modelId}: ${fieldName}.kind must be a string`);
  assert.ok(evidence.kind.trim(), `${claim.modelId}: ${fieldName}.kind must not be empty`);

  const paths = artifactPaths(evidence);
  assert.ok(paths.length > 0, `${claim.modelId}: ${fieldName} must define reportPath, receiptPath, or receiptPaths`);

  return paths.map((artifactPath) => {
    const context = `${claim.modelId}: ${fieldName} ${artifactPath}`;
    const artifact = readArtifact(artifactPath, context);
    assertArtifactModelMatches(claim, artifact, context);
    assertArtifactResultsPassed(artifact, context);
    return { artifactPath, artifact };
  });
}

function assertPerformanceEvidence(claim) {
  const artifacts = assertEvidenceArtifacts(claim, claim.performanceEvidence, 'performanceEvidence');
  assert.equal(
    typeof claim.performanceEvidence.metricPath,
    'string',
    `${claim.modelId}: performanceEvidence.metricPath must be a string`
  );
  assert.ok(
    claim.performanceEvidence.metricPath.trim(),
    `${claim.modelId}: performanceEvidence.metricPath must not be empty`
  );
  assert.equal(
    typeof claim.performanceEvidence.unit,
    'string',
    `${claim.modelId}: performanceEvidence.unit must be a string`
  );
  assert.equal(
    typeof claim.performanceEvidence.minValue,
    'number',
    `${claim.modelId}: performanceEvidence.minValue must be numeric`
  );
  assert.equal(
    Number.isFinite(claim.performanceEvidence.minValue),
    true,
    `${claim.modelId}: performanceEvidence.minValue must be finite`
  );

  const artifact = artifacts[0]?.artifact;
  const metric = getPath(artifact, claim.performanceEvidence.metricPath);
  assert.equal(
    typeof metric,
    'number',
    `${claim.modelId}: performance metric ${claim.performanceEvidence.metricPath} must resolve to a number`
  );
  assert.equal(
    Number.isFinite(metric),
    true,
    `${claim.modelId}: performance metric ${claim.performanceEvidence.metricPath} must be finite`
  );
  assert.ok(
    metric > claim.performanceEvidence.minValue,
    `${claim.modelId}: performance metric ${claim.performanceEvidence.metricPath} must be greater than ${claim.performanceEvidence.minValue}`
  );

  if (claim.mode === 'embedding') {
    const manifest = readLocalManifest(claim.modelId, `${claim.modelId}: embedding performance evidence`);
    const expectedEmbeddingDim = manifest?.architecture?.hiddenSize;
    assert.equal(
      getPath(artifact, 'metrics.semanticPassed'),
      true,
      `${claim.modelId}: embedding performance evidence must preserve semanticPassed=true`
    );
    assert.equal(
      getPath(artifact, 'metrics.embeddingDim'),
      expectedEmbeddingDim,
      `${claim.modelId}: embedding performance evidence must match the manifest hidden size`
    );
  }

  if (claim.mode === 'rerank') {
    assert.equal(
      getPath(artifact, 'metrics.semanticPassed'),
      true,
      `${claim.modelId}: rerank performance evidence must preserve semanticPassed=true`
    );
    assert.equal(
      typeof getPath(artifact, 'metrics.rerankMs'),
      'number',
      `${claim.modelId}: rerank performance evidence must preserve rerankMs`
    );
  }

  if (claim.mode === 'diffusion') {
    assertDiffusionPerformanceMatchesManifest(claim, artifact);
  }
}

function readLocalManifest(modelId, context) {
  const manifestPath = path.join('models', 'local', modelId, 'manifest.json');
  const absolutePath = path.join(REPO_ROOT, manifestPath);
  assert.equal(fs.existsSync(absolutePath), true, `${context}: local manifest must exist (${manifestPath})`);
  return JSON.parse(fs.readFileSync(absolutePath, 'utf8'));
}

function assertDiffusionPerformanceMatchesManifest(claim, artifact) {
  const context = `${claim.modelId}: diffusion performance evidence`;
  const manifest = readLocalManifest(claim.modelId, context);
  const expected = manifest?.inference?.diffusionGemma ?? null;
  if (!expected) {
    return;
  }
  const actual = artifact?.metrics?.performanceArtifact?.diffusionGemma ?? null;
  assert.ok(actual, `${context}: metrics.performanceArtifact.diffusionGemma is required`);
  assert.equal(
    actual.canvasLength,
    expected.canvasLength,
    `${context}: canvasLength must match the manifest contract`
  );
  assert.equal(
    actual.maxDenoisingSteps,
    expected.maxDenoisingSteps,
    `${context}: maxDenoisingSteps must match the manifest contract`
  );
  assert.equal(
    actual.maxNewTokens,
    expected.maxNewTokens,
    `${context}: maxNewTokens must match the manifest contract`
  );
}

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
    performanceEvidence: entry.performanceEvidence,
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
  assert.deepEqual(claim.surface, catalogEntry.surface, `${claim.modelId}: claim surface must match models/catalog.json`);
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
  assertEvidenceArtifacts(claim, claim.evidence, 'evidence');
  assertPerformanceEvidence(claim);

  if (claim.source === 'manual-review') {
    assert.equal(
      claim.evidence.kind,
      'manual-report',
      `${claim.modelId}: manual-review claims must point to a committed report artifact`
    );
  }
}

console.log('release-claim-policy.test: ok');
