import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const AF32_MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af32';
const AF16_MODEL_ID = 'gemma-4-31b-it-text-q4k-ehf16-af16';
const QWEN_MODEL_ID = 'qwen-3-6-27b-q4k-ehaf16';

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function stripUndefined(value) {
  if (Array.isArray(value)) {
    return value.map(stripUndefined);
  }
  if (!value || typeof value !== 'object') {
    return value;
  }
  const output = {};
  for (const key of Object.keys(value).sort()) {
    if (value[key] !== undefined) {
      output[key] = stripUndefined(value[key]);
    }
  }
  return output;
}

function hashJson(value) {
  return `sha256:${crypto.createHash('sha256').update(JSON.stringify(stripUndefined(value))).digest('hex')}`;
}

function hashText(value) {
  return `sha256:${crypto.createHash('sha256').update(String(value)).digest('hex')}`;
}

function catalogEntry(catalog, modelId) {
  return catalog.models.find((entry) => entry.modelId === modelId) ?? null;
}

const catalog = readJson('models/catalog.json');
const releaseClaims = readJson('tools/policies/release-claim-policy.json');
const af32Config = readJson(`src/config/conversion/gemma4/${AF32_MODEL_ID}.json`);
const af16Config = readJson(`src/config/conversion/gemma4/${AF16_MODEL_ID}.json`);
const af32ManifestPath = path.join('models', 'local', AF32_MODEL_ID, 'manifest.json');
const af16ManifestPath = path.join('models', 'local', AF16_MODEL_ID, 'manifest.json');
const af32ManifestText = fs.readFileSync(af32ManifestPath, 'utf8');
const af32Manifest = JSON.parse(af32ManifestText);
const af16Manifest = readJson(af16ManifestPath);
const profile = readJson('src/config/runtime/profiles/gemma4-31b-f16-activations-probe.json');

const af32Entry = catalogEntry(catalog, AF32_MODEL_ID);
const af16Entry = catalogEntry(catalog, AF16_MODEL_ID);
const qwenEntry = catalogEntry(catalog, QWEN_MODEL_ID);
const af32Claim = releaseClaims.claims.find((claim) => claim.modelId === AF32_MODEL_ID) ?? null;
const af16Claim = releaseClaims.claims.find((claim) => claim.modelId === AF16_MODEL_ID) ?? null;

assert.ok(af32Entry, `${AF32_MODEL_ID}: existing catalog entry must remain present`);
assert.ok(qwenEntry, `${QWEN_MODEL_ID}: existing Qwen catalog entry must remain present`);
assert.ok(af16Entry, `${AF16_MODEL_ID}: f16 activation catalog entry must be present`);

assert.equal(af32Config.output?.modelBaseId, AF32_MODEL_ID);
assert.equal(af16Config.output?.modelBaseId, AF16_MODEL_ID);
assert.equal(profile.model, AF16_MODEL_ID);

assert.equal(
  af32Entry.manifestVariantId,
  af32Manifest.artifactIdentity?.manifestVariantId,
  'af32 catalog entry must point at the checked-in af32 manifest variant id'
);
assert.equal(
  af16Manifest.artifactIdentity?.weightPackId,
  af32Manifest.artifactIdentity?.weightPackId,
  'af16 manifest variant must share the af32 weight pack id'
);
assert.equal(
  af16Manifest.artifactIdentity?.weightPackHash,
  af32Manifest.artifactIdentity?.weightPackHash,
  'af16 manifest variant must share the af32 weight pack hash'
);
assert.equal(
  af16Manifest.artifactIdentity?.shardSetHash,
  af32Manifest.artifactIdentity?.shardSetHash,
  'af16 manifest variant must share the af32 shard-set hash'
);
assert.notEqual(
  af16Manifest.artifactIdentity?.manifestVariantId,
  af32Manifest.artifactIdentity?.manifestVariantId,
  'af16 manifest variant must carry its own manifest variant id'
);
assert.equal(
  af16Entry.manifestVariantId,
  af16Manifest.artifactIdentity?.manifestVariantId,
  'af16 catalog entry must point at the f16 manifest variant id'
);
assert.equal(af16Entry.weightPackId, af32Manifest.artifactIdentity?.weightPackId);
assert.equal(af16Entry.weightsRefAllowed, true);
assert.equal(af16Entry.lifecycle?.availability?.hf, false);
assert.equal(af16Entry.lifecycle?.status?.runtime, 'active');
assert.equal(af16Entry.lifecycle?.status?.tested, 'verified');
assert.equal(af16Entry.lifecycle?.tested?.suite, 'inference');
assert.deepEqual(af16Entry.lifecycle?.tested?.surface, ['node']);
assert.equal(af16Entry.lifecycle?.tested?.result, 'pass');
assert.equal(af16Entry.lifecycle?.tested?.contracts?.executionContractOk, true);
assert.equal(af16Entry.demoVisible, false);

assert.equal(af16Manifest.quantizationInfo?.compute, 'f16');
assert.equal(af16Manifest.quantizationInfo?.variantTag, 'q4k-ehf16-af16');
assert.deepEqual(af16Manifest.weightsRef, af16Config.manifest?.weightsRef);
assert.equal(af16Manifest.weightsRef?.artifactRoot, `../${AF32_MODEL_ID}`);
assert.equal(af16Manifest.weightsRef?.manifestDigest, hashText(af32ManifestText));
assert.equal(af16Manifest.weightsRef?.shardSetHash, af32Manifest.artifactIdentity?.shardSetHash);
assert.equal(af16Manifest.artifactIdentity?.conversionConfigDigest, hashJson(af16Config));
assert.equal(af32Manifest.artifactIdentity?.conversionConfigDigest, hashJson(af32Config));

assert.deepEqual(
  af16Manifest.inference?.session?.compute?.defaults,
  {
    activationDtype: 'f16',
    mathDtype: 'f16',
    accumDtype: 'f16',
    outputDtype: 'f16',
  },
  'af16 manifest session defaults must make the f16 compute lane explicit'
);
assert.equal(af16Manifest.inference?.session?.kvcache?.kvDtype, 'f16');
assert.deepEqual(af16Manifest.inference?.largeWeights?.gpuResidentOverrides, []);

assert.ok(af32Claim, 'af32 primary must have package-visible runtime evidence before the af16 sibling is claimable');
assert.equal(
  af32Claim.evidence?.reportPath,
  'reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af32/2026-06-29T22-07-45.149Z.json'
);
assert.ok(af16Claim, 'af16 weights-ref sibling is claimable after the af32 primary has package-visible runtime evidence');
assert.equal(
  af16Claim.evidence?.reportPath,
  'reports/release-claims/gemma-4-31b-it-text-q4k-ehf16-af16/2026-06-29T22-09-42.445Z.json'
);
assert.equal(af16Claim.performanceEvidence?.metricPath, 'metrics.decodeTokensPerSec');

console.log('gemma4-31b-f16-variant-identity.test: ok');
