import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  buildModelReleasePlatformReport,
  validateModelReleasePlatform,
} from '../../tools/check-model-release-platform.js';

const policy = JSON.parse(await fs.readFile('tools/policies/model-release-platform.json', 'utf8'));
const matrix = JSON.parse(await fs.readFile('src/config/goal-completion-matrix.json', 'utf8'));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

{
  const report = await buildModelReleasePlatformReport();
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.commercialAssessment, 'unestablished');
  assert.equal(policy.positioning.entryProduct, 'Doppler Production Release');
  assert.equal(policy.positioning.recurringProduct, 'Doppler Release Operations');
  assert.equal(policy.positioning.initialIcp, 'TypeScript/Electron desktop products on Windows and macOS');
  assert.equal(policy.commercialOffer.activationAuthority, 'customer-controlled');
  assert.equal(policy.commercialOffer.selfPromotionAllowed, false);
  assert.equal(policy.referenceIntegrations.externalAdoptionEvidence, false);
  assert.equal(policy.referenceIntegrations.commercialDemandEvidence, false);
  assert.deepEqual(report.migrationSurfaces, ['openai-server', 'generation', 'embedding']);
  assert.equal(
    policy.apiConvergence.find((row) => row.id === 'electron-reranking')?.mode,
    'pack-authoritative'
  );
  assert.deepEqual(report.partialRequirements, []);
  assert.deepEqual(report.implementedPromotionGates, [
    'canonical-product-contract',
    'electron-reference-release',
    'pack-release-closure',
    'pack-first-electron-reranking',
    'doppler-release-command',
    'github-release-action',
  ]);
  assert.deepEqual(report.externalPromotionGates, [
    'electron-fleet-qualification',
    'revocation-and-customer-rollback',
    'paid-external-release-and-upgrade',
    'three-unrelated-design-partners',
  ]);
  assert.equal(
    policy.promotionSequence.find((row) => row.id === 'electron-fleet-qualification')?.blockerCode,
    'customer-electron-fleet-receipts-missing'
  );
  assert.equal(
    policy.promotionSequence.find((row) => row.id === 'three-unrelated-design-partners')?.blockerCode,
    'three-unrelated-design-partners-missing'
  );
  assert.ok(policy.pack.requiredReleaseElements.find(
    (row) => row.id === 'version-supersession-migration'
  )?.implementationState === 'implemented');
  assert.ok(policy.recovery.find(
    (row) => row.id === 'portable-state-snapshot-identity'
  )?.implementationState === 'implemented');
}

{
  const broken = clone(policy);
  const releaseCommand = broken.promotionSequence.find((row) => row.id === 'doppler-release-command');
  releaseCommand.blockerCode = 'paid-doppler-production-release-missing';
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(report.errors.includes(
    'promotionSequence.doppler-release-command: implemented gates must have blockerCode null'
  ));
}

{
  const broken = clone(policy);
  broken.providers.doeRequired = true;
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(report.errors.includes('providers.doeRequired must be false'), report.errors.join('\n'));
}

{
  const broken = clone(policy);
  const migration = broken.pack.requiredReleaseElements
    .find((row) => row.id === 'version-supersession-migration');
  migration.implementationState = 'partial';
  migration.blockerCode = null;
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes('pack.requiredReleaseElements.version-supersession-migration: partial rows require a blockerCode'),
    report.errors.join('\n')
  );
}

{
  const broken = clone(policy);
  broken.apiConvergence.find((row) => row.id === 'generation').evidencePaths = ['missing/path.js'];
  const report = await validateModelReleasePlatform(broken, matrix);
  assert.ok(
    report.errors.includes('apiConvergence.generation.evidencePaths references missing path missing/path.js'),
    report.errors.join('\n')
  );
}

{
  const broken = clone(matrix);
  const goal = broken.goals.find((row) => row.id === 'local-webgpu-product-surface');
  goal.rows = goal.rows.filter((row) => row.id !== 'doppler-production-release-offer');
  const report = await validateModelReleasePlatform(policy, broken);
  assert.ok(
    report.errors.includes('goal matrix is missing local-webgpu-product-surface/doppler-production-release-offer'),
    report.errors.join('\n')
  );
}

console.log('model-release-platform.test: ok');
